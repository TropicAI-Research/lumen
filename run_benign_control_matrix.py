#!/usr/bin/env python3
"""Plan or execute the locked exact-model benign-control matrix.

The script is intentionally opt-in: without ``--execute`` it only validates
the design, credentials, and planned commands.  Runs are model-by-model so
that failures can be resumed without repeating completed observations.
The routing manifest prevents retired model aliases from being replaced by a
different current model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Set

from benign_control import DESIGN_VERSION, LANGUAGES, PROMPT_TYPES


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = ROOT / "configs" / "models_benign_control"
DEFAULT_QUESTION_DIR = ROOT / "benign_control_questions"
DEFAULT_RESULTS_ROOT = ROOT / "eval_results_benign_control"
DEFAULT_LOCK = ROOT / "benign_control_lock.json"
DEFAULT_ROUTING = ROOT / "benign_control_routing.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_lock(lock_file: Path) -> dict:
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    if lock.get("design_version") != DESIGN_VERSION:
        raise SystemExit(
            f"Design version mismatch: lock={lock.get('design_version')} code={DESIGN_VERSION}"
        )
    for relative, expected in lock["files"].items():
        path = ROOT / relative
        observed = sha256(path)
        if observed != expected:
            raise SystemExit(
                f"Locked design file changed: {relative}\n"
                f"expected {expected}\nobserved {observed}\n"
                "Regenerate/review the design before spending API budget."
            )
    return lock


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overwriting exported variables."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())


def config_files(config_dir: Path, selected: Set[str]) -> List[Path]:
    files = sorted(config_dir.glob("*.json"))
    if selected:
        files = [
            path
            for path in files
            if path.stem in selected
            or json.loads(path.read_text(encoding="utf-8")).get("model_name") in selected
        ]
        found = {path.stem for path in files}
        missing = selected - found - {
            json.loads(path.read_text(encoding="utf-8")).get("model_name")
            for path in files
        }
        if missing:
            raise SystemExit(f"Unknown requested models: {sorted(missing)}")
    return files


def required_env_vars(configs: Iterable[Path]) -> Set[str]:
    required = set()
    for path in configs:
        config = json.loads(path.read_text(encoding="utf-8"))
        for value in config.values():
            if isinstance(value, str) and value.startswith("$"):
                required.add(value[1:])
    return required


def completed_question_ids(model_dir: Path) -> Set[str]:
    completed: Set[str] = set()
    for path in sorted(model_dir.glob("incremental_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("question_id") and not row.get("error"):
                    completed.add(row["question_id"])
    for path in sorted(model_dir.glob("results_*.json")):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows:
            if row.get("question_id") and not row.get("error"):
                completed.add(row["question_id"])
    return completed


def expected_ids(question_file: Path) -> Set[str]:
    return {
        row["question_id"]
        for row in json.loads(question_file.read_text(encoding="utf-8"))
    }


def new_inference_cost(results_root: Path) -> float:
    """Sum paid v2 inference once, excluding exact-protocol reused records."""
    artifacts = list(results_root.glob("*/*/*/results_*.json"))
    artifacts.extend(results_root.glob("*/*/*/incremental_*.jsonl"))
    seen = set()
    total = 0.0
    for path in sorted(artifacts):
        if path.suffix == ".jsonl":
            rows = []
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        else:
            try:
                rows = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
        for row in rows:
            if row.get("benign_control_reuse"):
                continue
            cost = row.get("total_cost")
            if not isinstance(cost, (int, float)) or cost <= 0:
                continue
            trace_ids = tuple(
                trace.get("response_id")
                for trace in (row.get("api_trace") or [])
                if trace.get("response_id")
            )
            if trace_ids:
                fingerprint = ("api_trace", trace_ids)
            else:
                fingerprint = (
                    "legacy",
                    path.parent.name,
                    row.get("question_id"),
                    row.get("prompt_type"),
                    row.get("prompt_tokens"),
                    row.get("completion_tokens"),
                    row.get("execution_time"),
                    row.get("assistant_response"),
                )
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            total += float(cost)
    return total


def append_log(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def parse_csv_set(value: str) -> Set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--question-dir", type=Path, default=DEFAULT_QUESTION_DIR)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--routing-manifest", type=Path, default=DEFAULT_ROUTING
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated config stems/model names. Empty means every exact routed model.",
    )
    parser.add_argument(
        "--languages",
        default=",".join(LANGUAGES),
        help="Comma-separated subset of pt,en,es,zh,hi.",
    )
    parser.add_argument(
        "--prompts",
        default=",".join(PROMPT_TYPES),
        help="Comma-separated subset of ethical,neutral.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument(
        "--show-commands",
        action="store_true",
        help="Print every command in dry-run mode (the default summary is concise).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call model APIs. Without this flag, print a dry-run plan.",
    )
    parser.add_argument(
        "--max-new-cost-usd",
        type=float,
        help=(
            "Required with --execute. Stop before a cell whose 1.5x estimated "
            "cost would exceed this cumulative v2 inference ceiling."
        ),
    )
    parser.add_argument(
        "--allow-missing-credentials",
        action="store_true",
        help="Dry-run despite missing provider credentials; forbidden with --execute.",
    )
    args = parser.parse_args()

    lock = verify_lock(args.lock_file)
    case_count = int(lock["case_count"])
    routing = json.loads(args.routing_manifest.read_text(encoding="utf-8"))
    exact_models = set(routing["exact_runnable_models"])
    selected_models = parse_csv_set(args.models)
    configs = config_files(args.config_dir, selected_models)
    config_stems = {path.stem for path in configs}
    if not selected_models and config_stems != exact_models:
        raise SystemExit(
            "Config directory does not match the routing lock: "
            f"missing={sorted(exact_models - config_stems)}, "
            f"unexpected={sorted(config_stems - exact_models)}"
        )
    if selected_models and not config_stems.issubset(exact_models):
        raise SystemExit(
            "Selected configs include models outside the routing lock: "
            f"{sorted(config_stems - exact_models)}"
        )

    languages = tuple(item for item in LANGUAGES if item in parse_csv_set(args.languages))
    prompts = tuple(item for item in PROMPT_TYPES if item in parse_csv_set(args.prompts))
    if not languages or set(languages) != parse_csv_set(args.languages):
        raise SystemExit(f"Invalid --languages: {args.languages}")
    if not prompts or set(prompts) != parse_csv_set(args.prompts):
        raise SystemExit(f"Invalid --prompts: {args.prompts}")

    load_dotenv(ROOT / ".env")
    required_env = required_env_vars(configs)
    missing_env = sorted(name for name in required_env if not os.environ.get(name))
    if missing_env and (args.execute or not args.allow_missing_credentials):
        raise SystemExit(
            "Missing required credentials: "
            + ", ".join(missing_env)
            + ("\nUse --allow-missing-credentials for planning only." if not args.execute else "")
        )
    if args.execute and (
        args.max_new_cost_usd is None or args.max_new_cost_usd <= 0
    ):
        raise SystemExit(
            "--execute requires a positive --max-new-cost-usd spending ceiling"
        )

    planned = len(configs) * len(languages) * len(prompts) * case_count
    print(f"Design:       {lock['design_version']}")
    print(f"Models:       {len(configs)}")
    print(f"Languages:    {', '.join(languages)}")
    print(f"Prompts:      {', '.join(prompts)}")
    print(f"Cases/cell:   {case_count}")
    print(f"Observations: {planned}")
    excluded = routing.get("excluded_retired_models", {})
    if excluded:
        print(
            "Excluded:     "
            + ", ".join(sorted(excluded))
            + " (retired; no model substitution)"
        )
    if missing_env:
        print(f"Missing credentials (dry-run only): {', '.join(missing_env)}")

    commands = []
    skipped_observations = 0
    for language in languages:
        question_file = (
            args.question_dir / f"questions_benign_control_{language}.json"
        )
        ids = expected_ids(question_file)
        if len(ids) != case_count:
            raise SystemExit(
                f"{question_file}: expected {case_count} unique ids"
            )
        for prompt in prompts:
            output_dir = args.results_root / language / prompt
            for config in configs:
                model_dir = output_dir / config.stem
                done = completed_question_ids(model_dir) & ids
                skipped_observations += len(done)
                if len(done) == case_count:
                    continue
                command = [
                    sys.executable,
                    str(ROOT / "eval_v3.py"),
                    "--config",
                    str(config),
                    "--questions",
                    str(question_file),
                    "--output",
                    str(output_dir),
                    "--prompt-type",
                    prompt,
                    "--workers",
                    str(args.workers),
                    "--resume",
                ]
                if args.sequential:
                    command.append("--sequential")
                commands.append((language, prompt, config.stem, len(done), command))

    print(f"Already complete: {skipped_observations} observations")
    print(f"Remaining jobs:    {len(commands)} model-language-prompt cells")
    print(f"Remaining calls:   {planned - skipped_observations}")
    current_cost = new_inference_cost(args.results_root)
    print(f"New v2 cost so far: ${current_cost:.4f}")
    if args.max_new_cost_usd is not None:
        print(f"New v2 cost ceiling: ${args.max_new_cost_usd:.2f}")

    for language, prompt, model, done, command in commands:
        if args.show_commands:
            print(f"[{language}/{prompt}/{model}] {case_count - done} remaining")
        if not args.execute and args.show_commands:
            print("  " + " ".join(command))

    if not args.execute:
        print("\nDry run only. Add --execute after reviewing the locked design and budget.")
        return
    if args.allow_missing_credentials:
        raise SystemExit("--allow-missing-credentials cannot be used with --execute")

    log_path = args.results_root / "matrix_run_log.jsonl"
    for index, (language, prompt, model, done, command) in enumerate(commands, start=1):
        spent = new_inference_cost(args.results_root)
        model_estimate = float(
            routing["estimated_cost_usd_by_model"][model]
        )
        estimated_cell = (
            (case_count - done)
            / int(lock["observations_per_model"])
            * model_estimate
        )
        guarded_cell = estimated_cell * 1.5
        if spent + guarded_cell > args.max_new_cost_usd:
            print(
                "\nStopped before the next cell to protect the spending ceiling: "
                f"spent=${spent:.4f}, next 1.5x estimate=${guarded_cell:.4f}, "
                f"ceiling=${args.max_new_cost_usd:.2f}."
            )
            return
        started = datetime.now(timezone.utc).isoformat()
        print(f"\n[{index}/{len(commands)}] {language}/{prompt}/{model}")
        append_log(log_path, {
            "event": "start",
            "at_utc": started,
            "design_version": DESIGN_VERSION,
            "language": language,
            "prompt": prompt,
            "model": model,
            "already_complete": done,
            "command": command,
        })
        completed = subprocess.run(command, cwd=ROOT, check=False)
        question_file = (
            args.question_dir / f"questions_benign_control_{language}.json"
        )
        cell_ids = expected_ids(question_file)
        valid_after = completed_question_ids(
            args.results_root / language / prompt / model
        ) & cell_ids
        append_log(log_path, {
            "event": "finish",
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "design_version": DESIGN_VERSION,
            "language": language,
            "prompt": prompt,
            "model": model,
            "returncode": completed.returncode,
            "valid_observations": len(valid_after),
        })
        if completed.returncode != 0:
            raise SystemExit(
                f"Stopped after {language}/{prompt}/{model}: "
                f"eval_v3 exited {completed.returncode}. Re-run with --execute to resume."
            )
        if len(valid_after) != case_count:
            raise SystemExit(
                f"Stopped after {language}/{prompt}/{model}: evaluator returned "
                f"success but only {len(valid_after)}/{case_count} observations "
                "are valid. "
                "Inspect the provider errors, then re-run with --execute to resume."
            )
        print(
            f"Cumulative new v2 inference cost: "
            f"${new_inference_cost(args.results_root):.4f} / "
            f"${args.max_new_cost_usd:.2f}"
        )

    print("\nFull requested matrix finished. Run score_benign_control.py next.")


if __name__ == "__main__":
    main()
