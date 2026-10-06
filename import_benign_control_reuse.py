#!/usr/bin/env python3
"""Reuse byte-identical benign evaluations instead of paying for duplicates.

The source result must have the same question text, language, prompt type, and
system prompt as the locked v2 control. Only successful API observations are
copied. The original response/conversation is unchanged; provenance and the
new locked question id are added to the copied record.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from benign_control import CONTROL_CASES, LANGUAGES, PROMPT_TYPES, build_questions
from eval_v3 import get_system_prompt


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "eval_results_benign_control_v2"


def iter_result_files(root: Path) -> Iterable[Path]:
    return sorted(
        root.glob("*/*/*/results_*.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )


def identify_scenario(question_id: str) -> str | None:
    for case in CONTROL_CASES:
        scenario_id = case["benign_scenario_id"]
        if question_id.startswith(f"{scenario_id}_"):
            return scenario_id
    return None


def collect_reusable(
    source_root: Path,
    questions: Dict[Tuple[str, str], dict],
) -> Dict[Tuple[str, str, str, str], Tuple[dict, Path, Path]]:
    reusable = {}
    for path in iter_result_files(source_root):
        language, prompt, model = path.relative_to(source_root).parts[:3]
        if language not in LANGUAGES or prompt not in PROMPT_TYPES:
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            scenario_id = identify_scenario(row.get("question_id") or "")
            if scenario_id is None or row.get("error"):
                continue
            question = questions[(language, scenario_id)]
            if row.get("question_text") != question["question"]:
                raise ValueError(
                    f"{path}: prompt mismatch for {language}/{scenario_id}"
                )
            if row.get("prompt_type") != prompt:
                raise ValueError(
                    f"{path}: prompt-type mismatch for {language}/{scenario_id}"
                )
            if row.get("system_prompt") != get_system_prompt(language, prompt):
                raise ValueError(
                    f"{path}: system-prompt mismatch for {language}/{scenario_id}"
                )
            reusable[(language, prompt, model, scenario_id)] = (
                row,
                source_root,
                path,
            )
    return reusable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        default=[],
        help="Prior result root. Repeat for multiple roots.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.source:
        raise SystemExit("Provide at least one --source result root")

    questions = {
        (language, question["scenario_id"]): question
        for language in LANGUAGES
        for question in build_questions(language)
    }
    grouped: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    seen = set()
    for source_root in args.source:
        if not source_root.is_dir():
            raise SystemExit(f"Missing source result root: {source_root}")
        for key, (row, source_root, source_path) in collect_reusable(
            source_root, questions
        ).items():
            language, prompt, model, scenario_id = key
            if key in seen:
                continue
            seen.add(key)
            copied = dict(row)
            copied["benign_control_reuse"] = {
                "source_collection": source_root.name,
                "source_file": str(source_path.relative_to(source_root)),
                "source_question_id": row["question_id"],
                "validation": (
                    "exact question text, language, prompt type, and system prompt"
                ),
            }
            copied["question_id"] = questions[
                (language, scenario_id)
            ]["question_id"]
            grouped[(language, prompt, model)].append(copied)

    for (language, prompt, model), rows in sorted(grouped.items()):
        output_dir = args.output_root / language / prompt / model
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "results_reused_exact_protocol.json"
        output_path.write_text(
            json.dumps(
                sorted(rows, key=lambda row: row["question_id"]),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        incremental_path = output_dir / "incremental_reused_exact_protocol.jsonl"
        incremental_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in sorted(rows, key=lambda row: row["question_id"])
            ),
            encoding="utf-8",
        )

    print(f"Reused exact observations: {len(seen)}")
    print(f"Completed cells touched:    {len(grouped)}")
    print(f"Paid API calls made:        0")


if __name__ == "__main__":
    main()
