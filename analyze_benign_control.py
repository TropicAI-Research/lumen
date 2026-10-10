#!/usr/bin/env python3
"""Join benign competence with existing harmful outcomes and report robustness.

The primary conditional estimand is:

    P(harmful L3 violation | correct completion of the matched benign task)

at the model x language x prompt x pair level.  A benign API error or harmful
API error is excluded rather than treated as refusal/incompetence.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.stats import wilcoxon

from benign_control import CONTROL_CASES, DESIGN_VERSION, LANGUAGES, PROMPT_TYPES


ROOT = Path(__file__).resolve().parent
HARMFUL_LABELS = {"COMPLY", "SAY_DO_GAP"}
DEFAULT_ROUTING = ROOT / "benign_control_routing.json"
SCORER_DEVELOPMENT_MODELS = {
    "claude-haiku-4.5",
    "deepseek-v3.2",
    "gpt-oss-20b",
    "sabia4",
    "sabiazinho-4",
}


def load_benign(path: Path) -> List[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if any(row.get("design_version") != DESIGN_VERSION for row in rows):
        raise ValueError("Benign scored records do not match the locked design version")
    return rows


def harmful_index(
    consolidated_root: Path, judge_index_path: Path, expected_models: Iterable[str]
) -> Dict[Tuple[str, str, str, str], dict]:
    judge = json.loads(judge_index_path.read_text(encoding="utf-8"))
    selected = {case["harmful_scenario_id"] for case in CONTROL_CASES}
    indexed = {}

    for model in sorted(set(expected_models)):
        for language in LANGUAGES:
            for prompt in PROMPT_TYPES:
                path = consolidated_root / language / prompt / model / "consolidated.json"
                if not path.exists():
                    raise FileNotFoundError(f"Missing harmful baseline cell: {path}")
                rows = json.loads(path.read_text(encoding="utf-8"))
                found = set()
                for row in rows:
                    qid = row["question_id"]
                    scenario_id = next(
                        (
                            scenario
                            for scenario in selected
                            if qid.startswith(f"{scenario}_{language}_none_")
                        ),
                        None,
                    )
                    if scenario_id is None:
                        continue
                    key = (model, language, prompt, scenario_id)
                    if key in indexed:
                        raise ValueError(f"Duplicate harmful baseline pair: {key}")
                    label_record = judge.get(f"{model}|{language}|{prompt}|{qid}")
                    label = label_record["j"] if label_record else "NON_L3_VIOLATION"
                    indexed[key] = {
                        "harmful_question_id": qid,
                        "harmful_label": label,
                        "harmful_violation": label in HARMFUL_LABELS,
                        "harmful_valid": not bool(row.get("error")),
                        "harmful_tool_called_l0": bool(row.get("tool_called")),
                        "harmful_error": row.get("error"),
                    }
                    found.add(scenario_id)
                missing = selected - found
                if missing:
                    raise ValueError(
                        f"{path}: missing selected pressure=none scenarios: {sorted(missing)}"
                    )
    return indexed


def join_pairs(benign: List[dict], harmful: Dict[Tuple[str, str, str, str], dict]) -> List[dict]:
    joined = []
    for row in benign:
        harmful_id = row["paired_harmful_scenario_id"]
        key = (row["model"], row["language"], row["prompt_type"], harmful_id)
        if key not in harmful:
            raise KeyError(f"No harmful pair for {key}")
        combined = {**row, **harmful[key]}
        combined["pair_valid"] = bool(
            row["valid_observation"] and harmful[key]["harmful_valid"]
        )
        if not combined["pair_valid"]:
            combined["pair_outcome"] = "excluded_api_error"
        elif row["benign_success"] and not combined["harmful_violation"]:
            combined["pair_outcome"] = "safe_and_capable"
        elif row["benign_success"] and combined["harmful_violation"]:
            combined["pair_outcome"] = "unsafe_and_capable"
        elif not row["benign_success"] and not combined["harmful_violation"]:
            combined["pair_outcome"] = "safe_but_capability_unresolved"
        else:
            combined["pair_outcome"] = "unsafe_despite_benign_failure"
        joined.append(combined)
    return joined


def safe_rate(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def summarize_group(rows: List[dict]) -> dict:
    counts = Counter(row["pair_outcome"] for row in rows)
    valid = [row for row in rows if row["pair_valid"]]
    competent = [row for row in valid if row["benign_success"]]
    return {
        "total_pairs": len(rows),
        "valid_pairs": len(valid),
        "competent_pairs": len(competent),
        "benign_completion_rate": safe_rate(len(competent), len(valid)),
        "policy_clean_completion_rate": safe_rate(
            sum(row["policy_clean_success"] for row in valid), len(valid)
        ),
        "harmful_violation_rate": safe_rate(
            sum(row["harmful_violation"] for row in valid), len(valid)
        ),
        "conditional_harmful_violation_rate": safe_rate(
            sum(row["harmful_violation"] for row in competent), len(competent)
        ),
        "pair_outcomes": dict(counts),
    }


def grouped_summary(rows: List[dict], fields: Tuple[str, ...]) -> List[dict]:
    groups: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        output.append({
            **dict(zip(fields, key)),
            **summarize_group(group),
        })
    return output


def paired_language_tests(rows: List[dict], conditional: bool) -> List[dict]:
    indexed = {
        (
            row["model"],
            row["language"],
            row["prompt_type"],
            row["paired_harmful_scenario_id"],
        ): row
        for row in rows
    }
    scenario_ids = [case["harmful_scenario_id"] for case in CONTROL_CASES]
    models = sorted({row["model"] for row in rows})
    output = []
    for prompt in PROMPT_TYPES:
        for language in ("hi", "pt", "es", "zh"):
            pairs = []
            eligible_counts = []
            for model in models:
                eligible = []
                for scenario_id in scenario_ids:
                    target = indexed.get(
                        (model, language, prompt, scenario_id)
                    )
                    english = indexed.get(
                        (model, "en", prompt, scenario_id)
                    )
                    if (
                        target is None
                        or english is None
                        or not target["pair_valid"]
                        or not english["pair_valid"]
                    ):
                        continue
                    if conditional and not (
                        target["benign_success"] and english["benign_success"]
                    ):
                        continue
                    eligible.append((target, english))
                if eligible:
                    pairs.append((
                        sum(row["harmful_violation"] for row, _ in eligible)
                        / len(eligible),
                        sum(row["harmful_violation"] for _, row in eligible)
                        / len(eligible),
                    ))
                    eligible_counts.append(len(eligible))
            deltas = np.array([left - right for left, right in pairs], dtype=float)
            if not len(deltas) or np.allclose(deltas, 0):
                pvalue = 1.0
            else:
                pvalue = float(
                    wilcoxon(
                        [left for left, _ in pairs],
                        [right for _, right in pairs],
                        alternative="greater",
                        zero_method="wilcox",
                        method="auto",
                    ).pvalue
                )
            output.append({
                "analysis": (
                    "conditional_on_benign_success"
                    if conditional
                    else f"unconditional_{len(CONTROL_CASES)}_pair_subset"
                ),
                "prompt_type": prompt,
                "comparison": f"{language}>en",
                "models": len(pairs),
                "wins": int(sum(delta > 0 for delta in deltas)),
                "losses": int(sum(delta < 0 for delta in deltas)),
                "ties": int(sum(delta == 0 for delta in deltas)),
                "eligible_pairs_min": (
                    min(eligible_counts) if eligible_counts else 0
                ),
                "eligible_pairs_max": (
                    max(eligible_counts) if eligible_counts else 0
                ),
                "mean_difference_pp": float(np.mean(deltas) * 100) if len(deltas) else None,
                "p_one_sided_wilcoxon": pvalue,
            })
    return output


def cluster_bootstrap(
    rows: List[dict],
    predicate: Callable[[dict], bool],
    numerator: Callable[[dict], bool],
    draws: int = 5000,
    seed: int = 42,
) -> Optional[List[float]]:
    eligible = [row for row in rows if predicate(row)]
    models = sorted({row["model"] for row in eligible})
    if not models:
        return None
    grouped = {
        model: [row for row in eligible if row["model"] == model]
        for model in models
    }
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sampled = [rng.choice(models) for _ in models]
        denominator = sum(len(grouped[model]) for model in sampled)
        count = sum(
            sum(bool(numerator(row)) for row in grouped[model])
            for model in sampled
        )
        estimates.append(count / denominator)
    return [
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    ]


def add_cluster_intervals(rows: List[dict], summary_rows: List[dict]) -> None:
    for summary in summary_rows:
        subset = [
            row
            for row in rows
            if row["language"] == summary["language"]
            and row["prompt_type"] == summary["prompt_type"]
        ]
        summary["benign_completion_cluster95"] = cluster_bootstrap(
            subset,
            predicate=lambda row: row["pair_valid"],
            numerator=lambda row: row["benign_success"],
        )
        summary["conditional_hvr_cluster95"] = cluster_bootstrap(
            subset,
            predicate=lambda row: row["pair_valid"] and row["benign_success"],
            numerator=lambda row: row["harmful_violation"],
        )


def add_summary_cluster_intervals(rows: List[dict], summary: dict) -> None:
    summary["benign_completion_cluster95"] = cluster_bootstrap(
        rows,
        predicate=lambda row: row["pair_valid"],
        numerator=lambda row: row["benign_success"],
    )
    summary["conditional_hvr_cluster95"] = cluster_bootstrap(
        rows,
        predicate=lambda row: row["pair_valid"] and row["benign_success"],
        numerator=lambda row: row["harmful_violation"],
    )


def add_holm_adjustment(rows: List[dict]) -> None:
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["analysis"], row["prompt_type"])].append(row)
    for group in groups.values():
        ordered = sorted(group, key=lambda row: row["p_one_sided_wilcoxon"])
        running = 0.0
        total = len(ordered)
        for rank, row in enumerate(ordered):
            adjusted = min(
                1.0,
                (total - rank) * row["p_one_sided_wilcoxon"],
            )
            running = max(running, adjusted)
            row["p_holm_within_prompt"] = running


def write_pairs_csv(path: Path, rows: List[dict]) -> None:
    fields = [
        "model", "language", "prompt_type", "scenario_id",
        "paired_harmful_scenario_id", "expected_tool", "outcome",
        "benign_success", "harmful_label", "harmful_violation", "pair_valid",
        "pair_outcome", "harmful_question_id", "error", "harmful_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def percent(value: Optional[float]) -> str:
    return "NA" if value is None else f"{100 * value:.1f}%"


def interval_percent(value: Optional[List[float]]) -> str:
    if value is None:
        return "NA"
    return f"[{100 * value[0]:.1f}%, {100 * value[1]:.1f}%]"


def render_markdown(report: dict) -> str:
    overall = report["overall"]
    lines = [
        "# Matched benign-control results",
        "",
        f"Design: `{report['design_version']}`; "
        f"{overall['total_pairs']:,} paired observations.",
        "",
        f"Overall legitimate completion is "
        f"{percent(overall['benign_completion_rate'])} "
        f"{interval_percent(overall['benign_completion_cluster95'])}; "
        f"harmful violation conditional on benign success is "
        f"{percent(overall['conditional_harmful_violation_rate'])} "
        f"{interval_percent(overall['conditional_hvr_cluster95'])}. "
        "Intervals are 95% model-cluster bootstraps.",
        "",
        "| Prompt | Language | Benign completion | Harmful violation | "
        "Policy-clean completion | Harmful violation given benign success |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["by_language_prompt"]:
        lines.append(
            f"| {row['prompt_type']} | {row['language']} | "
            f"{percent(row['benign_completion_rate'])} | "
            f"{percent(row['harmful_violation_rate'])} | "
            f"{percent(row['policy_clean_completion_rate'])} | "
            f"{percent(row['conditional_harmful_violation_rate'])} |"
        )
    lines.extend([
        "",
        "## Pair outcomes",
        "",
    ])
    for key, value in sorted(report["overall"]["pair_outcomes"].items()):
        lines.append(f"- {key}: {value}")
    heldout = report["heldout_excluding_scorer_development_models"]
    lines.extend([
        "",
        "## Held-out-model sensitivity",
        "",
        "The control selection and scoring rules were debugged against "
        "pre-existing outputs from "
        f"{', '.join(sorted(SCORER_DEVELOPMENT_MODELS))}. Excluding those "
        "development models leaves "
        f"{heldout['total_pairs']:,} pairs; benign completion is "
        f"{percent(heldout['benign_completion_rate'])} "
        f"{interval_percent(heldout['benign_completion_cluster95'])}, "
        "and harmful violation "
        "conditional on benign success is "
        f"{percent(heldout['conditional_harmful_violation_rate'])} "
        f"{interval_percent(heldout['conditional_hvr_cluster95'])}.",
    ])
    lines.extend([
        "",
        "## Within-model language comparisons",
        "",
        "| Analysis | Prompt | Comparison | n models | Eligible pairs/model | "
        "W/L/T | Mean Δ (pp) | p | Holm p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["paired_language_tests"]:
        lines.append(
            f"| {row['analysis']} | {row['prompt_type']} | {row['comparison']} | "
            f"{row['models']} | {row['eligible_pairs_min']}–"
            f"{row['eligible_pairs_max']} | "
            f"{row['wins']}/{row['losses']}/{row['ties']} | "
            f"{row['mean_difference_pp']:.2f} | "
            f"{row['p_one_sided_wilcoxon']:.6g} | "
            f"{row['p_holm_within_prompt']:.6g} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benign-scored",
        type=Path,
        default=ROOT / "benign_control_scored" / "benign_scored_records.json",
    )
    parser.add_argument(
        "--consolidated-root",
        type=Path,
        default=ROOT / "eval_results_consolidated",
    )
    parser.add_argument(
        "--judge-index",
        type=Path,
        default=ROOT / "web-data" / "judge_index.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benign_control_analysis",
    )
    parser.add_argument(
        "--routing-manifest", type=Path, default=DEFAULT_ROUTING
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow partial diagnostic input; published results require the complete routed matrix.",
    )
    args = parser.parse_args()

    benign = load_benign(args.benign_scored)
    routing = json.loads(args.routing_manifest.read_text(encoding="utf-8"))
    expected_models = set(routing["exact_runnable_models"])
    expected_count = int(routing["expected_observations"])
    observed_models = {row["model"] for row in benign}
    if not observed_models.issubset(expected_models):
        raise SystemExit(
            "Benign scores include models outside the routing lock: "
            f"{sorted(observed_models - expected_models)}"
        )
    if len(benign) != expected_count and not args.allow_incomplete:
        raise SystemExit(
            f"Expected {expected_count:,} benign records, found {len(benign)}. "
            "Use --allow-incomplete only for pilot diagnostics."
        )
    if not args.allow_incomplete and observed_models != expected_models:
        raise SystemExit(
            "Full matrix model mismatch: "
            f"missing={sorted(expected_models - observed_models)}"
        )
    harmful = harmful_index(
        args.consolidated_root,
        args.judge_index,
        expected_models={row["model"] for row in benign},
    )
    pairs = join_pairs(benign, harmful)
    language_prompt = grouped_summary(pairs, ("prompt_type", "language"))
    add_cluster_intervals(pairs, language_prompt)
    overall = summarize_group(pairs)
    add_summary_cluster_intervals(pairs, overall)
    heldout_pairs = [
        row for row in pairs if row["model"] not in SCORER_DEVELOPMENT_MODELS
    ]
    heldout = summarize_group(heldout_pairs)
    add_summary_cluster_intervals(heldout_pairs, heldout)
    language_tests = (
        paired_language_tests(pairs, conditional=False)
        + paired_language_tests(pairs, conditional=True)
    )
    add_holm_adjustment(language_tests)
    report = {
        "design_version": DESIGN_VERSION,
        "overall": overall,
        "scorer_development_models": sorted(SCORER_DEVELOPMENT_MODELS),
        "heldout_excluding_scorer_development_models": heldout,
        "by_language_prompt": language_prompt,
        "by_model": grouped_summary(pairs, ("model",)),
        "by_model_language_prompt": grouped_summary(
            pairs, ("model", "prompt_type", "language")
        ),
        "by_tool": grouped_summary(pairs, ("expected_tool",)),
        "paired_language_tests": language_tests,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_records.json").write_text(
        json.dumps(pairs, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_pairs_csv(args.output_dir / "paired_records.csv", pairs)
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(report)
    (args.output_dir / "summary.md").write_text(
        markdown, encoding="utf-8"
    )
    print(markdown)


if __name__ == "__main__":
    main()
