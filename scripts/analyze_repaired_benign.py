#!/usr/bin/env python3
"""Rejoin fixed benign observations to the corrected harmful response versions.

No benign response is regenerated or rescored. Completion across all 3,300
benign observations is distinct from the denominator of valid harmful pairs.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analyze_benign_control as original
from benign_control import CONTROL_CASES
from scripts.canonical_panel import iter_cases, compact, digest
from scripts.audit_benign_sources import relative_source


def load_versioned_pairs():
    """Resolve against saved responses now, never a stale aggregate report."""
    benign = original.load_benign(ROOT / "benign_control_scored/benign_scored_records.json")
    # Portable pointers in this derived join; the frozen score file is unchanged.
    benign = [{**r, "result_file": relative_source(r["result_file"])} for r in benign]
    models = {r["model"] for r in benign}
    scenarios = {case["harmful_scenario_id"] for case in CONTROL_CASES}
    harmful = {}
    for case in iter_cases("corrected"):
        if case["model"] not in models or case["scenario_id"] not in scenarios or case["pressure"] != "none":
            continue
        row = compact(case)
        valid = row["semantic_valid"] and row["input_matches_current"]
        key = (case["model"], case["language"], case["prompt"], case["scenario_id"])
        harmful[key] = {"harmful_question_id": case["question_id"],
            "harmful_label": case["label"] if row["l2"] else "NO_SUBSTANTIVE_ACTION" if valid else None,
            "harmful_violation": row["harmful"] if valid else None, "harmful_valid": valid,
            "harmful_tool_called_l0": row["l0"], "harmful_error": None if valid else case["status"],
            "harmful_source_path": case["source_path"], "harmful_source_record_sha256": case["record_sha256"],
            "harmful_label_source": case["label_source"], "harmful_reexecuted": case["repaired"],
            "harmful_judgment_pending": row["needs_judgment"]}
    pairs = original.join_pairs(benign, harmful)
    for row in pairs:
        if not row["pair_valid"]:
            row["pair_outcome"] = "excluded_missing_or_invalid_outcome"
    return benign, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--available-case", action="store_true",
                        help="Report available pairs as a defined cohort, excluding documented missing/API-failed outcomes. Never allows pending judgments.")
    args = parser.parse_args()
    benign, pairs = load_versioned_pairs()
    missing = sum(not r["pair_valid"] for r in pairs)
    if args.available_case and any(r.get("harmful_judgment_pending") for r in pairs):
        raise SystemExit("Available-case analysis requires all eligible semantic judgments first.")
    if missing and not (args.allow_incomplete or args.available_case):
        raise SystemExit(f"{missing} unavailable or unjudged harmful outcomes: no final paired results generated.")
    overall = original.summarize_group(pairs)
    original.add_summary_cluster_intervals(pairs, overall)
    heldout_pairs = [r for r in pairs if r["model"] not in original.SCORER_DEVELOPMENT_MODELS]
    heldout = original.summarize_group(heldout_pairs)
    original.add_summary_cluster_intervals(heldout_pairs, heldout)
    tests = original.paired_language_tests(pairs, conditional=False) + original.paired_language_tests(pairs, conditional=True)
    original.add_holm_adjustment(tests)
    per_model = original.grouped_summary(pairs, ("model",))
    for row in per_model:
        fixed = [r for r in benign if r["model"] == row["model"]]
        row["all_benign_trials"] = len(fixed)
        row["all_benign_completed"] = sum(r["benign_success"] for r in fixed)
    report = {"panel": "corrected", "counts_are_final": missing == 0,
        "available_case_analysis": args.available_case,
        "paired_records_sha256": digest(pairs),
        "excluded_pair_ids": [{k:r.get(k) for k in ["model","language","prompt_type","harmful_question_id","harmful_error"]}
                              for r in pairs if not r["pair_valid"]],
        "benign_rescored_or_rerun": False, "all_benign_trials": len(benign),
        "all_benign_completed": sum(r["benign_success"] for r in benign),
        "unavailable_or_unjudged_pairs": missing, "overall_paired": overall,
        "excluding_scorer_development_models": heldout, "by_model": per_model,
        "by_language_prompt": original.grouped_summary(pairs, ("prompt_type", "language")),
        "paired_language_tests": tests,
        "interpretation": "Benign completion uses all fixed benign observations. Harmful and conditional rates use only pairs with a valid versioned harmful outcome; unknown outcomes are not refusals."}
    out = ROOT / "analysis/revision"
    out.mkdir(parents=True, exist_ok=True)
    (out / "benign_paired_records.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + "\n")
    (out / "benign_statistics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k not in {"by_model", "by_language_prompt", "paired_language_tests"}}, indent=2))


if __name__ == "__main__":
    main()
