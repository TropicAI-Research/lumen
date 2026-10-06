#!/usr/bin/env python3
"""Replay stored calls only: no model API and no real external tool action."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metrics_reconciled import extract_tool_calls
from sandbox_env import simulate_case
from scripts.canonical_panel import iter_cases, compact
from metrics_reconciled import is_api_error

DEFENSIVE = {"message_to_reporting_channel", "report_to_defensive_channel"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=["recorded", "common", "corrected"], default="common")
    args = parser.parse_args()
    out = ROOT / "analysis/revision"
    out.mkdir(parents=True, exist_ok=True)
    label_counts = defaultdict(Counter)
    consequences = Counter()
    total = changed = missing_labels = 0
    archived_error_histories = 0
    historical = json.loads((ROOT / "web-data/judge_index.json").read_text()) if args.panel == "recorded" else {}
    with (out / (args.panel + "_sandbox.jsonl")).open("w") as stream:
        for case in iter_cases(args.panel):
            if args.panel == "recorded":
                # Reproduce an archived artifact, including its legacy selection.
                # This is not the corrected primary outcome or denominator.
                if case["key"] not in historical:
                    continue
                case["label"] = historical[case["key"]]["j"]
                archived_error_histories += is_api_error(case["record"])
            else:
                row = compact(case)
                if not row["valid"] or not row["l2"]:
                    continue
            replay = simulate_case(extract_tool_calls(case["record"]))
            observed = [e for e in replay["effects"] if e["changed"]]
            defensive_only = bool(observed) and all(e["kind"] in DEFENSIVE for e in observed)
            any_other_change = any(e["kind"] not in DEFENSIVE for e in observed)
            label = case["label"] or "UNJUDGED"
            total += 1
            changed += bool(replay["state_changed"])
            missing_labels += case["label"] is None
            label_counts[label].update({"total": 1, "state_changed": bool(replay["state_changed"]),
                "defensive_only_change": defensive_only, "other_change": any_other_change})
            consequences.update(replay["summary"])
            stream.write(json.dumps({"key": case["key"], "source_path": case["source_path"],
                "source_record_sha256": case["record_sha256"], "label": case["label"],
                **replay, "defensive_only_change": defensive_only, "other_change": any_other_change}, ensure_ascii=False) + "\n")
    report = {"panel": args.panel, "total_cases": total, "total_state_changed": changed,
        "archived_api_error_histories_in_legacy_cohort": archived_error_histories,
        "missing_semantic_labels": missing_labels, "by_label": dict(label_counts),
        "aggregate_consequences": dict(consequences),
        "sandbox_source_sha256": hashlib.sha256((ROOT / "sandbox_env.py").read_bytes()).hexdigest(),
        "interpretation": "Mechanical effects under the fixed synthetic state. Other change is not independently human-validated harmfulness, and does not predict deployment effects."}
    if args.panel == "recorded":
        expected = json.loads((ROOT / "data/calibration/sandbox_summary.json").read_text())
        assert total == expected["total_cases"]
        assert changed == expected["total_state_changed"]
        assert dict(consequences) == expected["aggregate_consequences"]
        for label, counts in expected["by_judge_label"].items():
            assert label_counts[label]["total"] == counts["total"]
            assert label_counts[label]["state_changed"] == counts["changed"]
        report["exactly_reproduces_original_counts_and_consequences"] = True
    (out / (args.panel + "_sandbox_summary.json")).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
