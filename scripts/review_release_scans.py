#!/usr/bin/env python3
"""Review redacted findings against exact, content-hashed false positives.

No broad rule exclusions. Exceptions require a verified whole-file hash,
line, and generic-api-key rule. Derived replay text must reproduce its
immutable source. Two matches occur inside typed provider-encrypted reasoning,
not authentication fields. Any new or changed finding blocks publication.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_language_repairs import save
from scripts.canonical_panel import digest
from metrics_reconciled import extract_tool_calls
from sandbox_env import simulate_case

KNOWN = {
    "eval_results_consolidated/en/ethical/gpt-5-mini/consolidated.json": (18212, "ec6d9f897e5224fc8b662d35b92ae064ac503970c58bbc3cd4ce17634bee0762", "Synthetic natural-language phrase about access dates"),
    "eval_results_consolidated/en/neutral/gpt-5-mini/consolidated.json": (12507, "71c0406211a367e476c83ad4e47f0a53b19abf2d223f8410a0157b7c036135e1", "Synthetic natural-language phrase about approved key actions"),
    "eval_results_consolidated/pt/neutral/grok-4/consolidated.json": (93269, "1adecdfac9ed42a2bc13c27e0a2d1cd6f3484b1ea2fc6dc389fd43aef9008b83", "Model-generated list of procedural requirements"),
    "analysis/revision/common_sandbox.jsonl": (11004, "9b1c9e038a6499907367dc7a70a7ce040e77d2a7c862d28d8116014c5742bab7", "Exact deterministic replay of the same synthetic procedural requirements"),
    "analysis/revision/recorded_sandbox.jsonl": (11515, "3344328722f9dd03a43f1a5c904389a36977ee7991657b8282c481353a7fd6c9", "Exact deterministic replay of the same synthetic procedural requirements"),
    "analysis/revision/corrected_sandbox.jsonl": (11660, "f9902704ff3fabb7edb5b69febcbf0f8ee2f2756d3669cdc6d4de1da4fa0e602", "Exact deterministic replay of the same synthetic procedural requirements"),
    "data/language_repair/judge_runs/cb519d217282646a65bd604a5e0073f6131b098583705d83c82196d420d4dcb9/turn_00.json": (28, "3a1def85d6bb8b2e8831d016f92e2ca1bc75763d1e10f96798f222072cf7b639", "Random key-like substring inside typed encrypted reasoning; not an authentication credential"),
    "data/language_repair/judge_runs/eb886084c977cd135f067b3fbe867972ec55facc009c76ce337a8f6476cf54fc/turn_00.json": (28, "059569fd813c534a0e4c553f1d72e515b77850f45f96c6c87b5976070526dc4c", "Random key-like substring inside typed encrypted reasoning; not an authentication credential"),
}
# The first three whole-file hashes were checked against original commit
# 7b4f0e81815733e90a1da9b21229479745fb7a88 before publication. Hashes let the
# same check run from the clean public history without importing old Git objects.


def verify_exception_sources(stage):
    for relative, (line, sha256, _) in KNOWN.items():
        data = (stage / relative).read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha256, relative
        if relative.startswith("analysis/revision/"):
            row = json.loads(data.splitlines()[line-1])
            assert row["key"] == "grok-4|pt|neutral|manipulation_roach_motel_pt_incremental_0552"
            source = json.loads((stage / row["source_path"]).read_text())
            original = next(r for r in source if r["question_id"] == row["key"].split("|")[-1])
            assert digest(original) == row["source_record_sha256"]
            replay = simulate_case(extract_tool_calls(original))
            assert all(row[k] == value for k, value in replay.items())
        elif "/judge_runs/" in relative:
            response = json.loads(data)["response"]
            assert response["model"] == "openai/o4-mini"
            detail, = response["choices"][0]["message"]["reasoning_details"]
            assert detail["type"] == "reasoning.encrypted" and detail["format"] == "openai-responses-v1"
            assert json.loads(data.splitlines()[line-1].decode().strip().removeprefix('"data": ').removesuffix(',')) == detail["data"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--tree-report", type=Path, required=True)
    parser.add_argument("--history-report", type=Path, required=True)
    parser.add_argument("--known-secret-report", type=Path, required=True)
    args = parser.parse_args()
    stage = args.directory.resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=stage, text=True).strip()
    assert subprocess.check_output(["git", "rev-list", "--count", "--all"], cwd=stage, text=True).strip() == "1"
    known_scan = json.loads(args.known_secret_report.read_text())
    assert not known_scan["matching_objects"] and not known_scan["credential_pattern_findings"]
    verify_exception_sources(stage)
    reports = {}
    for name, path in [("tree", args.tree_report), ("history", args.history_report)]:
        findings = json.loads(path.read_text())
        reviewed, unexpected = [], []
        for item in findings:
            relative = item["File"]
            if Path(relative).is_absolute():
                relative = str(Path(relative).relative_to(stage))
            accepted = (relative in KNOWN and item["RuleID"] == "generic-api-key"
                        and item["StartLine"] == KNOWN[relative][0])
            safe = {"file": relative, "rule": item["RuleID"], "line": item["StartLine"]}
            if accepted:
                safe["review_reason"] = KNOWN[relative][2]
            (reviewed if accepted else unexpected).append(safe)
        reports[name] = {"reviewed_false_positives": reviewed, "unexpected_findings": unexpected}
    passed = all(not r["unexpected_findings"] for r in reports.values())
    receipt = {"commit": head, "passed": passed, "generic_scanner": "Gitleaks 8.30.1",
        "reports": reports, "known_credential_and_shape_scan": "passed",
        "fresh_history_commits": 1, "credential_values_printed": False,
        "scope": "Exact staged files and the single-commit publication history; does not certify unrelated external copies."}
    save(ROOT / "analysis/verification/staged_history_security.json", receipt)
    print(json.dumps(receipt, indent=2))
    if not passed:
        raise SystemExit("Unexpected possible credentials: do not publish.")


if __name__ == "__main__":
    main()
