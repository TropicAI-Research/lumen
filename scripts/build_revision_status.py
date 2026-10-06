#!/usr/bin/env python3
"""Record artifact-derived progress, without network calls or paid work.

--freeze-analyses removes the working-copy banner only after every available
response has its required judgment, no runner is active, and the primary and
benign analyses have been regenerated. Missing runs remain explicit; freezing
an available-case analysis does not claim completion of unavailable cells.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_language_repair_run import summarize


def campaign_active():
    path = ROOT / "data/language_repair/.campaign.lock"
    if not path.exists():
        return False
    with path.open("r") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return False


def read(relative):
    path = ROOT / relative
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-analyses", action="store_true")
    args = parser.parse_args()
    runtime = summarize()
    active = campaign_active()
    primary = read("analysis/revision/common_statistics.json")
    corrected = read("analysis/revision/corrected_audit.json")
    benign = read("analysis/revision/benign_statistics.json")
    plan = read("data/language_repair/judge_plan.json")
    labels = [json.loads(p.read_text()) for p in (ROOT / "data/language_repair/judge_runs").glob("*/result.json")]
    valid_labels = [r for r in labels if not r.get("error") and r.get("label") in {"COMPLY", "SAY_DO_GAP", "REDIRECT", "AMBIGUOUS"}]
    ready = (not active and primary.get("audit", {}).get("counts_are_final") is True
             and corrected.get("missing_judgments") == 0
             and benign.get("available_case_analysis") is True
             and len(plan.get("tasks", [])) == len(valid_labels)
             and runtime["total_spent_or_reserved_usd"] <= 12)
    banner = ROOT / "Lumen_bench_For_NeurIPS_2026 (1)/generated/release_status.tex"
    if args.freeze_analyses:
        if not ready:
            raise SystemExit("Cannot freeze: a runner, missing judgment, or required analysis remains.")
        banner.write_text("% Analyses frozen on the explicitly documented available-response cohorts.\n")
    status = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_ready": bool(ready),
        "working_copy_banner_present": "Working copy" in banner.read_text(),
        "api_execution": {"local_paid_runner_active": active,
            "planned_affected_cells_all_32_models": 4608,
            "planned_available_model_reruns": runtime["planned"],
            "completed_affected_case_reruns": runtime["completed"],
            "failed_current_case_results": runtime["error_results"],
            "unexecuted_available_model_cases": runtime["unfinished"],
            "unavailable_exact_model_ids": runtime["unavailable_in_frozen_catalog"],
            "total_spent_or_reserved_usd": runtime["total_spent_or_reserved_usd"],
            "authorized_ceiling_usd": 12,
            "cost_by_basis_usd": runtime["inference_cost_by_basis_usd"],
            "cost_caveat": "Saved usage costs, estimates and unknown-request reserves; not a verified final invoice."},
        "judging": {"plan_tasks": len(plan.get("tasks", [])), "valid_new_labels": len(valid_labels),
            "plan_by_reason": dict(Counter(t["reason"] for t in plan.get("tasks", [])))},
        "primary_analysis": primary.get("audit", {}),
        "corrected_panel": corrected,
        "benign": {k: benign.get(k) for k in ["all_benign_trials", "all_benign_completed", "unavailable_or_unjudged_pairs", "available_case_analysis"]},
        "scope": "Corrected corpus, immutable historical responses, separate reruns, explicit missing outcomes. Portuguese-only human calibration. No arXiv submission.",
        "publication_checks": "See release build and security receipts; analysis readiness alone does not certify remote publication."
    }
    (ROOT / "revision_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    (ROOT / "data/language_repair/runtime_status.json").write_text(json.dumps(runtime, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"analysis_ready": bool(ready), "paid_runner_active": active,
        "completed_reruns": runtime["completed"], "costs_and_reserves_usd": runtime["total_spent_or_reserved_usd"],
        "new_judge_labels": len(valid_labels), "judge_plan": len(plan.get("tasks", []))}))


if __name__ == "__main__":
    main()
