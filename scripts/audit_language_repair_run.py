#!/usr/bin/env python3
"""Summarize saved repair requests and costs offline, without credentials."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/language_repair"


def summarize():
    plan = json.loads((OUT / "execution_plan.json").read_text())
    by_model = defaultdict(Counter)
    attempts = []
    cost_by_basis = Counter()
    translation = json.loads((OUT / "translation_response.json").read_text())
    translation_cost = float(translation.get("usage", {}).get("cost", 0))
    for path in sorted(OUT.rglob("turn_[0-9][0-9].json")):
        turn = json.loads(path.read_text())
        basis = turn.get("cost_basis", "unknown")
        cost_by_basis[basis] += turn.get("charged_or_estimated_cost_usd", 0)
    completed_pilot = defaultdict(list)
    for task in plan["tasks"]:
        model = task["model"]
        q = task["question"]
        by_model[model]["planned"] += 1
        path = OUT / "runs" / model / task["language"] / task["prompt"] / q["question_id"] / "result.json"
        if not path.exists():
            by_model[model]["unfinished"] += 1
            continue
        row = json.loads(path.read_text())
        status = "error" if row.get("error") else "completed"
        by_model[model][status] += 1
        by_model[model]["recorded_or_reserved_cost_usd"] += row.get("total_cost", 0)
        if status == "error":
            attempts.append({"model": model, "language": task["language"], "prompt": task["prompt"],
                             "question_id": q["question_id"], "error": row["error"],
                             "result_path": str(path.relative_to(ROOT))})
        elif q["scenario_id"] == "whistle_document_destruction" and q["pressure_type"] == "none":
            completed_pilot[model].append(row.get("total_cost", 0))
    projected = []
    for model, counts in by_model.items():
        pilot = completed_pilot[model]
        counts["pilot_completed"] = len(pilot)
        # This single scenario is not representative: show it as a diagnostic,
        # not an upper bound or a guarantee of affordable completion.
        projected.append({"model": model, "pilot_n": len(pilot),
                          "pilot_mean_case_usd": sum(pilot)/len(pilot) if pilot else None,
                          "pilot_scaled_144_case_usd": sum(pilot)*144/len(pilot) if pilot else None})
    summary = {"updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "planned": len(plan["tasks"]), "models": len(plan["routes"]),
        "completed": sum(v["completed"] for v in by_model.values()),
        "error_results": sum(v["error"] for v in by_model.values()),
        "unfinished": sum(v["unfinished"] for v in by_model.values()),
        "translation_cost_usd": translation_cost, "inference_cost_by_basis_usd": dict(cost_by_basis),
        "total_spent_or_reserved_usd": translation_cost + sum(cost_by_basis.values()),
        "authorized_total_ceiling_usd": 12,
        "failed_attempt_archives": len(list((OUT / "failed_attempts").rglob("result.json"))),
        "by_model": dict(by_model), "current_errors": attempts,
        "pilot_cost_diagnostic_not_bound": projected,
        "unavailable_in_frozen_catalog": plan["unavailable"]}
    return summary


if __name__ == "__main__":
    result = summarize()
    path = OUT / "runtime_status.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ["by_model", "current_errors", "pilot_cost_diagnostic_not_bound"]}, indent=2))
    for model, counts in result["by_model"].items():
        print(json.dumps({"model": model, **counts}))
