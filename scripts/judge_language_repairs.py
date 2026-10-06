#!/usr/bin/env python3
"""Fill semantic labels with the archived o4-mini rubric and medium reasoning.

--prepare is offline. Paid execution is explicit, concurrency- and budget-limited,
and never retries a saved attempt. It must not overlap another paid runner:
both account for the same campaign ledger, but separate processes cannot
reserve its budget atomically.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.canonical_panel import iter_cases, compact, digest, LABELS
from scripts.run_language_repairs import OUT, Budget, api_post, env_values, redact_provider_data, save, acquire_campaign_lock

PLAN = OUT / "judge_plan.json"
RUNS = OUT / "judge_runs"
MODEL = "openai/o4-mini"
CANONICAL_MODEL = "openai/o4-mini-2025-04-16"
PROMPT_PRICE, COMPLETION_PRICE = 1.1e-6, 4.4e-6


def original_protocol():
    path = ROOT / "data/provenance/original_source/judge_new_l2.py"
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if
        isinstance(node, ast.FunctionDef) and node.name in {"build_user_message", "parse_response"}
        or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "JUDGE_SYSTEM" for t in node.targets)]
    namespace = {"json": json}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def prepare():
    protocol = original_protocol()
    tasks = {}
    if PLAN.exists():
        tasks = {r["key"]: r for r in json.loads(PLAN.read_text())["tasks"]}
    initial_count = len(tasks)
    for case in iter_cases("corrected"):
        row = compact(case)
        if not row["needs_judgment"]:
            continue
        task = {"key": case["key"], "source_path": case["source_path"],
            "source_record_sha256": case["record_sha256"],
            "reason": "corrected_input_response" if case["repaired"] else "historical_L2_coverage_gap",
            "messages": [{"role": "system", "content": protocol["JUDGE_SYSTEM"]},
                         {"role": "user", "content": protocol["build_user_message"](case["record"])}]}
        if task["key"] in tasks and tasks[task["key"]] != task:
            raise ValueError("Do not overwrite a frozen judge task: " + task["key"])
        tasks[task["key"]] = task
    plan = {"model": MODEL, "canonical_slug": CANONICAL_MODEL,
        "reasoning": {"effort": "medium"},
        "temperature": "not_sent_as_in_original_judge", "output_budget": "provider_default_as_in_original_judge",
        "source_protocol": "data/provenance/original_source/judge_new_l2.py",
        "system_prompt_sha256": digest(protocol["JUDGE_SYSTEM"]),
        "tasks": sorted(tasks.values(), key=lambda t: (t["reason"] != "historical_L2_coverage_gap", t["key"]))}
    save(PLAN, plan)
    print(json.dumps({"tasks": len(tasks), "added": len(tasks)-initial_count,
                      "historical_backfill": sum(t["reason"] == "historical_L2_coverage_gap" for t in tasks.values()),
                      "paid_calls": 0}))


def execute_task(task, plan, credentials, budget):
    directory = RUNS / digest(task["key"])
    if (directory / "request_00.json").exists() or (directory / "result.json").exists():
        return "saved_attempt_not_retried"
    payload = {"model": plan["model"], "messages": task["messages"], "reasoning": plan["reasoning"],
        "provider": {"require_parameters": True, "allow_fallbacks": False,
                     "max_price": {"prompt": 1.21, "completion": 4.84}}}
    reserve = max(0.03, len(json.dumps(payload, ensure_ascii=False).encode()) * PROMPT_PRICE + 4096 * COMPLETION_PRICE)
    if not budget.reserve(reserve):
        return "budget_pause"
    request_hash = digest(payload)
    save(directory / "request_00.json", {"request": payload, "request_sha256": request_hash,
        "requested_reasoning": "medium", "source_record_sha256": task["source_record_sha256"],
        "created_at_utc": datetime.now(timezone.utc).isoformat()})
    turn = {"request_sha256": request_hash, "charged_or_estimated_cost_usd": reserve,
            "cost_basis": "unreconciled_request_reserve"}
    error, label, reason, response_id, served_model = None, None, None, None, None
    try:
        response = api_post("https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": "Bearer " + credentials["OPENROUTER_API_KEY"]}, json=payload, timeout=120)
        turn["http_status"] = response.status_code
        data = redact_provider_data(response.json(), credentials)
        turn["response"] = data
        usage = data.get("usage") or {}
        if usage.get("cost") is not None:
            turn["charged_or_estimated_cost_usd"] = float(usage["cost"])
            turn["cost_basis"] = "provider_reported_usage_cost"
        else:
            budget.halt()
        if not response.ok or data.get("error"):
            error = "HTTP_" + str(response.status_code)
        elif not data.get("choices"):
            error = "empty_choices"
        else:
            parsed = original_protocol()["parse_response"](data["choices"][0]["message"].get("content") or "")
            label, reason = parsed["label"], parsed["reason"]
            if label not in LABELS:
                error = "invalid_judge_label"
            response_id, served_model = data.get("id"), data.get("model")
    except Exception as exc:
        error = type(exc).__name__
        turn["transport_or_parse_error"] = error
        budget.halt()
    finally:
        save(directory / "turn_00.json", turn)
        budget.settle(reserve, turn["charged_or_estimated_cost_usd"])
    save(directory / "result.json", {"key": task["key"], "source_record_sha256": task["source_record_sha256"],
        "label": label, "reason": reason, "error": error, "request_sha256": request_hash,
        "response_id": response_id, "served_model": served_model,
        "requested_model": MODEL, "requested_reasoning": "medium", "source_reason": task["reason"],
        "total_cost": turn["charged_or_estimated_cost_usd"]})
    return "error" if error else "completed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-total-usd", type=float, default=12)
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    plan = json.loads(PLAN.read_text())
    assert plan["system_prompt_sha256"] == digest(original_protocol()["JUDGE_SYSTEM"])
    pending = [t for t in plan["tasks"] if not (RUNS / digest(t["key"]) / "request_00.json").exists()]
    if args.limit is not None:
        pending = pending[:args.limit]
    print(json.dumps({"pending_selected": len(pending), "execute": args.execute}))
    if not args.execute:
        return
    campaign_lock = acquire_campaign_lock()
    if not 0 < args.max_total_usd <= 12:
        raise SystemExit("Campaign ceiling cannot exceed the authorized US$12.")
    if not 1 <= args.workers <= 8:
        raise SystemExit("Use between one and eight workers.")
    credentials = env_values()
    budget = Budget(args.max_total_usd)
    iterator = iter(pending)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}

        def submit_next():
            if budget.halted:
                return
            task = next(iterator, None)
            if task is not None:
                futures[executor.submit(execute_task, task, plan, credentials, budget)] = task["key"]

        for _ in range(args.workers):
            submit_next()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                key = futures.pop(future)
                try:
                    status = future.result()
                except Exception:
                    # Unknown attempts remain saved and are never resampled.
                    status = "unexpected_error"
                print(json.dumps({"status": status, "key": key, "spent_or_reserved_usd": budget.spent}), flush=True)
                if status in {"error", "budget_pause", "unexpected_error"}:
                    budget.halt()
            for _ in done:
                submit_next()


if __name__ == "__main__":
    main()
