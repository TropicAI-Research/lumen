#!/usr/bin/env python3
"""Audit saved usage counters; do not infer provider invoices from them."""
from collections import Counter, defaultdict
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = defaultdict(Counter)
    for path in sorted((ROOT / "eval_results_consolidated").glob("*/*/*/consolidated.json")):
        model = path.parts[-2]
        counts = rows[model]
        config = json.loads((ROOT / "configs/models" / (model + ".json")).read_text())
        for record in json.loads(path.read_text()):
            counts["observations"] += 1
            counts["api_errors"] += bool(record.get("error"))
            for field in ["prompt_tokens", "completion_tokens", "reasoning_tokens", "total_cost"]:
                counts[field] += record.get(field, 0) or 0
            counts["zero_input_and_output_counters"] += not (record.get("prompt_tokens") or record.get("completion_tokens"))
            counts["reasoning_exceeds_completion_counter"] += (record.get("reasoning_tokens", 0) or 0) > (record.get("completion_tokens", 0) or 0)
        counts["archived_input_price_per_million"] = config["prompt_price_per_1k"] * 1000
        counts["archived_output_price_per_million"] = config["completion_price_per_1k"] * 1000
    for counts in rows.values():
        base = (counts["prompt_tokens"]*counts["archived_input_price_per_million"] +
                counts["completion_tokens"]*counts["archived_output_price_per_million"])/1e6
        counts["input_plus_completion_list_price_proxy"] = base
        counts["proxy_with_reasoning_added_again"] = base + counts["reasoning_tokens"]*counts["archived_output_price_per_million"]/1e6
    report = {"n_observations": sum(v["observations"] for v in rows.values()), "models": dict(rows),
        "saved_legacy_total_cost_sum": sum(v["total_cost"] for v in rows.values()),
        "input_plus_completion_proxy_sum": sum(v["input_plus_completion_list_price_proxy"] for v in rows.values()),
        "proxy_with_reasoning_added_again_sum": sum(v["proxy_with_reasoning_added_again"] for v in rows.values()),
        "zero_input_and_output_counter_observations": sum(v["zero_input_and_output_counters"] for v in rows.values()),
        "provider_invoices_available": False,
        "interpretation": "Alternative accounting conventions, not confidence bounds or verified spend. Reasoning may already be included in completion counters; some records have missing/zero usage. Historical counters and list-price snapshots cannot establish a complete billed total or a current reproduction quote."}
    out = ROOT / "analysis/verification/historical_cost_audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "models"}, indent=2))


if __name__ == "__main__":
    main()
