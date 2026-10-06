#!/usr/bin/env python3
"""Read-only provider preflight; never send an inference request."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/language_repair"


def check(item):
    model, route = item
    if route["credential_env"] != "OPENROUTER_API_KEY":
        return {"model": model, "direct_provider": True, "route": route}
    response = requests.get("https://openrouter.ai/api/v1/models/" + route["model"] + "/endpoints", timeout=30)
    response.raise_for_status()
    data = response.json()["data"]
    required = {"tools", "tool_choice", *route["inference_parameters"]}
    candidates = []
    for endpoint in data["endpoints"]:
        supported = set(endpoint.get("supported_parameters", []))
        if not required.issubset(supported) or endpoint.get("status") != 0:
            continue
        price = endpoint["pricing"]
        candidates.append({"provider_name": endpoint["provider_name"], "tag": endpoint["tag"],
            "prompt_price": float(price["prompt"]), "completion_price": float(price["completion"]),
            "quantization": endpoint.get("quantization"), "supported_parameters": sorted(supported)})
    candidates.sort(key=lambda x: 3000*x["prompt_price"] + 500*x["completion_price"])
    return {"model": model, "requested_model": route["model"], "required_parameters": sorted(required),
            "endpoints": data["endpoints"], "compatible_routes": candidates}


def main():
    plan = json.loads((OUT / "execution_plan.json").read_text())
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "inference_calls": 0, "models": []}
    with ThreadPoolExecutor(max_workers=6) as executor:
        for row in executor.map(check, plan["routes"].items()):
            report["models"].append(row)
            candidates = row.get("compatible_routes", [])
            print(json.dumps({"model": row["model"], "direct": row.get("direct_provider", False),
                "compatible_routes": len(candidates), "cheapest_compatible": candidates[0] if candidates else None}), flush=True)
    path = OUT / "endpoint_preflight.json"
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
