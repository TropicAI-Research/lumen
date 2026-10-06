#!/usr/bin/env python3
"""Check the saved effective repair requests against the frozen protocol.

This audits submitted payloads, not unverifiable server-side implementation.
It never makes model calls and never reads credentials.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_v3 import get_system_prompt, get_tool_schemas
from scripts.canonical_panel import digest


def main():
    base = ROOT / "data/language_repair"
    plan = json.loads((base / "execution_plan.json").read_text())
    tasks = {(r["model"], r["language"], r["prompt"], r["question"]["question_id"]): r for r in plan["tasks"]}
    amendments = {digest(json.loads(p.read_text())) for p in base.glob("provider_amendments*.json")}
    parameters, providers = Counter(), Counter()
    checked = []
    for path in sorted((base / "runs").glob("*/*/*/*/request_[0-9][0-9].json")):
        model, lang, prompt, qid = path.parts[-5:-1]
        task = tasks[(model, lang, prompt, qid)]
        route = plan["routes"][model]
        request = json.loads(path.read_text())
        payload = request["request"]
        assert int(path.stem.rsplit("_", 1)[-1]) < 5, path
        assert digest(payload) == request["request_sha256"], path
        assert payload["model"] == route["model"], path
        expected = route["inference_parameters"]
        assert request["requested_inference_parameters"] == expected, path
        for field in ["temperature", "max_tokens", "max_completion_tokens", "top_p", "top_k", "min_p", "stop", "reasoning", "reasoning_effort", "presence_penalty", "repetition_penalty"]:
            assert (field in payload) == (field in expected), (path, field)
            if field in expected:
                assert payload[field] == expected[field], (path, field)
        assert payload["tool_choice"] == "auto", path
        assert payload["tools"] == get_tool_schemas(lang), path
        assert payload["messages"][0] == {"role": "system", "content": get_system_prompt(lang, prompt)}, path
        assert payload["messages"][1] == {"role": "user", "content": task["question"]["question"]}, path
        for message in payload["messages"][2:]:
            assert message["role"] in {"assistant", "tool"}, path
            assert not any(k in message for k in ["reasoning", "reasoning_details", "reasoning_content"]), path
        amendment = request.get("routing_amendment_sha256")
        assert amendment is None or amendment in amendments, path
        if route["credential_env"] == "OPENROUTER_API_KEY":
            assert payload["provider"]["require_parameters"] is True, path
            assert payload["provider"]["allow_fallbacks"] is False, path
        parameters[json.dumps(expected, sort_keys=True)] += 1
        providers[(model, tuple(payload.get("provider", {}).get("order", [])))] += 1
        checked.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    report = {"request_count": len(checked), "effective_payload_parameters": dict(parameters),
              "routes": [{"model": k[0], "provider_order": list(k[1]), "requests": v} for k,v in sorted(providers.items())],
              "scope": "Saved submitted inference payloads in current run paths. Provider implementation is not independently observable. Archived failed attempts remain available separately.",
              "inference_recipe_matches_frozen_plan": True, "files": checked}
    (ROOT / "analysis/verification/repair_protocol_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k not in {"files","routes"}}))


if __name__ == "__main__":
    main()
