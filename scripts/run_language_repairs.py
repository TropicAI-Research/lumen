#!/usr/bin/env python3
"""Resume only corrected-language cases, retaining the original inference recipe.

No paid call is made without --execute. Raw requests (without credentials),
responses, usage, and served model identifiers are saved per turn. Existing
attempts, including failed ones, are never retried automatically.
"""
from __future__ import annotations

import argparse
import atexit
import copy
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_v3 import get_system_prompt, detect_refusal
from tool.generic_tools import get_tool_schemas
from tool.simulated_responses import execute_simulated_tool

OUT = ROOT / "data/language_repair"
RUNS = OUT / "runs"
SESSION_BUDGET = 12.0
TRANSPORT = threading.local()


def acquire_campaign_lock():
    """One paid process at a time shares this campaign's budget ledger."""
    handle = (OUT / ".campaign.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise SystemExit("Another paid campaign runner is active; no request sent.")
    atexit.register(handle.close)
    return handle


def api_post(url, **kwargs):
    # Reuse verified TLS connections per worker; never consult global account
    # credentials (.netrc) or change the user's authentication configuration.
    if not hasattr(TRANSPORT, "session"):
        TRANSPORT.session = requests.Session()
        TRANSPORT.session.trust_env = False
    # Some gateways keep a non-streaming generation alive by sending whitespace.
    # requests' read timeout resets on each byte, so timeout=120 alone can hang
    # indefinitely. Enforce the same requested HTTP deadline while reading the
    # envelope; do not change the model's token budget or reasoning settings.
    deadline_seconds = kwargs.get("timeout", 120)
    started = time.monotonic()
    response = TRANSPORT.session.post(url, stream=True, **kwargs)
    content = bytearray()
    try:
        for chunk in response.iter_content(chunk_size=1):
            if time.monotonic() - started > deadline_seconds:
                raise requests.Timeout("HTTP response exceeded its transport deadline")
            content.extend(chunk)
        response._content = bytes(content)
        response._content_consumed = True
        return response
    finally:
        response.close()


def redact_provider_data(value, credentials):
    """Remove credential echoes, including provider-masked partial keys."""
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if k.lower() in {"authorization", "api_key", "access_token"}
                    else redact_provider_data(v, credentials)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_provider_data(v, credentials) for v in value]
    if isinstance(value, str):
        for secret in credentials.values():
            if secret and len(secret) >= 8:
                value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"(?i)(API key provided:\s*)[^\s.]+", r"\1[REDACTED]", value)
        value = re.sub(r"[A-Za-z0-9_-]+\*{4,}[A-Za-z0-9_-]+", "[REDACTED]", value)
    return value


def malformed_tool_error(value):
    text = json.dumps(value, ensure_ascii=False).lower()
    return any(marker in text for marker in [
        "tool call validation failed", "tool_use_failed", "failed to parse tool",
        "invalid tool call", "did not match schema", "extra data: line",
        "failed to call a function", "failed_generation"])


def archive_failed_turn(directory):
    """Retry only an infrastructure-failed request, keeping successful turns.

    A malformed tool call is a model/protocol outcome, not permission to
    resample it. Repeated successful generations would both cost more and
    change the experiment's observed behavior.
    """
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text())
    if not result.get("error"):
        raise ValueError("Cannot retry a successful result")
    turns = sorted(directory.glob("turn_[0-9][0-9].json"))
    if not turns:
        raise ValueError("Missing failed request evidence")
    failed = turns[-1]
    event = json.loads(failed.read_text())
    http = event.get("http_status")
    body_error = (event.get("response") or {}).get("error") or {}
    provider_code = body_error.get("code") if isinstance(body_error, dict) else None
    malformed_model_output = malformed_tool_error(body_error)
    infrastructure = (event.get("transport_error") in {"SSLError", "Timeout", "ReadTimeout", "ConnectTimeout", "ConnectionError", "InterruptedUnobservedResponse"}
                      or http in {401, 403, 404, 408, 429, 500, 502, 503, 504}
                      or provider_code in {429, 500, 502, 503, 504})
    if not infrastructure or http == 400 or malformed_model_output:
        raise ValueError("This is not a retryable infrastructure failure")
    number = int(failed.stem.split("_")[-1])
    checkpoint = directory / "checkpoint.json"
    if number:
        if not checkpoint.exists() or json.loads(checkpoint.read_text())["next_turn"] != number:
            raise ValueError("Cannot resume failed turn without its exact preceding history")
    parent = OUT / "failed_turn_attempts" / directory.relative_to(RUNS)
    archive = parent / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive.mkdir(parents=True, exist_ok=False)
    for path in [result_path, failed, directory / f"request_{number:02d}.json"]:
        path.rename(archive / path.name)
    save(archive / "resume_manifest.json", {"retry_scope": "failed_request_only", "next_turn": number,
        "successful_prior_turns_retained": number, "active_directory_relative_to_campaign": str(directory.relative_to(OUT)),
        "prior_checkpoint_sha256": digest(json.loads(checkpoint.read_text())) if checkpoint.exists() else None})
    return archive


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def env_values():
    return {k.strip(): v.strip().strip(chr(34)).strip(chr(39)) for k, v in
            (line.split("=", 1) for line in (ROOT / ".env").read_text().splitlines()
             if "=" in line and not line.lstrip().startswith("#"))}


def record_interrupted_attempt(task, route, reason):
    """Account for a stopped worker without inventing a provider response.

    Administrative use only after the owning process has stopped. The result
    is an unknown API outcome, never a refusal, and is not retried by this
    runner. The request and any successful preceding turns remain untouched.
    """
    q = task["question"]
    directory = RUNS / task["model"] / task["language"] / task["prompt"] / q["question_id"]
    if (directory / "result.json").exists():
        raise ValueError("A saved result must not be overwritten")
    unmatched = [p for p in directory.glob("request_[0-9][0-9].json")
                 if not p.with_name(p.name.replace("request_", "turn_")).exists()]
    if len(unmatched) != 1:
        raise ValueError("Expected exactly one request with an unobserved response")
    request_path = unmatched[0]
    saved_request = json.loads(request_path.read_text())
    payload = saved_request["request"]
    reserve = saved_request.get("reserved_cost_usd", max(0.003, task["proxy_cost_usd"] * 2,
        len(json.dumps(payload, ensure_ascii=False).encode()) * route["prompt_price"]
        + route["inference_parameters"].get("max_tokens", 4096) * route["completion_price"]))
    save(request_path.with_name(request_path.name.replace("request_", "turn_")), {
        "request_sha256": saved_request["request_sha256"],
        "transport_error": "InterruptedUnobservedResponse", "administrative_reason": reason,
        "charged_or_estimated_cost_usd": reserve, "cost_basis": "unreconciled_request_reserve",
        "closed_at_utc": datetime.now(timezone.utc).isoformat(), "provider_response_observed": False})
    state_path = directory / "checkpoint.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    save(directory / "result.json", {"model": task["model"], "language": task["language"],
        "prompt_type": task["prompt"], "question_id": q["question_id"], "scenario_id": q["scenario_id"],
        "question_text": q["question"], "system_prompt": get_system_prompt(task["language"], task["prompt"]),
        "source_record_sha256": task["source_record_sha256"], "source_path": task["source_path"],
        "error": "InterruptedUnobservedResponse", "administrative_reason": reason,
        "conversation": state.get("messages"), "tools_called": state.get("names"),
        "api_trace": state.get("trace"), "assistant_response": None,
        "total_cost": sum(json.loads(p.read_text())["charged_or_estimated_cost_usd"]
                          for p in directory.glob("turn_[0-9][0-9].json")),
        "completed_at_utc": datetime.now(timezone.utc).isoformat()})


def prepare():
    manifest = json.loads((OUT / "repair_manifest.json").read_text())
    catalog_response = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
    catalog_response.raise_for_status()
    catalog = {m["id"]: m for m in catalog_response.json()["data"]}
    routes, unavailable = {}, []
    for path in sorted((ROOT / "configs/models").glob("*.json")):
        source = json.loads(path.read_text())
        original = source["model_name"]
        mode = "provider_defaults" if original.lower().startswith(("gpt-5", "o1", "o3", "o4")) else "temperature_0_max_tokens_1024"
        inference = {} if mode == "provider_defaults" else {"temperature": 0.0, "max_tokens": 1024}
        if "maritaca.ai" in source["api_base"]:
            route = {"api_base": source["api_base"], "model": original, "credential_env": "MARITACA_API_KEY",
                     "prompt_price": source["prompt_price_per_1k"] / 1000,
                     "completion_price": source["completion_price_per_1k"] / 1000,
                     "canonical_slug": original, "supported_parameters": None}
        else:
            slug = original if "/" in original else "openai/" + original
            match = catalog.get(slug) or next((m for m in catalog.values() if m.get("canonical_slug") == slug), None)
            if match is None:
                unavailable.append({"model": path.stem, "requested_model": original,
                                    "reason": "Exact model ID/canonical snapshot absent from current catalog; no substitute."})
                continue
            supported = set(match.get("supported_parameters", []))
            if not {"tools", "tool_choice"}.issubset(supported):
                unavailable.append({"model": path.stem, "reason": "Tool-calling route unavailable."})
                continue
            if not set(inference).issubset(supported):
                unavailable.append({"model": path.stem, "reason": "Route does not support all original inference parameters."})
                continue
            route = {"api_base": "https://openrouter.ai/api/v1", "model": match["id"],
                     "credential_env": "OPENROUTER_API_KEY", "canonical_slug": match.get("canonical_slug"),
                     "prompt_price": float(match["pricing"]["prompt"]),
                     "completion_price": float(match["pricing"]["completion"]),
                     "supported_parameters": match.get("supported_parameters")}
        routes[path.stem] = {**route, "original_model": original, "original_api_base": source["api_base"],
            "inference_parameters": inference, "thinking": "unspecified_provider_default_as_in_original_run",
            "max_iterations": 5, "tool_choice": "auto", "history": "original_assistant_content_and_tool_calls_only",
            "automatic_retries": 0}
    questions = {q["question_id"]: q for lang in ["zh", "hi"]
                 for q in json.loads((ROOT / f"questions_v3_{lang}.json").read_text())}
    tasks = []
    for model, route in routes.items():
        for lang in ["zh", "hi"]:
            for prompt in ["ethical", "neutral"]:
                path = ROOT / "eval_results_consolidated" / lang / prompt / model / "consolidated.json"
                old = {r["question_id"]: r for r in json.loads(path.read_text())}
                for change in manifest["records"]:
                    if change["language"] != lang:
                        continue
                    qid = change["question_id"]
                    record = old[qid]
                    if record["question_text"] != change["old_question"]:
                        raise ValueError(f"Recorded source input differs from repair manifest: {model} {qid}")
                    if record["system_prompt"] != get_system_prompt(lang, prompt):
                        raise ValueError(f"System prompt drift: {model} {lang} {prompt}")
                    proxy_cost = (record.get("prompt_tokens", 0) * route["prompt_price"]
                                  + (record.get("completion_tokens", 0) + record.get("reasoning_tokens", 0)) * route["completion_price"])
                    tasks.append({"model": model, "language": lang, "prompt": prompt, "question": questions[qid],
                                  "source_record_sha256": digest(record), "source_path": str(path.relative_to(ROOT)),
                                  "proxy_cost_usd": proxy_cost})
    frozen = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "routes": routes,
              "unavailable": unavailable, "tasks": tasks, "translation_model": manifest["translation_model"],
              "repair_manifest_sha256": digest(manifest),
              "system_prompt_hashes": {f"{l}/{p}": digest(get_system_prompt(l, p)) for l in ["zh", "hi"] for p in ["ethical", "neutral"]},
              "tool_schema_hashes": {l: digest(get_tool_schemas(l)) for l in ["zh", "hi"]},
              "original_history_policy": "Assistant tool-call turns contain role, content and tool_calls, followed by simulated tool messages; no added reasoning history fields."}
    path = OUT / "execution_plan.json"
    if path.exists():
        prior = json.loads(path.read_text())
        for field in ["routes", "tasks", "repair_manifest_sha256", "system_prompt_hashes", "tool_schema_hashes"]:
            if prior[field] != frozen[field]:
                raise SystemExit(f"Frozen plan differs in {field}; do not silently overwrite or mix protocols.")
    else:
        save(path, frozen)
    print(json.dumps({"runnable_models": len(routes), "tasks": len(tasks), "unavailable": unavailable,
                      "proxy_inference_cost_usd": sum(t["proxy_cost_usd"] for t in tasks)}))


class Budget:
    def __init__(self, ceiling, keep_going=False):
        self.lock = threading.Lock()
        self.ceiling = min(ceiling, SESSION_BUDGET)
        translation = OUT / "translation_response.json"
        self.spent = float(json.loads(translation.read_text()).get("usage", {}).get("cost", 0)) if translation.exists() else 0
        self.active = 0
        self.reserved = 0.0
        self.halted = False
        self.keep_going = keep_going
        for path in OUT.rglob("turn_[0-9][0-9].json"):
            self.spent += float(json.loads(path.read_text()).get("charged_or_estimated_cost_usd", 0))

    def reserve(self, amount):
        with self.lock:
            if self.halted or self.spent + self.reserved + amount > self.ceiling:
                return False
            self.reserved += amount
            return True

    def settle(self, reserved, spent):
        with self.lock:
            self.reserved -= reserved
            self.spent += spent
            if self.spent >= self.ceiling:
                self.halted = True

    def halt(self):
        with self.lock:
            self.halted = True


def validate_resumed_route(directory, state, route, allow_provider_change, schemas):
    """A provider-only amendment must not change an inference recipe or history."""
    prior_hash = state.get("route_sha256", digest(route))
    if prior_hash == digest(route):
        return
    if not allow_provider_change:
        raise ValueError("Routing changed during an unfinished conversation")
    original = json.loads((directory / "request_00.json").read_text())["request"]
    requested = {k: v for k, v in original.items()
                 if k not in {"model", "messages", "tools", "tool_choice", "provider"}}
    if original["model"] != route["model"] or requested != route["inference_parameters"]:
        raise ValueError("Provider amendment would change the model or inference recipe")
    if (original["messages"][:2] != state["messages"][:2] or original.get("tool_choice") != "auto"
            or original["tools"] != schemas):
        raise ValueError("Provider amendment would change the original prompts, schemas, or tool choice")
    save(directory / f"provider_change_before_turn_{state['next_turn']:02d}.json", {
        "prior_route_sha256": prior_hash, "new_route_sha256": digest(route),
        "next_turn": state["next_turn"], "history_sha256": digest(state["messages"]),
        "original_request_sha256": digest(original), "inference_recipe_preserved": True,
        "reason": route.get("reason", "Explicit provider-only amendment"),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat()})


def run_case(task, route, credentials, budget, allow_provider_change=False):
    model, lang, prompt = task["model"], task["language"], task["prompt"]
    q = task["question"]
    directory = RUNS / model / lang / prompt / q["question_id"]
    if (directory / "result.json").exists():
        return {"status": "reused", "model": model}
    directory.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "system", "content": get_system_prompt(lang, prompt)},
                {"role": "user", "content": q["question"]}]
    schemas = get_tool_schemas(lang)
    usage_totals = {k: 0 for k in ["prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"]}
    names, trace = [], []
    final, error, finish = "", None, None
    first_turn = 0
    checkpoint = directory / "checkpoint.json"
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state["source_record_sha256"] != task["source_record_sha256"]:
            raise ValueError("Checkpoint source differs from frozen plan")
        validate_resumed_route(directory, state, route, allow_provider_change, schemas)
        first_turn = state["next_turn"]
        messages, names, trace = state["messages"], state["names"], state["trace"]
        usage_totals = state["usage_totals"]
        final, finish = state["final"], state["finish"]
    if (directory / f"request_{first_turn:02d}.json").exists() or (directory / f"turn_{first_turn:02d}.json").exists():
        return {"status": "interrupted_attempt_not_retried", "model": model}
    start = time.monotonic()
    for turn in range(first_turn, 5):
        payload = {"model": route["model"], "messages": copy.deepcopy(messages), "tools": schemas,
                   "tool_choice": "auto", **route["inference_parameters"]}
        if route["credential_env"] == "OPENROUTER_API_KEY":
            payload["provider"] = {"require_parameters": True, "allow_fallbacks": False,
                "max_price": {"prompt": route["prompt_price"] * 1e6 * 1.10,
                              "completion": route["completion_price"] * 1e6 * 1.10}}
            if route.get("provider_order"):
                payload["provider"]["order"] = route["provider_order"]
        # This reserve is a spend-management estimate, not an inference limit.
        # Unbounded provider-default output settings are kept unchanged.
        reserve = max(0.003, task["proxy_cost_usd"] * 2,
                      len(json.dumps(payload, ensure_ascii=False).encode()) * route["prompt_price"]
                      + route["inference_parameters"].get("max_tokens", 4096) * route["completion_price"])
        if not budget.reserve(reserve):
            # No new request was sent. A future invocation can safely resume
            # from the last saved tool feedback without paying for it again.
            return {"status": "paused_before_request", "model": model,
                    "language": lang, "prompt": prompt, "question_id": q["question_id"],
                    "next_turn": turn}
        request_hash = digest(payload)
        save(directory / f"request_{turn:02d}.json", {"request": payload, "request_sha256": request_hash,
             "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "reserved_cost_usd": reserve,
             "requested_inference_parameters": route["inference_parameters"],
             "reasoning_setting": route["thinking"], "history_policy": route["history"],
             "route_sha256": digest(route), "routing_amendment_sha256": route.get("routing_amendment_sha256")})
        record = {"request_sha256": request_hash, "charged_or_estimated_cost_usd": 0.0}
        try:
            response = api_post(route["api_base"].rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + credentials[route["credential_env"]]}, json=payload, timeout=120)
            record["http_status"] = response.status_code
            try:
                data = response.json()
            except ValueError:
                data = {"error": {"message": "Non-JSON HTTP response"}}
            data = redact_provider_data(data, credentials)
            record["response"] = data
            usage = data.get("usage") or {}
            cost = usage.get("cost")
            if cost is None:
                if "prompt_tokens" in usage and "completion_tokens" in usage:
                    cost = usage["prompt_tokens"] * route["prompt_price"] + usage["completion_tokens"] * route["completion_price"]
                    record["cost_basis"] = "token_price_estimate"
                else:
                    # A failed generation may still consume tokens. Missing
                    # accounting is not evidence of zero cost.
                    cost = reserve
                    record["cost_basis"] = "unreconciled_missing_usage_reserve"
                    if not budget.keep_going:
                        budget.halt()
            else:
                record["cost_basis"] = "provider_reported_usage_cost"
            record["charged_or_estimated_cost_usd"] = float(cost)
            if not response.ok or data.get("error"):
                error = f"HTTP_{response.status_code}: " + str(data.get("error", {}))
                break
            for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                usage_totals[k] += usage.get(k, 0) or 0
            usage_totals["reasoning_tokens"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
            trace.append({"requested_model": route["model"], "served_model": data.get("model"),
                          "provider": data.get("provider"), "response_id": data.get("id"),
                          "system_fingerprint": data.get("system_fingerprint"), "request_sha256": request_hash})
            if not data.get("choices"):
                error = "empty_choices"
                break
            choice = data["choices"][0]
            message = choice["message"]
            finish = choice.get("finish_reason")
            final = message.get("content") or ""
            if not message.get("tool_calls"):
                break
            # Deliberately match the original history, not a new reasoning preset.
            calls = [{"id": c["id"], "type": "function", "function": {
                      "name": c["function"]["name"], "arguments": c["function"]["arguments"]}}
                     for c in message["tool_calls"]]
            messages.append({"role": "assistant", "content": final, "tool_calls": calls})
            for call in calls:
                fn = call["function"]
                names.append(fn["name"])
                messages.append({"role": "tool", "tool_call_id": call["id"],
                    "content": execute_simulated_tool(fn["name"], fn["arguments"], language=lang)})
        except requests.RequestException as exc:
            # An unknown request outcome is never automatically retried or
            # treated as free; reserve its estimated cost until reconciliation.
            error = type(exc).__name__
            record["transport_error"] = error
            record["charged_or_estimated_cost_usd"] = reserve
            record["cost_basis"] = "unreconciled_request_reserve"
        finally:
            save(directory / f"turn_{turn:02d}.json", record)
            budget.settle(reserve, record["charged_or_estimated_cost_usd"])
        if error:
            break
        save(checkpoint, {"source_record_sha256": task["source_record_sha256"], "route_sha256": digest(route),
            "next_turn": turn + 1, "messages": messages, "names": names,
            "trace": trace, "usage_totals": usage_totals, "final": final, "finish": finish})
    result = {"model": model, "question_id": q["question_id"], "scenario_id": q["scenario_id"],
        "category": q["expected_state"]["category"], "language": lang, "prompt_type": prompt,
        "pressure_type": q["pressure_type"], "explicitness": q["explicitness"], "moral_clarity": q["moral_clarity"],
        "question_text": q["question"], "system_prompt": get_system_prompt(lang, prompt),
        "tool_called": bool(names), "tools_called": names, "assistant_response": final,
        "refused": detect_refusal(final, lang) if not names else False, "conversation": messages,
        "api_trace": trace, "error": error, "finish_reason": finish,
        "execution_time": time.monotonic() - start, **usage_totals,
        "source_record_sha256": task["source_record_sha256"], "source_path": task["source_path"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    result["total_cost"] = sum(json.loads(p.read_text())["charged_or_estimated_cost_usd"] for p in directory.glob("turn_*.json"))
    save(directory / "result.json", result)
    return {"status": "error" if error else "completed", "model": model, "language": lang,
            "prompt": prompt, "question_id": q["question_id"], "error": error,
            "cost_usd": result["total_cost"], "tools": len(names), "finish_reason": finish}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--route-overrides", type=Path)
    parser.add_argument("--retry-failed", action="store_true",
                        help="Archive only an infrastructure-failed request before a supervised retry; retain all successful turns.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--per-model-workers", type=int, default=1,
                        help="Transport concurrency only; never changes a request or its inference settings.")
    parser.add_argument("--limit-per-model", type=int,
                        help="Select at most this many pending cases per model for an endpoint check.")
    parser.add_argument("--allow-provider-change-on-resume", action="store_true",
                        help="Explicitly allow a recorded provider-only amendment while preserving successful prior turns.")
    parser.add_argument("--keep-going", action="store_true",
                        help="Record errors without retrying; pause a model after three consecutive infrastructure errors. Malformed model outputs stay failed outcomes but do not imply endpoint downtime.")
    parser.add_argument("--max-total-usd", type=float, default=0.5)
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    if args.execute:
        campaign_lock = acquire_campaign_lock()
    plan = json.loads((OUT / "execution_plan.json").read_text())
    if args.route_overrides:
        amendments = json.loads(args.route_overrides.read_text())
        for model, changes in amendments["routes"].items():
            allowed = {"provider_order", "prompt_price", "completion_price", "reason"}
            if not set(changes).issubset(allowed) or model not in plan["routes"]:
                raise SystemExit("Route amendments cannot alter model IDs or inference settings.")
            plan["routes"][model].update(changes)
            plan["routes"][model]["routing_amendment_sha256"] = digest(amendments)
    manifest = json.loads((OUT / "repair_manifest.json").read_text())
    assert digest(manifest) == plan["repair_manifest_sha256"], "Repair manifest drift"
    for lang, hash_value in plan["tool_schema_hashes"].items():
        assert digest(get_tool_schemas(lang)) == hash_value, "Tool schema drift"
    for key, hash_value in plan["system_prompt_hashes"].items():
        assert digest(get_system_prompt(*key.split("/"))) == hash_value, "System prompt drift"
    tasks = [t for t in plan["tasks"] if not args.models or t["model"] in args.models]
    if args.smoke:
        tasks = [t for t in tasks if t["question"]["scenario_id"] == "whistle_document_destruction"
                 and t["question"]["pressure_type"] == "none"]
    if args.retry_failed:
        if not args.models:
            raise SystemExit("A supervised retry requires explicit --models selection.")
        for task in tasks:
            directory = RUNS / task["model"] / task["language"] / task["prompt"] / task["question"]["question_id"]
            result_path = directory / "result.json"
            if result_path.exists() and json.loads(result_path.read_text()).get("error"):
                print(json.dumps({"failed_attempt_to_archive": str(directory.relative_to(ROOT)),
                                  "execute": args.execute}), flush=True)
                if args.execute:
                    archive_failed_turn(directory)
    pending = [t for t in tasks if not (RUNS / t["model"] / t["language"] / t["prompt"] / t["question"]["question_id"] / "result.json").exists()]
    if args.limit_per_model is not None:
        if args.limit_per_model < 1:
            raise SystemExit("--limit-per-model must be positive")
        selected, counts = [], Counter()
        for task in pending:
            if counts[task["model"]] < args.limit_per_model:
                selected.append(task)
                counts[task["model"]] += 1
        pending = selected
    print(json.dumps({"selected": len(tasks), "pending": len(pending), "workers": args.workers,
                      "max_total_usd": args.max_total_usd, "execute": args.execute}), flush=True)
    if not args.execute:
        return
    if not 0 < args.max_total_usd <= SESSION_BUDGET:
        raise SystemExit("Explicit ceiling must be positive and at most the authorized US$12.")
    if not 1 <= args.workers <= 8:
        raise SystemExit("Use between one and eight workers.")
    if not 1 <= args.per_model_workers <= args.workers:
        raise SystemExit("Per-model concurrency must be positive and no greater than total workers.")
    credentials = env_values()
    for task in pending:
        if not credentials.get(plan["routes"][task["model"]]["credential_env"]):
            raise SystemExit("Missing credential for " + task["model"])
    budget = Budget(args.max_total_usd, keep_going=args.keep_going)
    queue = deque(pending)
    active_models = Counter()
    errors_by_model = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}

        def submit_next():
            if budget.halted:
                return
            # Model-level concurrency is explicit; the default remains one.
            for _ in range(len(queue)):
                task = queue.popleft()
                model = task["model"]
                if errors_by_model[model] >= 3:
                    continue
                if active_models[model] >= args.per_model_workers:
                    queue.append(task)
                    continue
                active_models[model] += 1
                future = executor.submit(run_case, task, plan["routes"][model], credentials, budget,
                                         args.allow_provider_change_on_resume)
                futures[future] = model
                return

        for _ in range(args.workers):
            submit_next()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                model = futures.pop(future)
                active_models[model] -= 1
                try:
                    outcome = future.result()
                except Exception as exc:
                    budget.halt()
                    # Do not print exception payloads that may contain a request.
                    outcome = {"status": "unexpected_error", "type": type(exc).__name__}
                print(json.dumps({**outcome, "cumulative_spent_or_reserved_usd": budget.spent}, ensure_ascii=False), flush=True)
                if outcome["status"] == "completed" or malformed_tool_error(outcome.get("error")):
                    errors_by_model[model] = 0
                if outcome["status"] == "error" and not malformed_tool_error(outcome.get("error")):
                    errors_by_model[model] += 1
                    if errors_by_model[model] == 3:
                        print(json.dumps({"status": "model_paused_after_three_consecutive_infrastructure_errors", "model": model}), flush=True)
                if (outcome["status"] == "error" and not args.keep_going) or outcome["status"] in {
                        "unexpected_error", "interrupted_attempt_not_retried", "paused_before_request"}:
                    budget.halt()
            for _ in done:
                submit_next()
    print(json.dumps({"spent_or_reserved_usd": budget.spent, "halted": budget.halted}), flush=True)


if __name__ == "__main__":
    main()
