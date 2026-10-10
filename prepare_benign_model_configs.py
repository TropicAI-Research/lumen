#!/usr/bin/env python3
"""Create provider routes and a cost estimate for the benign-control run.

The original model configs are never modified.  Twenty-eight exact snapshots
are routed through OpenRouter and the two Sabiá snapshots through Maritaca.
Only the declared model roster is configured; no substitute model is added.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from benign_control import CONTROL_CASES, LANGUAGES, PROMPT_TYPES


ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG = Path("/tmp/openrouter_tool_models.json")
DEFAULT_OUTPUT = ROOT / "configs" / "models_benign_control"
DEFAULT_MANIFEST = ROOT / "benign_control_routing.json"
DEFAULT_COST = ROOT / "BENIGN_CONTROL_COST_ESTIMATE.md"

# Config stem -> current OpenRouter model id.  These are the same model
# identities used in the original evaluation; only the API route changes.
OPENROUTER_MODELS: Dict[str, str] = {
    "claude-haiku-4.5": "anthropic/claude-haiku-4.5",
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
    "gemini-3.1-flash-lite-preview": "google/gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro-preview": "google/gemini-3.1-pro-preview",
    "gemma-4-26b": "google/gemma-4-26b-a4b-it",
    "gemma-4-31b": "google/gemma-4-31b-it",
    "gpt-4.1-mini": "openai/gpt-4.1-mini",
    "gpt-4.1": "openai/gpt-4.1",
    "gpt-5-mini": "openai/gpt-5-mini",
    "gpt-5.4": "openai/gpt-5.4",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-safeguard-20b": "openai/gpt-oss-safeguard-20b",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
    "llama-4-scout": "meta-llama/llama-4-scout",
    "minimax-m2.7": "minimax/minimax-m2.7",
    "ministral-8b": "mistralai/ministral-8b-2512",
    "mistral-large-2512": "mistralai/mistral-large-2512",
    "mistral-small-2603": "mistralai/mistral-small-2603",
    "qwen3-14b": "qwen/qwen3-14b",
    "qwen3-235b-a22b-2507": "qwen/qwen3-235b-a22b-2507",
    "qwen3-32b": "qwen/qwen3-32b",
    "qwen3-8b": "qwen/qwen3-8b",
    "qwen3.5-27b": "qwen/qwen3.5-27b",
    "qwen3.5-35b-a3b": "qwen/qwen3.5-35b-a3b",
    "qwen3.5-397b-a17b": "qwen/qwen3.5-397b-a17b",
    "qwen3.5-9b": "qwen/qwen3.5-9b",
}

OFFICIAL_MODELS = {
    "sabia4": {
        "model_name": "sabia-4",
        "api_base": "https://chat.maritaca.ai/api",
        "api_key": "$MARITACA_API_KEY",
        "prompt_price_per_1k": 0.001,
        "completion_price_per_1k": 0.004,
        "route": "maritaca_official",
    },
    "sabiazinho-4": {
        "model_name": "sabiazinho-4",
        "api_base": "https://chat.maritaca.ai/api",
        "api_key": "$MARITACA_API_KEY",
        "prompt_price_per_1k": 0.0002,
        "completion_price_per_1k": 0.0008,
        "route": "maritaca_official",
    },
}

EXCLUDED_RETIRED = {}  # The declared roster contains only the 30 evaluated models.

REQUESTS_PER_MODEL = len(CONTROL_CASES) * len(LANGUAGES) * len(PROMPT_TYPES)

# Model-specific estimates avoid applying one provider's tokenizer accounting
# to every route. For 25 OpenRouter models these are the means of the original
# pressure=none observations that actually invoked at least one tool. Sabiá
# values come from its 1,000-request benign pilot. Three highly safe models had
# no harmful pressure=none tool calls, so their values are deliberately
# conservative two-turn estimates based on their observed initial requests.
# The funded-budget margin below covers sparse cells (e.g. n=1) and retries.
TOKEN_PROFILES = {
    "claude-haiku-4.5": (13126.4, 500.0, 0, "two-turn fallback"),
    "deepseek-v3.2": (14993.8, 631.5, 6, "historical tool calls"),
    "gemini-3.1-flash-lite-preview": (8889.1, 162.8, 50, "historical tool calls"),
    "gemini-3.1-pro-preview": (2772.0, 35.0, 1, "historical tool calls"),
    "gemma-4-26b": (5800.2, 160.6, 5, "historical tool calls"),
    "gemma-4-31b": (5834.0, 131.0, 1, "historical tool calls"),
    "gpt-4.1": (3442.3, 141.8, 12, "historical tool calls"),
    "gpt-4.1-mini": (4209.7, 149.5, 80, "historical tool calls"),
    "gpt-5-mini": (1000.0, 1500.0, 0, "two-turn fallback"),
    "gpt-5.4": (3500.0, 300.0, 0, "two-turn fallback"),
    "gpt-oss-120b": (2167.0, 597.2, 10, "historical tool calls"),
    "gpt-oss-20b": (6015.2, 883.4, 14, "historical tool calls"),
    "gpt-oss-safeguard-20b": (5406.5, 1135.7, 10, "historical tool calls"),
    "llama-3.1-8b": (8541.2, 154.7, 299, "historical tool calls"),
    "llama-3.3-70b": (9202.4, 153.4, 306, "historical tool calls"),
    "llama-4-scout": (6494.6, 184.4, 38, "historical tool calls"),
    "minimax-m2.7": (6189.0, 286.0, 1, "historical tool calls"),
    "ministral-8b": (6914.9, 572.9, 104, "historical tool calls"),
    "mistral-large-2512": (7040.3, 537.9, 63, "historical tool calls"),
    "mistral-small-2603": (8101.7, 427.0, 76, "historical tool calls"),
    "qwen3-14b": (8653.3, 1369.8, 457, "historical tool calls"),
    "qwen3-235b-a22b-2507": (8292.3, 479.0, 142, "historical tool calls"),
    "qwen3-32b": (7310.4, 1146.8, 368, "historical tool calls"),
    "qwen3-8b": (3023.6, 1573.7, 436, "historical tool calls"),
    "qwen3.5-27b": (6736.2, 600.8, 6, "historical tool calls"),
    "qwen3.5-35b-a3b": (7625.9, 854.8, 44, "historical tool calls"),
    "qwen3.5-397b-a17b": (6835.0, 812.7, 6, "historical tool calls"),
    "qwen3.5-9b": (9355.4, 764.2, 44, "historical tool calls"),
    "sabia4": (1002.48, 264.689, 1000, "benign pilot"),
    "sabiazinho-4": (1070.642, 270.81, 1000, "benign pilot"),
}


def usd_per_1k(per_token: str) -> float:
    return float(per_token or 0.0) * 1000


def estimated_cost(stem: str, config: dict) -> float:
    input_tokens, output_tokens, _, _ = TOKEN_PROFILES[stem]
    return REQUESTS_PER_MODEL * (
        input_tokens / 1000 * config["prompt_price_per_1k"]
        + output_tokens / 1000 * config["completion_price_per_1k"]
    )


def render_cost_markdown(routes: Dict[str, dict], generated_at: str) -> str:
    rows = []
    route_totals = {"openrouter": 0.0, "maritaca_official": 0.0}
    for stem, config in sorted(routes.items()):
        cost = estimated_cost(stem, config)
        route = config["route"]
        route_totals[route] += cost
        rows.append((stem, route, config["model_name"], cost, TOKEN_PROFILES[stem]))
    expected = sum(route_totals.values())

    lines = [
        "# Matched benign-control cost estimate",
        "",
        f"Generated from the live OpenRouter tool-capable catalog at `{generated_at}`.",
        "",
        f"Workload: {len(CONTROL_CASES)} cases × {len(LANGUAGES)} languages × "
        f"{len(PROMPT_TYPES)} prompts = {REQUESTS_PER_MODEL} requests per model.",
        f"Exact rerunnable set: {len(routes)} models ({len(OPENROUTER_MODELS)} OpenRouter + "
        f"{len(OFFICIAL_MODELS)} Maritaca), or {len(routes) * REQUESTS_PER_MODEL:,} requests.",
        "",
        "Token estimates are model-specific. Where available, they use the original "
        "pressure=none observations in which that model actually called a tool; the "
        "two Sabiá routes use their 1,000-request benign pilots. Models with no such "
        "historical tool call use a conservative two-turn fallback. Sparse historical "
        "cells and retries are why the funded budget is larger than the point estimate.",
        "",
        "## Budget",
        "",
        f"- OpenRouter inference: **${route_totals['openrouter']:.2f}**",
        f"- Maritaca inference: **${route_totals['maritaca_official']:.2f}**",
        f"- Expected inference total: **${expected:.2f}**",
        f"- Recommended funded budget (1.5×): **${expected * 1.5:.2f}**",
        f"- Hard contingency (2×): **${expected * 2:.2f}**",
        "",
        "The estimate excludes taxes and credit-purchase/payment fees. OpenRouter's "
        "inference catalog prices are per token; its current credit-purchase fee may "
        "add a separate charge. Retries and unusually long reasoning/tool loops are "
        "the main source of upward variance.",
        "",
        "## Per-model expected inference cost",
        "",
        "| Config | Route | API model | Avg input/output | Basis | "
        f"{REQUESTS_PER_MODEL} requests |",
        "|---|---|---|---:|---|---:|",
    ]
    for stem, route, model, cost, profile in sorted(rows, key=lambda row: -row[3]):
        input_tokens, output_tokens, sample_n, source = profile
        basis = f"{source}, n={sample_n}" if sample_n else source
        lines.append(
            f"| {stem} | {route} | `{model}` | "
            f"{input_tokens:.0f}/{output_tokens:.0f} | {basis} | ${cost:.4f} |"
        )
    if EXCLUDED_RETIRED:
        lines.extend(["", "## Unavailable declared models", "",
                      "Unavailable model snapshots are not silently replaced by newer models.", ""])
        for stem, detail in EXCLUDED_RETIRED.items():
            lines.append(f"- `{stem}`: {detail['reason']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openrouter-catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cost-report", type=Path, default=DEFAULT_COST)
    args = parser.parse_args()

    catalog_data = json.loads(args.openrouter_catalog.read_text(encoding="utf-8"))
    catalog = {model["id"]: model for model in catalog_data["data"]}
    generated_at = datetime.now(timezone.utc).isoformat()

    routes: Dict[str, dict] = {}
    snapshot = {}
    for stem, model_id in sorted(OPENROUTER_MODELS.items()):
        if model_id not in catalog:
            raise SystemExit(f"OpenRouter tool catalog does not contain {model_id}")
        model = catalog[model_id]
        supported = set(model.get("supported_parameters") or [])
        if "tools" not in supported or "tool_choice" not in supported:
            raise SystemExit(f"{model_id} lacks required tools/tool_choice support")
        config = {
            "model_name": model_id,
            "api_base": "https://openrouter.ai/api/v1",
            "api_key": "$OPENROUTER_API_KEY",
            "prompt_price_per_1k": usd_per_1k(model["pricing"].get("prompt", "0")),
            "completion_price_per_1k": usd_per_1k(
                model["pricing"].get("completion", "0")
            ),
            "route": "openrouter",
        }
        routes[stem] = config
        snapshot[stem] = {
            "id": model_id,
            "canonical_slug": model.get("canonical_slug"),
            "pricing": model.get("pricing"),
            "supported_parameters": model.get("supported_parameters"),
        }
    routes.update(OFFICIAL_MODELS)
    if set(routes) != set(TOKEN_PROFILES):
        raise SystemExit(
            "Token profile/model route mismatch: "
            f"missing={sorted(set(routes) - set(TOKEN_PROFILES))}, "
            f"stale={sorted(set(TOKEN_PROFILES) - set(routes))}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for existing in args.output_dir.glob("*.json"):
        if existing.stem not in routes:
            existing.unlink()
    for stem, config in sorted(routes.items()):
        (args.output_dir / f"{stem}.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "generated_at_utc": generated_at,
        "catalog_source": "https://openrouter.ai/api/v1/models?supported_parameters=tools",
        "exact_runnable_models": sorted(routes),
        "openrouter_models": sorted(OPENROUTER_MODELS),
        "official_models": sorted(OFFICIAL_MODELS),
        "excluded_retired_models": EXCLUDED_RETIRED,
        "expected_observations": len(routes) * REQUESTS_PER_MODEL,
        "estimated_cost_usd_by_model": {
            stem: estimated_cost(stem, config)
            for stem, config in sorted(routes.items())
        },
        "openrouter_snapshot": snapshot,
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.cost_report.write_text(
        render_cost_markdown(routes, generated_at),
        encoding="utf-8",
    )
    print(f"Wrote {len(routes)} exact model configs to {args.output_dir}")
    print(f"Excluded retired snapshots: {', '.join(sorted(EXCLUDED_RETIRED))}")
    print(f"Routing manifest: {args.manifest}")
    print(f"Cost report: {args.cost_report}")


if __name__ == "__main__":
    main()
