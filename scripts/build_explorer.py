#!/usr/bin/env python3
"""Build the static, version-pinned research explorer without any API calls.

The canonical panel selects the exact response version; calibration remains
an explicitly separate sample. A judgment is never transferred to a different
response just because its trial key is the same. All browser data is compressed
and records are loaded in small model/language/prompt shards.
"""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metrics_reconciled import extract_tool_calls, READ_ONLY_TOOLS
from scripts.canonical_panel import iter_cases, compact, digest, key_for
from tool.generic_tools import get_tool_schemas

SITE = ROOT / "site"
DATA = SITE / "data"
# Public immutable dataset release. Do not silently follow mutable `main`.
DATASET = "Tropic-AI/lumen-bench"
REVISION = "f24c4b07689232159f5c259f7eb6b2fea7f5a6e1"
OUTCOMES = ["NO_SUBSTANTIVE_ACTION", "COMPLY", "SAY_DO_GAP", "REDIRECT", "AMBIGUOUS", "OUTPUT_ERROR"]
LANGUAGES = ["en", "pt", "es", "zh", "hi"]
PROMPTS = ["ethical", "neutral"]


def read(path):
    return json.loads((ROOT / path).read_text())


def read_lines(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def write_gzip(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(encode(value), compresslevel=9, mtime=0))


def body(record):
    """Display fields only. Full evidence stays linked at the dataset revision."""
    return {
        "question": record.get("question_text", ""),
        "system_prompt": record.get("system_prompt", ""),
        "response": record.get("assistant_response", ""),
        "calls": extract_tool_calls(record),
        "conversation": record.get("conversation") or [],
        "error": record.get("error"),
        "finish_reason": record.get("finish_reason"),
    }


def same_annotation(record, annotation):
    return (record.get("question_text") == annotation["question_text"]
            and record.get("assistant_response") == annotation["assistant_response"]
            and extract_tool_calls(record) == annotation["tool_calls"])


def record_lookup(path, cache):
    if path not in cache:
        content = (ROOT / path).read_text()
        try:
            records = json.loads(content)
        except json.JSONDecodeError:
            records = [json.loads(line) for line in content.splitlines() if line.strip()]
        if isinstance(records, dict):
            records = [records]
        cache[path] = {r["question_id"]: r for r in records}
    return cache[path]


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    models = read("configs/evaluation_models.json")["models"]
    model_ids = {m: i for i, m in enumerate(models)}
    corpus = read_lines("data/corpus/parallel_scenarios.jsonl")
    questions = []
    for lang in LANGUAGES:
        for q in read(f"questions_v3_{lang}.json"):
            questions.append({"id": q["question_id"], "scenario": q["scenario_id"],
                              "language": lang, "pressure": q["pressure_type"],
                              "category": q["expected_state"]["category"], "text": q["question"]})
    questions.sort(key=lambda q: (q["scenario"], LANGUAGES.index(q["language"]), q["id"]))
    qids = {q["id"]: i for i, q in enumerate(questions)}
    schemas = get_tool_schemas("en")
    tool_names = [t["function"]["name"] for t in schemas]
    tool_ids = {name: i for i, name in enumerate(tool_names)}
    assert len(tool_names) == 21
    historical = read("web-data/judge_index.json")
    sandbox = {r["key"]: r for r in read_lines("analysis/revision/corrected_sandbox.jsonl")}
    cross = read_lines("data/calibration/crossjudge_unique_triples.jsonl")
    human = read_lines("data/calibration/human_cases.jsonl")
    archival_cache = {}
    cross_by_key, human_by_key = {}, {}
    for index, row in enumerate(cross):
        key = key_for(row["model"], row["lang"], row["prompt"], row["question_id"])
        path = f'eval_results_consolidated/{row["lang"]}/{row["prompt"]}/{row["model"]}/consolidated.json'
        raw = record_lookup(path, archival_cache)[row["question_id"]]
        labels = [row["o4_mini_label"], row["opus_label"], row["gemini_label"]]
        row.update(id=index, key=key, labels=labels, agreement=len(set(labels)) == 1,
                   source_path=path, source_sha256=digest(raw), detail=body(raw),
                   rationale=historical.get(key, {}).get("r"), current_response_match=False)
        cross_by_key[key] = row
    for index, row in enumerate(human):
        key = key_for(row["model"], row["language"], row["prompt_type"], row["question_id"])
        row.update(id=index, key=key, agreement=row["human_label"] == row["judge_label"],
                   current_response_match=False)
        human_by_key[key] = row
    del archival_cache

    rows, shards, outcome_counts = [], defaultdict(dict), Counter()
    tool_counts, model_counts = Counter(), {m: Counter() for m in models}
    matched_cross, matched_human = 0, 0
    for case in iter_cases("corrected"):
        c = compact(case)
        raw = case["record"]
        assert raw is not None and c["input_matches_current"] and not c["needs_judgment"]
        outcome = "OUTPUT_ERROR" if not c["valid"] else case["label"] or "NO_SUBSTANTIVE_ACTION"
        calls = extract_tool_calls(raw)
        names = sorted({call["name"] for call in calls})
        # Malformed/unknown tool names are evidence too, not a reason to drop
        # a response. Reserve a separate bit; keep their literal text in detail.
        unknown = set(names) - tool_ids.keys()
        tool_mask = sum(1 << tool_ids[name] for name in names if name in tool_ids)
        if unknown:
            tool_mask |= 1 << len(tool_names)
        flags = sum((1 << i) if c[field] else 0 for i, field in enumerate(["valid", "l0", "l1", "l2"]))
        evidence = 0
        cross_match = cross_by_key.get(case["key"])
        human_match = human_by_key.get(case["key"])
        if cross_match and cross_match["source_sha256"] == c["record_sha256"]:
            cross_match["current_response_match"] = True
            evidence |= 1
            matched_cross += 1
        if human_match and same_annotation(raw, human_match):
            human_match["current_response_match"] = True
            human_match["source_path"] = case["source_path"]
            human_match["source_sha256"] = c["record_sha256"]
            evidence |= 2
            matched_human += 1
        replay = sandbox.get(case["key"])
        if replay:
            assert replay["source_record_sha256"] == c["record_sha256"]
            if replay["state_changed"]:
                evidence |= 4
        row = [model_ids[c["model"]], qids[c["question_id"]], PROMPTS.index(c["prompt"]),
               OUTCOMES.index(outcome), flags, tool_mask, evidence]
        rows.append(row)
        detail = {**body(raw), "key": c["key"], "source_path": c["source_path"],
                  "source_sha256": c["record_sha256"], "label": case["label"],
                  "label_source": c["label_source"], "mechanical": {k: c[k] for k in ["valid", "l0", "l1", "l2"]},
                  "cross_id": cross_match["id"] if evidence & 1 else None,
                  "human_id": human_match["id"] if evidence & 2 else None,
                  "sandbox": replay}
        if c["label_source"] == "web-data/judge_index.json":
            detail["reason"] = historical[c["key"]].get("r", "")
        elif c["label_source"]:
            judgment = read(c["label_source"])
            assert judgment["source_record_sha256"] == c["record_sha256"]
            detail["reason"] = judgment.get("reason", "")
        shard = f'{model_ids[c["model"]]:02d}-{c["language"]}-{c["prompt"]}'
        shards[shard][c["question_id"]] = detail
        outcome_counts[outcome] += 1
        tool_counts.update(names)
        model_counts[c["model"]].update([outcome])
        if len(rows) % 32400 == 0:
            print(json.dumps({"indexed_responses": len(rows)}), flush=True)
    assert len(rows) == 194400 and len({tuple(r[:3]) for r in rows}) == 194400
    assert sum(outcome_counts[l] for l in ["COMPLY", "SAY_DO_GAP"]) == 15940
    assert outcome_counts["OUTPUT_ERROR"] == 32 and outcome_counts["SAY_DO_GAP"] == 1441
    for name, records in sorted(shards.items()):
        write_gzip(DATA / "responses" / f"{name}.json.gz", records)
    del shards
    write_gzip(DATA / "index.json.gz", rows)
    write_gzip(DATA / "questions.json.gz", questions)
    write_gzip(DATA / "scenarios.json.gz", corpus)
    write_gzip(DATA / "judges.json.gz", {"cross": cross, "human": human, "human_summary": read("data/calibration/human_calibration.json")})
    write_gzip(DATA / "tools.json.gz", {"schemas": schemas, "counts": tool_counts,
                                       "read_only": sorted(READ_ONLY_TOOLS)})

    benign = read("analysis/revision/benign_paired_records.json")
    benign_sources = {key_for(r["model"], r["language"], r["prompt"], r["question_id"]): r
                      for r in read_lines("data/corpus/benign_response_index.jsonl")}
    benign_questions = {q["question_id"]: q for lang in LANGUAGES
                        for q in read(f"benign_control_questions/questions_benign_control_{lang}.json")}
    raw_cache, benign_shards, benign_index = {}, defaultdict(dict), []
    for pair in benign:
        key = key_for(pair["model"], pair["language"], pair["prompt_type"], pair["question_id"])
        source = benign_sources[key]
        raw = record_lookup(source["source_path"], raw_cache)[pair["question_id"]]
        assert digest(raw) == source["source_record_sha256"]
        entry = {**body(raw), "key": key, "score": pair,
                 "expected": benign_questions[pair["question_id"]]["expected_state"],
                 "source_path": source["source_path"], "source_sha256": source["source_record_sha256"]}
        benign_shards[pair["model"]][key] = entry
        benign_index.append({k: pair[k] for k in ["model", "language", "prompt_type", "question_id", "scenario_id",
                                                 "paired_harmful_scenario_id", "paired_harmful_category", "expected_tool",
                                                 "outcome", "benign_success", "pair_outcome", "harmful_question_id", "harmful_violation"]})
        benign_index[-1]["text"] = raw["question_text"]
        benign_index[-1]["key"] = key
    for model, records in benign_shards.items():
        write_gzip(DATA / "benign" / f"{model_ids[model]:02d}.json.gz", records)
    write_gzip(DATA / "benign.json.gz", {"rows": benign_index, "statistics": read("analysis/revision/benign_statistics.json")})
    write_gzip(DATA / "statistics.json.gz", read("analysis/revision/corrected_statistics.json"))
    provenance_files = ["configs/evaluation_models.json", "analysis/revision/corrected_records.csv.gz",
                        "data/calibration/crossjudge_unique_triples.jsonl", "data/calibration/human_cases.jsonl",
                        "analysis/revision/benign_paired_records.json", "analysis/revision/corrected_sandbox.jsonl"]
    manifest = {"schema": 1, "dataset": DATASET, "revision": REVISION,
                "models": models, "languages": LANGUAGES, "prompts": PROMPTS, "tools": tool_names,
                "outcomes": OUTCOMES, "counts": dict(outcome_counts), "model_counts": model_counts,
                "total": len(rows), "completed": len(rows) - outcome_counts["OUTPUT_ERROR"],
                "scenarios": len(corpus), "categories": sorted({q["category"] for q in questions}),
                "pressure": ["none", "authority", "urgency", "social", "justification", "incremental"],
                "l2": len(sandbox), "cross_n": len(cross), "human_n": len(human),
                "matched_cross": matched_cross, "matched_human": matched_human,
                "benign_n": len(benign), "benign_completed": sum(p["benign_success"] for p in benign),
                "source_hashes": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in provenance_files},
                "index_columns": ["model_index", "question_index", "prompt_index", "outcome_index",
                                  "mechanical_bits_valid_l0_l1_l2", "tool_bits", "evidence_bits_cross_human_state_change"]}
    (DATA / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"responses": len(rows), "cross_sample": len(cross), "human_sample": len(human),
                      "cross_current_matches": matched_cross, "human_current_matches": matched_human,
                      "benign_pairs": len(benign), "site_bytes": sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file())}), flush=True)


if __name__ == "__main__":
    main()
