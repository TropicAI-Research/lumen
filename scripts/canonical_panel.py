#!/usr/bin/env python3
"""Join immutable historical records, repaired runs, and versioned judgments.

Missing reruns, API errors, and pending judgments are never treated as safe.
No API calls. The recorded panel is archival; the common panel removes all
six affected base scenarios in every language. The corrected panel replaces
only runs whose inputs were defective, with explicit missing cells.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metrics_reconciled import classify_mechanical, is_l2_hard_positive, is_api_error

LANGUAGES = ["pt", "en", "es", "zh", "hi"]
LABELS = {"COMPLY", "SAY_DO_GAP", "REDIRECT", "AMBIGUOUS"}
HARMFUL = {"COMPLY", "SAY_DO_GAP"}
DEFECT_IDS = {"whistle_performance_fabricate", "whistle_intimidate_witness", "whistle_leak_identity",
              "whistle_bonus_revoke", "whistle_document_destruction", "whistle_monitor_communications"}


def release_models():
    roster = json.loads((ROOT / "configs/evaluation_models.json").read_text())["models"]
    if len(roster) != 30 or len(set(roster)) != 30:
        raise ValueError("The released panel must contain exactly 30 unique models")
    return frozenset(roster)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def key_for(model, language, prompt, question_id):
    return "|".join([model, language, prompt, question_id])


def load_new_labels():
    labels = {}
    for path in sorted((ROOT / "data/language_repair/judge_runs").glob("*/result.json")):
        row = json.loads(path.read_text())
        if row.get("error") or row.get("label") not in LABELS:
            continue
        key = row["key"]
        if key in labels:
            raise ValueError("Duplicate replacement judgment: " + key)
        labels[key] = {**row, "artifact_path": str(path.relative_to(ROOT))}
    return labels


def resolve_label(record, key, repaired, historical, additions):
    """Input/response changes always invalidate the old label for that key."""
    if record is None or is_api_error(record):
        return None, None
    if key in additions:
        entry = additions[key]
        if entry["source_record_sha256"] != digest(record):
            raise ValueError("Judgment belongs to another record version: " + key)
        return entry["label"], entry.get("artifact_path", "new_judgment")
    if not repaired and key in historical:
        return historical[key]["j"], "web-data/judge_index.json"
    return None, None


def validate_infrastructure_replacement(original, replacement, task, question):
    """An error-only overlay cannot replace a successful observed response."""
    if (not is_api_error(original) or original.get("tools_called")
            or any(m.get("role") not in {"system", "user"} or m.get("tool_calls") for m in original.get("conversation", []))
            or original.get("assistant_response") != "[ERROR] Connection error."):
        raise ValueError("Only documented pre-generation connection errors may use this overlay")
    if (task["kind"] != "original_api_error" or task["source_record_sha256"] != digest(original)
            or replacement["source_record_sha256"] != digest(original)):
        raise ValueError("Infrastructure replacement source mismatch")
    if (replacement["question_text"] != question["question"]
            or replacement["question_text"] != original["question_text"]
            or replacement["system_prompt"] != original["system_prompt"]
            or replacement["question_id"] != original["question_id"]):
        raise ValueError("Infrastructure recovery changed the original request")


def iter_cases(mode="corrected"):
    if mode not in {"recorded", "common", "corrected"}:
        raise ValueError(mode)
    questions = {q["question_id"]: q for language in LANGUAGES
                 for q in json.loads((ROOT / f"questions_v3_{language}.json").read_text())}
    historical = json.loads((ROOT / "web-data/judge_index.json").read_text())
    additions = load_new_labels()
    recovery_path = ROOT / "data/infrastructure_recovery_20261009/original_error_overrides.json"
    recovery_tasks = json.loads(recovery_path.read_text())["tasks"] if recovery_path.exists() and mode != "recorded" else []
    recovery = {t["key"]: t for t in recovery_tasks}
    if len(recovery) != len(recovery_tasks):
        raise ValueError("Duplicate infrastructure replacement key")
    seen = set()
    included_models = release_models()
    for path in sorted((ROOT / "eval_results_consolidated").glob("*/*/*/consolidated.json")):
        language, prompt, model = path.parts[-4:-1]
        if model not in included_models:
            continue
        for original in json.loads(path.read_text()):
            qid = original["question_id"]
            q = questions[qid]
            scenario = q["scenario_id"]
            if mode == "common" and scenario in DEFECT_IDS:
                continue
            key = key_for(model, language, prompt, qid)
            if key in seen:
                raise ValueError("Duplicate source record: " + key)
            seen.add(key)
            defective = language in {"hi", "zh"} and scenario in DEFECT_IDS
            repaired = mode == "corrected" and defective
            recovered = False
            source_path = str(path.relative_to(ROOT))
            record = original
            if repaired:
                new_path = ROOT / "data/language_repair/runs" / model / language / prompt / qid / "result.json"
                if new_path.exists():
                    record = json.loads(new_path.read_text())
                    if record["source_record_sha256"] != digest(original):
                        raise ValueError("Repair origin mismatch: " + key)
                    if record["question_text"] != q["question"]:
                        raise ValueError("Repaired input mismatch: " + key)
                    if record["system_prompt"] != original["system_prompt"]:
                        raise ValueError("System-prompt drift: " + key)
                    source_path = str(new_path.relative_to(ROOT))
                else:
                    record = None
                    source_path = None
            elif not defective and original["question_text"] != q["question"]:
                raise ValueError("Unexpected input mismatch: " + key)
            if key in recovery:
                if defective:
                    raise ValueError("Infrastructure overlay cannot override a language replacement")
                new_path = ROOT / "data/language_repair/runs" / model / language / prompt / qid / "result.json"
                if new_path.exists():
                    record = json.loads(new_path.read_text())
                    validate_infrastructure_replacement(original, record, recovery[key], q)
                    source_path = str(new_path.relative_to(ROOT))
                    recovered = True
            # Archival statistics retain only archival labels. Other panels
            # may fill previously unjudged, unchanged source records.
            label, label_source = resolve_label(record, key, repaired or recovered, historical,
                                                {} if mode == "recorded" else additions)
            status = ("missing_corrected_run" if record is None else "api_error" if is_api_error(record)
                      else "corrected_run" if repaired else "recorded_wrong_language" if defective
                      else "recovered_api_error" if recovered
                      else "original_valid_input")
            yield {"key": key, "model": model, "language": language, "prompt": prompt,
                "question_id": qid, "scenario_id": scenario, "category": original["category"],
                "pressure": original["pressure_type"], "record": record,
                "source_path": source_path, "original_source_path": str(path.relative_to(ROOT)),
                "record_sha256": digest(record) if record is not None else None,
                "original_record_sha256": digest(original), "repaired": repaired, "recovered": recovered,
                "status": status, "label": label, "label_source": label_source,
                "input_matches_current": record is not None and record["question_text"] == q["question"]}


def compact(case):
    row = {k: v for k, v in case.items() if k != "record"}
    record = case["record"]
    flags = classify_mechanical(record) if record is not None else {
        "valid": False, "l0": None, "l1": None, "l2": None}
    row.update(flags)
    label = case["label"]
    if label is not None and label not in LABELS:
        raise ValueError("Unknown label for " + case["key"])
    if label is not None and not flags["l2"]:
        raise ValueError("Semantic label outside reconciled L2: " + case["key"])
    needs_judge = bool(flags["valid"] and flags["l2"] and label is None)
    row.update({"l2_legacy_all_arguments": is_l2_hard_positive(record) if flags["valid"] else None,
        "needs_judgment": needs_judge, "judged_cohort": label is not None,
        "semantic_valid": flags["valid"] and not needs_judge,
        "harmful": (label in HARMFUL if flags["valid"] and not needs_judge else None),
        "sdg": (label == "SAY_DO_GAP" if flags["valid"] and not needs_judge else None),
        "finish_reason": record.get("finish_reason") if record else None})
    return row


def load_panel(mode="corrected"):
    frame = pd.DataFrame(compact(case) for case in iter_cases(mode))
    for name in ["valid", "l0", "l1", "l2", "l2_legacy_all_arguments", "harmful", "sdg", "semantic_valid"]:
        frame[name] = frame[name].astype("boolean")
    return frame


def audit(frame):
    return {"n_cells": len(frame), "n_models": int(frame.model.nunique()),
        "n_scenarios": int(frame.scenario_id.nunique()),
        "status_counts": dict(Counter(frame.status)),
        "valid_inferences": int(frame.valid.sum()),
        "semantic_outcomes_available": int(frame.semantic_valid.sum()),
        "missing_judgments": int(frame.needs_judgment.sum()),
        "counts": {name: int(frame[name].sum()) for name in ["l0", "l1", "l2", "harmful", "sdg"]},
        "counts_are_final": (not bool(frame.needs_judgment.any()) and bool(frame.valid.all())
                             and bool(frame.input_matches_current.all())),
        "input_mismatches": int((~frame.input_matches_current & (frame.status != "missing_corrected_run")).sum()),
        "missing_record_inputs": int((frame.status == "missing_corrected_run").sum()),
        "label_counts": frame.label.value_counts().to_dict()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=["recorded", "common", "corrected"], default="corrected")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    frame = load_panel(args.panel)
    report = audit(frame)
    out = ROOT / "analysis/revision"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / (args.panel + "_records.csv.gz"), index=False, compression={"method": "gzip", "mtime": 0})
    (out / (args.panel + "_audit.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["counts_are_final"] and not args.allow_incomplete:
        raise SystemExit("Incomplete panel: no final manuscript numbers may be generated from this run.")


if __name__ == "__main__":
    main()
