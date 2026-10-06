#!/usr/bin/env python3
"""Recover original evidence without modifying or executing the source project.

Only explicitly listed, secret-screened source files are archived. Human notes
and timestamps are not exported: the public calibration needs case content,
labels, and stable IDs, not annotation-interface metadata. No API calls.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metrics import extract_tool_calls, is_l2_hard_positive

LABELS = ["COMPLY", "SAY_DO_GAP", "REDIRECT", "AMBIGUOUS"]
SCRIPT_NAMES = ["metrics.py", "recompute_violations.py", "analysis_deep.py",
                "judge_new_l2.py", "eval_v3.py", "run_crossjudge_656.py",
                "annotation/compute_kappas.py", "annotation/build_cases.py",
                "annotation/annotator.html", "figures.py", "figures_extra.py",
                "statistical_analysis.py", "rerun_logistic_fe.py"]
SECRET = re.compile(rb"(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|"
                    rb"gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def export(path, data):
    if SECRET.search(data):
        raise ValueError(f"Possible credential in {path.name}; export blocked (value redacted).")
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"Existing evidence differs: {path}; not overwritten.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def recover(source):
    destination = ROOT / "data/provenance/original_source"
    sources = []
    for name in SCRIPT_NAMES:
        data = (source / name).read_bytes()
        export(destination / name, data)
        sources.append({"path": name, "sha256": sha(data), "bytes": len(data)})

    input_names = ["judge_combined_parsed.json", "annotation/cases_to_label.json",
                   "annotation/cases_meta.json", "annotation/annotations_2026-04-26T20-35-08.json"]
    content = {}
    for name in input_names:
        data = (source / name).read_bytes()
        sources.append({"path": name, "sha256": sha(data), "bytes": len(data)})
        content[name] = json.loads(data)
    original_judge = content[input_names[0]]
    grouped = defaultdict(list)
    for entry in original_judge.values():
        key = "|".join(entry[k] for k in ["model", "lang", "prompt", "question_id"])
        grouped[key].append(entry)
    for key, entries in grouped.items():
        if len({e["label"] for e in entries}) != 1:
            raise ValueError(f"Conflicting historical labels for {key}")
    released = json.loads((ROOT / "web-data/judge_index.json").read_text())
    assert set(released) == set(grouped), "Released and original judge cohorts differ"
    assert all(grouped[k][-1]["label"] == v["j"] for k, v in released.items())

    cases = {r["case_id"]: r for r in content[input_names[1]]}
    meta = {r["case_id"]: r for r in content[input_names[2]]}
    annotations = content[input_names[3]]["annotations"]
    assert len(cases) == len(meta) == len(annotations) == 100
    assert set(meta) == {r["case_id"] for r in annotations}
    matrix = [[0] * 4 for _ in range(4)]
    human_rows = []
    raw_cache = {}
    matched_interface_cases = 0
    for annotation in annotations:
        cid = annotation["case_id"]
        j = original_judge[cid]
        assert j["lang"] == "pt"
        human = annotation["label"]
        judge = meta[cid]["judge_label"]
        matrix[LABELS.index(human)][LABELS.index(judge)] += 1
        raw_path = ROOT / "eval_results_consolidated" / j["lang"] / j["prompt"] / j["model"] / "consolidated.json"
        if raw_path not in raw_cache:
            raw_cache[raw_path] = {r["question_id"]: r for r in json.loads(raw_path.read_text())}
        raw = raw_cache[raw_path][j["question_id"]]
        case = {"case_id": cid, "category": j["category"], "pressure_type": j["pressure_type"],
            "prompt_type": j["prompt"], "question_text": raw["question_text"],
            "assistant_response": raw.get("assistant_response") or "", "tool_calls": extract_tool_calls(raw)}
        if cid in cases:
            assert all(case[k] == cases[cid][k] for k in case), f"Saved annotation content differs for {cid}"
            matched_interface_cases += 1
        human_rows.append({**case, "language": "pt", "human_label": human,
            "judge_label": judge, "model": j["model"], "question_id": j["question_id"],
            "judge_reason": meta[cid]["judge_reason"],
            "content_provenance": "Raw evaluation joined through the annotation case ID and original judge metadata.",
            "also_matches_saved_annotation_interface_case": cid in cases,
            "protocol": "Single author; Portuguese only; model identity and judge label hidden in the annotation interface."})
    expected = json.loads((ROOT / "data/calibration/human_calibration.json").read_text())
    assert matrix == expected["confusion_matrix_4class"]["matrix_human_x_judge"]
    human_path = ROOT / "data/calibration/human_cases.jsonl"
    export(human_path, ("\n".join(json.dumps(r, ensure_ascii=False) for r in human_rows) + "\n").encode())

    # Extract only the original pure functions, never execute the project or
    # load its credentials. Both mechanical rules are audited, not conflated.
    tree = ast.parse((source / "recompute_violations.py").read_text())
    selected = [node for node in tree.body if
        isinstance(node, ast.FunctionDef) and node.name in {"classify_result", "has_nontrivial_params"}
        or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
            t.id in {"READ_ONLY_TOOLS", "TRIVIAL_VALUES"} for t in node.targets)]
    namespace = {"json": json}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "original_mechanical_rules", "exec"), namespace)
    classify = namespace["classify_result"]
    counts = Counter()
    unjudged = []
    identical = 0
    for path in sorted((ROOT / "eval_results_consolidated").glob("*/*/*/consolidated.json")):
        relative = path.relative_to(ROOT)
        original_path = source / relative
        data = path.read_bytes()
        assert original_path.read_bytes() == data, f"Original results differ: {relative}"
        identical += 1
        lang, prompt, model = path.parts[-4:-1]
        for record in json.loads(data):
            key = "|".join([model, lang, prompt, record["question_id"]])
            broad = classify(record)["L2"]
            strict = is_l2_hard_positive(record)
            judged = key in released
            counts["records"] += 1
            counts["original_any_argument_l2"] += bool(broad)
            counts["all_arguments_helper_l2"] += strict
            counts["historically_judged"] += judged
            counts["judged_outside_strict_helper"] += judged and not strict
            counts["strict_helper_without_judge"] += strict and not judged
            counts["judged_outside_broad_rule"] += judged and not broad
            if broad and not judged:
                unjudged.append({"key": key, "source_path": str(relative),
                                 "reason": "Original any-argument L2 rule passes; no historical semantic label."})
    report = {"source_folder_label": source.name, "sources": sources,
        "original_files_modified": False, "identical_result_files": identical,
        "human_calibration": {"n": 100, "language": "pt", "rater_count": 1,
            "confusion_matrix_human_x_judge": matrix, "matches_released_summary": True,
            "annotation_and_hidden_label_ids_match": True,
            "saved_interface_file_overlaps_annotation_export_in": matched_interface_cases,
            "all_overlapping_interface_contents_match_raw_evaluations": True,
            "case_content_recovered_from_raw_evaluations": len(human_rows),
            "export": str(human_path.relative_to(ROOT)), "export_sha256": sha(human_path.read_bytes())},
        "judge_recovery": {"original_entries": len(original_judge), "unique_cases": len(grouped),
            "redundant_entries": len(original_judge) - len(grouped), "conflicting_duplicates": 0,
            "matches_all_released_labels": True},
        "mechanical_selection_audit": dict(counts),
        "historical_analysis": "analysis_deep.py assigned L2 by judge-cohort membership, not by recomputing either mechanical rule.",
        "resolution_status": "Evidence recovered. A single explicit mechanical definition and complete judging coverage are still required for the corrected release.",
        "unjudged_under_original_broad_rule": unjudged}
    report_path = ROOT / "analysis/verification/original_evidence_audit.json"
    export(report_path, (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode())
    print(json.dumps({"human_cases_recovered": len(human_rows), "identical_result_files": identical,
        "redundant_judge_entries": len(original_judge) - len(grouped), **dict(counts),
        "unjudged_under_original_broad_rule": len(unjudged)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    recover(parser.parse_args().source.resolve())
