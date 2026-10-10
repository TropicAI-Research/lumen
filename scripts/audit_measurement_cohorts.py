#!/usr/bin/env python3
"""Audit the archived judge evidence and its language-repair overlap, offline.

Alternative judges apply to saved original responses, not to repaired reruns.
The common-scenario restriction excludes the six affected scenarios in every
language. It is a coverage sensitivity check, not a new judge experiment.
"""
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.canonical_panel import DEFECT_IDS
from scripts.reproduce_sensitivity import LABELS, HARMFUL, judge_uncertainty, kappa
from scripts.unpack_manuscript import PAPER, ensure_manuscript

JUDGES = ["o4_mini_label", "opus_label", "gemini_label"]


def summarize_triples(rows):
    if not rows:
        raise ValueError("An empty triple cohort has no agreement estimate.")
    values = [[r[j] for j in JUDGES] for r in rows]
    if any(label not in LABELS for row in values for label in row):
        raise ValueError("Invalid judge label; missing responses must not be silently classified.")
    votes = np.array([[row.count(label) for label in LABELS] for row in values])
    majority = [Counter(row).most_common(1)[0] for row in values]
    pairwise = []
    for a, b in itertools.combinations(JUDGES, 2):
        matrix = np.zeros((4, 4), dtype=int)
        for row in rows:
            matrix[LABELS.index(row[a]), LABELS.index(row[b])] += 1
        pairwise.append({"judges": [a, b],
            "four_class_agreement": sum(r[a] == r[b] for r in rows) / len(rows),
            "binary_agreement": sum((r[a] in HARMFUL) == (r[b] in HARMFUL) for r in rows) / len(rows),
            "comply_redirect_inversions": sum({r[a], r[b]} == {"COMPLY", "REDIRECT"} for r in rows),
            "cohen_kappa": kappa(matrix)})
    return {"n": len(rows), "unanimous": sum(len(set(row)) == 1 for row in values),
        "majority": sum(n >= 2 for _, n in majority),
        "majority_sdg": sum(label == "SAY_DO_GAP" and n >= 2 for label, n in majority),
        "sdg_counts": {j: sum(r[j] == "SAY_DO_GAP" for r in rows) for j in JUDGES},
        "fleiss_kappa": float(fleiss_kappa(votes)), "pairwise": pairwise}


def main():
    ensure_manuscript()
    paths = [ROOT / "data/calibration/crossjudge_triples.jsonl",
             ROOT / "data/calibration/human_calibration.json", ROOT / "web-data/judge_index.json"]
    rows = [json.loads(line) for line in paths[0].read_text().splitlines() if line]
    questions = {q["question_id"]: q for lang in ["pt", "en", "es", "zh", "hi"]
                 for q in json.loads((ROOT / f"questions_v3_{lang}.json").read_text())}
    historical = json.loads(paths[2].read_text())
    seen = {}
    common, invalid_input_keys, duplicate_keys = [], [], []
    for row in rows:
        key = "|".join([row["model"], row["lang"], row["prompt"], row["question_id"]])
        if historical[key]["j"] != row["o4_mini_label"]:
            raise ValueError("Mismatched original judge: " + key)
        if key in seen:
            if any(row[j] != seen[key][j] for j in JUDGES):
                raise ValueError("Conflicting duplicate cross-judge labels: " + key)
            duplicate_keys.append(key)
            continue
        seen[key] = row
        question = questions[row["question_id"]]
        if question["scenario_id"] not in DEFECT_IDS:
            common.append(row)
        elif row["lang"] in {"zh", "hi"}:
            invalid_input_keys.append(key)
    frame = pd.DataFrame({"label": [r["j"] for r in historical.values()], "judged_cohort": True})
    calibration = judge_uncertainty(frame)
    unique = list(seen.values())
    unique_summary = summarize_triples(unique)
    rng = np.random.default_rng(42)
    votes = np.array([[sum(row[j] == label for j in JUDGES) for label in LABELS] for row in unique])
    bootstrap = [float(fleiss_kappa(votes[rng.integers(len(votes), size=len(votes))])) for _ in range(1000)]
    unique_summary["fleiss_kappa_ci95"] = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    unique_summary["bootstrap_seed"] = 42
    unique_summary["bootstrap_resamples"] = 1000
    report = {"archived_triples": summarize_triples(rows), "unique_triples": unique_summary,
        "common_102_scenario_triples": summarize_triples(common),
        "duplicate_keys_identical_labels": duplicate_keys,
        "original_wrong_language_triples": len(invalid_input_keys),
        "original_wrong_language_keys": sorted(invalid_input_keys),
        "human_calibration_reweighting_archival_population": calibration,
        "scope": "Labels refer to original responses only. The 102-scenario restriction is not a rejudgment of corrected-input outputs or human multilingual validation.",
        "source_files": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
    out = ROOT / "analysis/verification"
    out.mkdir(parents=True, exist_ok=True)
    (out / "measurement_cohort_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (ROOT / "data/calibration/crossjudge_unique_triples.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in unique) + "\n")
    lines = [r"\begin{table}[htbp]", r"\centering\small",
        r"\caption{Say-do labels assigned by three judges to the same response sample. The second column restricts that sample to 102 scenarios whose inputs were unchanged across collection periods. Each label belongs to a specific stored response.}",
        r"\label{tab:crossjudge-cohorts}", r"\begin{tabular}{lrr}", r"\toprule",
        f"Judge & Unique sample ($n={len(unique)}$) & Common scenarios ($n={len(common)}$) " + r"\\", r"\midrule"]
    for name, key in [("o4-mini", "o4_mini_label"), ("Opus 4.7", "opus_label"),
                      ("Gemini 2.5 Pro", "gemini_label"), ("Strict majority", None)]:
        cells = []
        for cohort in [report["unique_triples"], report["common_102_scenario_triples"]]:
            count = cohort["sdg_counts"][key] if key else cohort["majority_sdg"]
            cells.append(f"{count}/{cohort['n']} ({100*count/cohort['n']:.2f}\\%)")
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    generated = PAPER / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "judge_cohorts.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ["source_files", "original_wrong_language_keys",
        "human_calibration_reweighting_archival_population"]}, indent=2))
    print(json.dumps({k:v for k,v in calibration.items() if k.startswith("human_")}, indent=2))


if __name__ == "__main__":
    main()
