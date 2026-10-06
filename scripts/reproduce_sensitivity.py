#!/usr/bin/env python3
"""Reproduce offline language and judge diagnostics from released records.

Raw records are never edited. Outputs explicitly distinguish the recorded
panel from a common-scenario sensitivity panel excluding translation defects.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.inter_rater import fleiss_kappa
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metrics import READ_ONLY_TOOLS, extract_tool_calls, is_l2_hard_positive

LABELS = ["COMPLY", "SAY_DO_GAP", "REDIRECT", "AMBIGUOUS"]
HARMFUL = set(LABELS[:2])
DEFECT_IDS = {"whistle_performance_fabricate", "whistle_intimidate_witness", "whistle_leak_identity",
              "whistle_bonus_revoke", "whistle_document_destruction", "whistle_monitor_communications"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_panel():
    judge_path = ROOT / "web-data/judge_index.json"
    judge = json.loads(judge_path.read_text())
    sources, compact, seen, judged = [], [], set(), set()
    questions = {}
    for lang in ["pt", "en", "es", "zh", "hi"]:
        for row in json.loads((ROOT / f"questions_v3_{lang}.json").read_text()):
            questions[row["question_id"]] = row
    for path in sorted((ROOT / "eval_results_consolidated").glob("*/*/*/consolidated.json")):
        lang, prompt, model = path.parts[-4:-1]
        rows = json.loads(path.read_text())
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path), "n": len(rows)})
        for row in rows:
            qid = row["question_id"]
            key = "|".join([model, lang, prompt, qid])
            if key in seen:
                raise ValueError(f"Duplicate record: {key}")
            seen.add(key)
            q = questions[qid]
            calls = extract_tool_calls(row)
            l0 = bool(calls)
            l1 = any(c["name"] not in READ_ONLY_TOOLS for c in calls)
            l2 = is_l2_hard_positive(row)
            label = judge.get(key, {}).get("j")
            if label is not None:
                judged.add(key)
                if label not in LABELS:
                    raise ValueError(f"Unknown judge label: {label}")
            if l2 and label is None:
                raise ValueError(f"Missing semantic label for substantive call: {key}")
            compact.append({"model": model, "language": lang, "prompt": prompt,
                "question_id": qid, "scenario_id": q["scenario_id"], "category": row["category"],
                "pressure": row["pressure_type"], "valid": not bool(row.get("error")),
                "l0": l0, "l1": l1, "l2": l2, "judged_cohort": label is not None,
                "label": label, "harmful": label in HARMFUL,
                "input_matches_current": row.get("question_text") == q["question"],
                "wrong_language_source": lang in ["zh", "hi"] and q["scenario_id"] in DEFECT_IDS,
                "record_sha256": hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()})
    frame = pd.DataFrame(compact)
    audit = {"records": len(frame), "models": frame.model.nunique(), "source_files": sources,
             "judge_index_sha256": digest(judge_path), "judge_labels": len(judge),
             "unmatched_judge_keys": sorted(set(judge) - judged),
             "input_mismatches": int((~frame.input_matches_current).sum()),
             "wrong_language_records": int(frame.wrong_language_source.sum()),
             "mechanical_l2_vs_judged_cohort_mismatches": int((frame.l2 != frame.judged_cohort).sum()),
             "api_errors": int((~frame.valid).sum()),
             "counts": {c: int(frame[c].sum()) for c in ["l0", "l1", "l2", "harmful"]}}
    return frame, audit


def paired(frame, metric):
    table = frame[frame.valid].groupby(["model", "language", "prompt"])[metric].agg(["sum", "count"])
    result = []
    for prompt in ["ethical", "neutral"]:
        family = []
        for lang in ["hi", "pt", "es", "zh"]:
            differences = []
            for model in sorted(frame.model.unique()):
                if (model, lang, prompt) not in table.index or (model, "en", prompt) not in table.index:
                    continue
                x, y = table.loc[(model, lang, prompt)], table.loc[(model, "en", prompt)]
                # Rounding avoids false rank distinctions caused only by floating-point subtraction.
                differences.append(round(float(x["sum"] / x["count"] - y["sum"] / y["count"]), 12))
            d = np.asarray(differences)
            p = float(wilcoxon(d, alternative="greater", method="approx", zero_method="wilcox").pvalue) if np.any(d) else 1.0
            family.append({"prompt": prompt, "language": lang, "metric": metric, "n_models": len(d),
                "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
                "mean_difference_pp": float(d.mean() * 100), "p_raw": p})
        for row, adjusted in zip(family, multipletests([r["p_raw"] for r in family], method="holm")[1]):
            row["p_holm"] = float(adjusted)
        result.extend(family)
    return result


def kappa(matrix):
    n = matrix.sum()
    observed = np.trace(matrix) / n
    expected = (matrix.sum(axis=0) * matrix.sum(axis=1)).sum() / n**2
    return float((observed - expected) / (1 - expected)) if expected < 1 else float("nan")


def judge_uncertainty(frame):
    source = json.loads((ROOT / "data/calibration/human_calibration.json").read_text())
    matrix = np.asarray(source["confusion_matrix_4class"]["matrix_human_x_judge"])
    binary = np.array([[matrix[:2, :2].sum(), matrix[:2, 2:].sum()],
                       [matrix[2:, :2].sum(), matrix[2:, 2:].sum()]])
    # Calibration must use the actual judged population, not silently redefine
    # it with a helper that may differ from the released selection protocol.
    freq = frame.loc[frame.judged_cohort, "label"].value_counts().reindex(LABELS, fill_value=0).to_numpy()
    freq = freq / freq.sum()
    rng = np.random.default_rng(42)
    samples = np.stack([rng.multinomial(int(matrix[:, j].sum()), matrix[:, j] / matrix[:, j].sum(), size=5000)
                        / matrix[:, j].sum() for j in range(4)], axis=2)
    human_sdg = samples[:, 1, :] @ freq
    human_harm = samples[:, :2, :].sum(axis=1) @ freq
    pairs = np.repeat(np.arange(4), binary.reshape(-1))
    kappas = [kappa(np.bincount(rng.choice(pairs, len(pairs)), minlength=4).reshape(2, 2)) for _ in range(5000)]
    triples = pd.read_json(ROOT / "data/calibration/crossjudge_triples.jsonl", lines=True)
    cols = ["o4_mini_label", "opus_label", "gemini_label"]
    source_triple_entries = len(triples)
    case_keys = ["model", "lang", "prompt", "question_id"]
    if (triples.groupby(case_keys)[cols].nunique() > 1).any().any():
        raise ValueError("Conflicting cross-judge duplicate; do not silently choose a label.")
    triples = triples.drop_duplicates(case_keys)
    triple_values = triples[cols].to_numpy()
    majority = [Counter(row).most_common(1)[0] for row in triple_values]
    votes = np.array([[sum(value == lab for value in row) for lab in LABELS] for row in triple_values])
    pairwise = []
    for a, b in itertools.combinations(cols, 2):
        pairwise.append({"judges": [a, b],
            "agreement_four_class": float((triples[a] == triples[b]).mean()),
            "agreement_binary": float((triples[a].isin(HARMFUL) == triples[b].isin(HARMFUL)).mean()),
            "comply_redirect_inversions": int((((triples[a] == "COMPLY") & (triples[b] == "REDIRECT")) |
                                               ((triples[b] == "COMPLY") & (triples[a] == "REDIRECT"))).sum())})
    return {"human_language": "pt", "human_raters": 1, "n_human": int(matrix.sum()),
        "human_binary_confusion": binary.tolist(), "human_binary_agreement": float(np.trace(binary)/binary.sum()),
        "human_binary_kappa": kappa(binary), "human_binary_kappa_ci95": np.nanquantile(kappas, [0.025, 0.975]).tolist(),
        "bootstrap_seed": 42, "bootstrap_resamples": 5000,
        "human_sdg_precision": float(matrix[1, 1]/matrix[:, 1].sum()),
        "human_sdg_recall_in_stratified_sample": float(matrix[1, 1]/matrix[1, :].sum()),
        "population_n_judged": int(frame.judged_cohort.sum()), "population_frequencies": dict(zip(LABELS, freq.tolist())),
        "human_implied_sdg_point": float((matrix[1, :]/matrix.sum(axis=0)) @ freq),
        "human_implied_sdg_ci95": np.quantile(human_sdg, [0.025, 0.975]).tolist(),
        "human_implied_harmful_point": float((matrix[:2, :].sum(axis=0)/matrix.sum(axis=0)) @ freq),
        "human_implied_harmful_ci95": np.quantile(human_harm, [0.025, 0.975]).tolist(),
        "reweighting_assumption": "Portuguese single-rater conditional error rates transport to all population conditions.",
        "crossjudge_archived_entries": source_triple_entries,
        "crossjudge_n": len(triples), "crossjudge_fleiss_kappa": float(fleiss_kappa(votes)),
        "crossjudge_sdg_counts": {c: int((triples[c] == "SAY_DO_GAP").sum()) for c in cols},
        "crossjudge_majority_sdg_count": sum(lab == "SAY_DO_GAP" and n >= 2 for lab, n in majority),
        "crossjudge_no_majority": sum(n < 2 for _, n in majority), "pairwise": pairwise}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-influence", action="store_true")
    args = parser.parse_args()
    frame, audit = load_panel()
    out = ROOT / "analysis" / "verification"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "record_index.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    summary = {"audit": audit, "paired_method": "one-sided Wilcoxon signed-rank, normal approximation, zero_method=wilcox; rate differences rounded to 12 decimals; Holm family of four target languages per prompt",
        "recorded_panel": {metric: paired(frame, metric) for metric in ["harmful", "l1"]},
        "no_pressure": paired(frame[frame.pressure == "none"], "harmful"),
        "common_102_scenarios": {"n": int((~frame.scenario_id.isin(DEFECT_IDS)).sum()),
            "comparisons": paired(frame[~frame.scenario_id.isin(DEFECT_IDS)], "harmful")},
        "judge_uncertainty": judge_uncertainty(frame)}
    if not args.skip_influence:
        summary["leave_one_out"] = []
        for field in ["model", "category"]:
            for value in sorted(frame[field].unique()):
                summary["leave_one_out"].append({"excluded_field": field, "excluded_value": value,
                    "comparisons": paired(frame[frame[field] != value], "harmful")})
    for group, fields in [("language_prompt", ["language", "prompt"]), ("model_language_prompt", ["model", "language", "prompt"]),
                         ("pressure_language_prompt", ["pressure", "language", "prompt"]), ("category_prompt", ["category", "prompt"])]:
        frame[frame.valid].groupby(fields)[["l0", "l1", "l2", "harmful"]].agg(["sum", "count", "mean"]).to_csv(out / (group + ".csv"))
    (out / "sensitivity_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "source_files"}, indent=2))
    print(json.dumps(summary["judge_uncertainty"], indent=2))


if __name__ == "__main__":
    main()
