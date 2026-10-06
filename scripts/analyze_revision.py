#!/usr/bin/env python3
"""Reproduce comparisons on explicit, versioned panels; never impute safe cells.

Outputs are diagnostics, not final manuscript tables, while any eligible
semantic label is missing. Comparison pairs always share a base scenario and
pressure condition; a missing translated response cannot change only one
side's denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.canonical_panel import load_panel, audit
from scripts.reproduce_sensitivity import judge_uncertainty

TARGETS = ["hi", "pt", "es", "zh"]
PAIR_KEYS = ["model", "prompt", "scenario_id", "pressure"]


def compare_differences(values, alternative="greater"):
    values = np.round(np.asarray(values, dtype=float), 12)
    if not len(values):
        return {"n_models": 0, "mean_difference_pp": None, "p_raw": None}
    p = float(wilcoxon(values, alternative=alternative, method="approx", zero_method="wilcox").pvalue) if np.any(values) else 1.0
    return {"n_models": len(values), "mean_difference_pp": float(values.mean()*100), "p_raw": p,
            "wins": int((values > 0).sum()), "losses": int((values < 0).sum()), "ties": int((values == 0).sum())}


def adjust_family(rows):
    eligible = [r for r in rows if r["p_raw"] is not None]
    for row, p in zip(eligible, multipletests([r["p_raw"] for r in eligible], method="holm")[1] if eligible else []):
        row["p_holm"] = float(p)
    return rows


def require_final_panel(panel_audit, allow_incomplete=False):
    if not panel_audit["counts_are_final"] and not allow_incomplete:
        raise ValueError("Incomplete or mismatched panel: missing responses, API errors, or pending labels. "
                         "No final manuscript statistics may be generated.")


def exclude_recorded_api_errors(frame):
    """Explicit available-case analysis; do not hide absent runs or labels.

    All removed IDs remain in the design audit. Pair construction below
    also removes each unavailable case's counterpart in the other language.
    """
    return frame.loc[frame.status != "api_error"].copy()


def select_available_cases(frame):
    """Define an explicit available-response cohort, never skip pending labels."""
    if frame.needs_judgment.any():
        raise ValueError("Available-case analysis requires every eligible semantic judgment.")
    omitted = frame.status.isin(["api_error", "missing_corrected_run"])
    selected = frame.loc[~omitted].copy()
    if not selected.semantic_valid.all() or not selected.input_matches_current.all():
        raise ValueError("Unexpected invalid observation in available-case cohort.")
    return selected


def matched_rates(frame, language, prompt, metric):
    valid = frame[frame.valid & frame.input_matches_current & frame[metric].notna() & (frame.prompt == prompt)]
    target = valid.loc[valid.language == language, PAIR_KEYS + [metric]].rename(columns={metric: "target"})
    english = valid.loc[valid.language == "en", PAIR_KEYS + [metric]].rename(columns={metric: "english"})
    merged = target.merge(english, on=PAIR_KEYS, validate="one_to_one")
    if merged.empty:
        return pd.DataFrame(columns=["model", "target", "english", "n_paired_trials"])
    aggregate = merged.groupby("model").agg(target=("target", "mean"), english=("english", "mean"),
                                             n_paired_trials=("target", "size")).reset_index()
    return aggregate


def paired(frame, metric):
    results = []
    for prompt in ["ethical", "neutral"]:
        family = []
        for language in TARGETS:
            rates = matched_rates(frame, language, prompt, metric)
            d = (rates.target-rates.english).astype(float).to_numpy()
            summary = compare_differences(d)
            family.append({"prompt": prompt, "language": language, "metric": metric, **summary,
                "n_paired_trials": int(rates.n_paired_trials.sum()),
                "trials_per_model_min": int(rates.n_paired_trials.min()) if len(rates) else None,
                "trials_per_model_max": int(rates.n_paired_trials.max()) if len(rates) else None,
                "model_rates": rates.to_dict("records")})
        results.extend(adjust_family(family))
    return results


def pressure_diagnostics(frame):
    """Exploratory interaction contrasts, not inferred psychological causes."""
    valid = frame[frame.semantic_valid & frame.input_matches_current]
    keys = ["model", "language", "prompt", "scenario_id"]
    baseline = valid.loc[valid.pressure == "none", keys + ["harmful"]].rename(columns={"harmful": "baseline"})
    effects, interactions = [], []
    for prompt in ["ethical", "neutral"]:
        interaction_family = []
        for pressure in sorted(set(valid.pressure) - {"none"}):
            subset = valid[(valid.pressure == pressure) & (valid.prompt == prompt)]
            merged = subset[keys + ["harmful"]].merge(baseline, on=keys, validate="one_to_one")
            merged["difference"] = merged.harmful.astype(float) - merged.baseline.astype(float)
            rates = merged.groupby(["model", "language"])["difference"].mean().unstack("language")
            for language in sorted(rates):
                values = rates[language].dropna().astype(float).to_numpy()
                effects.append({"prompt": prompt, "pressure": pressure, "language": language,
                                **compare_differences(values, alternative="two-sided")})
            # Interaction differences require all four observations on the same
            # scenario, not merely a shared model with different case coverage.
            english = merged[merged.language == "en"][["model", "scenario_id", "difference"]].rename(columns={"difference": "english_difference"})
            for language in TARGETS:
                target = merged[merged.language == language][["model", "scenario_id", "difference"]]
                common = target.merge(english, on=["model", "scenario_id"], validate="one_to_one")
                common["interaction"] = common.difference - common.english_difference
                by_model = common.groupby("model").interaction.mean()
                interaction_family.append({"prompt": prompt, "pressure": pressure, "language": language,
                    "n_four_cell_scenarios": len(common), **compare_differences(by_model.to_numpy(), alternative="two-sided")})
        interactions.extend(adjust_family(interaction_family))
    # Five pressure-vs-baseline contrasts form a family within each language
    # and prompt; twenty language-interaction contrasts form a prompt family.
    for prompt, language in itertools.product(["ethical", "neutral"], ["en", *TARGETS]):
        adjust_family([r for r in effects if r["prompt"] == prompt and r["language"] == language])
    return {"pressure_vs_none": effects, "language_interactions": interactions,
            "interpretation": "Exploratory behavioral contrasts, not causal identification of psychological mechanisms or model training differences."}


def logistic_regression(frame):
    """Original unregularized fixed-effects specification, plus cluster SEs."""
    import statsmodels.api as sm
    valid = frame[frame.semantic_valid & frame.input_matches_current].copy()
    counts = valid.groupby("model").harmful.agg(["sum", "count"])
    separated = list(counts.index[(counts["sum"] == 0) | (counts["sum"] == counts["count"])])
    valid = valid[~valid.model.isin(separated)]
    # Binomial sufficient statistics preserve the unregularized point fit.
    fields = ["model", "language", "prompt", "category", "pressure"]
    grouped = valid.groupby(fields).harmful.agg(["sum", "count"]).reset_index()
    predictors = {}
    references = {"language": "en", "prompt": "ethical", "category": "autonomy", "pressure": "authority",
                  "model": sorted(grouped.model.unique())[0]}
    for field, reference in references.items():
        for value in sorted(grouped[field].unique()):
            if value != reference:
                predictors[field + "_" + value] = (grouped[field] == value).astype(float)
    x = sm.add_constant(pd.DataFrame(predictors), prepend=True)
    y = np.column_stack([grouped["sum"].astype(float), (grouped["count"]-grouped["sum"]).astype(float)])
    glm = sm.GLM(y, x, family=sm.families.Binomial())
    conventional = glm.fit(maxiter=100)
    clustered = glm.fit(maxiter=100, cov_type="cluster", cov_kwds={"groups": grouped.model, "use_correction": True})
    if not conventional.converged or not clustered.converged:
        raise ValueError("Regression did not converge")
    coefficients = {}
    for term in x.columns:
        if term == "const" or term.startswith("model_"):
            continue
        beta = float(conventional.params[term])
        coefficients[term] = {"beta": beta, "odds_ratio": float(np.exp(beta)),
            "ordinary_se": float(conventional.bse[term]), "ordinary_p": float(conventional.pvalues[term]),
            "cluster_se": float(clustered.bse[term]), "cluster_p": float(clustered.pvalues[term]),
            "cluster_ci95": np.exp(clustered.conf_int().loc[term]).tolist()}
    return {"n_trials": len(valid), "n_binomial_cells": len(grouped), "n_models": int(valid.model.nunique()),
            "excluded_separated_models": separated, "reference_levels": references,
            "clustering": "model; finite-cluster approximation; point estimates equal the original unregularized specification",
            "coefficients": coefficients}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=["common", "corrected"], default="common")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--exclude-api-errors", action="store_true",
                        help="Explicitly exclude terminal API failures, preserving their IDs and source-panel audit. Does not permit pending runs or judgments.")
    parser.add_argument("--available-case", action="store_true",
                        help="Explicitly defined available-response cohort; lists unavailable/API-failed cells, refuses pending judgments.")
    parser.add_argument("--skip-influence", action="store_true")
    parser.add_argument("--regression", action="store_true")
    args = parser.parse_args()
    source_frame = load_panel(args.panel)
    source_audit = audit(source_frame)
    frame = (select_available_cases(source_frame) if args.available_case else
             exclude_recorded_api_errors(source_frame) if args.exclude_api_errors else source_frame)
    panel_audit = audit(frame)
    try:
        require_final_panel(panel_audit, args.allow_incomplete)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    out = ROOT / "analysis/revision"
    out.mkdir(parents=True, exist_ok=True)
    report = {"panel": args.panel, "audit": panel_audit, "source_panel_audit": source_audit,
        "exclude_api_errors": args.exclude_api_errors or args.available_case,
        "available_case_analysis": args.available_case,
        "excluded_api_error_keys": source_frame.loc[source_frame.status == "api_error", "key"].tolist() if args.exclude_api_errors or args.available_case else [],
        "excluded_unavailable_keys": source_frame.loc[source_frame.status == "missing_corrected_run", "key"].tolist() if args.available_case else [],
        "paired_method": "Model-paired, scenario/pressure-matched, one-sided Wilcoxon; zero_method=wilcox; normal approximation; differences rounded to 12 decimals; four-language Holm family per prompt.",
        "comparisons": {m: paired(frame, m) for m in ["harmful", "l1"]},
        "no_pressure": paired(frame[frame.pressure == "none"], "harmful"),
        "judge_uncertainty": judge_uncertainty(frame),
        "pressure_diagnostics": pressure_diagnostics(frame)}
    if args.exclude_api_errors or args.available_case:
        complete_models = source_frame.groupby("model").valid.all()
        complete_models = list(complete_models.index[complete_models])
        complete_frame = frame[frame.model.isin(complete_models)]
        report["complete_models_sensitivity"] = {"models": complete_models,
            "n_models": len(complete_models), "n_observations": len(complete_frame),
            "comparisons": paired(complete_frame, "harmful")}
    if not args.skip_influence:
        report["leave_one_out"] = []
        for field in ["model", "category"]:
            for value in sorted(frame[field].unique()):
                comparisons = paired(frame[frame[field] != value], "harmful")
                for row in comparisons:
                    row.pop("model_rates", None)
                report["leave_one_out"].append({"excluded_field": field, "excluded_value": value, "comparisons": comparisons})
    if args.regression and panel_audit["counts_are_final"]:
        report["regression"] = logistic_regression(frame)
    for name, groups in [("models", ["model"]), ("language_prompt", ["language", "prompt"]),
                         ("model_language_prompt", ["model", "language", "prompt"]),
                         ("category_prompt", ["category", "prompt"]),
                         ("pressure_language_prompt", ["pressure", "language", "prompt"])]:
        eligible = frame[frame.semantic_valid & frame.input_matches_current]
        eligible.groupby(groups)[["l0", "l1", "l2", "harmful", "sdg"]].agg(["sum", "count", "mean"]).to_csv(out / f"{args.panel}_{name}.csv")
    path = out / (args.panel + "_statistics.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(path.relative_to(ROOT)), "audit": panel_audit}, indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k != "model_rates"} for r in report["comparisons"]["harmful"]], indent=2))


if __name__ == "__main__":
    main()
