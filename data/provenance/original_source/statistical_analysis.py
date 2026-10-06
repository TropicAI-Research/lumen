#!/usr/bin/env python3
"""
Statistical analysis for Lumen-bench evaluation results.

Reads all incremental JSONL files from eval_results/ and computes:
1. Overall and per-prompt-type violation rates with 95% bootstrap CIs
2. McNemar's test for ethical vs neutral per model
3. Permutation test for language effects (EN vs HI gap)
4. Chi-squared test for pressure type effects
5. Cohen's h effect sizes for key comparisons
"""

import json
import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# ============================================================================
# Configuration
# ============================================================================
EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results")
N_BOOTSTRAP = 10_000
SEED = 42
ALPHA = 0.05

np.random.seed(SEED)


# ============================================================================
# Data loading
# ============================================================================
def load_all_data():
    """Load all incremental JSONL files, extracting model name from path."""
    records = []
    pattern = os.path.join(EVAL_DIR, "**", "incremental_*.jsonl")
    files = glob.glob(pattern, recursive=True)
    print(f"Found {len(files)} incremental JSONL files")

    for fpath in sorted(files):
        # Extract model name from path: eval_results/<lang>/<prompt_type>/<model>/file.jsonl
        rel = os.path.relpath(fpath, EVAL_DIR)
        parts = Path(rel).parts
        # Normal case: lang/prompt_type/model/file.jsonl (4 parts)
        # Anomalous case: lang/model/file.jsonl (3 parts) -- use prompt_type from data
        if len(parts) >= 4:
            path_model = parts[2]
        elif len(parts) == 3:
            path_model = parts[1]
        else:
            print(f"  WARNING: skipping file with unexpected path structure: {fpath}")
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  WARNING: invalid JSON at {fpath}:{line_num}")
                    continue

                # Skip records with errors
                if rec.get("error"):
                    continue

                # Extract base scenario from question_id
                qid = rec["question_id"]
                lang = rec["language"]
                pressure = rec["pressure_type"]
                suffix = f"_{lang}_{pressure}_"
                idx = qid.find(suffix)
                base_scenario = qid[:idx] if idx > 0 else qid

                rec["model"] = path_model
                rec["base_scenario"] = base_scenario

                # Ensure prompt_type exists (should be in data)
                if "prompt_type" not in rec:
                    if len(parts) >= 4:
                        rec["prompt_type"] = parts[1]
                    else:
                        continue

                records.append(rec)

    print(f"Loaded {len(records)} valid records")
    return records


def deduplicate_records(records):
    """If a model+language+prompt_type has duplicate question_ids, keep the latest."""
    # Group by (model, language, prompt_type, question_id)
    groups = defaultdict(list)
    for rec in records:
        key = (rec["model"], rec["language"], rec["prompt_type"], rec["question_id"])
        groups[key].append(rec)

    deduped = []
    duplicates = 0
    for key, recs in groups.items():
        if len(recs) > 1:
            duplicates += len(recs) - 1
        # Keep the last one (most recent file)
        deduped.append(recs[-1])

    if duplicates > 0:
        print(f"Removed {duplicates} duplicate records (kept latest)")
    print(f"After deduplication: {len(deduped)} records")
    return deduped


# ============================================================================
# Statistical functions
# ============================================================================
def cohen_h(p1, p2):
    """Compute Cohen's h effect size for two proportions."""
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def interpret_cohen_h(h):
    """Interpret Cohen's h magnitude."""
    ah = abs(h)
    if ah < 0.2:
        return "negligible"
    elif ah < 0.5:
        return "small"
    elif ah < 0.8:
        return "medium"
    else:
        return "large"


def bootstrap_violation_rate(records, n_iter=N_BOOTSTRAP):
    """
    Bootstrap CI for violation rate by resampling base scenarios.
    Each scenario maps to multiple questions (across pressures).
    Returns (point_estimate, ci_lower, ci_upper).
    """
    # Group by base_scenario
    scenario_groups = defaultdict(list)
    for rec in records:
        scenario_groups[rec["base_scenario"]].append(rec)

    scenarios = list(scenario_groups.keys())
    n_scenarios = len(scenarios)

    if n_scenarios == 0:
        return 0.0, 0.0, 0.0

    # Point estimate
    total = len(records)
    violations = sum(1 for r in records if r["tool_called"])
    point_est = violations / total if total > 0 else 0.0

    # Bootstrap
    boot_rates = np.empty(n_iter)
    for i in range(n_iter):
        # Resample scenarios with replacement
        sampled_indices = np.random.randint(0, n_scenarios, size=n_scenarios)
        num = 0
        den = 0
        for idx in sampled_indices:
            sc = scenarios[idx]
            group = scenario_groups[sc]
            den += len(group)
            num += sum(1 for r in group if r["tool_called"])
        boot_rates[i] = num / den if den > 0 else 0.0

    ci_lower = np.percentile(boot_rates, 2.5)
    ci_upper = np.percentile(boot_rates, 97.5)

    return point_est, ci_lower, ci_upper


def mcnemar_test(records_a, records_b, key_field="question_id"):
    """
    McNemar's test comparing two conditions paired by key_field.
    Compares tool_called outcomes.
    Returns (chi2, p_value, n_discordant_ab, n_discordant_ba).
    """
    # Build lookup for condition B
    b_lookup = {}
    for rec in records_b:
        b_lookup[rec[key_field]] = rec["tool_called"]

    # For paired scenarios, we need to match by base_scenario + pressure
    # Use a composite key: base_scenario + pressure_type
    a_by_key = {}
    for rec in records_a:
        k = (rec["base_scenario"], rec["pressure_type"], rec["language"])
        a_by_key[k] = rec["tool_called"]

    b_by_key = {}
    for rec in records_b:
        k = (rec["base_scenario"], rec["pressure_type"], rec["language"])
        b_by_key[k] = rec["tool_called"]

    # Find shared keys
    shared = set(a_by_key.keys()) & set(b_by_key.keys())

    if len(shared) == 0:
        return None, None, 0, 0

    # Count discordant pairs
    # b (cell where A=violation, B=no violation)
    # c (cell where A=no violation, B=violation)
    b_count = 0  # A violated, B did not
    c_count = 0  # A did not, B violated
    for k in shared:
        a_val = a_by_key[k]
        b_val = b_by_key[k]
        if a_val and not b_val:
            b_count += 1
        elif not a_val and b_val:
            c_count += 1

    n_disc = b_count + c_count
    if n_disc == 0:
        return 0.0, 1.0, b_count, c_count

    # McNemar's test with continuity correction
    chi2 = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return chi2, p_value, b_count, c_count


def permutation_test_languages(records, lang_a, lang_b, n_perm=10_000):
    """
    Permutation test for difference in violation rate between two languages.
    Returns (rate_a, rate_b, diff, p_value).
    """
    recs_a = [r for r in records if r["language"] == lang_a]
    recs_b = [r for r in records if r["language"] == lang_b]

    if not recs_a or not recs_b:
        return None, None, None, None

    rate_a = np.mean([r["tool_called"] for r in recs_a])
    rate_b = np.mean([r["tool_called"] for r in recs_b])
    observed_diff = rate_a - rate_b

    # Pool all records from both languages
    pooled = recs_a + recs_b
    n_a = len(recs_a)
    n_total = len(pooled)
    violations = np.array([r["tool_called"] for r in pooled], dtype=float)

    count_extreme = 0
    for _ in range(n_perm):
        perm = np.random.permutation(n_total)
        perm_rate_a = violations[perm[:n_a]].mean()
        perm_rate_b = violations[perm[n_a:]].mean()
        perm_diff = perm_rate_a - perm_rate_b
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_perm + 1)  # +1 for observed
    return rate_a, rate_b, observed_diff, p_value


def chi_squared_pressure(records):
    """
    Chi-squared test for association between pressure type and violation.
    Returns (chi2, p_value, dof, contingency_table_dict).
    """
    pressures = sorted(set(r["pressure_type"] for r in records))
    # Build contingency table: rows = pressure types, cols = [violated, not_violated]
    table = []
    table_dict = {}
    for p in pressures:
        p_recs = [r for r in records if r["pressure_type"] == p]
        violated = sum(1 for r in p_recs if r["tool_called"])
        not_violated = len(p_recs) - violated
        table.append([violated, not_violated])
        table_dict[p] = {"violated": violated, "total": len(p_recs),
                         "rate": violated / len(p_recs) if p_recs else 0}

    table = np.array(table)
    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    return chi2, p_value, dof, table_dict


# ============================================================================
# Main analysis
# ============================================================================
def main():
    print("=" * 80)
    print("LUMEN-BENCH -- STATISTICAL ANALYSIS")
    print("=" * 80)
    print()

    # Load data
    records = load_all_data()
    records = deduplicate_records(records)

    models = sorted(set(r["model"] for r in records))
    languages = sorted(set(r["language"] for r in records))
    prompt_types = sorted(set(r["prompt_type"] for r in records))
    pressures = sorted(set(r["pressure_type"] for r in records))

    print(f"\nModels ({len(models)}): {', '.join(models)}")
    print(f"Languages ({len(languages)}): {', '.join(languages)}")
    print(f"Prompt types ({len(prompt_types)}): {', '.join(prompt_types)}")
    print(f"Pressure types ({len(pressures)}): {', '.join(pressures)}")

    # Per-model record counts
    print(f"\nRecords per model:")
    for m in models:
        n = sum(1 for r in records if r["model"] == m)
        print(f"  {m}: {n}")

    # ==================================================================
    # 1. OVERALL VIOLATION RATE PER MODEL WITH BOOTSTRAP CIs
    # ==================================================================
    print("\n" + "=" * 80)
    print("1. OVERALL VIOLATION RATE PER MODEL (95% Bootstrap CI, 10k iterations)")
    print("   Resampling unit: base scenario (108 scenarios)")
    print("=" * 80)
    print(f"{'Model':<30} {'Violation Rate':>15} {'95% CI':>20} {'N':>6}")
    print("-" * 75)

    model_overall = {}
    for m in models:
        m_recs = [r for r in records if r["model"] == m]
        rate, ci_lo, ci_hi = bootstrap_violation_rate(m_recs)
        model_overall[m] = (rate, ci_lo, ci_hi, len(m_recs))
        print(f"{m:<30} {rate:>14.1%} [{ci_lo:>7.1%}, {ci_hi:>7.1%}] {len(m_recs):>6}")

    # ==================================================================
    # 2. VIOLATION RATE PER MODEL x PROMPT TYPE WITH BOOTSTRAP CIs
    # ==================================================================
    print("\n" + "=" * 80)
    print("2. VIOLATION RATE PER MODEL x PROMPT TYPE (95% Bootstrap CI)")
    print("=" * 80)

    model_prompt_rates = {}
    for pt in prompt_types:
        print(f"\n  --- Prompt type: {pt.upper()} ---")
        print(f"  {'Model':<30} {'Violation Rate':>15} {'95% CI':>20} {'N':>6}")
        print(f"  {'-' * 73}")
        for m in models:
            m_recs = [r for r in records if r["model"] == m and r["prompt_type"] == pt]
            if not m_recs:
                print(f"  {m:<30} {'N/A':>15} {'':>20} {0:>6}")
                continue
            rate, ci_lo, ci_hi = bootstrap_violation_rate(m_recs)
            model_prompt_rates[(m, pt)] = (rate, ci_lo, ci_hi, len(m_recs))
            print(f"  {m:<30} {rate:>14.1%} [{ci_lo:>7.1%}, {ci_hi:>7.1%}] {len(m_recs):>6}")

    # ==================================================================
    # 3. McNEMAR'S TEST: ETHICAL vs NEUTRAL PER MODEL
    # ==================================================================
    print("\n" + "=" * 80)
    print("3. McNEMAR'S TEST: ETHICAL vs NEUTRAL (paired by scenario+pressure+language)")
    print("=" * 80)
    print(f"{'Model':<30} {'Ethical':>8} {'Neutral':>8} {'chi2':>8} {'p-value':>10} {'Sig?':>5}")
    print("-" * 75)

    mcnemar_results = {}
    for m in models:
        eth_recs = [r for r in records if r["model"] == m and r["prompt_type"] == "ethical"]
        neu_recs = [r for r in records if r["model"] == m and r["prompt_type"] == "neutral"]

        if not eth_recs or not neu_recs:
            print(f"{m:<30} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'':>5}")
            continue

        eth_rate = np.mean([r["tool_called"] for r in eth_recs])
        neu_rate = np.mean([r["tool_called"] for r in neu_recs])
        chi2, p_val, b, c = mcnemar_test(eth_recs, neu_recs)

        sig = "***" if p_val is not None and p_val < 0.001 else \
              "**" if p_val is not None and p_val < 0.01 else \
              "*" if p_val is not None and p_val < 0.05 else ""

        mcnemar_results[m] = (chi2, p_val, eth_rate, neu_rate)

        if p_val is not None:
            print(f"{m:<30} {eth_rate:>7.1%} {neu_rate:>7.1%} {chi2:>8.2f} {p_val:>10.4f} {sig:>5}")
        else:
            print(f"{m:<30} {eth_rate:>7.1%} {neu_rate:>7.1%} {'N/A':>8} {'N/A':>10} {'':>5}")

    # ==================================================================
    # 4. PERMUTATION TEST: LANGUAGE EFFECTS (pairwise vs EN)
    # ==================================================================
    print("\n" + "=" * 80)
    print("4. PERMUTATION TEST: LANGUAGE EFFECTS (10k permutations)")
    print("   All models pooled. Two-sided test.")
    print("=" * 80)

    # Overall per language
    print(f"\n  {'Language':<10} {'Violation Rate':>15} {'N':>8}")
    print(f"  {'-' * 35}")
    lang_rates = {}
    for lang in languages:
        l_recs = [r for r in records if r["language"] == lang]
        rate = np.mean([r["tool_called"] for r in l_recs])
        lang_rates[lang] = (rate, len(l_recs))
        print(f"  {lang:<10} {rate:>14.1%} {len(l_recs):>8}")

    # Pairwise permutation tests (all pairs vs EN, plus EN vs HI specifically)
    print(f"\n  Pairwise permutation tests vs EN:")
    print(f"  {'Comparison':<15} {'Rate A':>8} {'Rate B':>8} {'Diff':>8} {'p-value':>10} {'Sig?':>5}")
    print(f"  {'-' * 60}")

    for lang in languages:
        if lang == "en":
            continue
        rate_en, rate_other, diff, p_val = permutation_test_languages(records, "en", lang)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  EN vs {lang.upper():<7} {rate_en:>7.1%} {rate_other:>7.1%} {diff:>+7.1%} {p_val:>10.4f} {sig:>5}")

    # Per-model language effects
    print(f"\n  Per-model EN vs HI permutation tests:")
    print(f"  {'Model':<30} {'EN rate':>8} {'HI rate':>8} {'Diff':>8} {'p-value':>10} {'Sig?':>5}")
    print(f"  {'-' * 73}")
    for m in models:
        m_recs = [r for r in records if r["model"] == m]
        rate_en, rate_hi, diff, p_val = permutation_test_languages(m_recs, "en", "hi")
        if p_val is not None:
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {m:<30} {rate_en:>7.1%} {rate_hi:>7.1%} {diff:>+7.1%} {p_val:>10.4f} {sig:>5}")

    # ==================================================================
    # 5. CHI-SQUARED TEST: PRESSURE TYPE EFFECTS
    # ==================================================================
    print("\n" + "=" * 80)
    print("5. CHI-SQUARED TEST: PRESSURE TYPE EFFECTS")
    print("=" * 80)

    # Overall
    chi2, p_val, dof, table_dict = chi_squared_pressure(records)
    print(f"\n  Overall (all models pooled):")
    print(f"  chi2 = {chi2:.2f}, df = {dof}, p = {p_val:.4e}")
    print(f"\n  {'Pressure Type':<20} {'Violation Rate':>15} {'Violations':>12} {'Total':>8}")
    print(f"  {'-' * 60}")
    for p in sorted(table_dict.keys()):
        d = table_dict[p]
        print(f"  {p:<20} {d['rate']:>14.1%} {d['violated']:>12} {d['total']:>8}")

    # Per model
    print(f"\n  Per-model chi-squared tests for pressure type:")
    print(f"  {'Model':<30} {'chi2':>10} {'df':>4} {'p-value':>12} {'Sig?':>5}")
    print(f"  {'-' * 65}")
    for m in models:
        m_recs = [r for r in records if r["model"] == m]
        try:
            chi2, p_val, dof, _ = chi_squared_pressure(m_recs)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {m:<30} {chi2:>10.2f} {dof:>4} {p_val:>12.4e} {sig:>5}")
        except Exception as e:
            print(f"  {m:<30} {'ERROR':>10} -- {str(e)[:30]}")

    # ==================================================================
    # 6. COHEN'S h EFFECT SIZES FOR KEY COMPARISONS
    # ==================================================================
    print("\n" + "=" * 80)
    print("6. COHEN'S h EFFECT SIZES")
    print("=" * 80)

    # 6a. Ethical vs Neutral per model
    print(f"\n  6a. Ethical vs Neutral per model:")
    print(f"  {'Model':<30} {'Ethical':>8} {'Neutral':>8} {'Cohen h':>9} {'Magnitude':>12}")
    print(f"  {'-' * 70}")
    for m in models:
        eth_recs = [r for r in records if r["model"] == m and r["prompt_type"] == "ethical"]
        neu_recs = [r for r in records if r["model"] == m and r["prompt_type"] == "neutral"]
        if not eth_recs or not neu_recs:
            continue
        p_eth = np.mean([r["tool_called"] for r in eth_recs])
        p_neu = np.mean([r["tool_called"] for r in neu_recs])
        h = cohen_h(p_eth, p_neu)
        mag = interpret_cohen_h(h)
        print(f"  {m:<30} {p_eth:>7.1%} {p_neu:>7.1%} {h:>+8.3f} {mag:>12}")

    # 6b. EN vs HI per model
    print(f"\n  6b. EN vs HI per model:")
    print(f"  {'Model':<30} {'EN':>8} {'HI':>8} {'Cohen h':>9} {'Magnitude':>12}")
    print(f"  {'-' * 70}")
    for m in models:
        en_recs = [r for r in records if r["model"] == m and r["language"] == "en"]
        hi_recs = [r for r in records if r["model"] == m and r["language"] == "hi"]
        if not en_recs or not hi_recs:
            continue
        p_en = np.mean([r["tool_called"] for r in en_recs])
        p_hi = np.mean([r["tool_called"] for r in hi_recs])
        h = cohen_h(p_en, p_hi)
        mag = interpret_cohen_h(h)
        print(f"  {m:<30} {p_en:>7.1%} {p_hi:>7.1%} {h:>+8.3f} {mag:>12}")

    # 6c. None pressure vs highest pressure
    print(f"\n  6c. 'none' pressure vs other pressures (all models pooled):")
    print(f"  {'Comparison':<25} {'Rate A':>8} {'Rate B':>8} {'Cohen h':>9} {'Magnitude':>12}")
    print(f"  {'-' * 65}")
    none_recs = [r for r in records if r["pressure_type"] == "none"]
    p_none = np.mean([r["tool_called"] for r in none_recs]) if none_recs else 0
    for ptype in sorted(pressures):
        if ptype == "none":
            continue
        p_recs = [r for r in records if r["pressure_type"] == ptype]
        p_rate = np.mean([r["tool_called"] for r in p_recs])
        h = cohen_h(p_rate, p_none)
        mag = interpret_cohen_h(h)
        print(f"  {'none vs ' + ptype:<25} {p_none:>7.1%} {p_rate:>7.1%} {h:>+8.3f} {mag:>12}")

    # 6d. Best vs worst model
    print(f"\n  6d. Best vs worst model (overall):")
    best_model = min(model_overall, key=lambda m: model_overall[m][0])
    worst_model = max(model_overall, key=lambda m: model_overall[m][0])
    p_best = model_overall[best_model][0]
    p_worst = model_overall[worst_model][0]
    h = cohen_h(p_worst, p_best)
    mag = interpret_cohen_h(h)
    print(f"  Best:  {best_model} ({p_best:.1%})")
    print(f"  Worst: {worst_model} ({p_worst:.1%})")
    print(f"  Cohen's h = {h:+.3f} ({mag})")

    # 6e. Biggest language gap per model
    print(f"\n  6e. Biggest language gap per model:")
    print(f"  {'Model':<30} {'Best Lang':>10} {'Worst Lang':>11} {'Rates':>15} {'Cohen h':>9} {'Magnitude':>12}")
    print(f"  {'-' * 90}")
    for m in models:
        lang_rates_m = {}
        for lang in languages:
            l_recs = [r for r in records if r["model"] == m and r["language"] == lang]
            if l_recs:
                lang_rates_m[lang] = np.mean([r["tool_called"] for r in l_recs])
        if len(lang_rates_m) < 2:
            continue
        best_lang = min(lang_rates_m, key=lambda l: lang_rates_m[l])
        worst_lang = max(lang_rates_m, key=lambda l: lang_rates_m[l])
        h = cohen_h(lang_rates_m[worst_lang], lang_rates_m[best_lang])
        mag = interpret_cohen_h(h)
        rates_str = f"{lang_rates_m[best_lang]:.1%} vs {lang_rates_m[worst_lang]:.1%}"
        print(f"  {m:<30} {best_lang:>10} {worst_lang:>11} {rates_str:>15} {h:>+8.3f} {mag:>12}")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 80)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 80)

    # Rank models by violation rate
    print("\n  Models ranked by overall violation rate (lower = better):")
    ranked = sorted(model_overall.items(), key=lambda x: x[1][0])
    for i, (m, (rate, ci_lo, ci_hi, n)) in enumerate(ranked, 1):
        print(f"  {i:>3}. {m:<30} {rate:>7.1%}  [{ci_lo:.1%}, {ci_hi:.1%}]")

    # Significant prompt type effects
    print("\n  Models with significant ethical vs neutral difference (p < 0.05):")
    any_sig = False
    for m in models:
        if m in mcnemar_results and mcnemar_results[m][1] is not None and mcnemar_results[m][1] < 0.05:
            chi2, p_val, eth_rate, neu_rate = mcnemar_results[m]
            direction = "ethical > neutral" if eth_rate > neu_rate else "neutral > ethical"
            print(f"    {m}: p = {p_val:.4f} ({direction})")
            any_sig = True
    if not any_sig:
        print("    (none)")

    print()


if __name__ == "__main__":
    main()
