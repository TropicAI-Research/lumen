#!/usr/bin/env python3
"""Generate audited tables for the 30-model, 108-scenario manuscript, offline."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_revision import paired, require_final_panel, select_available_cases
from scripts.build_supplement import tex
from scripts.canonical_panel import audit, load_panel

from scripts.unpack_manuscript import PAPER, ensure_manuscript
OUT = PAPER / "generated"
LANGS = {"en": "English", "es": "Spanish", "pt": "Portuguese", "zh": "Mandarin", "hi": "Hindi"}
TIERS = {
    "Q1 Budget": ["gpt-oss-20b", "gpt-oss-120b", "gpt-oss-safeguard-20b", "gemma-4-26b", "llama-3.1-8b", "llama-3.3-70b", "llama-4-scout", "ministral-8b", "qwen3.5-9b"],
    "Q2 Low": ["deepseek-v3.2", "gemma-4-31b", "minimax-m2.7", "mistral-small-2603", "qwen3-8b", "qwen3-32b", "sabiazinho-4"],
    "Q3 Mid": ["gpt-4.1-mini", "gpt-5-mini", "gemini-3.1-flash-lite-preview", "mistral-large-2512", "qwen3-14b", "qwen3-235b-a22b-2507", "qwen3.5-27b", "qwen3.5-35b-a3b"],
    "Q4 Premium": ["claude-haiku-4.5", "gpt-4.1", "gpt-5.4", "gemini-3.1-pro-preview", "qwen3.5-397b-a17b", "sabia4"],
}


def number(value):
    return f"{int(value):,}".replace(",", "{,}")


def pct(numerator, denominator=1, digits=2):
    return "---" if not denominator else f"{100*float(numerator)/float(denominator):.{digits}f}"


def pvalue(value):
    if value is None:
        return "---"
    return "$<0.0001$" if value < .0001 else f"{value:.4f}"


def table(caption, label, headers, rows, columns=None, size="small"):
    lines = [r"\begin{table}[htbp]", "\\centering\\" + size,
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{adjustbox}{max width=\linewidth}",
             r"\begin{tabular}{" + (columns or "l" + "r"*(len(headers)-1)) + "}",
             r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    lines += [" & ".join(str(v) for v in row) + r" \\" for row in rows]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]) + "\n"


def write(name, content):
    (OUT / name).write_text("% Generated offline by scripts/build_paper_results.py.\n" + content + "\n")


def main():
    ensure_manuscript()
    stats = json.loads((ROOT / "analysis/revision/corrected_statistics.json").read_text())
    frame = select_available_cases(load_panel("corrected"))
    current = audit(frame)
    require_final_panel(current)
    assert current == stats["audit"], "Regenerate statistics after changing inputs or labels."
    assert set(frame.model) == {m for members in TIERS.values() for m in members}
    OUT.mkdir(exist_ok=True)
    counts = current["counts"]
    j = stats["judge_uncertainty"]
    c = json.loads((ROOT / "analysis/verification/measurement_cohort_audit.json").read_text())["unique_triples"]
    matrix = np.asarray(json.loads((ROOT / "data/calibration/human_calibration.json").read_text())["confusion_matrix_4class"]["matrix_human_x_judge"])
    models = frame.groupby("model").harmful.agg(["sum", "count", "mean"])
    prices = json.loads((ROOT / "analysis/verification/historical_cost_audit.json").read_text())["models"]
    blended = {m: .25*p["archived_input_price_per_million"]+.75*p["archived_output_price_per_million"] for m,p in prices.items()}
    rho = float(spearmanr([blended[m] for m in models.index], models["mean"]).statistic)
    tier_tests = []
    for tier, members in TIERS.items():
        for row in paired(frame[frame.model.isin(members)], "harmful"):
            if row["prompt"] != "ethical":
                continue
            row["tier"] = tier
            if row["wins"]+row["losses"] < 5:
                row["p_raw"] = None
            tier_tests.append(row)
    eligible = [r for r in tier_tests if r["p_raw"] is not None]
    for row,p in zip(eligible, multipletests([r["p_raw"] for r in eligible], method="holm")[1]):
        row["p_holm_price_family"] = float(p)

    interactions = stats["pressure_diagnostics"]["language_interactions"]
    write("primary_pressure.tex", "For each non-baseline pressure, we subtract the no-pressure outcome in the same scenario. We compare this difference between each target language and English, retaining only matched four-cell sets and averaging within model. A two-sided signed-rank test compares these model-level differences. Holm adjustment treats the 20 language-by-pressure interactions within each prompt as one exploratory family. Positive differences indicate that the pressure increment is greater in the target language. These contrasts describe behavior, not culturally equivalent pressure strength or psychological causes.\n\n" + table(
        "Exploratory pressure-by-language interactions, using 30 model-level differences and matched outcomes within each scenario.", "tab:pressure-interactions",
        ["Prompt", "Pressure", "Language", "Interaction (pp)", "$p$ raw", "$p$ Holm"],
        [[r["prompt"], tex(r["pressure"]), LANGS[r["language"]], f"{r['mean_difference_pp']:+.2f}", pvalue(r["p_raw"]), pvalue(r["p_holm"])] for r in interactions], size="scriptsize"))

    sensitivity_rows = []
    for row in stats["comparisons"]["harmful"]:
        ranges = []
        for field in ["model", "category"]:
            values = [r["p_holm"] for x in stats["leave_one_out"] if x["excluded_field"] == field for r in x["comparisons"] if (r["prompt"], r["language"]) == (row["prompt"], row["language"])]
            ranges.append(f"{min(values):.4f}--{max(values):.4f}")
        no = next(r for r in stats["no_pressure"] if (r["prompt"], r["language"]) == (row["prompt"], row["language"]))
        sensitivity_rows.append([row["prompt"], LANGS[row["language"]], pvalue(no["p_holm"]), *ranges])
    sensitivity = table("Language sensitivity checks. Each analysis repeats four-language Holm adjustment within a prompt. Leave-one-out ranges are extrema across exclusions, not confidence intervals.", "tab:language-influence", ["Prompt", "Language", "No pressure $p$", "Leave-one-model $p$ range", "Leave-one-category $p$ range"], sensitivity_rows)
    complete = stats["complete_models_sensitivity"]
    sensitivity += f"Restricting to fully observed models gives {complete['n_models']} models and {number(complete['n_observations'])} observations. Models with any unavailable outcome are excluded entirely for this check.\n\n"
    sensitivity += table("Language comparisons restricted to fully observed models.", "tab:complete-models-language", ["Prompt", "Language", r"Mean $\Delta$ (pp)", "$p$ raw", "$p$ Holm"], [[r["prompt"], LANGS[r["language"]], f"{r['mean_difference_pp']:+.2f}", pvalue(r["p_raw"]), pvalue(r["p_holm"])] for r in complete["comparisons"]])
    write("primary_sensitivity.tex", sensitivity)

    replay = json.loads((ROOT / "analysis/revision/corrected_sandbox_summary.json").read_text())
    assert replay["total_cases"] == counts["l2"] and not replay["missing_semantic_labels"]
    macros = {"COMPLY": "cmply", "SAY_DO_GAP": "sdg", "REDIRECT": "redir", "AMBIGUOUS": "ambig"}
    write("primary_sandbox.tex", table(f"Replay of {number(counts['l2'])} substantive-call responses. Reporting-only tags use fixed handler heuristics; other state changes are not synonymous with harm.", "tab:sandbox-labels", ["Judge label", "Cases", "Changed", r"Changed (\%)", "Reporting only", "Other change"], [["\\"+macros[lab]+"{}", number(r["total"]), number(r["state_changed"]), pct(r["state_changed"],r["total"]), number(r["defensive_only_change"]), number(r["other_change"])] for lab,r in replay["by_label"].items()]))
    write("primary_consequences.tex", table("Aggregate effects across independent resets of the synthetic state. Effect counters are not numbers of real victims or independent incidents.", "tab:sandbox-consequences", ["Sandbox counter", "Total"], [[tex(k),number(v)] for k,v in replay["aggregate_consequences"].items() if k != "candidates_remaining"]))

    strata = ", ".join(str(x) for x in matrix.sum(axis=0))
    calibration = r"\subsection{Population-level reweighting}\label{app:judge-cal-reweight}" + "\n" + f"The Portuguese human calibration contains {matrix.sum()} cases, with {strata} observations in the COMPLY, SDG, REDIRECT, and AMBIGUOUS judge strata, respectively. Raw human label proportions therefore do not estimate a population share. We estimate human labels within each judge stratum and weight them by the stratum's frequency among all {number(counts['l2'])} substantive-call responses. For example, 14 of 21 judge-SDG cases and two of 24 judge-COMPLY cases are human-SDG. Both disagreement directions enter the estimate:\n"
    calibration += r"\[\widehat P(H=\sdg)=\sum_y P(J=y)\widehat P(H=\sdg\mid J=y).\]" + "\n"
    calibration += f"This gives {pct(j['human_implied_sdg_point'])}\\% [{pct(j['human_implied_sdg_ci95'][0])}\\%, {pct(j['human_implied_sdg_ci95'][1])}\\%], compared with {pct(counts['sdg'],counts['l2'])}\\% under the judge. We bootstrap independently within each judge stratum at its observed size (5,000 resamples, seed 42), holding population frequencies fixed. The interval is wide because rare calibration disagreements can receive large population weights. Combining COMPLY and SDG gives a human-implied harmful share of {pct(j['human_implied_harmful_point'])}\\% [{pct(j['human_implied_harmful_ci95'][0])}\\%, {pct(j['human_implied_harmful_ci95'][1])}\\%], versus {pct(counts['harmful'],counts['l2'])}\\% under the judge. These are sensitivity estimates, not bounds. They assume that one Portuguese author's conditional error rates transfer across conditions; they omit inter-rater and language-specific systematic uncertainty and do not establish invariance of paired tests or odds ratios.\n"
    write("primary_calibration.tex", calibration)

    human_rows = [["\\"+macros[lab]+"{}", *[str(v) for v in matrix[i]], str(matrix[i].sum())] for i,lab in enumerate(macros)]
    human_rows.append(["Total", *[str(v) for v in matrix.sum(axis=0)], str(matrix.sum())])
    human_tables = table("Human labels (rows) against o4-mini labels (columns), Portuguese-only model-blind calibration.", "tab:human-confusion", ["Human / judge", r"\cmply{}", r"\sdg{}", r"\redir{}", r"\ambig{}", "Total"], human_rows)
    rows = []
    for i,lab in enumerate(macros):
        p,r = matrix[i,i]/matrix[:,i].sum(),matrix[i,i]/matrix[i].sum()
        rows.append(["\\"+macros[lab]+"{}", f"{p:.3f}", f"{r:.3f}", f"{2*p*r/(p+r):.3f}"])
    human_tables += table("Judge precision, recall and F1 in the stratified human sample; these are not population-weighted estimates.", "tab:judge-cal", ["Label", "Precision", "Recall", "F1"], rows)
    write("human_calibration_tables.tex", human_tables)

    cross_rows = [[" / ".join(r["judges"]).replace("_label", "").replace("_", "-"), pct(r["four_class_agreement"]), f"{r['cohen_kappa']:.3f}", pct(r["binary_agreement"])] for r in c["pairwise"]]
    write("crossjudge_summary.tex", table(f"Agreement on the same {c['n']} unique stored responses. Binary grouping combines COMPLY with SDG and REDIRECT with AMBIGUOUS.", "tab:crossjudge", ["Judges", "4-class agree (\\%)", "$\\kappa$", "Binary agree (\\%)"], cross_rows))
    supported = [(r["prompt"],r["language"]) for r in stats["comparisons"]["harmful"] if r["p_holm"] < .05]
    report = {"panel": "corrected", "audit": current, "language_supported_after_holm": supported, "price_spearman": rho, "price_tests": tier_tests, "generated": []}
    for path in sorted(OUT.glob("*.tex")):
        report["generated"].append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (ROOT / "analysis/verification/paper_numbers_audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"observations": len(frame), "supported": supported, "price_spearman": rho, "network_calls": 0}))


if __name__ == "__main__":
    main()
