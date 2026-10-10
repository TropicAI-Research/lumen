#!/usr/bin/env python3
"""Render verified results in the original paper's table and figure layouts.

Offline only: consumes audited analysis files, never makes model requests.
Writes only the local manuscript working directory.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="lumen-plot-config-"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="lumen-font-cache-"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_paper_results import TIERS, number, pct, pvalue, table
from scripts.build_supplement import tex, prompt_tex
from scripts.canonical_panel import audit

from scripts.unpack_manuscript import PAPER, ensure_manuscript
OUT = PAPER / "generated"
LANGS = {"en": "English", "es": "Spanish", "pt": "Portuguese", "hi": "Hindi", "zh": "Mandarin"}
NAMES = {
    "claude-haiku-4.5": "Claude-Haiku-4.5", "deepseek-v3.2": "DeepSeek-V3.2",
    "gemini-3.1-flash-lite-preview": "Gemini-3.1-Flash-Lite", "gemini-3.1-pro-preview": "Gemini-3.1-Pro",
    "gemma-4-26b": "Gemma-4-26B", "gemma-4-31b": "Gemma-4-31B", "gpt-4.1": "GPT-4.1",
    "gpt-4.1-mini": "GPT-4.1-mini", "gpt-5-mini": "GPT-5-mini", "gpt-5.4": "GPT-5.4",
    "gpt-oss-120b": "GPT-OSS-120B", "gpt-oss-20b": "GPT-OSS-20B", "gpt-oss-safeguard-20b": "GPT-OSS-Safeguard-20B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "llama-3.3-70b": "Llama-3.3-70B", "llama-4-scout": "Llama-4-Scout", "minimax-m2.7": "MiniMax-M2.7",
    "ministral-8b": "Ministral-8B", "mistral-large-2512": "Mistral-Large-2512", "mistral-small-2603": "Mistral-Small-2603",
    "qwen3-14b": "Qwen3-14B", "qwen3-235b-a22b-2507": "Qwen3-235B-A22B", "qwen3-32b": "Qwen3-32B",
    "qwen3-8b": "Qwen3-8B", "qwen3.5-27b": "Qwen3.5-27B", "qwen3.5-35b-a3b": "Qwen3.5-35B-A3B",
    "qwen3.5-397b-a17b": "Qwen3.5-397B-A17B", "qwen3.5-9b": "Qwen3.5-9B", "sabia4": r"Sabi\'a-4", "sabiazinho-4": "Sabiazinho-4",
}
CLOSED = {m for m in NAMES if m.startswith(("claude-", "gemini-", "sabia", "gpt-4", "gpt-5"))}


def read(relative):
    return json.loads((ROOT / relative).read_text())


def write(name, content):
    (OUT / ("original_" + name + ".tex")).write_text(
        "% Generated offline by scripts/build_original_style_results.py.\n" + content + "\n")


def heat(value):
    for limit, color in [(0.5, "d4f0d4"), (2, "b8e6b8"), (5, "f0f4a8"), (10, "ffe080"), (20, "ffb070"), (35, "ff8080")]:
        if value < limit:
            return r"\cellcolor[HTML]{" + color + "}" + f"{value:.2f}"
    return r"\cellcolor[HTML]{e85050}" + f"{value:.2f}"


def render_pair_table(stats):
    lines = [r"\begin{table}[h]", r"\centering\small",
        r"\caption{Within-model language comparisons on matched scenarios and pressures. Cells give the share of 30 models above, below, or tied with English; colors indicate direction. Ties are mostly zero-rate pairs. One-sided Wilcoxon $p$-values are Holm-adjusted across four languages within each prompt; supported comparisons are bold.}",
        r"\label{tab:paired-langs}", r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{lcccccccc}", r"\toprule",
        r"& \multicolumn{4}{c}{\textit{Ethical prompt}} & \multicolumn{4}{c}{\textit{Neutral prompt}} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"Comparison & $>$EN & $<$EN & Tied & $p_{\rm Holm}$ & $>$EN & $<$EN & Tied & $p_{\rm Holm}$ \\", r"\midrule"]
    for lang in ["hi", "pt", "es", "zh"]:
        cells = [lang.upper() + " vs EN"]
        for prompt in ["ethical", "neutral"]:
            row = next(r for r in stats["comparisons"]["harmful"] if r["prompt"] == prompt and r["language"] == lang)
            cells += [r"\cellcolor[HTML]{ffd0d0}" + pct(row["wins"], row["n_models"], 1) + r"\%",
                      r"\cellcolor[HTML]{d4f0d4}" + pct(row["losses"], row["n_models"], 1) + r"\%", pct(row["ties"], row["n_models"], 1) + r"\%"]
            p = pvalue(row["p_holm"])
            cells.append(r"\textbf{" + p + "}" if row["p_holm"] < .05 else p)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    write("paired_languages", "\n".join(lines))


def render_examples():
    rows = read("data/corpus/manuscript_examples.json")["figure"]
    lines = [r"{\scriptsize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabularx}{\linewidth}{@{}l>{\raggedright\arraybackslash}p{0.32\linewidth}>{\raggedright\arraybackslash}X@{}}",
        r"\toprule", r" & \textbf{USER INPUTS} & \textbf{MODEL: LLAMA-3.3-70B} \\", r"\midrule"]
    for row in rows:
        lang = row["language"]
        font = {"hi": r"\hifont", "zh": r"\zhfont"}.get(lang, "")
        user = "{" + font + " " + prompt_tex(row["question_text"], lang) + "}"
        response = "{" + font + " " + prompt_tex(row["assistant_response"], lang) + "}"
        calls = []
        for call in row["tool_calls"]:
            args = tex(json.dumps(call["arguments"], ensure_ascii=False, separators=(",", ":")))
            args = args.replace(",", r",\allowbreak{}").replace(":", r":\allowbreak{}").replace(r"\_", r"\_\allowbreak{}")
            calls.append(r"{\ttfamily\seqsplit{" + tex(call["name"]) + "}} " + r"{\ttfamily " + args + "}")
        status = r"\textcolor{complyred}{\textbf{COMPLY}}" if calls else r"\textcolor{redirectgreen}{\textbf{REFUSAL; NO CALL}}"
        model = (r"\par ".join(calls) + r"\par\smallskip " if calls else "") + response + r"\par\smallskip " + status
        lines.append(r"\textbf{" + lang.upper() + "} & " + user + " & " + model + r" \\")
        lines.append(r"\addlinespace[5pt]")
    lines += [r"\bottomrule", r"\end{tabularx}}"]
    write("examples", "\n".join(lines))


def meta_odds(frame, members):
    x = frame[(frame.prompt == "ethical") & frame.model.isin(members) & frame.language.isin(["en", "hi"])]
    matched = x.pivot(index=["model", "scenario_id", "pressure"], columns="language", values="harmful").dropna()
    effects, variances = [], []
    for _, rows in matched.groupby(level="model"):
        n = len(rows)
        cells = np.array([rows.hi.sum(), n-rows.hi.sum(), rows.en.sum(), n-rows.en.sum()], dtype=float)
        if np.any(cells == 0):
            cells += .5
        a, b, c, d = cells
        effects.append(np.log(a*d/(b*c)))
        variances.append(np.sum(1/cells))
    y, v = np.array(effects), np.array(variances)
    w = 1/v
    fixed = np.sum(w*y)/w.sum()
    q = np.sum(w*(y-fixed)**2)
    df = len(y)-1
    tau2 = max(0, (q-df)/(w.sum()-np.sum(w*w)/w.sum()))
    weights = 1/(v+tau2)
    pooled = np.sum(weights*y)/weights.sum()
    se = np.sqrt(1/weights.sum())
    return {"or": float(np.exp(pooled)), "ci": [float(np.exp(pooled-1.96*se)), float(np.exp(pooled+1.96*se))],
            "i2": float(max(0, (q-df)/q)*100) if q else 0, "tau2": float(tau2), "models": len(y), "matched_pairs": len(matched)}


def render_figures():
    c = read("analysis/verification/measurement_cohort_audit.json")["unique_triples"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    names = ["o4-mini\nvs\nOpus 4.7", "o4-mini\nvs\nGemini 2.5 Pro", "Opus 4.7\nvs\nGemini 2.5 Pro"]
    vals = [r["cohen_kappa"] for r in c["pairwise"]]
    axes[0].barh(names, vals, color=["#984ea3", "#ff7f00", "#8c564b"], edgecolor="black", linewidth=.5)
    axes[0].set_xlim(.5, 1)
    axes[0].set_xlabel("Cohen's κ (4-class)")
    axes[0].set_title(f"Shared sample (n={c['n']}): pairwise κ", weight="bold", fontsize=11)
    for i, v in enumerate(vals):
        axes[0].text(v+.009, i, f"κ={v:.3f}", va="center", fontsize=10, weight="bold")
    counts = [c["unanimous"], c["majority"]-c["unanimous"], c["n"]-c["majority"]]
    percents = np.array(counts)*100/c["n"]
    axes[1].bar(["All 3\nagree", "2 of 3\nagree", "All differ"], percents,
                color=["#3cb44b", "#d49c1c", "#d73027"], edgecolor="black", linewidth=.5)
    axes[1].set_ylim(0, 110)
    axes[1].set_ylabel("Percentage of cases (%)")
    axes[1].set_title(f"Shared sample (n={c['n']}): Fleiss' κ={c['fleiss_kappa']:.3f}", weight="bold", fontsize=11)
    for i, (n, v) in enumerate(zip(counts, percents)):
        axes[1].text(i, v+1.3, f"{v:.1f}%\n(n={n})", ha="center", fontsize=10, weight="bold")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x" if ax is axes[0] else "y", linestyle=":", alpha=.3)
    for suffix in ["pdf", "png"]:
        fig.savefig(PAPER / f"figure/fig8_3judge_agreement.{suffix}", dpi=180)
    plt.close(fig)
    sandbox = read("analysis/revision/corrected_sandbox_summary.json")
    effects = sandbox["aggregate_consequences"]
    labels = {"modifications": "Records modified", "messages_sent": "Messages sent", "files_deleted": "File deletion entries",
              "surveillance_started": "Surveillance events", "candidates_excluded": "Job candidates excluded",
              "approvals": "Requests approved", "medical_unauthorized_accesses": "Medical-record access flags", "installs": "Software installs"}
    items = sorted(labels, key=lambda k: effects[k], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    vals = [effects[k] for k in items]
    ax.barh([labels[k] for k in items], vals, color="#d73027", edgecolor="black", linewidth=.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals)*1.13)
    for i, n in enumerate(vals):
        ax.text(n+max(vals)*.01, i, f"{n:,}", va="center", fontsize=10, weight="bold")
    ax.set_xlabel(f"Simulated effect counters across {sandbox['total_cases']:,} substantive-call cases")
    ax.set_title("Aggregate effects in the synthetic sandbox", fontsize=13, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=.3)
    for suffix in ["pdf", "png"]:
        fig.savefig(PAPER / f"figure/fig9_sandbox_consequences.{suffix}", dpi=180)
    plt.close(fig)


def main():
    ensure_manuscript()
    OUT.mkdir(exist_ok=True)
    stats = read("analysis/revision/corrected_statistics.json")
    frame = pd.read_csv(ROOT / "analysis/revision/corrected_records.csv.gz", low_memory=False)
    frame = frame.loc[~frame.status.isin(["api_error", "missing_corrected_run"])].copy()
    for col in ["valid", "l0", "l1", "l2", "harmful", "sdg", "semantic_valid"]:
        frame[col] = frame[col].astype("boolean")
    assert audit(frame) == stats["audit"], "Stale analysis inputs"
    models = frame.groupby("model")[["l0", "l1", "l2", "harmful", "sdg"]].sum().astype(int)
    models["n"] = frame.groupby("model").size()
    models["sdg_rate"] = models.sdg / models.l2.replace(0, np.nan)
    assert set(models.index) == set(NAMES) and len(CLOSED) == 9
    render_pair_table(stats)
    render_examples()
    ordered = models.sort_values(["sdg_rate", "sdg"], ascending=False, na_position="last")
    cells = [[NAMES[m], number(row.l2), heat(100*row.sdg_rate) if row.l2 else "---"] for m, row in ordered.iterrows()]
    write("per_model_sdg", table(
        r"Per-model say-do gap rates, ranked by SDG share of substantive-call cases. Colors run from rare (green) to frequent (red). Claude-Haiku-4.5, Gemini-3.1-Pro, and GPT-5.4 have no $L_2$-positive cases, leaving SDG undefined.",
        "tab:per-model-sdg", ["Model", "$n_{L_2}$", "SDG \\%"]*2,
        [cells[i]+cells[i+15] for i in range(15)], columns=r"lrr@{\hspace{2em}}lrr"))
    rows = []
    for m, row in models.assign(rate=models.harmful/models.n).sort_values("rate").iterrows():
        rows.append([NAMES[m], r"\textsf{" + ("C" if m in CLOSED else "O") + "}", number(row.n)] +
                    [heat(100*row[k]/row.n) for k in ["l0", "l1", "l2", "harmful"]])
    write("overall", table(r"Per-model outcomes across all ten language/prompt conditions, ranked by $L_3$-behavioral. $N$ excludes API failures. Access: C = closed weights; O = open weights. Only $L_3$ is a harmful-action measure. Colors run from low (green) to high (red).",
        "tab:per-model-overall", ["Model", "Acc.", "$N$", "$L_0$ (\\%)", "$L_1$ (\\%)", "$L_2$ (\\%)", "$L_3$-beh (\\%)"], rows, size="scriptsize"))
    tier_tests = read("analysis/verification/paper_numbers_audit.json")["price_tests"]
    tier_rows, summary, language_rows, meta = [], [], [], {}
    for tier, members in TIERS.items():
        cells = [tier]
        for lang in ["hi", "pt", "es", "zh"]:
            r = next(x for x in tier_tests if x["tier"] == tier and x["language"] == lang)
            cells.append("---" if r["p_raw"] is None else f"{r['wins']}/{len(members)} ({r['p_raw']:.3f}; {r['p_holm_price_family']:.3f})")
        tier_rows.append(cells)
        rates = 100*models.loc[members, "harmful"]/models.loc[members, "n"]
        summary.append([tier, str(len(members)), f"{rates.median():.2f}", f"{rates.mean():.2f}", f"{rates.min():.2f}--{rates.max():.2f}"])
        meta[tier] = m = meta_odds(frame, members)
        rates_lang = frame[(frame.prompt == "ethical") & frame.model.isin(members)].groupby("language").harmful.mean()
        language_rows.append([tier] + [pct(rates_lang[l]) for l in ["en", "es", "pt", "zh", "hi"]] +
                            [f"{m['or']:.2f} [{m['ci'][0]:.2f}, {m['ci'][1]:.2f}]", f"{m['i2']:.1f}\\%"])
    write("price_paired", table(r"Exploratory within-tier ethical-prompt language tests. Each cell gives models above English / tier size, followed by raw and Holm-adjusted $p$ (raw; adjusted). All testable tier/language contrasts form one Holm family. Dashes indicate fewer than five non-zero pairs; none passes the adjusted threshold.",
        "tab:price-paired-langs", ["Tier", "HI vs EN", "PT vs EN", "ES vs EN", "ZH vs EN"], tier_rows, size="scriptsize"))
    write("price_summary", table(r"Harmful-action percentages by fixed blended-price tier, across the 30-model panel. Prices are archived snapshots. Wide within-tier ranges preclude a deterministic interpretation of price.",
        "tab:price-quartile", ["Tier", "Models", "Median (\\%)", "Mean (\\%)", "Range (\\%)"], summary))
    write("price_languages", table(r"Ethical-prompt harmful-action percentages pooled within each price tier. Hindi/English ORs use the supplementary marginal random-effects approximation described in the text, not a paired test. $I^2$ describes between-model heterogeneity; sparse premium-tier events limit precision.",
        "tab:lang-by-price", ["Tier", "EN (\\%)", "ES (\\%)", "PT (\\%)", "ZH (\\%)", "HI (\\%)", "HI/EN OR [95\\% CI]", "$I^2$"], language_rows, size="scriptsize"))
    pressures = ["none", "social", "urgency", "authority", "incremental", "justification"]
    rows = [[tex(p)]+[pct(frame[(frame.pressure == p) & (frame.prompt == mode)].harmful.mean()) for mode in ["ethical", "neutral"]] for p in pressures]
    write("pressure", table(r"Harmful-action rate by pressure variant, pooled across the 30-model panel. Percentages describe the observed conditions, not their psychological mechanism.", "tab:pressure-rates", ["Pressure", "Ethical (\\%)", "Neutral (\\%)"], rows))
    cats = frame[frame.prompt == "neutral"].groupby("category").harmful.mean().sort_values(ascending=False)
    rows = [[tex(c)]+[pct(frame[(frame.category == c) & (frame.prompt == p)].harmful.mean()) for p in ["ethical", "neutral"]] for c in cats.index]
    write("categories", table(r"Harmful-action rate by ethical category, sorted by neutral-prompt rate. Category differences do not identify their cause.", "tab:category-rates", ["Category", "Ethical (\\%)", "Neutral (\\%)"], rows))
    pressure = (OUT / "primary_pressure.tex").read_text()
    write("pressure_interactions", pressure[pressure.index("For each non-baseline pressure"):])
    fit = stats["regression"]
    rows = [[tex(term), f"${r['beta']:+.3f}$", f"{r['odds_ratio']:.3f} [{r['cluster_ci95'][0]:.3f}, {r['cluster_ci95'][1]:.3f}]", pvalue(r["cluster_p"])] for term, r in fit["coefficients"].items()]
    write("regression", table(f"Logistic regression with model fixed effects ($N={number(fit['n_trials'])}$, {fit['n_models']} models). Reference levels: EN, ethical prompt, autonomy, authority pressure. Confidence intervals and nominal Wald tests cluster by model. Intercept and model coefficients are omitted for space.",
        "tab:logreg", ["Term", "$\\beta$", "OR [cluster 95\\% CI]", "Cluster $p$"], rows))
    rows = []
    for lang, name in LANGS.items():
        cells = [name]
        for prompt in ["ethical", "neutral", None]:
            f = frame[frame.language == lang]
            if prompt:
                f = f[f.prompt == prompt]
            a, n = int(f.sdg.sum()), int(f.l2.sum())
            lo, hi = proportion_confint(a, n, method="wilson")
            cells.append(f"{pct(a,n)}\\% [{pct(lo)}, {pct(hi)}]")
        rows.append(cells)
    write("sdg_languages", table(r"SDG share of $L_2$-positive cases by language and prompt. Wilson 95\% intervals summarize binomial counting uncertainty conditional on the judge labels, not annotation uncertainty or model-cluster uncertainty. These pooled shares do not replace the paired language tests.",
        "tab:saydo-by-language", ["Language", "Ethical prompt", "Neutral prompt", "Pooled"], rows))
    zh = frame[(frame.language == "zh") & (frame.prompt == "ethical")]
    total = int(zh.sdg.sum())
    pieces = []
    for model in ["llama-3.1-8b", "llama-3.3-70b"]:
        f = zh[zh.model == model]
        a, n = int(f.sdg.sum()), int(f.l2.sum())
        pieces.append(f"{NAMES[model]} contributes {a} of the {total} Mandarin ethical-prompt SDG cases ({pct(a,total,1)}\\%); its within-cell SDG share is {a}/{n} ({pct(a,n,1)}\\%).")
    write("sdg_family", " ".join(pieces))
    inventory = pd.read_json(ROOT / "data/corpus/parallel_scenarios.jsonl", lines=True)
    full_cats = inventory.drop_duplicates("scenario_id").groupby("category").size()
    common_cats = frame.drop_duplicates("scenario_id").groupby("category").size()
    assert full_cats.sum() == 108 and common_cats.sum() == 108
    write("category_inventory", table("Scenario counts in the authored benchmark inventory.",
        "tab:category-inventory", ["Category", "Scenarios"], [[tex(c),str(n)] for c,n in full_cats.items()]))
    render_figures()
    files = sorted(OUT.glob("original_*.tex")) + [PAPER / "figure/fig8_3judge_agreement.pdf", PAPER / "figure/fig9_sandbox_consequences.pdf"]
    report = {"analysis": stats["audit"], "supplementary_marginal_odds_ratios": meta,
              "files": [{"path":str(p.relative_to(ROOT)), "sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in files], "network_calls":0}
    (ROOT / "analysis/verification/original_style_results_audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"generated":len(files), "observations":len(frame), "network_calls":0}))


if __name__ == "__main__":
    main()
