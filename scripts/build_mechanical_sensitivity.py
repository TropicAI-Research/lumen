#!/usr/bin/env python3
"""Judge-free paired language checks on complete common-scenario inputs.

L1 includes defensive and ineffective calls; it is not semantic harmfulness.
No API calls. Missing semantic labels do not affect this mechanical outcome.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.canonical_panel import load_panel, audit
from scripts.analyze_revision import paired, select_available_cases
from scripts.unpack_manuscript import PAPER, ensure_manuscript


def main():
    ensure_manuscript()
    source_frame = load_panel("corrected")
    frame = select_available_cases(source_frame)
    if not bool(frame.valid.all() and frame.input_matches_current.all() and frame.l1.notna().all()):
        raise SystemExit("Common-scenario mechanical outcomes or inputs are incomplete.")
    comparisons = paired(frame, "l1")
    report = {"panel": "corrected", "metric": "l1", "n_scenarios": 108,
        "n_models": int(frame.model.nunique()), "n_observations": len(frame),
        "excluded_api_error_keys": source_frame.loc[source_frame.status == "api_error", "key"].tolist(),
        "semantic_judge_used_for_this_outcome": False, "comparisons": comparisons,
        "no_pressure": paired(frame[frame.pressure == "none"], "l1"),
        "full_panel_audit_including_separate_semantic_coverage": audit(frame),
        "method": "Same model, scenario and pressure in target and English. One-sided Wilcoxon, target greater than English, normal approximation, zero_method=wilcox, rounded differences at 12 decimals. Holm family of four languages per prompt.",
        "interpretation": "Non-read-only tool use, including defensive calls and ineffective attempts. Does not establish semantic harm, a say-do gap, or translation equivalence."}
    out = ROOT / "analysis/verification"
    out.mkdir(parents=True, exist_ok=True)
    (out / "mechanical_language_sensitivity.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [r"\begin{table}[htbp]", r"\centering\small",
        r"\caption{Judge-free language sensitivity on 108 scenarios and 30 models. The outcome is any non-read-only call ($L_1$), not harmfulness. Unavailable outcomes and their counterpart in the other language are excluded before pairing; complete models contribute 648 pairs. $\Delta$ is the mean within-model difference in percentage points; Holm correction uses four languages per prompt.}",
        r"\label{tab:mechanical-language}", r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Prompt & Language vs. EN & $\Delta$ (pp) & $>$ / $<$ / tied & Raw $p$ & Holm $p$ \\", r"\midrule"]
    for r in comparisons:
        lines.append(f"{r['prompt'].capitalize()} & {r['language'].upper()} & {r['mean_difference_pp']:.2f} & " +
                     f"{r['wins']} / {r['losses']} / {r['ties']} & {r['p_raw']:.4f} & {r['p_holm']:.4f} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (PAPER / "generated/mechanical_language.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps([{k:v for k,v in row.items() if k != "model_rates"} for row in comparisons], indent=2))


if __name__ == "__main__":
    main()
