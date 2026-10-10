#!/usr/bin/env python3
"""Build prompts and versioned benign tables; incomplete output requires opt-in.

This is offline. It never reruns or changes the fixed benign observations.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_v3 import get_system_prompt
from scripts.analyze_repaired_benign import load_versioned_pairs
from scripts.canonical_panel import digest
from scripts.unpack_manuscript import PAPER, ensure_manuscript

OUT = PAPER / "generated"


def tex(text):
    escapes = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$",
               "{": r"\{", "}": r"\}", "\\": r"\textbackslash{}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escapes.get(c, c) for c in str(text))


def prompt_tex(text, language):
    if language == "hi":
        # The Devanagari font does not contain Latin AI. Switch font only for
        # those glyphs; the released prompt string is not changed.
        return "".join((r"{\normalfont " + tex(part) + "}") if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", part)
                       else tex(part) for part in re.split(r"([A-Za-z][A-Za-z0-9_-]*)", text))
    if language == "zh":
        chunks = []
        for i, c in enumerate(text):
            is_han = "\u3400" <= c <= "\u9fff"
            prior_break = i and ("\u3400" <= text[i-1] <= "\u9fff" or text[i-1] in "，。；：！？、")
            if is_han and prior_break:
                chunks.append(r"\allowbreak{}")
            chunks.append(tex(c))
        return "".join(chunks)
    return tex(text)


def percentage(n, d):
    return "---" if not d else f"{100*n/d:.1f}\\%"


def benign_table(rows, allow_incomplete=False, available_case=False):
    missing = sum(not r["pair_valid"] for r in rows)
    if available_case and any(r.get("harmful_judgment_pending") for r in rows):
        raise ValueError("Available-case tables cannot omit pending semantic judgments.")
    if missing and not (allow_incomplete or available_case):
        raise ValueError(f"{missing} harmful outcomes are unavailable or unjudged; refusing a final benign table.")
    models = sorted({r["model"] for r in rows})
    lines = ["% Generated from current versioned harmful responses and fixed benign scores."]
    if missing and available_case:
        lines.append(f"The paired analysis has {len(rows)-missing:,} available harmful outcomes out of {len(rows):,} planned pairs. " +
                     f"The remaining {missing} outcomes are unavailable or API-failed and are excluded, never counted as refusals. " +
                     "All fixed benign observations remain in the benign-completion denominator.\n")
    elif missing:
        lines.append(r"\noindent\textbf{Incomplete paired analysis:} " +
                     f"{missing} harmful outcomes are unavailable or await judgment. " +
                     "The counts below are provisional; unavailable outcomes are not refusals.\n")
    lines.extend([r"\begin{table}[htbp]", r"\centering\small",
        r"\caption{Fixed benign completion and versioned paired harmful action. Benign completion uses all 110 benign trials per model. Harmful counts use only pairs with a valid response and, when needed, a semantic label. The final column restricts those available pairs to successful benign trials. A zero harmful count alone does not establish useful safe behavior.}",
        r"\label{tab:benign-per-model}", r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Model & Benign completed & Pairs available & Harmful / available & Harmful / capable available \\",
        r"\midrule"])
    for model in models:
        group = [r for r in rows if r["model"] == model]
        valid = [r for r in group if r["pair_valid"]]
        capable = [r for r in valid if r["benign_success"]]
        completed = sum(bool(r["benign_success"]) for r in group)
        harm = sum(bool(r["harmful_violation"]) for r in valid)
        conditional = sum(bool(r["harmful_violation"]) for r in capable)
        harm_cell = f"{harm}/{len(valid)}" if valid else "---"
        conditional_cell = f"{conditional}/{len(capable)}" if capable else "---"
        lines.append(f"{tex(model)} & {completed}/{len(group)} & {len(valid)}/{len(group)} & " +
                     f"{harm_cell} & {conditional_cell} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true", help="Explicitly mark provisional paired counts.")
    parser.add_argument("--available-case", action="store_true", help="Explicit available-case cohort; rejects any unjudged eligible response.")
    args = parser.parse_args()
    ensure_manuscript()
    _, rows = load_versioned_pairs()
    # Validate before writing any output, including the otherwise valid prompts.
    table = benign_table(rows, args.allow_incomplete, args.available_case)
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = []
    names = {"pt": "Portuguese", "en": "English", "es": "Spanish", "zh": "Chinese", "hi": "Hindi"}
    for lang, name in names.items():
        for condition in ["neutral", "ethical"]:
            prompts.append(r"\subsection*{" + name + ": " + condition + "}")
            font = {"zh": r"\zhfont", "hi": r"\hifont"}.get(lang, "")
            prompts.append(r"\begin{quote}{\small " + font + "\n" + prompt_tex(get_system_prompt(lang, condition), lang).replace("\n\n", "\n\\par\n") + "\n}\\end{quote}\n")
    (OUT / "system_prompts.tex").write_text("\n".join(prompts), encoding="utf-8")
    (OUT / "benign_tables.tex").write_text(table, encoding="utf-8")
    if args.available_case:
        report = json.loads((ROOT / "analysis/revision/benign_statistics.json").read_text())
        assert report["paired_records_sha256"] == digest(rows), "Regenerate the paired statistics before rendering their summary."
        for name, expected_completed in [("gemini-3.1-pro-preview", 110), ("claude-haiku-4.5", 61)]:
            example = [r for r in rows if r["model"] == name]
            assert len(example) == 110 and all(r["pair_valid"] for r in example), "Update the manuscript's per-model example if its coverage changes."
            assert sum(r["benign_success"] for r in example) == expected_completed
            assert not any(r["harmful_violation"] for r in example), "Update the manuscript's zero-harm example."
        overall = report["overall_paired"]
        outcomes = overall["pair_outcomes"]
        harm = outcomes.get("unsafe_and_capable", 0)
        completed = report["all_benign_completed"]
        total = report["all_benign_trials"]
        low, high = overall["conditional_hvr_cluster95"]
        summary = (r"\subsection{Distinguishing safety from tool competence}" + "\n"
            "To distinguish withholding a harmful action from inability to use the tools, we evaluated 11 matched benign tasks "
            "across 30 models, five languages, and both prompts. "
            f"Models complete {completed:,} of {total:,} benign requests ({100*completed/total:.1f}\\%). "
            f"Among {overall['valid_pairs']:,} pairs with available harmful outcomes, "
            f"{overall['competent_pairs']:,} also have successful benign completion. "
            f"In this demonstrated-completion subset, {harm} harmful actions remain: "
            f"{100*overall['conditional_harmful_violation_rate']:.2f}\\% "
            f"(model-cluster 95\\% CI [{100*low:.2f}\\%, {100*high:.2f}\\%]), "
            f"compared with {100*overall['harmful_violation_rate']:.2f}\\% before filtering for benign success. "
            "This shows harmful actions on operations that the same model can perform legitimately in the same language and prompt condition. "
            "It does not establish general capability or isolate ethical intent through minimal wording changes. "
            "Appendix~\\ref{app:benign-controls} gives the pairing design, missing outcomes, and all per-model results.\n")
        (OUT / "benign_summary.tex").write_text(summary)
        held = report["excluding_scorer_development_models"]
        held_low, held_high = held["conditional_hvr_cluster95"]
        detail = (f"Among the {overall['competent_pairs']:,} available pairs with benign completion, "
                  f"{outcomes.get('safe_and_capable', 0):,} have no harmful action and {harm} do. "
                  f"A further {outcomes.get('safe_but_capability_unresolved', 0)} non-violations have no benign completion "
                  "and do not establish safe-and-capable behavior. "
                  f"Excluding the five scorer-development models leaves {held['valid_pairs']:,} available pairs; "
                  f"harmful action among the {held['competent_pairs']:,} benign-success pairs is "
                  f"{100*held['conditional_harmful_violation_rate']:.2f}\\% "
                  f"(model-cluster 95\\% CI [{100*held_low:.2f}\\%, {100*held_high:.2f}\\%]).\n")
        (OUT / "benign_detail.tex").write_text(detail)
    print(json.dumps({"models": len({r["model"] for r in rows}),
                     "valid_pairs": sum(bool(r["pair_valid"]) for r in rows),
                     "total_pairs": len(rows), "system_prompts": len(names)*2}))


if __name__ == "__main__":
    main()
