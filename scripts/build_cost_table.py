#!/usr/bin/env python3
"""Render audited historical counters without claiming an unverifiable bill."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_supplement import tex
from scripts.unpack_manuscript import PAPER, ensure_manuscript


def main():
    ensure_manuscript()
    data = json.loads((ROOT / "analysis/verification/historical_cost_audit.json").read_text())
    rows = [r"\begin{table}[htbp]", r"\centering\small",
        r"\caption{Usage counters and input/output price snapshots for the 194{,}400 scheduled observations in the base response archive. Token columns are millions of tokens. Reasoning and completion counters can overlap. Zero I/O counts observations with zero input and completion counters, not free API calls. The table describes this archive, not total project expenditure.}",
        r"\label{tab:cost-breakdown}", r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{lrrrrrr}",
        r"\toprule", r"Model & Input \$/M & Output \$/M & Input (M) & Completion (M) & Reasoning (M) & Zero I/O \\", r"\midrule"]
    for model, counts in sorted(data["models"].items()):
        rows.append(tex(model) + " & " + " & ".join([
            f"{counts['archived_input_price_per_million']:.3f}",
            f"{counts['archived_output_price_per_million']:.3f}",
            f"{counts['prompt_tokens']/1e6:.3f}", f"{counts['completion_tokens']/1e6:.3f}",
            f"{counts['reasoning_tokens']/1e6:.3f}", str(counts['zero_input_and_output_counters'])]) + r" \\")
    rows.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"])
    path = PAPER / "generated/cost_counters.tex"
    path.write_text("\n".join(rows) + "\n")
    print(json.dumps({"models": len(data["models"]), "observations": data["n_observations"], "output": str(path.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
