#!/usr/bin/env python3
"""Render audited historical counters without claiming an unverifiable bill."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_supplement import tex


def main():
    data = json.loads((ROOT / "analysis/verification/historical_cost_audit.json").read_text())
    rows = [r"\begin{table}[htbp]", r"\centering\small",
        r"\caption{Historical usage counters and archived input/output price snapshots. Token columns are millions of tokens. Reasoning and completion counters are not guaranteed to be disjoint billing categories. Zero I/O counts observations with both input and completion counters equal to zero, not free API calls. This table excludes the later correction runs.}",
        r"\label{tab:cost-breakdown}", r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{lrrrrrr}",
        r"\toprule", r"Model & Input \$/M & Output \$/M & Input (M) & Completion (M) & Reasoning (M) & Zero I/O \\", r"\midrule"]
    for model, counts in sorted(data["models"].items()):
        rows.append(tex(model) + " & " + " & ".join([
            f"{counts['archived_input_price_per_million']:.3f}",
            f"{counts['archived_output_price_per_million']:.3f}",
            f"{counts['prompt_tokens']/1e6:.3f}", f"{counts['completion_tokens']/1e6:.3f}",
            f"{counts['reasoning_tokens']/1e6:.3f}", str(counts['zero_input_and_output_counters'])]) + r" \\")
    rows.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"])
    path = ROOT / "Lumen_bench_For_NeurIPS_2026 (1)/generated/cost_counters.tex"
    path.write_text("\n".join(rows) + "\n")
    print(json.dumps({"models": len(data["models"]), "observations": data["n_observations"], "output": str(path.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
