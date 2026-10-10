"""The original-style paper keeps its layout without reverting audited facts."""
import json
from pathlib import Path
import re
import subprocess
import unittest

from scripts.build_arxiv_release import collect_sources, ROOT

PAPER = ROOT / "Lumen_bench_original_corrigido"


class OriginalStyleManuscriptTests(unittest.TestCase):
    def setUp(self):
        self.main = (PAPER / "neurips_2026.tex").read_text()
        self.sources = collect_sources(PAPER)
        self.active = "\n".join(
            line for p in self.sources if p.suffix == ".tex"
            for line in (PAPER / p).read_text().splitlines()
            if not line.lstrip().startswith("%"))

    def test_original_main_sections_and_result_tables_remain(self):
        sections = re.findall(r"\\section\{([^}]+)\}", self.main)
        self.assertEqual(sections[:7], ["Introduction", "Related Work", "Benchmark Design", "Experimental Protocol",
                                       "Results", "Discussion and Limitations", "Conclusion"])
        self.assertEqual(sections[7:10], ["Datasheet for Datasets", "Scenarios: representative requests and full inventory",
                                         "Pressure variants: psychological grounding and templates"])
        self.assertNotIn(r"\input{generated/primary_results.tex}", self.main)
        for label in ["tab:paired-langs", "tab:per-model-sdg", "tab:price-tier-composition", "tab:price-paired-langs",
                      "tab:per-model-overall", "tab:saydo-by-language", "tab:lang-by-price"]:
            self.assertEqual(self.active.count(r"\label{" + label + "}"), 1)
        self.assertGreater(self.main.index(r"\input{evaluation_appendix.tex}"), self.main.index(r"\section{Per-model cost breakdown}"))

    def test_all_three_original_figure_concepts_remain(self):
        for label in ["fig:motivating", "fig:fig8", "fig:fig9"]:
            self.assertIn(r"\label{" + label + "}", self.main)
        for name in ["figure/fig8_3judge_agreement.pdf", "figure/fig9_sandbox_consequences.pdf"]:
            self.assertIn(Path(name), self.sources)
        examples = json.loads((ROOT / "data/corpus/manuscript_examples.json").read_text())["figure"]
        hindi = next(r for r in examples if r["language"] == "hi")
        rendered = (PAPER / "generated/original_examples.tex").read_text()
        self.assertIn(hindi["assistant_response"], rendered)
        self.assertIn("configure\\_system", rendered)
        self.assertNotIn(Path("figure/examples.pdf"), self.sources)

    def test_full_panel_numbers_and_annotation_scope(self):
        stats = json.loads((ROOT / "analysis/revision/corrected_statistics.json").read_text())["audit"]
        for n in [stats["n_cells"], stats["counts"]["l2"], stats["counts"]["sdg"]]:
            self.assertIn(f"{n:,}".replace(",", "{,}"), self.main)
        self.assertIn(f"{100*stats['counts']['sdg']/stats['counts']['l2']:.2f}", self.main)
        self.assertEqual(stats["n_models"], 30)
        self.assertEqual(stats["n_scenarios"], 108)
        self.assertIn("one Portuguese-speaking author", self.main)
        for old in ["conservative lower bound", "architecturally independent", "rules out translation noise",
                    "uniform across conditions", "no judge ever labels", "native human raters"]:
            self.assertNotIn(old, self.active)

    def test_claimed_language_decisions_match_adjusted_results(self):
        stats = json.loads((ROOT / "analysis/revision/corrected_statistics.json").read_text())
        actual = {(r["prompt"], r["language"]) for r in stats["comparisons"]["harmful"] if r["p_holm"] < .05}
        self.assertEqual(actual, {("ethical", "hi"), ("ethical", "pt"), ("neutral", "hi"), ("neutral", "es")})
        for row in stats["comparisons"]["harmful"]:
            self.assertIn(f"{row['p_holm']:.4f}", self.active)
        common = json.loads((ROOT / "analysis/revision/common_statistics.json").read_text())
        hi = next(r for r in common["comparisons"]["harmful"] if (r["prompt"],r["language"]) == ("neutral","hi"))
        self.assertIn(f"{hi['p_holm']:.4f}", self.main)
        self.assertIn("no within-tier test remains significant", self.main)

    def test_publication_destinations_and_no_revision_diary(self):
        self.assertIn("https://github.com/TropicAI-Research/lumen", self.active)
        self.assertIn("https://huggingface.co/datasets/Tropic-AI/lumen-bench", self.active)
        for old in ["anonymous.4open.science", "datasets/lumen-bench/lumen-bench"]:
            self.assertNotIn(old, self.active)
        self.assertIsNone(re.search(r"\b(?:reruns?|re-runs?|backfilled)\b|corrected-input|defective localizations", self.active, re.I))
        self.assertIsNone(re.search(r"(?:sk-or-v1-|sk-di-)[A-Za-z0-9_-]{12,}", self.active))

    def test_all_tex_references_resolve_once(self):
        labels = re.findall(r"\\label\{([^}]+)\}", self.active)
        self.assertEqual(len(labels), len(set(labels)))
        references = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", self.active)
        self.assertFalse(set(references)-set(labels))

    def test_regenerated_figures_use_the_current_cohorts(self):
        agreement = subprocess.run(["pdftotext", str(PAPER / "figure/fig8_3judge_agreement.pdf"), "-"],
                                   check=True, text=True, capture_output=True).stdout
        sandbox = subprocess.run(["pdftotext", str(PAPER / "figure/fig9_sandbox_consequences.pdf"), "-"],
                                 check=True, text=True, capture_output=True).stdout
        calibration = json.loads((ROOT / "analysis/verification/measurement_cohort_audit.json").read_text())
        replay = json.loads((ROOT / "analysis/revision/corrected_sandbox_summary.json").read_text())
        self.assertIn(str(calibration["unique_triples"]["n"]), agreement)
        self.assertIn(f"{replay['total_cases']:,}", sandbox)
        self.assertNotIn("production", sandbox)

    def test_excluded_models_are_absent_from_active_sources(self):
        self.assertIsNone(re.search(r"grok|xai|cartagena2026mindgap", self.active, re.I))


if __name__ == "__main__":
    unittest.main()
