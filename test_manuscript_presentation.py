"""Keep the final manuscript self-contained and its populations explicit."""
import json
from pathlib import Path
import re
import tempfile
import unittest

from scripts.build_arxiv_release import collect_sources, ROOT
from scripts.unpack_manuscript import unpack
from scripts.build_repair_coverage import coverage_status, serialize_coverage


class ManuscriptPresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="lumen-presentation-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.paper = unpack(Path(cls.temporary.name))

    def test_abstract_reports_total_trials_with_error_accounting_in_methods(self):
        text = (self.paper / "neurips_2026.tex").read_text()
        abstract = text.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]
        self.assertIn(r"The evaluation comprises 194{,}400 trials.", abstract)
        self.assertNotIn(r"194{,}368", abstract)
        self.assertIn("32 end in malformed tool-output errors", text)
        self.assertIn("excluded from harmful-action rates, not counted as refusals", text)

    def test_compact_coverage_preserves_records_and_unicode(self):
        report = {"scope": "Português / 中文", "totals": {"planned": 2},
                  "cells": [{"key": "one", "label": None}, {"key": "two", "needs_judgment": False}]}
        encoded = serialize_coverage(report)
        self.assertEqual(json.loads(encoded), report)
        self.assertEqual(sum(line.lstrip().startswith('{"key":') for line in encoded.splitlines()), 2)

    def test_active_text_has_no_revision_narrative(self):
        pattern = re.compile(
            r"\b(?:reruns?|re-runs?|reexecut\w*|re-execut\w*|backfilled)\b|"
            r"repair campaign|correction runs|corrected-input|defective localizations",
            re.IGNORECASE,
        )
        for relative in collect_sources(self.paper):
            if relative.suffix != ".tex":
                continue
            content = "\n".join(line for line in (self.paper / relative).read_text().splitlines()
                                if not line.lstrip().startswith("%"))
            self.assertIsNone(pattern.search(content), str(relative))

    def test_analysis_membership_and_missingness_remain_explicit(self):
        text = (self.paper / "evaluation_appendix.tex").read_text()
        stats = json.loads((ROOT / "analysis/revision/corrected_statistics.json").read_text())
        cells = f"{stats['audit']['n_cells']:,}".replace(",", "{,}")
        for marker in ["108 base scenarios", "30 models", "different", "collection periods",
                       "Client/API failures", cells, "source hashes", "missing outcomes rather than model refusals"]:
            self.assertIn(marker, text)
        self.assertIn(r"\input{generated/evaluation_coverage.tex}", text)

    def test_full_model_coverage_matches_the_analysis(self):
        report = json.loads((ROOT / "analysis/revision/repair_coverage.json").read_text())
        full = report["full_panel_coverage"]
        self.assertEqual(len(full["by_model"]), 30)
        for row in full["by_model"].values():
            self.assertEqual(row["planned"], 6480)
            self.assertEqual(row["planned"], sum(row.get(k, 0) for k in ["completed", "error", "not_executed"]))
        self.assertEqual(full["totals"]["planned"], 194400)
        self.assertEqual(full["totals"]["planned"], sum(full["totals"][k] for k in ["completed", "error", "not_executed"]))
        stats = json.loads((ROOT / "analysis/revision/corrected_statistics.json").read_text())
        self.assertEqual(full["totals"]["completed"], stats["audit"]["n_cells"])

    def test_coverage_never_relabels_errors_as_completed(self):
        self.assertEqual(coverage_status("original_valid_input"), "completed")
        self.assertEqual(coverage_status("corrected_run"), "completed")
        self.assertEqual(coverage_status("recovered_api_error"), "completed")
        self.assertEqual(coverage_status("api_error"), "error")
        self.assertEqual(coverage_status("missing_corrected_run"), "not_executed")
        with self.assertRaises(ValueError):
            coverage_status("recorded_wrong_language")


if __name__ == "__main__":
    unittest.main()
