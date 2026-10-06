"""Offline regression tests; no model calls or credential reads."""
import json
from pathlib import Path
import unittest

from all_scenarios_reduced import loc
from scripts.repair_language_inputs import audit, sources, digest


class LanguageRepairTests(unittest.TestCase):
    def test_no_english_fallback(self):
        for lang in ["zh", "hi"]:
            with self.assertRaisesRegex(ValueError, "fallback is not allowed"):
                loc(lang, "português", "English", "español")

    def test_unknown_language_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            loc("fr", "português", "English", "español")

    def test_explicit_translations_preserved(self):
        self.assertEqual(loc("zh", "p", "e", "s", "中文", "हिन्दी"), "中文")
        self.assertEqual(loc("hi", "p", "e", "s", "中文", "हिन्दी"), "हिन्दी")

    def test_all_scenario_ids_align(self):
        src = sources()
        for lang in src:
            self.assertEqual(len(src[lang]), 108)
            self.assertEqual(set(src[lang]), set(src["pt"]))

    def test_translation_status_is_explicit(self):
        report = audit()
        self.assertFalse(report["native_speaker_validation"])
        self.assertEqual(len(report["missing_translations"]), 0)

    def test_original_question_design_is_fixed(self):
        for lang in ["pt", "en", "es", "zh", "hi"]:
            rows = json.loads(Path(f"questions_v3_{lang}.json").read_text())
            self.assertEqual(len(rows), 648)
            self.assertEqual(len({r["question_id"] for r in rows}), 648)
            self.assertEqual({r["language"] for r in rows}, {lang})
            self.assertEqual(len({r["scenario_id"] for r in rows}), 108)

    def test_digest_is_content_specific(self):
        self.assertEqual(digest("中文"), digest("中文".encode()))
        self.assertNotEqual(digest("中文"), digest("English"))


if __name__ == "__main__":
    unittest.main()
