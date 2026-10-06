import unittest
from scripts.audit_measurement_cohorts import summarize_triples


class MeasurementCohortTests(unittest.TestCase):
    def test_three_different_labels_have_no_majority(self):
        rows = [{"o4_mini_label": "SAY_DO_GAP", "opus_label": "COMPLY", "gemini_label": "AMBIGUOUS"},
                {"o4_mini_label": "COMPLY", "opus_label": "COMPLY", "gemini_label": "COMPLY"}]
        result = summarize_triples(rows)
        self.assertEqual(result["majority"], 1)
        self.assertEqual(result["majority_sdg"], 0)
        self.assertEqual(result["unanimous"], 1)

    def test_binary_agreement_keeps_say_do_in_harmful_group(self):
        row = {"o4_mini_label": "SAY_DO_GAP", "opus_label": "COMPLY", "gemini_label": "REDIRECT"}
        result = summarize_triples([row])
        self.assertEqual(result["pairwise"][0]["binary_agreement"], 1)
        self.assertEqual(result["pairwise"][2]["comply_redirect_inversions"], 1)

    def test_invalid_label_is_not_nonviolation(self):
        with self.assertRaises(ValueError):
            summarize_triples([{"o4_mini_label": "ERROR", "opus_label": "COMPLY", "gemini_label": "REDIRECT"}])


if __name__ == "__main__":
    unittest.main()
