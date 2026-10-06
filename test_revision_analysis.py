import pandas as pd
import unittest
from scripts.analyze_revision import matched_rates, compare_differences, require_final_panel, exclude_recorded_api_errors, select_available_cases


class RevisionAnalysisTests(unittest.TestCase):
    def test_missing_target_removes_same_english_case(self):
        common = {"model": "model", "prompt": "ethical", "pressure": "none", "valid": True, "input_matches_current": True}
        frame = pd.DataFrame([
            {**common, "scenario_id": "one", "language": "en", "harmful": False},
            {**common, "scenario_id": "two", "language": "en", "harmful": True},
            {**common, "scenario_id": "one", "language": "hi", "harmful": True},
            {**common, "scenario_id": "two", "language": "hi", "harmful": None}])
        result = matched_rates(frame, "hi", "ethical", "harmful")
        self.assertEqual(result.n_paired_trials.iloc[0], 1)
        self.assertEqual(result.english.iloc[0], 0)
        self.assertEqual(result.target.iloc[0], 1)

    def test_all_ties_are_not_significant(self):
        self.assertEqual(compare_differences([0, 0, 0])["p_raw"], 1)

    def test_zero_pending_labels_does_not_make_missing_runs_final(self):
        with self.assertRaises(ValueError):
            require_final_panel({"counts_are_final": False, "missing_judgments": 0})

    def test_diagnostics_require_explicit_opt_in(self):
        require_final_panel({"counts_are_final": False}, allow_incomplete=True)

    def test_complete_panel_can_generate_final_statistics(self):
        require_final_panel({"counts_are_final": True})

    def test_explicit_error_exclusion_keeps_pending_runs(self):
        frame = pd.DataFrame({"status": ["api_error", "missing_corrected_run", "original_valid_input"]})
        result = exclude_recorded_api_errors(frame)
        self.assertEqual(list(result.status), ["missing_corrected_run", "original_valid_input"])

    def test_available_case_excludes_only_declared_unknown_statuses(self):
        frame = pd.DataFrame({"status": ["api_error", "missing_corrected_run", "original_valid_input"],
            "needs_judgment": [False, False, False], "semantic_valid": [False, False, True],
            "input_matches_current": [True, False, True]})
        self.assertEqual(list(select_available_cases(frame).index), [2])

    def test_available_case_does_not_hide_unjudged_or_wrong_language_records(self):
        for pending, matches in [(True, True), (False, False)]:
            frame = pd.DataFrame({"status": ["corrected_run"], "needs_judgment": [pending],
                "semantic_valid": [not pending], "input_matches_current": [matches]})
            with self.assertRaises(ValueError):
                select_available_cases(frame)


if __name__ == "__main__":
    unittest.main()
