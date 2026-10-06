import unittest
from scripts.build_supplement import benign_table


class SupplementTests(unittest.TestCase):
    def rows(self):
        return [
            {"model": "a", "benign_success": True, "pair_valid": True, "harmful_violation": True},
            {"model": "a", "benign_success": True, "pair_valid": False, "harmful_violation": None},
            {"model": "b", "benign_success": False, "pair_valid": False, "harmful_violation": None},
        ]

    def test_default_refuses_incomplete_harmful_pairs(self):
        with self.assertRaises(ValueError):
            benign_table(self.rows())

    def test_benign_denominator_does_not_shrink_with_harmful_coverage(self):
        text = benign_table(self.rows(), allow_incomplete=True)
        self.assertIn("a & 2/2 & 1/2 & 1/1 & 1/1", text)
        self.assertIn("b & 0/1 & 0/1 & --- & ---", text)
        self.assertIn("Incomplete paired analysis", text)

    def test_complete_pairs_have_no_incomplete_warning(self):
        text = benign_table(self.rows()[:1])
        self.assertNotIn("Incomplete paired analysis", text)

    def test_available_case_table_identifies_exclusions(self):
        text = benign_table(self.rows(), available_case=True)
        self.assertIn("1 available harmful outcomes out of 3 planned pairs", text)
        self.assertNotIn("provisional", text)

    def test_available_case_does_not_hide_pending_judgments(self):
        rows = self.rows()
        rows[1]["harmful_judgment_pending"] = True
        with self.assertRaises(ValueError):
            benign_table(rows, available_case=True)


if __name__ == "__main__":
    unittest.main()
