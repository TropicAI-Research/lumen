import unittest

from scripts.build_paper_results import TIERS, number, pct, pvalue, table


class PaperResultsTest(unittest.TestCase):
    def test_tiers_are_unique_and_complete(self):
        models = [m for group in TIERS.values() for m in group]
        self.assertEqual(len(models), 30)
        self.assertEqual(len(set(models)), 30)

    def test_denominators_and_rounding(self):
        self.assertEqual(number(195840), "195{,}840")
        self.assertEqual(pct(1396, 16753), "8.33")
        self.assertEqual(pct(0, 0), "---")
        self.assertEqual(pvalue(0.0651212), "0.0651")

    def test_table_is_valid_tex(self):
        result = table("Caption", "tab:test", ["A", "B"], [["x", "1"]])
        self.assertIn(r"\centering\small", result)
        self.assertIn("x & 1 " + r"\\", result)
        self.assertEqual(result.count(r"\begin{table}"), 1)
        self.assertEqual(result.count(r"\end{table}"), 1)


if __name__ == "__main__":
    unittest.main()
