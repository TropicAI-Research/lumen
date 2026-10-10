"""Offline invariants for the static explorer, not new model evaluations."""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"


def packed(name):
    return json.loads(gzip.decompress((SITE / "data" / name).read_bytes()))


class ExplorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((SITE / "data/manifest.json").read_text())
        cls.index = packed("index.json.gz")
        cls.questions = packed("questions.json.gz")

    def test_complete_panel_and_distinct_outcomes(self):
        self.assertEqual(len(self.index), 194400)
        self.assertEqual(len({tuple(row[:3]) for row in self.index}), 194400)
        self.assertEqual(Counter(row[0] for row in self.index), {i:6480 for i in range(30)})
        self.assertEqual(Counter(self.manifest["outcomes"][r[3]] for r in self.index), self.manifest["counts"])
        self.assertEqual(sum(bool(row[4] & 1) for row in self.index), 194368)
        self.assertEqual(sum(bool(row[4] & 8) for row in self.index), 16822)
        self.assertTrue(all(not row[4] for row in self.index if row[3] == 5))

    def test_current_sources_match_the_pinned_manifest(self):
        self.assertRegex(self.manifest["revision"], r"^[0-9a-f]{40}$")
        for path, expected in self.manifest["source_hashes"].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected)

    def test_parallel_scenario_coverage(self):
        scenarios = packed("scenarios.json.gz")
        self.assertEqual(len(scenarios), 108)
        self.assertEqual(len(self.questions), 3240)
        self.assertEqual(len({q["id"] for q in self.questions}), 3240)
        for row in scenarios:
            self.assertEqual(set(row["pressure_variants"]), {"en", "pt", "es", "zh", "hi"})
            self.assertEqual(row["source_language"], "pt")
            self.assertFalse(row["native_translation_validation"])
            self.assertTrue(all(len(v) == 6 for v in row["pressure_variants"].values()))

    def test_calibration_stays_on_its_own_response_version(self):
        judges = packed("judges.json.gz")
        self.assertEqual(len(judges["cross"]), 607)
        self.assertEqual(len(judges["human"]), 83)
        self.assertEqual([sum(r["labels"][i]=="SAY_DO_GAP" for r in judges["cross"]) for i in range(3)], [53,50,42])
        self.assertEqual(sum(r["current_response_match"] for r in judges["cross"]), 597)
        self.assertEqual(sum(r["current_response_match"] for r in judges["human"]), 83)
        self.assertEqual({r["language"] for r in judges["human"]}, {"pt"})
        self.assertEqual(sum(bool(r[6] & 1) for r in self.index), 597)
        self.assertEqual(sum(bool(r[6] & 2) for r in self.index), 83)

    def test_benign_pairs_preserve_fixed_scores(self):
        benign=packed("benign.json.gz")
        self.assertEqual(len(benign["rows"]), 3300)
        self.assertEqual(sum(r["benign_success"] for r in benign["rows"]), 2733)
        self.assertEqual(sum(r["benign_success"] and r["harmful_violation"] for r in benign["rows"]), 271)
        keys = {r["key"] for r in benign["rows"]}
        actual = set()
        for path in (SITE / "data/benign").glob("*.json.gz"):
            for key, value in json.loads(gzip.decompress(path.read_bytes())).items():
                self.assertEqual(key, value["key"])
                self.assertTrue((ROOT / value["source_path"]).is_file())
                actual.add(key)
        self.assertEqual(keys,actual)

    def test_every_exported_response_hash_and_label_matches_analysis(self):
        with gzip.open(ROOT / "analysis/revision/corrected_records.csv.gz", "rt") as stream:
            source = {r["key"]:r for r in csv.DictReader(stream)}
        actual = set()
        paths = list((SITE / "data/responses").glob("*.json.gz"))
        self.assertEqual(len(paths),300)
        for path in paths:
            shard = json.loads(gzip.decompress(path.read_bytes()))
            self.assertEqual(len(shard),648)
            for detail in shard.values():
                key=detail["key"]; expected=source[key];actual.add(key)
                self.assertEqual(detail["source_sha256"],expected["record_sha256"])
                self.assertEqual(detail["source_path"],expected["source_path"])
                self.assertEqual(detail["label"] or "",expected["label"])
                self.assertEqual(detail["mechanical"]["valid"],expected["valid"]=="True")
                if detail["sandbox"]:
                    self.assertEqual(detail["sandbox"]["source_record_sha256"],detail["source_sha256"])
                self.assertNotIn("api_trace", detail)
        self.assertEqual(actual,set(source))

    def test_static_assets_resolve_without_live_api(self):
        html=(SITE / "index.html").read_text()
        for link in re.findall(r'(?:src|href)="([^"]+)"',html):
            if not link.startswith(("https://","#")):
                self.assertTrue((SITE / link).is_file(),link)
        self.assertTrue((SITE / ".nojekyll").exists())
        self.assertNotIn("onclick=",html)
        self.assertIn("script-src 'self'",html)
        js=(SITE / "app.js").read_text()
        self.assertNotIn("api.openrouter",js)
        self.assertNotIn("Authorization",js)
        self.assertNotIn("eval(",js)
        self.assertIn("from './core.js'",js)
        self.assertLess(sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file()),100_000_000)


if __name__ == "__main__":
    unittest.main()
