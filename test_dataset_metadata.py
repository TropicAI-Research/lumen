import hashlib
import tempfile
import unittest
from pathlib import Path
from scripts.build_dataset_metadata import unique_judgments, validate_resources


class MetadataTests(unittest.TestCase):
    def test_duplicate_judgments_are_not_extra_cases(self):
        row = {"model": "m", "lang": "en", "prompt": "neutral", "question_id": "q", "label": "COMPLY"}
        self.assertEqual(len(unique_judgments([row, dict(row)])), 1)
        with self.assertRaises(ValueError):
            unique_judgments([row, {**row, "label": "REDIRECT"}])

    def test_hashes_and_extraction_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = b"id,value\na,one\n"
            (root / "table.csv").write_bytes(payload)
            metadata = {"distribution": [{"@id": "table.csv", "sha256": hashlib.sha256(payload).hexdigest()}],
                "recordSet": [{"field": [{"@id": "table/id", "source": {
                    "fileObject": {"@id": "table.csv"}, "extract": {"column": "id"}}}]}]}
            self.assertEqual(validate_resources(metadata, root), {"table.csv": 1})
            metadata["recordSet"][0]["field"][0]["source"]["extract"]["column"] = "absent"
            with self.assertRaises(ValueError):
                validate_resources(metadata, root)

    def test_url_is_not_a_checksum(self):
        metadata = {"distribution": [{"@id": "file", "sha256": "https://example.org/issue"}], "recordSet": []}
        with self.assertRaises(ValueError):
            validate_resources(metadata, Path("/tmp"))


if __name__ == "__main__":
    unittest.main()
