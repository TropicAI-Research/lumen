"""The public landing page stays small; complete sources remain reproducible."""
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

from scripts.unpack_manuscript import ROOT, unpack


class PublicationDocsTests(unittest.TestCase):
    def test_readme_has_no_dataset_metadata_header_or_duplicate_sources(self):
        text = (ROOT / "README.md").read_text()
        self.assertTrue(text.startswith("# Lumen-bench\n"))
        self.assertLess(len(text), 4000)
        for unwanted in ["config_name:", "license:", "original_corrigido", "manuscript source"]:
            self.assertNotIn(unwanted, text)

    def test_relative_readme_links_resolve(self):
        text = (ROOT / "README.md").read_text()
        links = re.findall(r"\]\(([^)]+)\)", text)
        for link in links:
            if not link.startswith(("https://", "#")):
                self.assertTrue((ROOT / link).is_file(), link)

    def test_huggingface_metadata_remains_separate_and_complete(self):
        text = (ROOT / "configs/dataset_card.yaml").read_text()
        self.assertEqual(text.count("config_name:"), 5)
        for path in re.findall(r"path: (.+)", text):
            self.assertTrue((ROOT / path).is_file(), path)

    def test_unpack_preserves_the_complete_verified_archive(self):
        with tempfile.TemporaryDirectory(prefix="lumen-unpack-test-") as tmp:
            target = unpack(Path(tmp))
            manifest = json.loads((ROOT / "release/build_manifest.json").read_text())
            self.assertEqual(len(list(target.rglob("*.*"))), len(manifest["source_files"]))
            for row in manifest["source_files"]:
                name = "neurips_2026.tex" if row["archive_path"] == "main.tex" else row["archive_path"]
                self.assertEqual(hashlib.sha256((target / name).read_bytes()).hexdigest(), row["sha256"])

    def test_unpack_never_overwrites_existing_edits(self):
        with tempfile.TemporaryDirectory(prefix="lumen-unpack-test-") as tmp:
            target = Path(tmp)
            edited = target / "neurips_2026.tex"
            edited.write_text("A local author edit")
            with self.assertRaises(FileExistsError):
                unpack(target)
            self.assertEqual(edited.read_text(), "A local author edit")

    def test_unpack_rejects_a_symlink_destination(self):
        with tempfile.TemporaryDirectory(prefix="lumen-unpack-test-") as tmp:
            parent = Path(tmp)
            target = parent / "real"
            target.mkdir()
            linked = parent / "linked"
            linked.symlink_to(target, target_is_directory=True)
            with self.assertRaises(AssertionError):
                unpack(linked)
            self.assertEqual(list(target.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
