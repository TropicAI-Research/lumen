import unittest
from scripts.audit_release_files import allowed, inspect_bytes


class ReleaseFileTests(unittest.TestCase):
    def test_excludes_private_or_unrelated_files(self):
        for name in [".env", ".env.backup", "data/language_repair/budget.json", "NEURIPS_2026_REBUTTAL.md",
                     "Lumen_bench_For_NeurIPS_2026/neurips_2026.tex", "project.zip",
                     "analysis/verification/github_destination_audit.json", "data/other/.env", "../outside.py",
                     "scripts/__pycache__/runner.pyc", ".git/config", "release/private.zip", "release/.env"]:
            self.assertFalse(allowed(name), name)

    def test_keeps_auditable_sources_and_not_root_pdf(self):
        for name in [".env.example", "scripts/canonical_panel.py", "data/language_repair/runs/m/q/result.json",
                     "eval_results_consolidated/pt/ethical/m/consolidated.json",
                     "eval_results_benign_control_v2/pt/ethical/m/results_20260726_170000.json",
                     "eval_results_benign_control_v2/pt/ethical/m/incremental_20260726_170000.jsonl",
                     "eval_results_benign_control_v2/pt/ethical/m/incremental_reused_exact_protocol.jsonl",
                     "Lumen_bench_For_NeurIPS_2026 (1)/neurips_2026.tex", "release/lumen-bench.pdf", "release/lumen-bench-arxiv.tar.gz"]:
            self.assertTrue(allowed(name), name)
        self.assertFalse(allowed("Lumen_bench_For_NeurIPS_2026 (1)/neurips_2026.pdf"))

    def test_findings_contain_no_secret_values(self):
        fake = b"sk-or-v1-" + b"q" * 64
        result = inspect_bytes(b"value=" + fake, {"TEST_CREDENTIAL": fake})
        self.assertIn("provider_key_shape", result)
        self.assertIn("configured_value:TEST_CREDENTIAL", result)
        self.assertNotIn(fake.decode(), str(result))

    def test_prefix_embedded_in_opaque_blob_is_not_a_standalone_key(self):
        blob = b"aaaaask-" + b"q" * 80
        self.assertNotIn("provider_key_shape", inspect_bytes(blob, {}))
        self.assertIn("configured_value:TEST", inspect_bytes(blob, {"TEST": blob[5:]}))


if __name__ == "__main__":
    unittest.main()
