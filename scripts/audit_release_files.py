#!/usr/bin/env python3
"""Hash an explicit publication file set and screen it for credentials locally.

This does not upload, copy, initialize Git, or change repository visibility.
The file list is deliberately narrower than the working directory. It excludes
credentials, budget-key identifiers, private access checks, rebuttals, obsolete
manuscript copies, build caches, and unrelated files. A clean result is not a
publication-readiness decision and does not replace the final history scan.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_language_repairs import env_values

ROOT_NAMES = {
    "README.md", "LICENSE", ".env.example", ".gitignore", ".gitleaksignore", "requirements.txt", "croissant.json",
    "all_scenarios_reduced.py", "generate_questions_v3.py", "eval_v3.py", "metrics.py", "metrics_reconciled.py",
    "sandbox_env.py", "sandbox_run.py", "official_prices.json", "analyze_benign_control.py", "benign_control.py",
    "benign_scenarios.py", "benign_translations.json", "benign_control_lock.json", "benign_control_routing.json",
    "generate_benign_control.py", "score_benign_control.py", "run_benign_control_matrix.py",
    "import_benign_control_reuse.py", "prepare_benign_model_configs.py", "run_safeguard_remainder.py", "revision_status.json",
    *{f"questions_v3_{lang}.json" for lang in ["pt", "en", "es", "zh", "hi"]},
}
TREES = {
    "scripts", "tool", "configs/models", "configs/models_benign_control", "eval_results_consolidated",
    "eval_results_benign_control_v2", "benign_control_questions", "benign_control_scored_v2",
    "data/corpus", "data/tables", "data/calibration", "data/provenance/original_source",
    "data/language_repair/runs", "data/language_repair/judge_runs", "data/language_repair/original_questions",
    "data/language_repair/failed_attempts", "data/language_repair/failed_turn_attempts",
    "analysis/revision",
}
REPAIR_FILES = {
    "execution_plan.json", "judge_plan.json", "repair_manifest.json", "source_before.json",
    "translations.json", "translation_request.json", "translation_response.json", "translation_parsing.json",
    "provider_amendments.json", "provider_amendments_v1.json", "provider_amendments_v3.json", "provider_amendments_v4.json", "provider_amendments_v5.json", "provider_amendments_v6.json",
    "endpoint_preflight.json", "endpoint_preflight_v3.json",
    "provider_amendments_v7.json",
}
VERIFICATION_FILES = {
    "original_evidence_audit.json", "measurement_cohort_audit.json", "mechanical_language_sensitivity.json",
    "dataset_metadata_audit.json", "historical_cost_audit.json",
    "legacy_api_error_audit.json", "paper_numbers_audit.json", "repair_protocol_audit.json",
    "benign_source_audit.json",
}
MANUSCRIPT = "Lumen_bench_For_NeurIPS_2026 (1)"
SECRET_PATTERNS = {
    "provider_key_shape": re.compile(rb"(?<![A-Za-z0-9_-])(?:sk-(?:or-v1-|ant-api\d+-)?[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9_]{24,}|github_pat_[A-Za-z0-9_]{24,}|xai-[A-Za-z0-9_-]{24,}|AIza[A-Za-z0-9_-]{30,})"),
    "numeric_provider_key_shape": re.compile(rb"(?<![A-Za-z0-9_])\d{12,30}_[a-f0-9]{16,64}(?![A-Za-z0-9_])"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "authorization_header": re.compile(rb"(?i)authorization[\s\"':=]+bearer\s+[a-z0-9_.-]{24,}"),
    "credential_url": re.compile(rb"https?://[^\s/:\"<>]+:[^\s/@\"<>]{12,}@"),
}


def allowed(relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        return False
    if any(part in {".git", "__pycache__", ".codex", ".agents", ".aws"} for part in path.parts):
        return False
    if path.name.startswith(".env") and str(path) != ".env.example":
        return False
    if str(path) in ROOT_NAMES or (len(path.parts) == 1 and re.fullmatch(r"test_[a-z_0-9]+\.py", path.name)):
        return True
    if path.parent.as_posix() == "data/language_repair":
        return path.name in REPAIR_FILES
    if path.parent.as_posix() == "analysis/verification":
        return path.name in VERIFICATION_FILES
    if path.parent.as_posix() == "release":
        return path.name in {"lumen-bench.pdf", "lumen-bench-arxiv.tar.gz", "build_manifest.json"}
    if str(path) == "web-data/judge_index.json":
        return True
    if path.parts[0] == MANUSCRIPT:
        # All current figures are generated from records in TeX. Old PDF figures
        # contain stale counts and are deliberately not part of this release.
        return path.suffix in {".tex", ".sty", ".bib"}
    for tree in TREES:
        if str(path).startswith(tree + "/"):
            if tree == "eval_results_benign_control_v2":
                return (path.name == "incremental_reused_exact_protocol.jsonl" or
                        bool(re.fullmatch(r"results_[0-9_]+\.json|incremental_[0-9_]+\.jsonl", path.name)))
            if tree.startswith("eval_results_"):
                return path.name == "consolidated.json"
            return path.suffix in {".py", ".json", ".jsonl", ".csv", ".html"} or (tree == "analysis/revision" and path.name.endswith(".csv.gz"))
    return False


def inspect_bytes(data, configured):
    findings = [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(data)]
    findings.extend("configured_value:" + name for name, value in configured.items() if value in data)
    return findings


def main():
    configured = {name: value.encode() for name, value in env_values().items() if value and len(value) >= 8}
    candidates = [p for p in ROOT.iterdir() if p.is_file() and allowed(p.name)]
    for tree in sorted(TREES | {MANUSCRIPT, "analysis/verification", "data/language_repair", "release"}):
        base = ROOT / tree
        if base.exists():
            candidates.extend(p for p in base.rglob("*") if p.is_file() and allowed(p.relative_to(ROOT).as_posix()))
    candidates.append(ROOT / "web-data/judge_index.json")
    manifest, findings = [], []
    for path in sorted(set(candidates)):
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError("Symlink or external source in publication set.")
        data = path.read_bytes()
        relative = path.relative_to(ROOT).as_posix()
        manifest.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        for kind in inspect_bytes(data, configured):
            findings.append({"path": relative, "rule": kind})
    out = ROOT / "analysis/verification"
    benign_audit = json.loads((out / "benign_source_audit.json").read_text())
    missing_raw = set(benign_audit["source_files"]) - {row["path"] for row in manifest}
    assert not missing_raw, "Publication would omit raw benign evidence: " + repr(sorted(missing_raw))
    (out / "release_candidate_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    report = {"files_scanned": len(manifest), "bytes_scanned": sum(r["bytes"] for r in manifest),
        "configured_values_checked": len(configured), "pattern_findings": findings,
        "credential_values_printed": False, "remote_writes": False, "publication_ready": False,
        "scope": "Explicit candidate file list; current contents only. Does not certify unpublished future edits or remote history.",
        "generic_scanner_note": "Local prefix/header/private-key pattern checks. A separate Gitleaks scan of the staged files and history is required before publication."}
    (out / "release_candidate_security.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if findings:
        raise SystemExit("Publication candidate has possible credentials; values were not printed.")


if __name__ == "__main__":
    main()
