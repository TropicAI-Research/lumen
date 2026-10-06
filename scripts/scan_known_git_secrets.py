#!/usr/bin/env python3
"""Read-only check of the local Git object store and reflog messages.

Only counts and Git object IDs are reported. This complements a generic
secret scanner; it does not claim to discover unknown or external copies.
"""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_language_repairs import env_values
from scripts.audit_release_files import SECRET_PATTERNS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/verification/local_known_secret_scan.json")
    args = parser.parse_args()
    repository = args.repository.resolve()
    configured = {name: value.encode() for name, value in env_values().items() if value and len(value) >= 8}
    metadata = subprocess.run(["git", "cat-file", "--batch-all-objects",
                               "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
                              cwd=repository, capture_output=True, check=True).stdout
    objects = [(parts[0], parts[1], int(parts[2])) for line in metadata.splitlines()
               if len(parts := line.split()) == 3]
    matches = []
    pattern_findings = []
    total_bytes = 0
    with subprocess.Popen(["git", "cat-file", "--batch"], cwd=repository, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
        for oid, object_type, expected_size in objects:
            proc.stdin.write(oid+b"\n")
            proc.stdin.flush()
            header = proc.stdout.readline().split()
            if len(header) != 3 or int(header[2]) != expected_size:
                raise ValueError("Unexpected Git object response")
            content = proc.stdout.read(expected_size)
            if len(content) != expected_size or proc.stdout.read(1) != b"\n":
                raise ValueError("Incomplete Git object read")
            total_bytes += expected_size
            for name, value in configured.items():
                if value in content:
                    matches.append({"object_id": oid.decode(), "object_type": object_type.decode(), "credential_variable": name})
            for rule, pattern in SECRET_PATTERNS.items():
                if pattern.search(content):
                    pattern_findings.append({"object_id": oid.decode(), "object_type": object_type.decode(), "rule": rule})
        proc.stdin.close()
        proc.wait(timeout=10)
    for scope, command in [("reflog_messages", ["git", "reflog", "show", "--all", "--format=%gs"]),
                           ("reference_names", ["git", "for-each-ref", "--format=%(refname)"])]:
        content = subprocess.run(command, cwd=repository, capture_output=True, check=True).stdout
        total_bytes += len(content)
        for name, value in configured.items():
            if value in content:
                matches.append({"scope": scope, "credential_variable": name})
        for rule, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                pattern_findings.append({"scope": scope, "rule": rule})
    report = {"scope": "all local Git objects including unreachable objects, plus reflog messages and reference names",
        "blob_count": sum(t == b"blob" for _, t, _ in objects), "object_count": len(objects), "bytes_scanned": total_bytes,
        "configured_values_checked": len(configured), "matching_objects": matches,
        "credential_pattern_findings": pattern_findings,
        "credential_values_printed": False, "read_only": True,
        "limitation": "Exact configured values plus local credential-shape patterns. Final staged-tree and full generic history scanning are still required; remote or external copies are not certified."}
    output = args.output
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if matches or pattern_findings:
        raise SystemExit("Publication blocked: a configured credential or possible credential pattern appears in local Git history.")


if __name__ == "__main__":
    main()
