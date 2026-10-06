#!/usr/bin/env python3
"""Download version-pinned public calibration evidence (no model calls)."""
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "calibration"
SOURCE = "lumen-bench/lumen-bench"
REVISION = "e08bb5397a994f13855135809e5fa88a350e60bb"
FILES = ("human_calibration.json", "crossjudge_triples.jsonl", "crossjudge_agreement.json",
         "judge_labels.jsonl", "sandbox_summary.json")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"source_dataset": SOURCE, "source_revision": REVISION, "files": []}
    for name in FILES:
        path = OUT / name
        url = f"https://huggingface.co/datasets/{SOURCE}/resolve/{REVISION}/{name}"
        if not path.exists():
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
        content = path.read_bytes()
        if name.endswith(".jsonl"):
            rows = [json.loads(line) for line in content.splitlines() if line.strip()]
            count = len(rows)
        else:
            json.loads(content)
            count = None
        entry = {"path": str(path.relative_to(ROOT)), "source_url": url,
                 "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content), "rows": count}
        manifest["files"].append(entry)
        print(json.dumps({"file": name, "bytes": len(content), "rows": count}))
    (OUT / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
