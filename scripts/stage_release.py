#!/usr/bin/env python3
"""Copy only audited publication files to an isolated staging directory.

No upload or visibility change. Static mode permits preparing immutable raw
evidence while corrected analyses are still being finalized. Every copied
source must match the screened manifest; unrelated workspace files cannot
enter the stage through a recursive workspace copy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
STATIC = ("eval_results_consolidated/", "eval_results_benign_control_v2/",
          "benign_control_questions/", "benign_control_scored_v2/", "data/calibration/",
          "data/provenance/original_source/", "data/language_repair/original_questions/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    security = json.loads((ROOT / "analysis/verification/release_candidate_security.json").read_text())
    assert not security["pattern_findings"], "Candidate failed credential screening"
    records = json.loads((ROOT / "analysis/verification/release_candidate_manifest.json").read_text())
    selected = [r for r in records if not args.static or r["path"].startswith(STATIC)]
    stage = args.directory.resolve() if args.directory else Path(tempfile.mkdtemp(prefix="lumen-public-"))
    if not stage.name.startswith("lumen-public-") or stage.parent != Path("/tmp"):
        raise SystemExit("Staging must use a dedicated /tmp/lumen-public-* directory.")
    stage.mkdir(exist_ok=True)
    for row in selected:
        source = ROOT / row["path"]
        data = source.read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"], "Changed after screening: " + row["path"]
        target = stage / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == row["sha256"]:
            continue
        shutil.copy2(source, target)
    receipt = {"stage": str(stage), "mode": "static" if args.static else "full",
               "files": len(selected), "bytes": sum(r["bytes"] for r in selected),
               "source_manifest_sha256": hashlib.sha256((ROOT / "analysis/verification/release_candidate_manifest.json").read_bytes()).hexdigest(),
               "uploaded": False}
    (ROOT / "analysis/verification/local_staging.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
