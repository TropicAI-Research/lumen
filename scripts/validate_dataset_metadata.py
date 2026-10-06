#!/usr/bin/env python3
"""Validate every declared record with mlcroissant using local resource files.

Install mlcroissant==1.1.1 in a separate environment if needed. No downloads
or model calls are made: each content URL is mapped to its checksummed CSV.
"""
import importlib.metadata
import json
from pathlib import Path
import sys

import mlcroissant as mlc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_dataset_metadata import validate_resources


def main():
    metadata = json.loads((ROOT / "croissant.json").read_text())
    expected = validate_resources(metadata, ROOT)
    mapping = {r["@id"]: str(ROOT / r["@id"]) for r in metadata["distribution"]}
    dataset = mlc.Dataset(jsonld=metadata, mapping=mapping)
    counts = {r.id: sum(1 for _ in dataset.records(record_set=r.id)) for r in dataset.metadata.record_sets}
    for recordset in metadata["recordSet"]:
        file_id = recordset["field"][0]["source"]["fileObject"]["@id"]
        assert counts[recordset["@id"]] == expected[file_id]
    path = ROOT / "analysis/verification/dataset_metadata_audit.json"
    report = json.loads(path.read_text())
    report.update({"official_mlcroissant_validation_performed": True,
                   "official_validator_version": importlib.metadata.version("mlcroissant"),
                   "official_extracted_record_counts": counts,
                   "local_mapping_prevented_downloads": True,
                   "metadata_sha256": __import__("hashlib").sha256((ROOT / "croissant.json").read_bytes()).hexdigest(),
                   "publication_date_note": "datePublished intentionally omitted until publication; its absence is a recommended-field warning, not a schema failure."})
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"official_validation": "passed", "extracted_records": counts}))


if __name__ == "__main__":
    main()
