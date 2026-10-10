#!/usr/bin/env python3
"""Generate real CSV resources and checksummed Croissant metadata offline.

The descriptor covers inputs and calibration/control artifacts. The
full response panel and raw response/provenance files remain
separate release resources. No upload, account access, or model API call.
Croissant extraction follows the official CSV-column specification:
https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html
"""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyze_benign_control import load_benign
from scripts.audit_benign_sources import relative_source


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def unique_judgments(rows):
    seen = {}
    for row in rows:
        key = "|".join(row[c] for c in ["model", "lang", "prompt", "question_id"])
        if key in seen and row["label"] != seen[key]["label"]:
            raise ValueError("Conflicting historical judgments: " + key)
        if key not in seen:
            seen[key] = {"key": key, **row}
    return list(seen.values())


def flat(row):
    return {key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict))
            else "" if value is None else str(value).lower() if isinstance(value, bool) else value
            for key, value in row.items()}


def validate_resources(metadata, root):
    files = {d["@id"]: d for d in metadata["distribution"]}
    fields_by_file, counts = {}, {}
    for name, resource in files.items():
        if not re.fullmatch(r"[0-9a-f]{64}", resource["sha256"]):
            raise ValueError("Not a SHA-256 checksum: " + name)
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != resource["sha256"]:
            raise ValueError("Resource checksum mismatch: " + name)
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields_by_file[name] = set(reader.fieldnames or [])
            counts[name] = sum(1 for _ in reader)
    for recordset in metadata["recordSet"]:
        for field in recordset["field"]:
            source = field["source"]
            if source["extract"]["column"] not in fields_by_file[source["fileObject"]["@id"]]:
                raise ValueError("Field cannot be extracted: " + field["@id"])
    return counts


def main():
    parallel_source = ROOT / "data/corpus/parallel_scenarios.jsonl"
    triple_source = ROOT / "data/calibration/crossjudge_unique_triples.jsonl"
    human_source = ROOT / "data/calibration/human_cases.jsonl"
    judge_source = ROOT / "data/calibration/judge_labels.jsonl"
    benign_source = ROOT / "benign_control_scored/benign_scored_records.json"
    parallel = []
    for row in jsonl(parallel_source):
        parallel.append({**{k:v for k,v in row.items() if k != "base_texts"},
                         **{"text_"+lang: text for lang, text in row["base_texts"].items()}})
    tables = [
        ("parallel_inputs", parallel, "108 Portuguese-authored scenarios with translations and pressure variants. No native translation validation.", parallel_source),
        ("crossjudge", jsonl(triple_source), "Unique stored responses from the declared model roster, each with three model-judge labels. Labels apply to these exact responses only.", triple_source),
        ("human_calibration", jsonl(human_source), "Portuguese cases labelled by one author, restricted to the model roster; unequal retained judge-label strata.", human_source),
        ("historical_judge_labels", unique_judgments(jsonl(judge_source)), "Unique source-response judgments for the declared models. Redundant entries remain in the source JSONL; the first recorded reason is retained here.", judge_source),
        ("benign_controls", [{**r, "result_file":relative_source(r["result_file"])} for r in load_benign(benign_source)], "3300 fixed benign observations and mechanical completion scores. Corresponding harmful outcomes and conditional analyses are linked separately in analysis/revision/.", benign_source),
    ]
    context = {"@language": "en", "@vocab": "https://schema.org/", "sc": "https://schema.org/",
        "cr": "http://mlcommons.org/croissant/", "dct": "http://purl.org/dc/terms/",
        "conformsTo": "dct:conformsTo", "dataType": {"@id": "cr:dataType", "@type": "@vocab"}}
    # Standard context keys used by the official mlcroissant implementation.
    # Keep the generator dependency-free; validate with mlcroissant separately.
    context["rai"] = "http://mlcommons.org/croissant/RAI/"
    context.update({k: "cr:"+k for k in ["recordSet", "field", "source", "fileObject", "extract", "column",
        "citeAs", "equivalentProperty", "fileProperty", "fileSet", "format", "includes", "isLiveDataset",
        "jsonPath", "key", "md5", "parentField", "path", "references", "regex", "repeated", "replace",
        "samplingRate", "separator", "subField", "transform"]})
    context.update({k: {"@id": "cr:" + k, "@type": "@json"} for k in ["data", "examples"]})
    metadata = {"@context": context, "@type": "sc:Dataset", "name": "Lumen-bench",
        "description": "Multilingual simulated tool-calling safety benchmark. This descriptor covers parallel inputs, original human/model judge evidence, and fixed benign controls. Corrected-response evaluation results are versioned separately and must not be inferred from these calibration tables.",
        "url": "https://huggingface.co/datasets/Tropic-AI/lumen-bench",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "version": "2.1.0", "datePublished": "2026-10-10", "dateModified": "2026-10-10",
        "citeAs": "Bonás, Giovana Kerche; Laitz, Thiago; Almeida, Thales Sales; Santos, João Guilherme Alves (2026). Lumen-bench: Cross-Lingual Safety Gaps and the Say-Do Problem in Tool-Calling Large Language Models. https://github.com/TropicAI-Research/lumen",
        "creator": {"@type": "sc:Organization", "name": "Tropic AI", "url": "https://github.com/TropicAI-Research/lumen"},
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "inLanguage": ["pt", "en", "es", "zh", "hi"], "distribution": [], "recordSet": []}
    sources = []
    for name, raw_rows, description, source_path in tables:
        rows = [flat(row) for row in raw_rows]
        columns = sorted({key for row in rows for key in row})
        path = ROOT / "data/tables" / (name + ".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        relative = path.relative_to(ROOT).as_posix()
        metadata["distribution"].append({"@type": "cr:FileObject", "@id": relative,
            "name": name, "description": description, "encodingFormat": "text/csv",
            "contentUrl": "https://huggingface.co/datasets/Tropic-AI/lumen-bench/resolve/main/" + relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        metadata["recordSet"].append({"@type": "cr:RecordSet", "@id": name, "name": name,
            "description": description, "field": [{"@type": "cr:Field", "@id": name + "/" + col,
                "name": col, "dataType": "sc:Text", "source": {"fileObject": {"@id": relative},
                    "extract": {"column": col}}} for col in columns]})
        sources.append({"path": str(source_path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()})
    counts = validate_resources(metadata, ROOT)
    (ROOT / "croissant.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    report = {"uploaded": False, "checks": "Local CSV extraction, field existence, row counts, and SHA-256 checksums.",
        "official_mlcroissant_validation_performed": False,
        "resource_rows": counts, "source_files": sources,
        "scope": "Inputs, calibration and fixed benign controls only. No claim of complete corrected response coverage.",
        "specification": "https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html"}
    (ROOT / "analysis/verification/dataset_metadata_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"uploaded": False, "resource_rows": counts, "local_resource_checks_passed": True}, indent=2))


if __name__ == "__main__":
    main()
