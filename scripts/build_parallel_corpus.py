#!/usr/bin/env python3
"""Create a side-by-side corpus and mechanical integrity report, not validation."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.repair_language_inputs import sources, audit


def main():
    source = sources()
    questions = {lang: json.loads((ROOT / f"questions_v3_{lang}.json").read_text()) for lang in source}
    by_scenario = {lang: {sid: [q for q in rows if q["scenario_id"] == sid] for sid in source[lang]}
                   for lang, rows in questions.items()}
    out = ROOT / "data/corpus"
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with (out / "parallel_scenarios.jsonl").open("w") as stream:
        for sid in source["pt"]:
            category = by_scenario["pt"][sid][0]["expected_state"]["category"]
            counts[category] += 1
            row = {"scenario_id": sid, "category": category, "source_language": "pt",
                "source_authoring": "Authors; coauthor review and suggestions.",
                "translation_model": "anthropic/claude-sonnet-4.6",
                "native_translation_validation": False,
                "base_texts": {lang: source[lang][sid]["question"] for lang in source},
                "pressure_variants": {lang: {q["pressure_type"]: {"question_id": q["question_id"], "question": q["question"]}
                                             for q in by_scenario[lang][sid]} for lang in source}}
            assert all(len(row["pressure_variants"][lang]) == 6 for lang in source)
            assert all(row["pressure_variants"][lang]["none"]["question"] == row["base_texts"][lang] for lang in source)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {**audit(), "category_counts": dict(sorted(counts.items())),
        "question_count_per_language": {lang: len(rows) for lang, rows in questions.items()},
        "test_scope": "Explicit field coverage, stable IDs, script presence and source alignment only; not a semantic-equivalence or native-speaker assessment.",
        "source_files": [{"path": f"questions_v3_{lang}.json", "sha256": hashlib.sha256((ROOT/f"questions_v3_{lang}.json").read_bytes()).hexdigest()} for lang in source]}
    (out / "integrity_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "source_files"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
