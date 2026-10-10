#!/usr/bin/env python3
"""Verify every fixed benign score against its released raw response.

The frozen score file is not rewritten. Regeneration is in memory only, with
a portable source index and checksums exported for independent inspection.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from score_benign_control import classify_record, load_questions
from metrics_reconciled import is_api_error
from scripts.canonical_panel import digest


def relative_source(value):
    parts = Path(value).parts
    index = parts.index("eval_results_benign_control")
    return Path(*parts[index:]).as_posix()


def main():
    scored_path = ROOT / "benign_control_scored/benign_scored_records.json"
    scored = json.loads(scored_path.read_text())
    questions = load_questions(ROOT / "benign_control_questions")
    cache, files, rows, differences = {}, {}, [], []
    for saved in scored:
        relative = relative_source(saved["result_file"])
        path = ROOT / relative
        if relative not in cache:
            content = path.read_text()
            try:
                records = json.loads(content)
            except json.JSONDecodeError:
                records = [json.loads(line) for line in content.splitlines() if line.strip()]
            if isinstance(records, dict):
                records = [records]
            cache[relative] = {r["question_id"]:r for r in records}
            files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        record = cache[relative][saved["question_id"]]
        recalculated = classify_record(record, questions[saved["question_id"]])
        changed = [k for k,v in recalculated.items() if saved[k] != v]
        if changed:
            differences.append({"model":saved["model"],"question_id":saved["question_id"],"fields":changed})
        assert not is_api_error(record), (relative,saved["question_id"])
        rows.append({"model": saved["model"], "language": saved["language"], "prompt": saved["prompt_type"],
                     "question_id": saved["question_id"], "source_path": relative,
                     "source_record_sha256": digest(record), "source_file_sha256": files[relative],
                     "fixed_score": saved["outcome"], "benign_success": saved["benign_success"]})
    assert not differences, differences[:10]
    index = ROOT / "data/corpus/benign_response_index.jsonl"
    index.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    report = {"fixed_scores_unchanged": True, "raw_source_files":len(files), "observations":len(rows),
              "outcomes":dict(Counter(r["fixed_score"] for r in rows)), "all_scores_reproduced":True,
              "api_failures":0, "fixed_score_sha256":hashlib.sha256(scored_path.read_bytes()).hexdigest(),
              "source_files":files, "portable_index":str(index.relative_to(ROOT))}
    (ROOT / "analysis/verification/benign_source_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "source_files"}))


if __name__ == "__main__":
    main()
