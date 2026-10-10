#!/usr/bin/env python3
"""Build explicit coverage of the declared 30-model design, without API calls."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.canonical_panel import iter_cases, compact, release_models
from scripts.build_paper_results import table, number, pvalue
from scripts.build_supplement import tex
from scripts.unpack_manuscript import PAPER, ensure_manuscript


def coverage_status(status):
    if status in {"original_valid_input", "corrected_run", "recovered_api_error"}:
        return "completed"
    if status == "api_error":
        return "error"
    if status == "missing_corrected_run":
        return "not_executed"
    raise ValueError("Unknown coverage status: " + status)


def serialize_coverage(report):
    """Keep every field while avoiding a GitHub-oversized pretty-printed file."""
    header = {key: value for key, value in report.items() if key != "cells"}
    text = json.dumps(header, ensure_ascii=False, indent=2).rstrip()
    entries = [json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in report["cells"]]
    return text[:-1] + ',\n  "cells": [\n    ' + ',\n    '.join(entries) + '\n  ]\n}\n'


def main():
    ensure_manuscript()
    full, localized = defaultdict(Counter), defaultdict(Counter)
    cells = []
    for case in iter_cases("corrected"):
        row = compact(case)
        assert not row["needs_judgment"], "Judge every eligible response before freezing coverage."
        counts = {"planned": 1, coverage_status(case["status"]): 1}
        full[case["model"]].update(counts)
        if case["repaired"]:
            localized[case["model"]].update(counts)
        cells.append({k: row[k] for k in ["key", "model", "language", "prompt", "question_id", "scenario_id", "pressure", "status", "source_path", "record_sha256", "label", "label_source", "needs_judgment"]})
    assert set(full) == release_models() and all(r["planned"] == 6480 for r in full.values())
    assert all(r["planned"] == 144 for r in localized.values())
    totals = {k: sum(r[k] for r in full.values()) for k in ["planned", "completed", "error", "not_executed"]}
    localized_totals = {k: sum(r[k] for r in localized.values()) for k in totals}
    stats_path = ROOT / "analysis/revision/corrected_statistics.json"
    stats = json.loads(stats_path.read_text())
    assert totals["completed"] == stats["audit"]["n_cells"]
    assert totals["planned"] == 194400 == sum(totals[k] for k in ["completed", "error", "not_executed"])
    report = {"totals": localized_totals, "by_model": dict(sorted(localized.items())),
              "full_panel_coverage": {"totals": totals, "by_model": dict(sorted(full.items()))},
              "corrected_available_case_audit": stats["audit"], "missing_semantic_judgments": 0,
              "corrected_statistics_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
              "scope": "30 declared models; unknown outcomes are not refusals or safe actions.", "cells": cells}
    serialized = serialize_coverage(report)
    assert json.loads(serialized) == report, "Serialization must preserve every coverage field"
    assert len(serialized.encode()) < 100 * 1024 * 1024, "Use a different storage layout before publication"
    (ROOT / "analysis/revision/repair_coverage.json").write_text(serialized)
    text = (f"The design schedules {number(totals['planned'])} observations: {number(totals['completed'])} completed responses, "
            f"{totals['error']} errors, and {totals['not_executed']} cases with no available response. "
            "Errors and unavailable outcomes remain unknown, rather than being counted as refusals. "
            "Malformed outputs are not resampled until success. Every eligible substantive-call response has a semantic label. "
            "The record index links each scheduled cell to its input, outcome status, source, and content hash.\n\n")
    text += table("Coverage of all 108 scenarios, five languages, six pressure conditions and two prompts. Each model has 6,480 scheduled observations; completed responses enter the analysis.", "tab:evaluation-coverage", ["Model", "Completed", "Error", "Unavailable"], [[tex(m),number(r["completed"]),str(r["error"]),str(r["not_executed"])] for m,r in sorted(full.items())]+[["Total",number(totals["completed"]),str(totals["error"]),str(totals["not_executed"])]] ,size="scriptsize")
    common = json.loads((ROOT / "analysis/revision/common_statistics.json").read_text())
    assert common["audit"]["n_models"] == 30 and common["audit"]["n_scenarios"] == 102
    text += ("A supplementary comparison restricts all languages to 102 base scenarios whose inputs were unchanged across collection periods. "
             f"This subset contains {number(common['audit']['n_cells'])} observations. It checks sensitivity to input-version coverage; "
             "it does not isolate provider or time effects or validate translation equivalence.\n\n")
    text += table("Language sensitivity on the shared 102-scenario subset. Four language comparisons form one Holm family within each prompt.", "tab:extended-language", ["Prompt", "Language", r"Mean $\Delta$ (pp)", "$p$ raw", "$p$ Holm"], [[r["prompt"],r["language"].upper(),f"{r['mean_difference_pp']:+.2f}",pvalue(r["p_raw"]),pvalue(r["p_holm"])] for r in common["comparisons"]["harmful"]])
    (PAPER / "generated/evaluation_coverage.tex").write_text("% Generated offline from the complete design.\n"+text)
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
