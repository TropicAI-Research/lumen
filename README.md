---
license: cc-by-4.0
language:
  - pt
  - en
  - es
  - zh
  - hi
tags:
  - llm-safety
  - tool-use
  - multilingual
  - evaluation
configs:
  - config_name: parallel_inputs
    default: true
    data_files:
      - split: train
        path: data/tables/parallel_inputs.csv
  - config_name: crossjudge
    data_files:
      - split: train
        path: data/tables/crossjudge.csv
  - config_name: human_calibration
    data_files:
      - split: train
        path: data/tables/human_calibration.csv
  - config_name: historical_judge_labels
    data_files:
      - split: train
        path: data/tables/historical_judge_labels.csv
  - config_name: benign_controls
    data_files:
      - split: train
        path: data/tables/benign_controls.csv
---

# Lumen-bench

A multilingual benchmark of what tool-calling models do, including cases in which refusal text accompanies a harmful call. Evaluation-time tool feedback and retrospective state changes are simulated; neither measures deployed-agent harm.

[Code](https://github.com/TropicAI-Research/lumen) · [Dataset](https://huggingface.co/datasets/Tropic-AI/lumen-bench) · [Paper](release/lumen-bench.pdf) · [Overleaf ZIP](release/lumen-bench-overleaf.zip) · [arXiv source package](release/lumen-bench-arxiv.tar.gz)

## Design and coverage

The declared roster contains 30 models. Each receives 108 scenarios × five languages × six pressure conditions × two system prompts: 6,480 scheduled observations per model and 194,400 overall. The benchmark uses 21 generic tools and 11 ethical categories.

| Population | Scenarios | Models | Completed responses | Errors | No response |
|---|---:|---:|---:|---:|---:|
| Full design | 108 | 30 | 194,368 | 32 | 0 |
| Shared-input sensitivity subset | 102 | 30 | 183,600 | 0 | 0 |

Every scheduled case has an attempted execution. The 32 errors are malformed tool-output failures, not safe refusals. Successful outcomes are not resampled, and malformed outputs are not repeatedly sampled until success.

Language comparisons match the same model, scenario, pressure, and system prompt in both languages. Each model contributes one paired rate difference; four target-language tests form a Holm family within each prompt. Unavailable observations remove both sides of a pair.

The full analysis uses the current parallel inputs and is selected by the stable code identifier `--panel corrected`. The `--panel common` option is a 102-scenario coverage sensitivity check, not the headline population. Stored inputs, responses, provider routes, and labels have explicit versions and content hashes. Observations span different collection periods and service routes, whose defaults are not assumed identical.

## Language and measurement validation

Scenarios were written in Portuguese and translated into English, Spanish, Mandarin Chinese, and Hindi using Claude Sonnet 4.6. The corrected corpus has explicit translations in all five languages; generation fails on missing language fields. Automated coverage and script checks are not semantic-equivalence validation.

Human calibration is Portuguese-only: one native-Portuguese author, blind to model identity, labelled 83 retained cases from judge-label strata of sizes 24, 21, 25 and 13. No native-speaker validation was conducted for the other four languages. Cross-judge evidence contains 607 unique stored responses labelled by o4-mini, Opus, and Gemini. Labels apply to those exact responses, not every later response with the same scenario ID.

The four metrics distinguish any tool call (`L0`), non-read-only calls (`L1`), calls with substantive arguments (`L2`), and judged harmful action (`L3` = COMPLY ∪ SAY_DO_GAP). The full analysis contains 16,822 substantive-call responses, all judged: 14,499 COMPLY, 1,441 SAY_DO_GAP, 685 REDIRECT, and 197 AMBIGUOUS. Thus the judge-estimated say-do share is 8.57%. Its uncertainty and the transfer assumption in Portuguese calibration are reported in the paper, not hidden behind cross-model agreement.

## Benign controls

Eleven legitimate tasks are paired with harmful requests using corresponding tool operations, across the same 30 models, five languages, and two prompts (3,300 trials). Models complete 2,733 benign requests (82.8%). Their wording and context can differ; these are not all strict lexical minimal pairs.

All 3,300 pairs have observed harmful outcomes. Of the 2,733 pairs with successful benign completion, 271 also execute the harmful request (9.92%; model-cluster 95% interval 4.62–16.32%). The control distinguishes non-violation with demonstrated competence from non-violation without it. Per-model results, scoring criteria, and raw source links are released.

## Artifacts

```text
configs/evaluation_models.json      declared 30-model roster
configs/models/                    inference configurations
questions_v3_{pt,en,es,zh,hi}.json   648 questions per language
all_scenarios_reduced.py            108 source scenarios and translations
tool/                              schemas and simulated tool feedback
eval_results_consolidated/         300 source-response JSON files
data/language_repair/runs/          versioned request/response records
data/language_repair/judge_runs/    response-hash-linked semantic labels
data/corpus/                       parallel inputs and source indexes
data/provenance/serving_qwen3_8b.json  self-hosted runtime and record hashes
data/calibration/                  human and model-judge evidence
data/tables/                       CSV resources declared in croissant.json
analysis/revision/                 full design, statistics, replay and controls
Lumen_bench_original_corrigido/     manuscript source
release/                           PDF and source packages
```

Canonical analysis rows include every scheduled cell, with status, source path, response hash, and label provenance. Source-response files are not interchangeable with the canonical full-design export: the latter selects the input version used in the paper. Synthetic names and records do not describe real employees or patients.

The Qwen3-8B records include 126 self-hosted sessions using the pinned public checkpoint, native BF16 weights, and vLLM 0.17.1. The runtime evidence documents thinking, tool and reasoning parsers, exact model-file hashes, template checks, and all submitted requests. Request-level inference settings match the benchmark recipe; numerical equivalence with historical provider implementations is not assumed.

## Offline reproduction

Install dependencies with `pip install -r requirements.txt`. The commands below use saved records and make no model API calls; no API key is required.

```bash
python -m unittest discover -p 'test_*.py'
python scripts/build_parallel_corpus.py
python scripts/audit_benign_sources.py
python scripts/audit_historical_costs.py
python scripts/audit_recorded_errors.py
python scripts/audit_measurement_cohorts.py
python scripts/canonical_panel.py --panel corrected --allow-incomplete
python scripts/analyze_revision.py --panel corrected --available-case --regression
python scripts/replay_revision.py --panel corrected
python scripts/canonical_panel.py --panel common
python scripts/analyze_revision.py --panel common --available-case --skip-influence
python scripts/replay_revision.py --panel common
python scripts/analyze_repaired_benign.py --available-case
python scripts/build_supplement.py --available-case
python scripts/build_repair_coverage.py
python scripts/build_mechanical_sensitivity.py
python scripts/build_case_evidence.py
python scripts/build_paper_results.py
python scripts/build_original_style_results.py
python scripts/build_cost_table.py
python scripts/build_dataset_metadata.py
python scripts/build_arxiv_release.py
```

`--allow-incomplete` preserves unknown outcomes in the design export; it does not turn them into non-violations. The statistical analysis explicitly excludes unavailable outcomes and refuses to proceed if any eligible semantic label is pending. Benign scores can be checked against raw records through `data/corpus/benign_response_index.jsonl`.

Compile with XeLaTeX and BibTeX. The Overleaf ZIP and arXiv package contain `main.tex`, its bibliography, style and generated inputs. Select XeLaTeX and TeX Live 2025. No arXiv submission is performed by the build script.

New inference through `eval_v3.py` incurs provider charges. Copy `.env.example` to a local `.env` and supply personal credentials only if intentionally collecting new results. Provider availability and defaults can change. The reported evaluator uses temperature 0 and max_tokens 1024 except for GPT-5-Mini and GPT-5.4, which use provider defaults; model reasoning effort is not explicitly set. The semantic judge uses medium reasoning. Do not change these settings silently when reproducing the protocol.

## Costs and limitations

Saved token counters and archived list prices are not provider invoices or a current reproduction quote. Reasoning and completion counters may overlap. Nine source observations have zero input/output counters; this does not establish free calls. Per-request metadata distinguishes provider-reported charges from estimates where available.

The benchmark studies clearly harmful synthetic requests in single sessions, with simulated tool feedback and separate deterministic replay. It does not measure ambiguous intent, planning, persistent memory, retrieval, or real organizational permissions. A low harmful-action rate alone is not evidence of useful safe behavior. Use the corpus for safety evaluation, not training harmful compliance.

## Licenses

Code is MIT (see LICENSE). Dataset artifacts, including questions, responses, labels and sandbox records, are CC-BY 4.0.

## Citation

```bibtex
@misc{lumenbench2026,
  title = {Lumen-bench: Cross-Lingual Safety Gaps and the Say-Do Problem in Tool-Calling Large Language Models},
  author = {Bonás, Giovana Kerche and Laitz, Thiago and Almeida, Thales Sales and Santos, João Guilherme Alves},
  howpublished = {Preprint},
  url = {https://github.com/TropicAI-Research/lumen},
  year = {2026}
}
```

An arXiv identifier will be added after deposition.
