# Reproducing Lumen-bench

The commands in this guide use released records, not new model inference. No API key is required. Dependencies in `requirements.txt` were tested with Python 3.13.

## Data and analysis populations

The full design contains 30 models × 108 scenarios × five languages × six pressure conditions × two system prompts, or 194,400 trials. There are 194,368 completed responses and 32 malformed tool-output errors. Errors remain in the records with unknown harmful-action outcomes and are never counted as refusals.

`analysis/revision/corrected_records.csv.gz` is the analysis-ready full-design export. The stable command-line identifier `--panel corrected` selects this population. The `--panel common` option selects a separate 102-scenario sensitivity subset with 183,600 completed responses; it is not the headline population.

Each row links its model, language, prompt, and question ID to a source path, response hash, and judgment provenance. Use this export rather than treating the original `eval_results_consolidated/` files alone as the analysis population: the export selects the appropriate input and response versions and joins labels to the exact responses they describe. Missing values in the harmful-action field are not zeros.

| Files | Purpose |
|---|---|
| `all_scenarios_reduced.py`, `questions_v3_*.json` | Source scenarios and 648 questions per language |
| `configs/evaluation_models.json`, `configs/models/` | Model roster and inference settings |
| `tool/` | Tool schemas and simulated evaluation-time feedback |
| `eval_results_consolidated/` | 300 original response JSON files |
| `data/language_repair/runs/`, `data/language_repair/judge_runs/` | Versioned responses and response-linked semantic judgments |
| `data/corpus/` | Parallel inputs and source indexes |
| `data/calibration/` | Human annotations and cross-judge evidence |
| `data/provenance/` | Source and serving-runtime evidence |
| `analysis/revision/` | Canonical exports, statistical analyses, controls, and deterministic sandbox replay |

The five Hugging Face viewer configurations under `data/tables/` are auxiliary tables, not the complete response panel. They contain parallel inputs, cross-judge labels, human calibration, historical judge labels, and benign controls. The viewer's `train` split is a storage label, not a recommendation to train models to comply with harmful requests. `croissant.json` describes these tables, and `configs/dataset_card.yaml` supplies the Hugging Face metadata.

## Offline workflow

Run from the repository root. The scripts preserve the distinction between unknown outcomes and non-violations and reject analyses with pending eligible semantic judgments.

```bash
pip install -r requirements.txt
python -m unittest discover -p 'test_*.py'
python scripts/build_parallel_corpus.py
python scripts/audit_benign_sources.py
python scripts/audit_historical_costs.py
python scripts/audit_recorded_errors.py
python scripts/audit_measurement_cohorts.py
python scripts/canonical_panel.py --panel corrected --allow-incomplete
python scripts/analyze_revision.py --panel corrected --available-case --regression
python scripts/replay_revision.py --panel corrected
python scripts/analyze_repaired_benign.py --available-case
```

`--allow-incomplete` retains all scheduled rows in the export, including the 32 unknown outcomes. `--available-case` excludes those unknown outcomes from the statistical analysis. Language tests retain a scenario/pressure pair only when both language outcomes are available for the same model and prompt. The four target-language comparisons form one Holm family within each prompt.

To reproduce the shared-input sensitivity check:

```bash
python scripts/canonical_panel.py --panel common
python scripts/analyze_revision.py --panel common --available-case --skip-influence
python scripts/replay_revision.py --panel common
```

The benign control covers 11 legitimate tasks across the same 30 models, five languages, and two prompts, for 3,300 pairs. There are 2,733 successful benign completions and 271 harmful actions within that subset. These pairs use corresponding tool operations but are not all strict lexical minimal pairs. Raw benign evidence is indexed by `data/corpus/benign_response_index.jsonl`.

`benign_scenarios.py` contains those 11 source tasks; `benign_control_questions/` contains the exact five-language evaluation questions and mechanical completion criteria. `eval_results_benign_control/` stores their recorded responses, and `benign_control_scored/` stores completion scores. These files describe the same experiment, not separate control sets.

## Paper sources

`release/lumen-bench-overleaf.zip` and `release/lumen-bench-arxiv.tar.gz` contain the complete manuscript sources. They are packaged once, without a duplicate source folder in the repository. Upload the ZIP to Overleaf and select XeLaTeX and TeX Live 2025, or unpack locally:

```bash
python scripts/unpack_manuscript.py
```

This creates a local, ignored `manuscript/` directory and verifies the source hashes in `release/build_manifest.json`. Existing source files are not overwritten. After the analysis workflow above, regenerate the tables and packages with:

```bash
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

The last command requires XeLaTeX and BibTeX, and produces the PDF and both source archives. It does not submit to arXiv.

## Measurement and scope

The metrics distinguish any tool call (`L0`), non-read-only calls (`L1`), calls with substantive arguments (`L2`), and judged harmful action (`L3`, the union of `COMPLY` and `SAY_DO_GAP`). Human calibration contains 83 Portuguese cases labeled by one native-Portuguese author under a model-blind protocol. Cross-judge evidence contains 607 unique responses labeled by o4-mini, Opus, and Gemini. The paper reports calibration uncertainty and its language-transfer assumptions.

Scenarios were authored in Portuguese and translated with Claude Sonnet 4.6. No native-speaker validation was performed for the other four languages. Automated language and coverage checks do not establish semantic equivalence. The evaluation uses synthetic, clearly harmful requests in single sessions, simulated tool feedback, and retrospective sandbox replay. It does not measure deployed-agent harm, persistent memory, planning, or ethically ambiguous requests.

## New inference and costs

Running `eval_v3.py` makes provider requests and can incur charges. Only for intentional new data collection, copy `.env.example` to a private `.env` and provide your own credentials. Never commit that file.

The reported evaluator uses temperature 0 and max_tokens 1024, except GPT-5-Mini and GPT-5.4, which use provider defaults; model reasoning effort is not explicitly set. The semantic judge uses medium reasoning. Provider availability and defaults may change, so preserve and inspect the actual request settings when reproducing the protocol.

The Qwen3-8B evidence includes 126 self-hosted sessions. `data/provenance/serving_qwen3_8b.json` records the pinned public checkpoint, BF16 weights, vLLM 0.17.1 runtime, tool and reasoning parsers, thinking-template checks, and request hashes. Matching request-level settings does not establish numerical equivalence with historical provider implementations.

Archived token counters and list prices are not provider invoices or a current cost quote. Reasoning and completion counters may overlap; nine source observations have zero input/output counters, which does not establish free inference. Per-request metadata distinguishes provider-reported charges from estimates where available.
