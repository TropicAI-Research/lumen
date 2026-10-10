# Lumen-bench

A multilingual benchmark for evaluating what tool-calling language models **do**, not just what they say. It measures harmful tool use, including cases where a model refuses in words but still performs the requested action.

[Paper](https://huggingface.co/datasets/Tropic-AI/lumen-bench/resolve/main/release/lumen-bench.pdf) · [Dataset](https://huggingface.co/datasets/Tropic-AI/lumen-bench) · [Code](https://github.com/TropicAI-Research/lumen)

The benchmark evaluates **30 models** on **108 scenarios**, in Portuguese, English, Spanish, Mandarin Chinese, and Hindi, with six pressure conditions, two system prompts, and 21 tools: **194,400 trials**. It also includes 3,300 benign-control trials.

## Repository structure

| Location | Contents |
|---|---|
| `questions_v3_*.json`, `tool/`, `configs/` | Evaluation questions, tools, and model settings |
| `eval_results_consolidated/`, `eval_results_benign_control/` | Recorded model responses |
| `benign_control_questions/`, `benign_control_scored/` | The 11 benign tasks and completion scores for all 3,300 trials |
| `data/` | Parallel corpus, response records, human calibration, and judge labels |
| `analysis/revision/` | Analysis-ready records, statistics, benign controls, and sandbox results |
| `scripts/` | Data preparation and reproduction scripts |
| `release/` | Paper PDF, Overleaf ZIP, and arXiv source package |
| `site/` (GitHub) | Interactive explorer: responses, judges, scenarios, and benign controls |

For analysis, start with [`corrected_records.csv.gz`](https://huggingface.co/datasets/Tropic-AI/lumen-bench/resolve/main/analysis/revision/corrected_records.csv.gz): it links each trial to its input, response, and judgment. It retains all 194,400 trials; 32 malformed tool-output errors are excluded from harmful-action rates, not counted as refusals.

## Reproduce the results

```bash
pip install -r requirements.txt
python scripts/analyze_revision.py --panel corrected --available-case --regression
```

This uses saved records, requires no API key, and makes no model API calls. See the [reproduction guide](docs/reproduction.md) for the full workflow, data fields, and paper compilation.

The [explorer guide](docs/explorer.md) explains local preview and GitHub Pages deployment.

All scenarios and tool effects are synthetic. Human calibration was conducted in Portuguese only; the benchmark does not establish safety in deployed agents.

## License and citation

Code: [MIT](LICENSE). Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Giovana Kerche Bonás, Thiago Laitz, Thales Sales Almeida, and João Guilherme Alves Santos. *Lumen-bench: Cross-Lingual Safety Gaps and the Say-Do Problem in Tool-Calling Large Language Models.* 2026.
