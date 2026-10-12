# Lumen-bench Explorer

A static, read-only interface for the benchmark, using TropicAI's organization artwork and coral/teal palette. All interface text is English; requests and responses retain their recorded languages. The explorer makes no model API calls, has no backend, and needs no credentials in the browser.

## Interactive exploration

- **Playground:** choose a prompt language on the world map, then compare two recorded responses while changing language, model, social pressure, or system instruction. The other displayed settings and scenario stay matched. Open the exact requests, proposed calls, and original evidence. Map locations are navigation anchors for languages; the benchmark does not measure country-level outcomes.
- **Condition matrix:** click any of the five-language by six-pressure cells to inspect the recorded outcome for the selected model, scenario, and instruction. There is one response per cell.
- **Model × language atlas:** explore all 150 model-language cells. Filter scenarios and pressure, switch between harmful-action and say–do rates, and compare ethical and neutral instructions. Each selection exposes its numerator, denominator, and output errors. Counts are descriptive; no inferential test is recalculated.
- **Predict the action:** read a model's prose before revealing its proposed tool calls and stored outcome. Examples are selected for exploration and do not form a representative sample.

Every view can be shared through its URL fragment. Guided examples are selected from the released observations. Neither responses nor labels are generated in the browser.

## What you can inspect

The **Results** view presents the paper's findings before the individual records. It includes harmful-action rates under both prompts for all 30 models, the say-do share with its substantive-call denominator, descriptive pressure-condition results, and the eight within-model language comparisons. The **Benign-control results** section reproduces Table 26 for all 30 models, with completion counts, harmful actions before and after conditioning on benign completion, the published model-cluster confidence interval, the four paired outcomes, and summaries by language and prompt. These are the released results, not new model evaluations. Tables export as CSV; model names and paired-outcome counts open the underlying records.

Direct links: [results](https://tropicai-research.github.io/lumen/#results), [benign-control results](https://tropicai-research.github.io/lumen/#results?section=benign), and [paired language comparisons](https://tropicai-research.github.io/lumen/#results?section=languages).

The response browser covers all 194,400 trials, including the 32 recorded output errors. Filters select model, language, system prompt, scenario category, pressure condition, outcome, tool, and additional evidence. Open a record to inspect the request, final assistant text, tool arguments, judge rationale, sandbox replay when available, and the recorded message trace. Every record links to its source at an immutable dataset revision and displays its SHA-256.

Separate views expose the parallel scenario corpus, the 607-response three-judge sample, the 83 Portuguese human annotations, and the 3,300 benign controls with their mechanical completion criteria and harmful counterparts. Ten three-judge cases use a different response version from the analysis panel; the browser preserves their original evidence and never transfers those labels to the current response.

Filtering is descriptive. It does not recompute hypothesis tests or turn the results into model safety rankings. “No substantive action” is not synonymous with refusal or competence. The methods view explains these distinctions and links to the reproduction artifacts.

## Build and preview locally

From the released repository:

```bash
pip install -r requirements.txt
python scripts/build_explorer.py --dataset-revision DATASET_COMMIT
python -m unittest test_explorer
node site/test-core.mjs
node site/test-lab.mjs
python -m http.server 8768 --bind 127.0.0.1 --directory site
```

Replace `DATASET_COMMIT` with the dataset's immutable 40-character commit SHA. Open `http://127.0.0.1:8768`. Use a current browser with the standard `DecompressionStream` API. Opening `index.html` directly with `file://` will not work, because the browser must fetch local data files over HTTP.

The builder is offline and deterministic. It reads the canonical response selector, validates source hashes, and produces a compressed index plus lazily loaded response shards under `site/data/`. Generated data is about 78 MB; a visit downloads only the index and the response shards it needs, not the full underlying dataset. No third-party fonts, analytics, or external JavaScript are used. The world map geometry is included in `site/world.js`; no map service is contacted.

The interaction code is in `site/lab.js` and `site/lab.css`. Run `python scripts/check_lab_browser.py --url http://127.0.0.1:8768/ --output /tmp/lumen-browser-checks` with Playwright and Chrome installed to verify the map, comparisons, atlas, prediction exercise, and mobile layouts.

Before changing the input data, publish and verify an immutable dataset revision, then pass that commit to `--dataset-revision`. Use the same revision for the paper link in `site/index.html`. The public source files must match the manifest's hashes. Do not point production data links to mutable `main` or include credentials in `site/`.

## Publish on GitHub Pages

The deployment branch is `gh-pages` in `TropicAI-Research/lumen`; its root contains only the contents of `site/`, not the full benchmark repository. `.nojekyll` disables Jekyll processing. This keeps the published site below GitHub Pages' size limit.

In repository **Settings → Pages**, select **Deploy from a branch**, choose **gh-pages** and **/ (root)**, then save. When GitHub finishes deployment, the expected URL is:

`https://tropicai-research.github.io/lumen/`

Repository visibility and organization billing settings remain under the owner's control. GitHub Free organizations require a public repository for Pages; do not upgrade a plan or change visibility as part of a build. Configuring Pages through a fine-grained personal token requires the repository's Pages read/write permission, separately from Contents write permission.

GitHub's instructions: [configure a publishing source](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site), [Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits).
