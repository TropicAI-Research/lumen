# Lumen-bench Explorer

A static, read-only interface for the benchmark, using TropicAI's organization artwork and coral/teal palette. All interface text is English; requests and responses retain their recorded languages. The explorer makes no model API calls, has no backend, and needs no credentials in the browser.

## What you can inspect

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
python -m http.server 8768 --bind 127.0.0.1 --directory site
```

Replace `DATASET_COMMIT` with the dataset's immutable 40-character commit SHA. Open `http://127.0.0.1:8768`. Use a current browser with the standard `DecompressionStream` API. Opening `index.html` directly with `file://` will not work, because the browser must fetch local data files over HTTP.

The builder is offline and deterministic. It reads the canonical response selector, validates source hashes, and produces a compressed index plus lazily loaded response shards under `site/data/`. Generated data is about 78 MB; a visit downloads only the index and the response shards it needs, not the full underlying dataset. No third-party fonts, analytics, or external JavaScript are used.

Before changing the input data, publish and verify an immutable dataset revision, then pass that commit to `--dataset-revision`. Use the same revision for the paper link in `site/index.html`. The public source files must match the manifest's hashes. Do not point production data links to mutable `main` or include credentials in `site/`.

## Publish on GitHub Pages

The deployment branch is `gh-pages` in `TropicAI-Research/lumen`; its root contains only the contents of `site/`, not the full benchmark repository. `.nojekyll` disables Jekyll processing. This keeps the published site below GitHub Pages' size limit.

In repository **Settings → Pages**, select **Deploy from a branch**, choose **gh-pages** and **/ (root)**, then save. When GitHub finishes deployment, the expected URL is:

`https://tropicai-research.github.io/lumen/`

Repository visibility and organization billing settings remain under the owner's control. GitHub Free organizations require a public repository for Pages; do not upgrade a plan or change visibility as part of a build. Configuring Pages through a fine-grained personal token requires the repository's Pages read/write permission, separately from Contents write permission.

GitHub's instructions: [configure a publishing source](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site), [Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits).
