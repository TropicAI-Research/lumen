#!/usr/bin/env python3
"""
Build the 75 new cases for human annotation, stratified by judge label.

Combined with the existing 25 in human_ratings_corrected.csv, this brings
the calibration set to ~25 per judge-label class (target: 25/25/25/25).

Run:
    python3 annotation/build_cases.py

Output:
    annotation/cases_to_label.json   — list of cases for the HTML annotator
    annotation/cases_meta.json       — sampling metadata (judge label hidden from annotator)
"""
import json, os, random, glob
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
random.seed(SEED)

# Fresh-start design: 100 cases, 25 per judge label, single consistent protocol.
# The original 25-case set is preserved on disk for provenance but not used in
# the new calibration analysis.
TARGET_PER_LABEL = 25
NEW_PER_LABEL = {l: TARGET_PER_LABEL for l in ['COMPLY', 'SAY_DO_GAP', 'REDIRECT', 'AMBIGUOUS']}

print(f"Cases per class (target): {NEW_PER_LABEL}")
print(f"Total: {sum(NEW_PER_LABEL.values())}\n")

# ----- 1. Load primary-judge labels -----
jc = json.load(open(ROOT / 'judge_combined_parsed.json'))
print(f"Total L2-positive cases (all langs): {len(jc)}")

# Filter to PT
pt_cases = {k: v for k, v in jc.items() if v.get('lang') == 'pt'}
print(f"PT L2-positive cases: {len(pt_cases)}")
print(f"PT label distribution: {Counter(v['label'] for v in pt_cases.values())}\n")

# ----- 2. (legacy 25-case set was retired; sampling pool is the full PT L2-positive set) -----
available = pt_cases
print(f"PT cases available for sampling: {len(available)}")
print(f"Available label distribution: {Counter(v['label'] for v in available.values())}\n")

# ----- 3. Stratified sampling -----
by_label = defaultdict(list)
for k, v in available.items():
    by_label[v['label']].append(k)

sampled_keys = []
for label, n_needed in NEW_PER_LABEL.items():
    pool = by_label.get(label, [])
    if len(pool) < n_needed:
        raise RuntimeError(f"Not enough {label} cases: need {n_needed}, have {len(pool)}")
    picked = random.sample(pool, n_needed)
    sampled_keys.extend(picked)
    print(f"  {label}: sampled {n_needed} of {len(pool)}")

print(f"\nTotal sampled: {len(sampled_keys)}\n")

# ----- 4. Fetch raw case data from eval_results/pt/<prompt>/<model>/results_*.json -----
# Cache: load each results file once
results_cache = {}
def load_results(model, prompt):
    cache_key = (model, prompt)
    if cache_key in results_cache:
        return results_cache[cache_key]
    pattern = ROOT / 'eval_results' / 'pt' / prompt / model / 'results_*.json'
    matches = glob.glob(str(pattern))
    if not matches:
        results_cache[cache_key] = None
        return None
    matches.sort()
    by_qid = {}
    for path in matches:  # merge multiple runs (some models have split runs)
        for r in json.load(open(path)):
            by_qid[r['question_id']] = r
    # Fallback: also try incremental_*.jsonl for models run via batch API
    if not by_qid:
        for path in glob.glob(str(ROOT / 'eval_results' / 'pt' / prompt / model / 'incremental_*.jsonl')):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        by_qid[r['question_id']] = r
    results_cache[cache_key] = by_qid
    return by_qid

cases_for_annotator = []
cases_meta = []
missing = []
for k in sampled_keys:
    rec = jc[k]
    model = rec['model']
    prompt = rec['prompt']
    qid = rec['question_id']
    results = load_results(model, prompt)
    if results is None or qid not in results:
        missing.append(k)
        continue
    raw = results[qid]
    # Build a clean record for the annotator (without the judge label)
    annot_record = {
        'case_id': k,
        'category': rec['category'],
        'pressure_type': rec['pressure_type'],
        'prompt_type': prompt,
        # model_name intentionally omitted from the annotator file (blind review)
        'question_text': raw.get('question_text', raw.get('conversation', [{}, {}])[1].get('content', '')),
        'assistant_response': raw.get('assistant_response') or '',
        'tool_calls': [],
    }
    # Extract tool calls from conversation
    for msg in raw.get('conversation', []):
        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            for tc in msg['tool_calls']:
                fn = tc.get('function', {})
                args_raw = fn.get('arguments', '{}')
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except Exception:
                    args = args_raw
                annot_record['tool_calls'].append({
                    'name': fn.get('name', ''),
                    'arguments': args,
                })
    cases_for_annotator.append(annot_record)
    # Meta record holds the hidden judge label + reasoning
    cases_meta.append({
        'case_id': k,
        'judge_label': rec['label'],
        'judge_reason': rec.get('reason', ''),
    })

print(f"Built {len(cases_for_annotator)} annotator records ({len(missing)} missing raw data)")
if missing:
    print("Missing examples:", missing[:5])

# ----- 5. Save -----
out_dir = ROOT / 'annotation'
random.Random(SEED).shuffle(cases_for_annotator)  # randomize annotation order
# Reorder meta to match shuffled annotator order
order_map = {c['case_id']: i for i, c in enumerate(cases_for_annotator)}
cases_meta.sort(key=lambda m: order_map[m['case_id']])

with open(out_dir / 'cases_to_label.json', 'w') as f:
    json.dump(cases_for_annotator, f, ensure_ascii=False, indent=2)
with open(out_dir / 'cases_meta.json', 'w') as f:
    json.dump(cases_meta, f, ensure_ascii=False, indent=2)

print(f"\nWritten:")
print(f"  {out_dir / 'cases_to_label.json'} ({os.path.getsize(out_dir / 'cases_to_label.json'):,} bytes)")
print(f"  {out_dir / 'cases_meta.json'}     ({os.path.getsize(out_dir / 'cases_meta.json'):,} bytes)")
