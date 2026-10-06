#!/usr/bin/env python3
"""Deep statistical analysis: regression, bootstrap CIs, SAY_DO_GAP breakdowns."""
import json, glob, sys, os, random
from collections import defaultdict, Counter
sys.path.insert(0,'.')
from metrics import is_l2_hard_positive, extract_tool_calls, READ_ONLY_TOOLS as READ_ONLY

EXCLUDED = {'sabia31','sabiazinho-3'}
OUT_DIR = 'analysis'
os.makedirs(OUT_DIR, exist_ok=True)

parsed = json.load(open('judge_combined_parsed.json'))
# Reverse lookup (lang, model, prompt, question_id) -> custom_id, built from parsed entries
rev = {(v['lang'], v['model'], v['prompt'], v['question_id']): k for k, v in parsed.items()}

def load_results(path):
    """Load results from JSON list or JSONL (batch) format."""
    if path.endswith('.jsonl'):
        out = []
        for line in open(path):
            line = line.strip()
            if not line: continue
            try: out.append(json.loads(line))
            except: pass
        return out
    return json.load(open(path))

# Per-case records: one row per (question, model) with all features + outcome.
# We read from eval_results_consolidated/ where each cell has been deduplicated
# by question_id (latest non-error attempt wins). See consolidate_eval_results.py
# for the consolidation logic. If consolidated dir is missing, fall back to raw.
records = []
paths = []
if os.path.isdir('eval_results_consolidated'):
    paths = sorted(glob.glob('eval_results_consolidated/*/*/*/consolidated.json'))
else:
    seen_dirs = set()
    for path in sorted(glob.glob('eval_results/*/*/*/results_*.json')):
        d = '/'.join(path.split('/')[:-1])
        seen_dirs.add(d); paths.append(path)
    for path in sorted(glob.glob('eval_results/*/*/*/incremental_batch*.jsonl')):
        d = '/'.join(path.split('/')[:-1])
        if d not in seen_dirs:
            paths.append(path)

for path in paths:
    parts = path.split('/')
    lang, prompt, model = parts[-4], parts[-3], parts[-2]
    if model in EXCLUDED: continue
    try:
        results = load_results(path)
    except: continue
    for r in results:
        if r.get('error'): continue
        calls = extract_tool_calls(r)
        l0 = len(calls) > 0
        l1 = any(c['name'] not in READ_ONLY for c in calls)
        cid = rev.get((lang, model, prompt, r.get('question_id','')))
        # L2 ground truth: a case was L2-positive iff it was submitted to the judge
        l2 = cid is not None
        judge_label = parsed.get(cid, {}).get('label') if cid else None
        records.append({
            'lang': lang, 'prompt': prompt, 'model': model,
            'category': r.get('category'),
            'pressure_type': r.get('pressure_type'),
            'explicitness': r.get('explicitness'),
            'moral_clarity': r.get('moral_clarity'),
            'question_id': r.get('question_id'),
            'L0': int(l0), 'L1': int(l1), 'L2': int(l2),
            'judge_label': judge_label or 'REFUSAL',  # L2-negative = refused (no action)
            'L3_behavioral': int(judge_label in {'COMPLY','SAY_DO_GAP'}),
            'L3_strict': int(judge_label == 'COMPLY'),
            'SAY_DO_GAP': int(judge_label == 'SAY_DO_GAP'),
            'REDIRECT': int(judge_label == 'REDIRECT'),
            'AMBIGUOUS': int(judge_label == 'AMBIGUOUS'),
        })
print(f"Total records: {len(records)}")

# Save records for later use
with open(f'{OUT_DIR}/records.json','w') as f:
    json.dump(records, f)

# === 1. Bootstrap CIs on overall violation rates ===
def bootstrap_ci(values, n_boot=1000, ci=0.95):
    if not values: return (0, 0, 0)
    n = len(values)
    mean = sum(values)/n
    samples = []
    for _ in range(n_boot):
        s = [values[random.randrange(n)] for _ in range(n)]
        samples.append(sum(s)/n)
    samples.sort()
    lo = samples[int(n_boot*(1-ci)/2)]
    hi = samples[int(n_boot*(1+ci)/2)]
    return mean, lo, hi

random.seed(42)
print("\n=== Table: violation rate by language × prompt × metric (with 95% bootstrap CIs) ===")
print(f"{'lang':3s} {'prompt':8s} {'n':>6s}  {'L0 (95% CI)':>22s}  {'L1':>22s}  {'L3_beh':>22s}  {'L3_str':>22s}")
results_table = []
for lang in ['en','pt','es','zh','hi']:
    for prompt in ['ethical','neutral']:
        sub = [r for r in records if r['lang']==lang and r['prompt']==prompt]
        if not sub: continue
        n = len(sub)
        l0_m, l0_lo, l0_hi = bootstrap_ci([r['L0'] for r in sub])
        l1_m, l1_lo, l1_hi = bootstrap_ci([r['L1'] for r in sub])
        l3b_m, l3b_lo, l3b_hi = bootstrap_ci([r['L3_behavioral'] for r in sub])
        l3s_m, l3s_lo, l3s_hi = bootstrap_ci([r['L3_strict'] for r in sub])
        print(f"{lang:3s} {prompt:8s} {n:>6d}  {l0_m*100:5.2f} [{l0_lo*100:4.2f},{l0_hi*100:4.2f}]   {l1_m*100:5.2f} [{l1_lo*100:4.2f},{l1_hi*100:4.2f}]   {l3b_m*100:5.2f} [{l3b_lo*100:4.2f},{l3b_hi*100:4.2f}]   {l3s_m*100:5.2f} [{l3s_lo*100:4.2f},{l3s_hi*100:4.2f}]")
        results_table.append({
            'lang':lang,'prompt':prompt,'n':n,
            'L0':{'mean':l0_m,'lo':l0_lo,'hi':l0_hi},
            'L1':{'mean':l1_m,'lo':l1_lo,'hi':l1_hi},
            'L3_behavioral':{'mean':l3b_m,'lo':l3b_lo,'hi':l3b_hi},
            'L3_strict':{'mean':l3s_m,'lo':l3s_lo,'hi':l3s_hi},
        })
with open(f'{OUT_DIR}/violation_rate_table.json','w') as f:
    json.dump(results_table, f, indent=2)

# === 2. Wilcoxon signed-rank test: HI vs EN pairs by model ===
from itertools import groupby
print("\n=== Cross-lingual test: HI vs EN (paired by model, ethical prompt) ===")
def get_rate(lang, prompt, model, outcome):
    sub = [r for r in records if r['lang']==lang and r['prompt']==prompt and r['model']==model]
    if not sub: return None
    return sum(r[outcome] for r in sub) / len(sub)

models_available = sorted(set(r['model'] for r in records))
for outcome in ['L0','L3_behavioral','L3_strict']:
    diffs = []
    for m in models_available:
        en = get_rate('en','ethical',m,outcome)
        hi = get_rate('hi','ethical',m,outcome)
        if en is not None and hi is not None:
            diffs.append((m, en, hi, hi-en))
    # Wilcoxon sign test (manual, no scipy dep needed)
    pos = sum(1 for _,_,_,d in diffs if d > 0)
    neg = sum(1 for _,_,_,d in diffs if d < 0)
    ties = sum(1 for _,_,_,d in diffs if d == 0)
    mean_diff = sum(d for _,_,_,d in diffs)/len(diffs) if diffs else 0
    print(f"  {outcome}: n={len(diffs)}, HI>EN={pos}, HI<EN={neg}, ties={ties}, mean(HI-EN)={mean_diff*100:.2f}pp")

# === 3. SAY_DO_GAP: by model family and language ===
print("\n=== SAY_DO_GAP by model (all prompts, L2-positive denominator) ===")
model_say = defaultdict(lambda: [0,0])
for r in records:
    if r['L2']:
        model_say[r['model']][0] += r['SAY_DO_GAP']
        model_say[r['model']][1] += 1
rows = [(m, sdg, t, sdg/t) for m,(sdg,t) in model_say.items() if t>=20]
for m,sdg,t,rate in sorted(rows, key=lambda x:-x[3])[:15]:
    print(f"  {m:30s}  {sdg:4d}/{t:4d} = {rate*100:5.1f}%")

# === 4. Category × pressure cross-tab for SAY_DO_GAP ===
print("\n=== SAY_DO_GAP rate by category (L2-positive denominator) ===")
cat_say = defaultdict(lambda: [0,0])
for r in records:
    if r['L2']:
        cat_say[r['category']][0] += r['SAY_DO_GAP']
        cat_say[r['category']][1] += 1
for c,(sdg,t) in sorted(cat_say.items(), key=lambda x:-x[1][0]/(x[1][1] or 1)):
    if t: print(f"  {c:30s}  {sdg:4d}/{t:4d} = {sdg/t*100:5.1f}%")

# === 5. Pressure type effect on L3_behavioral ===
print("\n=== L3_behavioral rate by pressure type (neutral prompt) ===")
press_eff = defaultdict(lambda: [0,0])
for r in records:
    if r['prompt']=='neutral':
        press_eff[r['pressure_type']][0] += r['L3_behavioral']
        press_eff[r['pressure_type']][1] += 1
for p,(v,t) in sorted(press_eff.items(), key=lambda x:-x[1][0]/(x[1][1] or 1)):
    if t: print(f"  {p:15s}  {v}/{t} = {v/t*100:5.2f}%")

# === 6. Logistic regression via pure Python (no scipy required) ===
# Fit L3_behavioral ~ lang + prompt + category + pressure + model, report OR
# Use simple coordinate descent on categorical dummies.
print("\n=== Logistic regression (simple) on L3_behavioral ===")
# Build feature matrix: one-hot
import math
def fit_logistic(X, y, lr=0.5, n_iter=200, reg=0.01):
    n = len(y); d = len(X[0])
    w = [0.0]*d
    for it in range(n_iter):
        # gradient
        grad = [0.0]*d
        for i in range(n):
            z = sum(w[j]*X[i][j] for j in range(d))
            p = 1/(1+math.exp(-z)) if z > -30 else 0.0
            err = p - y[i]
            for j in range(d):
                grad[j] += err * X[i][j]
        for j in range(d):
            w[j] -= lr * (grad[j]/n + reg*w[j])
    return w

langs_l = ['pt','es','zh','hi']  # en reference
prompts_l = ['neutral']  # ethical reference
cats_sorted = sorted(set(r['category'] for r in records if r['category']))[1:]  # first as ref
press_l = sorted(set(r['pressure_type'] for r in records if r['pressure_type']))[1:]  # first as ref

# Build features
feat_names = ['intercept'] + [f'lang_{l}' for l in langs_l] + [f'prompt_{p}' for p in prompts_l] + \
             [f'cat_{c}' for c in cats_sorted] + [f'press_{p}' for p in press_l]
X, y = [], []
for r in records:
    if r['category'] is None or r['pressure_type'] is None: continue
    row = [1.0]  # intercept
    for l in langs_l: row.append(1.0 if r['lang']==l else 0.0)
    for p in prompts_l: row.append(1.0 if r['prompt']==p else 0.0)
    for c in cats_sorted: row.append(1.0 if r['category']==c else 0.0)
    for p in press_l: row.append(1.0 if r['pressure_type']==p else 0.0)
    X.append(row)
    y.append(r['L3_behavioral'])
print(f"  n={len(y)}  features={len(feat_names)}  positive rate={sum(y)/len(y)*100:.2f}%")
w = fit_logistic(X, y, lr=0.3, n_iter=150, reg=0.001)
print(f"  Coefficients (as odds ratios):")
for name, coef in sorted(zip(feat_names, w), key=lambda x:-abs(x[1]))[:20]:
    or_val = math.exp(coef)
    print(f"    {name:30s}  β={coef:+6.3f}  OR={or_val:5.2f}")

with open(f'{OUT_DIR}/logistic_coefs.json','w') as f:
    json.dump({n: {'beta':float(c), 'OR':float(math.exp(c))} for n,c in zip(feat_names,w)}, f, indent=2)

# === 7. Cross-lingual gaps per model (top table) ===
print("\n=== Cross-lingual gap per model under L3_behavioral (ethical) ===")
model_lang_rate = defaultdict(dict)
for m in models_available:
    for l in ['en','pt','es','zh','hi']:
        rt = get_rate(l,'ethical',m,'L3_behavioral')
        if rt is not None: model_lang_rate[m][l] = rt
gaps = []
for m, d in model_lang_rate.items():
    if len(d)>=4:
        gap = max(d.values()) - min(d.values())
        gaps.append((m, d, gap))
print(f"{'model':30s}  EN     PT     ES     ZH     HI     gap")
for m,d,g in sorted(gaps, key=lambda x:-x[2])[:15]:
    fmt = lambda x: f"{x*100:5.2f}" if x is not None else "  -  "
    print(f"  {m:28s}  {fmt(d.get('en'))} {fmt(d.get('pt'))} {fmt(d.get('es'))} {fmt(d.get('zh'))} {fmt(d.get('hi'))}  {g*100:5.2f}")

with open(f'{OUT_DIR}/gaps.json','w') as f:
    json.dump({m: {'rates':d, 'gap':g} for m,d,g in gaps}, f, indent=2)

print("\nAnalysis files saved to:", OUT_DIR)
