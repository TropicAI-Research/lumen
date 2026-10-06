#!/usr/bin/env python3
"""Generate paper figures from aggregated data."""
import json, sys, os
from collections import defaultdict, Counter

# Optional: try matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    print("matplotlib not available, installing...")
    os.system("pip install --quiet matplotlib numpy")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True

OUT = 'analysis/figures'
os.makedirs(OUT, exist_ok=True)

records = json.load(open('analysis/records.json'))
table = json.load(open('analysis/violation_rate_table.json'))
gaps = json.load(open('analysis/gaps.json'))
judge_combined = json.load(open('judge_combined_parsed.json'))

LANGS = ['en','pt','es','zh','hi']
LANG_NAMES = {'en':'English','pt':'Portuguese','es':'Spanish','zh':'Chinese','hi':'Hindi'}
PROMPTS = ['ethical','neutral']

# Color palette
C = {
    'comply': '#d73027', 'saydo': '#9061c2', 'redirect': '#1a9850', 'ambig': '#fdae61', 'refusal':'#4575b4',
    'l0':'#bababa', 'l1':'#878787', 'l2':'#4d4d4d', 'l3beh':'#d73027', 'l3str':'#a50026',
    'en':'#377eb8','pt':'#4daf4a','es':'#ff7f00','zh':'#984ea3','hi':'#e41a1c',
}

plt.rcParams.update({
    'font.family':'sans-serif','font.size':10,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.titlesize':11,'axes.labelsize':10,
    'xtick.labelsize':9,'ytick.labelsize':9,
    'legend.fontsize':9,'figure.dpi':150,
})

# ============ Figure 1: Violation rate by language with CIs, L0 vs L3_behavioral ============
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax_i, prompt in enumerate(PROMPTS):
    ax = axes[ax_i]
    rows = [r for r in table if r['prompt']==prompt]
    rows.sort(key=lambda x: LANGS.index(x['lang']))
    x = np.arange(len(rows))
    w = 0.28
    l0 = [r['L0']['mean']*100 for r in rows]
    l0_err = [[(r['L0']['mean']-r['L0']['lo'])*100 for r in rows],
              [(r['L0']['hi']-r['L0']['mean'])*100 for r in rows]]
    l3b = [r['L3_behavioral']['mean']*100 for r in rows]
    l3b_err = [[(r['L3_behavioral']['mean']-r['L3_behavioral']['lo'])*100 for r in rows],
               [(r['L3_behavioral']['hi']-r['L3_behavioral']['mean'])*100 for r in rows]]
    l3s = [r['L3_strict']['mean']*100 for r in rows]
    l3s_err = [[(r['L3_strict']['mean']-r['L3_strict']['lo'])*100 for r in rows],
               [(r['L3_strict']['hi']-r['L3_strict']['mean'])*100 for r in rows]]
    ax.bar(x-w, l0, w, yerr=l0_err, color=C['l0'], label='L0 (any tool)', capsize=3, edgecolor='black', linewidth=0.3)
    ax.bar(x, l3b, w, yerr=l3b_err, color=C['l3beh'], label='L3_behav (COMPLY+SAY_DO_GAP)', capsize=3, edgecolor='black', linewidth=0.3)
    ax.bar(x+w, l3s, w, yerr=l3s_err, color=C['l3str'], label='L3_strict (COMPLY only)', capsize=3, edgecolor='black', linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels([LANG_NAMES[r['lang']] for r in rows], rotation=20, ha='right')
    ax.set_ylabel('Violation rate (%)')
    ax.set_title(f'System prompt: {prompt}', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    if ax_i == 0: ax.legend(loc='upper left')
plt.suptitle('Violation rate across languages and metric definitions', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig1_violation_by_language.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT}/fig1_violation_by_language.pdf', bbox_inches='tight')
plt.close()
print("fig1 saved")

# ============ Figure 2: Heatmap model × language (L3_behavioral, ethical) ============
model_lang = defaultdict(lambda: {l:[0,0] for l in LANGS})
for r in records:
    if r['prompt']!='ethical': continue
    if r['lang'] not in LANGS: continue
    model_lang[r['model']][r['lang']][1] += 1
    if r['L3_behavioral']: model_lang[r['model']][r['lang']][0] += 1
# Compute rates
model_rates = {}
for m, d in model_lang.items():
    if all(d[l][1]>0 for l in LANGS):
        model_rates[m] = [d[l][0]/d[l][1]*100 for l in LANGS]
# Sort by max rate
model_rates = dict(sorted(model_rates.items(), key=lambda x: -max(x[1])))

fig, ax = plt.subplots(figsize=(7, 8))
data = np.array([model_rates[m] for m in model_rates])
im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max(40, data.max()))
ax.set_xticks(range(len(LANGS))); ax.set_xticklabels([LANG_NAMES[l] for l in LANGS])
ax.set_yticks(range(len(model_rates))); ax.set_yticklabels(list(model_rates.keys()))
# annotate
for i in range(len(model_rates)):
    for j in range(len(LANGS)):
        v = data[i,j]
        ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                fontsize=8, color='white' if v > data.max()*0.5 else 'black')
plt.colorbar(im, ax=ax, label='L3_behavioral violation rate (%)', shrink=0.7)
ax.set_title(f'Cross-lingual safety variation (ethical prompt, {len(model_rates)} models)', fontweight='bold')
ax.set_xlabel('Language')
plt.tight_layout()
plt.savefig(f'{OUT}/fig2_heatmap_model_lang.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT}/fig2_heatmap_model_lang.pdf', bbox_inches='tight')
plt.close()
print("fig2 saved")

# ============ Figure 3: SAY_DO_GAP prevalence by model ============
# Source: judge_combined_parsed.json (canonical 17,128 judgments matching Table 14)
model_sdg = defaultdict(lambda: [0,0])
for v in judge_combined.values():
    model_sdg[v['model']][1] += 1
    if v.get('label') == 'SAY_DO_GAP':
        model_sdg[v['model']][0] += 1
model_sdg_rate = {m: (sdg/t, t) for m,(sdg,t) in model_sdg.items() if t>=20}
model_sdg_rate = dict(sorted(model_sdg_rate.items(), key=lambda x: -x[1][0]))

fig, ax = plt.subplots(figsize=(9, 5))
names = list(model_sdg_rate.keys())[:15]
rates = [model_sdg_rate[n][0]*100 for n in names]
ns = [model_sdg_rate[n][1] for n in names]
bars = ax.barh(names, rates, color=C['saydo'], edgecolor='black', linewidth=0.3)
ax.invert_yaxis()
ax.set_xlabel('SAY_DO_GAP rate (% of L2-positive cases)')
ax.set_title('Say-Do gap prevalence by model\n(tool call executes harm while text refuses)', fontweight='bold')
for bar, r, n in zip(bars, rates, ns):
    ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2, f'{r:.1f}% (n={n})',
            va='center', fontsize=8)
ax.grid(axis='x', alpha=0.3, linestyle=':')
ax.set_xlim(0, max(rates)*1.25)
plt.tight_layout()
plt.savefig(f'{OUT}/fig3_saydo_by_model.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT}/fig3_saydo_by_model.pdf', bbox_inches='tight')
plt.close()
print("fig3 saved")

# ============ Figure 4: Label distribution by language (stacked bars) ============
# Source: judge_combined_parsed.json (canonical 17,128 judgments)
lang_dist = defaultdict(Counter)
for v in judge_combined.values():
    lang_dist[v['lang']][v.get('label')] += 1

fig, ax = plt.subplots(figsize=(9, 4.5))
labels = ['COMPLY','SAY_DO_GAP','REDIRECT','AMBIGUOUS']
colors = [C['comply'], C['saydo'], C['redirect'], C['ambig']]
data_pct = np.zeros((len(LANGS), len(labels)))
totals = []
for i, l in enumerate(LANGS):
    tot = sum(lang_dist[l][k] for k in labels)
    totals.append(tot)
    for j, k in enumerate(labels):
        data_pct[i,j] = lang_dist[l][k]/tot*100 if tot else 0
x = np.arange(len(LANGS))
bottom = np.zeros(len(LANGS))
for j, (lbl, c) in enumerate(zip(labels, colors)):
    ax.bar(x, data_pct[:,j], bottom=bottom, color=c, label=lbl, edgecolor='white', linewidth=1)
    # annotate significant segments
    for i, v in enumerate(data_pct[:,j]):
        if v >= 3:
            ax.text(x[i], bottom[i]+v/2, f'{v:.1f}%', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
    bottom += data_pct[:,j]
ax.set_xticks(x); ax.set_xticklabels([f"{LANG_NAMES[l]}\n(n={t})" for l,t in zip(LANGS, totals)])
ax.set_ylabel('Composition of tool-calling behavior (%)')
ax.set_title('Distribution of judge labels across languages\n(among cases with substantive tool calls)', fontweight='bold')
ax.legend(loc='upper right', ncol=4, bbox_to_anchor=(1, 1.10))
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_label_dist_by_language.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT}/fig4_label_dist_by_language.pdf', bbox_inches='tight')
plt.close()
print("fig4 saved")

# ============ Figure 5: SAY_DO_GAP by category ============
# Source: judge_combined_parsed.json (canonical 17,128 judgments)
cat_sdg = defaultdict(lambda: [0,0])
for v in judge_combined.values():
    cat_sdg[v['category']][1] += 1
    if v.get('label') == 'SAY_DO_GAP':
        cat_sdg[v['category']][0] += 1
cat_sdg_r = {c: (s/t, t) for c,(s,t) in cat_sdg.items() if t}
cat_sdg_r = dict(sorted(cat_sdg_r.items(), key=lambda x:-x[1][0]))

fig, ax = plt.subplots(figsize=(9, 4))
names = list(cat_sdg_r.keys())
rates = [cat_sdg_r[n][0]*100 for n in names]
ns = [cat_sdg_r[n][1] for n in names]
bars = ax.barh(names, rates, color=C['saydo'], edgecolor='black', linewidth=0.3)
ax.invert_yaxis()
ax.set_xlabel('SAY_DO_GAP rate (% of L2-positive)')
ax.set_title('Say-Do gap by ethical category', fontweight='bold')
for bar, r, n in zip(bars, rates, ns):
    ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2, f'{r:.1f}% (n={n})', va='center', fontsize=8)
ax.grid(axis='x', alpha=0.3, linestyle=':')
ax.set_xlim(0, max(rates)*1.25)
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_saydo_by_category.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT}/fig5_saydo_by_category.pdf', bbox_inches='tight')
plt.close()
print("fig5 saved")

# ============ Figure 6: Cross-lingual gap with CIs, top 10 models ============
# Use gap data
top_gaps = sorted(gaps.items(), key=lambda x:-x[1]['gap'])[:10]
fig, ax = plt.subplots(figsize=(9, 5.5))
for i, (model, d) in enumerate(top_gaps):
    y = len(top_gaps)-1-i  # reverse
    vals = [d['rates'].get(l,0)*100 for l in LANGS]
    for j, (l, v) in enumerate(zip(LANGS, vals)):
        ax.plot([v], [y], 'o', color=C[l], markersize=10, markeredgecolor='black', markeredgewidth=0.5, label=LANG_NAMES[l] if i==0 else None)
    # line connecting min to max
    mn, mx = min(vals), max(vals)
    ax.plot([mn, mx], [y, y], '-', color='gray', alpha=0.4, zorder=0)
    ax.text(max(vals)*1.02+0.5, y, f' gap={d["gap"]*100:.1f}pp', va='center', fontsize=8)
ax.set_yticks(range(len(top_gaps)))
ax.set_yticklabels([m for m,_ in reversed(top_gaps)])
ax.set_xlabel('L3_behavioral violation rate (%)')
ax.set_title('Cross-lingual safety gap: top 10 models\n(each dot = one language, ethical prompt)', fontweight='bold')
ax.legend(loc='lower right', ncol=5, fontsize=8)
ax.grid(axis='x', alpha=0.3, linestyle=':')
ax.set_xlim(-1, max(d['gap']*100 for _,d in top_gaps)*1.15 + 10)
plt.tight_layout()
plt.savefig(f'{OUT}/fig6_gap_top_models.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT}/fig6_gap_top_models.pdf', bbox_inches='tight')
plt.close()
print("fig6 saved")

# ============ Figure 7: Scaling comparison (Llama vs Qwen3 vs Qwen 3.5 families) ============
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

def fam_rates(models, prompt='neutral'):
    out = {}
    for m in models:
        vals = []
        for l in LANGS:
            sub = [r for r in records if r['model']==m and r['lang']==l and r['prompt']==prompt]
            if sub:
                vals.append(sum(r['L3_behavioral'] for r in sub)/len(sub)*100)
            else:
                vals.append(None)
        if all(v is not None for v in vals): out[m] = vals
    return out

panels = [
    ('Llama family',      ['llama-3.1-8b','llama-3.3-70b','llama-4-scout']),
    ('Qwen3 family',      ['qwen3-8b','qwen3-14b','qwen3-32b','qwen3-235b-a22b-2507']),
    ('Qwen 3.5 family',   ['qwen3.5-9b','qwen3.5-27b','qwen3.5-35b-a3b','qwen3.5-397b-a17b']),
]
x = np.arange(len(LANGS))
for ax, (title, fam) in zip(axes, panels):
    ms = fam_rates(fam)
    n = len(ms)
    w = 0.8 / max(n,1)
    for i, (m, vals) in enumerate(ms.items()):
        ax.bar(x + (i - (n-1)/2)*w, vals, w, label=m, edgecolor='black', linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels([LANG_NAMES[l] for l in LANGS], rotation=20, ha='right')
    ax.set_ylabel('L3_behavioral violation rate (%)')
    ax.set_title(f'{title} — neutral prompt', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3, linestyle=':')

plt.tight_layout()
plt.savefig(f'{OUT}/fig7_llama_family.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT}/fig7_llama_family.pdf', bbox_inches='tight')
plt.close()
print("fig7 saved")

print(f"\nAll figures saved to {OUT}")
