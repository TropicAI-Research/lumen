#!/usr/bin/env python3
"""Generate additional figures: 3-judge agreement, sandbox consequences."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

OUT = 'analysis/figures'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family':'sans-serif','font.size':10,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.titlesize':11,'axes.labelsize':10,
    'xtick.labelsize':9,'ytick.labelsize':9,
    'legend.fontsize':9,'figure.dpi':150,
})

C = {
    'comply': '#d73027', 'saydo': '#9061c2', 'redirect': '#1a9850', 'ambig': '#fdae61',
    'judge_o4': '#377eb8', 'judge_opus': '#984ea3', 'judge_gemini': '#ff7f00',
}

# =============================================================================
# Figure 8: 3-judge agreement on a single coherent stratified sample (n=639)
# =============================================================================
import json as _json
_parsed = _json.load(open('crossjudge_656_parsed.json'))
_n = _parsed['n']
_all3 = sum(1 for c in _parsed['cases'] if (
    _parsed['o4_mini'][c] == _parsed['opus'][c] == _parsed['gemini'][c]
))
_majority = sum(1 for c in _parsed['cases'] if len({_parsed['o4_mini'][c], _parsed['opus'][c], _parsed['gemini'][c]}) <= 2)
SAMPLES = [
    dict(
        label=f'Single stratified sample (n={_n})',
        pairs=[
            ('o4-mini', 'Opus 4.7',        _parsed['agreement']['pairwise_cohen_kappa']['o4-mini_x_opus']),
            ('o4-mini', 'Gemini 2.5 Pro',  _parsed['agreement']['pairwise_cohen_kappa']['o4-mini_x_gemini']),
            ('Opus 4.7', 'Gemini 2.5 Pro', _parsed['agreement']['pairwise_cohen_kappa']['opus_x_gemini']),
        ],
        all3=_all3, maj_only=_majority - _all3, diff=_n - _majority, total=_n,
        fleiss=_parsed['agreement']['fleiss_kappa'],
    ),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                         gridspec_kw={'width_ratios':[1.2, 1]})
axes = [axes]  # keep iteration shape

for row_i, s in enumerate(SAMPLES):
    # Left: pairwise kappas
    ax = axes[row_i][0]
    names = [f"{a}\nvs\n{b}" for a, b, _ in s['pairs']]
    kappas = [p[2] for p in s['pairs']]
    colors_bar = [C['judge_opus'], C['judge_gemini'], '#8c564b']
    bars = ax.barh(names, kappas, color=colors_bar,
                   edgecolor='black', linewidth=0.4)
    for bar, k in zip(bars, kappas):
        ax.text(k + 0.01, bar.get_y() + bar.get_height()/2, f'κ={k:.3f}',
                va='center', fontsize=10, fontweight='bold')
    ax.axvline(0.8, color='gray', linestyle=':', alpha=0.6, linewidth=0.8)
    ax.axvline(0.6, color='gray', linestyle=':', alpha=0.4, linewidth=0.6)
    ax.set_xlabel("Cohen's κ (4-class)")
    ax.set_xlim(0.5, 1.0)
    ax.set_title(f"{s['label']} — pairwise κ", fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle=':')

    # Right: agreement breakdown
    ax = axes[row_i][1]
    vals_n = [s['all3'], s['maj_only'], s['diff']]
    vals = [v / s['total'] * 100 for v in vals_n]
    bars = ax.bar(['All 3\nagree', '2 of 3\nagree', 'All differ'],
                  vals,
                  color=['#3fb950','#d29922','#f85149'],
                  edgecolor='black', linewidth=0.4)
    for bar, v, cnt in zip(bars, vals, vals_n):
        ax.text(bar.get_x()+bar.get_width()/2, v+1,
                f'{v:.1f}%\n(n={cnt})',
                ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('Percentage of cases (%)')
    ax.set_ylim(0, 110)
    ax.set_title(f"{s['label']} — Fleiss' κ = {s['fleiss']:.3f}",
                 fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle=':')

plt.tight_layout()
plt.savefig(f'{OUT}/fig8_3judge_agreement.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT}/fig8_3judge_agreement.pdf', bbox_inches='tight')
plt.close()
print("fig8 saved")

# =============================================================================
# Figure 9: Sandbox aggregate consequences
# =============================================================================
summary = json.load(open('analysis/sandbox_summary.json'))
agg = summary['aggregate_consequences']

# Filter to harmful consequences (exclude read-only and benign)
label_map = {
    'candidates_excluded': 'Job candidates excluded\n(discriminatory filters)',
    'files_deleted': 'Files deleted\n(incident logs, evidence)',
    'modifications': 'Records modified',
    'messages_sent': 'Messages sent',
    'surveillance_started': 'Surveillance events',
    'medical_unauthorized_accesses': 'Medical records\naccessed unauthorizedly',
    'approvals': 'Requests approved\n(incl. with bypass)',
    'installs': 'Software installs\n(incl. backdoors)',
    'pii_records_leaked': 'PII records leaked',
    'transactions': 'Transactions processed',
}
items = []
for key, label in label_map.items():
    if key in agg and agg[key] > 0:
        items.append((label, agg[key]))
items.sort(key=lambda x: -x[1])

fig, ax = plt.subplots(figsize=(9, 5))
labels = [x[0] for x in items]
vals = [x[1] for x in items]
colors_bar = ['#d73027' if v > 100 else '#fdae61' for v in vals]
bars = ax.barh(labels, vals, color=colors_bar, edgecolor='black', linewidth=0.4)
ax.invert_yaxis()
for bar, v in zip(bars, vals):
    ax.text(bar.get_width() + max(vals)*0.01, bar.get_y()+bar.get_height()/2,
            f'{v:,}', va='center', fontsize=9, fontweight='bold')
ax.set_xlabel('Number of simulated consequences (across 9,871 L2-positive cases)')
ax.set_title('Aggregate harmful consequences in sandbox\n(what would have happened if tools had been connected to production)', fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle=':')
ax.set_xlim(0, max(vals)*1.15)
plt.tight_layout()
plt.savefig(f'{OUT}/fig9_sandbox_consequences.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT}/fig9_sandbox_consequences.pdf', bbox_inches='tight')
plt.close()
print("fig9 saved")

# =============================================================================
# Figure 10: Sandbox state-change rate by judge label
# =============================================================================
data_by_label = summary['by_judge_label']
order = ['COMPLY', 'SAY_DO_GAP', 'REDIRECT', 'AMBIGUOUS']
rates = []
ns = []
colors_bar = [C['comply'], C['saydo'], C['redirect'], C['ambig']]
for lbl in order:
    if lbl in data_by_label:
        d = data_by_label[lbl]
        rates.append(d['changed']/d['total']*100)
        ns.append(d['total'])
    else:
        rates.append(0); ns.append(0)

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(order))
bars = ax.bar(x, rates, color=colors_bar, edgecolor='black', linewidth=0.4, width=0.6)
for bar, r, n in zip(bars, rates, ns):
    ax.text(bar.get_x()+bar.get_width()/2, r+1.5, f'{r:.1f}%\n(n={n:,})',
            ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(order)
ax.set_ylabel('% of cases causing state change in sandbox')
ax.set_title('Sandbox validation of L3 judge labels\n(harmful state change rate by judge classification)',
             fontweight='bold')
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3, linestyle=':')
# Annotation
ax.text(1, 95, 'COMPLY and SAY_DO_GAP cause\nstate change at same rate',
        ha='center', fontsize=8, style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
plt.tight_layout()
plt.savefig(f'{OUT}/fig10_sandbox_by_label.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT}/fig10_sandbox_by_label.pdf', bbox_inches='tight')
plt.close()
print("fig10 saved")

print(f"\nAll saved to {OUT}")
