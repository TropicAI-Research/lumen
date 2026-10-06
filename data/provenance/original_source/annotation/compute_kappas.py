#!/usr/bin/env python3
"""
Compute judge-vs-human kappas after annotation is complete.

Uses ONLY the 100-case fresh-start set (not the legacy 25 from
human_ratings_corrected.csv). The legacy set is preserved on disk for
provenance but not used in this pipeline.

Inputs:
  annotation/cases_meta.json      — hidden judge labels for the 100 cases
  annotation/annotations_*.json   — exported annotations from annotator.html

Outputs:
  analysis/human_calibration.json — full statistics for paper §3.2.1
  prints summary table

Run:
  python3 annotation/compute_kappas.py
  # or pass an explicit annotations file:
  python3 annotation/compute_kappas.py annotation/annotations_<timestamp>.json
"""
import json, sys, glob
import numpy as np
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ['COMPLY', 'SAY_DO_GAP', 'REDIRECT', 'AMBIGUOUS']
L2I = {l: i for i, l in enumerate(LABELS)}

def cohen_kappa(rater_a, rater_b, n_classes):
    n = len(rater_a)
    if n == 0:
        return float('nan')
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for x, y in zip(rater_a, rater_b):
        cm[x, y] += 1
    po = np.trace(cm) / n
    pe = sum((cm.sum(axis=0)[c] / n) * (cm.sum(axis=1)[c] / n) for c in range(n_classes))
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else float('nan')

def bootstrap_ci(rater_a, rater_b, n_classes, B=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(rater_a)
    ks = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        a = [rater_a[i] for i in idx]
        b = [rater_b[i] for i in idx]
        k = cohen_kappa(a, b, n_classes)
        if not np.isnan(k):
            ks.append(k)
    if not ks:
        return float('nan'), float('nan'), float('nan')
    ks = np.array(ks)
    return float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5)), float(ks.std())

def per_class_pr(judge_labels, human_labels, target_class):
    tp = sum(1 for j, h in zip(judge_labels, human_labels) if j == target_class and h == target_class)
    fp = sum(1 for j, h in zip(judge_labels, human_labels) if j == target_class and h != target_class)
    fn = sum(1 for j, h in zip(judge_labels, human_labels) if j != target_class and h == target_class)
    p = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    r = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1 = 2 * p * r / (p + r) if (p + r) > 0 and not np.isnan(p) and not np.isnan(r) else float('nan')
    return {'precision': p, 'recall': r, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn,
            'support_judge': tp + fp, 'support_human': tp + fn}

def main():
    if len(sys.argv) < 2:
        candidates = sorted(glob.glob(str(ROOT / 'annotation' / 'annotations_*.json')))
        if not candidates:
            print("Usage: compute_kappas.py [path/to/annotations_*.json]")
            print(f"No annotations_*.json found in {ROOT / 'annotation'}")
            sys.exit(1)
        annot_path = Path(candidates[-1])
        print(f"Using latest: {annot_path}")
    else:
        annot_path = Path(sys.argv[1])

    annots = json.load(open(annot_path))
    meta = json.load(open(ROOT / 'annotation' / 'cases_meta.json'))
    meta_by_id = {m['case_id']: m for m in meta}

    human, judge = [], []
    skipped = 0
    for a in annots['annotations']:
        if a['label'] is None:
            skipped += 1
            continue
        cid = a['case_id']
        if cid not in meta_by_id:
            print(f"  warn: case_id {cid} not in cases_meta.json — skipping")
            continue
        human.append(a['label'])
        judge.append(meta_by_id[cid]['judge_label'])

    n = len(human)
    print(f"\n=== Calibration set (fresh, n={n}/{len(meta)}) ===")
    if skipped:
        print(f"  ({skipped} cases unlabeled — excluded)")
    print(f"  Human label distribution: {Counter(human)}")
    print(f"  Judge label distribution: {Counter(judge)}")

    if n == 0:
        print("No labeled cases. Run the annotator first.")
        sys.exit(0)

    h4 = [L2I[h] for h in human]
    j4 = [L2I[j] for j in judge]
    hb = [1 if h == 'COMPLY' else 0 for h in human]
    jb = [1 if j == 'COMPLY' else 0 for j in judge]

    k4 = cohen_kappa(h4, j4, 4)
    kb = cohen_kappa(hb, jb, 2)
    ci_lo_4, ci_hi_4, se_4 = bootstrap_ci(h4, j4, 4)
    ci_lo_b, ci_hi_b, se_b = bootstrap_ci(hb, jb, 2)
    agree_b = sum(1 for x, y in zip(hb, jb) if x == y) / n * 100
    agree_4 = sum(1 for x, y in zip(h4, j4) if x == y) / n * 100

    print(f"\n=== o4-mini vs human ===")
    print(f"  Raw agreement (4-class): {agree_4:.1f}% ({sum(1 for x,y in zip(h4,j4) if x==y)}/{n})")
    print(f"  Raw agreement (binary):  {agree_b:.1f}%")
    print(f"  Cohen κ 4-class: {k4:.3f}  95% CI [{ci_lo_4:.3f}, {ci_hi_4:.3f}]  SE={se_4:.3f}")
    print(f"  Cohen κ binary:  {kb:.3f}  95% CI [{ci_lo_b:.3f}, {ci_hi_b:.3f}]  SE={se_b:.3f}")

    print(f"\n=== Per-class precision/recall (judge as predictor, human as ground truth) ===")
    pr_table = {}
    for label in LABELS:
        pr = per_class_pr(judge, human, label)
        pr_table[label] = pr
        print(f"  {label:12s} P={pr['precision']:.3f}  R={pr['recall']:.3f}  F1={pr['f1']:.3f}  "
              f"(judge n={pr['support_judge']}, human n={pr['support_human']})")

    cm_4 = np.zeros((4, 4), dtype=int)
    for h, j in zip(h4, j4):
        cm_4[h, j] += 1
    print(f"\n=== Confusion matrix (rows=human, cols=judge) ===")
    print(f"{'':14s}" + ''.join(f"{l:>13s}" for l in LABELS))
    for i, l in enumerate(LABELS):
        print(f"{l:14s}" + ''.join(f"{cm_4[i,c]:>13d}" for c in range(4)))

    output = {
        'design': '100 cases, stratified 25 per judge label (COMPLY/SAY_DO_GAP/REDIRECT/AMBIGUOUS), '
                  'Portuguese only, blind annotation (model name hidden), single rater, '
                  'fresh-start set (legacy 25-case set is not used in this analysis).',
        'n_total_in_set': len(meta),
        'n_annotated': n,
        'human_label_distribution': dict(Counter(human)),
        'judge_label_distribution': dict(Counter(judge)),
        'o4_mini_vs_human': {
            'kappa_4class': float(k4),
            'kappa_4class_ci95': [ci_lo_4, ci_hi_4],
            'kappa_4class_se': se_4,
            'kappa_binary': float(kb),
            'kappa_binary_ci95': [ci_lo_b, ci_hi_b],
            'kappa_binary_se': se_b,
            'agreement_4class_pct': agree_4,
            'agreement_binary_pct': agree_b,
        },
        'per_class_pr': pr_table,
        'confusion_matrix_4class': {
            'labels': LABELS,
            'matrix_human_x_judge': cm_4.tolist(),
        },
        'methodology': {
            'sampling': 'Stratified by primary-judge label (o4-mini-2025-04-16): 25 cases sampled '
                        'from each of COMPLY, SAY_DO_GAP, REDIRECT, AMBIGUOUS. Population: PT '
                        'L2-positive cases across the 32-model panel.',
            'rater_count': 1,
            'languages_validated': ['pt'],
            'cross_lingual_generalization': 'inferred from cross-judge κ over n=656 multilingual cases',
            'blind_to_model_identity': True,
            'bootstrap_resamples': 1000,
            'seed': 42,
        },
    }

    out_path = ROOT / 'analysis' / 'human_calibration.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nWritten: {out_path}")

if __name__ == '__main__':
    main()
