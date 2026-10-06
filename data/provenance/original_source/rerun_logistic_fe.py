"""Re-run the logistic regression with model fixed effects via statsmodels (unregularized).

This replaces regression_with_fe.py which used sklearn LogisticRegression(C=1.0),
introducing L2 shrinkage that biased category coefficients away from the paper's values.
The paper Tab 26 needs unregularized MLE.

Drops the 3 perfect-separation models: claude-haiku-4.5, gemini-3.1-pro-preview, gpt-5.4.
N = 207,360 - 3*6,480 = 187,920 obs from 29 models.

Outputs analysis/logistic_coefs.json with {beta, OR, ci_lo, ci_hi, p, se} per coef.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
records = json.load(open(ROOT / "analysis/records.json"))
print(f"Records loaded: {len(records):,}")

df = pd.DataFrame(records)
df = df.dropna(subset=["category", "pressure_type", "L3_behavioral"])
print(f"After drop NA: {len(df):,}")

# Drop perfect-separation models
zero_models = ["claude-haiku-4.5", "gemini-3.1-pro-preview", "gpt-5.4"]
df = df[~df["model"].isin(zero_models)].reset_index(drop=True)
print(f"After dropping zero-violation models: {len(df):,} (expected 187,920)")

# Build design matrix
y = df["L3_behavioral"].values

X_cols = {}
# Language (ref EN)
for lang in ["pt", "es", "zh", "hi"]:
    X_cols[f"lang_{lang}"] = (df["lang"] == lang).astype(int).values

# Prompt (ref ethical)
X_cols["prompt_neutral"] = (df["prompt"] == "neutral").astype(int).values

# Category (ref autonomy)
cats = sorted(df["category"].unique())
cats.remove("autonomy")
for c in cats:
    X_cols[f"cat_{c}"] = (df["category"] == c).astype(int).values

# Pressure (ref authority)
press = sorted(df["pressure_type"].unique())
press.remove("authority")
for p in press:
    X_cols[f"press_{p}"] = (df["pressure_type"] == p).astype(int).values

# Model fixed effects (drop alphabetically first)
models = sorted(df["model"].unique())
ref_model = models[0]
print(f"Reference model: {ref_model}; FE for {len(models)-1} other models")
for m in models[1:]:
    X_cols[f"model_{m}"] = (df["model"] == m).astype(int).values

X = pd.DataFrame(X_cols)
X = sm.add_constant(X, prepend=True)

print(f"Design matrix: {X.shape}")
print("Fitting...")
model = sm.Logit(y, X)
result = model.fit(disp=False, method="newton", maxiter=100)

print(f"\nN = {len(y):,}")
print(f"McFadden Pseudo R² = {result.prsquared:.4f}")
print(f"Log-Likelihood = {result.llf:.2f}")
print(f"LLR p = {result.llr_pvalue:.2e}")

# Extract coefficients (drop model FE from output)
out = {}
for name in X.columns:
    if name.startswith("model_") or name == "const":
        continue
    beta = float(result.params[name])
    se = float(result.bse[name])
    pval = float(result.pvalues[name])
    or_val = float(np.exp(beta))
    ci_lo = float(np.exp(beta - 1.96 * se))
    ci_hi = float(np.exp(beta + 1.96 * se))
    out[name] = {
        "beta": round(beta, 6),
        "OR": round(or_val, 6),
        "ci_lo": round(ci_lo, 6),
        "ci_hi": round(ci_hi, 6),
        "se": round(se, 6),
        "p": pval,
    }

# Intercept
out["intercept"] = {
    "beta": round(float(result.params["const"]), 6),
    "OR": round(float(np.exp(result.params["const"])), 6),
}
out["_meta"] = {
    "n_obs": int(len(y)),
    "mcfadden_r2": round(float(result.prsquared), 6),
    "loglik": round(float(result.llf), 2),
    "null_loglik": round(float(result.llnull), 2),
    "llr_pvalue": float(result.llr_pvalue),
}

with open(ROOT / "analysis/logistic_coefs.json", "w") as f:
    json.dump(out, f, indent=2, default=float)

print("\nKey ORs:")
for k in [
    "lang_pt", "lang_es", "lang_zh", "lang_hi",
    "prompt_neutral",
    "cat_complicity", "cat_legal", "cat_loyalty_conflict", "cat_privacy",
    "cat_sycophancy", "cat_whistleblowing", "cat_justice_non_discrimination",
    "press_social", "press_none", "press_justification",
]:
    if k in out:
        c = out[k]
        print(f"  {k:40s} β={c['beta']:+7.4f} OR={c['OR']:6.3f} "
              f"[{c['ci_lo']:.3f},{c['ci_hi']:.3f}] p={c['p']:.2e}")
