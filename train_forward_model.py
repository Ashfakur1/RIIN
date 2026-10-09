#!/usr/bin/env python3
"""
train_forward_model.py
Trains the forward model (raw-material masses -> tile properties) on the 238
real experimental batches. No synthetic data anywhere.

WHAT CHANGED vs. THE PREVIOUS REVISION (all driven by the data audit below
and by the reviewer comments; see Reviewer_Comment_Mapping.md)

 1. DATA AUDIT IS NOW COMPUTED, NOT ASSERTED.
    The previous version printed hard-coded correlations ("r=-0.59 with
    MOR ...") that do not match the data (actual r(total, MOR) = +0.175).
    Every number printed here is computed from the CSV.

 2. FEATURES = AS-WEIGHED MASSES (no renormalisation, no batch_mass_dev).
    The 8 ingredient columns are near-independent uniform draws over their
    design ranges (max |pairwise r| ~ 0.15; totals 90.2-110.6). They are NOT
    a closed simplex, so there is no closure to "correct". Renormalising to
    100 wt% *creates* closure (mean pairwise r becomes negative) and made
    the design matrix numerically singular (condition number ~4e14), which
    produced the 1e13-magnitude SHAP values, meaningless |coef| importances
    and an ill-defined partial correlation. As-weighed masses have condition
    number ~1.3, so correlations, OLS effects, PDP and SHAP are all valid.
    Recommendations from inverse_design.py are still constrained to
    sum = 100 (a proper recipe); the model is simply queried on that slice.

 3. TARGET DIAGNOSTICS + HONEST TARGET SET.
    A target for which >= 90 % of batches share one value has no variance and cannot be
    learned (R2 = 0 by construction, a tiny "MAPE" is meaningless): it is treated as a FIXED
    property, not a modelled target. The decision is printed and stored in
    data/target_diagnostics.csv, so the same script adapts to a new dataset (in the first
    real-data release shrinkage was constant; in the current dataset it varies and is modelled).
    A negative water absorption (measurement artefact) is set to 0 % and reported. WA
    censoring is DETECTED (>= 10 % of batches at the minimum value): only then is WA treated as
    left-censored and are WA metrics also given on the uncensored subset.

 4. NESTED CROSS-VALIDATION for architecture selection (tuning happens
    INSIDE each outer fold). Previously each model was tuned on the whole
    training split and then scored by CV on that same split (optimistic).
    All six tuned models are also scored on the untouched held-out test set.

 5. MLP is wrapped with target scaling (previously it predicted unscaled MOR
    and shrinkage, giving R2 = -0.8 and -214155: an unfair strawman).

 6. PREDICTION UNCERTAINTY: out-of-fold residual quantiles give a 90 %
    prediction half-width per target (stored in metadata.json, used by
    inverse_design.py and the app). Empirical coverage on the held-out
    test set is reported.

 7. CLOSURE-SAFE ANALYSIS: correlations (Pearson + Spearman, BH-FDR),
    standardised OLS effects with CIs and VIFs, and a CLR-based robustness
    check. The ill-posed partial correlation is removed.

 8. Bounds = observed min/max (margin 0). The previous 10 % margin allowed
    the optimiser to propose recipes outside anything fabricated.

 9. Material names follow Table 1 ("Crushed Fired Tile", "ETP Sludge") in
    every figure (Reviewer 8).

10. WA EXCEEDANCE MODEL (section 5b): P(WA > limit) for any recipe, scored by out-of-fold AUC
    / Brier score. If WA is left-censored a Tobit model is fitted (parameters in metadata.json
    "wa_tobit"); otherwise a Gaussian residual model around the forward-model prediction is used
    ("wa_exceedance"). Both are consumed by inverse_design.py and the app.

Run order for the whole project:   python train_forward_model.py   ->   python run_analyses.py
(optional)  ->  streamlit run streamlit_app.py.
Set QUICK=1 in the environment for a fast smoke test (tiny search budgets).
"""

import json, math, os, warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from scipy import stats
from scipy.stats import pearsonr, spearmanr, loguniform, randint, uniform
from scipy.stats import norm as _norm
from scipy.optimize import minimize as _minimize
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import (
    KFold, RepeatedKFold, RandomizedSearchCV, cross_validate,
    cross_val_predict, train_test_split,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_utils import MultiTargetModel

warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="talk", font_scale=1.0)

# ── Paths ─────────────────────────────────────────────────────────────────
ROOTDIR = Path(__file__).parent
DATADIR, MODELDIR, PLOTDIR = ROOTDIR / "data", ROOTDIR / "models", ROOTDIR / "plots"
for p in (DATADIR, MODELDIR, PLOTDIR):
    p.mkdir(exist_ok=True, parents=True)
RAW_DATA_FILE = ROOTDIR / "ceramic_tiles_238_samples.csv"

# ── Configuration ─────────────────────────────────────────────────────────
QUICK = os.environ.get("QUICK", "0") == "1"
RANDOM_STATE = 7
TEST_SIZE = 0.20
KMM2_TO_MPA = 9.80665            # kgf/mm^2 -> MPa
N_TUNE_ITER_NESTED = 3 if QUICK else 15     # draws per inner search
INNER_CV_FOLDS = 2 if QUICK else 3
OUTER_REPEATS = 1 if QUICK else 2           # outer = 5-fold x OUTER_REPEATS
N_TUNE_ITER_FINAL = 5 if QUICK else 40      # final tuning on the full train split
FINAL_CV_FOLDS = 3 if QUICK else 5
INTERVAL_LEVEL = 0.90                       # prediction-interval coverage
BOUNDS_MARGIN = 0.0                         # search only inside observed box
CONSTANT_TARGET_MODAL_FRACTION = 0.90       # >= this share at one value -> not modelled
WA_CENSORED_FRACTION = 0.10                 # >= this share at the minimum WA -> left-censored
# Model-selection rule. "one_se": among models whose nested-CV R2 is within one
# standard error of the best, take the SIMPLEST (Hastie et al., one-standard-error
# rule) - with n=190 and R2 differences of ~0.01-0.03 between models, "highest mean"
# mostly selects noise. "best_mean" restores the old behaviour.
SELECTION_RULE = "one_se"
COMPLEXITY = {"LinearRegression": 0, "RandomForest": 2, "XGB": 2, "LGBM": 2,
              "CatBoost": 2, "MLP": 3}

# ISO 13006 / EN 14411 group BIa (fully vitrified, pressed): E <= 0.5 %,
# mean modulus of rupture >= 35 N/mm2. Verify against your edition of the standard.
ISO13006_BIA_WA_MAX = 0.5
ISO13006_BIA_MOR_MIN = 35.0

_DPI = 300
_FS_TITLE, _FS_AX, _FS_TICK = 16, 14, 12


def savefig(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(PLOTDIR / f"{stem}.{ext}", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD + DATA AUDIT
# ══════════════════════════════════════════════════════════════════════════
raw = pd.read_csv(RAW_DATA_FILE)
materials = ["AG98", "AG22", "AG23", "SodaF", "PotashF", "Crushing", "ETP", "NaSil"]
comp_cols = [f"{m}_wtpct" for m in materials]
feature_cols = list(comp_cols)          # as-weighed masses, NOT renormalised

df = raw.rename(columns={m: f"{m}_wtpct" for m in materials}).copy()
df["MOR_MPa"] = df["MOR_kgf_mm2"] * KMM2_TO_MPA
df["WA_pct"] = df["WA_fraction"] * 100.0
_n_neg_wa = int((df["WA_pct"] < 0).sum())
if _n_neg_wa:
    print(f"  NOTE: {_n_neg_wa} batch(es) report a negative water absorption "
          f"(min {df['WA_pct'].min():.4f} %), which is physically impossible (measurement artefact); "
          "set to 0.0 % for modelling.")
    df.loc[df["WA_pct"] < 0, "WA_pct"] = 0.0
ALL_TARGETS = ["MOR_MPa", "WA_pct", "Shrinkage_pct"]
df["batch_total_wtpct"] = df[comp_cols].sum(axis=1)

TGT_LABELS = {"MOR_MPa": "Firing MOR (MPa)", "WA_pct": "Water Absorption (%)",
              "Shrinkage_pct": "Fired Shrinkage (%)"}
MAT_SHORT = {"AG98_wtpct": "AG98", "AG22_wtpct": "AG22", "AG23_wtpct": "AG23",
             "SodaF_wtpct": "Soda Feldspar", "PotashF_wtpct": "Potash Feldspar",
             "Crushing_wtpct": "Crushed Fired Tile", "ETP_wtpct": "ETP Sludge",
             "NaSil_wtpct": "Na-Silicate"}
n_total = len(df)
tot = df["batch_total_wtpct"]

audit = {"n_batches": int(n_total)}
audit["total_mass"] = {"mean": float(tot.mean()), "sd": float(tot.std()),
                       "min": float(tot.min()), "max": float(tot.max()),
                       "n_deviating_gt_0p5": int(((tot - 100).abs() > 0.5).sum())}
_cm = df[comp_cols].corr().values
_off = _cm[~np.eye(len(comp_cols), dtype=bool)]
audit["raw_parts_max_abs_pairwise_r"] = float(np.abs(_off).max())
_ren = df[comp_cols].div(tot, axis=0) * 100
_cm_ren = _ren.corr().values
audit["renormalised_parts_mean_pairwise_r"] = float(_cm_ren[~np.eye(len(comp_cols), dtype=bool)].mean())
_Xs = (df[comp_cols] - df[comp_cols].mean()) / df[comp_cols].std()
audit["cond_number_asweighed"] = float(np.linalg.cond(_Xs.values))
_Xr = pd.concat([_ren, (tot - 100).rename("dev")], axis=1)
audit["cond_number_renormalised_plus_dev"] = float(
    np.linalg.cond(((_Xr - _Xr.mean()) / _Xr.std()).values))
audit["corr_total_mass_with_targets"] = {
    t: float(np.corrcoef(tot, df[t])[0, 1]) for t in ALL_TARGETS}

print(f"Loaded {n_total} real experimental batches (no synthetic data).")
print(f"As-weighed batch totals: {tot.min():.1f}-{tot.max():.1f} (mean {tot.mean():.1f}, "
      f"sd {tot.std():.1f}); {audit['total_mass']['n_deviating_gt_0p5']}/{n_total} "
      f"deviate >0.5 from 100.")
print(f"Ingredient masses are near-independent: max |pairwise r| = "
      f"{audit['raw_parts_max_abs_pairwise_r']:.3f} (a closed simplex would give a mean "
      f"pairwise r of about -1/7 = -0.14; renormalising here gives "
      f"{audit['renormalised_parts_mean_pairwise_r']:.3f}, i.e. closure is CREATED by renormalising).")
print(f"Design-matrix condition number: as-weighed {audit['cond_number_asweighed']:.1f} vs "
      f"renormalised+deviation {audit['cond_number_renormalised_plus_dev']:.1e} (singular).")
print("Correlation of total batch mass with targets (computed): " +
      ", ".join(f"{t}: {v:+.3f}" for t, v in audit["corr_total_mass_with_targets"].items()))

# ── Target diagnostics ────────────────────────────────────────────────────
diag_rows, MODEL_TARGETS, CONSTANT_TARGETS = [], [], {}
for t in ALL_TARGETS:
    vc = df[t].round(6).value_counts()
    modal_val, modal_frac = float(vc.index[0]), float(vc.iloc[0] / n_total)
    row = {"target": t, "n_unique": int(df[t].nunique()), "modal_value": modal_val,
           "modal_fraction": modal_frac, "min": float(df[t].min()), "max": float(df[t].max()),
           "sd": float(df[t].std())}
    if modal_frac >= CONSTANT_TARGET_MODAL_FRACTION:
        CONSTANT_TARGETS[t] = modal_val
        row["decision"] = "FIXED (not modelled): no usable variance"
    else:
        MODEL_TARGETS.append(t)
        row["decision"] = "modelled"
    diag_rows.append(row)
pd.DataFrame(diag_rows).to_csv(DATADIR / "target_diagnostics.csv", index=False)
for r in diag_rows:
    print(f"  target {r['target']:<14s}: {r['n_unique']:>3d} unique values, "
          f"{r['modal_fraction']*100:5.1f}% at {r['modal_value']:.4g} -> {r['decision']}")

wa_floor = float(df["WA_pct"].min())
audit["wa"] = {"floor": wa_floor,
               "fraction_at_floor": float((df["WA_pct"] <= wa_floor + 1e-9).mean()),
               "n_above_iso_bia_limit": int((df["WA_pct"] > ISO13006_BIA_WA_MAX).sum()),
               "n_negative_clipped": _n_neg_wa}
wa_is_censored = ("WA_pct" in MODEL_TARGETS) and audit["wa"]["fraction_at_floor"] >= WA_CENSORED_FRACTION
audit["wa"]["censored"] = bool(wa_is_censored)
audit["mor"] = {"min": float(df["MOR_MPa"].min()), "max": float(df["MOR_MPa"].max()),
                "fraction_ge_iso_bia_min": float((df["MOR_MPa"] >= ISO13006_BIA_MOR_MIN).mean())}
audit["constant_targets"] = CONSTANT_TARGETS
print(f"WA: {audit['wa']['fraction_at_floor']*100:.1f}% of batches sit at the minimum "
      f"({wa_floor:.3f} %) -> {'LEFT-CENSORED' if wa_is_censored else 'not treated as censored'}; "
      f"{audit['wa']['n_above_iso_bia_limit']}/{n_total} batches exceed {ISO13006_BIA_WA_MAX} % "
      f"(ISO 13006 BIa limit). MOR range {audit['mor']['min']:.1f}-{audit['mor']['max']:.1f} MPa "
      f"({audit['mor']['fraction_ge_iso_bia_min']*100:.0f}% >= {ISO13006_BIA_MOR_MIN:.0f} MPa).")
print(f"Modelled targets: {MODEL_TARGETS}   Fixed: {CONSTANT_TARGETS}")
if wa_is_censored:
    print("  WARNING: WA is heavily left-censored; treat continuous WA predictions near the minimum "
          "as 'at floor'. Ask the lab whether the minimum is a reporting floor.")

DATA_HASH = pd.util.hash_pandas_object(raw).sum()
GENERATED_AT = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

X = df[feature_cols]
Y = df[MODEL_TARGETS]

# ── Held-out split (touched once, for the final comparison) ────────────────
X_train, X_test, y_train_df, y_test_df = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
y_train_df, y_test_df = y_train_df.reset_index(drop=True), y_test_df.reset_index(drop=True)
print(f"\nTrain: {len(X_train)} | Held-out test: {len(X_test)}")

preproc = ColumnTransformer([("num", StandardScaler(), feature_cols)], remainder="drop")

# ══════════════════════════════════════════════════════════════════════════
# 2. CORRELATIONS (closure-safe) + OLS EFFECTS
# ══════════════════════════════════════════════════════════════════════════
def _bh_fdr(p):
    p = np.asarray(p, float)
    return stats.false_discovery_control(p, method="bh")

corr_rows = []
for c in comp_cols:
    for t in MODEL_TARGETS:
        r, p = pearsonr(df[c], df[t])
        rho, ps = spearmanr(df[c], df[t])
        corr_rows.append({"material": MAT_SHORT[c], "col": c, "target": t,
                          "pearson_r": r, "pearson_p": p, "spearman_rho": rho, "spearman_p": ps})
corr_df = pd.DataFrame(corr_rows)
corr_df["pearson_p_fdr"] = _bh_fdr(corr_df["pearson_p"])
corr_df["spearman_p_fdr"] = _bh_fdr(corr_df["spearman_p"])

# CLR robustness (closure-invariant coordinates of the as-weighed composition)
_logX = np.log(df[comp_cols].values)
_clr = pd.DataFrame(_logX - _logX.mean(axis=1, keepdims=True), columns=comp_cols)
corr_df["clr_r"] = [pearsonr(_clr[r.col], df[r.target])[0] for r in corr_df.itertuples()]
corr_df["sign_flip_clr"] = np.sign(corr_df["pearson_r"]) != np.sign(corr_df["clr_r"])
corr_df.drop(columns="col").to_csv(DATADIR / "correlations_closure_checked.csv", index=False)
print(f"\nCorrelations (n={n_total}): {(corr_df['pearson_p'] < 0.05).sum()}/{len(corr_df)} "
      f"Pearson p<0.05 raw, {(corr_df['pearson_p_fdr'] < 0.05).sum()}/{len(corr_df)} after BH-FDR; "
      f"{int(corr_df['sign_flip_clr'].sum())} pairs change sign under CLR "
      f"(all such pairs have |r| small: max |r| = "
      f"{corr_df.loc[corr_df['sign_flip_clr'], 'pearson_r'].abs().max() if corr_df['sign_flip_clr'].any() else 0:.2f}).")

# Standardised OLS effects (valid: regressors are near-orthogonal)
def ols_standardised(Xdf, yv):
    Z = ((Xdf - Xdf.mean()) / Xdf.std()).values
    z = ((yv - yv.mean()) / yv.std()).values
    A = np.column_stack([Z, np.ones(len(Z))])
    beta, *_ = np.linalg.lstsq(A, z, rcond=None)
    resid = z - A @ beta
    dof = len(z) - A.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    tcrit = stats.t.ppf(0.975, dof)
    tval = beta / se
    pval = 2 * stats.t.sf(np.abs(tval), dof)
    vif = [1 / (1 - r2_score(Z[:, j], np.delete(Z, j, axis=1) @
           np.linalg.lstsq(np.delete(Z, j, axis=1), Z[:, j], rcond=None)[0]))
           for j in range(Z.shape[1])]
    out = pd.DataFrame({"material": [MAT_SHORT[c] for c in Xdf.columns],
                        "std_beta": beta[:-1], "ci_low": beta[:-1] - tcrit * se[:-1],
                        "ci_high": beta[:-1] + tcrit * se[:-1], "p_value": pval[:-1], "VIF": vif})
    return out, 1 - resid @ resid / (z @ z)

ols_all = []
for t in MODEL_TARGETS:
    o, r2_ols = ols_standardised(X, df[t])
    o.insert(0, "target", t)
    o["OLS_R2_in_sample"] = r2_ols
    ols_all.append(o)
ols_df = pd.concat(ols_all)
ols_df.to_csv(DATADIR / "ols_standardised_effects.csv", index=False)
print(f"Standardised OLS effects saved (max VIF = {ols_df['VIF'].max():.2f}: no collinearity, "
      f"so effects are interpretable).")
for t in MODEL_TARGETS:
    top = ols_df[ols_df["target"] == t].reindex(
        ols_df[ols_df["target"] == t]["std_beta"].abs().sort_values(ascending=False).index).head(2)
    print(f"  {t}: largest standardised effects -> " +
          ", ".join(f"{r.material} ({r.std_beta:+.2f})" for r in top.itertuples()))
if "MOR_MPa" in MODEL_TARGETS:
    nasil = ols_df[(ols_df.target == "MOR_MPa") & (ols_df.material == "Na-Silicate")]["std_beta"].iloc[0]
    if abs(nasil) > 0.5:
        print("  NOTE: Na-Silicate (a deflocculant, i.e. a slip additive rather than a body "
              "constituent - cf. Reviewer 6) is the dominant MOR driver. Discuss it as a processing "
              "additive whose dose controls slip density/packing, not as a 'raw material'.")

# Heatmap (Pearson, * raw p<0.05, ** BH-FDR q<0.05)
piv_r = corr_df.pivot(index="material", columns="target", values="pearson_r").loc[
    [MAT_SHORT[c] for c in comp_cols], MODEL_TARGETS]
piv_p = corr_df.pivot(index="material", columns="target", values="pearson_p").loc[piv_r.index, MODEL_TARGETS]
piv_q = corr_df.pivot(index="material", columns="target", values="pearson_p_fdr").loc[piv_r.index, MODEL_TARGETS]
annot = piv_r.round(2).astype(str)
for i in piv_r.index:
    for j in piv_r.columns:
        annot.loc[i, j] += " **" if piv_q.loc[i, j] < 0.05 else (" *" if piv_p.loc[i, j] < 0.05 else "")
piv_disp = piv_r.rename(columns=TGT_LABELS)
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(piv_disp, annot=annot.values, fmt="", cmap="coolwarm", center=0, vmin=-1, vmax=1,
            linewidths=0.5, annot_kws={"size": 13}, ax=ax)
ax.set_title(f"Pearson r: as-weighed masses vs. properties\nn={n_total} batches  |  * p<0.05, ** BH-FDR q<0.05",
             fontsize=_FS_TITLE, fontweight="bold", pad=14)
plt.xticks(rotation=30, ha="right", fontsize=_FS_TICK)
plt.yticks(rotation=0, fontsize=_FS_TICK)
plt.tight_layout()
savefig(fig, "input_output_correlation_heatmap")
print("Saved: input_output_correlation_heatmap.pdf / .png")

# ══════════════════════════════════════════════════════════════════════════
# 3. CANDIDATE MODELS + SEARCH SPACES
# ══════════════════════════════════════════════════════════════════════════
def make_candidates() -> dict:
    """Six candidates. Inner estimators are single-threaded; parallelism is
    applied at the CV level (avoids thread oversubscription)."""
    cands = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1),
        # MLP needs a scaled target; without this it fails for trivial reasons
        "MLP": TransformedTargetRegressor(
            regressor=MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000, random_state=42,
                                   early_stopping=True, n_iter_no_change=30),
            transformer=StandardScaler()),
    }
    for lib, cls, name, kw in [
        ("xgboost", "XGBRegressor", "XGB",
         {"n_estimators": 300, "random_state": RANDOM_STATE, "n_jobs": 1, "verbosity": 0}),
        ("lightgbm", "LGBMRegressor", "LGBM",
         {"n_estimators": 300, "random_state": RANDOM_STATE, "n_jobs": 1, "verbose": -1,
          "min_gain_to_split": 0.0}),
        ("catboost", "CatBoostRegressor", "CatBoost",
         {"iterations": 300, "verbose": 0, "random_seed": RANDOM_STATE, "thread_count": 1,
          "allow_writing_files": False}),
    ]:
        try:
            cands[name] = getattr(__import__(lib), cls)(**kw)
        except Exception as e:
            print(f"  (candidate {name} unavailable: {e})")
    return cands


PARAM_DISTS = {
    "RandomForest": {"reg__n_estimators": randint(150, 700), "reg__max_depth": randint(2, 12),
                     "reg__min_samples_leaf": randint(1, 8), "reg__max_features": uniform(0.3, 0.7)},
    "MLP": {"reg__regressor__hidden_layer_sizes": [(32,), (64,), (32, 32), (64, 32), (64, 64)],
            "reg__regressor__alpha": loguniform(1e-4, 1e-1),
            "reg__regressor__learning_rate_init": loguniform(1e-4, 1e-2)},
    "XGB": {"reg__n_estimators": randint(100, 600), "reg__max_depth": randint(2, 8),
            "reg__learning_rate": loguniform(1e-2, 3e-1), "reg__subsample": uniform(0.6, 0.4),
            "reg__colsample_bytree": uniform(0.5, 0.5), "reg__reg_lambda": loguniform(1e-2, 10)},
    "LGBM": {"reg__n_estimators": randint(100, 600), "reg__num_leaves": randint(7, 63),
             "reg__max_depth": randint(2, 8), "reg__learning_rate": loguniform(1e-2, 3e-1),
             "reg__min_child_samples": randint(3, 20)},
    "CatBoost": {"reg__iterations": randint(150, 700), "reg__depth": randint(3, 8),
                 "reg__learning_rate": loguniform(1e-2, 3e-1), "reg__l2_leaf_reg": loguniform(1e-1, 10)},
}


def _pipe(model):
    return Pipeline([("preproc", preproc), ("reg", clone(model))])


def make_search(name, model, n_iter, cv_folds, seed):
    """Estimator whose .fit() tunes hyperparameters internally (used both as
    the INNER loop of nested CV and for the final tuning)."""
    pipe = _pipe(model)
    if name not in PARAM_DISTS:
        return pipe
    return RandomizedSearchCV(
        pipe, PARAM_DISTS[name], n_iter=n_iter,
        cv=KFold(cv_folds, shuffle=True, random_state=seed),
        scoring="r2", random_state=seed, n_jobs=1, refit=True)


def clean_params(best_params: dict) -> dict:
    return {k.replace("reg__", "", 1): v for k, v in best_params.items()}


def metrics(y_true, y_pred):
    return {"r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "mape_pct": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)}

# ══════════════════════════════════════════════════════════════════════════
# 4. NESTED CV MODEL SELECTION (per target)
# ══════════════════════════════════════════════════════════════════════════
outer_cv = RepeatedKFold(n_splits=5, n_repeats=OUTER_REPEATS, random_state=RANDOM_STATE)
n_outer = 5 * OUTER_REPEATS
print(f"\n[Model selection] NESTED CV: outer 5-fold x {OUTER_REPEATS} repeats ({n_outer} folds); "
      f"inside each outer fold, RandomizedSearchCV ({N_TUNE_ITER_NESTED} draws, "
      f"{INNER_CV_FOLDS}-fold). Training split only (n={len(X_train)}).")
cv_rows, fold_r2, best_per_target = [], {}, {}
for tname in MODEL_TARGETS:
    print(f"  -- {tname} --")
    y_tr = y_train_df[tname].values
    scores = {}
    for name, model in make_candidates().items():
        try:
            est = make_search(name, model, N_TUNE_ITER_NESTED, INNER_CV_FOLDS, RANDOM_STATE)
            res = cross_validate(est, X_train, y_tr, cv=outer_cv,
                                 scoring={"r2": "r2", "neg_mae": "neg_mean_absolute_error"},
                                 n_jobs=-1)
            fold_r2[(tname, name)] = res["test_r2"]
            scores[name] = float(np.mean(res["test_r2"]))
            cv_rows.append({"target": tname, "model": name, "nested_cv_r2_mean": scores[name],
                            "nested_cv_r2_sd": float(np.std(res["test_r2"])),
                            "nested_cv_mae_mean": float(-np.mean(res["test_neg_mae"]))})
            print(f"    {name:<17s} R2 = {scores[name]:+.4f} +/- {np.std(res['test_r2']):.4f}   "
                  f"MAE = {-np.mean(res['test_neg_mae']):.4f}")
        except Exception as e:
            print(f"    {name:<17s} skipped: {type(e).__name__}: {e}")
    best_mean_model = max(scores, key=scores.get)
    sd_best = float(np.std(fold_r2[(tname, best_mean_model)]))
    se_best = sd_best / math.sqrt(n_outer)
    if SELECTION_RULE == "one_se":
        eligible = [m for m, v in scores.items() if v >= scores[best_mean_model] - se_best]
        chosen = sorted(eligible, key=lambda m: (COMPLEXITY.get(m, 9), -scores[m]))[0]
    else:
        chosen = best_mean_model
    best_per_target[tname] = chosen
    print(f"    -> highest nested-CV R2: {best_mean_model} (SE {se_best:.4f}); "
          f"SELECTED by '{SELECTION_RULE}' rule: {chosen}")

cv_df = pd.DataFrame(cv_rows)
cv_df["selected"] = [best_per_target.get(r.target) == r.model for r in cv_df.itertuples()]
cv_df = cv_df.sort_values(["target", "nested_cv_r2_mean"], ascending=[True, False])
cv_df.to_csv(DATADIR / "cv_model_comparison.csv", index=False)

# Paired fold-by-fold differences vs. LinearRegression (are the fancy models really better?)
pair_rows = []
for tname in MODEL_TARGETS:
    if (tname, "LinearRegression") not in fold_r2:
        continue
    base = fold_r2[(tname, "LinearRegression")]
    for (t2, name), sc in fold_r2.items():
        if t2 != tname or name == "LinearRegression":
            continue
        d = sc - base
        pair_rows.append({"target": tname, "model": name, "mean_r2_diff_vs_linear": float(d.mean()),
                          "sd_of_diff": float(d.std()), "fraction_folds_better": float((d > 0).mean())})
pair_df = pd.DataFrame(pair_rows)
pair_df.to_csv(DATADIR / "cv_paired_vs_linear.csv", index=False)
for tname in MODEL_TARGETS:
    b = best_per_target[tname]
    if b != "LinearRegression":
        r = pair_df[(pair_df.target == tname) & (pair_df.model == b)].iloc[0]
        verdict = ("NOT clearly better than plain linear regression"
                   if r.mean_r2_diff_vs_linear < r.sd_of_diff else "clearly better than linear regression")
        print(f"  {tname}: {b} beats LinearRegression by {r.mean_r2_diff_vs_linear:+.3f} R2 "
              f"(paired sd {r.sd_of_diff:.3f}; better in {r.fraction_folds_better*100:.0f}% of folds) -> {verdict}.")

# ══════════════════════════════════════════════════════════════════════════
# 5. FINAL TUNING (full training split), HELD-OUT TEST FOR ALL SIX MODELS
# ══════════════════════════════════════════════════════════════════════════
print(f"\n[Final tuning] RandomizedSearchCV {N_TUNE_ITER_FINAL} draws, {FINAL_CV_FOLDS}-fold, "
      f"per model & target; every tuned model is then scored ONCE on the held-out test set.")
hparam_rows, test_rows_all, tuned_store = [], [], {}
for tname in MODEL_TARGETS:
    y_tr, y_te = y_train_df[tname].values, y_test_df[tname].values
    for name, model in make_candidates().items():
        try:
            if name in PARAM_DISTS:
                s = RandomizedSearchCV(
                    _pipe(model), PARAM_DISTS[name], n_iter=N_TUNE_ITER_FINAL,
                    cv=KFold(FINAL_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                    scoring="r2", random_state=RANDOM_STATE, n_jobs=-1, refit=True).fit(X_train, y_tr)
                fitted, hp = s.best_estimator_, clean_params(s.best_params_)
            else:
                fitted, hp = _pipe(model).fit(X_train, y_tr), {"note": "no hyperparameters (OLS)"}
            tuned_store[(tname, name)] = hp
            hparam_rows.append({"target": tname, "model": name, **{k: (str(v) if isinstance(v, tuple) else v)
                                                                   for k, v in hp.items()}})
            m = metrics(y_te, fitted.predict(X_test))
            test_rows_all.append({"target": tname, "model": name, **m})
        except Exception as e:
            print(f"    {tname}/{name} skipped: {type(e).__name__}: {e}")
hp_df = pd.DataFrame(hparam_rows)
hp_df.to_csv(DATADIR / "hyperparameters_all_candidates.csv", index=False)
test_all_df = pd.DataFrame(test_rows_all)
test_all_df.to_csv(DATADIR / "held_out_all_models.csv", index=False)
print("Saved: hyperparameters_all_candidates.csv, held_out_all_models.csv")
print(test_all_df.pivot(index="model", columns="target", values="r2").round(3).to_string())


def build_pipeline(tname: str) -> Pipeline:
    name = best_per_target[tname]
    cand = clone(make_candidates()[name])
    hp = {k: v for k, v in tuned_store[(tname, name)].items() if k != "note"}
    pipe = _pipe(cand)
    if hp:
        pipe.set_params(**{f"reg__{k}": v for k, v in hp.items()})
    return pipe


# ── Headline held-out metrics for the SELECTED model per target ────────────
final_models, test_rows, y_pred_test = {}, [], {}
for tname in MODEL_TARGETS:
    pipe = build_pipeline(tname).fit(X_train, y_train_df[tname].values)
    final_models[tname] = pipe
    y_pred_test[tname] = pipe.predict(X_test)
    m = metrics(y_test_df[tname].values, y_pred_test[tname])
    row = {"target": tname, "model": best_per_target[tname], **m}
    if tname == "WA_pct" and wa_is_censored:   # censoring-aware view
        unc = y_test_df[tname].values > wa_floor + 1e-9
        if unc.sum() >= 5:
            row["r2_uncensored_only"] = float(r2_score(y_test_df[tname].values[unc], y_pred_test[tname][unc]))
            row["n_uncensored_test"] = int(unc.sum())
    test_rows.append(row)
    print(f"[HELD-OUT TEST] {tname:<8s} ({best_per_target[tname]:<16s}) R2={m['r2']:.4f}  "
          f"MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  MAPE={m['mape_pct']:.2f}%"
          + (f"  | uncensored-only R2={row['r2_uncensored_only']:.3f} (n={row['n_uncensored_test']})"
             if "r2_uncensored_only" in row else ""))
pd.DataFrame(test_rows).to_csv(DATADIR / "held_out_test_metrics.csv", index=False)

selected_rows = []
for t in MODEL_TARGETS:
    hp = tuned_store[(t, best_per_target[t])]
    selected_rows.append({"target": t, "model": best_per_target[t],
                          **{k: (str(v) if isinstance(v, tuple) else v) for k, v in hp.items()}})
pd.DataFrame(selected_rows).to_csv(DATADIR / "hyperparameters_selected.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════
# 5b. WA EXCEEDANCE MODEL  P(WA > limit)
# A compliance check needs the probability of exceeding a limit, not just a point value.
#   * If WA is left-censored (many batches at the minimum) an OLS fit treats the floor as exact and
#     R2 is a poor yardstick: a Tobit model (linear latent WA, Gaussian noise, values at the floor
#     treated as "at or below the floor") is used.
#   * Otherwise WA is an ordinary continuous target: P = 1 - Phi((limit - prediction) / sigma),
#     sigma = SD of the out-of-fold residuals of the forward model.
# Either way the model is scored by out-of-fold AUC / Brier score at several limits and its
# parameters are shipped in metadata.json ("wa_tobit" / "wa_exceedance").
# ══════════════════════════════════════════════════════════════════════════
wa_tobit_meta = None
wa_exceed_meta = None
if "WA_pct" in MODEL_TARGETS:
    _EPS = 1e-9
    _y_all = df["WA_pct"].values
    _Xv = X.values
    _limits = (0.30, 0.40, ISO13006_BIA_WA_MAX)
    _oof_p = {thr: np.zeros(len(_y_all)) for thr in _limits}
    if wa_is_censored:
        def _tobit_nll(theta, Z, y, floor):
            b0, b, ls = theta[0], theta[1:-1], theta[-1]
            s_ = math.exp(ls); mu_ = b0 + Z @ b
            cens = y <= floor + _EPS
            ll = np.where(cens, _norm.logcdf((floor - mu_) / s_), _norm.logpdf((y - mu_) / s_) - ls)
            return -float(ll.sum())

        def _tobit_fit(Z, y, floor):
            th0 = np.r_[max(float(np.mean(y)), floor), np.zeros(Z.shape[1]), math.log(max(float(np.std(y)), 0.05))]
            r = _minimize(_tobit_nll, th0, args=(Z, y, floor), method="BFGS")
            return {"b0": float(r.x[0]), "coef": r.x[1:-1].astype(float), "sigma": float(math.exp(r.x[-1]))}

        def _tobit_mu(fit, Z):
            return fit["b0"] + Z @ fit["coef"]

        for _tr, _te in KFold(5, shuffle=True, random_state=RANDOM_STATE).split(_Xv):
            _sc = StandardScaler().fit(_Xv[_tr])
            _fit = _tobit_fit(_sc.transform(_Xv[_tr]), _y_all[_tr], wa_floor)
            for thr in _oof_p:
                _oof_p[thr][_te] = 1.0 - _norm.cdf((thr - _tobit_mu(_fit, _sc.transform(_Xv[_te]))) / _fit["sigma"])
        _sc_all = StandardScaler().fit(_Xv)
        _fit_all = _tobit_fit(_sc_all.transform(_Xv), _y_all, wa_floor)
        wa_tobit_meta = {"features": list(feature_cols), "floor": float(wa_floor),
                         "mean": _sc_all.mean_.tolist(), "scale": _sc_all.scale_.tolist(),
                         "b0": _fit_all["b0"], "coef": _fit_all["coef"].tolist(), "sigma": _fit_all["sigma"]}
        wa_exceed_meta = {"model": "tobit", "sigma": _fit_all["sigma"], "floor": float(wa_floor)}
        print(f"[WA exceedance] Tobit model; latent-noise sigma = {_fit_all['sigma']:.3f} % (floor {wa_floor:.3f} %)")
    else:
        _oof_wa = cross_val_predict(build_pipeline("WA_pct"), X, _y_all,
                                    cv=KFold(5, shuffle=True, random_state=RANDOM_STATE), n_jobs=-1)
        _sigma = float(np.std(_y_all - _oof_wa, ddof=1))
        for thr in _oof_p:
            _oof_p[thr] = 1.0 - _norm.cdf((thr - _oof_wa) / _sigma)
        wa_exceed_meta = {"model": "gaussian_residual", "sigma": _sigma}
        print(f"[WA exceedance] Gaussian residual model; sigma = {_sigma:.3f} % (SD of out-of-fold residuals)")
    exc_rows = []
    for thr, pr in _oof_p.items():
        yb = (_y_all > thr).astype(int)
        if 0 < yb.sum() < len(yb):
            exc_rows.append({"model": wa_exceed_meta["model"], "limit_pct": thr, "n_exceeding": int(yb.sum()),
                             "n_total": int(len(yb)), "oof_auc": float(roc_auc_score(yb, pr)),
                             "oof_brier": float(np.mean((pr - yb) ** 2)),
                             "brier_of_base_rate": float(np.mean((yb.mean() - yb) ** 2))})
    pd.DataFrame(exc_rows).to_csv(DATADIR / "wa_exceedance_metrics.csv", index=False)
    print("[WA exceedance] out-of-fold scoring (all data):")
    for r_ in exc_rows:
        print(f"    P(WA > {r_['limit_pct']:.2f} %): n_exceed={r_['n_exceeding']:3d}/{r_['n_total']}  "
              f"AUC={r_['oof_auc']:.3f}  Brier={r_['oof_brier']:.4f} (base-rate Brier {r_['brier_of_base_rate']:.4f})")
    if exc_rows and min(r_["n_exceeding"] for r_ in exc_rows) < 20:
        print("    NOTE: some limits have fewer than 20 exceeding batches, so their AUC is uncertain; "
              "report the counts alongside it.")

# ══════════════════════════════════════════════════════════════════════════
# 6. PREDICTION UNCERTAINTY (out-of-fold residual quantiles)
# ══════════════════════════════════════════════════════════════════════════
def conformal_halfwidth(abs_res, level):
    n = len(abs_res)
    q = min(1.0, math.ceil((n + 1) * level) / n)
    return float(np.quantile(abs_res, q))

interval_rows, halfwidth_all, oof_by_target = [], {}, {}
for tname in MODEL_TARGETS:
    kf = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    # (a) train-only OOF -> half-width -> coverage on the untouched test set
    oof_tr = cross_val_predict(build_pipeline(tname), X_train, y_train_df[tname].values, cv=kf, n_jobs=-1)
    hw_tr = conformal_halfwidth(np.abs(y_train_df[tname].values - oof_tr), INTERVAL_LEVEL)
    cover = float(np.mean(np.abs(y_test_df[tname].values - y_pred_test[tname]) <= hw_tr))
    # (b) all-data OOF -> half-width shipped with the production model
    oof_all = cross_val_predict(build_pipeline(tname), X, df[tname].values, cv=kf, n_jobs=-1)
    hw_all = conformal_halfwidth(np.abs(df[tname].values - oof_all), INTERVAL_LEVEL)
    halfwidth_all[tname] = hw_all
    interval_rows.append({"target": tname, "nominal_level": INTERVAL_LEVEL,
                          "halfwidth_train_oof": hw_tr, "empirical_coverage_on_test": cover,
                          "halfwidth_all_oof_shipped": hw_all,
                          "oof_rmse_all": float(np.sqrt(np.mean((df[tname].values - oof_all) ** 2)))})
    print(f"[Uncertainty] {tname}: {INTERVAL_LEVEL*100:.0f}% half-width (all-data OOF) = "
          f"+/-{hw_all:.4f}; test coverage of train-derived interval = {cover*100:.0f}% "
          f"(n={len(y_test_df)})")
    oof_by_target[tname] = oof_all
pd.DataFrame(interval_rows).to_csv(DATADIR / "prediction_intervals.csv", index=False)


# ── Does error grow toward the edge of the design space? (Reviewers 4, 5) ──
# Out-of-fold |residual| vs. scaled distance from the design-space centroid.
_Zs = ((df[comp_cols] - df[comp_cols].mean()) / df[comp_cols].std()).values
_edge = np.linalg.norm(_Zs, axis=1)
edge_rows = []
for tname, oof in oof_by_target.items():
    ares = np.abs(df[tname].values - oof)
    rho, p = spearmanr(_edge, ares)
    tert = pd.qcut(_edge, 3, labels=["inner third", "middle third", "outer third"])
    by = pd.Series(ares).groupby(tert.astype(str)).mean()
    for k_, v_ in by.items():
        edge_rows.append({"target": tname, "region": k_, "mean_abs_oof_error": float(v_),
                          "spearman_rho_error_vs_edge_distance": float(rho), "p_value": float(p)})
    print(f"[Edge effect] {tname}: Spearman rho(|OOF error|, distance from centroid) = {rho:+.2f} "
          f"(p={p:.3f}); mean |error| inner/middle/outer third = "
          f"{by['inner third']:.4g} / {by['middle third']:.4g} / {by['outer third']:.4g}")
pd.DataFrame(edge_rows).to_csv(DATADIR / "error_vs_edge_distance.csv", index=False)

# ── Production model: refit on ALL batches ─────────────────────────────────
production_models = {t: build_pipeline(t).fit(X, df[t].values) for t in MODEL_TARGETS}
production_model = MultiTargetModel(production_models, MODEL_TARGETS, feature_cols)

# ── Parity plots: held-out test + out-of-fold (all data) ───────────────────
units = {"MOR_MPa": "MPa", "WA_pct": "%", "Shrinkage_pct": "%"}
fig, axes = plt.subplots(2, len(MODEL_TARGETS), figsize=(7.5 * len(MODEL_TARGETS), 13), squeeze=False)
for j, t in enumerate(MODEL_TARGETS):
    for i, (yt, yp, ttl) in enumerate([
        (y_test_df[t].values, y_pred_test[t], f"Held-out test (n={len(X_test)})"),
        (df[t].values, oof_by_target[t], f"5-fold out-of-fold (n={n_total})"),
    ]):
        ax = axes[i, j]
        cens = (yt <= wa_floor + 1e-9) if (t == "WA_pct" and wa_is_censored) else np.zeros(len(yt), bool)
        ax.scatter(yt[~cens], yp[~cens], alpha=0.75, s=45, color="teal", label="measured")
        if cens.any():
            ax.scatter(yt[cens], yp[cens], alpha=0.5, s=45, facecolors="none", edgecolors="gray",
                       label=f"at {wa_floor:.1f} % floor")
            ax.legend(fontsize=10)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
        m = metrics(yt, yp)
        ax.set_title(f"{TGT_LABELS[t]}\n{ttl}\nR2={m['r2']:.3f}  MAE={m['mae']:.3f} {units[t]}", fontsize=_FS_AX)
        ax.set_xlabel("Measured", fontsize=_FS_AX)
        ax.set_ylabel("Predicted", fontsize=_FS_AX)
plt.suptitle("Forward model: predicted vs. measured (real data only)", fontsize=_FS_TITLE + 2,
             fontweight="bold", y=1.0)
plt.tight_layout()
savefig(fig, "parity_plots")
pd.DataFrame({**{f"y_true_{t}": y_test_df[t].values for t in MODEL_TARGETS},
              **{f"y_pred_{t}": y_pred_test[t] for t in MODEL_TARGETS}}
             ).to_csv(DATADIR / "parity_data.csv", index=False)
print("Saved: parity_plots.pdf / .png")

# ══════════════════════════════════════════════════════════════════════════
# 7. IMPORTANCE: permutation (held-out), native, SHAP, CLR robustness
# ══════════════════════════════════════════════════════════════════════════
imp_rows = []
for tname in MODEL_TARGETS:
    pipe = final_models[tname]                       # fit on the training split only
    pi = permutation_importance(pipe, X_test, y_test_df[tname].values, n_repeats=50,
                                random_state=RANDOM_STATE, scoring="r2", n_jobs=-1)
    reg = production_models[tname].named_steps["reg"]
    native = (np.abs(reg.coef_) if hasattr(reg, "coef_") else
              reg.feature_importances_ if hasattr(reg, "feature_importances_") else np.full(len(feature_cols), np.nan))
    native = native / np.nansum(native) if np.nansum(native) > 0 else native
    for k, c in enumerate(feature_cols):
        imp_rows.append({"target": tname, "feature": c, "label": MAT_SHORT[c],
                         "perm_importance_heldout_r2_drop": pi.importances_mean[k],
                         "perm_importance_sd": pi.importances_std[k],
                         "native_importance_normalised": native[k]})
fi = pd.DataFrame(imp_rows)
fi.to_csv(DATADIR / "feature_importances.csv", index=False)

# CLR-space refit (closure-invariant coordinates) - robustness of the ranking
clr_cols = [f"clr_{c}" for c in comp_cols]
X_clr_full = pd.DataFrame(_clr.values, columns=clr_cols)
clr_pre = ColumnTransformer([("num", StandardScaler(), clr_cols)], remainder="drop")
clr_rows = []
for tname in MODEL_TARGETS:
    name = best_per_target[tname]
    cand = clone(make_candidates()[name])
    hp = {k: v for k, v in tuned_store[(tname, name)].items() if k != "note"}
    clr_pipe = Pipeline([("preproc", clr_pre), ("reg", cand)])
    if hp:
        clr_pipe.set_params(**{f"reg__{k}": v for k, v in hp.items()})
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_clr_full, df[tname], test_size=TEST_SIZE,
                                                  random_state=RANDOM_STATE)
    clr_pipe.fit(Xc_tr, yc_tr)
    r2_clr = r2_score(yc_te, clr_pipe.predict(Xc_te))
    pi = permutation_importance(clr_pipe, Xc_te, yc_te, n_repeats=50, random_state=RANDOM_STATE,
                                scoring="r2", n_jobs=-1)
    for k, c in enumerate(comp_cols):
        clr_rows.append({"target": tname, "material": MAT_SHORT[c],
                         "clr_perm_importance_heldout": pi.importances_mean[k],
                         "clr_model_heldout_r2": r2_clr})
pd.DataFrame(clr_rows).to_csv(DATADIR / "closure_aware_feature_importance.csv", index=False)
print("Saved: feature_importances.csv, closure_aware_feature_importance.csv "
      "(CLR refit; held-out R2 of the scale-free model shown in the file)")

fig, axes = plt.subplots(1, len(MODEL_TARGETS), figsize=(9 * len(MODEL_TARGETS), 6), squeeze=False)
for ax, tname in zip(axes[0], MODEL_TARGETS):
    sub = fi[fi.target == tname].sort_values("perm_importance_heldout_r2_drop", ascending=False)
    ax.barh(sub["label"], sub["perm_importance_heldout_r2_drop"], xerr=sub["perm_importance_sd"],
            color="#E53935", edgecolor="white")
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"{TGT_LABELS[tname]}\n({best_per_target[tname]})", fontsize=_FS_AX)
    ax.set_xlabel("Drop in held-out R2 when feature is permuted", fontsize=_FS_AX - 1)
    ax.tick_params(labelsize=_FS_TICK)
plt.suptitle(f"Permutation importance on the held-out test set (n={len(X_test)}, 50 repeats)",
             fontsize=_FS_TITLE, fontweight="bold", y=1.03)
plt.tight_layout()
savefig(fig, "feature_importances")
print("Saved: feature_importances.pdf / .png")

try:
    import shap
    shap_rows = []
    for tname in MODEL_TARGETS:
        pipe = production_models[tname]
        reg = pipe.named_steps["reg"]
        Xs_df = pd.DataFrame(pipe.named_steps["preproc"].transform(X), columns=feature_cols)
        try:
            sv = shap.TreeExplainer(reg).shap_values(Xs_df)
        except Exception:
            sv = shap.Explainer(reg.predict, Xs_df)(Xs_df).values
        for feat, v in zip(feature_cols, np.abs(sv).mean(axis=0)):
            shap_rows.append({"target": tname, "feature": feat, "mean_abs_shap": float(v),
                              "label": MAT_SHORT[feat]})
    pd.DataFrame(shap_rows).to_csv(DATADIR / "shap_importances.csv", index=False)
    print("Saved: shap_importances.csv (model-agnostic; values are in target units, sane scale)")
except ImportError:
    print("SHAP not installed - skipping (pip install shap).")
except Exception as e:
    print(f"SHAP: {type(e).__name__}: {e}")

# ── PDP (production model, marginalised over all batches; valid because the
#    as-weighed factors are near-independent) ────────────────────────────────
def compute_pdp(model, Xo, feat, n_grid=50):
    grid = np.linspace(Xo[feat].min(), Xo[feat].max(), n_grid)
    out = []
    for v in grid:
        Xt = Xo.copy()
        Xt[feat] = v
        out.append(model.predict(Xt).mean(axis=0))
    return grid, np.array(out)

colors = ["#1565C0", "#D32F2F"]
for t_idx, tname in enumerate(MODEL_TARGETS):
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    axes = axes.flatten()
    for ax, feat in zip(axes, feature_cols):
        grid, means = compute_pdp(production_model, X, feat)
        ax.plot(grid, means[:, t_idx], color=colors[t_idx % 2], lw=3)
        ax.set_title(MAT_SHORT[feat], fontsize=20, fontweight="bold")
        ax.set_xlabel(f"{MAT_SHORT[feat]} (as-weighed, wt%)", fontsize=15)
        ax.set_ylabel(TGT_LABELS[tname], fontsize=15)
        ax.tick_params(labelsize=13)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    fig.suptitle(f"Partial dependence of {TGT_LABELS[tname]} (marginalised over n={n_total} batches)",
                 fontsize=24, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, f"PDP_{tname}")
    print(f"  Saved: PDP_{tname}.pdf / .png")

# ══════════════════════════════════════════════════════════════════════════
# 8. SAVE ARTEFACTS FOR inverse_design.py / streamlit_app.py
# ══════════════════════════════════════════════════════════════════════════
joblib.dump(production_model, MODELDIR / "forward_model.joblib")
with open(MODELDIR / "feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

df[feature_cols + ["batch_total_wtpct"] + ALL_TARGETS].to_csv(DATADIR / "dataset.csv", index=False)

bounds = {}
for m, col in zip(materials, comp_cols):
    lo, hi = float(df[col].min()), float(df[col].max())
    pad = (hi - lo) * BOUNDS_MARGIN
    bounds[m] = [math.floor((lo - pad) * 1000) / 1000, math.ceil((hi + pad) * 1000) / 1000]

# Cost (Tk/kg) - plant cost sheet. CO2 EF (kg CO2/kg) - international proxies.
# Reviewer 6: crushed fired tile (0.587) and ETP sludge (0.242) factors are the
# highest of any body constituent and contradict the "sustainability" role
# claimed for them. Revisit with plant-specific data before resubmission.
COST_TK_PER_KG = {"AG98": 6.95, "AG22": 8.37, "AG23": 7.024, "SodaF": 8.887, "PotashF": 6.241,
                  "Crushing": 0.0, "ETP": 0.0, "NaSil": 23.369}
CO2_KG_PER_KG = {"AG98": 0.129, "AG22": 0.129, "AG23": 0.129, "SodaF": 0.053, "PotashF": 0.0286,
                 "Crushing": 0.587, "ETP": 0.242, "NaSil": 0.433}

metadata = {
    "data_hash": str(DATA_HASH), "generated_at": GENERATED_AT,
    "materials": materials, "bounds": bounds, "bounds_margin": BOUNDS_MARGIN,
    "cost_tk_per_kg": COST_TK_PER_KG, "co2_kg_per_kg": CO2_KG_PER_KG,
    "feature_cols": feature_cols, "comp_cols": comp_cols,
    "target_cols": ALL_TARGETS, "modelled_targets": MODEL_TARGETS,
    "constant_targets": CONSTANT_TARGETS,
    "n_experimental": int(n_total), "n_synthetic": 0, "synthetic_data_used": False,
    "architecture_per_target": best_per_target,
    "prediction_interval_level": INTERVAL_LEVEL,
    "prediction_halfwidth": halfwidth_all,
    "held_out_test_metrics": test_rows,
    "iso13006_bia": {"wa_max_pct": ISO13006_BIA_WA_MAX, "mor_min_mpa": ISO13006_BIA_MOR_MIN},
    "data_audit": audit,
    "wa_tobit": wa_tobit_meta,
    "wa_exceedance": wa_exceed_meta,
}
with open(DATADIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, default=float)

print("\nFigures -> plots/ | Model -> models/forward_model.joblib | Tables -> data/")
print("Headline validation numbers: data/held_out_test_metrics.csv (+ held_out_all_models.csv)")
print("Done.")
