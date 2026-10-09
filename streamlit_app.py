#!/usr/bin/env python3
"""
streamlit_app.py
Ceramic tile inverse-design decision-support tool.
Input: target minimum MOR and maximum water absorption.
Output: recommended recipes from three methods (+ a Pareto view), with
prediction intervals, spec-compliance flags and training-domain warnings.

CHANGES vs. THE PREVIOUS REVISION (see Reviewer_Comment_Mapping.md)
  * Shrinkage removed as an input: it is constant (12.5 %) in the training data
    and is shown as a fixed property.
  * MOR is a MINIMUM spec and WA a MAXIMUM spec (one-sided), consistent with the
    optimiser. Tab text no longer claims "symmetric" penalties.
  * Every optimiser result shows 90 % prediction intervals and whether the spec
    is met (i) by the point estimate and (ii) at the pessimistic interval bound.
    Optional "safety margin" slider (sidebar) makes the optimiser require (ii).
  * Out-of-range targets and infeasible searches are reported in the UI (they
    were previously only Python warnings) (Reviewer 2, point 5).
  * Load / import failures are shown as a readable error instead of a blank page
    (Reviewer 7 could not run the app).
  * Scope-and-limitations text is generated from the metrics files, so it can
    no longer go stale (the old text said WA R^2 was negative).
  * New tab: Pareto front (NSGA-II, no weights) next to the scalarised result.

SECOND REVISION (matches the updated backend)
  * Safety margin now defaults to the backend value (k = 1): at k = 0 the optimiser sits on the
    spec boundary and a built batch would fail about half the time.
  * Bayesian results come from the production-mode optimiser (cheapest spec-compliant recipe +
    local polish). The UI reports whether the spec is met at margin k, whether the polish was
    applied, and shows the backend warning when no compliant recipe exists (e.g. a tight WA
    target combined with a large margin).
  * WA is left-censored in the data; where the censored (Tobit) model is available the UI shows
    P(WA > limit) and the out-of-fold AUC of that model.
  * Scope text generated from files: WA floor, shrinkage ceiling, process data, CO2 caveat.
  * Pareto tab: larger trial range and a warning when few compliant trials were found, because
    the front is then under-sampled.

THIRD REVISION (new, larger dataset)
  * Shrinkage varies in the new data, so it is modelled and is an OPTIONAL two-sided design target
    (target +/- tolerance, tick-box). If a dataset ever has constant shrinkage the app falls back to
    showing it as a fixed property.
  * Water absorption is no longer left-censored: the scope text and the exceedance probability
    follow metadata.json (Tobit if censored, Gaussian residual model otherwise).
  * The scope text no longer states anything about the data that is not read from the files
    (fraction of batches meeting 35 MPa, batch-total range, WA censoring, shrinkage).
"""

import json
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", page_title="Ceramic Tile Composition Designer")

BASE_DIR = Path(__file__).parent
DATADIR = BASE_DIR / "data"

# ── Robust start-up: show a readable error rather than a blank page ──────────
_missing = [str(p.relative_to(BASE_DIR)) for p in
            (DATADIR / "dataset.csv", DATADIR / "metadata.json",
             BASE_DIR / "models" / "forward_model.joblib", BASE_DIR / "models" / "feature_cols.json")
            if not p.exists()]
if _missing:
    st.error("Required files not found: " + ", ".join(_missing) +
             ". Run `python train_forward_model.py` first.")
    st.stop()

try:
    import inverse_design as idsg
    from inverse_design import (
        TARGET_COLS, FIXED_PROPERTIES, SHRINK_AVAILABLE, clamp_targets, clamp_shrinkage, target_status,
        inverse_non_optimized,
        inverse_optimized, inverse_bayesian_optimization, inverse_pareto, refresh_prices,
        get_price_info, TGT_LABELS, MAT_SHORT,
    )
except Exception:
    st.error("The inverse-design backend failed to load. Details:")
    st.code(traceback.format_exc())
    st.stop()


def _show_df(df):
    """st.dataframe that works on old and new Streamlit (width="stretch" only exists in recent
    releases; older ones raise TypeError, newer ones deprecate use_container_width)."""
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


def _read_optional_csv(name: str):
    f = DATADIR / name
    return pd.read_csv(f) if f.exists() else None


@st.cache_data
def _load_json(path: str, mtime: float) -> dict:
    with open(path) as f:
        return json.load(f)


@st.cache_data
def _load_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def _p(name: str) -> Path:
    return DATADIR / name


meta = _load_json(str(_p("metadata.json")), _p("metadata.json").stat().st_mtime)
dataset = _load_csv(str(_p("dataset.csv")), _p("dataset.csv").stat().st_mtime)
materials = meta["materials"]
N_EXP = meta.get("n_experimental", len(dataset))
MOR_min, MOR_max = float(dataset["MOR_MPa"].min()), float(dataset["MOR_MPa"].max())
WA_min, WA_max = float(dataset["WA_pct"].min()), float(dataset["WA_pct"].max())
HW = meta.get("prediction_halfwidth", {})
ISO = meta.get("iso13006_bia", {"wa_max_pct": 0.5, "mor_min_mpa": 35.0})
held = {r["target"]: r for r in meta.get("held_out_test_metrics", [])}
arch = meta.get("architecture_per_target", {})
audit = meta.get("data_audit", {})

_exceed_df = _read_optional_csv("wa_exceedance_metrics.csv")
_n_shr_const = int((dataset["Shrinkage_pct"] == dataset["Shrinkage_pct"].max()).sum()) \
    if "Shrinkage_pct" in dataset.columns else None
_FS_TITLE, _FS_AX, _FS_TICK, _FS_LABEL, _FS_ANNOT = 13, 11, 10, 10, 8
UNITS = {"MOR_MPa": "MPa", "WA_pct": "%", "Shrinkage_pct": "%"}

for key in ["res_nn", "res_opt", "res_bay", "trial_vals", "identical", "tgt_nn", "tgt_opt",
            "tgt_bay", "tgt_all", "res_all", "pareto", "tgt_par"]:
    st.session_state.setdefault(key, None)

st.title("Ceramic Tile Inverse Composition Design")
st.caption(f"Forward model trained on {N_EXP} real experimental batches (no synthetic data). "
           "AKIJ Ceramics Ltd., Bangladesh.")

# ── Scope & limitations (generated from the metrics files) ───────────────────
_exceed_line = ""
if _exceed_df is not None and len(_exceed_df):
    _model_name = {"tobit": "A censored (Tobit) model", "gaussian_residual": "A Gaussian residual model"}.get(
        str(_exceed_df["model"].iloc[0]), "A model")
    _bits = [f"limit {r.limit_pct:g} %: {int(r.n_exceeding)}/{int(r.n_total)} batches above, "
             f"out-of-fold AUC {r.oof_auc:.2f}" for r in _exceed_df.itertuples()]
    _exceed_line = (f"{_model_name} estimates P(WA > limit); its out-of-fold exceedance scores are — "
                    + "; ".join(_bits) + ". Scores at limits with few exceeding batches are uncertain.")


def _metric_line(t: str) -> str:
    r = held.get(t)
    if not r:
        return "n/a"
    return (f"{arch.get(t, '?')}: held-out R² = {r['r2']:.2f}, MAE = {r['mae']:.3g} {UNITS[t]}, "
            f"90 % interval ±{HW.get(t, float('nan')):.3g} {UNITS[t]}")


with st.expander("Model Scope & Limitations (read before use)", expanded=False):
    bounds_rows = "\n".join(
        f"| {MAT_SHORT.get(m, m)} | {meta['bounds'][m][0]:.2f} | {meta['bounds'][m][1]:.2f} |" for m in materials)
    _wa = audit.get("wa", {})
    if _wa.get("censored"):
        _wa_text = (f"About {_wa.get('fraction_at_floor', float('nan')) * 100:.0f} % of training batches sit at the "
                    f"{WA_min:.2f} % lowest reported value (left-censored: the true value may be lower), and only "
                    f"{_wa.get('n_above_iso_bia_limit', '?')} exceed {ISO['wa_max_pct']} % (ISO 13006 group BIa limit). "
                    "WA point predictions are therefore low-confidence; they are best used as a compliance check "
                    f"against a maximum. {_exceed_line}")
    else:
        _wa_text = (f"WA is a continuous measured target ({_wa.get('n_above_iso_bia_limit', '?')} of {N_EXP} batches "
                    f"exceed {ISO['wa_max_pct']} %, the ISO 13006 group BIa limit; one or more negative readings, "
                    f"if any, were set to 0 %). It is the weakest of the modelled properties, so prefer a safety "
                    f"margin and read the exceedance probability. {_exceed_line}")
    if SHRINK_AVAILABLE:
        _shrink_text = (f"Shrinkage is modelled ({_metric_line('Shrinkage_pct')}). It is an optional two-sided design "
                        "target: tick the box and give a target and tolerance; recipes are then required to keep "
                        "|predicted shrinkage − target| plus the safety margin within the tolerance. Leave it "
                        "unticked to ignore shrinkage in the search (the predicted value is still reported).")
    else:
        _shrink_text = (f"Shrinkage is recorded as {FIXED_PROPERTIES.get('Shrinkage_pct', 'n/a')} % in "
                        f"{_n_shr_const if _n_shr_const is not None else 'almost all'} of {N_EXP} batches, so it shows "
                        "no usable variance; it is not modelled and not an input.")
    _frac_mor = audit.get("mor", {}).get("fraction_ge_iso_bia_min", float("nan")) * 100
    _tm = audit.get("total_mass", {})
    _mq = "; ".join(f"{TGT_LABELS.get(t, t)} — {_metric_line(t)}" for t in TARGET_COLS)
    st.markdown(f"""
**Training scope.** One homogeneous tile body, laboratory-scale preparation with firing in an
industrial roller kiln, at AKIJ Ceramics Ltd., trained on {N_EXP} real batches. Process held fixed:
rapid ball mill (alumina balls, 30 min); slip viscosity 25–35 s, density 1.67–1.70 g/cm³, residue
0.7–0.9 % (45 µm) and 0.5–0.8 % (850 µm); dried-powder moisture 6.0–6.5 %; green specimens
108–110 × 54–55 × 9–10 mm pressed at 100 bar; dried at 160–200 °C for 45 min; fired at 1210 °C
(90 min total cycle). This is a template for ONE production line and ONE product: other lines or
raw materials need their own data. The bounds below are the range actually fabricated (no
extrapolation margin).

**Model quality (held-out test set, from `held_out_test_metrics.csv`).**
{_mq}.
Prediction intervals are out-of-fold residual quantiles: they are the honest measure of
how far a predicted property can be trusted. Property differences between two recipes that are
smaller than the interval cannot be resolved by the model.

**Water absorption.** {_wa_text}

**Shrinkage.** {_shrink_text}

**Strength.** {_frac_mor:.0f} % of the training batches reach the ISO 13006 BIa minimum of {ISO['mor_min_mpa']:.0f} MPa.

**Sodium silicate** is a deflocculant (slip additive). In this data its dose is the strongest
MOR driver; treat it as a process additive, not as an ordinary body constituent.

**Recipes.** Optimiser recipes sum to 100 wt%. Real batches were weighed as independent masses
(totals {_tm.get('min', float('nan')):.0f}–{_tm.get('max', float('nan')):.0f}), so the nearest-neighbour methods return the as-weighed recipe with its total shown.

**Raw-material specificity.** Predictions apply to the raw materials of this supplier; other
sources may shift the response. **Extrapolation.** Recipes farther from real batches than the
training-domain threshold are flagged and penalised. **CO₂.** Emission factors are literature /
EPD proxies (comparative indicators, raw-material scope only: firing and drying energy are not
included) and differ between countries and allocation methods. Crushed fired tile and ETP sludge
carry the highest factors of any body constituent, so how waste burdens are allocated changes
the CO₂ ranking of recipes. Edit the dated `*_CO2.csv` file to use your own factors; the
robustness analysis shows the recommended recipe typically changes by only a few wt% and costs
a few percent in cost/CO₂ when factors change, with the largest effect for crushed tile, AG98
and the highest-MOR targets.

**Industrial use.** Re-train on facility-specific data with process parameters held fixed.

| Material | Min (wt%) | Max (wt%) |
|----------|-----------|-----------|
{bounds_rows}
""")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Optimiser settings")
    k_safety = st.slider(
        "Safety margin (× 90 % prediction half-width)", 0.0, 2.0, float(idsg.SAFETY_FACTOR), 0.5,
        help="0 = recipe must meet the spec at the point estimate only (about half of such recipes "
             "would fail if built). 1 (default) = must meet it at the pessimistic end of the 90 % "
             "prediction interval (MOR − hw, WA + hw). 2 may be infeasible for tight WA targets. "
             "Applies to Bayesian and Pareto searches.")
    st.divider()
    st.header("Fixed process parameters")
    st.caption("Held constant for all batches; not model inputs.")
    for label, val in [("Pressing pressure (bar)", 100), ("Dryer time (min)", 45),
                       ("Kiln cycle (min)", 90), ("Kiln temperature (°C)", 1210)]:
        st.number_input(label, value=float(val), disabled=True)
    if not SHRINK_AVAILABLE:
        st.number_input("Fired shrinkage (%) — fixed in data",
                        value=float(FIXED_PROPERTIES.get("Shrinkage_pct", 12.5)), disabled=True)
    st.divider()
    st.subheader("Raw-material price / CO₂ database")
    if st.button("Refresh prices now"):
        refresh_prices()
        st.success("Price / CO₂ tables reloaded.")
    _pinfo = get_price_info()
    for _label, _key in [("Cost table", "cost"), ("CO₂ table", "co2")]:
        _e = _pinfo.get(_key, {})
        if _e.get("file") is not None:
            st.write(f"**{_label}:** `{Path(_e['file']).name}`")
        else:
            st.write(f"**{_label}:** fallback values (metadata.json)")

refresh_prices()

# ── Targets ──────────────────────────────────────────────────────────────────
st.subheader("Target specification")
st.caption("MOR is a **minimum** requirement and water absorption a **maximum** limit (one-sided "
           "specifications): a recipe that exceeds the MOR value or stays below the WA value is acceptable. "
           + ("Shrinkage is an optional **two-sided** target (target ± tolerance)." if SHRINK_AVAILABLE else
              "Shrinkage is shown for completeness but is fixed in the training data, so it cannot be designed."))
c1, c2, c3 = st.columns(3)
with c1:
    MOR_in = st.number_input(f"MOR (MPa) — minimum required  [{MOR_min:.1f} – {MOR_max:.1f}]",
                             min_value=0.0, max_value=200.0, value=round((MOR_min + MOR_max) / 2, 1),
                             step=0.5, format="%.1f",
                             help=(f"ISO 13006 BIa minimum is {ISO['mor_min_mpa']:.0f} MPa; "
                                   f"{audit.get('mor', {}).get('fraction_ge_iso_bia_min', float('nan')) * 100:.0f} % of "
                                   "the training batches reach it."))
with c2:
    WA_in = st.number_input(f"Water absorption (%) — maximum allowed  [{WA_min:.2f} – {WA_max:.2f}]",
                            min_value=0.0, max_value=20.0, value=float(ISO["wa_max_pct"]),
                            step=0.01, format="%.2f",
                            help=f"ISO 13006 group BIa limit: {ISO['wa_max_pct']} %.")
S_in = S_tol_in = None
with c3:
    if SHRINK_AVAILABLE:
        S_min, S_max = float(dataset["Shrinkage_pct"].min()), float(dataset["Shrinkage_pct"].max())
        use_S = st.checkbox("Constrain shrinkage", value=False,
                            help="Require the predicted fired shrinkage to lie within target ± tolerance "
                                 "(including the safety margin).")
        S_in = st.number_input(f"Shrinkage target (%)  [{S_min:.2f} – {S_max:.2f}]", min_value=0.0, max_value=30.0,
                               value=round(float(dataset["Shrinkage_pct"].median()), 2), step=0.05, format="%.2f",
                               disabled=not use_S)
        S_tol_in = st.number_input("Shrinkage tolerance ± (%-points)", min_value=0.05, max_value=5.0,
                                   value=float(idsg.DEFAULT_SHRINK_TOL), step=0.05, format="%.2f",
                                   disabled=not use_S,
                                   help=f"The 90 % prediction half-width of shrinkage is ±{HW.get('Shrinkage_pct', float('nan')):.2f}; "
                                        "with a safety margin k the tolerance must exceed k times that to be achievable.")
        if not use_S:
            S_in = S_tol_in = None
    else:
        st.number_input("Fired shrinkage (%) — fixed, not an input",
                        value=float(FIXED_PROPERTIES.get("Shrinkage_pct", 12.5)), disabled=True,
                        help=(f"Recorded as {FIXED_PROPERTIES.get('Shrinkage_pct', 12.5)} % in "
                              f"{_n_shr_const if _n_shr_const is not None else 'almost all'} of {N_EXP} batches, "
                              "so the data contain no usable variation: shrinkage is neither modelled nor optimised."))
_status = target_status(MOR_in, WA_in, S_in)
for msg in _status["messages"]:
    st.warning(msg)
MOR_c, WA_c = clamp_targets(MOR_in, WA_in)
S_c = clamp_shrinkage(S_in)
S_KW = {"S_tgt": S_c, "S_tol": S_tol_in} if S_c is not None else {}
TGT = {"MOR_MPa": MOR_c, "WA_pct": WA_c, "S_tgt": S_c, "S_tol": S_tol_in if S_c is not None else None}

# ── Helpers ──────────────────────────────────────────────────────────────────
def _fmt_interval(res: dict, t: str) -> str:
    iv = res.get("pred_interval_90", {}).get(t)
    return f"[{iv[0]:.3g}, {iv[1]:.3g}]" if iv else "measured"


def _table(res: dict) -> pd.DataFrame:
    row = {MAT_SHORT.get(m, m): v for m, v in res["composition_wtpct"].items()}
    row["Total (wt%)"] = res.get("batch_total_wtpct", 100.0)
    for t in TARGET_COLS:
        row[TGT_LABELS[t]] = res["predicted"][t]
        row[TGT_LABELS[t] + " 90% interval"] = _fmt_interval(res, t)
    row["Cost (Tk/kg)"] = res["cost_Tk_per_kg"]
    row["CO₂ (kg/kg)"] = res["CO2_kg_per_kg"]
    return pd.DataFrame([row])


def _comp_bar(res: dict, title: str, color: str):
    comp = res["composition_wtpct"]
    labels = [MAT_SHORT.get(m, m) for m in comp]
    vals = list(comp.values())
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(range(len(labels)), vals, color=color, alpha=0.85, edgecolor="white")
    ymax = max(vals) if vals else 1.0
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + ymax * 0.01, f"{h:.1f}", ha="center",
                va="bottom", fontsize=_FS_ANNOT, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=_FS_TICK)
    ax.set_ylabel("Mass (wt%)", fontsize=_FS_AX)
    ax.set_title(title, fontsize=_FS_AX, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.4, axis="y")
    ax.set_ylim(0, ymax * 1.15)
    plt.tight_layout()
    return fig



def _margins(res: dict, tgt: dict) -> dict:
    """Positive = spec met."""
    m = {"MOR margin (MPa)": res["predicted"]["MOR_MPa"] - tgt["MOR_MPa"],
         "WA margin (%)": tgt["WA_pct"] - res["predicted"]["WA_pct"]}
    if tgt.get("S_tgt") is not None and "Shrinkage_pct" in res["predicted"]:
        m["Shrinkage margin (%-pts)"] = tgt["S_tol"] - abs(res["predicted"]["Shrinkage_pct"] - tgt["S_tgt"])
    return m


def _margin_bars(res: dict, tgt: dict, title: str):
    m = _margins(res, tgt)
    fig, ax = plt.subplots(figsize=(5, 3.8))
    colors = ["#2E7D32" if v >= 0 else "#C62828" for v in m.values()]
    bars = ax.bar(list(m.keys()), list(m.values()), color=colors, alpha=0.85, edgecolor="white")
    span = max(abs(v) for v in m.values()) or 1.0
    for b, v in zip(bars, m.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.03 * span if v >= 0 else -0.03 * span),
                f"{v:+.3g}", ha="center", va="bottom" if v >= 0 else "top", fontsize=_FS_ANNOT + 1,
                fontweight="bold")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Spec margin (green = met)", fontsize=_FS_AX)
    ax.set_title(title, fontsize=_FS_AX, fontweight="bold")
    ax.tick_params(axis="x", labelsize=_FS_TICK, rotation=10)
    ax.grid(True, ls="--", alpha=0.4, axis="y")
    ax.set_ylim(-1.4 * span, 1.4 * span)
    plt.tight_layout()
    return fig


def _spec_flags(res: dict, tgt: dict) -> None:
    m = _margins(res, tgt)
    ok_point = all(v >= 0 for v in m.values())
    hw_ok = (res["predicted"]["MOR_MPa"] - HW.get("MOR_MPa", 0) >= tgt["MOR_MPa"] and
             res["predicted"]["WA_pct"] + HW.get("WA_pct", 0) <= tgt["WA_pct"])
    if tgt.get("S_tgt") is not None and "Shrinkage_pct" in res["predicted"]:
        hw_ok = hw_ok and (abs(res["predicted"]["Shrinkage_pct"] - tgt["S_tgt"]) + HW.get("Shrinkage_pct", 0)
                           <= tgt["S_tol"])
    if "pred_interval_90" in res:
        (st.success if ok_point else st.error)(
            f"Spec met by the point estimate: {'yes' if ok_point else 'NO'}. "
            f"Spec met at the pessimistic 90 % bound: {'yes' if hw_ok else 'no'}.")
    else:
        (st.success if ok_point else st.error)(
            f"Measured properties of this real batch meet the spec: {'yes' if ok_point else 'NO'}.")
    if res.get("p_WA_exceeds_target") is not None:
        st.caption(f"Probability that WA exceeds {tgt['WA_pct']:.2f} % (model estimate): "
                   f"{res['p_WA_exceeds_target']:.1%}.")
    if res.get("trust_distance") is not None:
        st.caption(f"Distance to nearest real batch: {res['trust_distance']:.2f} "
                   f"(training-domain threshold {res['trust_threshold']:.2f}).")


def _metrics_row(res: dict):
    a, b = st.columns(2)
    a.metric("Batch cost", f"{res['cost_Tk_per_kg']:.4f} Tk/kg")
    b.metric("CO₂ (proxy factors)", f"{res['CO2_kg_per_kg']:.5f} kg/kg")


def _show_result(res: dict, tgt: dict, color: str, label: str):
    st.markdown("#### Recommended recipe")
    _show_df(_table(res))
    _spec_flags(res, tgt)
    c1, c2 = st.columns(2)
    with c1:
        fig = _comp_bar(res, "Recipe (as-weighed wt%)", color)
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        fig = _margin_bars(res, tgt, label)
        st.pyplot(fig)
        plt.close(fig)
    _metrics_row(res)


st.subheader("Inverse design method")
tabs = st.tabs(["Non-Optimised (NN)", "Cost + CO₂ Optimised (NN)", "Bayesian Optimisation",
                "Compare All Methods", "Pareto front"])

# ── Tab 1 ────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("Returns the **real fabricated batch** whose measured MOR and WA are closest to the "
                "targets (scaled distance). No cost or CO₂ consideration; a baseline. Properties "
                "shown are **measured**, not predicted.")
    if st.button("Run Non-Optimised", key="run_nn"):
        st.session_state["tgt_nn"] = dict(TGT)
        st.session_state["res_nn"] = inverse_non_optimized(MOR_c, WA_c, **S_KW)
    if st.session_state["res_nn"] is not None:
        _show_result(st.session_state["res_nn"], st.session_state["tgt_nn"], "#E24A33",
                     "Margin vs. target")

# ── Tab 2 ────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("Among the **10 nearest real batches**, picks the lowest combined cost + CO₂ rank "
                "(latest price table). If it equals Method 1 the neighbourhood has no cost/CO₂ diversity.")
    if st.button("Run Cost + CO₂ Optimised", key="run_opt"):
        refresh_prices()
        st.session_state["tgt_opt"] = dict(TGT)
        st.session_state["res_opt"], st.session_state["identical"] = inverse_optimized(MOR_c, WA_c, **S_KW)
    if st.session_state["res_opt"] is not None:
        if st.session_state["identical"]:
            st.warning("Methods 1 and 2 returned the same batch for this target.")
        _show_result(st.session_state["res_opt"], st.session_state["tgt_opt"], "#348ABD",
                     "Margin vs. target")

# ── Tab 3 ────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown(
        "Searches the **continuous recipe space** (sum = 100 wt%, within the fabricated ranges) with "
        "Optuna TPE, minimising a **weighted-sum (scalarised)** objective: range-normalised cost + "
        "range-normalised CO₂ + *w* × one-sided spec violation (MOR below its minimum, WA above its "
        "maximum) + guardrails. Property violations are weighted *w* : 1 against cost and CO₂ "
        f"(*w* = {idsg.PROPERTY_PENALTY_WEIGHT:g}). The recipe returned is the cheapest trial that meets "
        "the spec at the chosen safety margin (if any was found), followed by a short local polish "
        "that stays inside the training domain. This is **not** a Pareto solver and its solutions "
        "are not guaranteed Pareto-optimal — see the Pareto tab. Typical run time: a few seconds.")
    n_trials = st.slider("Optimisation trials", 50, 400, 200, 50)
    if st.button("Run Bayesian Optimisation", key="run_bayes"):
        refresh_prices()
        st.session_state["tgt_bay"] = dict(TGT)
        with st.spinner(f"Running {n_trials} trials..."):
            res, tv, _ = inverse_bayesian_optimization(MOR_c, WA_c, n_trials=n_trials, k_sigma=k_safety, **S_KW)
        st.session_state["res_bay"], st.session_state["trial_vals"] = res, tv
    if st.session_state["res_bay"] is not None:
        res, tgt, tv = st.session_state["res_bay"], st.session_state["tgt_bay"], st.session_state["trial_vals"]
        if not res["feasible"]:
            st.error(f"No recipe inside the training-data domain was found in {res['n_trials']} trials. "
                     "The result below is the least-bad candidate: relax the target, add trials, or "
                     "collect real batches in this region.")
        else:
            st.caption(f"{res['n_feasible_trials']}/{res['n_trials']} trials inside the training domain; "
                       f"safety margin k = {res['safety_factor']:g}"
                       f"{'; local polish applied' if res.get('polished') else ''}.")
        if res.get("warning"):
            st.warning(res["warning"])
        elif res.get("meets_target_at_k") is not None:
            (st.success if res["meets_target_at_k"] else st.error)(
                f"Spec met with safety margin k = {res['safety_factor']:g}: "
                f"{'yes' if res['meets_target_at_k'] else 'NO'}.")
        _show_result(res, tgt, "#8EBA42", "Margin vs. target (point estimate)")
        st.markdown("#### Convergence")
        fig, ax = plt.subplots(figsize=(8, 3.8))
        t = np.arange(1, len(tv) + 1)
        ax.plot(t, tv, color="#BDBDBD", lw=1.0, alpha=0.7, label="Trial objective")
        ax.plot(t, np.minimum.accumulate(tv), color="#E53935", lw=2.2, label="Best so far")
        ax.set_xlabel("Trial", fontsize=_FS_AX)
        ax.set_ylabel("Scalarised objective", fontsize=_FS_AX)
        ax.legend(fontsize=_FS_LABEL)
        ax.grid(True, ls="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ── Tab 4 ────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("Runs all three methods on the same target. Cost and CO₂ changes are relative to "
                "the Non-Optimised (NN) batch. **Cost/CO₂ differences of a few percent are small "
                "compared with the price and emission-factor uncertainty and should be read as indicative.**")
    n_trials_cmp = st.slider("Bayesian trials (comparison)", 50, 400, 200, 50)
    if st.button("Compare All Methods", key="run_all", type="primary"):
        refresh_prices()
        st.session_state["tgt_all"] = dict(TGT)
        with st.spinner("Running all methods..."):
            r1 = inverse_non_optimized(MOR_c, WA_c, **S_KW)
            r2, ident = inverse_optimized(MOR_c, WA_c, **S_KW)
            r3, _, _ = inverse_bayesian_optimization(MOR_c, WA_c, n_trials=n_trials_cmp, k_sigma=k_safety, **S_KW)
        st.session_state["res_all"] = {"Non-Optimised (NN)": r1, "Cost + CO₂ Optimised (NN)": r2,
                                       "Bayesian Optimisation": r3}
        st.session_state["identical"] = ident
    if st.session_state["res_all"] is not None:
        results, tgt = st.session_state["res_all"], st.session_state["tgt_all"]
        colors = ["#E24A33", "#348ABD", "#8EBA42"]
        bay = results["Bayesian Optimisation"]
        if st.session_state["identical"]:
            st.warning("Methods 1 and 2 returned the same batch.")
        if not bay["feasible"]:
            st.error("The Bayesian result is UNRELIABLE (no recipe inside the training domain found).")
        if bay.get("warning"):
            st.warning(bay["warning"])

        st.subheader("Recommended recipes")
        cmp_df = pd.DataFrame({
            n: {**{MAT_SHORT.get(m, m): v for m, v in r["composition_wtpct"].items()},
                "Total (wt%)": r.get("batch_total_wtpct", 100.0),
                **{TGT_LABELS[t]: r["predicted"][t] for t in TARGET_COLS},
                **{k_: round(v_, 3) for k_, v_ in _margins(r, tgt).items()},
                "P(WA > max)": (f"{r['p_WA_exceeds_target']:.1%}" if r.get("p_WA_exceeds_target") is not None else "—"),
                "Cost (Tk/kg)": r["cost_Tk_per_kg"], "CO₂ (kg/kg)": r["CO2_kg_per_kg"]}
            for n, r in results.items()}).T
        _show_df(cmp_df)

        st.subheader("Cost & CO₂ change vs. Non-Optimised baseline")
        b_cost, b_co2 = results["Non-Optimised (NN)"]["cost_Tk_per_kg"], results["Non-Optimised (NN)"]["CO2_kg_per_kg"]
        red = pd.DataFrame([{"Method": n, "Cost (Tk/kg)": r["cost_Tk_per_kg"],
                             "Cost change (%)": round((r["cost_Tk_per_kg"] / b_cost - 1) * 100, 2),
                             "CO₂ (kg/kg)": r["CO2_kg_per_kg"],
                             "CO₂ change (%)": round((r["CO2_kg_per_kg"] / b_co2 - 1) * 100, 2)}
                            for n, r in results.items()]).set_index("Method")
        _show_df(red)

        st.subheader("Composition comparison")
        comp_df = pd.DataFrame({n: r["composition_wtpct"] for n, r in results.items()})
        x = np.arange(len(comp_df))
        bw = 0.26
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for off, (n, c) in zip((-bw, 0, bw), zip(results, colors)):
            ax.bar(x + off, comp_df[n], bw, label=n, color=c, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([MAT_SHORT.get(m, m) for m in comp_df.index], rotation=35, ha="right",
                           fontsize=_FS_TICK)
        ax.set_ylabel("Mass (wt%)", fontsize=_FS_AX)
        ax.legend(fontsize=_FS_LABEL - 1)
        ax.grid(True, ls="--", alpha=0.4, axis="y")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Cost vs. CO₂")
        fig, ax = plt.subplots(figsize=(6, 4))
        for (n, r), c, mk in zip(results.items(), colors, ["o", "s", "*"]):
            ax.scatter(r["cost_Tk_per_kg"], r["CO2_kg_per_kg"], s=170, marker=mk, color=c, label=n, zorder=3)
        ax.set_xlabel("Cost (Tk/kg)", fontsize=_FS_AX)
        ax.set_ylabel("CO₂ (kg/kg, proxy factors)", fontsize=_FS_AX)
        ax.legend(fontsize=_FS_LABEL - 1)
        ax.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Spec margins (positive = met)")
        _mkeys = list(_margins(next(iter(results.values())), tgt).keys())
        fig, axes = plt.subplots(1, len(_mkeys), figsize=(4.5 * len(_mkeys), 3.6), squeeze=False)
        for ax, key in zip(axes[0], _mkeys):
            vals = [_margins(r, tgt)[key] for r in results.values()]
            ax.bar(range(3), vals, color=colors, alpha=0.85)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(range(3))
            ax.set_xticklabels(["NN", "NN+cost/CO₂", "Bayesian"], fontsize=_FS_TICK)
            ax.set_title(key, fontsize=_FS_AX)
            ax.grid(True, ls="--", alpha=0.4, axis="y")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ── Tab 5: Pareto ────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("Genuinely multi-objective search (**NSGA-II**, no weights): minimises cost, CO₂ and "
                "spec violation simultaneously; the training-domain condition is a hard constraint. "
                "Shows every domain-feasible recipe found; the **Pareto set** of spec-compliant recipes "
                "is the cost–CO₂ trade-off curve.")
    n_par = st.slider("NSGA-II trials", 200, 2000, 800, 200,
                      help="More trials give a denser front, especially for demanding targets where few "
                           "recipes are compliant.")
    if st.button("Compute Pareto front", key="run_pareto"):
        refresh_prices()
        st.session_state["tgt_par"] = dict(TGT)
        with st.spinner("Searching..."):
            st.session_state["pareto"] = inverse_pareto(MOR_c, WA_c, n_trials=n_par, k_sigma=k_safety, **S_KW)
    pdf = st.session_state["pareto"]
    if pdf is not None:
        if pdf.empty or not pdf["compliant"].any():
            st.error("No spec-compliant recipe inside the training domain was found. Relax the target "
                     "or reduce the safety margin.")
        else:
            front = pdf[pdf["pareto"] & pdf["compliant"]].sort_values("cost_Tk_per_kg")
            _n_comp = int(pdf["compliant"].sum())
            st.caption(f"{_n_comp} spec-compliant recipes among {len(pdf)} domain-feasible trials; "
                       f"{len(front)} on the Pareto set (safety margin k = {k_safety:g}).")
            if _n_comp < 100:
                st.warning("Few compliant trials were found, so this front is probably under-sampled "
                           "(cheaper compliant recipes may exist). Increase the number of trials or "
                           "lower the safety margin.")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.scatter(pdf["cost_Tk_per_kg"], pdf["CO2_kg_per_kg"], c=np.where(pdf["compliant"], "#9E9E9E", "#EF9A9A"),
                       s=18, label="feasible trials (grey: compliant, red: violates spec)")
            ax.scatter(front["cost_Tk_per_kg"], front["CO2_kg_per_kg"], color="#1565C0", s=45,
                       label="Pareto set (compliant)", zorder=3)
            ax.set_xlabel("Cost (Tk/kg)", fontsize=_FS_AX)
            ax.set_ylabel("CO₂ (kg/kg, proxy factors)", fontsize=_FS_AX)
            ax.legend(fontsize=_FS_LABEL - 1)
            ax.grid(True, ls="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            _pc = ["cost_Tk_per_kg", "CO2_kg_per_kg", "MOR_pred", "WA_pred"] + \
                  (["Shrinkage_pred"] if SHRINK_AVAILABLE and "Shrinkage_pred" in front.columns else [])
            show = front[_pc + list(materials)].copy()
            show.columns = ["Cost (Tk/kg)", "CO₂ (kg/kg)", "MOR pred (MPa)", "WA pred (%)"] + \
                           (["Shrinkage pred (%)"] if len(_pc) == 5 else []) + \
                           [MAT_SHORT.get(m, m) for m in materials]
            _show_df(show.round(4))
            st.caption("Predicted MOR/WA carry the model's 90 % intervals "
                       f"(±{HW.get('MOR_MPa', float('nan')):.3g} MPa, ±{HW.get('WA_pct', float('nan')):.3g} %).")