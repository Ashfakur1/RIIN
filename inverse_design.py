#!/usr/bin/env python3
"""
inverse_design.py
Inverse design of ceramic tile recipes (backend library imported by
streamlit_app.py and run_analyses.py; no plotting, no file output).

WHAT THE THREE METHODS DO
  1. Nearest Neighbour (non-optimised): the real fabricated batch whose MEASURED
     properties are closest to the target (scaled MOR, WA space; shrinkage too if a band
     is given) AMONG BATCHES THAT MEET THE SPEC (MOR >= target, WA <= target, |S - S*| <=
     tolerance if a band is given); falls back to the nearest batch overall if none does.
     Baseline.
  2. Nearest Neighbour (cost + CO2): among the 10 nearest spec-compliant real
     batches, the one with the lowest combined cost + CO2 percentile rank.
  3. Bayesian optimisation (Optuna TPE): continuous search over recipes on the
     sum = 100 wt% slice, minimising a SCALARISED objective
         f = cost~ + CO2~ + w * [ max(0, MOR* - MOR_eff)/r_MOR
                                  + max(0, WA_eff - WA*)/r_WA
                                  + max(0, |S - S*| + k*hw_S - tol)/r_S  (only if a band is given) ]
             + guardrails
     (cost~, CO2~ range-normalised; MOR* is a MINIMUM spec, WA* a MAXIMUM spec).
     This is a weighted-sum scalarisation of a multi-objective problem, NOT a
     Pareto solver (Reviewers 4 and 7); solutions are not guaranteed Pareto-
     optimal. inverse_pareto() below solves the genuinely multi-objective
     problem (NSGA-II) so the two can be compared.
     ON THE WEIGHT: cost~ and CO2~ each span [0, 1] over the dataset; a property
     shortfall of 1 % of that property's dataset range therefore costs
     w * 0.01, while 1 % of the cost (or CO2) range costs 0.01. The exchange
     rate is exactly w : 1 (= 10 : 1 at baseline). The former "roughly 7.5 to 1"
     statement has no derivation in the code and should be removed from the
     manuscript; run_analyses.py sensitivity sweeps w instead (Reviewers 1, 2, 4, 8).

THE THREE PROPERTIES
  MOR is a MINIMUM spec and WA a MAXIMUM spec (one-sided). Shrinkage, when it is a modelled target
  (it is not if the training data show no variation), is an OPTIONAL two-sided band: target S* +/-
  tolerance (S_tgt, S_tol; default tolerance DEFAULT_SHRINK_TOL). With S_tgt=None it is ignored
  and only reported as a prediction with its interval.

DESIGN NOTES
  * Model inputs are AS-WEIGHED masses (see train_forward_model.py); optimiser recipes are
    constrained to sum = 100 wt% and queried on that slice.
  * Trust region = 95th percentile of leave-one-out nearest-neighbour distances between REAL
    batches (a data-defined applicability domain).
  * SAFETY_FACTOR k (default 1) requires MOR - k*hw >= target, WA + k*hw <= target and
    |S - S*| + k*hw <= tol, hw = 90 % prediction half-width from out-of-fold residuals
    (metadata.json). At k = 0 the optimiser sits on the spec boundary, so a built batch fails
    about half the time.
  * production_mode=True (default) in inverse_bayesian_optimization(): (a) the returned recipe is
    the cheapest DOMAIN-FEASIBLE trial that meets the spec at margin k, if one was found; (b) a
    short SLSQP polish removes optimiser noise near the boundary. The analysis scripts pass
    production_mode=False for the raw-search studies so they measure the search itself.
  * PROPERTY_PENALTY_WEIGHT w = 10 (exchange rate exactly w : 1; see ON THE WEIGHT above).
  * The Optuna seed is a parameter. Cost / CO2 are per kg of batch: sum(x_m * c_m) / sum(x_m).
  * WA_PENALTY_SYMMETRIC switch answers Reviewer 2's symmetric-penalty question.
  * Each trial stores the composition it actually evaluated (no drift between the optimiser's
    parameters and the reported recipe).
  * p_WA_exceeds_target = P(WA > target): from the Tobit model if WA is left-censored in the
    training data, otherwise from a Gaussian residual model (metadata.json "wa_exceedance").
  * WA_DESIGN_LIMIT (default None) optionally tightens the WA limit used INSIDE the search.
    Caution: k times the WA half-width is added, so very small limits can make the problem
    infeasible.

PRICES / CO2 are read at run time from dated CSVs (model_utils.load_price_table); the
forward model is never retrained when prices change.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize as _sp_minimize
from scipy.stats import norm as _norm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from model_utils import MultiTargetModel, load_price_table  # MultiTargetModel is needed to unpickle forward_model

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).parent
DATADIR = BASE_DIR / "data"
MODELDIR = BASE_DIR / "models"
COST_DIR = DATADIR / "cost"
CO2_DIR = DATADIR / "co2"

# ── Load real dataset, model, metadata ───────────────────────────────────────
dataset = pd.read_csv(DATADIR / "dataset.csv")
forward_model = joblib.load(MODELDIR / "forward_model.joblib")
with open(DATADIR / "metadata.json") as f:
    meta = json.load(f)
with open(MODELDIR / "feature_cols.json") as f:
    model_feature_cols: list[str] = json.load(f)

materials = meta["materials"]
comp_cols = meta.get("comp_cols", [f"{m}_wtpct" for m in materials])
bounds = meta["bounds"]
TARGET_COLS = list(meta["modelled_targets"])            # e.g. ["MOR_MPa", "WA_pct", "Shrinkage_pct"]
FIXED_PROPERTIES = dict(meta.get("constant_targets", {}))   # targets with no variance (not modelled)
SHRINK_AVAILABLE = "Shrinkage_pct" in TARGET_COLS
HALFWIDTH = dict(meta.get("prediction_halfwidth", {t: 0.0 for t in TARGET_COLS}))
_WA_TOBIT = meta.get("wa_tobit")            # censored WA model (only if WA is left-censored)
_WA_EXCEED = meta.get("wa_exceedance")      # {"model": "tobit" | "gaussian_residual", "sigma": ...}
_WA_CENSORED = bool(meta.get("data_audit", {}).get("wa", {}).get("censored", False))

_fallback_cost = meta.get("cost_tk_per_kg", {m: 1.0 for m in materials})
_fallback_co2 = {m: float(np.mean(v)) if isinstance(v, (list, tuple)) else float(v)
                 for m, v in meta.get("co2_kg_per_kg", {m: 0.0 for m in materials}).items()}

# ── Optimiser configuration (module-level so sensitivity scripts can sweep) ──
PROPERTY_PENALTY_WEIGHT = 10.0    # w: property : cost exchange rate is exactly w : 1
EXTRAP_PENALTY_WEIGHT = 10.0      # trust-region guardrail
SANITY_PENALTY_WEIGHT = 20.0      # implausible-prediction guardrail
PENALTY_CAP = 100.0
WA_PENALTY_SYMMETRIC = False      # False: WA is a maximum spec (one-sided)
SAFETY_FACTOR = 1.0               # k in units of the 90 % prediction half-width
WA_DESIGN_LIMIT = None            # optional tighter WA limit used inside the search (see docstring)
COMPLIANCE_TOL = 1e-9             # violation <= tol counts as meeting the spec
POLISH_MARGIN = 1e-4              # spec margin (fraction of property range) kept by the polish
DEFAULT_SHRINK_TOL = 0.75         # default shrinkage tolerance (absolute %-points) around S*
TRUST_REGION_PERCENTILE = 95.0    # of leave-one-out NN distances between real batches
PROPERTY_SANITY_MARGIN = 0.5
PREDICTION_CLAMP_MULTIPLE = 5.0

# ── Module state populated by refresh_prices() ───────────────────────────────
cost_dict: dict[str, float] = {}
co2_dict: dict[str, float] = {}
price_info: dict = {"cost": {}, "co2": {}}
_pr: dict[str, float] = {}
_prop_ranges: dict[str, float] = {}
_cost_min = _cost_max = _cost_range = 0.0
_co2_min = _co2_max = _co2_range = 0.0
_prop_scaler = None
_nbrs = None
_comp_scaler = None
_comp_nbrs = None
_trust_region_threshold = float("inf")


def _per_kg(frame: pd.DataFrame, table: dict) -> pd.Series:
    """sum(x_m * v_m) / sum(x_m) for as-weighed batches."""
    num = sum(frame[f"{m}_wtpct"] * table[m] for m in materials)
    den = sum(frame[f"{m}_wtpct"] for m in materials)
    return num / den


def refresh_prices() -> None:
    """Reload the latest dated cost / CO2 tables and recompute every quantity
    derived from them. Never retrains or touches the forward model."""
    global cost_dict, co2_dict, price_info, dataset
    global _pr, _prop_ranges, _cost_min, _cost_max, _cost_range
    global _co2_min, _co2_max, _co2_range
    global _prop_scaler, _nbrs, _comp_scaler, _comp_nbrs, _trust_region_threshold

    cost_dict, cost_meta = load_price_table(
        COST_DIR, keyword="Cost", value_col="Price_Tk_per_kg",
        materials=materials, fallback=_fallback_cost)
    co2_dict, co2_meta = load_price_table(
        CO2_DIR, keyword="CO2", value_col="CO2_kg_per_kg",
        materials=materials, fallback=_fallback_co2)
    price_info = {"cost": cost_meta, "co2": co2_meta}

    dataset["cost_Tk_per_kg"] = _per_kg(dataset, cost_dict)
    dataset["CO2_kg_per_kg"] = _per_kg(dataset, co2_dict)

    _pr = {}
    for t in TARGET_COLS:
        _pr[f"{t}_min"] = float(dataset[t].min())
        _pr[f"{t}_max"] = float(dataset[t].max())
    _prop_ranges = {t: max(_pr[f"{t}_max"] - _pr[f"{t}_min"], 1e-9) for t in TARGET_COLS}

    _cost_min, _cost_max = float(dataset["cost_Tk_per_kg"].min()), float(dataset["cost_Tk_per_kg"].max())
    _cost_range = max(_cost_max - _cost_min, 1e-9)
    _co2_min, _co2_max = float(dataset["CO2_kg_per_kg"].min()), float(dataset["CO2_kg_per_kg"].max())
    _co2_range = max(_co2_max - _co2_min, 1e-9)

    if _prop_scaler is None:
        _prop_scaler = StandardScaler()
        Xp = _prop_scaler.fit_transform(dataset[TARGET_COLS].values)
        _nbrs = NearestNeighbors(n_neighbors=min(10, len(dataset))).fit(Xp)

    if _comp_scaler is None:
        _comp_scaler = StandardScaler()
        Xc = _comp_scaler.fit_transform(dataset[comp_cols].values)
        _comp_nbrs = NearestNeighbors(n_neighbors=1).fit(Xc)
        # Applicability domain: how far apart are neighbouring REAL batches?
        d2, _ = NearestNeighbors(n_neighbors=2).fit(Xc).kneighbors(Xc)
        loo = d2[:, 1]                       # distance to nearest OTHER real batch
        _trust_region_threshold = float(np.percentile(loo, TRUST_REGION_PERCENTILE))
        print(f"  Composition trust-region threshold: {_trust_region_threshold:.4f} "
              f"(scaled units; {TRUST_REGION_PERCENTILE:.0f}th percentile of leave-one-out "
              f"nearest-neighbour distances among {len(dataset)} real batches)")


def get_price_info() -> dict:
    return price_info


refresh_prices()

TGT_LABELS = {"MOR_MPa": "Firing MOR (MPa)", "WA_pct": "Water Absorption (%)",
              "Shrinkage_pct": "Fired Shrinkage (%)"}
MAT_SHORT = {"AG98": "AG98", "AG22": "AG22", "AG23": "AG23", "SodaF": "Soda Feldspar",
             "PotashF": "Potash Feldspar", "Crushing": "Crushed Fired Tile",
             "ETP": "ETP Sludge", "NaSil": "Na-Silicate"}


# ── Helpers ──────────────────────────────────────────────────────────────────
def build_input_row(comp: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([{f"{m}_wtpct": comp[m] for m in materials}])[model_feature_cols]


def target_status(MOR: float, WA: float, S: float | None = None) -> dict:
    """What clamp_targets() would do, for display in the app (no warnings)."""
    out = {"clamped": False, "messages": []}
    checks = [("MOR", MOR, "MOR_MPa"), ("WA", WA, "WA_pct")]
    if S is not None and SHRINK_AVAILABLE:
        checks.append(("Shrinkage", S, "Shrinkage_pct"))
    for name, val, key in checks:
        lo, hi = _pr[f"{key}_min"], _pr[f"{key}_max"]
        if val < lo or val > hi:
            out["clamped"] = True
            out["messages"].append(
                f"{name} target {val:.3f} is outside the fabricated range "
                f"[{lo:.3f}, {hi:.3f}] and was clipped to {min(max(val, lo), hi):.3f}.")
    if _WA_CENSORED and WA <= _pr["WA_pct_min"] + 0.02:
        out["messages"].append(
            f"WA targets at or below ~{_pr['WA_pct_min']:.2f} % sit at the lowest reported value of the "
            "training data (many batches are recorded there, i.e. left-censored); the "
            "model cannot resolve values below it.")
    return out


def clamp_targets(MOR: float, WA: float) -> tuple[float, float]:
    """Clip user targets to the fabricated range (issues warnings)."""
    MOR_c = float(np.clip(MOR, _pr["MOR_MPa_min"], _pr["MOR_MPa_max"]))
    WA_c = float(np.clip(WA, _pr["WA_pct_min"], _pr["WA_pct_max"]))
    if MOR != MOR_c:
        warnings.warn(f"MOR_MPa clamped: {MOR:.3f} -> {MOR_c:.3f}")
    if WA != WA_c:
        warnings.warn(f"WA_pct clamped: {WA:.4f} -> {WA_c:.4f}")
    return MOR_c, WA_c


def clamp_shrinkage(S: float | None) -> float | None:
    """Clip a shrinkage target to the fabricated range; None if shrinkage is not a modelled target."""
    if S is None or not SHRINK_AVAILABLE:
        return None
    S_c = float(np.clip(S, _pr["Shrinkage_pct_min"], _pr["Shrinkage_pct_max"]))
    if S != S_c:
        warnings.warn(f"Shrinkage_pct clamped: {S:.3f} -> {S_c:.3f}")
    return S_c


def _enforce_bounds(comp: dict[str, float]) -> dict[str, float]:
    """Project onto {sum = 100, lo <= x_m <= hi} by iterative water-filling."""
    lo = {m: bounds[m][0] for m in materials}
    hi = {m: bounds[m][1] for m in materials}
    x = {m: float(np.clip(comp[m], lo[m], hi[m])) for m in materials}
    for _ in range(100):
        diff = 100.0 - sum(x.values())
        if abs(diff) < 1e-9:
            break
        room = ({m: hi[m] - x[m] for m in materials if hi[m] - x[m] > 1e-12} if diff > 0
                else {m: x[m] - lo[m] for m in materials if x[m] - lo[m] > 1e-12})
        tot = sum(room.values())
        if not room or tot <= 1e-12:
            break
        for m, r in room.items():
            x[m] += diff * (r / tot)
        x = {m: float(np.clip(x[m], lo[m], hi[m])) for m in materials}
    return x


def _complete_composition(free: dict[str, float]) -> tuple[dict[str, float], float]:
    """7 sampled materials -> full recipe: SodaF is the balance; out-of-bounds
    balance gets a soft penalty and the recipe is projected back into bounds."""
    soda = 100.0 - sum(free.values())
    lo, hi = bounds["SodaF"]
    pen = (lo - soda) * 50.0 if soda < lo else ((soda - hi) * 50.0 if soda > hi else 0.0)
    comp = dict(free)
    comp["SodaF"] = float(np.clip(soda, lo, hi))
    return _enforce_bounds(comp), float(pen)


def _summarize(row: pd.Series) -> dict:
    total = float(sum(row[f"{m}_wtpct"] for m in materials))
    return {
        "composition_wtpct": {m: round(float(row[f"{m}_wtpct"]), 4) for m in materials},
        "batch_total_wtpct": round(total, 3),
        "predicted": {t: round(float(row[t]), 5) for t in TARGET_COLS},   # MEASURED values
        "fixed_properties": dict(FIXED_PROPERTIES),
        "cost_Tk_per_kg": round(float(row["cost_Tk_per_kg"]), 4),
        "CO2_kg_per_kg": round(float(row["CO2_kg_per_kg"]), 5),
        "source": "real_experimental_batch (properties are MEASURED, not predicted)",
    }


def _composition_extrapolation_score(comp: dict[str, float]) -> float:
    x_sc = _comp_scaler.transform(np.array([[comp[m] for m in materials]]))
    d, _ = _comp_nbrs.kneighbors(x_sc, n_neighbors=1)
    return float(d[0][0])


def _property_sanity_penalty(pred_raw: dict[str, float]) -> float:
    pen = 0.0
    for key, val in pred_raw.items():
        lo, hi, rng = _pr[f"{key}_min"], _pr[f"{key}_max"], _prop_ranges[key]
        margin = rng * PROPERTY_SANITY_MARGIN
        if val < lo - margin:
            pen += (lo - margin - val) / rng
        elif val > hi + margin:
            pen += (val - hi - margin) / rng
    return pen


def _predict_clamped(comp: dict[str, float]) -> tuple[dict, dict]:
    raw = np.asarray(forward_model.predict(build_input_row(comp))[0], dtype=float)
    raw_d, cl_d = {}, {}
    for i, key in enumerate(TARGET_COLS):
        lo, hi, rng = _pr[f"{key}_min"], _pr[f"{key}_max"], _prop_ranges[key]
        raw_d[key] = float(raw[i])
        cl_d[key] = float(max(np.clip(raw[i], lo - PREDICTION_CLAMP_MULTIPLE * rng,
                                      hi + PREDICTION_CLAMP_MULTIPLE * rng), 0.0))
    return raw_d, cl_d


def _cost_co2(comp: dict[str, float]) -> tuple[float, float]:
    total = sum(comp[m] for m in materials)
    return (sum(comp[m] * cost_dict[m] for m in materials) / total,
            sum(comp[m] * co2_dict[m] for m in materials) / total)


def _evaluate(comp: dict[str, float], MOR_tgt: float, WA_tgt: float,
              k: float, soda_pen: float = 0.0, S_tgt: float | None = None,
              S_tol: float | None = None) -> dict:
    """Everything the optimisers need about one recipe."""
    raw, pred = _predict_clamped(comp)
    cost, co2 = _cost_co2(comp)
    MOR_eff = pred["MOR_MPa"] - k * HALFWIDTH.get("MOR_MPa", 0.0)
    WA_eff = pred["WA_pct"] + k * HALFWIDTH.get("WA_pct", 0.0)
    mor_short = max(0.0, MOR_tgt - MOR_eff) / _prop_ranges["MOR_MPa"]
    wa_dev = (abs(WA_eff - WA_tgt) if WA_PENALTY_SYMMETRIC else max(0.0, WA_eff - WA_tgt)) \
        / _prop_ranges["WA_pct"]
    s_dev = 0.0
    if S_tgt is not None and SHRINK_AVAILABLE:
        tol = DEFAULT_SHRINK_TOL if S_tol is None else S_tol
        s_dev = max(0.0, abs(pred["Shrinkage_pct"] - S_tgt) + k * HALFWIDTH.get("Shrinkage_pct", 0.0) - tol) \
            / _prop_ranges["Shrinkage_pct"]
    violation = mor_short + wa_dev + s_dev              # unweighted, range-normalised
    extrap = _composition_extrapolation_score(comp)
    overflow = max(0.0, extrap - _trust_region_threshold)
    sanity_raw = _property_sanity_penalty(raw)
    return {"pred": pred, "raw": raw, "cost": cost, "co2": co2,
            "norm_cost": (cost - _cost_min) / _cost_range,
            "norm_co2": (co2 - _co2_min) / _co2_range,
            "violation": violation, "soda_pen": soda_pen,
            "extrap": extrap, "overflow": overflow, "sanity_raw": sanity_raw,
            "feasible": (overflow == 0.0) and (sanity_raw == 0.0)}



def wa_exceed_probability(comp: dict[str, float], limit: float) -> float | None:
    """P(WA > limit). Tobit model if WA is left-censored in the training data, otherwise a Gaussian
    residual model around the forward-model prediction; None if neither is in metadata.json."""
    if _WA_TOBIT:
        x = np.array([[float(comp[f[:-len("_wtpct")]]) for f in _WA_TOBIT["features"]]])
        z = (x - np.array(_WA_TOBIT["mean"])) / np.array(_WA_TOBIT["scale"])
        mu = float(_WA_TOBIT["b0"] + float(z.ravel() @ np.array(_WA_TOBIT["coef"])))
        return float(1.0 - _norm.cdf((limit - mu) / _WA_TOBIT["sigma"]))
    if _WA_EXCEED and _WA_EXCEED.get("model") == "gaussian_residual" and "WA_pct" in TARGET_COLS:
        mu = _predict_clamped(comp)[0]["WA_pct"]
        return float(1.0 - _norm.cdf((limit - mu) / _WA_EXCEED["sigma"]))
    return None


def _design_wa(WA_tgt: float) -> float:
    return WA_tgt if WA_DESIGN_LIMIT is None else min(WA_tgt, WA_DESIGN_LIMIT)


def _score(e: dict) -> float:
    """The scalarised objective of one evaluated recipe (no soda-balance term)."""
    extrap_pen = min(e["overflow"] / max(_trust_region_threshold, 1e-9) * EXTRAP_PENALTY_WEIGHT, PENALTY_CAP)
    sanity_pen = min(e["sanity_raw"] * SANITY_PENALTY_WEIGHT, PENALTY_CAP)
    return e["norm_cost"] + e["norm_co2"] + PROPERTY_PENALTY_WEIGHT * e["violation"] + extrap_pen + sanity_pen


def _polish(comp0: dict[str, float], MOR_tgt: float, WA_tgt: float, k: float,
            S_tgt: float | None = None, S_tol: float | None = None) -> tuple[dict, bool]:
    """Local SLSQP refinement of a TPE recipe: minimise cost~ + CO2~ subject to the spec at
    margin k and the SodaF bounds. The result is accepted only if it is inside the training
    domain and has a better scalarised score than the starting recipe."""
    free = _free_materials()

    def full(x):
        c = {m: float(v) for m, v in zip(free, x)}
        c["SodaF"] = 100.0 - sum(c.values())
        return c

    def f(x):
        cost, co2 = _cost_co2(full(x))
        return (cost - _cost_min) / _cost_range + (co2 - _co2_min) / _co2_range

    def g_mor(x):
        return (_predict_clamped(full(x))[1]["MOR_MPa"] - k * HALFWIDTH.get("MOR_MPa", 0.0) - MOR_tgt) \
            / _prop_ranges["MOR_MPa"] - POLISH_MARGIN

    def g_wa(x):
        return (WA_tgt - _predict_clamped(full(x))[1]["WA_pct"] - k * HALFWIDTH.get("WA_pct", 0.0)) \
            / _prop_ranges["WA_pct"] - POLISH_MARGIN

    lo_s, hi_s = bounds["SodaF"]
    cons = [{"type": "ineq", "fun": g_mor}, {"type": "ineq", "fun": g_wa},
            {"type": "ineq", "fun": lambda x: (100.0 - sum(x)) - lo_s},
            {"type": "ineq", "fun": lambda x: hi_s - (100.0 - sum(x))}]
    if S_tgt is not None and SHRINK_AVAILABLE:
        tol = DEFAULT_SHRINK_TOL if S_tol is None else S_tol
        hw_s = HALFWIDTH.get("Shrinkage_pct", 0.0)
        # |S - S*| + k*hw <= tol  <=>  two linear-in-S inequalities (smooth for SLSQP)
        cons.append({"type": "ineq", "fun": lambda x: (tol - k * hw_s - (_predict_clamped(full(x))[1]["Shrinkage_pct"] - S_tgt))
                     / _prop_ranges["Shrinkage_pct"] - POLISH_MARGIN})
        cons.append({"type": "ineq", "fun": lambda x: (tol - k * hw_s + (_predict_clamped(full(x))[1]["Shrinkage_pct"] - S_tgt))
                     / _prop_ranges["Shrinkage_pct"] - POLISH_MARGIN})
    x0 = np.array([comp0[m] for m in free])
    try:
        r = _sp_minimize(f, x0, method="SLSQP", constraints=cons,
                         bounds=[tuple(bounds[m]) for m in free], options={"maxiter": 200, "ftol": 1e-10})
        cand = _enforce_bounds(full(r.x))
    except Exception:
        return comp0, False
    e_old = _evaluate(comp0, MOR_tgt, WA_tgt, k, S_tgt=S_tgt, S_tol=S_tol)
    best, best_score = comp0, _score(e_old)
    # The unconstrained optimum of a (near-)linear problem sits at the edge of the design box,
    # usually outside the training-data domain. Step toward it only as far as the trust region
    # and the spec allow: evaluate points along the segment comp0 -> cand.
    for t in (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05):
        pt = _enforce_bounds({m: comp0[m] + t * (cand[m] - comp0[m]) for m in materials})
        e_t = _evaluate(pt, MOR_tgt, WA_tgt, k, S_tgt=S_tgt, S_tol=S_tol)
        if e_t["feasible"] and e_t["violation"] <= max(COMPLIANCE_TOL, e_old["violation"]) \
                and _score(e_t) < best_score - 1e-12:
            best, best_score = pt, _score(e_t)
    return best, best is not comp0


def _interval(pred: dict) -> dict:
    return {t: [round(pred[t] - HALFWIDTH.get(t, 0.0), 5), round(pred[t] + HALFWIDTH.get(t, 0.0), 5)]
            for t in TARGET_COLS}


def evaluate_composition(comp: dict[str, float], MOR_tgt: float | None = None,
                         WA_tgt: float | None = None, S_tgt: float | None = None,
                         S_tol: float | None = None) -> dict:
    """Predict a given recipe (used by the app and by run_analyses.py)."""
    comp = {m: float(comp[m]) for m in materials}
    e = _evaluate(comp, MOR_tgt if MOR_tgt is not None else _pr["MOR_MPa_min"],
                  WA_tgt if WA_tgt is not None else _pr["WA_pct_max"], k=0.0, S_tgt=S_tgt, S_tol=S_tol)
    return {"predicted": {t: round(e["pred"][t], 5) for t in TARGET_COLS},
            "pred_interval_90": _interval(e["pred"]),
            "cost_Tk_per_kg": round(e["cost"], 4), "CO2_kg_per_kg": round(e["co2"], 5),
            "trust_distance": e["extrap"], "trust_threshold": _trust_region_threshold,
            "inside_training_domain": e["overflow"] == 0.0,
            "p_WA_exceeds_limit": wa_exceed_probability(comp, WA_tgt if WA_tgt is not None else _pr["WA_pct_max"])}


# ── Methods 1 and 2: nearest neighbours over the REAL batches ───────────────
def _nn_pool(MOR: float, WA: float, k: int | None = None, S_tgt: float | None = None,
             S_tol: float | None = None) -> tuple[np.ndarray, int]:
    """Indices of real batches ordered by scaled distance to the target, restricted to
    batches whose MEASURED properties meet the spec (MOR >= target, WA <= target and, if a
    shrinkage target is given, |S - S*| <= tolerance). Distance uses only the specified
    properties. If no real batch meets the spec, falls back to all batches (pool size 0 is
    returned so callers can warn). Using the same spec as the optimiser makes the baselines
    like-for-like; a plain nearest batch fails the MOR minimum roughly half the time."""
    MOR, WA = clamp_targets(MOR, WA)
    S_tgt = clamp_shrinkage(S_tgt)
    cols, qv = ["MOR_MPa", "WA_pct"], [MOR, WA]
    if S_tgt is not None:
        cols.append("Shrinkage_pct"); qv.append(S_tgt)
    idx = [TARGET_COLS.index(c) for c in cols]
    mu, sc = _prop_scaler.mean_[idx], _prop_scaler.scale_[idx]
    dist = np.linalg.norm((dataset[cols].values - mu) / sc - (np.array(qv) - mu) / sc, axis=1)
    ok = (dataset["MOR_MPa"].values >= MOR) & (dataset["WA_pct"].values <= WA)
    if S_tgt is not None:
        tol = DEFAULT_SHRINK_TOL if S_tol is None else S_tol
        ok &= np.abs(dataset["Shrinkage_pct"].values - S_tgt) <= tol
    pool = np.where(ok)[0]
    n_ok = len(pool)
    if n_ok == 0:
        pool = np.arange(len(dataset))
    order = pool[np.argsort(dist[pool])]
    return (order if k is None else order[:k]), n_ok


def inverse_non_optimized(MOR_MPa: float, WA_pct: float, S_tgt: float | None = None,
                          S_tol: float | None = None) -> dict:
    """Method 1: closest spec-compliant real batch (falls back to closest batch overall)."""
    idx, n_ok = _nn_pool(MOR_MPa, WA_pct, k=1, S_tgt=S_tgt, S_tol=S_tol)
    out = _summarize(dataset.iloc[idx[0]])
    out["n_spec_compliant_batches"] = n_ok
    return out


def inverse_optimized(MOR_MPa: float, WA_pct: float, S_tgt: float | None = None,
                      S_tol: float | None = None) -> tuple[dict, bool]:
    """Method 2: among the 10 closest spec-compliant real batches, lowest cost + CO2 rank."""
    idxs, n_ok = _nn_pool(MOR_MPa, WA_pct, k=10, S_tgt=S_tgt, S_tol=S_tol)
    sub = dataset.iloc[idxs].copy()
    sub["_score"] = sub["cost_Tk_per_kg"].rank(pct=True) + sub["CO2_kg_per_kg"].rank(pct=True)
    best = sub["_score"].idxmin()
    identical = (best == dataset.index[idxs[0]])
    if identical:
        print("  NOTE: Methods 1 and 2 returned the same batch (no cost/CO2 diversity "
              "among the nearest neighbours for this target).")
    out = _summarize(dataset.loc[best])
    out["n_spec_compliant_batches"] = n_ok
    return out, identical


# ── Method 3: scalarised Bayesian optimisation (TPE) ─────────────────────────
def _free_materials() -> list[str]:
    return [m for m in materials if m != "SodaF"]


def inverse_bayesian_optimization(MOR_MPa_tgt: float, WA_tgt: float, n_trials: int = 200,
                                  sampler=None, seed: int = 42, k_sigma: float | None = None,
                                  production_mode: bool = True, S_tgt: float | None = None,
                                  S_tol: float | None = None
                                  ) -> tuple[dict, list[float], optuna.Study]:
    """Scalarised BO. `sampler` swaps the acquisition strategy (e.g. GPSampler)
    on the IDENTICAL objective; `seed` seeds the default TPE sampler;
    `k_sigma` overrides SAFETY_FACTOR for this call. `production_mode` (default True) returns the
    cheapest spec-compliant domain-feasible trial when one exists and applies a local polish;
    analysis scripts pass False to study the raw search on the identical objective."""
    MOR_tgt, WA_req = clamp_targets(MOR_MPa_tgt, WA_tgt)
    WA_tgt = _design_wa(WA_req)                     # limit used inside the search
    S_tgt = clamp_shrinkage(S_tgt)
    S_tol = (DEFAULT_SHRINK_TOL if S_tol is None else S_tol) if S_tgt is not None else None
    k = SAFETY_FACTOR if k_sigma is None else k_sigma
    trial_vals: list[float] = []

    def _objective(trial: optuna.Trial) -> float:
        free = {m: trial.suggest_float(f"c_{m}", bounds[m][0], bounds[m][1]) for m in _free_materials()}
        comp, soda_pen = _complete_composition(free)
        e = _evaluate(comp, MOR_tgt, WA_tgt, k, soda_pen, S_tgt, S_tol)
        extrap_pen = min(e["overflow"] / max(_trust_region_threshold, 1e-9) * EXTRAP_PENALTY_WEIGHT, PENALTY_CAP)
        sanity_pen = min(e["sanity_raw"] * SANITY_PENALTY_WEIGHT, PENALTY_CAP)
        obj = (e["norm_cost"] + e["norm_co2"] + PROPERTY_PENALTY_WEIGHT * e["violation"]
               + soda_pen + extrap_pen + sanity_pen)
        trial.set_user_attr("feasible", bool(e["feasible"]))
        trial.set_user_attr("violation", float(e["violation"]))
        trial.set_user_attr("extrapolation_score", e["extrap"])
        trial.set_user_attr("comp", comp)
        trial_vals.append(obj)
        return obj

    sampler = sampler if sampler is not None else optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

    feasible_trials = [t for t in study.trials
                       if t.value is not None and t.user_attrs.get("feasible", False)]
    compliant_trials = [t for t in feasible_trials if t.user_attrs.get("violation", 1.0) <= COMPLIANCE_TOL]
    warning_msg = None
    if feasible_trials:
        pool = compliant_trials if (production_mode and compliant_trials) else feasible_trials
        best_trial = min(pool, key=lambda t: t.value)
    else:
        best_trial = study.best_trial
        warning_msg = ("Bayesian optimisation found NO recipe inside the training-data domain for "
                       f"this target in {n_trials} trials; returning the least-bad trial. Treat with "
                       "caution: relax the target, add trials, or add real batches in this region.")
        warnings.warn(warning_msg)

    best_c = best_trial.user_attrs["comp"]
    polished = False
    if production_mode and feasible_trials:
        best_c, polished = _polish(best_c, MOR_tgt, WA_tgt, k, S_tgt, S_tol)
    e = _evaluate(best_c, MOR_tgt, WA_req, 0.0, S_tgt=S_tgt, S_tol=S_tol)
    e_lo = _evaluate(best_c, MOR_tgt, WA_req, 1.0, S_tgt=S_tgt, S_tol=S_tol)   # pessimistic bounds
    e_k = _evaluate(best_c, MOR_tgt, WA_req, k, S_tgt=S_tgt, S_tol=S_tol)      # margin actually requested
    meets_k = bool(e_k["violation"] <= COMPLIANCE_TOL)
    if feasible_trials and not meets_k and warning_msg is None:
        warning_msg = (f"No recipe meeting the spec with safety margin k = {k:g} was found inside the "
                       "training domain. The returned recipe is the closest one; lower k, relax the "
                       "target, or add real batches in this region.")
    n_feas = len(feasible_trials)
    print(f"  Bayesian Optimisation: {n_feas}/{n_trials} trials were inside the training domain; "
          f"result selected from those." if n_feas else
          f"  Bayesian Optimisation: 0/{n_trials} trials inside the training domain - see warning.")
    pred = e["pred"]
    result = {
        "composition_wtpct": {m: round(best_c[m], 4) for m in materials},
        "batch_total_wtpct": round(sum(best_c.values()), 3),
        "predicted": {t: round(pred[t], 5) for t in TARGET_COLS},
        "pred_interval_90": _interval(pred),
        "fixed_properties": dict(FIXED_PROPERTIES),
        "meets_target_point_estimate": bool(pred["MOR_MPa"] >= MOR_tgt and pred["WA_pct"] <= WA_req
                                            and (S_tgt is None or abs(pred["Shrinkage_pct"] - S_tgt) <= S_tol)),
        "shrinkage_target": S_tgt, "shrinkage_tolerance": S_tol,
        "meets_target_with_90pct_margin": bool(e_lo["violation"] <= COMPLIANCE_TOL),
        "meets_target_at_k": meets_k, "polished": polished,
        "p_WA_exceeds_target": wa_exceed_probability(best_c, WA_req),
        "cost_Tk_per_kg": round(e["cost"], 4), "CO2_kg_per_kg": round(e["co2"], 5),
        "trust_distance": round(e["extrap"], 4), "trust_threshold": round(_trust_region_threshold, 4),
        "feasible": bool(feasible_trials), "n_feasible_trials": n_feas, "n_trials": n_trials,
        "warning": warning_msg, "safety_factor": k,
    }
    return result, trial_vals, study


# ── Genuinely multi-objective search (NSGA-II) ──────────────────────────────
def _pareto_mask(F: np.ndarray) -> np.ndarray:
    """True for non-dominated rows of F (all objectives minimised)."""
    n = len(F)
    keep = np.ones(n, bool)
    for i in range(n):
        dominated = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if dominated.any():
            keep[i] = False
    return keep


def inverse_pareto(MOR_MPa_tgt: float, WA_tgt: float, n_trials: int = 400, seed: int = 0,
                   k_sigma: float | None = None, warm_start: list[dict] | None = None,
                   S_tgt: float | None = None, S_tol: float | None = None) -> pd.DataFrame:
    """NSGA-II over three objectives (norm. cost, norm. CO2, spec violation)
    with the trust region as a constraint. Returns every domain-feasible trial
    with a `pareto` flag (non-dominated among the three objectives) and a
    `compliant` flag (violation within COMPLIANCE_TOL). No weights are involved.
    `warm_start`: optional list of recipes (wt% dicts) queued as the first trials, so the search
    cannot miss regions already known to be good (e.g. the scalarised-BO solutions)."""
    MOR_tgt, WA_tgt = clamp_targets(MOR_MPa_tgt, WA_tgt)
    WA_tgt = _design_wa(WA_tgt)
    S_tgt = clamp_shrinkage(S_tgt)
    S_tol = (DEFAULT_SHRINK_TOL if S_tol is None else S_tol) if S_tgt is not None else None
    k = SAFETY_FACTOR if k_sigma is None else k_sigma

    def _obj(trial: optuna.Trial):
        free = {m: trial.suggest_float(f"c_{m}", bounds[m][0], bounds[m][1]) for m in _free_materials()}
        comp, soda_pen = _complete_composition(free)
        e = _evaluate(comp, MOR_tgt, WA_tgt, k, soda_pen, S_tgt, S_tol)
        trial.set_user_attr("comp", comp)
        trial.set_user_attr("e", {kk: e[kk] for kk in ("cost", "co2", "violation", "extrap",
                                                       "overflow", "sanity_raw", "pred")})
        trial.set_user_attr("constraint", [e["overflow"], e["sanity_raw"], soda_pen])
        return e["norm_cost"], e["norm_co2"], e["violation"]

    sampler = optuna.samplers.NSGAIISampler(
        seed=seed, population_size=40,
        constraints_func=lambda t: t.user_attrs["constraint"])
    study = optuna.create_study(directions=["minimize"] * 3, sampler=sampler)
    for comp0 in (warm_start or []):
        study.enqueue_trial({f"c_{m}": float(np.clip(comp0[m], *bounds[m])) for m in _free_materials()})
    study.optimize(_obj, n_trials=n_trials, show_progress_bar=False)

    rows = []
    for t in study.trials:
        if t.state.name != "COMPLETE" or max(t.user_attrs["constraint"]) > 0:
            continue
        e, comp = t.user_attrs["e"], t.user_attrs["comp"]
        rows.append({"cost_Tk_per_kg": e["cost"], "CO2_kg_per_kg": e["co2"], "violation": e["violation"],
                     "MOR_pred": e["pred"]["MOR_MPa"], "WA_pred": e["pred"]["WA_pct"],
                     "Shrinkage_pred": e["pred"].get("Shrinkage_pct", float("nan")),
                     "trust_distance": e["extrap"], **{m: comp[m] for m in materials}})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    F = df[["cost_Tk_per_kg", "CO2_kg_per_kg", "violation"]].values
    df["pareto"] = _pareto_mask(F)
    df["compliant"] = df["violation"] <= COMPLIANCE_TOL
    return df
