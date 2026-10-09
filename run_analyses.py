#!/usr/bin/env python3
"""
run_analyses.py
All post-training analyses of the inverse-design framework in ONE script. It replaces the former
sensitivity_analysis.py, tpe_vs_ei_comparison.py, pareto_analysis.py and co2_robustness.py; every
analysis, setting, seed and output file name is unchanged.

PROJECT LAYOUT AND RUN ORDER
  model_utils.py          MultiTargetModel (needed to unpickle the model) + dated price/CO2 loader
  train_forward_model.py  1) trains the forward model on the 238 real batches -> data/, models/
  inverse_design.py       backend library (three methods + NSGA-II); imported, never run directly
  run_analyses.py         2) this file: the four analyses below
  streamlit_app.py        3) decision-support app:  streamlit run streamlit_app.py

USAGE
  python run_analyses.py                     # all four analyses, in the order below
  python run_analyses.py sensitivity         # one analysis
  python run_analyses.py samplers pareto     # several
  QUICK=1 python run_analyses.py all         # smoke test (tiny budgets)
Requires train_forward_model.py to have been run.

THE FOUR ANALYSES
  sensitivity  Sensitivity of the Bayesian-optimisation recommendation to the design choices that
               reviewers asked to have justified (Reviewers 1, 2, 4, 7, 8).
                 PROPERTY_PENALTY_WEIGHT w   [1, 2.5, 5, 10, 20]  property : cost exchange rate is w : 1
                 EXTRAP_PENALTY_WEIGHT       [5, 10, 20]          trust-region guardrail
                 SANITY_PENALTY_WEIGHT       [10, 20, 40]         implausible-prediction guardrail
                 WA_PENALTY_SYMMETRIC        [False, True]        one-sided vs symmetric WA penalty
                 SAFETY_FACTOR k             [0, 0.5, 1, 2]       margin in 90 % half-widths
               Repeats use different Optuna seeds; targets are specification-anchored (MOR minimum
               at dataset quantiles; WA maximum at the ISO 13006 BIa limit; a tight-WA case at the
               20th WA percentile of the data; and, if shrinkage is modelled, a mid-MOR case with a
               shrinkage band = median +/- DEFAULT_SHRINK_TOL).
               The sweeps run in raw-search mode (production_mode=False) to isolate each weight; a
               final production-mode check reports how often the DELIVERED method meets the spec at
               k = 0, 1, 2. NOTE on the ratio: cost~ and CO2~ each span [0, 1], so the exchange
               rate is exactly w : 1 (not "7.5 : 1"; Reviewer 8).
  samplers     TPE (production sampler) vs GP_EI (Optuna GPSampler = "vanilla" BO; Reviewer 7) vs
               Random (no-model baseline) on the IDENTICAL objective, guardrails and feasibility
               logic; 5 seeds; paired one-sided Wilcoxon tests on the final objective. Caveat for
               the paper: the objective is non-smooth (max(), caps, hard guardrails), a known
               weakness for GP surrogates, so TPE's advantage is an empirical result for THIS
               objective, not a general claim. Pooled pairs share seeds across four targets and are
               not fully independent. Degrades to TPE + Random if torch/GPSampler is unavailable.
  pareto       NSGA-II over (norm. cost, norm. CO2, spec violation) with the trust region as a
               constraint and no weights, compared with the scalarised-BO solutions. Two fronts per
               scenario: INDEPENDENT (no knowledge of BO) and SEEDED (BO recipes queued as the
               first trials). A BO recipe counts as dominated only if beaten by more than 0.1 % on
               an objective. 'bo_cost_below_independent_front_min' > 0 flags an under-sampled
               independent front; judge dominance with the seeded front then. Only BO recipes that
               meet the spec at margin k are compared. The cost and CO2 spans are small in absolute
               terms (narrow design box): compare them with the model's prediction uncertainty.
  co2          Are the CO2 conclusions robust to the emission factors? (Reviewers 1 and 6)
               Q1 rank stability: Spearman rho of CO2/kg across the 238 real batches (and overlap of
                  the 10 lowest-CO2 batches) between the baseline and each alternative factor set.
               Q2 recommendation stability / regret: what is lost by using the recipe optimised
                  under the BASELINE factors instead of the one optimised under the alternative
                  factors (both evaluated with the alternative factors)?
                    objective regret = score(baseline recipe) - score(alternative-optimal recipe),
                    score = cost~ + CO2~ (the optimiser's own objective, renormalised with the
                    alternative factors), reported x100 = "points of range". Cost and CO2 are
                    normalised by their ranges across the batches (printed when the analysis
                    starts; they are narrow relative to the mean), so a 2 % cost and 4 % CO2 penalty
                    can already be many points: quote the regret to rank scenarios and
                    cost_diff_pct / co2_diff_pct to describe losses in physical terms. A CO2-only
                    regret can be negative legitimately (the optimiser trades cost against CO2).
                  The search is stochastic (single runs differ by 1-4 % in CO2 for identical
                  factors), so each recipe is the best of BEST_OF seeds and the baseline recipe is
                  offered as a candidate in every scenario: regret >= 0 by construction and is a
                  LOWER BOUND. The 'reseed' scenario (same factors, new seeds) is the noise floor.
               Scenarios: reseed; each of the 8 factors x0.5 / x1.5; crushed_tile_zero and
               waste_zero (cut-off allocation); AG98_april_0.48; MC_<i> (all factors x U(0.5, 1.5)).

OUTPUT (data/ and plots/)
  sensitivity: sensitivity_analysis.csv, sensitivity_summary_<w>.csv, sensitivity_by_target_<w>.csv,
               sensitivity_production_mode(.csv|_summary.csv), plots/sensitivity_<w>.pdf/.png
  samplers:    tpe_vs_ei_comparison.csv, tpe_vs_ei_summary.csv, tpe_vs_ei_paired_tests.csv,
               (tpe_vs_ei_failures.csv), plots/tpe_vs_ei_convergence.pdf/.png
  pareto:      pareto_<scenario>.csv, pareto_seeded_<scenario>.csv, pareto_summary.csv,
               plots/pareto_front.pdf/.png
  co2:         co2_rank_stability.csv, co2_robustness_runs.csv, co2_robustness_summary.csv,
               plots/co2_robustness.png/.pdf
"""
from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

import inverse_design as idsg

ROOT = Path(__file__).parent
DATADIR, PLOTDIR = ROOT / "data", ROOT / "plots"
DATADIR.mkdir(exist_ok=True, parents=True)
PLOTDIR.mkdir(exist_ok=True, parents=True)

QUICK = os.environ.get("QUICK", "0") == "1"
dataset = pd.read_csv(DATADIR / "dataset.csv")
WA_SPEC = float(idsg.meta.get("iso13006_bia", {}).get("wa_max_pct", 0.5))
materials = idsg.materials


def _targets(with_extra: bool, quick_keys: tuple) -> dict:
    """name -> (MOR minimum [MPa], WA maximum [%]) or, for a shrinkage case, (MOR, WA, S target,
    S tolerance). Anchored to dataset quantiles and ISO 13006. `with_extra` adds a tight-WA case
    (20th WA percentile of the data) and, if shrinkage is modelled, a shrinkage-band case."""
    q = lambda p: float(dataset["MOR_MPa"].quantile(p))
    t = {"low_MOR": (q(0.10), WA_SPEC), "mid_MOR": (q(0.50), WA_SPEC), "high_MOR": (q(0.90), WA_SPEC)}
    if with_extra:
        t["tight_WA"] = (q(0.50), round(float(dataset["WA_pct"].quantile(0.20)), 2))
        if idsg.SHRINK_AVAILABLE:
            t["mid_MOR_shrink"] = (q(0.50), WA_SPEC, round(float(dataset["Shrinkage_pct"].median()), 2),
                                   float(idsg.DEFAULT_SHRINK_TOL))
    return {k: t[k] for k in quick_keys if k in t} if QUICK else t


def _skw(v: tuple) -> dict:
    """Shrinkage keyword arguments for the idsg functions (empty for 2-property targets)."""
    return {"S_tgt": v[2], "S_tol": v[3]} if len(v) >= 4 else {}


@contextlib.contextmanager
def _patched(name: str, value):
    original = getattr(idsg, name)
    setattr(idsg, name, value)
    try:
        yield
    finally:
        setattr(idsg, name, original)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 1. SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════════════════════
def sensitivity() -> None:
    n_trials = 60 if QUICK else 200
    n_seeds = 2 if QUICK else 5
    prod_k = [0.0, 1.0, 2.0]
    targets = _targets(True, ("mid_MOR", "tight_WA", "mid_MOR_shrink"))
    baseline = {n: getattr(idsg, n) for n in (
        "PROPERTY_PENALTY_WEIGHT", "EXTRAP_PENALTY_WEIGHT", "SANITY_PENALTY_WEIGHT",
        "WA_PENALTY_SYMMETRIC", "SAFETY_FACTOR")}
    sweeps = {
        "PROPERTY_PENALTY_WEIGHT": [1.0, 2.5, 5.0, 10.0, 20.0],
        "EXTRAP_PENALTY_WEIGHT": [5.0, 10.0, 20.0],
        "SANITY_PENALTY_WEIGHT": [10.0, 20.0, 40.0],
        "WA_PENALTY_SYMMETRIC": [False, True],
        "SAFETY_FACTOR": [0.0, 0.5, 1.0, 2.0],
    }
    if QUICK:
        sweeps = {"PROPERTY_PENALTY_WEIGHT": [2.5, 5.0], "SAFETY_FACTOR": [0.0, 1.0]}

    def run_one(weight_name, weight_val, target_name, tv, seed, nn_ref) -> dict:
        mor_t, wa_t, skw = tv[0], tv[1], _skw(tv)
        with _patched(weight_name, weight_val):
            result, _, _ = idsg.inverse_bayesian_optimization(
                mor_t, wa_t, n_trials=n_trials, seed=seed, production_mode=False, **skw)
        pred, comp = result["predicted"], result["composition_wtpct"]
        return {
            "weight_name": weight_name, "weight_value": float(weight_val), "target_name": target_name,
            "seed": seed, "target_MOR_min": mor_t, "target_WA_max": wa_t,
            "pred_MOR": pred["MOR_MPa"], "pred_WA": pred["WA_pct"],
            "target_S": skw.get("S_tgt", np.nan), "pred_S": pred.get("Shrinkage_pct", np.nan),
            "S_margin": (skw["S_tol"] - abs(pred["Shrinkage_pct"] - skw["S_tgt"])) if skw else np.nan,
            "MOR_margin": pred["MOR_MPa"] - mor_t,              # >= 0 means spec met
            "WA_margin": wa_t - pred["WA_pct"],                 # >= 0 means spec met
            "spec_met_point": result["meets_target_point_estimate"],
            "spec_met_with_90pct_margin": result["meets_target_with_90pct_margin"],
            "cost_Tk_per_kg": result["cost_Tk_per_kg"], "CO2_kg_per_kg": result["CO2_kg_per_kg"],
            "cost_change_vs_NN_pct": (result["cost_Tk_per_kg"] / nn_ref["cost_Tk_per_kg"] - 1) * 100,
            "CO2_change_vs_NN_pct": (result["CO2_kg_per_kg"] / nn_ref["CO2_kg_per_kg"] - 1) * 100,
            "feasible": result["feasible"], "trust_ratio": result["trust_distance"] / result["trust_threshold"],
            **{f"comp_{m}": comp[m] for m in materials},
        }

    def production_check(nn_refs) -> None:
        rows = []
        for k in prod_k:
            for tname, tv in targets.items():
                mor_t, wa_t = tv[0], tv[1]
                for seed in range(n_seeds):
                    res, _, _ = idsg.inverse_bayesian_optimization(
                        mor_t, wa_t, n_trials=n_trials, seed=seed, k_sigma=k, production_mode=True, **_skw(tv))
                    nn = nn_refs[tname]
                    rows.append({
                        "k": k, "target_name": tname, "seed": seed,
                        "spec_met_point": res["meets_target_point_estimate"],
                        "spec_met_at_k": res["meets_target_at_k"],
                        "spec_met_with_90pct_margin": res["meets_target_with_90pct_margin"],
                        "polished": res["polished"], "feasible": res["feasible"],
                        "no_compliant_recipe_found": res["warning"] is not None,
                        "cost_Tk_per_kg": res["cost_Tk_per_kg"], "CO2_kg_per_kg": res["CO2_kg_per_kg"],
                        "cost_change_vs_NN_pct": (res["cost_Tk_per_kg"] / nn["cost_Tk_per_kg"] - 1) * 100,
                        "CO2_change_vs_NN_pct": (res["CO2_kg_per_kg"] / nn["CO2_kg_per_kg"] - 1) * 100,
                        "trust_ratio": res["trust_distance"] / res["trust_threshold"],
                        "p_WA_exceeds_target": res["p_WA_exceeds_target"]})
        d = pd.DataFrame(rows)
        d.to_csv(DATADIR / "sensitivity_production_mode.csv", index=False)
        summ = d.groupby(["k", "target_name"]).agg(
            spec_met_at_k=("spec_met_at_k", "mean"), spec_met_90=("spec_met_with_90pct_margin", "mean"),
            polished=("polished", "mean"), no_compliant=("no_compliant_recipe_found", "mean"),
            cost=("cost_Tk_per_kg", "mean"), CO2=("CO2_kg_per_kg", "mean"),
            cost_vs_NN_pct=("cost_change_vs_NN_pct", "mean"), CO2_vs_NN_pct=("CO2_change_vs_NN_pct", "mean"),
            trust_ratio_max=("trust_ratio", "max"), p_WA_exceeds=("p_WA_exceeds_target", "mean")).reset_index()
        summ.to_csv(DATADIR / "sensitivity_production_mode_summary.csv", index=False)
        print("\n=== PRODUCTION MODE (the delivered method): spec compliance by safety factor and target ===")
        print(summ.round(4).to_string(index=False))
        print("\nOverall by k:")
        print(d.groupby("k")[["spec_met_at_k", "spec_met_with_90pct_margin", "polished"]].mean().round(3).to_string())

    idsg.refresh_prices()
    nn_refs = {n: idsg.inverse_non_optimized(t[0], t[1], **_skw(t)) for n, t in targets.items()}
    rows = []
    for wname, values in sweeps.items():
        print(f"\n=== {wname} (baseline {baseline[wname]}): {values} ===")
        for val in values:
            for tname, tv in targets.items():
                for seed in range(n_seeds):
                    rows.append(run_one(wname, val, tname, tv, seed, nn_refs[tname]))
            print(f"  {wname}={val}: done")
    df = pd.DataFrame(rows)
    df.to_csv(DATADIR / "sensitivity_analysis.csv", index=False)
    print(f"\nSaved: data/sensitivity_analysis.csv ({len(df)} runs)")

    for wname in sweeps:
        sub = df[df.weight_name == wname]
        agg = sub.groupby("weight_value").agg(
            MOR_margin_mean=("MOR_margin", "mean"), MOR_margin_sd=("MOR_margin", "std"),
            WA_margin_mean=("WA_margin", "mean"), WA_margin_sd=("WA_margin", "std"),
            spec_met_point=("spec_met_point", "mean"),
            spec_met_with_margin=("spec_met_with_90pct_margin", "mean"),
            cost_mean=("cost_Tk_per_kg", "mean"), cost_sd=("cost_Tk_per_kg", "std"),
            CO2_mean=("CO2_kg_per_kg", "mean"), CO2_sd=("CO2_kg_per_kg", "std"),
            cost_change_vs_NN_pct=("cost_change_vs_NN_pct", "mean"),
            CO2_change_vs_NN_pct=("CO2_change_vs_NN_pct", "mean"),
            feasible_share=("feasible", "mean"),
        ).reset_index()
        agg.to_csv(DATADIR / f"sensitivity_summary_{wname}.csv", index=False)
        per_t = sub.groupby(["weight_value", "target_name"]).agg(
            MOR_margin=("MOR_margin", "mean"), spec_met_point=("spec_met_point", "mean"),
            spec_met_with_margin=("spec_met_with_90pct_margin", "mean"),
            cost=("cost_Tk_per_kg", "mean"), CO2=("CO2_kg_per_kg", "mean")).reset_index()
        per_t.to_csv(DATADIR / f"sensitivity_by_target_{wname}.csv", index=False)
        print(f"\n{wname}\n" + agg[["weight_value", "MOR_margin_mean", "WA_margin_mean", "spec_met_point",
                                     "spec_met_with_margin", "cost_mean", "CO2_mean",
                                     "feasible_share"]].round(4).to_string(index=False))

        fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
        x = agg["weight_value"].astype(float)
        axes[0].errorbar(x, agg["MOR_margin_mean"], agg["MOR_margin_sd"], fmt="o-", capsize=3, label="MOR margin (MPa)")
        axes[0].axhline(0, color="k", lw=0.8)
        axes[0].set_title("Achieved MOR margin (pred - target)")
        axes[1].errorbar(x, agg["WA_margin_mean"], agg["WA_margin_sd"], fmt="s-", capsize=3, color="#6A1B9A")
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].set_title("Achieved WA margin (target - pred, %)")
        axes[2].errorbar(x, agg["cost_mean"], agg["cost_sd"], fmt="o-", capsize=3, color="#D32F2F")
        axes[2].set_title("Cost (Tk/kg)")
        axes[3].errorbar(x, agg["CO2_mean"], agg["CO2_sd"], fmt="o-", capsize=3, color="#388E3C")
        axes[3].set_title("CO2 (kg/kg)")
        for ax in axes:
            ax.axvline(float(baseline[wname]), color="gray", ls="--", alpha=0.6)
            ax.set_xlabel(wname)
            ax.grid(True, ls="--", alpha=0.4)
        fig.suptitle(f"Sensitivity to {wname} (mean +/- sd over {len(targets)} targets x {n_seeds} seeds; "
                     f"dashed = baseline)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(PLOTDIR / f"sensitivity_{wname}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)

    production_check(nn_refs)

    for name, val in baseline.items():
        assert getattr(idsg, name) == val, f"{name} not restored"
    print("\nAll constants restored to baseline. Sensitivity done.")


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 2. SAMPLER COMPARISON (TPE vs GP-EI vs Random)
# ══════════════════════════════════════════════════════════════════════════════════════════════
def _gp_sampler_usable() -> tuple[bool, str]:
    """GPSampler imports torch lazily, only after its startup trials (default 10).
    Probe with 15 trials so a broken torch install is detected up front."""
    if not hasattr(optuna.samplers, "GPSampler"):
        return False, "optuna.samplers.GPSampler not found (need optuna>=3.6)"
    try:
        s = optuna.create_study(sampler=optuna.samplers.GPSampler(seed=0))
        s.optimize(lambda t: t.suggest_float("x", 0.0, 1.0) ** 2, n_trials=15)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def samplers() -> None:
    n_trials = 40 if QUICK else 200
    seeds = [0, 1] if QUICK else [0, 1, 2, 3, 4]
    targets = _targets(True, ("mid_MOR", "mid_MOR_shrink"))

    idsg.refresh_prices()
    gp_ok, gp_err = _gp_sampler_usable()
    samplers_ = {"TPE": lambda s: optuna.samplers.TPESampler(seed=s),
                 "Random": lambda s: optuna.samplers.RandomSampler(seed=s)}
    if gp_ok:
        samplers_["GP_EI"] = lambda s: optuna.samplers.GPSampler(seed=s)
    else:
        print("\n" + "=" * 78 + f"\nGP_EI unavailable: {gp_err}\nUsual cause: torch is missing or broken "
              "(CUDA wheel without driver, missing VC++ redistributable, MKL/OpenMP clash).\n"
              "Fix: pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
              "Continuing with TPE + Random only.\n" + "=" * 78 + "\n")

    rows, traj, failures = [], {}, []
    for tname, tv in targets.items():
        mor_t, wa_t = tv[0], tv[1]
        for sname, factory in samplers_.items():
            traj[(sname, tname)] = []
            for seed in seeds:
                print(f"target={tname}  sampler={sname}  seed={seed}")
                t0 = time.perf_counter()
                try:
                    res, vals, study = idsg.inverse_bayesian_optimization(
                        mor_t, wa_t, n_trials=n_trials, sampler=factory(seed), production_mode=False, **_skw(tv))
                except Exception as e:
                    print(f"  FAILED ({type(e).__name__}: {e}) - skipped")
                    failures.append({"sampler": sname, "target": tname, "seed": seed, "error": str(e)})
                    continue
                el = time.perf_counter() - t0
                rb = np.minimum.accumulate(np.asarray(vals))
                traj[(sname, tname)].append(rb)
                n_feas = sum(1 for t in study.trials if t.user_attrs.get("feasible", False))
                p = res["predicted"]
                rows.append({
                    "sampler": sname, "target_name": tname, "seed": seed,
                    "final_best_objective": float(rb[-1]), "elapsed_seconds": el,
                    "feasible_rate": n_feas / n_trials, "recommendation_feasible": res["feasible"],
                    "MOR_margin": p["MOR_MPa"] - mor_t, "WA_margin": wa_t - p["WA_pct"],
                    "spec_met_point": res["meets_target_point_estimate"],
                    "spec_met_with_90pct_margin": res["meets_target_with_90pct_margin"],
                    "cost_Tk_per_kg": res["cost_Tk_per_kg"], "CO2_kg_per_kg": res["CO2_kg_per_kg"]})

    df = pd.DataFrame(rows)
    if df.empty:
        print("No successful runs.")
        return
    df.to_csv(DATADIR / "tpe_vs_ei_comparison.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(DATADIR / "tpe_vs_ei_failures.csv", index=False)
    print(f"\nSaved: data/tpe_vs_ei_comparison.csv ({len(df)} runs)")

    summary = df.groupby(["target_name", "sampler"]).agg(
        mean_final_objective=("final_best_objective", "mean"),
        sd_final_objective=("final_best_objective", "std"),
        mean_elapsed_s=("elapsed_seconds", "mean"),
        mean_feasible_rate=("feasible_rate", "mean"),
        spec_met_share=("spec_met_point", "mean"),
        spec_met_with_margin_share=("spec_met_with_90pct_margin", "mean"),
        mean_MOR_margin=("MOR_margin", "mean"), mean_WA_margin=("WA_margin", "mean"),
        mean_cost=("cost_Tk_per_kg", "mean"), mean_CO2=("CO2_kg_per_kg", "mean")).reset_index()
    summary.to_csv(DATADIR / "tpe_vs_ei_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print("\nNOTE: spec_met_share is the POINT-estimate spec; spec_met_with_margin_share uses the "
          "pessimistic 90 % bounds and is the relevant one for k >= 1.")

    # Paired tests on the final objective (same target & seed), TPE vs each rival
    piv = df.pivot_table(index=["target_name", "seed"], columns="sampler", values="final_best_objective")
    test_rows = []
    for rival in [c for c in piv.columns if c != "TPE"]:
        pair = piv[["TPE", rival]].dropna()
        if len(pair) >= 6 and (pair["TPE"] - pair[rival]).abs().sum() > 0:
            stat, p = wilcoxon(pair["TPE"], pair[rival], alternative="less")
            test_rows.append({"comparison": f"TPE < {rival}", "n_pairs": len(pair),
                              "n_pairs_TPE_better": int((pair["TPE"] < pair[rival]).sum()),
                              "median_diff": float((pair["TPE"] - pair[rival]).median()), "wilcoxon_p": float(p)})
    if test_rows:
        pd.DataFrame(test_rows).to_csv(DATADIR / "tpe_vs_ei_paired_tests.csv", index=False)
        print("\nPaired one-sided Wilcoxon (H1: TPE reaches a LOWER final objective):")
        print(pd.DataFrame(test_rows).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        print("Report p-values in scientific notation or as 'p < 0.001' (they are tiny, not zero).")

    colors = {"TPE": "#1565C0", "GP_EI": "#D32F2F", "Random": "#757575"}
    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5), squeeze=False)
    for ax, tname in zip(axes[0], targets):
        for sname in samplers_:
            runs = traj.get((sname, tname), [])
            if not runs:
                continue
            arr = np.array(runs)
            x = np.arange(1, arr.shape[1] + 1)
            ax.plot(x, arr.mean(0), label=sname, color=colors[sname], lw=2)
            ax.fill_between(x, arr.min(0), arr.max(0), color=colors[sname], alpha=0.15)
        ax.set_title(tname)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Running-best objective")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.4)
    fig.suptitle(f"Search-strategy convergence (n={len(seeds)} seeds, {n_trials} trials)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(PLOTDIR / f"tpe_vs_ei_convergence.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved: plots/tpe_vs_ei_convergence.pdf / .png")


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 3. PARETO (NSGA-II) vs SCALARISED BO
# ══════════════════════════════════════════════════════════════════════════════════════════════
def _front_2d(df):
    comp = df[df.compliant]
    if not len(comp):
        return comp
    return comp[idsg._pareto_mask(comp[["cost_Tk_per_kg", "CO2_kg_per_kg"]].values)]


def _n_dominated(front, bo_pts, tol):
    """Number of BO points MATERIALLY dominated: some front point is no worse on both objectives
    and better by more than `tol` (relative) on at least one. Differences below tol are numerical
    noise (e.g. the recipe being re-evaluated after clipping to the bounds), not dominance."""
    n = 0
    for c, o in bo_pts:
        if len(front) and ((front.cost_Tk_per_kg <= c) & (front.CO2_kg_per_kg <= o) &
                           ((front.cost_Tk_per_kg < c * (1 - tol)) | (front.CO2_kg_per_kg < o * (1 - tol)))).any():
            n += 1
    return n


def pareto() -> None:
    n_trials = 150 if QUICK else 4000
    n_bo_seeds = 2 if QUICK else 5
    dom_tol = 1e-3                       # a BO recipe is dominated only if beaten by > 0.1 % on an objective
    scen = _targets(False, ("mid_MOR",))

    idsg.refresh_prices()
    fig, axes = plt.subplots(1, len(scen), figsize=(6.2 * len(scen), 5), squeeze=False)
    summ = []
    for ax, (name, (mor, wa)) in zip(axes[0], scen.items()):
        bo_rows, bo_comps = [], []
        for s in range(n_bo_seeds):
            r, _, _ = idsg.inverse_bayesian_optimization(mor, wa, n_trials=200, seed=s)
            ev = idsg.evaluate_composition(r["composition_wtpct"], mor, wa)   # same basis as the fronts
            bo_rows.append((ev["cost_Tk_per_kg"], ev["CO2_kg_per_kg"], bool(r["meets_target_at_k"])))
            bo_comps.append(r["composition_wtpct"])
        bo = np.array(bo_rows, float)
        ok = bo[:, 2] > 0.5                       # BO recipes that meet the spec at margin k
        bo_pts = [(c, o) for (c, o, f) in bo_rows if f]

        df_ind = idsg.inverse_pareto(mor, wa, n_trials=n_trials, seed=0)
        df_seed = idsg.inverse_pareto(mor, wa, n_trials=n_trials, seed=0, warm_start=bo_comps)
        df_ind.to_csv(DATADIR / f"pareto_{name}.csv", index=False)
        df_seed.to_csv(DATADIR / f"pareto_seeded_{name}.csv", index=False)
        f_ind, f_seed = _front_2d(df_ind), _front_2d(df_seed)

        comp = df_ind[df_ind.compliant]
        ax.scatter(comp.cost_Tk_per_kg, comp.CO2_kg_per_kg, s=10, c="#BDBDBD", label="compliant trials (independent)")
        ax.scatter(f_ind.cost_Tk_per_kg, f_ind.CO2_kg_per_kg, s=30, c="#1565C0", label="front, independent NSGA-II")
        if len(f_seed):
            ax.scatter(f_seed.cost_Tk_per_kg, f_seed.CO2_kg_per_kg, s=14, c="#EF6C00", marker="x",
                       label="front, BO-seeded NSGA-II")
        ax.scatter(bo[ok, 0], bo[ok, 1], marker="*", s=180, c="#D32F2F", edgecolors="k",
                   label=f"scalarised BO ({int(ok.sum())}/{n_bo_seeds} meet spec at k)")
        ax.set_title(f"{name}: MOR >= {mor:.1f} MPa, WA <= {wa} %  (k = {idsg.SAFETY_FACTOR:g})")
        ax.set_xlabel("Cost (Tk/kg)"); ax.set_ylabel("CO2 (kg/kg)")
        ax.grid(True, ls="--", alpha=.4); ax.legend(fontsize=7)

        below = int(sum(1 for c, _ in bo_pts if len(f_ind) and c < f_ind.cost_Tk_per_kg.min() - 1e-9))
        summ.append({
            "scenario": name, "n_trials_each_front": n_trials, "k": idsg.SAFETY_FACTOR,
            "n_domain_feasible_indep": len(df_ind), "n_compliant_indep": int(df_ind.compliant.sum()),
            "n_front_2d_indep": len(f_ind), "n_front_2d_seeded": len(f_seed),
            "front_cost_min": f_ind.cost_Tk_per_kg.min() if len(f_ind) else np.nan,
            "front_cost_max": f_ind.cost_Tk_per_kg.max() if len(f_ind) else np.nan,
            "front_CO2_min": f_ind.CO2_kg_per_kg.min() if len(f_ind) else np.nan,
            "front_CO2_max": f_ind.CO2_kg_per_kg.max() if len(f_ind) else np.nan,
            "n_bo_compliant": len(bo_pts),
            "bo_dominated_by_independent_front": f"{_n_dominated(f_ind, bo_pts, dom_tol)}/{len(bo_pts)}",
            "bo_dominated_by_seeded_front": f"{_n_dominated(f_seed, bo_pts, dom_tol)}/{len(bo_pts)}",
            "bo_cost_below_independent_front_min": f"{below}/{len(bo_pts)}",
            "bo_cost_mean": float(np.mean([c for c, _ in bo_pts])) if bo_pts else np.nan,
            "bo_CO2_mean": float(np.mean([o for _, o in bo_pts])) if bo_pts else np.nan})
    out = pd.DataFrame(summ)
    out.to_csv(DATADIR / "pareto_summary.csv", index=False)
    print(out.round(4).to_string(index=False))
    print("\nReading guide: 'bo_cost_below_independent_front_min' > 0 means the independent front did not "
          "reach BO's cost, i.e. it is under-sampled at that target; judge dominance with the seeded front.")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(PLOTDIR / f"pareto_front.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved: data/pareto_*.csv, data/pareto_seeded_*.csv, data/pareto_summary.csv, plots/pareto_front.pdf/.png")


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 4. CO2-FACTOR ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════════════════════════
@contextlib.contextmanager
def _co2_factors(factors: dict):
    """Temporarily replace the CO2 factors AND everything derived from them."""
    saved = (idsg.co2_dict, idsg.dataset["CO2_kg_per_kg"].copy(),
             idsg._co2_min, idsg._co2_max, idsg._co2_range)
    idsg.co2_dict = dict(factors)
    idsg.dataset["CO2_kg_per_kg"] = idsg._per_kg(idsg.dataset, idsg.co2_dict)
    idsg._co2_min = float(idsg.dataset["CO2_kg_per_kg"].min())
    idsg._co2_max = float(idsg.dataset["CO2_kg_per_kg"].max())
    idsg._co2_range = max(idsg._co2_max - idsg._co2_min, 1e-9)
    try:
        yield
    finally:
        (idsg.co2_dict, idsg.dataset["CO2_kg_per_kg"],
         idsg._co2_min, idsg._co2_max, idsg._co2_range) = saved


def _co2_of(comp: dict, f: dict) -> float:
    tot = sum(comp[m] for m in materials)
    return sum(comp[m] * f[m] for m in materials) / tot


def co2() -> None:
    n_trials = 60 if QUICK else 200
    best_of = 2 if QUICK else 3          # search seeds per (scenario, target); the best recipe is kept
    n_mc = 5 if QUICK else 30
    spread = 0.5                         # +/-50 %
    reseed_offset = 100
    targets = _targets(False, ("mid_MOR",))

    def best_recipe(mor, wa, f, seed0, seed_recipe=None) -> dict:
        """Best of best_of searches under factors f: lowest normalised cost + CO2 among recipes that
        meet the spec at margin k (all recipes if none do). `seed_recipe` is added as an extra
        candidate. Must be called inside _co2_factors(f)."""
        cands = []
        if seed_recipe is not None:
            mor_c, wa_c = idsg.clamp_targets(mor, wa)
            e_s = idsg._evaluate(seed_recipe, mor_c, wa_c, idsg.SAFETY_FACTOR)
            ok_s = bool(e_s["feasible"] and e_s["violation"] <= idsg.COMPLIANCE_TOL)
            cost_s = idsg._cost_co2(seed_recipe)[0]
            score_s = ((cost_s - idsg._cost_min) / idsg._cost_range
                       + (_co2_of(seed_recipe, f) - idsg._co2_min) / idsg._co2_range)
            cands.append((not ok_s, score_s, {"composition_wtpct": dict(seed_recipe), "meets_target_at_k": ok_s,
                                              "cost_Tk_per_kg": cost_s}))
        for s in range(best_of):
            r, _, _ = idsg.inverse_bayesian_optimization(mor, wa, n_trials=n_trials, seed=seed0 + s)
            score = ((r["cost_Tk_per_kg"] - idsg._cost_min) / idsg._cost_range
                     + (_co2_of(r["composition_wtpct"], f) - idsg._co2_min) / idsg._co2_range)
            cands.append((not r["meets_target_at_k"], score, r))
        cands.sort(key=lambda t: (t[0], t[1]))
        return cands[0][2]

    def build_scenarios(base: dict) -> dict:
        sc = {"reseed": dict(base)}
        for m in materials:
            for mult in (1 - spread, 1 + spread):
                sc[f"{m}_x{mult:g}"] = {**base, m: base[m] * mult}
        sc["crushed_tile_zero"] = {**base, "Crushing": 0.0}
        sc["waste_zero"] = {**base, "Crushing": 0.0, "ETP": 0.0}
        sc["AG98_april_0.48"] = {**base, "AG98": 0.48}
        rng = np.random.default_rng(2026)
        for i in range(n_mc):
            sc[f"MC_{i:02d}"] = {m: base[m] * float(rng.uniform(1 - spread, 1 + spread)) for m in materials}
        return sc

    idsg.refresh_prices()
    base = dict(idsg.co2_dict)
    scen = build_scenarios(base)
    print("Baseline CO2 factors:", {m: round(v, 4) for m, v in base.items()})
    _c, _e = idsg.dataset["cost_Tk_per_kg"], idsg.dataset["CO2_kg_per_kg"]
    print(f"Cost range across the {len(dataset)} batches: {_c.min():.3f}-{_c.max():.3f} Tk/kg "
          f"({(_c.max() / _c.min() - 1) * 100:.1f} % of the minimum); CO2 range: {_e.min():.4f}-{_e.max():.4f} "
          f"kg/kg ({(_e.max() / _e.min() - 1) * 100:.1f} %). Regret 'points' are relative to these ranges.")

    # ---- Q1: rank stability across the real batches ----
    co2_base = idsg._per_kg(dataset, base)
    lowest = set(co2_base.nsmallest(10).index)
    q1 = []
    for name, f in scen.items():
        v = idsg._per_kg(dataset, f)
        q1.append({"scenario": name, "spearman_rho_vs_baseline": float(spearmanr(co2_base, v)[0]),
                   "top10_lowest_CO2_overlap": len(lowest & set(v.nsmallest(10).index))})
    q1 = pd.DataFrame(q1)
    q1.to_csv(DATADIR / "co2_rank_stability.csv", index=False)

    # ---- Q2: recommendation stability and regret ----
    base_recipes = {}
    with _co2_factors(base):
        for tname, (mor, wa) in targets.items():
            base_recipes[tname] = best_recipe(mor, wa, base, 0)["composition_wtpct"]

    rows = []
    for i, (name, f) in enumerate(scen.items(), 1):
        print(f"[{i}/{len(scen)}] {name}")
        with _co2_factors(f):
            for tname, (mor, wa) in targets.items():
                seed0 = reseed_offset if name == "reseed" else 0
                b = base_recipes[tname]
                r = best_recipe(mor, wa, f, seed0, seed_recipe=b)
                alt = r["composition_wtpct"]
                c_alt, c_base = _co2_of(alt, f), _co2_of(b, f)
                k_alt, k_base = idsg._cost_co2(alt)[0], idsg._cost_co2(b)[0]
                sc_alt = (k_alt - idsg._cost_min) / idsg._cost_range + (c_alt - idsg._co2_min) / idsg._co2_range
                sc_base = (k_base - idsg._cost_min) / idsg._cost_range + (c_base - idsg._co2_min) / idsg._co2_range
                rows.append({
                    "scenario": name, "target_name": tname,
                    "CO2_alt_recipe_under_alt": c_alt, "CO2_baseline_recipe_under_alt": c_base,
                    "objective_regret_points": (sc_base - sc_alt) * 100.0,
                    "cost_diff_pct": (k_base / k_alt - 1.0) * 100.0,
                    "co2_diff_pct": (c_base / c_alt - 1.0) * 100.0,
                    "wtpct_moved": 0.5 * sum(abs(alt[m] - b[m]) for m in materials),
                    "max_single_change_wtpct": max(abs(alt[m] - b[m]) for m in materials),
                    "alt_meets_spec_at_k": bool(r["meets_target_at_k"])})
    runs = pd.DataFrame(rows)
    runs.to_csv(DATADIR / "co2_robustness_runs.csv", index=False)
    summ = runs.groupby("scenario").agg(
        regret_mean_pts=("objective_regret_points", "mean"), regret_max_pts=("objective_regret_points", "max"),
        cost_diff_mean_pct=("cost_diff_pct", "mean"), co2_diff_mean_pct=("co2_diff_pct", "mean"),
        wtpct_moved_mean=("wtpct_moved", "mean"), max_single_change_mean=("max_single_change_wtpct", "mean"),
        alt_meets_spec_share=("alt_meets_spec_at_k", "mean")).reset_index()
    summ = summ.merge(q1, on="scenario")
    summ.to_csv(DATADIR / "co2_robustness_summary.csv", index=False)

    noise = summ[summ.scenario == "reseed"].iloc[0]
    print("\n=== One-at-a-time and named scenarios ===")
    named = summ[~summ.scenario.str.startswith("MC_")].sort_values("regret_mean_pts", ascending=False)
    print(named.round(3).to_string(index=False))
    mc = summ[summ.scenario.str.startswith("MC_")]
    if len(mc):
        print(f"\n=== Monte-Carlo (each factor x U({1-spread:g}, {1+spread:g}), n = {len(mc)}) ===")
        print(f"Spearman rho of CO2 ranking across the 238 batches: min {mc.spearman_rho_vs_baseline.min():.3f}, "
              f"median {mc.spearman_rho_vs_baseline.median():.3f}")
        print(f"Objective regret of using the baseline recipe (points of normalised range): median "
              f"{mc.regret_mean_pts.median():.2f}, 95th pct {mc.regret_mean_pts.quantile(.95):.2f}, max {mc.regret_mean_pts.max():.2f}")
        print(f"  ...of which baseline recipe is dearer by {mc.cost_diff_mean_pct.median():+.2f} % (median) and "
              f"dirtier by {mc.co2_diff_mean_pct.median():+.2f} % (median) than the alternative-optimal recipe")
        print(f"Recipe shift (wt% moved): median {mc.wtpct_moved_mean.median():.2f}")
    print(f"\nNoise floor ('reseed'): objective regret {noise.regret_mean_pts:.2f} points, wt% moved "
          f"{noise.wtpct_moved_mean:.2f}. Regret is >= 0 by construction; values near the 'reseed' level "
          "are search differences, not a factor effect.")
    print(f"Share of runs where the alternative-optimal recipe differs from the baseline recipe "
          f"(wt% moved > 0.01): {(runs.wtpct_moved > 0.01).mean():.0%}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    nm = named[named.scenario != "reseed"].sort_values("spearman_rho_vs_baseline")
    ax[0].barh(nm.scenario, nm.spearman_rho_vs_baseline, color="#1565C0")
    if len(mc):
        ax[0].axvspan(mc.spearman_rho_vs_baseline.min(), mc.spearman_rho_vs_baseline.max(), color="#EF6C00",
                      alpha=.25, label="Monte-Carlo range")
        ax[0].legend(fontsize=8)
    ax[0].set_xlabel("Spearman rho of CO2 ranking vs baseline (238 batches)")
    ax[0].set_xlim(max(0, min(nm.spearman_rho_vs_baseline.min(), 0.5) - .05), 1.0); ax[0].grid(True, ls="--", alpha=.4)
    nr = named.sort_values("regret_mean_pts")
    yy = np.arange(len(nr))
    ax[1].barh(yy - 0.2, nr.cost_diff_mean_pct, height=0.4, color="#EF6C00", label="baseline recipe dearer by (%)")
    ax[1].barh(yy + 0.2, nr.co2_diff_mean_pct, height=0.4, color="#D32F2F", label="baseline recipe dirtier by (%)")
    ax[1].set_yticks(yy); ax[1].set_yticklabels(nr.scenario)
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_xlabel("Penalty of using the baseline recipe under the alternative factors (%), sorted by objective regret")
    ax[1].legend(fontsize=8); ax[1].grid(True, ls="--", alpha=.4)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PLOTDIR / f"co2_robustness.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    idsg.refresh_prices()          # restore baseline state
    print("Saved: data/co2_rank_stability.csv, co2_robustness_runs.csv, co2_robustness_summary.csv, plots/co2_robustness.*")


# ══════════════════════════════════════════════════════════════════════════════════════════════
ANALYSES = {"sensitivity": sensitivity, "samplers": samplers, "pareto": pareto, "co2": co2}


def main(argv: list[str]) -> None:
    wanted = [a.lower() for a in argv] or ["all"]
    if "all" in wanted:
        wanted = list(ANALYSES)
    unknown = [a for a in wanted if a not in ANALYSES]
    if unknown:
        sys.exit(f"Unknown analysis {unknown}. Choose from: {', '.join(ANALYSES)} or 'all'.")
    for name in wanted:
        print("\n" + "#" * 100 + f"\n# {name.upper()}\n" + "#" * 100)
        ANALYSES[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
