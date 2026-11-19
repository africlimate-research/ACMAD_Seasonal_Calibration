"""Lab 1, Lab 4D and Lab 5 helpers: bias by amount, sampling sensitivity, leakage.

No Streamlit imports. IDR always via ``fit_idr`` (``increasing=True``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Dataset
from .experiment import (Settings, categories, category_frequencies, fit_idr, run_pooled,
                         tercile_probs)
from .location import local_terciles
from .scores import bss


# --- Lab 1B -------------------------------------------------------------------

@dataclass(frozen=True)
class BiasByAmount:
    edges: np.ndarray       # (n_bins + 1,) SEAS5 bin edges (mm)
    mean_bias: np.ndarray   # (n_bins,) mean SEAS5 - CHIRPS (mm)
    q25: np.ndarray
    q75: np.ndarray
    mean_x: np.ndarray
    mean_y: np.ndarray
    overall_mean: float


def bias_by_amount(ds: Dataset, n_bins: int = 10) -> BiasByAmount:
    """Bin every point-season by SEAS5 quantile and summarise SEAS5 - CHIRPS in each bin."""
    x, y = ds.X.ravel(), ds.Y.ravel()
    d = x - y
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    k = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
    stat = lambda f: np.array([f(b) for b in range(n_bins)])  # noqa: E731
    return BiasByAmount(
        edges=edges,
        mean_bias=stat(lambda b: d[k == b].mean()),
        q25=stat(lambda b: np.percentile(d[k == b], 25)),
        q75=stat(lambda b: np.percentile(d[k == b], 75)),
        mean_x=stat(lambda b: x[k == b].mean()),
        mean_y=stat(lambda b: y[k == b].mean()),
        overall_mean=float(d.mean()),
    )


# --- Lab 4D / trap E: how stable are the scores? -------------------------------

def seed_sensitivity(ds: Dataset, s: Settings = Settings(), n_seeds: int = 50) -> np.ndarray:
    """Pair-sampling sensitivity: BSS (n_seeds, 3) when a different random set of pairs is drawn."""
    out = []
    for k in range(n_seeds):
        r = run_pooled(ds, Settings(s.seed + 1 + k, s.n_train, s.n_test, s.split_year))
        out.append(bss(r.P, r.obs_cat, r.clim))
    return np.array(out)


def pair_bootstrap_bss(P: np.ndarray, cat: np.ndarray, ref, n_boot: int = 500, seed: int = 0) -> np.ndarray:
    """Resample test *pairs* as if they were independent (the naive view, for trap E)."""
    rng = np.random.default_rng(seed)
    ref = np.broadcast_to(np.asarray(ref, float), P.shape)
    n = len(cat)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        out.append(bss(P[i], cat[i], ref[i]))
    return np.array(out)


def bss_by_year(P: np.ndarray, cat: np.ndarray, ref, years: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BSS (n_years, 3) for each held-out season separately, with pair counts."""
    ref = np.broadcast_to(np.asarray(ref, float), P.shape)
    uy = np.unique(years)
    vals = np.array([bss(P[years == y], cat[years == y], ref[years == y]) for y in uy])
    return uy, vals, np.array([(years == y).sum() for y in uy])


# --- Trap B: leakage ----------------------------------------------------------------

@dataclass(frozen=True)
class Leakage:
    bss: np.ndarray        # (3,)
    bs_ref: np.ndarray     # (3,) Brier score of the reference
    n_test: int
    local: bool
    P: np.ndarray          # (n_test, 3) forecasts that were scored
    cat: np.ndarray        # (n_test,) observed categories
    ref: np.ndarray        # (n_test, 3) reference probabilities
    year: np.ndarray       # (n_test,) season of each scored pair


def leakage_experiment(ds: Dataset, s: Settings = Settings(), *, local: bool = False,
                       leak_cutoffs: bool = False, leak_reference: bool = False,
                       leak_model: bool = False) -> Leakage:
    """Score the same test pairs while letting test-year information leak into one or more parts.

    local=False: the notebook's pooled IDR, pooled cutoffs and pooled climatology.
    local=True:  anomaly IDR with each location's own cutoffs and reference (trap A row ③).
    leak_cutoffs:   tercile cutoffs from all 25 years (changes *which event* is scored).
    leak_reference: climatology from all 25 years (changes *the yardstick*).
    leak_model:     the test pairs are added to the IDR training pairs.
    """
    r = run_pooled(ds, s)
    smp = r.sample
    every = np.ones(ds.n_years, bool)
    cut_rows = every if leak_cutoffs else smp.train_rows
    ref_rows = every if leak_reference else smp.train_rows
    xf, yf, lf = smp.x_train, smp.y_train, smp.loc_train
    if leak_model:
        xf, yf, lf = (np.r_[xf, smp.x_test], np.r_[yf, smp.y_test], np.r_[lf, smp.loc_test])

    if not local:
        q1, q2 = np.quantile(ds.Y[cut_rows].ravel(), [1 / 3, 2 / 3])
        model = fit_idr(xf, yf) if leak_model else r.model
        P = tercile_probs(model, smp.x_test, q1, q2)
        cat = categories(smp.y_test, q1, q2)
        ref = np.broadcast_to(category_frequencies(categories(ds.Y[ref_rows].ravel(), q1, q2)), P.shape)
        return Leakage(bss(P, cat, ref), _bs_ref(ref, cat), len(cat), False, P, cat, np.asarray(ref), smp.year_test)

    # Local: same eligible locations in every variant, standardised with training-year statistics.
    Xtr, Ytr = ds.X[smp.train_rows], ds.Y[smp.train_rows]
    mx, sx, my, sy = Xtr.mean(0), Xtr.std(0), Ytr.mean(0), Ytr.std(0)
    good = (sy > 1) & (sx > 0) & local_terciles(ds, smp.train_rows).distinct & local_terciles(ds, every).distinct
    Q1, Q2 = np.quantile(ds.Y[cut_rows], [1 / 3, 2 / 3], axis=0)
    rc = categories(ds.Y[ref_rows], Q1, Q2)
    REF = np.stack([(rc == j).mean(0) for j in range(3)], axis=1)

    g = good[lf]
    model = fit_idr((xf[g] - mx[lf[g]]) / sx[lf[g]], (yf[g] - my[lf[g]]) / sy[lf[g]])
    ok = good[smp.loc_test]
    le = smp.loc_test[ok]
    P = tercile_probs(model, (smp.x_test[ok] - mx[le]) / sx[le], (Q1[le] - my[le]) / sy[le], (Q2[le] - my[le]) / sy[le])
    cat = categories(smp.y_test[ok], Q1[le], Q2[le])
    return Leakage(bss(P, cat, REF[le]), _bs_ref(REF[le], cat), int(ok.sum()), True, P, cat, REF[le], smp.year_test[ok])


def _bs_ref(ref: np.ndarray, cat: np.ndarray) -> np.ndarray:
    return np.array([np.mean((ref[:, j] - (cat == j)) ** 2) for j in range(3)])
