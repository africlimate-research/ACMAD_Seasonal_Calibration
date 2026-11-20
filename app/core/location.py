"""Per-location statistics, local terciles and the pooling comparison (trap A, plan §2b)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Dataset
from .experiment import (Settings, categories, fit_idr, run_pooled, tercile_probs)
from .scores import bss


def per_location_r(ds: Dataset, rows: np.ndarray | None = None) -> np.ndarray:
    """Year-to-year Pearson r between SEAS5 and CHIRPS at each location (NaN where undefined)."""
    X = ds.X if rows is None else ds.X[rows]
    Y = ds.Y if rows is None else ds.Y[rows]
    Xa, Ya = X - X.mean(0), Y - Y.mean(0)
    den = np.sqrt((Xa ** 2).sum(0) * (Ya ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (Xa * Ya).sum(0) / den, np.nan)


def pooled_r(ds: Dataset, anomalies: bool = False) -> float:
    X, Y = ds.X, ds.Y
    if anomalies:
        X, Y = X - X.mean(0), Y - Y.mean(0)
    return float(np.corrcoef(X.ravel(), Y.ravel())[0, 1])


def location_by_wetness(ds: Dataset, percentile: float) -> int:
    """Index of the location whose mean OND rainfall is closest to the given percentile."""
    m = ds.Y.mean(0)
    target = np.percentile(m, percentile)
    return int(np.argmin(np.abs(m - target)))


@dataclass(frozen=True)
class LocalTerciles:
    q1: np.ndarray          # (L,) training-year 1/3 quantile per location
    q2: np.ndarray          # (L,)
    reference: np.ndarray   # (L, 3) empirical training frequencies under these cutoffs

    @property
    def distinct(self) -> np.ndarray:
        return self.q1 < self.q2


def local_terciles(ds: Dataset, train_rows: np.ndarray) -> LocalTerciles:
    Ytr = ds.Y[train_rows]
    q1, q2 = np.quantile(Ytr, [1 / 3, 2 / 3], axis=0)
    cat = categories(Ytr, q1, q2)
    ref = np.stack([(cat == j).mean(0) for j in range(3)], axis=1)
    return LocalTerciles(q1=q1, q2=q2, reference=ref)


@dataclass(frozen=True)
class PoolingComparison:
    pooled: np.ndarray          # ① BSS (3,)
    pooled_local: np.ndarray    # ② BSS (3,)
    anomaly_local: np.ndarray   # ③ BSS (3,)
    n_test_pairs: int           # pairs scored in every row (if common) or in ②
    n_test_excluded: int
    n_locations_equal_cutoffs: int
    n_train_pairs_anomaly: int
    common: bool
    pooled_all: np.ndarray          # ① BSS on all test pairs (the notebook's number)
    pooled_excluded: np.ndarray     # ① BSS on the left-out pairs only
    excluded_median_mm: float       # median mean-OND rainfall at the left-out pairs' locations


def pooling_comparison(ds: Dataset, s: Settings = Settings(), common: bool = True) -> PoolingComparison:
    """Trap A. With common=True all three rows use the same eligible test pairs (the app default).
    With common=False, ② uses only the equal-cutoff exclusion (matches the review's -0.212/-0.044/-0.165)."""
    r = run_pooled(ds, s)
    smp = r.sample
    lt = local_terciles(ds, smp.train_rows)
    Xtr, Ytr = ds.X[smp.train_rows], ds.Y[smp.train_rows]
    mx, sx, my, sy = Xtr.mean(0), Xtr.std(0), Ytr.mean(0), Ytr.std(0)
    good = (sy > 1) & (sx > 0) & lt.distinct          # eligible for anomaly IDR

    # ③ anomaly IDR: fit on standardised training pairs at eligible locations.
    gt = good[smp.loc_train]
    l_tr = smp.loc_train[gt]
    model_a = fit_idr((smp.x_train[gt] - mx[l_tr]) / sx[l_tr], (smp.y_train[gt] - my[l_tr]) / sy[l_tr])

    ok_anom = good[smp.loc_test]
    ok_local = lt.distinct[smp.loc_test]
    ok2 = ok_anom if common else ok_local

    def score_pooled_local(mask):
        le = smp.loc_test[mask]
        P = tercile_probs(r.model, smp.x_test[mask], lt.q1[le], lt.q2[le])
        return bss(P, categories(smp.y_test[mask], lt.q1[le], lt.q2[le]), lt.reference[le])

    le = smp.loc_test[ok_anom]
    P3 = tercile_probs(model_a, (smp.x_test[ok_anom] - mx[le]) / sx[le],
                       (lt.q1[le] - my[le]) / sy[le], (lt.q2[le] - my[le]) / sy[le])
    anomaly = bss(P3, categories(smp.y_test[ok_anom], lt.q1[le], lt.q2[le]), lt.reference[le])

    m1 = ok_anom if common else np.ones_like(ok_anom)
    pooled = bss(r.P[m1], r.obs_cat[m1], r.clim)

    return PoolingComparison(
        pooled=pooled, pooled_local=score_pooled_local(ok2), anomaly_local=anomaly,
        n_test_pairs=int(ok2.sum()), n_test_excluded=int((~ok2).sum()),
        n_locations_equal_cutoffs=int((~lt.distinct).sum()),
        n_train_pairs_anomaly=int(gt.sum()), common=common,
        pooled_all=bss(r.P, r.obs_cat, r.clim),
        pooled_excluded=bss(r.P[~ok2], r.obs_cat[~ok2], r.clim) if (~ok2).sum() > 20 else np.full(3, np.nan),
        excluded_median_mm=float(np.median(ds.Y.mean(0)[smp.loc_test[~ok2]])) if (~ok2).any() else float("nan"),
    )
