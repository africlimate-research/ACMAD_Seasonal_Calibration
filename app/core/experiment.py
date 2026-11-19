"""Train/test sampling, IDR fitting and tercile probabilities.

Default settings reproduce notebook cells 9-16 exactly. IDR always comes from
``idr_xarray.IDRModel`` through its public API, with ``increasing=True``
(``increasing=False`` has a known predict() bug — see plan §2e).
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from .data import REPO_ROOT, Dataset

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from idr_xarray import IDRDistribution, IDRModel  # noqa: E402

CATEGORIES = ("BN", "NN", "AN")
CATEGORY_NAMES = ("Below normal", "Near normal", "Above normal")


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    n_train: int = 1200
    n_test: int = 1500
    split_year: int = 2015      # train = years <= split_year, test = the rest

    @property
    def is_notebook_default(self) -> bool:
        return self == Settings()


@dataclass(frozen=True)
class Sample:
    """Sampled point-year pairs, keeping each pair's location and year."""

    x_train: np.ndarray
    y_train: np.ndarray
    loc_train: np.ndarray
    year_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    loc_test: np.ndarray
    year_test: np.ndarray
    train_rows: np.ndarray      # bool mask over ds.years
    test_rows: np.ndarray

    @property
    def n_test_years(self) -> int:
        return len(np.unique(self.year_test))


def draw_sample(ds: Dataset, s: Settings = Settings()) -> Sample:
    """Notebook cell 9: one generator, train draw then test draw, on year-major flattened arrays."""
    train_rows = ds.years <= s.split_year
    test_rows = ~train_rows
    L = ds.n_locations
    Xtr, Ytr = ds.X[train_rows], ds.Y[train_rows]
    Xte, Yte = ds.X[test_rows], ds.Y[test_rows]
    rng = np.random.default_rng(s.seed)
    it = rng.choice(Xtr.size, size=min(s.n_train, Xtr.size), replace=False)
    ie = rng.choice(Xte.size, size=min(s.n_test, Xte.size), replace=False)
    return Sample(
        x_train=Xtr.ravel()[it], y_train=Ytr.ravel()[it],
        loc_train=it % L, year_train=ds.years[train_rows][it // L],
        x_test=Xte.ravel()[ie], y_test=Yte.ravel()[ie],
        loc_test=ie % L, year_test=ds.years[test_rows][ie // L],
        train_rows=train_rows, test_rows=test_rows,
    )


def fit_idr(x: np.ndarray, y: np.ndarray) -> IDRModel:
    return IDRModel(increasing=True).fit(x, y)


# ---------------------------------------------------------------------------
# Evaluating predictive CDFs
# ---------------------------------------------------------------------------

def cdf_at_rows(dist: IDRDistribution, t) -> np.ndarray:
    """F_i(t_i): each forecast's CDF at its own threshold (t may be a scalar)."""
    n = dist.cdf.shape[0]
    t = np.broadcast_to(np.asarray(t, dtype=float), (n,))
    thr = np.concatenate([[-np.inf], dist.thresholds])
    cdf = np.hstack([np.zeros((n, 1)), dist.cdf])
    return cdf[np.arange(n), np.searchsorted(thr, t, side="right") - 1]


def tercile_probs(model: IDRModel, x: np.ndarray, q1, q2) -> np.ndarray:
    """(n, 3) array of P(BN), P(NN), P(AN). q1/q2 may be scalars or per-forecast arrays."""
    d = model.predict(np.atleast_1d(x))
    a, b = cdf_at_rows(d, q1), cdf_at_rows(d, q2)
    return np.column_stack([a, b - a, 1 - b])


def categories(y: np.ndarray, q1, q2) -> np.ndarray:
    """0 = BN (y <= q1), 1 = NN, 2 = AN (y > q2). Same as np.digitize(y, [q1, q2], right=True)."""
    return (np.asarray(y) > q1).astype(int) + (np.asarray(y) > q2).astype(int)


def category_frequencies(cat: np.ndarray) -> np.ndarray:
    return np.array([(cat == j).mean() for j in range(3)])


def threshold_curve(model: IDRModel, x_train: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Lab 2A: fitted P(Y <= t | x) at each distinct training SEAS5 value (public API only)."""
    xs = np.unique(x_train)
    return xs, cdf_at_rows(model.predict(xs), t)


def predictive_cdf(model: IDRModel, x: float) -> tuple[np.ndarray, np.ndarray]:
    d = model.predict(np.array([x]))
    return d.thresholds, d.cdf[0]


def predictive_quantiles(model: IDRModel, x: float, qs=(0.1, 0.5, 0.9)) -> np.ndarray:
    return np.atleast_1d(model.predict(np.array([x])).quantile(np.asarray(qs)))


@dataclass(frozen=True)
class LinearGaussian:
    """Comparison model: y = a + b x + N(0, sigma^2)."""

    a: float
    b: float
    sigma: float

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray) -> "LinearGaussian":
        b, a = np.polyfit(x, y, 1)
        return cls(a=float(a), b=float(b), sigma=float(np.std(y - (a + b * x), ddof=2)))

    def cdf(self, x: float, t: np.ndarray) -> np.ndarray:
        return ndtr((np.asarray(t) - (self.a + self.b * x)) / self.sigma)


# ---------------------------------------------------------------------------
# The notebook experiment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PooledResult:
    sample: Sample
    model: IDRModel
    q1: float
    q2: float
    P: np.ndarray          # (n_test, 3) forecast probabilities
    obs_cat: np.ndarray    # (n_test,) observed category 0/1/2
    clim: np.ndarray       # (3,) training-years climatological frequencies
    y_train_pool: np.ndarray  # all training-year CHIRPS values (for cutoffs)


def run_pooled(ds: Dataset, s: Settings = Settings()) -> PooledResult:
    """Notebook cells 9, 13 and 16: pooled IDR, pooled terciles, pooled climatology."""
    smp = draw_sample(ds, s)
    model = fit_idr(smp.x_train, smp.y_train)
    y_pool = ds.Y[smp.train_rows].ravel()
    q1, q2 = (float(v) for v in np.quantile(y_pool, [1 / 3, 2 / 3]))
    P = tercile_probs(model, smp.x_test, q1, q2)
    return PooledResult(
        sample=smp, model=model, q1=q1, q2=q2, P=P,
        obs_cat=categories(smp.y_test, q1, q2),
        clim=category_frequencies(categories(y_pool, q1, q2)),
        y_train_pool=y_pool,
    )


def shuffle_experiment(ds: Dataset, s: Settings = Settings(), n_shuffles: int = 100) -> dict[str, np.ndarray]:
    """Trap C / plan §2c: BSS (n_shuffles, 3) for each control; test cases fixed."""
    from .scores import bss

    r = run_pooled(ds, s)
    smp = r.sample
    right, wrong, both = [], [], []
    for k in range(n_shuffles):
        xs = smp.x_train.copy()
        np.random.default_rng(k).shuffle(xs)
        m = fit_idr(xs, smp.y_train)          # reused for the shuffled-test control below
        right.append(bss(tercile_probs(m, smp.x_test, r.q1, r.q2), r.obs_cat, r.clim))
        # Wrong monotone direction via sign flip (NOT increasing=False, which is buggy).
        wrong.append(bss(tercile_probs(fit_idr(-xs, smp.y_train), -smp.x_test, r.q1, r.q2), r.obs_cat, r.clim))
        xe = smp.x_test.copy()
        np.random.default_rng(1000 + k).shuffle(xe)
        both.append(bss(tercile_probs(m, xe, r.q1, r.q2), r.obs_cat, r.clim))
    fit_clim = category_frequencies(categories(smp.y_train, r.q1, r.q2))
    const = bss(np.tile(fit_clim, (len(smp.y_test), 1)), r.obs_cat, r.clim)
    return {
        "real": bss(r.P, r.obs_cat, r.clim)[None, :],
        "shuffled": np.array(right),
        "wrong_direction": np.array(wrong),
        "shuffled_train_and_test": np.array(both),
        "constant_climatology": const[None, :],
    }
