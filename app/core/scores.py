"""Verification scores: Brier, BSS, reliability, Murphy decomposition, ROC, bootstraps."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def brier(p: np.ndarray, o: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(o, float)) ** 2))


def bss(P: np.ndarray, cat: np.ndarray, ref) -> np.ndarray:
    """BSS per category. ref is (3,) or per-forecast (n, 3) reference probabilities."""
    ref = np.broadcast_to(np.asarray(ref, float), P.shape)
    out = np.empty(3)
    for j in range(3):
        o = (cat == j).astype(float)
        out[j] = 1 - brier(P[:, j], o) / brier(ref[:, j], o)
    return out


def brier_table(P: np.ndarray, cat: np.ndarray, ref) -> list[dict]:
    """Notebook cell 16 table rows."""
    ref = np.broadcast_to(np.asarray(ref, float), P.shape)
    rows = []
    for j in range(3):
        o = (cat == j).astype(float)
        bs, br = brier(P[:, j], o), brier(ref[:, j], o)
        rows.append({"bs": bs, "bs_ref": br, "bss": 1 - bs / br})
    return rows


def bin_index(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-width bins on [0, 1]; last bin includes 1 (as in notebook cell 17)."""
    return np.clip(np.floor(np.asarray(p) * n_bins).astype(int), 0, n_bins - 1)


@dataclass(frozen=True)
class Reliability:
    edges: np.ndarray
    mean_p: np.ndarray    # per non-empty bin
    freq: np.ndarray
    count: np.ndarray


def reliability(p: np.ndarray, o: np.ndarray, n_bins: int = 8) -> Reliability:
    """All non-empty bins (the chart greys out sparse ones; the notebook dropped n < 10)."""
    k = bin_index(p, n_bins)
    keep = [b for b in range(n_bins) if (k == b).any()]
    return Reliability(
        edges=np.linspace(0, 1, n_bins + 1),
        mean_p=np.array([p[k == b].mean() for b in keep]),
        freq=np.array([o[k == b].mean() for b in keep]),
        count=np.array([(k == b).sum() for b in keep]),
    )


@dataclass(frozen=True)
class Murphy:
    bs_raw: float
    bs_binned: float
    reliability: float
    resolution: float
    uncertainty: float

    @property
    def within_bin(self) -> float:
        return self.bs_raw - self.bs_binned


def murphy(p: np.ndarray, o: np.ndarray, n_bins: int = 8) -> Murphy:
    """Exact decomposition of the *binned* Brier score: BS_binned = REL - RES + UNC."""
    p, o = np.asarray(p, float), np.asarray(o, float)
    k = bin_index(p, n_bins)
    obar = o.mean()
    rel = res = 0.0
    p_binned = np.empty_like(p)
    for b in np.unique(k):
        m = k == b
        pk, ok = p[m].mean(), o[m].mean()
        p_binned[m] = pk
        rel += m.sum() * (pk - ok) ** 2
        res += m.sum() * (ok - obar) ** 2
    n = len(p)
    return Murphy(bs_raw=brier(p, o), bs_binned=brier(p_binned, o),
                  reliability=rel / n, resolution=res / n, uncertainty=obar * (1 - obar))


@dataclass(frozen=True)
class Roc:
    fpr: np.ndarray
    tpr: np.ndarray
    cutoffs: np.ndarray
    auc: float


def roc(p: np.ndarray, o: np.ndarray) -> Roc | None:
    o = np.asarray(o).astype(int)
    if np.unique(o).size < 2:
        return None
    fpr, tpr, cut = roc_curve(o, p)
    return Roc(fpr=fpr, tpr=tpr, cutoffs=cut, auc=float(roc_auc_score(o, p)))


def confusion(p: np.ndarray, o: np.ndarray, cutoff: float) -> dict[str, int]:
    """Warn when p >= cutoff."""
    warn, o = np.asarray(p) >= cutoff, np.asarray(o).astype(bool)
    return {"hits": int((warn & o).sum()), "misses": int((~warn & o).sum()),
            "false_alarms": int((warn & ~o).sum()), "correct_negatives": int((~warn & ~o).sum())}


def year_bootstrap_bss(P: np.ndarray, cat: np.ndarray, ref, years: np.ndarray,
                       n_boot: int = 500, seed: int = 0) -> np.ndarray:
    """Held-out-season sensitivity: resample test *years* with the fitted model held fixed."""
    rng = np.random.default_rng(seed)
    ref = np.broadcast_to(np.asarray(ref, float), P.shape)
    uy = np.unique(years)
    idx_by_year = {y: np.flatnonzero(years == y) for y in uy}
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_year[y] for y in rng.choice(uy, size=len(uy), replace=True)])
        out.append(bss(P[idx], cat[idx], ref[idx]))
    return np.array(out)
