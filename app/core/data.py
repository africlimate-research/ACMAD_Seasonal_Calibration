"""Load and validate the practical's data files. No Streamlit imports here."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


@dataclass(frozen=True)
class Dataset:
    """Matched OND seasons (rows) x locations (columns)."""

    X: np.ndarray          # SEAS5 OND predictor, mm, shape (years, locations)
    Y: np.ndarray          # CHIRPS observed OND total, mm, same shape
    years: np.ndarray      # (years,) 2001..2025
    wmo_mean: np.ndarray   # (locations,) WMO-referenced local mean
    wmo_std: np.ndarray    # (locations,) WMO-referenced local std

    @property
    def n_years(self) -> int:
        return self.X.shape[0]

    @property
    def n_locations(self) -> int:
        return self.X.shape[1]


def load_data(data_dir: Path | str = DATA_DIR) -> Dataset:
    d = Path(data_dir)
    X = np.load(d / "seas5_OND.npy")
    Y = np.load(d / "chirps_obs_OND.npy")
    years = np.load(d / "pred_years_OND.npy")
    wmo_mean = np.load(d / "y_mean_wmo_OND.npy")
    wmo_std = np.load(d / "y_std_wmo_OND.npy")
    # Same checks as notebook cell 5.
    if not (X.shape == Y.shape == (len(years), len(wmo_mean))):
        raise ValueError(f"Shape mismatch: X{X.shape} Y{Y.shape} years{years.shape}")
    if not np.array_equal(years, np.arange(2001, 2026)):
        raise ValueError("Expected years 2001–2025")
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        raise ValueError("Non-finite values in X or Y")
    return Dataset(X=X, Y=Y, years=years, wmo_mean=wmo_mean, wmo_std=wmo_std)
