"""Step-by-step pool-adjacent-violators, for DISPLAY ONLY (Lab 2B).

Fitting always uses idr_xarray.IDRModel; a test checks that the final step here
equals IDRModel's fitted P(Y <= t | x) on the same points.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Block:
    start: int      # first group index (inclusive)
    stop: int       # last group index (exclusive)
    value: float    # weighted mean of the indicator over the block
    weight: float


@dataclass(frozen=True)
class PavaStep:
    blocks: tuple[Block, ...]      # blocks covering every group (unprocessed groups are singletons)
    merged: tuple[int, int] | None  # (start, stop) of the block just formed, if any
    note: str


def group_by_x(x: np.ndarray, indicator: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct x (ascending), mean indicator per x, and count per x."""
    ux, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    means = np.bincount(inv, weights=indicator.astype(float)) / cnt
    return ux, means, cnt.astype(float)


def pava_steps_antitonic(values: np.ndarray, weights: np.ndarray) -> list[PavaStep]:
    """Non-increasing weighted isotonic regression, recording each merge."""
    n = len(values)
    singletons = [Block(i, i + 1, float(values[i]), float(weights[i])) for i in range(n)]
    steps = [PavaStep(tuple(singletons), None, "Start: the raw share of dry outcomes at each SEAS5 value.")]
    stack: list[Block] = []
    for i in range(n):
        stack.append(singletons[i])
        while len(stack) >= 2 and stack[-2].value < stack[-1].value - 1e-12:
            b2, b1 = stack.pop(), stack.pop()
            w = b1.weight + b2.weight
            merged = Block(b1.start, b2.stop, (b1.value * b1.weight + b2.value * b2.weight) / w, w)
            stack.append(merged)
            note = (f"Violation: a wetter forecast had a higher share of dry outcomes "
                    f"({b1.value:.2f} < {b2.value:.2f}). Pool them → {merged.value:.2f}.")
            steps.append(PavaStep(tuple(stack) + tuple(singletons[i + 1:]), (merged.start, merged.stop), note))
    steps.append(PavaStep(tuple(stack), None, "Done: the fitted values never increase as SEAS5 gets wetter."))
    return steps


def expand(step: PavaStep, n_groups: int) -> np.ndarray:
    """Fitted value per group for a given step."""
    out = np.empty(n_groups)
    for b in step.blocks:
        out[b.start:b.stop] = b.value
    return out
