"""Bin assignment for calibration estimators, with the edge cases named.

Two strategies are offered, and the choice changes every number downstream:

``equal_width``
    ``n_bins`` bins of width ``1 / n_bins`` on [0, 1], the convention of
    Naeini et al. (2015) and of the usual reliability diagram. Bins are
    ``[e_k, e_{k+1})`` with the last bin closed on the right so that a
    forecast of exactly 1.0 lands in the top bin. Bins can be empty, and on a
    forecast concentrated near the base rate most of them are.

``equal_mass``
    Bin edges at the sample quantiles of the forecast, so each bin holds as
    close to ``n / n_bins`` samples as ties allow. Edges depend on the sample,
    so two samples from the same forecaster are binned differently; this is
    the price of the variance reduction. Repeated forecast values cannot be
    split, so with heavy ties the occupied-bin count falls below ``n_bins``.

References
----------
Naeini, M. P., Cooper, G. F. and Hauskrecht, M. (2015). "Obtaining well
calibrated probabilities using Bayesian binning." *Proceedings of the 29th
AAAI Conference on Artificial Intelligence*, 2901-2907.
PMID 25927013; PMCID PMC4410090.

Nixon, J., Dusenberry, M. W., Zhang, L., Jerfel, G. and Tran, D. (2019).
"Measuring calibration in deep learning." *CVPR 2019 Workshops*.
arXiv:1904.01685. (adaptive, i.e. equal-mass, binning)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["BINNING_STRATEGIES", "assign_bins", "bin_edges"]

#: The binning strategies this package implements.
BINNING_STRATEGIES: tuple[str, ...] = ("equal_width", "equal_mass")


def _check_n_bins(n_bins: int) -> int:
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)):
        raise TypeError(f"n_bins must be an integer, got {type(n_bins).__name__}")
    n = int(n_bins)
    if n < 1:
        raise ValueError(f"n_bins must be at least 1, got {n}")
    return n


def bin_edges(
    forecasts: ArrayLike,
    *,
    n_bins: int = 10,
    strategy: str = "equal_width",
) -> np.ndarray:
    """Return ``n_bins + 1`` monotone non-decreasing bin edges.

    For ``equal_width`` the edges do not depend on ``forecasts`` at all; the
    argument is accepted so that both strategies share a signature.

    Raises
    ------
    ValueError
        If ``strategy`` is not in :data:`BINNING_STRATEGIES` or ``n_bins < 1``.
    """
    n = _check_n_bins(n_bins)
    f = np.asarray(forecasts, dtype=float)
    if strategy == "equal_width":
        return np.linspace(0.0, 1.0, n + 1)
    if strategy == "equal_mass":
        if f.size == 0:
            raise ValueError("equal_mass binning needs at least one forecast")
        quantiles = np.linspace(0.0, 1.0, n + 1)
        edges = np.quantile(f, quantiles, method="linear")
        edges[0] = min(0.0, float(edges[0]))
        edges[-1] = max(1.0, float(edges[-1]))
        return np.maximum.accumulate(edges)
    raise ValueError(
        f"unknown binning strategy {strategy!r}; expected one of {BINNING_STRATEGIES}"
    )


def assign_bins(
    forecasts: ArrayLike,
    *,
    n_bins: int = 10,
    strategy: str = "equal_width",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(labels, edges)``: a bin index per forecast, and the edges used.

    ``labels`` are in ``[0, n_bins - 1]``. Empty bins keep their index, so the
    label set may have gaps; callers that summarise per bin should drop the
    empty ones rather than assume ``n_bins`` groups.
    """
    n = _check_n_bins(n_bins)
    f = np.asarray(forecasts, dtype=float)
    edges = bin_edges(f, n_bins=n, strategy=strategy)
    if n == 1:
        return np.zeros(f.shape, dtype=np.int64), edges
    labels = np.digitize(f, edges[1:-1], right=False).astype(np.int64)
    return np.clip(labels, 0, n - 1), edges
