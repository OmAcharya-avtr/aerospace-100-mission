"""Residual-window features for the learned detector/isolator.

Sixteen features are computed from a window of ``W`` normalised residuals and
**nothing else**.  No true state, no fault label, no plant parameter and no
quantity the flight software would not already hold enters the feature vector,
so a model trained on them is using the same information the chi-squared and
CUSUM baselines use.

Under the fault-free hypothesis every residual is ``N(0, I_2)``, which fixes
the expected value of most of these features and makes them interpretable:

===========================  ==================================================
``mean_ch{c}``               0
``std_ch{c}``                1
``slope_ch{c}``              0
``autocorr1_ch{c}``          0 (a consistent filter's innovation is white)
``max_abs_ch{c}``            E[max of W standard normals]
``cusum_range_ch{c}``        the range of a random walk, scaled by sqrt(W)
``mean_nis``                 2 (chi-squared with 2 dof)
``max_nis``                  -
``corr_01``                  0
``exceed_frac``              0.01, the tail mass above ``chi2.isf(0.01, 2)``
===========================  ==================================================

The design intent behind each group: the means catch a bias, the slopes and
``cusum_range`` catch a drift or runaway, ``std`` and ``autocorr1`` catch a
stuck sensor (whose residual becomes a smooth random walk rather than white
noise), and ``mean_nis`` / ``exceed_frac`` reproduce what the chi-squared test
already sees.  Whether the extra features earn their keep is a measured
question, answered by the feature-importance table in ``MODEL_CARD.md``.

Units: all features are dimensionless because the residual is normalised.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

__all__ = [
    "N_FEATURES",
    "NIS_EXCEEDANCE_THRESHOLD",
    "feature_names",
    "window_features",
    "feature_matrix",
]

#: Threshold used by the ``exceed_frac`` feature: the 99th percentile of a
#: chi-squared with 2 degrees of freedom, 9.21034037...
NIS_EXCEEDANCE_THRESHOLD: float = float(chi2.isf(0.01, 2))

#: Number of features produced by :func:`window_features`.
N_FEATURES: int = 16


def feature_names() -> tuple[str, ...]:
    """Feature names in the order :func:`window_features` returns them."""
    names: list[str] = []
    for c in (0, 1):
        names += [
            f"mean_ch{c}",
            f"std_ch{c}",
            f"slope_ch{c}",
            f"autocorr1_ch{c}",
            f"max_abs_ch{c}",
            f"cusum_range_ch{c}",
        ]
    names += ["mean_nis", "max_nis", "corr_01", "exceed_frac"]
    return tuple(names)


def _lag1_autocorr(x: NDArray[np.float64]) -> float:
    if x.size < 3:
        return 0.0
    xc = x - float(np.mean(x))
    denom = float(np.dot(xc, xc))
    if denom < 1e-30:
        return 0.0
    return float(np.dot(xc[:-1], xc[1:]) / denom)


def window_features(residual_window: ArrayLike) -> NDArray[np.float64]:
    """Sixteen features of one residual window.

    Parameters
    ----------
    residual_window : array_like, shape (W, 2)
        Normalised residuals, ``W >= 3``.

    Returns
    -------
    ndarray, shape (16,)
        Features in :func:`feature_names` order, all dimensionless and all
        finite (degenerate cases such as a constant channel return 0 for the
        undefined correlation rather than ``nan``).

    Raises
    ------
    ValueError
        If the window is not ``(W, 2)`` with ``W >= 3``, or is not finite.
    """
    r = np.atleast_2d(np.asarray(residual_window, dtype=float))
    if r.ndim != 2 or r.shape[1] != 2:
        raise ValueError(f"residual_window must be (W, 2), got shape {r.shape}")
    if r.shape[0] < 3:
        raise ValueError(f"window must hold at least 3 samples, got {r.shape[0]}")
    if not np.all(np.isfinite(r)):
        raise ValueError("residual_window must be finite")

    w = r.shape[0]
    idx = np.arange(w, dtype=float)
    idx_c = idx - float(np.mean(idx))
    idx_ss = float(np.dot(idx_c, idx_c))

    out: list[float] = []
    for c in (0, 1):
        x = r[:, c]
        xc = x - float(np.mean(x))
        slope = float(np.dot(idx_c, xc) / idx_ss) if idx_ss > 0.0 else 0.0
        cs = np.cumsum(xc)
        out += [
            float(np.mean(x)),
            float(np.std(x)),
            slope * float(w),
            _lag1_autocorr(x),
            float(np.max(np.abs(x))),
            float((np.max(cs) - np.min(cs)) / np.sqrt(w)),
        ]

    nis = np.sum(r * r, axis=1)
    s0 = float(np.std(r[:, 0]))
    s1 = float(np.std(r[:, 1]))
    if s0 < 1e-12 or s1 < 1e-12:
        corr = 0.0
    else:
        cov01 = float(np.mean((r[:, 0] - np.mean(r[:, 0])) * (r[:, 1] - np.mean(r[:, 1]))))
        corr = cov01 / (s0 * s1)
    out += [
        float(np.mean(nis)),
        float(np.max(nis)),
        corr,
        float(np.mean(nis > NIS_EXCEEDANCE_THRESHOLD)),
    ]
    return np.asarray(out, dtype=float)


def feature_matrix(
    residual: ArrayLike, window: int, stride: int = 1, start: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Sliding-window feature matrix.

    Parameters
    ----------
    residual : array_like, shape (N, 2)
        Full normalised residual sequence.
    window : int
        Window length ``W``, at least 3.
    stride : int
        Step between consecutive window *end* indices, at least 1.
    start : int
        Index of the first window's **end** sample.  Values below ``W - 1``
        are raised to ``W - 1``.

    Returns
    -------
    (features, end_index) : tuple
        ``features`` has shape ``(n_windows, 16)``; ``end_index[i]`` is the
        index of the last sample in window ``i``, so a detector decision made
        from that window is attributed to that sample.
    """
    r = np.atleast_2d(np.asarray(residual, dtype=float))
    w = int(window)
    st = int(stride)
    if w < 3:
        raise ValueError(f"window must be >= 3, got {window}")
    if st < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if r.ndim != 2 or r.shape[1] != 2:
        raise ValueError(f"residual must be (N, 2), got shape {r.shape}")
    if not np.all(np.isfinite(r)):
        raise ValueError("residual must be finite")
    first = max(int(start), w - 1)
    ends = np.arange(first, r.shape[0], st, dtype=np.intp)
    if ends.size == 0:
        return np.zeros((0, N_FEATURES)), ends
    return _vectorised_features(r, ends, w), ends


def _vectorised_features(
    r: NDArray[np.float64], ends: NDArray[np.intp], w: int
) -> NDArray[np.float64]:
    """Batched equivalent of :func:`window_features`.

    Kept separate from the reference single-window implementation so the two
    can be checked against each other; ``tests/test_features.py`` asserts they
    agree to 1e-12 on random data, which is what allows the fast path to be
    used everywhere without doubt about what it computes.
    """
    starts = ends - w + 1
    offs = np.arange(w)
    idx = starts[:, None] + offs[None, :]
    win = r[idx]  # (n, w, 2)

    mean = win.mean(axis=1)  # (n, 2)
    centred = win - mean[:, None, :]
    std = np.sqrt((centred**2).mean(axis=1))

    t = offs.astype(float)
    t_c = t - t.mean()
    t_ss = float(np.dot(t_c, t_c))
    if t_ss > 0:
        slope = np.einsum("j,ijc->ic", t_c, centred) / t_ss * float(w)
    else:
        slope = np.zeros_like(mean)

    denom = (centred**2).sum(axis=1)
    num = (centred[:, :-1, :] * centred[:, 1:, :]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        autocorr = np.where(denom > 1e-30, num / np.where(denom > 1e-30, denom, 1.0), 0.0)

    max_abs = np.abs(win).max(axis=1)
    cs = np.cumsum(centred, axis=1)
    cusum_range = (cs.max(axis=1) - cs.min(axis=1)) / np.sqrt(float(w))

    nis = (win * win).sum(axis=2)
    mean_nis = nis.mean(axis=1)
    max_nis = nis.max(axis=1)
    cov01 = (centred[:, :, 0] * centred[:, :, 1]).mean(axis=1)
    ok = (std[:, 0] > 1e-12) & (std[:, 1] > 1e-12)
    prod = std[:, 0] * std[:, 1]
    corr = np.where(ok, cov01 / np.where(ok, prod, 1.0), 0.0)
    exceed = (nis > NIS_EXCEEDANCE_THRESHOLD).mean(axis=1)

    cols = []
    for c in (0, 1):
        cols += [
            mean[:, c],
            std[:, c],
            slope[:, c],
            autocorr[:, c],
            max_abs[:, c],
            cusum_range[:, c],
        ]
    cols += [mean_nis, max_nis, corr, exceed]
    return np.stack(cols, axis=1)
