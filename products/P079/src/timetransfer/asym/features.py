"""Window features for the delay-asymmetry corrector, and why they are these.

What a corrector is allowed to see
----------------------------------
The target is the mean asymmetry bias of a window of exchanges,
``b = mean_k(d_ab,k - d_ba,k) / 2`` [s]. The features must be computable from
the four timestamps alone, and they must not leak the clock offset, because an
estimator that has learned to read ``theta`` off the features has learned
nothing about asymmetry.

The forward and reverse dwell times are

    fwd_k = t2 - t1 = f_ab + q_ab,k + theta_k
    rev_k = t4 - t3 = f_ba + q_ba,k - theta_k

Both carry ``theta``. **Centring each direction on its own window minimum**

    fwd_c = fwd - min(fwd),   rev_c = rev - min(rev)

removes ``theta`` (constant to within the clock's drift over the window) and
the floor ``f`` together, leaving the queueing excursions. That is the whole
feature construction, and it has a consequence worth being blunt about: the
static asymmetry ``f_ab - f_ba`` is *also* removed, so it is not in the
features. The corrector can only reach the static part through whatever
correlation the generating process puts between it and the shape of the
queueing. On a real path that correlation may be strong, weak or absent, and
:mod:`timetransfer.asym.data` declares exactly what correlation its generator
has.

The sixteen features
--------------------
Eight per direction, computed on the centred dwells, plus differences:

``mean``, ``median``, ``std``, ``iqr``, ``p90``, ``mean - median`` (a
skewness proxy that needs no third moment), and the count and span of the
window. The differences ``mean_fwd_c - mean_rev_c``, ``median_fwd_c -
median_rev_c`` and ``std_fwd_c - std_rev_c`` are included explicitly rather
than left for the model to form, because the median difference *is* the
median-filtered baseline and a model that cannot see it cannot be fairly
compared against it.

Units are seconds throughout, except the dimensionless count.
"""

from __future__ import annotations

import numpy as np

from ..errors import ConfigurationError, InsufficientDataError
from ..exchange import ExchangeSet
from ..timebase import NS_PER_S

__all__ = ["FEATURE_NAMES", "MIN_WINDOW", "centred_dwells_s", "window_features"]

MIN_WINDOW = 8
"""Smallest window the feature set is defined for; below it the quantiles are meaningless."""

FEATURE_NAMES: tuple[str, ...] = (
    "fwd_mean_s",
    "fwd_median_s",
    "fwd_std_s",
    "fwd_iqr_s",
    "fwd_p90_s",
    "fwd_mean_minus_median_s",
    "rev_mean_s",
    "rev_median_s",
    "rev_std_s",
    "rev_iqr_s",
    "rev_p90_s",
    "rev_mean_minus_median_s",
    "mean_difference_s",
    "median_difference_s",
    "std_difference_s",
    "rtt_excess_mean_s",
)
"""Names of the sixteen features, in the order :func:`window_features` returns them."""


def centred_dwells_s(exchanges: ExchangeSet) -> tuple[np.ndarray, np.ndarray]:
    """Forward and reverse dwells [s], each centred on its own window minimum.

    Centring removes the clock offset and the direction's delay floor together;
    see the module docstring for what that costs.

    Parameters
    ----------
    exchanges:
        An exchange set with at least :data:`MIN_WINDOW` exchanges.

    Returns
    -------
    tuple of ndarray
        ``(fwd_centred_s, rev_centred_s)``, both >= 0 with at least one zero.
    """
    if exchanges.n < MIN_WINDOW:
        raise InsufficientDataError(
            f"window features need at least {MIN_WINDOW} exchanges, got {exchanges.n}",
            n_usable=exchanges.n,
            n_required=MIN_WINDOW,
        )
    fwd = exchanges.forward_dwell_ns().astype(np.float64) / NS_PER_S
    rev = exchanges.reverse_dwell_ns().astype(np.float64) / NS_PER_S
    return fwd - float(np.min(fwd)), rev - float(np.min(rev))


def window_features(exchanges: ExchangeSet) -> np.ndarray:
    """The sixteen window features of :data:`FEATURE_NAMES`, in seconds.

    Parameters
    ----------
    exchanges:
        An exchange set with at least :data:`MIN_WINDOW` exchanges.

    Returns
    -------
    ndarray
        Shape ``(16,)``, ``float64``. Every entry is finite; a degenerate
        window (all dwells identical) gives zeros rather than ``nan``.
    """
    fwd_c, rev_c = centred_dwells_s(exchanges)
    rtt = exchanges.round_trip_ns().astype(np.float64) / NS_PER_S
    fwd_q = np.percentile(fwd_c, [25.0, 75.0, 90.0])
    rev_q = np.percentile(rev_c, [25.0, 75.0, 90.0])
    fwd_mean = float(np.mean(fwd_c))
    fwd_median = float(np.median(fwd_c))
    rev_mean = float(np.mean(rev_c))
    rev_median = float(np.median(rev_c))
    fwd_std = float(np.std(fwd_c, ddof=1))
    rev_std = float(np.std(rev_c, ddof=1))
    features = np.array(
        [
            fwd_mean,
            fwd_median,
            fwd_std,
            float(fwd_q[1] - fwd_q[0]),
            float(fwd_q[2]),
            fwd_mean - fwd_median,
            rev_mean,
            rev_median,
            rev_std,
            float(rev_q[1] - rev_q[0]),
            float(rev_q[2]),
            rev_mean - rev_median,
            fwd_mean - rev_mean,
            fwd_median - rev_median,
            fwd_std - rev_std,
            float(np.mean(rtt) - np.min(rtt)),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(features)):
        raise ConfigurationError(
            "window features are not finite; the exchange set contains degenerate timestamps"
        )
    return features
