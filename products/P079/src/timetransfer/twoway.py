"""Baseline 1: the symmetric-delay two-way offset estimator, and its exact bias.

The estimator
-------------
With the notation of :mod:`timetransfer.exchange` and a remote clock offset
``theta`` held constant over one exchange, the two observable dwell times are

    t2 - t1 = d_ab + theta                                                (1)
    t4 - t3 = d_ba - theta                                                (2)

Subtracting (2) from (1) and halving gives the standard estimator

    theta_hat = ((t2 - t1) - (t4 - t3)) / 2                               (3)
              = theta + (d_ab - d_ba) / 2

and adding them gives the round-trip, which is free of ``theta``:

    d_hat = ((t2 - t1) + (t4 - t3)) / 2 = (d_ab + d_ba) / 2               (4)

Equations (3) and (4) are the offset and delay of Mills, "Internet time
synchronization: the Network Time Protocol", *IEEE Transactions on
Communications* 39(10):1482-1493 (1991), section III, and the mean path delay
of IEEE Std 1588-2019, section 7.4.2. Units: seconds throughout; the
implementation differences integer nanoseconds and converts once.

The closed-form bias
--------------------
Define the exchange asymmetry ``A = d_ab - d_ba`` [s]. From (3),

    theta_hat - theta = A / 2                                             (5)

exactly, with no approximation and no dependence on the magnitude of either
delay, on the turnaround ``t3 - t2``, or on the round trip. Three consequences
are worth stating because each is a testable claim:

* **Exact under symmetry.** ``A = 0`` gives ``theta_hat = theta`` with no error
  term at all -- in exact arithmetic. In floating point the estimator inherits
  the quantisation of the four timestamps,
  ``u_q = delta / sqrt(12)`` (:mod:`timetransfer.timebase`), so the
  implementation claim is *agreement to within rounding*, not equality.
* **Linear, slope one half.** The bias is ``A/2`` regardless of how ``A``
  arises, so a declared static asymmetry of 1 ms puts exactly 500 us on the
  offset. ``validation/validate_symmetric_bias.py`` fits that slope.
* **Unobservable.** Equations (1) and (2) are two equations in the three
  unknowns ``theta``, ``d_ab`` and ``d_ba``. ``theta`` and ``A/2`` enter the
  observable difference identically, so no amount of data from the link alone
  separates them. Averaging reduces the *variable* part of ``A`` and does
  nothing whatever to the static part. This is the ceiling on every estimator
  in this package, including the learned one.

Validity range
--------------
``theta`` constant over one exchange. With a fractional frequency offset ``a``
and a round trip ``R``, treating ``theta`` as constant introduces an additional
term of order ``a * R / 2``: at ``a = 1e-6`` and ``R = 10 ms`` that is 5 ns, and
:func:`offset_skew_correction_s` returns it so a caller can check it against
the quantisation floor rather than assume it is negligible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import ConfigurationError, InsufficientDataError
from .exchange import ExchangeSet
from .timebase import NS_PER_S

__all__ = [
    "TwoWayEstimate",
    "asymmetry_bias_s",
    "asymmetry_from_bias_s",
    "offset_skew_correction_s",
    "symmetric_offsets",
    "two_way_estimate",
]


def asymmetry_bias_s(asymmetry_s: float | np.ndarray) -> float | np.ndarray:
    """Bias [s] that an exchange asymmetry puts on the symmetric estimator.

    ``bias = asymmetry / 2``, equation (5) of the module docstring. Exact.

    Parameters
    ----------
    asymmetry_s:
        ``d_ab - d_ba`` [s]. May be negative; a negative asymmetry biases the
        estimate low.
    """
    value = np.asarray(asymmetry_s, dtype=np.float64) / 2.0
    if not np.all(np.isfinite(value)):
        raise ConfigurationError(f"asymmetry_s must be finite, got {asymmetry_s!r}")
    return float(value) if np.ndim(asymmetry_s) == 0 else value


def asymmetry_from_bias_s(bias_s: float | np.ndarray) -> float | np.ndarray:
    """Inverse of :func:`asymmetry_bias_s`: ``asymmetry = 2 * bias`` [s]."""
    value = np.asarray(bias_s, dtype=np.float64) * 2.0
    if not np.all(np.isfinite(value)):
        raise ConfigurationError(f"bias_s must be finite, got {bias_s!r}")
    return float(value) if np.ndim(bias_s) == 0 else value


def offset_skew_correction_s(
    round_trip_s: float | np.ndarray, skew_s_per_s: float
) -> float | np.ndarray:
    """Magnitude [s] of the term dropped by treating ``theta`` as constant.

    The estimator attributes its result to the exchange midpoint. Over a round
    trip ``R`` the remote clock's offset moves by ``a * R``, and the symmetric
    combination retains half of it, so the neglected term is bounded by
    ``|a| * R / 2``.

    Parameters
    ----------
    round_trip_s:
        Measured round trip ``d_ab + d_ba`` [s], must be >= 0.
    skew_s_per_s:
        Fractional frequency offset [s/s].

    Returns
    -------
    float or ndarray
        ``|skew| * round_trip / 2`` [s]. Compare it against the quantisation
        floor before calling it negligible.
    """
    rtt = np.asarray(round_trip_s, dtype=np.float64)
    if np.any(rtt < 0.0):
        raise ConfigurationError("round_trip_s must be >= 0")
    value = abs(float(skew_s_per_s)) * rtt / 2.0
    return float(value) if np.ndim(round_trip_s) == 0 else value


def symmetric_offsets(exchanges: ExchangeSet) -> tuple[np.ndarray, np.ndarray]:
    """Per-exchange symmetric offset and mean path delay [s].

    Parameters
    ----------
    exchanges:
        A non-empty exchange set.

    Returns
    -------
    tuple of ndarray
        ``(offset_s, delay_s)``, equations (3) and (4). Both are computed from
        the exact integer dwell times and divided once, so the only rounding is
        the final conversion to seconds.
    """
    if exchanges.n == 0:
        raise InsufficientDataError(
            "symmetric_offsets needs at least one exchange", n_usable=0, n_required=1
        )
    fwd = exchanges.forward_dwell_ns().astype(np.float64)
    rev = exchanges.reverse_dwell_ns().astype(np.float64)
    offset_s = 0.5 * (fwd - rev) / NS_PER_S
    delay_s = 0.5 * (fwd + rev) / NS_PER_S
    return offset_s, delay_s


@dataclass(frozen=True)
class TwoWayEstimate:
    """Result of the symmetric-delay estimator over a set of exchanges.

    Attributes
    ----------
    offset_s:
        Per-exchange offset estimates [s], equation (3).
    delay_s:
        Per-exchange mean path delay estimates [s], equation (4).
    midpoint_s:
        Local-clock midpoint of each exchange [s]; the epoch each offset
        belongs to.
    offset_mean_s, offset_median_s:
        Point summaries [s]. The median is reported alongside the mean because
        the queueing component of the delay is right-skewed, so the two differ
        systematically and the difference is informative.
    delay_min_s:
        Smallest mean path delay observed [s]. The minimum-filtered delay is
        the usual proxy for the path floor; it is a biased-low estimator of it
        and is reported, not corrected.
    n:
        Number of exchanges used.
    skew_correction_bound_s:
        ``|a| * R_max / 2`` evaluated at ``assumed_skew_s_per_s``, the bound on
        the term dropped by treating the offset as constant within an exchange.
    assumed_skew_s_per_s:
        The skew used for that bound. Zero means the bound was not requested.
    """

    offset_s: np.ndarray
    delay_s: np.ndarray
    midpoint_s: np.ndarray
    offset_mean_s: float
    offset_median_s: float
    delay_min_s: float
    n: int
    skew_correction_bound_s: float
    assumed_skew_s_per_s: float

    def corrected_offset_s(self, bias_s: float | np.ndarray) -> np.ndarray:
        """Offsets with an asymmetry bias removed: ``offset - bias`` [s].

        ``bias_s`` is ``A/2``, whatever produced it -- a declared value, the
        median-filtered estimator, or the learned corrector. Subtracting is the
        whole of the correction; equation (5) leaves nothing else to do.
        """
        bias = np.asarray(bias_s, dtype=np.float64)
        if bias.ndim not in (0, 1):
            raise ConfigurationError(f"bias_s must be scalar or 1-D, got shape {bias.shape}")
        if bias.ndim == 1 and bias.size != self.n:
            raise ConfigurationError(
                f"bias_s has {bias.size} entries, expected {self.n}"
            )
        return self.offset_s - bias

    def as_text(self) -> str:
        """Compact human-readable summary."""
        return (
            f"n exchanges          : {self.n}\n"
            f"offset mean          : {self.offset_mean_s:.9e} s\n"
            f"offset median        : {self.offset_median_s:.9e} s\n"
            f"path delay min       : {self.delay_min_s:.9e} s\n"
            f"skew correction bound: {self.skew_correction_bound_s:.3e} s "
            f"(at a = {self.assumed_skew_s_per_s:.3e} s/s)"
        )


def two_way_estimate(
    exchanges: ExchangeSet, *, assumed_skew_s_per_s: float = 0.0
) -> TwoWayEstimate:
    """Run the symmetric-delay estimator over ``exchanges``.

    This is Baseline 1. It assumes ``d_ab == d_ba`` and is exact under that
    assumption; where the assumption fails the error is ``A/2`` exactly, and
    nothing in this function tries to hide that.

    Parameters
    ----------
    exchanges:
        A non-empty exchange set.
    assumed_skew_s_per_s:
        Fractional frequency offset [s/s] used only to report
        :attr:`TwoWayEstimate.skew_correction_bound_s`. It does not change any
        estimate.

    Returns
    -------
    TwoWayEstimate
    """
    offset_s, delay_s = symmetric_offsets(exchanges)
    rtt_s = exchanges.round_trip_ns().astype(np.float64) / NS_PER_S
    bound = float(
        np.max(offset_skew_correction_s(rtt_s, float(assumed_skew_s_per_s)))
    )
    return TwoWayEstimate(
        offset_s=offset_s,
        delay_s=delay_s,
        midpoint_s=exchanges.midpoint_s(),
        offset_mean_s=float(np.mean(offset_s)),
        offset_median_s=float(np.median(offset_s)),
        delay_min_s=float(np.min(delay_s)),
        n=exchanges.n,
        skew_correction_bound_s=bound,
        assumed_skew_s_per_s=float(assumed_skew_s_per_s),
    )
