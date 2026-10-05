"""EWMA and CUSUM control charts operating on standardised telemetry deviates.

Both charts take the *standardised deviate* ``u_t = (x_t - mu0) / sigma`` so that
every threshold is in sigma units and the designs in :mod:`telemetryool.arl`
apply directly.  Both are implemented in two forms:

``run`` / ``statistic``
    Single sequence, returns the full statistic trace for inspection.
``run_windows``
    ``(n_windows, window_length)`` block form, state reset at the start of each
    row, used by the Monte-Carlo harness.  No Python loop over windows.

References
----------
Page, E. S. (1954). "Continuous Inspection Schemes." *Biometrika* 41(1/2),
    100-115.
Roberts, S. W. (1959). "Control Chart Tests Based on Geometric Moving Averages."
    *Technometrics* 1(3), 239-250.  Origin of the EWMA chart.
Lucas, J. M. and Saccucci, M. S. (1990). "Exponentially Weighted Moving Average
    Control Schemes: Properties and Enhancements." *Technometrics* 32(1), 1-12.
Montgomery, D. C. (2013). *Introduction to Statistical Quality Control*,
    7th ed., Wiley, ch. 9.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .arl import ewma_sigma_z
from .runs import first_run_index

__all__ = ["ChartRun", "EwmaChart", "CusumChart"]


@dataclass(frozen=True)
class ChartRun:
    """Trace of one chart applied to one sequence.

    Attributes
    ----------
    statistic
        Shape ``(n,)`` for EWMA (the smoothed statistic ``z_t``, sigma units) or
        ``(2, n)`` for CUSUM (upper arm then lower arm, sigma units).
    upper_limit, lower_limit
        Shape ``(n,)``, the control limits actually applied at each sample,
        sigma units.  Constant unless ``time_varying_limits`` is set.
    breach
        Shape ``(n,)`` bool: statistic outside its limits at that sample,
        before persistence.
    alarm_index
        Index at which the persistence requirement is first met, or ``-1``.
    """

    statistic: NDArray[np.float64]
    upper_limit: NDArray[np.float64]
    lower_limit: NDArray[np.float64]
    breach: NDArray[np.bool_]
    alarm_index: int


def _as_2d(u: NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(u, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D array of deviates, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("input array is empty")
    return arr


@dataclass(frozen=True)
class EwmaChart:
    """Two-sided EWMA chart on standardised deviates.

    ``z_t = lam * u_t + (1 - lam) * z_{t-1}``, ``z_0 = 0`` (Roberts 1959).
    Control limits are ``+/- limit_mult * sigma_z`` with
    ``sigma_z = sqrt(lam / (2 - lam))`` (steady state), or, when
    ``time_varying_limits`` is true, ``+/- limit_mult * sigma_z *
    sqrt(1 - (1 - lam)**(2 (t + 1)))`` (Montgomery 2013 ch. 9).

    Parameters
    ----------
    lam
        Smoothing weight in (0, 1], dimensionless.  Small values detect small
        sustained shifts; ``lam = 1`` reduces the chart to a Shewhart chart.
    limit_mult
        Control-limit multiplier ``L``, dimensionless, > 0.  Obtain it from
        :func:`telemetryool.arl.design_ewma_L` rather than guessing.
    persistence
        Consecutive breaching samples required to raise an alarm.  The
        :mod:`telemetryool.arl` EWMA designs assume ``persistence = 1``; any
        other value changes the delivered ``alpha_W`` and must be calibrated
        empirically.
    time_varying_limits
        Use the exact finite-``t`` limits instead of the steady-state ones.
        This shortens ARL0 relative to the steady-state design, and
        :func:`telemetryool.arl.ewma_arl_markov` does **not** model it.

    Validity
    --------
    The designed false-alarm rate holds for independent normal deviates with
    known ``sigma``.  Under serial correlation it does not; the error is
    measured in ``validation/validate_far_design.py``.
    """

    lam: float
    limit_mult: float
    persistence: int = 1
    time_varying_limits: bool = False

    def __post_init__(self) -> None:
        if not (0.0 < float(self.lam) <= 1.0):
            raise ValueError(f"lam must lie in (0, 1], got {self.lam!r}")
        if float(self.limit_mult) <= 0.0:
            raise ValueError(f"limit_mult must be > 0, got {self.limit_mult!r}")
        if not isinstance(self.persistence, (int, np.integer)) or self.persistence < 1:
            raise ValueError(f"persistence must be an integer >= 1, got {self.persistence!r}")

    @property
    def sigma_z(self) -> float:
        """Steady-state standard deviation of ``z``, sigma units (dimensionless)."""
        return ewma_sigma_z(self.lam)

    def limits(self, n: int) -> NDArray[np.float64]:
        """Upper control limit at each of ``n`` samples, sigma units."""
        base = float(self.limit_mult) * self.sigma_z
        if not self.time_varying_limits:
            return np.full(n, base)
        t = np.arange(1, n + 1, dtype=float)
        return base * np.sqrt(1.0 - (1.0 - float(self.lam)) ** (2.0 * t))

    def statistic(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """EWMA statistic ``z`` for a batch of sequences, shape ``(n_rows, n)``."""
        arr = _as_2d(u)
        lam = float(self.lam)
        z = np.empty_like(arr)
        prev = np.zeros(arr.shape[0])
        for t in range(arr.shape[1]):
            prev = lam * arr[:, t] + (1.0 - lam) * prev
            z[:, t] = prev
        return z

    def breach_mask(self, u: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Per-sample out-of-limit mask, shape ``(n_rows, n)``, before persistence."""
        arr = _as_2d(u)
        z = self.statistic(arr)
        lim = self.limits(arr.shape[1])[None, :]
        return np.abs(z) > lim

    def run_windows(self, u: NDArray[np.float64]) -> NDArray[np.int64]:
        """Alarm index per window, shape ``(n_windows,)``; ``-1`` where no alarm."""
        return first_run_index(self.breach_mask(u), int(self.persistence))

    def run(self, u: NDArray[np.float64]) -> ChartRun:
        """Apply the chart to a single sequence and return the full trace."""
        arr = _as_2d(u)
        if arr.shape[0] != 1:
            raise ValueError("run() takes a single sequence; use run_windows() for a batch")
        z = self.statistic(arr)[0]
        lim = self.limits(arr.shape[1])
        breach = np.abs(z) > lim
        idx = int(first_run_index(breach[None, :], int(self.persistence))[0])
        return ChartRun(z, lim, -lim, breach, idx)


@dataclass(frozen=True)
class CusumChart:
    """Two-sided tabular CUSUM on standardised deviates (Page 1954).

    ``C_plus_t  = max(0, C_plus_{t-1}  + u_t - k)``
    ``C_minus_t = max(0, C_minus_{t-1} - u_t - k)``
    with ``C_plus_0 = C_minus_0 = 0``; a breach is ``max(C_plus, C_minus) > h``.

    Parameters
    ----------
    k
        Reference value in sigma units, > 0.  ``k = delta_target / 2`` is the
        likelihood-ratio-optimal choice for detecting a shift of
        ``delta_target`` sigma (Page 1954; Montgomery 2013 ch. 9).
    h
        Decision interval in sigma units, > 0.  Obtain it from
        :func:`telemetryool.arl.design_cusum_h`.
    persistence
        Consecutive breaching samples required to raise.  The
        :mod:`telemetryool.arl` CUSUM designs assume ``persistence = 1``.

    Validity
    --------
    Independent normal deviates, known ``sigma``.  The CUSUM arms are not reset
    after an alarm inside a window: the window formulation only asks whether an
    alarm occurred, so reset policy does not enter.
    """

    k: float
    h: float
    persistence: int = 1

    def __post_init__(self) -> None:
        if float(self.k) <= 0.0:
            raise ValueError(f"k must be > 0, got {self.k!r}")
        if float(self.h) <= 0.0:
            raise ValueError(f"h must be > 0, got {self.h!r}")
        if not isinstance(self.persistence, (int, np.integer)) or self.persistence < 1:
            raise ValueError(f"persistence must be an integer >= 1, got {self.persistence!r}")

    def arms(self, u: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Upper and lower CUSUM arms, each shape ``(n_rows, n)``, sigma units."""
        arr = _as_2d(u)
        k = float(self.k)
        c_up = np.empty_like(arr)
        c_dn = np.empty_like(arr)
        up = np.zeros(arr.shape[0])
        dn = np.zeros(arr.shape[0])
        for t in range(arr.shape[1]):
            up = np.maximum(0.0, up + arr[:, t] - k)
            dn = np.maximum(0.0, dn - arr[:, t] - k)
            c_up[:, t] = up
            c_dn[:, t] = dn
        return c_up, c_dn

    def breach_mask(self, u: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Per-sample out-of-limit mask, shape ``(n_rows, n)``, before persistence."""
        c_up, c_dn = self.arms(u)
        h = float(self.h)
        return (c_up > h) | (c_dn > h)

    def run_windows(self, u: NDArray[np.float64]) -> NDArray[np.int64]:
        """Alarm index per window, shape ``(n_windows,)``; ``-1`` where no alarm."""
        return first_run_index(self.breach_mask(u), int(self.persistence))

    def run(self, u: NDArray[np.float64]) -> ChartRun:
        """Apply the chart to a single sequence and return the full trace."""
        arr = _as_2d(u)
        if arr.shape[0] != 1:
            raise ValueError("run() takes a single sequence; use run_windows() for a batch")
        c_up, c_dn = self.arms(arr)
        stat = np.vstack([c_up[0], c_dn[0]])
        h = float(self.h)
        breach = (c_up[0] > h) | (c_dn[0] > h)
        idx = int(first_run_index(breach[None, :], int(self.persistence))[0])
        n = arr.shape[1]
        return ChartRun(stat, np.full(n, h), np.zeros(n), breach, idx)
