"""Analytic change detectors on the normalised residual.

Every detector here consumes the dimensionless normalised residual
``z`` of equation (6) in :mod:`twininvalidate.twin`, which is i.i.d. N(0, 1)
under the in-control hypothesis, and produces a non-negative **statistic
path** of the same shape. An alarm is raised the first time the statistic
exceeds a threshold.

Statistic paths are computed independently of the threshold -- none of the
three recursions below depends on ``h`` -- so a whole delay-against-false-alarm
curve is produced from one simulation by sweeping thresholds over a stored
statistic matrix. That is what makes the Monte-Carlo budget in the README
affordable on two cores.

Detectors
---------
CUSUM
    Page, E. S. (1954), "Continuous Inspection Schemes", *Biometrika* 41(1/2),
    100-115, DOI 10.1093/biomet/41.1-2.100. Two-sided tabular form.
EWMA
    Roberts, S. W. (1959), "Control Chart Tests Based on Geometric Moving
    Averages", *Technometrics* 1, 239-250,
    DOI 10.1080/00401706.1959.10489860. Design guidance for the smoothing
    constant: Lucas, J. M. & Saccucci, M. S. (1990), "Exponentially Weighted
    Moving Average Control Schemes: Properties and Enhancements",
    *Technometrics* 32, 1-12, DOI 10.1080/00401706.1990.10484583.
Windowed GLR
    Willsky, A. S. & Jones, H. L. (1976), "A Generalized Likelihood Ratio
    Approach to the Detection and Estimation of Jumps in Linear Systems",
    *IEEE Transactions on Automatic Control* 21, 108-112. The windowed
    maximum-over-onset form.

General background, including the optimality properties of CUSUM and the
average-run-length framework: Basseville, M. & Nikiforov, I. V. (1993),
*Detection of Abrupt Changes: Theory and Application*, Prentice-Hall; Lorden,
G. (1971), "Procedures for reacting to a change in distribution", *Annals of
Mathematical Statistics* 42(6), 1897-1908.

Validity range
--------------
All three statistics assume the residual is unit-variance, zero-mean and white
in control. All three are **mean-shift** statistics: none of them is the
likelihood-ratio test for a change in residual *variance*, and the measured
consequence of that mis-specification is the central negative result of this
package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_CUSUM_REFERENCE = 0.25
"""Declared CUSUM reference value ``k``, dimensionless.

Set to half the declared design shift ``delta_ref = 0.5`` standard deviations,
which is the reference value for which the one-sided CUSUM is the optimal
sequential test for that shift (Page 1954; Basseville & Nikiforov 1993). It is
declared in advance and is **not** re-chosen after seeing any detection delay.
"""

DEFAULT_EWMA_LAMBDA = 0.10
"""Declared EWMA smoothing constant, dimensionless, in the 0.05-0.25 range
recommended by Lucas & Saccucci (1990) for small shifts."""

DEFAULT_GLR_WINDOW = 100
"""Declared GLR window length in samples (5 s at the reference 20 Hz)."""


def _as_2d(z: np.ndarray) -> np.ndarray:
    """Validate and reshape a residual stream to ``(n_runs, n_samples)``."""
    arr = np.asarray(z, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"residuals must be 1-D or 2-D, got ndim={arr.ndim}")
    if arr.shape[1] == 0:
        raise ValueError("residual streams must contain at least one sample")
    if not np.all(np.isfinite(arr)):
        raise ValueError("residuals must be finite; found NaN or inf")
    return arr


def cusum_statistic(z: np.ndarray, reference: float = DEFAULT_CUSUM_REFERENCE) -> np.ndarray:
    """Two-sided tabular CUSUM statistic path (Page 1954).

    Recursions, with ``S+_{-1} = S-_{-1} = 0``:

        S+_k = max(0, S+_{k-1} + z_k - k)
        S-_k = max(0, S-_{k-1} - z_k - k)
        g_k  = max(S+_k, S-_k)

    Parameters
    ----------
    z:
        Normalised residuals, shape ``(n_runs, n_samples)`` or ``(n_samples,)``.
        Dimensionless.
    reference:
        Reference value ``k``, dimensionless, strictly positive. Must be
        declared before the experiment.

    Returns
    -------
    Statistic path, shape ``(n_runs, n_samples)``, dimensionless and
    non-negative.
    """
    arr = _as_2d(z)
    if reference <= 0.0:
        raise ValueError(f"CUSUM reference value must be positive, got {reference}")
    n_runs, n_samples = arr.shape
    out = np.empty((n_runs, n_samples))
    s_up = np.zeros(n_runs)
    s_dn = np.zeros(n_runs)
    for k in range(n_samples):
        zk = arr[:, k]
        s_up = np.maximum(0.0, s_up + zk - reference)
        s_dn = np.maximum(0.0, s_dn - zk - reference)
        out[:, k] = np.maximum(s_up, s_dn)
    return out


def ewma_statistic(z: np.ndarray, lam: float = DEFAULT_EWMA_LAMBDA) -> np.ndarray:
    """EWMA statistic path, standardised by its asymptotic variance (Roberts 1959).

    Recursion with ``A_{-1} = 0``:

        A_k = lam * z_k + (1 - lam) * A_{k-1},
        g_k = |A_k| / sqrt(lam / (2 - lam)).

    The divisor is the **asymptotic** standard deviation of ``A_k`` for
    unit-variance white input. The exact time-varying standard deviation is
    ``sqrt(lam (1 - (1-lam)^{2(k+1)}) / (2 - lam))``, which is smaller early
    on, so using the asymptotic value makes the chart slightly *less* sensitive
    in its first few dozen samples. That choice is deliberate -- it keeps the
    in-control false-alarm rate from being inflated at start-up -- and its
    effect is visible in the first-sample known-answer test.

    Parameters
    ----------
    z:
        Normalised residuals, dimensionless.
    lam:
        Smoothing constant in ``(0, 1]``, dimensionless.

    Returns
    -------
    Statistic path, shape ``(n_runs, n_samples)``, dimensionless.
    """
    arr = _as_2d(z)
    if not 0.0 < lam <= 1.0:
        raise ValueError(f"EWMA lambda must lie in (0, 1], got {lam}")
    sigma = np.sqrt(lam / (2.0 - lam))
    n_runs, n_samples = arr.shape
    out = np.empty((n_runs, n_samples))
    a = np.zeros(n_runs)
    for k in range(n_samples):
        a = lam * arr[:, k] + (1.0 - lam) * a
        out[:, k] = np.abs(a) / sigma
    return out


def glr_statistic(z: np.ndarray, window: int = DEFAULT_GLR_WINDOW) -> np.ndarray:
    """Windowed GLR statistic path for an unknown mean shift (Willsky & Jones 1976).

    For i.i.d. ``z ~ N(mu, 1)`` over the last ``n`` samples, the log-likelihood
    ratio of ``H1: mu != 0`` against ``H0: mu = 0``, with ``mu`` replaced by its
    maximum-likelihood estimate ``mu_hat = S_n / n`` where ``S_n`` is the sum of
    those ``n`` samples, is

        log Lambda_n = S_n^2 / (2 n).

    The onset is unknown, so the statistic maximises over window lengths:

        g_k = max_{1 <= n <= min(window, k+1)} S_n(k)^2 / (2 n).

    Parameters
    ----------
    z:
        Normalised residuals, dimensionless.
    window:
        Maximum window length in samples, a positive integer. Declared in
        advance; it bounds both the longest change the test can accumulate and
        the per-sample cost.

    Returns
    -------
    Statistic path, shape ``(n_runs, n_samples)``, in nats (a log-likelihood
    ratio), non-negative.
    """
    arr = _as_2d(z)
    if window < 1:
        raise ValueError(f"GLR window must be at least 1 sample, got {window}")
    n_runs, n_samples = arr.shape
    window = int(min(window, n_samples))
    cum = np.concatenate([np.zeros((n_runs, 1)), np.cumsum(arr, axis=1)], axis=1)
    out = np.zeros((n_runs, n_samples))
    for n in range(1, window + 1):
        # S_n(k) = cum[k+1] - cum[k+1-n], defined for k >= n-1.
        sums = cum[:, n:] - cum[:, :-n]
        np.maximum(out[:, n - 1 :], sums * sums / (2.0 * n), out=out[:, n - 1 :])
    return out


DEFAULT_VARCUSUM_REFERENCE = 0.031
"""Declared reference value for the variance-CUSUM oracle, dimensionless.

Half the design shift in ``z^2`` units, by the same Page (1954) rule used for
the mean CUSUM. The declared ``noise_variance`` scenario multiplies the
asset's **process-noise covariance** by 2, which raises the **residual**
variance only to about 1.06, because the measurement noise ``R`` dominates the
innovation variance ``S`` on the reference channel. The shift in ``E[z^2]`` is
therefore about 0.062 and the reference value is 0.031.

The 1.062 figure came from a 300-run measurement made while the scenarios were
being declared, before any detection delay had been computed. A later and
longer measurement, the one printed by ``validation/validate_twin.py``, gives
1.0585, i.e. a reference value of 0.029. The declared 0.031 was **not**
revised to match, because revising a declared constant after a delay is known
is the one thing this package refuses to do; the 7 % difference is recorded
here instead.

This number is where the oracle gets its oracle status: it is derived from the
post-change residual variance of the specific scenario, which a deployed
monitor does not know. The first version of this package set it to 0.5 on the
mistaken premise that a process-noise doubling doubles the residual variance.
The consequence was measurable -- the "oracle" was then *slower* than the
mis-specified mean-shift baselines it was meant to bound, 506 samples against
213 -- and the episode is recorded in ``validation/VALIDATION.md``.
"""


def variance_cusum_statistic(
    z: np.ndarray, reference: float = DEFAULT_VARCUSUM_REFERENCE
) -> np.ndarray:
    """One-sided CUSUM on ``z^2 - 1``: the correctly specified variance test.

    **This is not one of the three declared baselines.** It exists only to
    measure how much the three mean-shift baselines lose on a change they are
    mis-specified for, which is the central negative result of this package. It
    is labelled an *oracle* throughout because it is given the change type in
    advance, which no deployed monitor is.

    Recursion, with ``V_{-1} = 0``:

        V_k = max(0, V_{k-1} + (z_k^2 - 1) - c)

    Under the in-control hypothesis ``z^2 ~ chi^2_1``, so ``E[z^2 - 1] = 0``
    and the drift is negative by ``c``; after a variance multiplier ``rho`` the
    drift becomes ``rho - 1 - c``, positive for ``rho > 1 + c``. This is the
    standard Page construction applied to the sufficient statistic for a scale
    change in a zero-mean Gaussian (Basseville & Nikiforov 1993).

    Parameters
    ----------
    z:
        Normalised residuals, dimensionless.
    reference:
        Reference value ``c`` in ``z^2`` units, strictly positive.

    Returns
    -------
    Statistic path, shape ``(n_runs, n_samples)``, dimensionless.
    """
    arr = _as_2d(z)
    if reference <= 0.0:
        raise ValueError(f"variance CUSUM reference must be positive, got {reference}")
    n_runs, n_samples = arr.shape
    out = np.empty((n_runs, n_samples))
    v = np.zeros(n_runs)
    for k in range(n_samples):
        v = np.maximum(0.0, v + arr[:, k] ** 2 - 1.0 - reference)
        out[:, k] = v
    return out


DETECTORS = {
    "cusum": cusum_statistic,
    "ewma": ewma_statistic,
    "glr": glr_statistic,
    "varcusum": variance_cusum_statistic,
}
"""Name to statistic function, for the CLI and the sweep helpers.

``cusum``, ``ewma`` and ``glr`` are the three declared analytic baselines.
``varcusum`` is the oracle variance test of
:func:`variance_cusum_statistic` and is reported separately everywhere.
"""

BASELINES = ("cusum", "ewma", "glr")
"""The three declared analytic baselines, in the order the spec names them."""


def statistic_path(z: np.ndarray, name: str, **kwargs: float) -> np.ndarray:
    """Dispatch to one of :data:`DETECTORS` by name."""
    if name not in DETECTORS:
        raise ValueError(f"unknown detector {name!r}; choose from {sorted(DETECTORS)}")
    return DETECTORS[name](z, **kwargs)  # type: ignore[operator]


def first_alarm(statistic: np.ndarray, threshold: float) -> np.ndarray:
    """First sample index at which ``statistic`` exceeds ``threshold``.

    Parameters
    ----------
    statistic:
        Statistic path, shape ``(n_runs, n_samples)``.
    threshold:
        Alarm threshold in the statistic's own units.

    Returns
    -------
    Integer array of shape ``(n_runs,)``. Entry ``-1`` means the run was
    right-censored: no alarm occurred within the simulated horizon. Callers
    must handle ``-1``; silently dropping censored runs biases every average
    run length downwards.
    """
    stat = _as_2d(statistic)
    exceed = stat > threshold
    any_alarm = exceed.any(axis=1)
    idx = np.argmax(exceed, axis=1)
    return np.where(any_alarm, idx, -1).astype(np.int64)


@dataclass(frozen=True)
class DetectorSpec:
    """A detector with its declared design constants, excluding the threshold.

    The threshold is deliberately *not* part of this object: it is produced by
    :mod:`twininvalidate.thresholds` from a declared false-alarm target, and
    keeping the two apart is what stops a threshold being chosen after a delay
    is known.
    """

    name: str
    reference: float = DEFAULT_CUSUM_REFERENCE
    lam: float = DEFAULT_EWMA_LAMBDA
    window: int = DEFAULT_GLR_WINDOW
    var_reference: float = DEFAULT_VARCUSUM_REFERENCE

    def __post_init__(self) -> None:
        if self.name not in DETECTORS:
            raise ValueError(f"unknown detector {self.name!r}; choose from {sorted(DETECTORS)}")

    def kwargs(self) -> dict[str, float]:
        """The keyword arguments this detector's statistic function takes."""
        if self.name == "cusum":
            return {"reference": self.reference}
        if self.name == "ewma":
            return {"lam": self.lam}
        if self.name == "varcusum":
            return {"reference": self.var_reference}
        return {"window": self.window}

    def statistic(self, z: np.ndarray) -> np.ndarray:
        """Statistic path for ``z``, shape ``(n_runs, n_samples)``."""
        return statistic_path(z, self.name, **self.kwargs())

    def label(self) -> str:
        """Short human-readable label including the declared design constant."""
        if self.name == "cusum":
            return f"CUSUM (k={self.reference:g})"
        if self.name == "ewma":
            return f"EWMA (lambda={self.lam:g})"
        if self.name == "varcusum":
            return f"variance-CUSUM oracle (c={self.var_reference:g})"
        return f"GLR (window={self.window:d})"
