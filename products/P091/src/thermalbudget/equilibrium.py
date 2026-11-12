"""Build step 1: the equilibrium test. Equilibrium is tested, never assumed.

Every sustained figure this product reports is an average over a window in which
the die temperature is taken to have stopped rising. That assumption is the one
most easily wrong and the most expensive to be wrong about: averaging a run that
is still warming reports a sustained throughput that the board would not in fact
sustain, and the error is in the flattering direction.

So the window is tested. Take the final ``window_s`` seconds of the run, fit a
straight line to the die temperature against time by ordinary least squares, and
compare the fitted slope against a declared threshold. A run whose slope exceeds
the threshold is reported ``not-equilibrated`` and is NOT averaged; it appears in
the report as a named refusal.

Three details that change the answer, each implemented rather than assumed.

**Serial correlation.** Consecutive one-second samples of a thermal channel are
strongly dependent, so the ordinary least-squares standard error of the slope is
too small, often by a factor of several. The slope ESTIMATE is unbiased; its
uncertainty is not. Both standard errors are reported: the textbook one and an
AR(1)-corrected one that inflates the residual variance by
``(1 + rho) / (1 - rho)`` with ``rho`` the lag-1 autocorrelation of the
residuals. This is the classical first-order correction for a regression with
AR(1) errors; see Greene, *Econometric Analysis*, 8th edition, chapter 20
(generalised regression with autocorrelated disturbances), and Newey & West
(1987), *Econometrica* 55(3):703--708, for the heteroskedasticity- and
autocorrelation-consistent family this is the simplest member of. The confidence
interval quoted by this module is the corrected one; the uncorrected one is
printed beside it so the size of the correction is visible.

**The threshold is a declared criterion, not a fact about boards.** There is no
physical constant that says when a die has stopped warming. So the verdict is
reported together with the measured slope, its interval, and a ladder of other
thresholds at which the verdict would change
(:func:`equilibrium_sensitivity`). A reader who disagrees with the default can
read their own answer off the ladder rather than having to re-run anything.

**Missing samples.** A ``-1`` reading is ``nan`` by the time it reaches here and
is dropped. If fewer than ``min_samples`` finite samples remain, no slope is
computed and the verdict is ``INSUFFICIENT_SAMPLES`` -- which is not a pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .provenance import Origin

#: Length of the final window the test is applied to, seconds.
DEFAULT_WINDOW_S: float = 120.0

#: Declared slope threshold, degrees Celsius per second. 0.005 degC/s is 18 degC
#: per hour, and over the 120 s default window it bounds the die-temperature
#: drift inside the averaging window at 0.6 degC. It is a criterion, not a
#: measurement, and :func:`equilibrium_sensitivity` exists because it is.
DEFAULT_SLOPE_THRESHOLD_C_PER_S: float = 0.005

#: Ladder of thresholds the verdict is reported across, degrees Celsius per second.
THRESHOLD_LADDER: tuple[float, ...] = (0.001, 0.002, 0.005, 0.010, 0.020)

#: Fewest finite samples that may produce a slope.
DEFAULT_MIN_SAMPLES: int = 30

#: Coverage of the reported slope interval.
CONFIDENCE: float = 0.95


@dataclass(frozen=True)
class LinearFit:
    """Ordinary least-squares straight-line fit, with both standard errors.

    Attributes
    ----------
    slope, intercept:
        Fit coefficients. ``slope`` is in y-units per x-unit.
    slope_stderr_ols:
        Textbook standard error, valid only for independent residuals.
    slope_stderr_ar1:
        Standard error after the AR(1) inflation described in the module
        docstring. Never smaller than the OLS value when ``rho > 0``.
    residual_autocorrelation:
        Lag-1 autocorrelation ``rho`` of the residuals, clipped into
        ``(-0.999, 0.999)`` before use so the inflation factor stays finite.
    n:
        Number of finite samples the fit used.
    rms_residual:
        Root-mean-square residual, in y-units.
    """

    slope: float
    intercept: float
    slope_stderr_ols: float
    slope_stderr_ar1: float
    residual_autocorrelation: float
    n: int
    rms_residual: float

    @property
    def inflation(self) -> float:
        """Ratio of the AR(1) standard error to the OLS one."""
        if self.slope_stderr_ols == 0.0:
            return math.nan
        return self.slope_stderr_ar1 / self.slope_stderr_ols


def fit_linear(x: np.ndarray, y: np.ndarray) -> LinearFit:
    """Least-squares straight line through the finite pairs of ``x`` and ``y``.

    Parameters
    ----------
    x, y:
        Equal-length arrays. Pairs where either entry is non-finite are dropped.

    Raises
    ------
    ValueError
        If the arrays differ in length, fewer than three finite pairs remain, or
        every retained ``x`` is identical so the slope is not identifiable.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = int(x.size)
    if n < 3:
        raise ValueError(f"a straight-line fit needs at least 3 finite pairs, got {n}")
    span = float(x.max() - x.min())
    if span <= 0.0:
        raise ValueError("every x is identical; the slope is not identifiable")

    centred_x = x - x.mean()
    sxx = float(np.sum(centred_x**2))
    slope = float(np.sum(centred_x * (y - y.mean())) / sxx)
    intercept = float(y.mean() - slope * x.mean())
    residual = y - (intercept + slope * x)
    dof = n - 2
    sigma2 = float(np.sum(residual**2) / dof)
    stderr_ols = math.sqrt(sigma2 / sxx)

    if n >= 4 and np.std(residual) > 0.0:
        lead, lag = residual[1:], residual[:-1]
        denominator = float(np.sum(residual**2))
        rho = float(np.sum(lead * lag) / denominator) if denominator > 0.0 else 0.0
    else:
        rho = 0.0
    rho = float(np.clip(rho, -0.999, 0.999))
    inflation = math.sqrt((1.0 + rho) / (1.0 - rho)) if rho > -1.0 else 1.0
    return LinearFit(
        slope=slope,
        intercept=intercept,
        slope_stderr_ols=stderr_ols,
        slope_stderr_ar1=stderr_ols * inflation,
        residual_autocorrelation=rho,
        n=n,
        rms_residual=float(math.sqrt(np.mean(residual**2))),
    )


@dataclass(frozen=True)
class EquilibriumVerdict:
    """Outcome of the equilibrium test on one run.

    Attributes
    ----------
    label:
        Run label.
    origin:
        Origin of the trace the verdict was computed from.
    status:
        ``"EQUILIBRATED"``, ``"NOT_EQUILIBRATED"`` or ``"INSUFFICIENT_SAMPLES"``.
        Only the first permits averaging.
    slope_c_per_s:
        Fitted die-temperature slope over the window, ``nan`` when no fit was
        possible.
    slope_ci_c_per_s:
        AR(1)-corrected two-sided interval on the slope at :data:`CONFIDENCE`.
    threshold_c_per_s, window_s:
        The criterion applied.
    n_samples, n_missing:
        Finite and non-finite sample counts inside the window.
    mean_die_c:
        Mean die temperature over the window, for reference, ``nan`` if no
        finite sample.
    reason:
        One sentence saying why the status is what it is.
    fit:
        The underlying fit, or ``None`` when none was possible.
    """

    label: str
    origin: Origin
    status: str
    slope_c_per_s: float
    slope_ci_c_per_s: tuple[float, float]
    threshold_c_per_s: float
    window_s: float
    n_samples: int
    n_missing: int
    mean_die_c: float
    reason: str
    fit: LinearFit | None = None

    @property
    def equilibrated(self) -> bool:
        """True only for ``EQUILIBRATED``. ``INSUFFICIENT_SAMPLES`` is not a pass."""
        return self.status == "EQUILIBRATED"

    @property
    def interval_inside_threshold(self) -> bool:
        """Whether the whole slope interval lies inside the threshold band.

        Stricter than :attr:`equilibrated`, which tests the point estimate as
        the specification states. Reported beside it rather than replacing it.
        """
        low, high = self.slope_ci_c_per_s
        if not (math.isfinite(low) and math.isfinite(high)):
            return False
        return abs(low) <= self.threshold_c_per_s and abs(high) <= self.threshold_c_per_s


def _z_for(confidence: float) -> float:
    """Two-sided normal quantile for ``confidence``, via the error function."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    lo, hi = 0.0, 40.0
    target = confidence
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.erf(mid / math.sqrt(2.0)) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def equilibrium_test(
    trace,
    *,
    window_s: float = DEFAULT_WINDOW_S,
    threshold_c_per_s: float = DEFAULT_SLOPE_THRESHOLD_C_PER_S,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    confidence: float = CONFIDENCE,
) -> EquilibriumVerdict:
    """Test whether ``trace`` has reached thermal equilibrium by its final window.

    Parameters
    ----------
    trace:
        A :class:`~thermalbudget.trace.Trace`.
    window_s:
        Length of the final window, seconds.
    threshold_c_per_s:
        Declared slope criterion, degrees Celsius per second.
    min_samples:
        Fewest finite die-temperature samples that may produce a slope.
    confidence:
        Coverage of the reported slope interval.

    Returns
    -------
    EquilibriumVerdict

    Raises
    ------
    ValueError
        On a non-positive window or threshold.
    """
    if window_s <= 0.0:
        raise ValueError(f"window_s must be positive, got {window_s}")
    if threshold_c_per_s <= 0.0:
        raise ValueError(f"threshold_c_per_s must be positive, got {threshold_c_per_s}")

    tail = trace.tail(window_s)
    temperature = tail.die_temperature_c
    time_s = tail.t_rel
    finite = np.isfinite(temperature) & np.isfinite(time_s)
    n_finite = int(np.sum(finite))
    n_missing = int(temperature.size - n_finite)
    mean_die = float(np.mean(temperature[finite])) if n_finite else math.nan

    if n_finite < min_samples:
        return EquilibriumVerdict(
            label=trace.label,
            origin=trace.origin,
            status="INSUFFICIENT_SAMPLES",
            slope_c_per_s=math.nan,
            slope_ci_c_per_s=(math.nan, math.nan),
            threshold_c_per_s=threshold_c_per_s,
            window_s=window_s,
            n_samples=n_finite,
            n_missing=n_missing,
            mean_die_c=mean_die,
            reason=(
                f"{n_finite} finite die-temperature samples in the final {window_s:g} s, "
                f"fewer than the declared minimum of {min_samples}; no slope was computed "
                "and this is not a pass"
            ),
        )

    fit = fit_linear(time_s, temperature)
    half_width = _z_for(confidence) * fit.slope_stderr_ar1
    interval = (fit.slope - half_width, fit.slope + half_width)
    passed = abs(fit.slope) <= threshold_c_per_s
    return EquilibriumVerdict(
        label=trace.label,
        origin=trace.origin,
        status="EQUILIBRATED" if passed else "NOT_EQUILIBRATED",
        slope_c_per_s=fit.slope,
        slope_ci_c_per_s=interval,
        threshold_c_per_s=threshold_c_per_s,
        window_s=window_s,
        n_samples=fit.n,
        n_missing=n_missing,
        mean_die_c=mean_die,
        reason=(
            f"fitted slope {fit.slope:+.6f} degC/s over the final {window_s:g} s "
            f"{'is within' if passed else 'EXCEEDS'} the declared threshold "
            f"{threshold_c_per_s:g} degC/s "
            f"(AR(1) rho {fit.residual_autocorrelation:+.3f}, standard-error inflation "
            f"{fit.inflation:.2f}x)"
        ),
        fit=fit,
    )


def equilibrium_sensitivity(
    trace,
    *,
    window_s: float = DEFAULT_WINDOW_S,
    ladder: tuple[float, ...] = THRESHOLD_LADDER,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> dict[float, str]:
    """Verdict status at each threshold in ``ladder``, so the criterion is visible.

    Returns
    -------
    dict
        Threshold in degrees Celsius per second mapped to the status string.
    """
    return {
        threshold: equilibrium_test(
            trace, window_s=window_s, threshold_c_per_s=threshold, min_samples=min_samples
        ).status
        for threshold in ladder
    }
