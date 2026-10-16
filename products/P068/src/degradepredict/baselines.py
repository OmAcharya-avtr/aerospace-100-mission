"""Analytic degradation baselines with prediction intervals on the crossing time.

These are written and validated before any learned model exists, and they are
what the learned forecaster has to beat.

The common structure
--------------------
All three shipped baselines are ordinary least squares of a transformed metric
on a transformed time::

    g(y_j) = a + b * h(t_j) + e_j,    e_j ~ N(0, sigma**2) i.i.d.

with ``g`` the metric link and ``h(t) = t**z`` the time link:

====================  ==========  =======  ===========================================
baseline              g           z        mean shape it is exact for
====================  ==========  =======  ===========================================
``linear``            identity    1        ``y = y0 - b t``
``exponential``       natural log 1        ``y = y0 exp(-k t)``
``powerlaw-fade``     identity    declared ``y = y0 (1 - B t**z)``
====================  ==========  =======  ===========================================

The crossing time is then the value of ``t`` at which the fitted line reaches
``g(limit)``, which is an *inverse regression* (calibration) problem, not a
forward prediction. Two intervals are computed for it:

**Delta method.** With ``w = t**z``, the solution is
``w* = xbar + (g(L) - ubar) / b`` and
``var(w*) = (sigma**2 / b**2) * (1/n + (w* - xbar)**2 / Sxx)``.
This is the standard inverse-regression variance; see Draper, N. R. and Smith,
H. (1998), *Applied Regression Analysis*, 3rd ed., Wiley, section 3.2.

**Fieller's exact set.** The set of ``w`` whose fitted mean response is within
``t_crit`` standard errors of ``g(L)`` is the solution of a quadratic in
``u = w - xbar``::

    (b**2 - t_crit**2 sigma**2 / Sxx) u**2
        - 2 b (g(L) - ubar) u
        + ((g(L) - ubar)**2 - t_crit**2 sigma**2 / n)  <=  0

Fieller, E. C. (1954), "Some problems in interval estimation", Journal of the
Royal Statistical Society Series B 16(2):175-185. The leading coefficient is
``b**2 (1 - g)`` with ``g = t_crit**2 sigma**2 / (Sxx b**2)``; when ``g >= 1``
the slope is not significantly different from zero at that level and the exact
set is **not a bounded interval** -- it is a half line or the whole line. The
delta-method interval is the ``g -> 0`` limit of the Fieller set and is always
bounded, so it silently understates the uncertainty exactly when the data are
least informative. ``validation/validate_baselines.py`` measures the agreement
as a function of ``g``.

Units and assumptions
---------------------
Times are days of service age; the metric is whatever the caller supplies
(percent of beginning-of-life capacity in the shipped generator). The error
model assumes additive homoscedastic independent Gaussian residuals *on the
transformed scale*. For ``exponential`` that means multiplicative lognormal
noise on the metric, which the shipped generator does **not** produce -- the
generator adds Gaussian noise on the raw scale. That mismatch is deliberate and
declared: it is the ordinary situation in which a practitioner log-transforms to
get a linear fit.

Validity range
--------------
Extrapolation from an observation window of length ``t0`` to a crossing at
``T >> t0`` is extrapolation, and the interval produced here only covers
*parameter* uncertainty under the assumed shape. It carries no term for the
shape being wrong. When the shape is wrong the interval is too narrow, which is
visible in the measured coverage and is the main finding of this repository.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .forecast import TransformedNormalForecast
from .limits import LimitSpec

__all__ = [
    "BaselineFit",
    "DegradationFitError",
    "exponential_baseline",
    "fit_transformed_ols",
    "linear_baseline",
]


class DegradationFitError(ValueError):
    """Raised when a baseline cannot name a crossing time from the window given.

    The two causes are a fitted slope that is not in the degrading direction
    (the window shows no decline, so the model has no crossing to report) and a
    window too short or too degenerate to estimate a residual variance. Callers
    that must score every unit have to declare a fallback and count how often it
    was used; see :func:`degradepredict.benchmark.evaluate_model`.
    """


@dataclass(frozen=True)
class BaselineFit:
    """Result of one analytic baseline fit.

    Attributes
    ----------
    model:
        Label.
    intercept, slope:
        OLS coefficients on the transformed scale, ``g(y) = a + b * t**z``.
    sigma:
        Residual standard deviation on the transformed scale, ``sqrt(RSS/(n-2))``.
    n_samples:
        Number of observations used.
    z:
        Time-link exponent, dimensionless.
    metric_link:
        ``"identity"`` or ``"log"``; says which scale ``intercept``, ``slope``
        and ``sigma`` live on.
    crossing_days:
        Point estimate of the crossing time, days of service age.
    se_w:
        Delta-method standard error of ``w = T**z``, units ``day**z``.
    fieller_g:
        Fieller's ``g`` at ``level``. ``g >= 1`` means the exact set is unbounded.
    fieller_interval_days:
        Exact Fieller interval on the crossing time at ``level``, days. Entries
        may be ``-inf`` or ``+inf``.
    delta_interval_days:
        Delta-method interval on the crossing time at ``level``, days.
    level:
        Nominal two-sided confidence level the intervals were computed at.
    forecast:
        Predictive distribution over *remaining* life from the forecast origin.
    """

    model: str
    intercept: float
    slope: float
    sigma: float
    n_samples: int
    z: float
    metric_link: str
    crossing_days: float
    se_w: float
    fieller_g: float
    fieller_interval_days: tuple[float, float]
    delta_interval_days: tuple[float, float]
    level: float
    forecast: TransformedNormalForecast


def _validate_window(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.ndim != 1 or y.ndim != 1:
        raise ValueError("times and values must be one-dimensional")
    if t.shape != y.shape:
        raise ValueError(f"shape mismatch: times {t.shape} values {y.shape}")
    if t.size < 3:
        raise ValueError(
            f"need at least three samples to estimate a residual variance, got {t.size}"
        )
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(y))):
        raise ValueError("times and values must all be finite")
    if np.any(t < 0):
        raise ValueError("times must be non-negative service ages")
    return t, y


def fit_transformed_ols(
    times_days: np.ndarray,
    values: np.ndarray,
    limit: LimitSpec,
    *,
    z: float = 1.0,
    log_metric: bool = False,
    level: float = 0.90,
    model: str = "transformed-ols",
    origin_days: float | None = None,
) -> BaselineFit:
    """Fit ``g(y) = a + b t**z`` and invert it for the crossing of ``limit``.

    Parameters
    ----------
    times_days:
        Sample service ages in days, length >= 3, non-negative.
    values:
        Metric values at those times, same units as ``limit``.
    limit:
        The declared limit. Only ``direction="below"`` is supported here, which
        is the downward-degrading case the shipped generator produces.
    z:
        Time-link exponent, dimensionless, strictly positive.
    log_metric:
        If true the metric is log-transformed before fitting, which makes the
        model exponential. Requires strictly positive values and a strictly
        positive limit.
    level:
        Two-sided nominal confidence level for the Fieller and delta intervals.
    origin_days:
        Forecast origin. Defaults to the last sample time. Remaining life is
        measured forward from here.

    Returns
    -------
    BaselineFit

    Raises
    ------
    ValueError
        On malformed input, non-positive values under ``log_metric``, or a
        non-positive ``z``.
    DegradationFitError
        If the fitted slope is not strictly negative on the transformed scale,
        or the residual variance is zero so no interval exists.
    """
    t, y = _validate_window(times_days, values)
    if limit.direction != "below":
        raise ValueError("fit_transformed_ols handles downward-degrading limits only")
    if not np.isfinite(z) or z <= 0:
        raise ValueError(f"z must be finite and positive, got {z!r}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level!r}")

    if log_metric:
        if np.any(y <= 0) or limit.value <= 0:
            raise ValueError("log_metric requires strictly positive values and limit")
        u = np.log(y)
        u_limit = float(np.log(limit.value))
    else:
        u = y
        u_limit = float(limit.value)

    x = np.power(t, z)
    n = int(t.size)
    xbar = float(x.mean())
    ubar = float(u.mean())
    sxx = float(np.sum((x - xbar) ** 2))
    if sxx <= 0:
        raise DegradationFitError("all sample times coincide after the time transform")
    b = float(np.sum((x - xbar) * (u - ubar)) / sxx)
    a = ubar - b * xbar
    resid = u - (a + b * x)
    s2 = float(np.sum(resid**2) / (n - 2))
    if not np.isfinite(s2) or s2 <= 0:
        raise DegradationFitError("zero residual variance: no interval is defined")
    if b >= 0:
        raise DegradationFitError(
            f"fitted slope {b:.6g} is not negative on the transformed scale; this window "
            f"shows no decline and the model has no crossing to report"
        )

    w_star = xbar + (u_limit - ubar) / b
    if w_star <= 0:
        raise DegradationFitError(
            f"the fitted line reaches the limit at a non-positive transformed time "
            f"{w_star:.6g}; the window already sits at or below the limit"
        )
    var_w = (s2 / b**2) * (1.0 / n + (w_star - xbar) ** 2 / sxx)
    se_w = float(np.sqrt(var_w))
    crossing = float(w_star ** (1.0 / z))

    t_crit = float(stats.t.ppf(0.5 * (1.0 + level), df=n - 2))
    g_fieller = float(t_crit**2 * s2 / (sxx * b**2))
    lo_w, hi_w = _fieller_set(b, u_limit - ubar, s2, sxx, n, t_crit, xbar)
    fieller_days = (_to_days(lo_w, z, default=-np.inf), _to_days(hi_w, z, default=np.inf))
    delta_lo_w = w_star - t_crit * se_w
    delta_hi_w = w_star + t_crit * se_w
    delta_days = (
        float(max(delta_lo_w, 0.0) ** (1.0 / z)),
        float(max(delta_hi_w, 0.0) ** (1.0 / z)),
    )

    origin = float(t[-1]) if origin_days is None else float(origin_days)
    if crossing <= origin:
        raise DegradationFitError(
            f"the fitted crossing {crossing:.6g} d is not after the forecast origin "
            f"{origin:.6g} d; the window already indicates a crossing"
        )
    forecast = TransformedNormalForecast(
        mu_w=float(w_star),
        sd_w=se_w,
        z=float(z),
        origin_days=origin,
        fieller_g=g_fieller,
        model=model,
    )
    return BaselineFit(
        model=model,
        intercept=float(a),
        slope=b,
        sigma=float(np.sqrt(s2)),
        n_samples=n,
        z=float(z),
        metric_link="log" if log_metric else "identity",
        crossing_days=crossing,
        se_w=se_w,
        fieller_g=g_fieller,
        fieller_interval_days=fieller_days,
        delta_interval_days=delta_days,
        level=float(level),
        forecast=forecast,
    )


def _to_days(w: float, z: float, default: float) -> float:
    """Map a transformed-time endpoint back to days, or return the declared default.

    An endpoint of ``-inf`` or ``+inf`` is an unbounded Fieller endpoint and maps
    to ``default``. A finite non-positive endpoint means the confidence set
    reaches back past service age zero, which is reported as 0.0 days rather
    than as a complex root.
    """
    if not np.isfinite(w):
        return float(default)
    if w <= 0.0:
        return 0.0
    return float(w ** (1.0 / z))


def _fieller_set(
    b: float, d: float, s2: float, sxx: float, n: int, t_crit: float, xbar: float
) -> tuple[float, float]:
    """Fieller confidence set for the transformed crossing time ``w``.

    ``d = g(L) - ubar``. Solves ``A u**2 - 2 b d u + (d**2 - t**2 s2 / n) <= 0``
    for ``u = w - xbar`` with ``A = b**2 - t**2 s2 / sxx``, then returns
    ``xbar + u`` for both roots. Returns ``(-inf, inf)`` when ``A <= 0`` or the
    discriminant is negative, which is the unbounded case ``g >= 1``.
    """
    a_coef = b**2 - t_crit**2 * s2 / sxx
    b_coef = -2.0 * b * d
    c_coef = d**2 - t_crit**2 * s2 / n
    disc = b_coef**2 - 4.0 * a_coef * c_coef
    if a_coef > 0 and disc >= 0:
        root = np.sqrt(disc)
        u_lo = (-b_coef - root) / (2.0 * a_coef)
        u_hi = (-b_coef + root) / (2.0 * a_coef)
        return xbar + min(u_lo, u_hi), xbar + max(u_lo, u_hi)
    return -np.inf, np.inf


def linear_baseline(
    times_days: np.ndarray,
    values: np.ndarray,
    limit: LimitSpec,
    *,
    level: float = 0.90,
    origin_days: float | None = None,
) -> BaselineFit:
    """Straight-line fade fit: ``y = y0 - b t``, ``b`` in metric units per day.

    Assumptions: the fade is linear over the extrapolation range and the
    residuals are additive Gaussian. Validity: a first-order expansion, so it is
    only credible when the extrapolation distance is small compared with the
    curvature scale of the true fade. It has no mechanism behind it.
    """
    return fit_transformed_ols(
        times_days, values, limit, z=1.0, log_metric=False, level=level,
        model="linear", origin_days=origin_days,
    )


def exponential_baseline(
    times_days: np.ndarray,
    values: np.ndarray,
    limit: LimitSpec,
    *,
    level: float = 0.90,
    origin_days: float | None = None,
) -> BaselineFit:
    """Exponential fade fit: ``y = y0 exp(-k t)``, ``k`` in 1/day.

    Fitted by ordinary least squares of ``log y`` on ``t``, so the implied noise
    model is multiplicative lognormal on the metric. Requires strictly positive
    metric values and a strictly positive limit. Assumptions: a loss rate
    proportional to the remaining metric, and constant ``k``. Validity: no
    mechanism is claimed; this is a declared alternative shape.
    """
    return fit_transformed_ols(
        times_days, values, limit, z=1.0, log_metric=True, level=level,
        model="exponential", origin_days=origin_days,
    )
