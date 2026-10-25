"""Analytic growth models for a single object's divergence curve.

This is the classical baseline and it is implemented before anything learned,
because the specification predicts in writing that an analytic form will match
at most altitudes and because a weak baseline manufactures a learned win. The
forms are not arbitrary -- each is what the dynamics produce under a stated
dominant error term.

Linear in time
    A constant error in mean motion. Along-track position error from a
    semi-major-axis error ``da`` is ``-(3/2) n (da/a) t``: constant rate,
    linear growth. This is the regime above the altitude where drag matters,
    because then nothing is changing ``a``.

Quadratic in time
    A constant error in the *rate* of change of semi-major axis, which is what
    a mis-modelled drag gives: ``da(t) = adot_err t``, so along-track error
    integrates to ``t^2``. This is the regime where drag matters and the
    atmosphere model is the disagreement.

Power law in time
    ``A t^p`` with ``p`` free. Not a separate physical regime but the
    measurement of which of the two above is in force, and the honest form to
    fit when the answer is not known in advance. The fitted ``p`` against
    altitude is the central physical result of this package and is plotted in
    ``screenshots/growth_exponent_vs_altitude.png``.

Drag-dominated exponential in time
    ``A (exp(t/tau) - 1)``. At very low altitude the altitude loss feeds back
    into the density, so the error growth is faster than any power law over a
    long enough span. The form is written with the ``- 1`` so that it is zero
    at zero age, as the divergence curve is by construction.

Fitting
-------
All four are fitted by linear least squares **in log space**: the residual of
a divergence curve is multiplicative, not additive -- a 10 per cent error at
20 km and a 10 per cent error at 2 km are the same modelling error -- and a
linear-space fit is dominated by the last few points of a curve that spans two
decades. The consequence is stated plainly: these fits minimise relative
error, so none of them is the maximum-likelihood fit under additive Gaussian
noise, and the reported residual spread is a spread of log ratios.

Rank deficiency
---------------
:func:`fit_power_law` refuses a design matrix whose age column has no spread,
rather than letting ``numpy.linalg.lstsq`` return a least-norm solution with a
zero exponent and a zero uncertainty. That failure mode is the reason the
refusal is explicit: a pseudo-inverse of a rank-deficient matrix returns a
confident-looking answer of exactly zero, and Hypothesis found exactly that
shape of defect elsewhere in this mission by generating repeated abscissae.
``tests/test_growth.py`` has the regression test with literal inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .errors import InsufficientDataError

_MIN_DRIFT_KM = 1e-9
"""Drift values at or below this are dropped before taking a logarithm.

A divergence curve can contain an exact zero at zero age by construction, and
a value below a nanometre is below every numerical floor this package
measures. Dropping rather than clipping is deliberate: clipping would invent a
data point at the floor and bias the fitted exponent.
"""


@dataclass(frozen=True)
class GrowthFit:
    """A fitted growth model and the spread of its own residuals.

    Attributes
    ----------
    form
        ``"linear"``, ``"quadratic"``, ``"power"`` or ``"drag_exponential"``.
    amplitude_km
        The coefficient ``A`` in km, at an age of one day for the power forms.
    exponent
        ``p`` for the power forms; ``1.0`` and ``2.0`` for linear and
        quadratic; ``nan`` for the exponential form.
    timescale_days
        ``tau`` for the exponential form; ``nan`` otherwise.
    exponent_stderr
        Standard error of ``p`` from the least-squares parameter covariance,
        or ``nan`` where ``p`` was not a free parameter.
    log_residual_std
        Standard deviation of ``log(observed / predicted)`` over the fitted
        points. This is a spread of *ratios*: 0.1 means about 10 per cent.
    n_points
        Number of points the fit actually used after the floor was applied.
    """

    form: str
    amplitude_km: float
    exponent: float
    timescale_days: float
    exponent_stderr: float
    log_residual_std: float
    n_points: int

    def predict_km(self, ages_days: np.ndarray | float) -> np.ndarray:
        """Predicted drift, km, at ``ages_days``."""
        t = np.asarray(ages_days, dtype=float)
        if self.form == "drag_exponential":
            return self.amplitude_km * (np.expm1(t / self.timescale_days))
        return self.amplitude_km * t**self.exponent

    def interval_km(
        self, ages_days: np.ndarray | float, *, coverage: float = 0.90
    ) -> tuple[np.ndarray, np.ndarray]:
        """Multiplicative interval from the fitted log-residual spread.

        The interval is ``predict * exp(+/- z sigma)`` with ``z`` the normal
        quantile for ``coverage``. It is a **nominal** interval: it assumes the
        log residuals are Gaussian and that the spread measured on the fitted
        points carries over to the ages being predicted, and neither assumption
        is checked here. Whether it covers is measured, not assumed, and the
        measurement is :mod:`orbitdrift.calibration` applied to held-out
        objects. The naive coverage of this interval is in the coverage table
        of ``validation/validate_coverage_table.py``, where it is the row that
        shows what assuming costs.
        """
        if not 0.0 < coverage < 1.0:
            raise ValueError(f"coverage must be in (0, 1); got {coverage}")
        from scipy.stats import norm

        z = float(norm.ppf(0.5 + 0.5 * coverage))
        centre = self.predict_km(ages_days)
        spread = np.exp(z * self.log_residual_std)
        return centre / spread, centre * spread


def _clean(ages_days: np.ndarray, drift_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(ages_days, dtype=float).ravel()
    y = np.asarray(drift_km, dtype=float).ravel()
    if t.shape != y.shape:
        raise InsufficientDataError(
            f"ages ({t.shape}) and drift ({y.shape}) must have the same shape"
        )
    keep = np.isfinite(t) & np.isfinite(y) & (t > 0.0) & (y > _MIN_DRIFT_KM)
    return t[keep], y[keep]


def _log_residual_std(y: np.ndarray, predicted: np.ndarray, n_parameters: int) -> float:
    residual = np.log(y) - np.log(predicted)
    dof = max(1, residual.size - n_parameters)
    return float(np.sqrt(np.sum(residual**2) / dof))


def fit_power_law(
    ages_days: np.ndarray, drift_km: np.ndarray, *, exponent: float | None = None
) -> GrowthFit:
    """Fit ``A t^p`` in log space; ``exponent`` fixes ``p`` instead of fitting it.

    With ``exponent=1.0`` this is the linear-in-time form and with
    ``exponent=2.0`` the quadratic-in-time form, each fitted with one free
    parameter.

    Raises
    ------
    InsufficientDataError
        With fewer than two usable points; or, when ``p`` is free, when every
        usable age is the same value, which makes the design matrix
        rank-deficient. ``numpy.linalg.lstsq`` would return an exponent of
        exactly zero with a standard error of exactly zero for that input,
        which looks like a measurement and is not one.
    """
    t, y = _clean(ages_days, drift_km)
    if t.size < 2:
        raise InsufficientDataError(
            f"need at least 2 points with a positive age and a drift above "
            f"{_MIN_DRIFT_KM} km; got {t.size}"
        )
    log_t = np.log(t)
    log_y = np.log(y)

    if exponent is not None:
        p = float(exponent)
        log_amplitude = float(np.mean(log_y - p * log_t))
        fit = GrowthFit(
            form={1.0: "linear", 2.0: "quadratic"}.get(p, "power"),
            amplitude_km=float(np.exp(log_amplitude)),
            exponent=p,
            timescale_days=float("nan"),
            exponent_stderr=float("nan"),
            log_residual_std=0.0,
            n_points=int(t.size),
        )
        return GrowthFit(
            fit.form,
            fit.amplitude_km,
            p,
            float("nan"),
            float("nan"),
            _log_residual_std(y, fit.predict_km(t), 1),
            int(t.size),
        )

    spread = float(np.ptp(log_t))
    if spread <= 1e-12 or t.size < 3:
        raise InsufficientDataError(
            f"a free exponent needs at least 3 points at 2 or more distinct ages; "
            f"got {t.size} points spanning {spread:.3e} in log age. Refusing rather "
            "than returning the least-norm solution of a rank-deficient system, "
            "which is an exponent of exactly zero with an uncertainty of exactly "
            "zero and is not a measurement."
        )
    design = np.column_stack([np.ones_like(log_t), log_t])
    solution, *_ = np.linalg.lstsq(design, log_y, rcond=None)
    log_amplitude, p = float(solution[0]), float(solution[1])
    residual = log_y - design @ solution
    dof = max(1, log_y.size - 2)
    variance = float(np.sum(residual**2) / dof)
    covariance = variance * np.linalg.inv(design.T @ design)
    return GrowthFit(
        form="power",
        amplitude_km=float(np.exp(log_amplitude)),
        exponent=p,
        timescale_days=float("nan"),
        exponent_stderr=float(np.sqrt(covariance[1, 1])),
        log_residual_std=float(np.sqrt(variance)),
        n_points=int(t.size),
    )


def fit_linear(ages_days: np.ndarray, drift_km: np.ndarray) -> GrowthFit:
    """Fit ``A t`` -- the constant-mean-motion-error form."""
    return fit_power_law(ages_days, drift_km, exponent=1.0)


def fit_quadratic(ages_days: np.ndarray, drift_km: np.ndarray) -> GrowthFit:
    """Fit ``A t^2`` -- the constant-drag-rate-error form."""
    return fit_power_law(ages_days, drift_km, exponent=2.0)


def fit_drag_exponential(
    ages_days: np.ndarray,
    drift_km: np.ndarray,
    *,
    timescale_bounds_days: tuple[float, float] = (0.05, 2000.0),
) -> GrowthFit:
    """Fit ``A (exp(t/tau) - 1)`` in log space, with ``tau`` found by a 1-D search.

    For a fixed ``tau`` the model is linear in ``log A``, so the profile
    objective is a function of ``tau`` alone. Its derivative is monotone in the
    regime of interest and :func:`scipy.optimize.brentq` is used on that
    derivative; if the derivative does not change sign across the bounds, the
    boundary with the lower objective is returned and the fit is flagged by a
    ``timescale_days`` equal to that boundary. A ``tau`` at the upper bound
    means the data do not distinguish this form from the linear one, because
    ``exp(t/tau) - 1 -> t/tau`` as ``tau`` grows, and
    ``validation/validate_growth_baseline.py`` reports how often that happens.
    """
    t, y = _clean(ages_days, drift_km)
    if t.size < 3:
        raise InsufficientDataError(
            f"the exponential form has two parameters and needs at least 3 points; got {t.size}"
        )
    log_y = np.log(y)

    def objective(tau: float) -> float:
        basis = np.log(np.expm1(t / tau))
        return float(np.var(log_y - basis))

    def derivative(tau: float) -> float:
        step = 1e-4 * tau
        return (objective(tau + step) - objective(tau - step)) / (2.0 * step)

    low, high = timescale_bounds_days
    try:
        if derivative(low) * derivative(high) < 0.0:
            tau = float(brentq(derivative, low, high, xtol=1e-6, rtol=1e-8))
        else:
            tau = low if objective(low) <= objective(high) else high
    except (ValueError, OverflowError, FloatingPointError):
        tau = high
    basis = np.log(np.expm1(t / tau))
    log_amplitude = float(np.mean(log_y - basis))
    amplitude = float(np.exp(log_amplitude))
    fit = GrowthFit(
        "drag_exponential", amplitude, float("nan"), tau, float("nan"), 0.0, int(t.size)
    )
    return GrowthFit(
        "drag_exponential",
        amplitude,
        float("nan"),
        tau,
        float("nan"),
        _log_residual_std(y, fit.predict_km(t), 2),
        int(t.size),
    )


FORMS = {
    "linear": fit_linear,
    "quadratic": fit_quadratic,
    "power": fit_power_law,
    "drag_exponential": fit_drag_exponential,
}
"""Fit function for each analytic form, by name."""


def fit_all_forms(ages_days: np.ndarray, drift_km: np.ndarray) -> dict[str, GrowthFit]:
    """Fit every form in :data:`FORMS` and return them keyed by name.

    A form that cannot be fitted is omitted rather than returned with filler
    values, so a caller iterating the result never sees a fabricated fit.
    """
    out: dict[str, GrowthFit] = {}
    for name, function in FORMS.items():
        try:
            out[name] = function(ages_days, drift_km)
        except (InsufficientDataError, ValueError, FloatingPointError):
            continue
    return out


def best_form(fits: dict[str, GrowthFit]) -> str:
    """Name of the form with the smallest log-residual spread.

    Ties are broken by the order of :data:`FORMS`, which puts the forms with
    fewer free parameters first, so a tie goes to the simpler model. No
    information criterion is applied: the forms have one or two parameters over
    the same points and the spread is reported alongside, so a reader can see
    how little separates them when little does.
    """
    if not fits:
        raise InsufficientDataError("no form could be fitted")
    return min(fits, key=lambda name: (fits[name].log_residual_std, list(FORMS).index(name)))
