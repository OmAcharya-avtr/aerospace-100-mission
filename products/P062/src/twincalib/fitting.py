"""Weighted nonlinear least squares, which for this noise model is maximum
likelihood.

The baseline estimator of the package: fit the physics model alone, by itself,
with no learned component anywhere near it. It is implemented and measured
before the residual model exists, and the residual model is scored against it.

The estimator
-------------
For independent Gaussian errors of known standard deviation ``sigma_i`` the
negative log-likelihood of a parameter vector is, up to a constant that does
not depend on the parameters,

``-ln L(theta) = 0.5 * sum_i ((y_i - f_i(theta)) / sigma_i)**2 = 0.5 * chi2``

so minimising the weighted sum of squares *is* maximum likelihood (Bard 1974,
section 5-4; Seber and Wild 1989, section 2.1). The minimisation is done over
``ln theta`` rather than ``theta``:

* every parameter of both models is a physically positive quantity, and the
  logarithm enforces that without a bound constraint;
* the parameters span 11 orders of magnitude, and in log coordinates the
  trust-region step is scale free.

``scipy.optimize.least_squares`` with the default trust-region-reflective
algorithm does the minimisation.

The covariance
--------------
The Gauss-Newton covariance of the estimate in log-parameter coordinates is
``(J_w^T J_w)^-1`` with ``J_w`` the weighted log-sensitivity matrix at the
optimum (Bard 1974, eq. 7-7-5). It is the inverse Fisher information for this
noise model, so it is the Cramer-Rao bound as well as the asymptotic covariance,
and the square roots of its diagonal are directly *relative* standard
deviations. Two things about it are said out loud wherever it is reported:

1. It is a **local quadratic approximation**. It is exactly the curvature at the
   optimum and nothing else. Where the likelihood is not quadratic it is wrong,
   and :mod:`twincalib.profile` exists to measure by how much.
2. It assumes the model is right. The reduced chi-squared returned by
   :func:`fit` is the check on that assumption; a value far above 1 with a
   correct ``sigma`` means model error, not parameter error, and no covariance
   matrix will tell you which parameter is to blame.

References
----------
Bard, Y., *Nonlinear Parameter Estimation*, Academic Press, 1974.

Seber, G. A. F. and Wild, C. J., *Nonlinear Regression*, Wiley, 1989. Chapter 2
on the weighted least-squares estimator and Chapter 5 on its covariance.

Branch, M. A., Coleman, T. F. and Li, Y. (1999), "A subspace, interior, and
conjugate gradient method for large-scale bound-constrained minimization
problems", *SIAM Journal on Scientific Computing* 21(1), 1-23. The
trust-region-reflective algorithm behind ``scipy.optimize.least_squares``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .identifiability import null_direction_loading
from .model import Inputs, TwinModel
from .sensitivity import DEFAULT_LOG_STEP, SensitivityMatrix, log_sensitivity

__all__ = [
    "DIVERGENCE_PENALTY",
    "CalibrationProblem",
    "FitResult",
    "GuardedResidual",
    "chi_square",
    "fit",
    "residual_vector",
]

DIVERGENCE_PENALTY: float = 1.0e6
"""Weighted residual returned for a parameter vector the integrator rejects.

A fixed-step explicit scheme diverges when a trial parameter vector makes the
fastest time constant short against the design's step. Raising from inside the
residual function would abort the whole fit, so instead every component of the
weighted residual is set to this value: a finite, enormous cost that the
trust-region step walks away from. Each such evaluation is counted and the count
is reported in :class:`FitResult`, because a fit that spent most of its
evaluations in the rejected region is not a fit anyone should trust.
"""


class GuardedResidual:
    """Weighted residuals in log-parameter coordinates, with a divergence guard.

    Calling the instance with ``x = ln theta`` returns
    ``(y - f(exp(x))) / sigma``, dimensionless, or a vector filled with
    :data:`DIVERGENCE_PENALTY` if the model refuses that parameter vector.
    ``n_rejected`` counts how often that happened.
    """

    def __init__(self, problem: CalibrationProblem) -> None:
        self.problem = problem
        self.n_rejected = 0
        self._penalty = np.full(problem.n_observations, DIVERGENCE_PENALTY)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        try:
            return residual_vector(self.problem, np.exp(np.asarray(x, dtype=float)))
        except ValueError:
            self.n_rejected += 1
            return self._penalty


@dataclass(frozen=True)
class CalibrationProblem:
    """Everything the estimator needs: a model, a design and some data.

    Attributes
    ----------
    model:
        The calibration model. Its parameters are what is estimated.
    inputs:
        Exogenous inputs on the integration grid.
    initial_state:
        Initial state, in the model's state units. Treated as known.
    channels:
        Measured state indices.
    sample_stride:
        Observation decimation of the integration grid.
    y:
        Observations, K, shape ``(n_obs,)``, channel-major.
    sigma:
        Noise standard deviation, K, scalar or shape ``(n_obs,)``.
    offset:
        Optional additive correction to the model prediction, K, shape
        ``(n_obs,)``. This is the hook the residual model plugs into: fitting
        the physics with ``offset = -g(x)`` for a learned discrepancy ``g``
        is what :mod:`twincalib.absorption` does to measure how much the
        physics parameters move when a residual model is present.
    """

    model: TwinModel
    inputs: Inputs
    initial_state: tuple[float, ...]
    channels: tuple[int, ...]
    sample_stride: int
    y: np.ndarray
    sigma: np.ndarray
    offset: np.ndarray | None = None

    def __post_init__(self) -> None:
        y = np.asarray(self.y, dtype=float).ravel()
        if y.size == 0:
            raise ValueError("y must not be empty")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must be finite")
        sig = np.broadcast_to(np.asarray(self.sigma, dtype=float), y.shape).astype(float)
        if np.any(sig <= 0.0) or not np.all(np.isfinite(sig)):
            raise ValueError("sigma must be finite and strictly positive")
        if self.offset is not None:
            off = np.asarray(self.offset, dtype=float).ravel()
            if off.shape != y.shape:
                raise ValueError(f"offset must have shape {y.shape}, got {off.shape}")
            object.__setattr__(self, "offset", off)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "sigma", sig)
        object.__setattr__(self, "channels", tuple(int(c) for c in self.channels))
        object.__setattr__(self, "initial_state", tuple(float(x) for x in self.initial_state))

    @property
    def n_observations(self) -> int:
        """Number of scalar observations, dimensionless."""
        return int(self.y.size)

    @property
    def n_parameters(self) -> int:
        """Number of free parameters, dimensionless."""
        return int(self.model.n_parameters)

    @property
    def degrees_of_freedom(self) -> int:
        """``n_obs - n_par``, dimensionless. Negative is rejected by :func:`fit`."""
        return self.n_observations - self.n_parameters

    def predict(self, theta: Sequence[float]) -> np.ndarray:
        """Model prediction plus any offset, K, shape ``(n_obs,)``."""
        pred = self.model.observe(
            theta, self.inputs, self.initial_state, self.channels, self.sample_stride
        )
        return pred if self.offset is None else pred + self.offset


def residual_vector(problem: CalibrationProblem, theta: Sequence[float]) -> np.ndarray:
    """Weighted residuals ``(y - f(theta)) / sigma``, dimensionless."""
    return (problem.y - problem.predict(theta)) / problem.sigma


def chi_square(problem: CalibrationProblem, theta: Sequence[float]) -> float:
    """``chi2 = sum(((y - f) / sigma)**2)``, dimensionless."""
    r = residual_vector(problem, theta)
    return float(r @ r)


@dataclass(frozen=True)
class FitResult:
    """The outcome of one weighted least-squares fit.

    Attributes
    ----------
    theta:
        Estimate, in the model's parameter units.
    parameter_names:
        Names in the order of ``theta``.
    chi2:
        Weighted sum of squares at the optimum, dimensionless.
    reduced_chi2:
        ``chi2 / (n_obs - n_par)``, dimensionless. With a correct ``sigma`` and
        a correct model this is about 1; values well above 1 mean the model
        cannot reproduce the data to within the stated noise, which is model
        error and not parameter error.
    neg_log_likelihood:
        ``0.5 * chi2``, dimensionless: the negative log-likelihood up to an
        additive constant independent of the parameters.
    covariance_log:
        Gauss-Newton covariance in log-parameter coordinates, shape ``(p, p)``,
        dimensionless.
    relative_sd:
        ``sqrt(diag(covariance_log))``, dimensionless: a one-sigma *relative*
        standard deviation per parameter under the quadratic approximation.
    correlation:
        Parameter correlation matrix, shape ``(p, p)``, dimensionless.
    sensitivity:
        The sensitivity matrix at the optimum, reused by the identifiability and
        absorption analyses so that they see exactly the same Jacobian.
    rmse:
        Root-mean-square residual in K (unweighted), a number a thermal engineer
        can read directly.
    success:
        Whether the optimiser reported convergence.
    status, message:
        The optimiser's own status code and message, passed through unaltered.
    n_function_evaluations:
        Number of residual evaluations, each one a full simulation of the record.
    covariance_singular:
        True when the Gauss-Newton matrix had to be pseudo-inverted. The
        covariance and correlation are then not trustworthy and
        :meth:`caveats` says so.
    n_rejected_evaluations:
        How many trial parameter vectors the integrator refused, see
        :data:`DIVERGENCE_PENALTY`.
    """

    theta: np.ndarray
    parameter_names: tuple[str, ...]
    chi2: float
    reduced_chi2: float
    neg_log_likelihood: float
    covariance_log: np.ndarray
    relative_sd: np.ndarray
    correlation: np.ndarray
    sensitivity: SensitivityMatrix
    rmse: float
    success: bool
    status: int
    message: str
    n_function_evaluations: int
    covariance_singular: bool
    n_rejected_evaluations: int = 0

    def relative_error(self, theta_true: Sequence[float]) -> np.ndarray:
        """Signed relative error against a known truth, dimensionless.

        ``(theta_hat - theta_true) / theta_true``. Only meaningful for synthetic
        data; a real calibration has no truth to compare with.
        """
        t = np.asarray(theta_true, dtype=float).ravel()
        if t.shape != self.theta.shape:
            raise ValueError(f"theta_true must have shape {self.theta.shape}")
        if np.any(t == 0.0):
            raise ValueError("theta_true must not contain zeros")
        return (self.theta - t) / t

    def z_score(self, theta_true: Sequence[float]) -> np.ndarray:
        """Error in units of the quadratic-approximation standard deviation.

        ``ln(theta_hat/theta_true) / relative_sd``, dimensionless. A value much
        larger than 2 means either an unlucky draw, a biased estimator or a
        covariance that understates the uncertainty; in this package the third
        is the usual explanation and the profile intervals are the check.
        """
        t = np.asarray(theta_true, dtype=float).ravel()
        if t.shape != self.theta.shape:
            raise ValueError(f"theta_true must have shape {self.theta.shape}")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(self.theta / t) / self.relative_sd

    def caveats(self) -> list[str]:
        """Warnings that must travel with any reported parameter value."""
        out: list[str] = []
        if not self.success:
            out.append(f"the optimiser did not report convergence: {self.message}")
        if self.n_rejected_evaluations:
            out.append(
                f"{self.n_rejected_evaluations} trial parameter vectors were "
                "refused by the integrator and scored with the divergence "
                "penalty; the search region reaches the scheme's stability limit"
            )
        if self.covariance_singular:
            out.append(
                "the Gauss-Newton matrix is singular: the covariance below is a "
                "pseudo-inverse and the standard deviations in the deficient "
                "directions are not finite, whatever the table prints"
            )
        if self.reduced_chi2 > 2.0:
            out.append(
                f"reduced chi-squared is {self.reduced_chi2:.3f}, so the model "
                "does not reproduce the data to within the stated measurement "
                "noise; this is model error and no parameter covariance "
                "accounts for it"
            )
        worst = float(np.max(self.relative_sd)) if self.relative_sd.size else 0.0
        if worst > 0.5:
            name = self.parameter_names[int(np.argmax(self.relative_sd))]
            out.append(
                f"the worst relative standard deviation is {worst:.3g} on "
                f"{name}, i.e. above 50 per cent; do not report that parameter "
                "as determined"
            )
        return out


def fit(
    problem: CalibrationProblem,
    theta0: Sequence[float],
    log_step: float = DEFAULT_LOG_STEP,
    xtol: float = 1e-10,
    ftol: float = 1e-12,
    max_nfev: int | None = 400,
    bounds_factor: float = 1e4,
) -> FitResult:
    """Fit the model to the data by weighted nonlinear least squares.

    Parameters
    ----------
    problem:
        Model, design and data.
    theta0:
        Starting parameters, in model units. All entries must be positive.
    log_step:
        Central-difference step used for the covariance at the optimum. The
        optimiser's own internal Jacobian uses SciPy's default ``2-point``
        scheme; the reported covariance is recomputed here with
        :func:`~twincalib.sensitivity.log_sensitivity` so that it is the same
        matrix the identifiability analysis uses.
    xtol, ftol, max_nfev:
        Passed to ``scipy.optimize.least_squares``.
    bounds_factor:
        The search is bounded to ``theta0 / bounds_factor`` to
        ``theta0 * bounds_factor`` in log coordinates, i.e. plus or minus
        ``ln(bounds_factor)``. The bound exists so that a structurally
        non-identifiable direction cannot walk to infinity and take the
        integrator with it; a fit that ends on the bound is reported as not
        converged.

    Returns
    -------
    FitResult

    Raises
    ------
    ValueError
        If the problem has fewer observations than parameters, or ``theta0`` is
        invalid, or the tolerances are not positive.
    """
    if problem.degrees_of_freedom <= 0:
        raise ValueError(
            f"{problem.n_observations} observations cannot determine "
            f"{problem.n_parameters} parameters; need at least "
            f"{problem.n_parameters + 1}"
        )
    for name, value in (("xtol", xtol), ("ftol", ftol)):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if bounds_factor <= 1.0:
        raise ValueError("bounds_factor must exceed 1")
    start = problem.model.check_parameters(theta0)
    x0 = np.log(start)
    half = float(np.log(bounds_factor))

    residuals = GuardedResidual(problem)

    sol = least_squares(
        residuals,
        x0,
        bounds=(x0 - half, x0 + half),
        xtol=xtol,
        ftol=ftol,
        gtol=1e-12,
        max_nfev=max_nfev,
    )
    theta_hat = np.exp(sol.x)
    sens = log_sensitivity(
        problem.model,
        theta_hat,
        problem.inputs,
        problem.initial_state,
        problem.channels,
        problem.sigma,
        problem.sample_stride,
        log_step=log_step,
    )
    gauss_newton = sens.fisher()
    singular = False
    try:
        cond = float(np.linalg.cond(gauss_newton))
        if not np.isfinite(cond) or cond > 1.0 / np.finfo(float).eps:
            raise np.linalg.LinAlgError("ill-conditioned")
        cov = np.linalg.inv(gauss_newton)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(gauss_newton, rcond=1e-10)
        singular = True
    diag = np.diag(cov).astype(float)
    rel_sd = np.where(diag > 0.0, np.sqrt(np.maximum(diag, 0.0)), np.inf)
    if singular:
        # A pseudo-inverse reports a finite variance for a parameter the data
        # does not constrain, because it drops the infinite direction. Put the
        # infinity back, so that the profile grid in twincalib.profile is sized
        # from "unknown" rather than from a number that is not there.
        loading = null_direction_loading(sens, rank_tol=1e-8)
        rel_sd = np.where(loading > 1e-6, np.inf, rel_sd)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.sqrt(np.outer(diag, diag))
        corr = np.where(scale > 0.0, cov / scale, np.nan)
    np.fill_diagonal(corr, 1.0)
    r = residual_vector(problem, theta_hat)
    chi2 = float(r @ r)
    raw = problem.y - problem.predict(theta_hat)
    on_bound = bool(np.any(np.abs(sol.x - x0) > half - 1e-9))
    return FitResult(
        theta=theta_hat,
        parameter_names=tuple(problem.model.parameter_names),
        chi2=chi2,
        reduced_chi2=chi2 / problem.degrees_of_freedom,
        neg_log_likelihood=0.5 * chi2,
        covariance_log=cov,
        relative_sd=rel_sd,
        correlation=corr,
        sensitivity=sens,
        rmse=float(np.sqrt(np.mean(raw**2))),
        success=bool(sol.success) and not on_bound,
        status=int(sol.status),
        message=(
            "a parameter reached the search bound, so this is not an interior "
            "optimum: " + str(sol.message)
            if on_bound
            else str(sol.message)
        ),
        n_function_evaluations=int(sol.nfev),
        covariance_singular=singular,
        n_rejected_evaluations=int(residuals.n_rejected),
    )
