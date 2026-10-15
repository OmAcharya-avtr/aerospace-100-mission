"""Parameter sensitivities and the Fisher information matrix.

Everything downstream -- identifiability, the covariance of a fit, the
projection decomposition of a learned residual -- is built on one object: the
sensitivity matrix of the predicted observations with respect to the parameters,
in log-parameter coordinates and weighted by the measurement noise.

Why log-parameter coordinates
-----------------------------
The six parameters of the two-node twin span 11 orders of magnitude in
magnitude and have five different units (J/K, W/K, W/K^4, W). A condition
number computed on the raw Jacobian is then dominated by the choice of units and
says nothing. Differentiating with respect to ``ln theta`` instead gives

``dy/d(ln theta_j) = theta_j dy/dtheta_j``

which has the unit of ``y`` for every column, so the columns are comparable and
the singular-value spectrum is a statement about the experiment rather than
about the unit system. This is the standard relative-sensitivity scaling; see
Brun, Reichert and Kuensch (2001), section 2.2.

Weighting
---------
Rows are divided by the per-observation noise standard deviation, so that the
Gram matrix of the result is the Fisher information for independent Gaussian
errors of known variance:

``F = J_w^T J_w``,   ``J_w[i, j] = (1/sigma_i) dy_i/d(ln theta_j)``

For that model the Fisher information is exactly ``J^T Sigma^-1 J`` (Bard 1974,
Chapter 7), and the Cramer-Rao bound says the covariance of any unbiased
estimator is at least ``F^-1`` (Rao 1945; Cramer 1946).

References
----------
Bard, Y., *Nonlinear Parameter Estimation*, Academic Press, 1974. Chapter 7:
the covariance of a weighted least-squares estimate and its Gauss-Newton
approximation.

Brun, R., Reichert, P. and Kuensch, H. R. (2001), "Practical identifiability
analysis of large environmental simulation models", *Water Resources Research*
37(4), 1015-1030. Relative sensitivity scaling, the sensitivity measure
``delta_msqr`` and the collinearity index.

Cramer, H., *Mathematical Methods of Statistics*, Princeton University Press,
1946, section 32.3. Rao, C. R. (1945), *Bulletin of the Calcutta Mathematical
Society* 37, 81-91. The information inequality.

Fornberg, B. (1988), "Generation of finite difference formulas on arbitrarily
spaced grids", *Mathematics of Computation* 51(184), 699-706. The central
difference used here is the two-point case; its truncation error is
``O(h^2) y'''``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .model import Inputs, TwinModel

__all__ = [
    "DEFAULT_LOG_STEP",
    "SensitivityMatrix",
    "fisher_information",
    "log_sensitivity",
    "predicted_observations",
]

DEFAULT_LOG_STEP: float = 1e-4
"""Default central-difference step in log-parameter space, dimensionless.

Chosen by a step study rather than by rule of thumb: see
``validation/validate_sensitivity_output.txt``, which sweeps the step over seven
decades against the closed-form sensitivity of the single-node model and reports
the error at each. The optimum for a central difference is near
``eps**(1/3) ~ 6e-6`` for a well-scaled function, but the integrator's own
truncation error acts as noise on the function, which pushes the optimum
upwards; 1e-4 is inside the measured flat region for both shipped models.
"""


def predicted_observations(
    model: TwinModel,
    theta: Sequence[float],
    inputs: Inputs,
    initial_state: Sequence[float],
    channels: Sequence[int],
    sample_stride: int = 1,
) -> np.ndarray:
    """Model prediction of the observation vector, K, shape ``(n_obs,)``.

    A thin alias for ``model.observe`` kept so that callers of this module never
    need to import the model class.
    """
    return model.observe(theta, inputs, initial_state, channels, sample_stride)


@dataclass(frozen=True)
class SensitivityMatrix:
    """A weighted log-parameter sensitivity matrix and the pieces it came from.

    Attributes
    ----------
    jacobian_log:
        ``dy/d(ln theta)``, K per natural log unit, shape ``(n_obs, n_par)``.
    weighted:
        ``jacobian_log / sigma[:, None]``, dimensionless, shape
        ``(n_obs, n_par)``. This is the matrix whose Gram matrix is the Fisher
        information.
    prediction:
        Model prediction at ``theta``, K, shape ``(n_obs,)``.
    sigma:
        Per-observation noise standard deviation, K, shape ``(n_obs,)``.
    theta:
        The parameter vector the matrix was evaluated at.
    parameter_names:
        Names in column order.
    log_step:
        The central-difference step that was used, dimensionless.
    """

    jacobian_log: np.ndarray
    weighted: np.ndarray
    prediction: np.ndarray
    sigma: np.ndarray
    theta: np.ndarray
    parameter_names: tuple[str, ...]
    log_step: float

    @property
    def n_observations(self) -> int:
        """Number of scalar observations, dimensionless."""
        return int(self.weighted.shape[0])

    @property
    def n_parameters(self) -> int:
        """Number of parameters, dimensionless."""
        return int(self.weighted.shape[1])

    def fisher(self) -> np.ndarray:
        """Fisher information in log-parameter coordinates, shape ``(p, p)``.

        ``F = J_w^T J_w``, symmetrised explicitly so that the returned matrix is
        symmetric to the last bit rather than only up to the rounding of the
        product.
        """
        f = self.weighted.T @ self.weighted
        return 0.5 * (f + f.T)

    def column_norms(self) -> np.ndarray:
        """Euclidean norm of each weighted column, dimensionless, shape ``(p,)``."""
        return np.linalg.norm(self.weighted, axis=0)

    def delta_msqr(self) -> np.ndarray:
        """Brun et al. (2001) sensitivity measure, shape ``(p,)``.

        ``delta_j = ||J_w[:, j]|| / sqrt(n_obs)``: the root-mean-square change
        in the observations, in units of the measurement noise, caused by a
        change of one natural log unit in parameter ``j``. A value of 1 means a
        100 per cent relative change in that parameter moves the data by about
        one standard deviation per point, which is barely detectable.
        """
        return self.column_norms() / np.sqrt(self.n_observations)


def log_sensitivity(
    model: TwinModel,
    theta: Sequence[float],
    inputs: Inputs,
    initial_state: Sequence[float],
    channels: Sequence[int],
    sigma: Sequence[float] | float,
    sample_stride: int = 1,
    log_step: float = DEFAULT_LOG_STEP,
) -> SensitivityMatrix:
    """Central-difference sensitivity of the observations to ``ln theta``.

    Parameters
    ----------
    model:
        The calibration model.
    theta:
        Parameter vector to differentiate at. All entries must be positive,
        which the model's own validation enforces, because the logarithm is
        taken.
    inputs, initial_state, channels, sample_stride:
        The observation design, as for ``model.observe``.
    sigma:
        Measurement noise standard deviation, K. A scalar is broadcast.
    log_step:
        Relative step ``h``. Column ``j`` is
        ``(y(theta_j e^h) - y(theta_j e^-h)) / (2 h)``, which is a central
        difference with respect to ``ln theta_j`` with truncation error
        ``O(h^2)``.

    Returns
    -------
    SensitivityMatrix

    Raises
    ------
    ValueError
        If ``log_step`` is not strictly positive and below 1, or if ``sigma`` is
        not strictly positive, or if its shape does not match the observations.
    """
    if not np.isfinite(log_step) or not 0.0 < log_step < 1.0:
        raise ValueError("log_step must be finite and lie in (0, 1)")
    values = model.check_parameters(theta)
    base = model.observe(values, inputs, initial_state, channels, sample_stride)
    raw_sigma = np.asarray(sigma, dtype=float)
    if raw_sigma.ndim > 1 or (raw_sigma.ndim == 1 and raw_sigma.shape != base.shape):
        raise ValueError(
            f"sigma must be a scalar or have shape {base.shape}, got {raw_sigma.shape}"
        )
    sig = np.broadcast_to(raw_sigma, base.shape).astype(float)
    if np.any(sig <= 0.0) or not np.all(np.isfinite(sig)):
        raise ValueError("sigma must be finite and strictly positive")
    cols = np.empty((base.size, values.size), dtype=float)
    scale_up = float(np.exp(log_step))
    scale_dn = float(np.exp(-log_step))
    for j in range(values.size):
        up = values.copy()
        dn = values.copy()
        up[j] *= scale_up
        dn[j] *= scale_dn
        y_up = model.observe(up, inputs, initial_state, channels, sample_stride)
        y_dn = model.observe(dn, inputs, initial_state, channels, sample_stride)
        cols[:, j] = (y_up - y_dn) / (2.0 * log_step)
    return SensitivityMatrix(
        jacobian_log=cols,
        weighted=cols / sig[:, None],
        prediction=base,
        sigma=sig,
        theta=values,
        parameter_names=tuple(model.parameter_names),
        log_step=float(log_step),
    )


def fisher_information(sens: SensitivityMatrix) -> np.ndarray:
    """Fisher information matrix of a sensitivity matrix, shape ``(p, p)``.

    Equivalent to ``sens.fisher()``; provided as a free function because that is
    how the formula is written in Bard (1974, eq. 7-7-5).
    """
    return sens.fisher()
