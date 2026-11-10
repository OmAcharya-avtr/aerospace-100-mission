"""Baseline 2: fitted correlation-length models over site separation.

Three fitted forms, in increasing capacity, all of them isotropic in the sense
that they see only a scalar separation:

**(23) The specified two-parameter exponential.** The model the specification
names,

    rho(d) = rho_0 exp( -d / L ),   0 <= rho_0 <= 1,  L > 0

fitted by least squares over the implied correlations of equation (11) across
training pairs, pooled over regimes. Two parameters: a nugget and a correlation
length. The exponential form is the Matern covariance at smoothness one half
(Stein 1999 chapter 2) and is the standard first choice for a correlation-length
description of a geophysical field; ITU-R P.1815 and the site-diversity material
of P.618 use separation-parameterised diversity statistics of the same shape.

**(24) The per-regime exponential.** The same form fitted independently within
each declared seasonal regime, four times two parameters. This is the
strengthened baseline. The build guide of this mission is explicit that where a
specification names a baseline that measures weakly, the strongest honest
baseline is used and the specified one is reported beside it, so both are fitted
and both are reported everywhere.

**(25) Identifiability of the pair ``(rho_0, L)``.** The two parameters are not
separately well determined by a finite set of separations. Along the direction
that holds ``rho_0 exp(-d_bar / L)`` fixed at the mean sampled separation
``d_bar``, the residual sum of squares changes only at second order in the
curvature of the sampled separation range. The consequence is measured rather than argued:
``validation/validate_correlation.py`` reports the fitted ``(rho_0, L)`` against
the generator's truth AND the maximum absolute error of the fitted correlation
FUNCTION over the sampled separations, and the second is far smaller than the
first. The figure of merit for a correlation model is the function it implies at
the separations that exist, not either parameter on its own, and this product
quotes the function error as the primary number with the parameter errors beside
it.

**(26) Building a correlation matrix from a pair model.** A network's joint
unavailability needs a full correlation matrix, and a matrix assembled
entry-by-entry from a pair model need not be positive semi-definite. The
exponential form of equation (23) is a positive-definite function of Euclidean
distance on R^n for every length scale -- it is the Matern covariance at
smoothness one half, Stein 1999 chapter 2 -- but that guarantee does not reach
either of the two cases this module actually assembles matrices for. A
great-circle distance is not a Euclidean distance, and the LEARNED model of
``sitediverse.learned`` is not of the exponential form at all.

``matrix_from_pairs`` therefore projects onto the nearest positive-definite
matrix in the Frobenius norm by eigenvalue clipping
and REPORTS the projection it made: the least eigenvalue before clipping and the
Frobenius distance moved. A projection that moves the matrix appreciably is a
statement about the model, and it is printed rather than absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

#: Lower and upper bounds on ``rho_0`` of equation (23). The upper bound is 1.0:
#: a correlation above one is not a correlation.
NUGGET_BOUNDS: tuple[float, float] = (0.0, 1.0)

#: Lower and upper bounds on ``L`` of equation (23), kilometres. The upper bound
#: of 30000 km exceeds half the Earth's circumference, so a fit that reaches it
#: is saying the field is uniformly correlated and is reported as such.
LENGTH_BOUNDS: tuple[float, float] = (1.0, 30000.0)

#: Declared starting point of the least-squares fit.
START: tuple[float, float] = (0.9, 600.0)

#: Smallest eigenvalue the projection of equation (26) will accept.
EIGENVALUE_FLOOR: float = 1e-8


@dataclass(frozen=True)
class ExponentialModel:
    """A fitted two-parameter exponential correlation-length model, equation (23).

    Attributes
    ----------
    nugget:
        ``rho_0``, dimensionless.
    length_km:
        ``L``, kilometres.
    residual_rms:
        Root-mean-square residual of the fit against the implied correlations it
        was fitted to, dimensionless.
    sample_count:
        Number of training points.
    regime:
        The regime this model was fitted on, or ``None`` for the pooled fit of
        equation (23).
    at_bound:
        Names of the parameters the optimiser left sitting on a bound. A fit on
        a bound is not an estimate and the flag is carried so that no report
        quotes one as if it were.
    """

    nugget: float
    length_km: float
    residual_rms: float
    sample_count: int
    regime: int | None
    at_bound: tuple[str, ...]

    def predict(self, separation_km: np.ndarray | float) -> np.ndarray:
        """``rho(d)`` of equation (23) at the given separations."""
        distance = np.asarray(separation_km, dtype=float)
        if np.any(distance < 0.0):
            raise ValueError("separation_km must be non-negative")
        return self.nugget * np.exp(-distance / self.length_km)

    def e_folding_km(self) -> float:
        """Separation at which the correlation falls to ``rho_0 / e``."""
        return self.length_km


def fit_exponential(
    separation_km: np.ndarray,
    implied_rho: np.ndarray,
    regime: int | None = None,
    start: tuple[float, float] = START,
) -> ExponentialModel:
    """Fit equation (23) by least squares.

    Parameters
    ----------
    separation_km:
        ``(n,)`` great-circle separations, km.
    implied_rho:
        ``(n,)`` implied latent correlations from equation (11).
    regime:
        Regime label to record on the result, or ``None`` for a pooled fit.
    start:
        Starting point, ``(rho_0, L)``.

    Returns
    -------
    ExponentialModel
    """
    distance = np.asarray(separation_km, dtype=float)
    target = np.asarray(implied_rho, dtype=float)
    if distance.shape != target.shape:
        raise ValueError(
            f"separation_km and implied_rho must have the same shape, got "
            f"{distance.shape} and {target.shape}"
        )
    if distance.ndim != 1:
        raise ValueError(f"inputs must be one-dimensional, got {distance.ndim}D")
    if distance.size < 2:
        raise ValueError(
            f"at least 2 points are needed to fit 2 parameters, got {distance.size}"
        )
    if np.any(distance < 0.0):
        raise ValueError("separation_km must be non-negative")

    def residual(parameters: np.ndarray) -> np.ndarray:
        return parameters[0] * np.exp(-distance / parameters[1]) - target

    solution = least_squares(
        residual,
        np.asarray(start, dtype=float),
        bounds=(
            (NUGGET_BOUNDS[0], LENGTH_BOUNDS[0]),
            (NUGGET_BOUNDS[1], LENGTH_BOUNDS[1]),
        ),
    )
    nugget, length = float(solution.x[0]), float(solution.x[1])
    on_bound: list[str] = []
    for name, value, (low, high) in (
        ("nugget", nugget, NUGGET_BOUNDS),
        ("length_km", length, LENGTH_BOUNDS),
    ):
        if abs(value - low) <= 1e-9 * max(1.0, abs(low)) or abs(
            value - high
        ) <= 1e-9 * max(1.0, abs(high)):
            on_bound.append(name)
    return ExponentialModel(
        nugget=nugget,
        length_km=length,
        residual_rms=float(np.sqrt(np.mean(solution.fun**2))),
        sample_count=int(distance.size),
        regime=regime,
        at_bound=tuple(on_bound),
    )


@dataclass(frozen=True)
class PerRegimeModel:
    """Equation (24): one exponential model per declared regime."""

    models: tuple[ExponentialModel, ...]

    def predict(
        self, separation_km: np.ndarray, regime: np.ndarray
    ) -> np.ndarray:
        """``rho`` per point, each point using its own regime's model."""
        distance = np.asarray(separation_km, dtype=float)
        label = np.asarray(regime, dtype=int)
        if distance.shape != label.shape:
            raise ValueError(
                f"separation_km and regime must have the same shape, got "
                f"{distance.shape} and {label.shape}"
            )
        if label.size and (label.min() < 0 or label.max() >= len(self.models)):
            raise ValueError(
                f"regime labels must lie in [0, {len(self.models)}), got "
                f"[{int(label.min())}, {int(label.max())}]"
            )
        out = np.empty(distance.shape, dtype=float)
        for index, model in enumerate(self.models):
            mask = label == index
            if np.any(mask):
                out[mask] = model.predict(distance[mask])
        return out

    @property
    def residual_rms(self) -> float:
        """Sample-weighted root-mean-square residual across the regimes."""
        weights = np.array([model.sample_count for model in self.models], dtype=float)
        squares = np.array([model.residual_rms**2 for model in self.models])
        return float(np.sqrt(np.sum(weights * squares) / np.sum(weights)))


def fit_per_regime(
    separation_km: np.ndarray,
    implied_rho: np.ndarray,
    regime: np.ndarray,
    regime_count: int,
    start: tuple[float, float] = START,
) -> PerRegimeModel:
    """Fit equation (24): one equation (23) model per regime."""
    label = np.asarray(regime, dtype=int)
    models = []
    for index in range(regime_count):
        mask = label == index
        if np.count_nonzero(mask) < 2:
            raise ValueError(
                f"regime {index} has {int(np.count_nonzero(mask))} training "
                "points, which is too few to fit 2 parameters"
            )
        models.append(
            fit_exponential(
                np.asarray(separation_km, dtype=float)[mask],
                np.asarray(implied_rho, dtype=float)[mask],
                regime=index,
                start=start,
            )
        )
    return PerRegimeModel(models=tuple(models))


@dataclass(frozen=True)
class ProjectedMatrix:
    """A correlation matrix and the projection of equation (26) applied to it.

    Attributes
    ----------
    matrix:
        The positive-definite matrix, with a unit diagonal.
    least_eigenvalue_before:
        The smallest eigenvalue of the raw assembled matrix. Negative means the
        pair model was not consistent with any joint Gaussian field.
    frobenius_shift:
        ``||projected - raw||_F``. Zero when no projection was needed.
    projected:
        Whether any eigenvalue had to be clipped.
    """

    matrix: np.ndarray
    least_eigenvalue_before: float
    frobenius_shift: float
    projected: bool


def matrix_from_pairs(
    pair_rho: np.ndarray, floor: float = EIGENVALUE_FLOOR
) -> ProjectedMatrix:
    """Equation (26): assemble and, if needed, project a correlation matrix.

    Parameters
    ----------
    pair_rho:
        ``(n, n)`` symmetric matrix of pairwise correlations. The diagonal is
        overwritten with ones.
    floor:
        Smallest eigenvalue accepted.

    Returns
    -------
    ProjectedMatrix
    """
    raw = np.asarray(pair_rho, dtype=float).copy()
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError(f"pair_rho must be square, got shape {raw.shape}")
    if floor <= 0.0:
        raise ValueError(f"floor must be strictly positive, got {floor!r}")
    raw = 0.5 * (raw + raw.T)
    np.fill_diagonal(raw, 1.0)
    eigenvalues, vectors = np.linalg.eigh(raw)
    least = float(eigenvalues.min())
    if least >= floor:
        return ProjectedMatrix(raw, least, 0.0, False)
    clipped = vectors @ np.diag(np.maximum(eigenvalues, floor)) @ vectors.T
    clipped = 0.5 * (clipped + clipped.T)
    scale = np.sqrt(np.diag(clipped))
    normalised = clipped / np.outer(scale, scale)
    np.fill_diagonal(normalised, 1.0)
    return ProjectedMatrix(
        matrix=normalised,
        least_eigenvalue_before=least,
        frobenius_shift=float(np.linalg.norm(normalised - raw, ord="fro")),
        projected=True,
    )
