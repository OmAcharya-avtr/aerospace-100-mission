"""The Webb distribution: counting statistics of an avalanche photodiode.

An APD does not multiply every primary carrier by the same factor. The gain is
a random variable, and the resulting output-count distribution is neither
Poisson nor Gaussian. Webb, McIntyre and Conradi gave the approximation that
the optical-receiver literature has used ever since (P. P. Webb, R. J.
McIntyre and J. Conradi, "Properties of avalanche photodiodes", *RCA Review*
**35**(2):234-278, 1974).

Written in the **gain-normalised** output variable ``y = N_out / G`` --- output
electrons divided by the mean gain, so that ``y`` is directly comparable with
the primary photoelectron count --- the Webb density is

    p(y) = (2 pi m F)^(-1/2) * (1 + u)^(-3/2)
           * exp( -(y - m)^2 / (2 m F (1 + u)) ),      u = (y - m) / delta,

    delta = m F / (F - 1),                                                (W1)

with support ``y > m - delta``. Here

===========  ==========================================================
``m``        mean number of primary photoelectrons per observation (-)
``F``        APD excess noise factor, ``F >= 1`` (-)
``G``        mean avalanche gain (-), used only to rescale ``y`` to electrons
===========  ==========================================================

**Moments.** Equation (W1) is not merely moment-matched to second order; its
first three moments are exact closed forms, and this module states them and
:mod:`validation.validate_webb_limits` measures them:

    E[y]            = m
    Var[y]          = m F                                                 (W2)
    E[(y - m)^3]    = 3 m F (F - 1)
    skewness        = 3 (F - 1) / sqrt(m F)

The third-moment and skewness forms were derived in this repository by exact
numerical quadrature against the candidate closed forms, not quoted from the
1974 paper; the validation script reproduces that derivation and its residuals.

**The two limits, which is how this implementation is checked without a lab.**

*Gaussian limit.* At ``F = 1`` the shape term ``(1 + u)`` is identically 1 and
(W1) collapses **exactly** to ``N(m, m)``. For ``F`` slightly above 1 the
density is a Gaussian of variance ``m F`` perturbed by a skew of order
``(F - 1) / sqrt(m F)``. The validation measures the sup-norm difference
against ``N(m, m F)`` as ``F -> 1``.

*Poisson limit.* The APD with no excess noise is a unity-gain counter, whose
primary-carrier count is Poisson with mean ``m``. The Webb density is
continuous, so it cannot *be* Poisson: at ``F = 1`` its third central moment
is 0 where the Poisson's is ``m``. The honest statement of the Poisson limit is
convergence of the **integer-binned** Webb probabilities to the Poisson pmf at
a rate set by the Gaussian approximation to the Poisson, i.e. total variation
falling as ``m^(-1/2)``. That is what :func:`binned_pmf` exists for and what
the validation measures. Anyone who reports a tighter Poisson agreement than
``m^(-1/2)`` has made an error.

**Validity.** (W1) is an approximation to McIntyre's exact gain distribution,
accurate for ``m F`` large enough that the continuous description is
meaningful and for ``F`` not far above 1. It is not valid for ``m`` of order
one, where the discreteness of the primary count dominates; use
:mod:`photoncount.poisson` there. Nothing in this module models dead time.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "WebbParameters",
    "apd_excess_noise_factor",
    "binned_pmf",
    "cdf",
    "gaussian_limit_pdf",
    "moments",
    "pdf",
    "sample",
    "shape_parameter",
    "support_lower_bound",
]


class WebbParameters:
    """Validated Webb parameters.

    Parameters
    ----------
    mean_primary:
        ``m``, mean primary photoelectrons per observation, ``> 0`` (-).
    excess_noise_factor:
        ``F >= 1`` (-). ``F = 1`` is the no-excess-noise (Gaussian) limit.
    gain:
        ``G > 0`` (-), mean avalanche gain. Only used to convert between the
        gain-normalised variable ``y`` and output electrons.
    """

    __slots__ = ("excess_noise_factor", "gain", "mean_primary")

    def __init__(
        self,
        mean_primary: float,
        excess_noise_factor: float = 1.0,
        gain: float = 1.0,
    ) -> None:
        m = float(mean_primary)
        f = float(excess_noise_factor)
        g = float(gain)
        if not np.isfinite(m) or m <= 0.0:
            raise ValueError(f"mean_primary must be finite and > 0, got {mean_primary!r}")
        if not np.isfinite(f) or f < 1.0:
            raise ValueError(f"excess_noise_factor must be finite and >= 1, got {f!r}")
        if not np.isfinite(g) or g <= 0.0:
            raise ValueError(f"gain must be finite and > 0, got {g!r}")
        self.mean_primary = m
        self.excess_noise_factor = f
        self.gain = g

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"WebbParameters(mean_primary={self.mean_primary!r}, "
            f"excess_noise_factor={self.excess_noise_factor!r}, gain={self.gain!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WebbParameters):
            return NotImplemented
        return (
            self.mean_primary == other.mean_primary
            and self.excess_noise_factor == other.excess_noise_factor
            and self.gain == other.gain
        )


def apd_excess_noise_factor(gain: float, ionisation_ratio: float) -> float:
    """APD excess noise factor ``F`` from mean gain and ionisation ratio.

    ``F = k G + (2 - 1/G)(1 - k)``, the standard relation for a uniform
    multiplication region, with ``k`` the ratio of hole to electron ionisation
    coefficients (``0 <= k <= 1``) and ``G >= 1``. Limits: ``k = 0`` gives
    ``F -> 2`` for large ``G`` (the best an APD can do); ``k = 1`` gives
    ``F = G`` (the worst). Reproduced from the APD properties surveyed in Webb,
    McIntyre & Conradi (1974); this module does not re-derive it and does not
    claim it for a non-uniform multiplication region.
    """
    g = float(gain)
    k = float(ionisation_ratio)
    if not np.isfinite(g) or g < 1.0:
        raise ValueError(f"gain must be finite and >= 1, got {g!r}")
    if not np.isfinite(k) or not 0.0 <= k <= 1.0:
        raise ValueError(f"ionisation_ratio must be in [0, 1], got {k!r}")
    return k * g + (2.0 - 1.0 / g) * (1.0 - k)


def shape_parameter(params: WebbParameters) -> float:
    """``delta = m F / (F - 1)`` in gain-normalised units (-).

    Returns ``inf`` at ``F = 1``, which is the exact Gaussian case.
    """
    f = params.excess_noise_factor
    if f == 1.0:
        return float("inf")
    return params.mean_primary * f / (f - 1.0)


def support_lower_bound(params: WebbParameters) -> float:
    """Lower edge of the support, ``m - delta``, in gain-normalised units.

    ``-inf`` at ``F = 1``. Note this is often negative, i.e. the Webb density
    assigns probability to a negative gain-normalised count. That is an
    artefact of a continuous approximation to a non-negative count and is one
    reason :func:`binned_pmf` clips at zero and reports the clipped mass.
    """
    return params.mean_primary - shape_parameter(params)


def pdf(y: np.ndarray | float, params: WebbParameters) -> np.ndarray:
    """Webb density in the gain-normalised variable ``y`` (per unit ``y``).

    Zero outside the support. Equation (W1).
    """
    yy = np.atleast_1d(np.asarray(y, dtype=float))
    m = params.mean_primary
    f = params.excess_noise_factor
    var = m * f
    out = np.zeros_like(yy)
    if f == 1.0:
        return np.exp(-((yy - m) ** 2) / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)
    delta = shape_parameter(params)
    u = (yy - m) / delta
    ok = u > -1.0
    if np.any(ok):
        uo = u[ok]
        dy = yy[ok] - m
        out[ok] = (
            (1.0 + uo) ** -1.5
            * np.exp(-(dy**2) / (2.0 * var * (1.0 + uo)))
            / np.sqrt(2.0 * np.pi * var)
        )
    return out


def gaussian_limit_pdf(y: np.ndarray | float, params: WebbParameters) -> np.ndarray:
    """``N(m, m F)`` density, the ``F -> 1`` limit of (W1), per unit ``y``."""
    yy = np.atleast_1d(np.asarray(y, dtype=float))
    var = params.mean_primary * params.excess_noise_factor
    return np.exp(-((yy - params.mean_primary) ** 2) / (2.0 * var)) / np.sqrt(
        2.0 * np.pi * var
    )


def moments(params: WebbParameters) -> dict[str, float]:
    """Closed-form moments (W2), in gain-normalised units.

    Keys: ``mean``, ``variance``, ``third_central``, ``skewness``.
    Multiply ``mean`` by ``G`` and ``variance`` by ``G**2`` for output
    electrons; ``skewness`` is scale-free.
    """
    m = params.mean_primary
    f = params.excess_noise_factor
    var = m * f
    third = 3.0 * m * f * (f - 1.0)
    return {
        "mean": m,
        "variance": var,
        "third_central": third,
        "skewness": third / var**1.5,
    }


def _grid(params: WebbParameters, n_points: int, n_sigma: float) -> np.ndarray:
    m = params.mean_primary
    sd = np.sqrt(m * params.excess_noise_factor)
    lo = max(support_lower_bound(params), m - n_sigma * sd)
    hi = m + n_sigma * sd + 4.0 * sd * (params.excess_noise_factor - 1.0) * n_sigma
    return np.linspace(lo, hi, int(n_points))


def cdf(
    y: np.ndarray | float,
    params: WebbParameters,
    n_points: int = 200_001,
    n_sigma: float = 40.0,
) -> np.ndarray:
    """Webb CDF by trapezoidal quadrature of (W1) on a fixed grid.

    The grid spans the support from ``max(m - delta, m - n_sigma sd)`` upward
    and is normalised so the CDF reaches 1 at its top. Quadrature error on the
    default grid is measured in ``validation/validate_webb_limits.py``; it is
    not a tolerance to be assumed.
    """
    if params.excess_noise_factor == 1.0:
        return np.atleast_1d(
            stats.norm.cdf(
                np.asarray(y, dtype=float),
                loc=params.mean_primary,
                scale=np.sqrt(params.mean_primary),
            )
        )
    grid = _grid(params, n_points, n_sigma)
    dens = pdf(grid, params)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
    cum /= cum[-1]
    return np.atleast_1d(np.interp(np.asarray(y, dtype=float), grid, cum, left=0.0, right=1.0))


def binned_pmf(
    params: WebbParameters,
    k_max: int,
    n_points: int = 200_001,
) -> np.ndarray:
    """Integer-binned Webb probabilities for ``k = 0 .. k_max`` (-).

    Bin ``k`` is ``CDF(k + 1/2) - CDF(k - 1/2)``, with bin 0 absorbing all mass
    below ``1/2`` (including the negative-``y`` mass the continuous
    approximation puts there) and the array renormalised to sum to 1. This is
    the only form in which a Webb density may be compared with a Poisson pmf.
    """
    kk = int(k_max)
    if kk < 0:
        raise ValueError(f"k_max must be >= 0, got {kk!r}")
    edges = np.arange(0, kk + 2, dtype=float) - 0.5
    cum = cdf(edges, params, n_points=n_points)
    probs = np.diff(cum)
    probs[0] += cum[0]
    total = probs.sum()
    if total <= 0.0:
        raise ValueError("binned_pmf produced zero total mass; k_max is far from the mean")
    return probs / total


def sample(
    params: WebbParameters,
    size: int,
    rng: np.random.Generator,
    n_points: int = 200_001,
) -> np.ndarray:
    """Draw ``size`` gain-normalised Webb variates by inverse-CDF interpolation.

    Deterministic for a given ``rng`` state. The returned values are continuous
    (gain-normalised electrons), not integer counts; round or bin them if you
    need counts.
    """
    n = int(size)
    if n < 0:
        raise ValueError(f"size must be >= 0, got {n!r}")
    if params.excess_noise_factor == 1.0:
        return rng.normal(
            params.mean_primary, np.sqrt(params.mean_primary), size=n
        )
    grid = _grid(params, n_points, 40.0)
    dens = pdf(grid, params)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
    cum /= cum[-1]
    return np.interp(rng.random(n), cum, grid)
