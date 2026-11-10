"""Gaussian-copula joint unavailability, and the correlation implied by a record.

What this module computes
-------------------------
The probability that EVERY station in a set is cloudy at the same epoch, under a
Gaussian copula with a given correlation matrix and given marginal cloud
probabilities. For two stations it is computed exactly, by quadrature on an
identity; for more than two it is computed by Monte Carlo with a reported
standard error, and is never called closed form.

**(7) Sklar's theorem and the Gaussian copula.** Any joint distribution with
continuous margins factors into its margins and a unique copula (Sklar 1959,
*Publications de l'Institut de Statistique de l'Universite de Paris* 8, 229;
see Nelsen 2006, *An Introduction to Copulas*, 2nd ed., chapter 2 for the
statement and its binary-margin caveat). Choosing the Gaussian copula for a
binary cloud indicator is a MODELLING ASSUMPTION and not a consequence of the
theorem: for discrete margins the copula is not unique, so "the" copula of a
binary cloud field does not exist. What equation (8) therefore asserts is a
latent-variable model -- the indicator is a thresholded Gaussian -- which is
exactly how ``sitediverse.field`` generates the record and is a statement about
the field, not about the margins. The assumption is declared in README.md under
Limitations and is the first thing a reader applying this to a real record
should doubt.

**(8) Joint unavailability of two stations.** With marginal cloud probabilities
``q_i``, ``q_j`` at some epoch and latent correlation ``rho``,

    P(both cloudy) = Phi_2( h, k ; rho ),   h = Phi^{-1}(q_i), k = Phi^{-1}(q_j)

the standard bivariate normal distribution function.

**(9) The identity this is computed by.** The derivative of the bivariate normal
distribution function with respect to the correlation is the bivariate normal
DENSITY,

    d/d rho Phi_2(h, k; rho) = phi_2(h, k; rho)
    phi_2(h, k; r) = exp( -(h^2 - 2 r h k + k^2) / (2 (1 - r^2)) )
                     / ( 2 pi sqrt(1 - r^2) )

so, integrating from the independent case,

    Phi_2(h, k; rho) = Phi(h) Phi(k) + integral_0^rho phi_2(h, k; r) dr.

This is the reduction formula of Plackett 1954 (*Biometrika* 41, 351-360); the
derivative relation is also given by Sheppard 1899. Genz 2004 (*Statistics and
Computing* 14, 251-260) is the modern reference for computing bivariate and
trivariate normal probabilities numerically and is what SciPy's own multivariate
normal CDF implements -- which makes SciPy an INDEPENDENT implementation of the
same quantity by different mathematics, and it is used as a cross-check in
``validation/validate_copula.py``, never inside the product.

Equation (9) is not merely a way to compute equation (8). It is the reason this
product can report an overstatement without cancellation: the FIRST term is
exactly the independence baseline, and the integral is exactly the amount by
which correlation raises the joint unavailability above it. The difference
``Phi_2 - Phi(h)Phi(k)`` is computed as an integral of a positive integrand and
never as a subtraction of two nearly equal numbers. For two stations 7000 km
apart the two probabilities agree to six figures and the subtraction loses them
all; the integral returns the small difference to full relative precision. This
is the complement-space rule of the build guide applied to a difference rather
than to a product.

The integral is evaluated with piecewise Gauss-Legendre quadrature, ``SEGMENTS``
intervals of ``NODES`` points each, over ``[0, rho]``. The integrand is analytic
and bounded on that interval for ``|rho| < 1``, so the convergence is spectral;
the measured agreement with SciPy's independent implementation is reported in
``validation/validate_copula_output.txt``.

**(10) Joint unavailability of n stations.** For ``n > 2`` the orthant
probability has no elementary closed form. It is estimated here by Monte Carlo
over the pair (epoch, latent draw): an epoch is drawn uniformly from the record
and a correlated latent vector is drawn, and the estimate is the fraction of
draws below the epoch's thresholds. The estimator is unbiased for the
epoch-averaged orthant probability and its standard error is the binomial
``sqrt(p (1 - p) / M)``, which is REPORTED beside every n-station figure. It is
named Monte Carlo everywhere and never closed form.

**(11) The implied correlation.** Reading equation (8) backwards: given an
observed joint cloud rate ``P`` over a set of epochs and the marginal
climatology at those epochs, the implied latent correlation is the ``rho``
solving

    (1/T) sum_t Phi_2( h_it, h_jt ; rho )  =  P.

The left-hand side is strictly increasing in ``rho`` -- its derivative is the
mean of ``phi_2 > 0`` by equation (9) -- so the root is unique where it exists,
and Brent's method finds it. Where ``P`` lies outside the attainable range the
bracket end is returned and FLAGGED, because that is a real outcome for a
far-separated pair whose observed joint rate falls below the independent product
through sampling noise. This is the binary analogue of a tetrachoric
correlation (Pearson 1900); it is called the implied correlation here rather
than tetrachoric because the margins vary from epoch to epoch, which the
classical tetrachoric estimator does not allow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

#: Gauss-Legendre nodes per segment of the equation (9) quadrature. DECLARED.
NODES: int = 32

#: Segments the interval ``[0, rho]`` is cut into. DECLARED.
SEGMENTS: int = 4

#: The widest correlation the root finder of equation (11) will bracket. Not
#: 1.0: ``phi_2`` has an integrable singularity as ``|r| -> 1`` and the
#: quadrature's error grows there, so the bracket stops short and the end is
#: reported as reached rather than silently returned as an estimate.
RHO_MIN: float = -0.95
RHO_MAX: float = 0.995

_LEGENDRE_NODES, _LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(NODES)
_SEGMENT_EDGES = np.linspace(0.0, 1.0, SEGMENTS + 1)


def bivariate_density(
    h: np.ndarray | float, k: np.ndarray | float, rho: np.ndarray | float
) -> np.ndarray:
    """``phi_2(h, k; rho)`` of equation (9), the standard bivariate normal density.

    Parameters
    ----------
    h, k:
        Probit levels, dimensionless. Broadcast against one another.
    rho:
        Correlation in ``(-1, 1)``. A value with ``|rho| >= 1`` raises.

    Returns
    -------
    numpy.ndarray
        Density, strictly positive.
    """
    correlation = np.asarray(rho, dtype=float)
    if np.any(np.abs(correlation) >= 1.0):
        raise ValueError(
            "rho must satisfy |rho| < 1 for the bivariate normal density; got a "
            f"value of magnitude {float(np.max(np.abs(correlation))):.6f}"
        )
    left = np.asarray(h, dtype=float)
    right = np.asarray(k, dtype=float)
    gap = 1.0 - correlation**2
    quadratic = left**2 - 2.0 * correlation * left * right + right**2
    return np.exp(-quadratic / (2.0 * gap)) / (2.0 * np.pi * np.sqrt(gap))


def correlation_excess(
    h: np.ndarray | float, k: np.ndarray | float, rho: np.ndarray | float
) -> np.ndarray:
    """The integral of equation (9): ``Phi_2(h,k;rho) - Phi(h) Phi(k)``.

    This is the amount by which correlation raises the joint unavailability of a
    pair above the independence assumption, computed as an integral of a
    positive integrand rather than as a difference of two nearly equal
    probabilities. It is the headline quantity of this product at pair level.

    Returns
    -------
    numpy.ndarray
        The excess. Has the sign of ``rho`` and is zero exactly at ``rho = 0``.
    """
    left = np.asarray(h, dtype=float)
    right = np.asarray(k, dtype=float)
    correlation = np.asarray(rho, dtype=float)
    if np.any(np.abs(correlation) >= 1.0):
        raise ValueError(
            "rho must satisfy |rho| < 1; got a value of magnitude "
            f"{float(np.max(np.abs(correlation))):.6f}"
        )
    left, right, correlation = np.broadcast_arrays(left, right, correlation)
    edges = correlation[..., None] * _SEGMENT_EDGES
    lower = edges[..., :-1]
    upper = edges[..., 1:]
    middle = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    sample = middle[..., None] + half_width[..., None] * _LEGENDRE_NODES
    density = bivariate_density(
        left[..., None, None], right[..., None, None], sample
    )
    return np.sum(
        half_width[..., None] * _LEGENDRE_WEIGHTS * density, axis=(-1, -2)
    )


def bivariate_orthant(
    h: np.ndarray | float, k: np.ndarray | float, rho: np.ndarray | float
) -> np.ndarray:
    """``Phi_2(h, k; rho)`` of equation (8), via the identity of equation (9).

    Parameters
    ----------
    h, k:
        Probit levels of the two marginal cloud probabilities.
    rho:
        Latent correlation, ``|rho| < 1``.

    Returns
    -------
    numpy.ndarray
        Joint probability that both standard normal variates fall below their
        levels. In ``[0, 1]``.
    """
    independent = norm.cdf(np.asarray(h, dtype=float)) * norm.cdf(
        np.asarray(k, dtype=float)
    )
    return independent + correlation_excess(h, k, rho)


@dataclass(frozen=True)
class EpochNodes:
    """A weighted set of ``(h, k)`` nodes standing in for a run of epochs.

    The epoch-averaged joint unavailability of a pair is a mean over epochs of
    equation (8). Evaluating it at every epoch of an eight-year three-hourly
    record costs 23360 quadratures per correlation value, and the root finder of
    equation (11) needs about twenty such evaluations per pair. The epochs are
    therefore binned on their ``(h, k)`` values into a two-dimensional histogram
    of ``bins`` by ``bins`` cells, and the mean is taken over the occupied cells
    with the cell frequencies as weights.

    The node of a cell is the MEAN of the epochs in it, not the cell centre.
    This matters by nearly three orders of magnitude and was measured, not
    assumed: at the declared 48 bins the centre form reaches a worst error of
    4.4e-04 against the exact epoch-by-epoch mean and the cell-mean form
    reaches 6.0e-07. The reason is that the cell mean makes the first moment of
    each cell exact, so the leading term of the error is the cell's second
    moment against the curvature of equation (8) rather than its first moment
    against the gradient. The centre form's error was also NOT monotone in the
    bin count, because moving the bin edges moves the centres; the cell-mean
    form's is.

    The binning is an approximation and its error is MEASURED, not assumed:
    ``validation/validate_copula.py`` reports the difference between the binned
    mean and the exact epoch-by-epoch mean for every declared pair, and the
    headline figures of this product are computed epoch by epoch rather than
    from nodes. The nodes exist to make the FITTING loops affordable.
    """

    h: np.ndarray
    k: np.ndarray
    weight: np.ndarray
    epochs: int
    bins: int

    @property
    def node_count(self) -> int:
        """Number of occupied cells."""
        return int(self.h.size)

    def joint(self, rho: float) -> float:
        """Weighted mean of equation (8) over the nodes."""
        return float(
            np.sum(
                self.weight
                * bivariate_orthant(self.h, self.k, np.full(self.h.shape, rho))
            )
        )


#: Bins per axis of the ``EpochNodes`` histogram. DECLARED. The measured binning
#: error at this value is reported in validation/validate_copula_output.txt.
EPOCH_BINS: int = 48


def epoch_nodes(
    threshold_i: np.ndarray, threshold_j: np.ndarray, bins: int = EPOCH_BINS
) -> EpochNodes:
    """Bin a pair of epoch threshold series into weighted ``(h, k)`` nodes."""
    left = np.asarray(threshold_i, dtype=float)
    right = np.asarray(threshold_j, dtype=float)
    if left.shape != right.shape:
        raise ValueError(
            f"threshold series must have the same shape, got {left.shape} and "
            f"{right.shape}"
        )
    if left.ndim != 1:
        raise ValueError(f"threshold series must be one-dimensional, got {left.ndim}D")
    if left.size == 0:
        raise ValueError("threshold series must not be empty")
    if bins < 2:
        raise ValueError(f"bins must be at least 2, got {bins!r}")
    counts, edges_left, edges_right = np.histogram2d(left, right, bins=bins)
    sum_left, _, _ = np.histogram2d(
        left, right, bins=(edges_left, edges_right), weights=left
    )
    sum_right, _, _ = np.histogram2d(
        left, right, bins=(edges_left, edges_right), weights=right
    )
    occupied = counts > 0
    return EpochNodes(
        h=sum_left[occupied] / counts[occupied],
        k=sum_right[occupied] / counts[occupied],
        weight=counts[occupied] / float(left.size),
        epochs=int(left.size),
        bins=int(bins),
    )


def pair_joint_exact(
    threshold_i: np.ndarray, threshold_j: np.ndarray, rho: float
) -> float:
    """Epoch-by-epoch mean of equation (8). No binning, no Monte Carlo.

    This is the reference form. It is what every headline figure in this
    product uses and what ``EpochNodes.joint`` is measured against.
    """
    left = np.asarray(threshold_i, dtype=float)
    right = np.asarray(threshold_j, dtype=float)
    if left.shape != right.shape:
        raise ValueError(
            f"threshold series must have the same shape, got {left.shape} and "
            f"{right.shape}"
        )
    return float(
        np.mean(bivariate_orthant(left, right, np.full(left.shape, float(rho))))
    )


def pair_excess_exact(
    threshold_i: np.ndarray, threshold_j: np.ndarray, rho: float
) -> float:
    """Epoch-by-epoch mean of the equation (9) excess, without cancellation."""
    left = np.asarray(threshold_i, dtype=float)
    right = np.asarray(threshold_j, dtype=float)
    return float(
        np.mean(correlation_excess(left, right, np.full(left.shape, float(rho))))
    )


@dataclass(frozen=True)
class ImpliedCorrelation:
    """The result of inverting equation (11) for one pair over one epoch set.

    Attributes
    ----------
    rho:
        The implied latent correlation.
    at_bracket_end:
        ``True`` when the observed joint rate lies outside the range the copula
        can reach over ``[RHO_MIN, RHO_MAX]`` and the bracket end was returned.
        A far-separated pair whose observed joint rate falls below the
        independent product through sampling noise lands here, and the flag is
        carried rather than hidden because such a pair contributes a biased
        point to any fit.
    observed:
        The joint cloud rate the inversion targeted.
    reproduced:
        The joint rate the returned ``rho`` reproduces. Equal to ``observed`` to
        the root finder's tolerance unless ``at_bracket_end``.
    """

    rho: float
    at_bracket_end: bool
    observed: float
    reproduced: float

    @property
    def residual(self) -> float:
        """``reproduced - observed``. Zero to tolerance for an interior root."""
        return self.reproduced - self.observed


def implied_correlation(
    nodes: EpochNodes, observed_joint: float, tolerance: float = 1e-10
) -> ImpliedCorrelation:
    """Invert equation (11) for the latent correlation of one pair.

    Parameters
    ----------
    nodes:
        Weighted epoch nodes for the pair, from :func:`epoch_nodes`.
    observed_joint:
        The observed joint cloud rate over the same epochs, in ``[0, 1]``.
    tolerance:
        Absolute tolerance on ``rho`` passed to Brent's method.

    Returns
    -------
    ImpliedCorrelation
    """
    target = float(observed_joint)
    if not 0.0 <= target <= 1.0:
        raise ValueError(
            f"observed_joint must lie in [0, 1], got {target!r}"
        )
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be strictly positive, got {tolerance!r}")

    def gap(rho: float) -> float:
        return nodes.joint(rho) - target

    low = gap(RHO_MIN)
    high = gap(RHO_MAX)
    if low > 0.0:
        return ImpliedCorrelation(RHO_MIN, True, target, nodes.joint(RHO_MIN))
    if high < 0.0:
        return ImpliedCorrelation(RHO_MAX, True, target, nodes.joint(RHO_MAX))
    root = float(brentq(gap, RHO_MIN, RHO_MAX, xtol=tolerance))
    return ImpliedCorrelation(root, False, target, nodes.joint(root))


@dataclass(frozen=True)
class MonteCarloJoint:
    """An ``n``-station orthant estimate of equation (10) with its error.

    Attributes
    ----------
    estimate:
        The estimated epoch-averaged joint unavailability.
    standard_error:
        Binomial standard error of the estimate, ``sqrt(p (1 - p) / samples)``.
        This is the MONTE CARLO error of the estimator and says nothing about
        the record's sampling uncertainty, which is
        ``sitediverse.uncertainty``'s business.
    samples:
        Number of draws.
    """

    estimate: float
    standard_error: float
    samples: int

    @property
    def relative_error(self) -> float:
        """``standard_error / estimate``, or ``inf`` at an estimate of zero."""
        return (
            self.standard_error / self.estimate if self.estimate > 0.0 else float("inf")
        )


def network_joint_monte_carlo(
    thresholds: np.ndarray,
    correlation: np.ndarray,
    samples: int,
    seed: int,
) -> MonteCarloJoint:
    """Equation (10): the ``n``-station joint unavailability, by Monte Carlo.

    Parameters
    ----------
    thresholds:
        ``(epochs, n)`` probit thresholds of the stations in the network.
    correlation:
        ``(n, n)`` latent correlation matrix. Must be positive definite; a
        matrix that is not is a modelling error and raises rather than being
        silently repaired.
    samples:
        Number of (epoch, latent) draws.
    seed:
        Integer seed.

    Returns
    -------
    MonteCarloJoint
    """
    level = np.asarray(thresholds, dtype=float)
    if level.ndim != 2:
        raise ValueError(f"thresholds must be two-dimensional, got {level.ndim}D")
    count = level.shape[1]
    matrix = np.asarray(correlation, dtype=float)
    if matrix.shape != (count, count):
        raise ValueError(
            f"correlation must be ({count}, {count}) to match thresholds, got "
            f"{matrix.shape}"
        )
    if samples < 2:
        raise ValueError(f"samples must be at least 2, got {samples!r}")
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        least = float(np.min(np.linalg.eigvalsh(matrix)))
        raise ValueError(
            "correlation matrix is not positive definite (least eigenvalue "
            f"{least:.6e}); build it with sitediverse.correlation.matrix_from_pairs, "
            "which reports the projection it had to make"
        ) from error
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((int(samples), count)) @ factor.T
    epoch = rng.integers(0, level.shape[0], int(samples))
    hit = np.all(latent < level[epoch], axis=1)
    rate = float(hit.mean())
    return MonteCarloJoint(
        estimate=rate,
        standard_error=float(np.sqrt(rate * (1.0 - rate) / float(samples))),
        samples=int(samples),
    )
