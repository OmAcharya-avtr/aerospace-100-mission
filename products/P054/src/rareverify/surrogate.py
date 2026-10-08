"""Learned surrogate of the limit-state function, used to steer sampling.

What this is and what it is for
-------------------------------
The analytic baseline in :mod:`rareverify.tilting` needs the design point of
the limit state.  For the linear and lognormal instances that point is known in
closed form and costs nothing.  For a limit state that is only available as a
simulator, it is not, and the standard move is to fit a cheap surrogate to a
small number of expensive evaluations and locate the design point on the
surrogate instead.  That is what this module does, with a Gaussian-process
regressor as the surrogate, which is the method family of Echard, B., Gayton,
N. and Lemaire, M. (2011), "AK-MCS: an active learning reliability method
combining Kriging and Monte Carlo Simulation", Structural Safety 33(2):145-154.
This is a simplified, non-adaptive version of that idea: a one-shot design, no
active learning loop.

The surrogate is used in two distinct ways and the difference matters:

1. **Surrogate as the estimator** (:func:`surrogate_probability`): the failure
   probability is computed from the surrogate's own predicted limit state.
   This is *biased by the surrogate's error* and the bias is not bounded by
   anything the surrogate reports.  It is implemented so the bias can be
   measured, and it is reported with a 2-sigma band from the Gaussian-process
   posterior.
2. **Surrogate as the proposal** (:func:`surrogate_guided_importance_sampling`):
   the surrogate supplies only the mean shift, and every accepted sample is
   evaluated on the **true** limit state.  This estimator is unbiased for any
   surrogate, good or bad; a bad surrogate costs variance, not correctness.
   This is the configuration a verification campaign should use and the one
   benchmarked against the analytic baseline.

Uncertainty output
------------------
The Gaussian process supplies a posterior standard deviation at every point.
:func:`surrogate_design_point` uses it to produce a confidence output rather
than a point estimate: the design point is located three times, on the
posterior mean and on the mean plus and minus ``k`` posterior standard
deviations, giving a reliability-index band and hence a failure-probability
band.  :attr:`SurrogateFit.straddle_fraction` additionally reports, for a given
sample, the fraction of points where the posterior band crosses zero, i.e.
where the surrogate does not know which side of the limit state it is on.

Information the surrogate is given
----------------------------------
The training design is radial: directions uniform on the unit sphere, radii
uniform on ``[0, radius_scale * |Phi^-1(p_prior)|]``.  This uses a declared
prior order of magnitude for the failure probability, ``p_prior``.  That is a
real advantage handed to the surrogate and it is stated here rather than
hidden: without it, a design drawn from the input distribution itself would
contain no point anywhere near a ``1e-4`` failure boundary and the surrogate
would be useless.  The analytic baseline is given the exact design point, so
the comparison is not unfair in the surrogate's disfavour either; both sides
get prior information and the comparison is of what they do with it.

Compute
-------
A Gaussian-process fit is ``O(n_train^3)``; ``n_train`` is capped at 2000 and
the measured fit times on this container are in
validation/validate_surrogate.py.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from .estimate import RareEventEstimate
from .limitstates import LimitState
from .tilting import MeanShiftTilt, find_design_point_radial, importance_sampling

__all__ = [
    "SurrogateFit",
    "SurrogateLimitState",
    "fit_surrogate",
    "radial_design",
    "surrogate_design_point",
    "surrogate_guided_importance_sampling",
    "surrogate_probability",
]

MAX_TRAIN = 2000


def radial_design(
    dimension: int,
    n_train: int,
    p_prior: float = 1e-4,
    radius_scale: float = 1.4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Radial training design in standard normal space.

    Parameters
    ----------
    dimension
        Input dimension, ``>= 1``.
    n_train
        Number of design points, ``1 <= n_train <= 2000``.
    p_prior
        Declared prior order of magnitude for the failure probability, in
        ``(0, 0.5)``.  Sets the maximum design radius through
        ``|Phi^-1(p_prior)|``.
    radius_scale
        Multiplier on that radius, dimensionless, ``> 0``.
    rng
        Random generator.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_train, dimension)``, in standard-normal units.
    """
    if dimension < 1:
        raise ValueError(f"dimension must be at least 1, got {dimension}")
    if not 1 <= n_train <= MAX_TRAIN:
        raise ValueError(f"n_train must lie in [1, {MAX_TRAIN}], got {n_train}")
    if not 0.0 < p_prior < 0.5:
        raise ValueError(f"p_prior must lie strictly in (0, 0.5), got {p_prior}")
    if radius_scale <= 0.0:
        raise ValueError(f"radius_scale must be positive, got {radius_scale}")
    generator = np.random.default_rng(0) if rng is None else rng
    r_max = radius_scale * abs(float(stats.norm.ppf(p_prior)))
    directions = generator.standard_normal((n_train, dimension))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    directions /= norms
    radii = r_max * generator.random((n_train, 1))
    return directions * radii


@dataclass
class SurrogateFit:
    """A fitted Gaussian-process surrogate of a limit-state function.

    Attributes
    ----------
    model
        The fitted ``GaussianProcessRegressor``.
    dimension
        Input dimension.
    n_train
        Training-set size, equal to the number of true limit-state evaluations
        spent before any sampling begins.
    x_train, g_train
        Training design and the true limit-state values at it.
    fit_seconds
        Measured wall-clock fit time on this container.
    p_prior, radius_scale
        The prior information the design was built from.
    """

    model: GaussianProcessRegressor
    dimension: int
    n_train: int
    x_train: np.ndarray
    g_train: np.ndarray
    fit_seconds: float
    p_prior: float
    radius_scale: float
    diagnostics: dict[str, float] = field(default_factory=dict)

    def predict(
        self, x: np.ndarray, return_std: bool = False, chunk: int = 20_000
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Posterior mean (and optionally standard deviation) of ``g``.

        Parameters
        ----------
        x
            Shape ``(n, dimension)``.
        return_std
            Also return the posterior standard deviation, in ``g`` units.
        chunk
            Rows per prediction chunk, to bound peak memory.

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
        """
        arr = np.atleast_2d(np.asarray(x, dtype=float))
        if arr.shape[1] != self.dimension:
            raise ValueError(
                f"x must have {self.dimension} columns, got shape {arr.shape}"
            )
        means: list[np.ndarray] = []
        stds: list[np.ndarray] = []
        for start in range(0, arr.shape[0], chunk):
            block = arr[start : start + chunk]
            if return_std:
                mean, std = self.model.predict(block, return_std=True)
                stds.append(np.asarray(std, dtype=float))
            else:
                mean = self.model.predict(block)
            means.append(np.asarray(mean, dtype=float))
        mean_all = np.concatenate(means)
        if return_std:
            return mean_all, np.concatenate(stds)
        return mean_all

    def straddle_fraction(self, x: np.ndarray, k: float = 2.0) -> float:
        """Fraction of points whose posterior band ``mean +- k std`` contains 0.

        This is the surrogate's own statement of where it does not know which
        side of the limit state it is on.  Dimensionless, in ``[0, 1]``.
        """
        if k <= 0.0:
            raise ValueError(f"k must be positive, got {k}")
        mean, std = self.predict(x, return_std=True)
        return float(np.mean(np.abs(mean) <= k * std))

    def surrogate_limit_state(self, sigma_offset: float = 0.0) -> SurrogateLimitState:
        """Wrap the posterior as a limit state, optionally offset in sigma units.

        ``sigma_offset = +2`` gives the optimistic boundary (surrogate plus two
        posterior standard deviations, so less failure) and ``-2`` the
        pessimistic one.
        """
        return SurrogateLimitState(self, sigma_offset=sigma_offset)


class SurrogateLimitState(LimitState):
    """The surrogate posterior presented as a limit state.

    ``g_hat(x) = mean(x) + sigma_offset * std(x)``.  Used for the pure-surrogate
    probability estimate and for locating the design point of the posterior
    band.  It has no reference probability: any probability computed from it is
    an estimate of the *surrogate's* failure probability, not the model's.
    """

    reference_kind = "none"

    def __init__(self, fit: SurrogateFit, sigma_offset: float = 0.0) -> None:
        self.fit = fit
        self.sigma_offset = float(sigma_offset)
        self.name = f"surrogate[{self.sigma_offset:+.1f}sigma]"
        self.dimension = fit.dimension

    def g(self, x: np.ndarray) -> np.ndarray:
        if self.sigma_offset == 0.0:
            return np.asarray(self.fit.predict(x), dtype=float)
        mean, std = self.fit.predict(x, return_std=True)
        return mean + self.sigma_offset * std

    def analytic_probability(self) -> float:
        raise NotImplementedError(
            "a surrogate limit state has no reference probability; it is a fitted "
            "approximation, and its failure probability is not the model's"
        )

    def design_point(self) -> np.ndarray:
        raise NotImplementedError(
            "use surrogate_design_point(), which also returns the posterior band"
        )


def fit_surrogate(
    limit_state: LimitState,
    n_train: int = 300,
    p_prior: float = 1e-4,
    radius_scale: float = 1.4,
    rng: np.random.Generator | None = None,
    n_restarts: int = 0,
) -> SurrogateFit:
    """Fit a Gaussian-process surrogate to ``n_train`` true limit-state values.

    Parameters
    ----------
    limit_state
        The true limit state.  Exactly ``n_train`` evaluations are spent.
    n_train
        Design size, ``1 <= n_train <= 2000``.  The fit is ``O(n_train^3)``.
    p_prior, radius_scale
        Passed to :func:`radial_design`.
    rng
        Random generator.
    n_restarts
        ``n_restarts_optimizer`` for the marginal-likelihood optimisation.
        Default 0, i.e. one L-BFGS run from the initial kernel.  Raising it to
        1 or 2 was measured in validation/validate_surrogate.py and changed the
        recovered reliability index in the fourth decimal place or not at all,
        at two to three times the fit cost, so the compute budget is spent on
        samples instead.

    Returns
    -------
    SurrogateFit

    Notes
    -----
    Kernel: ``ConstantKernel * RBF + WhiteKernel``, with anisotropic RBF length
    scales.  The limit states here are deterministic, so the marginal
    likelihood drives the noise term to its lower bound; that bound is set to
    ``1e-8 * var(g_train)`` rather than to zero because a strictly noise-free
    Gaussian process produces numerically negative posterior variances on this
    scikit-learn version, which then propagate into the uncertainty output.
    The fitted noise level is reported in
    ``SurrogateFit.diagnostics["noise_level"]`` so it can be checked against the
    ``g`` scale rather than assumed negligible.

    For an exactly linear limit state the RBF length scale saturates at its
    upper bound, because a linear function is the infinite-length-scale limit
    of an RBF; scikit-learn emits a ``ConvergenceWarning`` saying so.  That is
    the correct diagnosis, not a tuning failure, and it is one structural
    reason the surrogate has nothing to add on the smooth instance.
    """
    if not 1 <= n_train <= MAX_TRAIN:
        raise ValueError(f"n_train must lie in [1, {MAX_TRAIN}], got {n_train}")
    generator = np.random.default_rng(0) if rng is None else rng
    x_train = radial_design(
        limit_state.dimension,
        n_train,
        p_prior=p_prior,
        radius_scale=radius_scale,
        rng=generator,
    )
    g_train = np.asarray(limit_state.g(x_train), dtype=float)
    scale = float(np.std(g_train)) or 1.0
    kernel = ConstantKernel(scale**2, (1e-6, 1e10)) * RBF(
        np.ones(limit_state.dimension), (1e-2, 1e4)
    ) + WhiteKernel(1e-4 * scale**2, (1e-8 * scale**2, 1e1 * scale**2))
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=int(n_restarts),
        random_state=int(generator.integers(0, 2**31 - 1)),
    )
    start = time.perf_counter()
    model.fit(x_train, g_train)
    fit_seconds = time.perf_counter() - start
    residual = g_train - np.asarray(model.predict(x_train), dtype=float)
    return SurrogateFit(
        model=model,
        dimension=limit_state.dimension,
        n_train=int(n_train),
        x_train=x_train,
        g_train=g_train,
        fit_seconds=fit_seconds,
        p_prior=float(p_prior),
        radius_scale=float(radius_scale),
        diagnostics={
            "train_rmse": float(np.sqrt(np.mean(residual**2))),
            "g_train_std": scale,
            "noise_level": float(model.kernel_.k2.noise_level),
            "log_marginal_likelihood": float(
                model.log_marginal_likelihood_value_
            ),
        },
    )


@dataclass(frozen=True)
class SurrogateDesignPoint:
    """Design point located on a surrogate, with its posterior band.

    Attributes
    ----------
    point
        Design point of the posterior mean, shape ``(d,)``, standard-normal
        units.
    beta
        ``|point|``, the surrogate's reliability index, dimensionless.
    beta_lower, beta_upper
        Reliability indices of the design points located on the posterior mean
        minus and plus ``k_sigma`` posterior standard deviations.  Not
        necessarily ordered as named if the surrogate is poor; they are
        reported as found.
    probability, probability_lower, probability_upper
        ``Phi(-beta)`` for each of the three, i.e. the surrogate's
        first-order failure probability and its band.  Dimensionless.
    k_sigma
        Band width in posterior standard deviations.
    converged
        Whether the constrained optimisation met its constraint tolerance on
        the posterior mean.
    """

    point: np.ndarray
    beta: float
    beta_lower: float
    beta_upper: float
    probability: float
    probability_lower: float
    probability_upper: float
    k_sigma: float
    converged: bool

    def describe(self) -> str:
        """One-line summary. Contains no I/O."""
        return (
            f"surrogate design point |x|={self.beta:.4f} "
            f"(band {self.beta_lower:.4f}..{self.beta_upper:.4f}) "
            f"p_first_order={self.probability:.4e} "
            f"[{min(self.probability_lower, self.probability_upper):.4e}, "
            f"{max(self.probability_lower, self.probability_upper):.4e}] "
            f"converged={self.converged}"
        )


def surrogate_design_point(
    fit: SurrogateFit,
    k_sigma: float = 2.0,
    n_directions: int = 192,
    rng: np.random.Generator | None = None,
) -> SurrogateDesignPoint:
    """Locate the design point on the surrogate and band it with the posterior.

    Spends no true limit-state evaluations.

    Parameters
    ----------
    fit
        Fitted surrogate.
    k_sigma
        Band half-width in posterior standard deviations, ``> 0``.
    n_directions
        Directions used by the vectorised ray search
        (:func:`rareverify.tilting.find_design_point_radial`).
    rng
        Random generator for the directions.

    Returns
    -------
    SurrogateDesignPoint
    """
    if k_sigma <= 0.0:
        raise ValueError(f"k_sigma must be positive, got {k_sigma}")
    generator = np.random.default_rng(0) if rng is None else rng
    mean_point, converged = find_design_point_radial(
        fit.surrogate_limit_state(0.0).g,
        fit.dimension,
        n_directions=n_directions,
        rng=generator,
    )
    betas = []
    for offset in (-k_sigma, k_sigma):
        point, _ = find_design_point_radial(
            fit.surrogate_limit_state(offset).g,
            fit.dimension,
            n_directions=64,
            rng=generator,
        )
        betas.append(float(np.linalg.norm(point)))
    beta = float(np.linalg.norm(mean_point))
    return SurrogateDesignPoint(
        point=mean_point,
        beta=beta,
        beta_lower=betas[0],
        beta_upper=betas[1],
        probability=float(stats.norm.cdf(-beta)),
        probability_lower=float(stats.norm.cdf(-betas[0])),
        probability_upper=float(stats.norm.cdf(-betas[1])),
        k_sigma=float(k_sigma),
        converged=converged,
    )


def surrogate_probability(
    fit: SurrogateFit,
    n_samples: int = 50_000,
    tilt: MeanShiftTilt | None = None,
    rng: np.random.Generator | None = None,
) -> RareEventEstimate:
    """Failure probability of the *surrogate*, by importance sampling on it.

    This estimator never touches the true limit state after training, so its
    ``true_evaluations`` is just ``fit.n_train``.  It is **biased as an estimate
    of the model's failure probability** by exactly the surrogate's error near
    the limit state, and that bias is what validation/validate_surrogate.py
    measures.  Provided so the bias is on the record, not as a recommended
    estimator.
    """
    generator = np.random.default_rng(0) if rng is None else rng
    surrogate = fit.surrogate_limit_state(0.0)
    if tilt is None:
        design = surrogate_design_point(fit, rng=generator)
        tilt = MeanShiftTilt(design.point, label="surrogate-design-point")
    est = importance_sampling(
        surrogate, tilt, n_samples, rng=generator, batch_size=20_000
    )
    diagnostics = dict(est.diagnostics)
    diagnostics["n_train"] = float(fit.n_train)
    return RareEventEstimate(
        method="surrogate-only",
        estimate=est.estimate,
        standard_error=est.standard_error,
        n_samples=est.n_samples,
        true_evaluations=fit.n_train,
        n_failures=est.n_failures,
        is_binomial=False,
        effective_sample_size=est.effective_sample_size,
        wall_seconds=est.wall_seconds,
        standard_error_kind="weighted-clt",
        contributions=est.contributions,
        diagnostics=diagnostics,
    )


def surrogate_guided_importance_sampling(
    limit_state: LimitState,
    fit: SurrogateFit,
    n_samples: int,
    rng: np.random.Generator | None = None,
    k_sigma: float = 2.0,
    batch_size: int = 250_000,
) -> tuple[RareEventEstimate, SurrogateDesignPoint]:
    """Importance sampling on the TRUE limit state, tilted by the surrogate.

    The surrogate supplies ``theta`` only.  Every sample is evaluated on the
    true limit state, so the estimator is unbiased whatever the surrogate does;
    a poor surrogate shows up as variance, not as bias.

    ``true_evaluations`` is ``fit.n_train + n_samples``, which is the accounting
    that makes the comparison against the analytic baseline fair: the training
    budget is spent on the real simulator.

    Returns
    -------
    tuple
        ``(estimate, design_point)``.  The design point carries the surrogate's
        uncertainty band, which is the AI component's confidence output.
    """
    generator = np.random.default_rng(0) if rng is None else rng
    design = surrogate_design_point(fit, k_sigma=k_sigma, rng=generator)
    tilt = MeanShiftTilt(design.point, label="surrogate-design-point")
    est = importance_sampling(
        limit_state, tilt, n_samples, rng=generator, batch_size=batch_size
    )
    diagnostics = dict(est.diagnostics)
    diagnostics.update(
        {
            "n_train": float(fit.n_train),
            "surrogate_beta": design.beta,
            "surrogate_beta_lower": design.beta_lower,
            "surrogate_beta_upper": design.beta_upper,
            "surrogate_fit_seconds": fit.fit_seconds,
            "surrogate_train_rmse": fit.diagnostics.get("train_rmse", math.nan),
        }
    )
    guided = RareEventEstimate(
        method="surrogate-guided-is",
        estimate=est.estimate,
        standard_error=est.standard_error,
        n_samples=est.n_samples,
        true_evaluations=fit.n_train + est.n_samples,
        n_failures=est.n_failures,
        is_binomial=False,
        effective_sample_size=est.effective_sample_size,
        wall_seconds=est.wall_seconds + fit.fit_seconds,
        standard_error_kind="weighted-clt",
        contributions=est.contributions,
        diagnostics=diagnostics,
    )
    return guided, design
