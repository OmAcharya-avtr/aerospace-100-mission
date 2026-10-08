"""The estimator result type and the intervals that are valid for it.

Every estimator in this package returns a :class:`RareEventEstimate`.  The
reason it is a single type is the one modelling error this package is built to
prevent: a Clopper-Pearson or Wilson interval is an interval **for a binomial
proportion**, and the importance-sampling and subset-simulation estimators are
not binomial proportions.  Quoting a Clopper-Pearson interval around a weighted
estimate is not conservative, it is wrong, and the amount by which it is wrong
is not bounded.  :func:`binomial_interval` therefore refuses to produce one for
a weighted estimator, and :func:`interval_for` dispatches to a normal or
bootstrap interval on the weighted mean instead.

The normal interval for a weighted estimator uses the central limit theorem on
the per-sample contributions ``w_i * 1[g(x_i) <= 0]``; its validity depends on
those contributions having a finite variance that the sample actually sees,
which is exactly what fails when the tilt is badly chosen.  The reported
effective sample size is the diagnostic for that failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .intervals import METHODS, ProportionInterval, proportion_interval

__all__ = [
    "EstimateInterval",
    "RareEventEstimate",
    "binomial_interval",
    "bootstrap_interval",
    "counting_noise_floor",
    "interval_for",
    "normal_interval",
]


def counting_noise_floor(p: float, n: int) -> float:
    """Standard deviation ``sqrt(p (1 - p) / n)`` of a crude Monte-Carlo estimate.

    This is the floor against which every known-answer tolerance in this
    package is stated: an estimator cannot be asked to agree with an analytic
    answer more closely than the counting noise of the run that produced it.

    Parameters
    ----------
    p
        True failure probability, dimensionless, in ``[0, 1]``.
    n
        Number of independent runs, ``>= 1``.

    Returns
    -------
    float
        Standard deviation of the crude estimator, dimensionless.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must lie in [0, 1], got {p}")
    return math.sqrt(p * (1.0 - p) / n)


@dataclass(frozen=True)
class RareEventEstimate:
    """The result of one rare-event probability estimation run.

    Attributes
    ----------
    method
        Short estimator name, e.g. ``"crude"``, ``"importance-sampling"``.
    estimate
        Estimated failure probability of the simulated model, dimensionless.
    standard_error
        Estimated standard deviation of :attr:`estimate`, dimensionless.  For
        the crude estimator this is the binomial plug-in
        ``sqrt(p_hat (1 - p_hat) / n)``; for a weighted estimator it is the
        sample standard error of the contributions.
    n_samples
        Number of samples drawn by the estimator.
    true_evaluations
        Number of evaluations of the *true* limit-state function.  This is the
        budget that matters for a surrogate method, because the surrogate's
        training set is paid for in true evaluations.
    n_failures
        Count of drawn samples that landed in the failure region.  For a
        weighted estimator this is a count under the sampling distribution, not
        under the model, so it is a diagnostic and not an estimate.
    is_binomial
        True only when :attr:`estimate` is ``n_failures / n_samples`` from
        independent draws of the model's own input distribution.  Binomial
        confidence intervals are valid only then.
    effective_sample_size
        ``(sum c)^2 / sum c^2`` over the per-sample contributions ``c``; equals
        :attr:`n_failures` for the crude estimator.
    wall_seconds
        Measured wall-clock time of the run on this container.  Reported for
        budgeting only; it is a property of this machine and this
        implementation, not of the method.
    standard_error_kind
        How :attr:`standard_error` was obtained: ``"binomial-plugin"``,
        ``"weighted-clt"``, or
        ``"subset-independence-lower-bound"``.  The last one is **optimistic**:
        subset simulation's samples within a level come from Markov chains and
        are positively correlated, so the independence formula understates the
        true variance.  Any interval built on it is narrower than it should be,
        and the only honest variance for subset simulation in this package
        comes from independent replications
        (:mod:`rareverify.benchmark`).
    contributions
        The non-zero per-sample contributions, kept so that
        :func:`bootstrap_interval` can resample them exactly.  Zero-valued
        contributions are not stored; ``n_samples`` records how many there
        were.
    diagnostics
        Free-form numeric diagnostics, e.g. maximum weight, levels used.
    """

    method: str
    estimate: float
    standard_error: float
    n_samples: int
    true_evaluations: int
    n_failures: int
    is_binomial: bool
    effective_sample_size: float
    wall_seconds: float
    standard_error_kind: str = "weighted-clt"
    contributions: np.ndarray = field(default_factory=lambda: np.zeros(0))
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def coefficient_of_variation(self) -> float:
        """``standard_error / estimate``, or ``inf`` when the estimate is zero."""
        return math.inf if self.estimate == 0.0 else self.standard_error / self.estimate

    def describe(self) -> str:
        """One-line summary. Contains no I/O."""
        return (
            f"{self.method}: p_hat={self.estimate:.6e} se={self.standard_error:.6e} "
            f"cov={self.coefficient_of_variation:.4f} n={self.n_samples} "
            f"evals={self.true_evaluations} ess={self.effective_sample_size:.2f}"
        )


@dataclass(frozen=True)
class EstimateInterval:
    """A confidence interval for a :class:`RareEventEstimate`.

    Attributes
    ----------
    lower, upper
        Limits in probability units on ``[0, 1]``.
    confidence
        Nominal confidence level.
    kind
        ``"clopper-pearson"``, ``"wilson"``, ``"normal-weighted"`` or
        ``"bootstrap-weighted"``.
    rationale
        Why this kind was used, so the interval cannot be quoted without its
        justification.
    """

    lower: float
    upper: float
    confidence: float
    kind: str
    rationale: str

    @property
    def width(self) -> float:
        """Interval width (dimensionless)."""
        return self.upper - self.lower

    def describe(self) -> str:
        """One-line summary. Contains no I/O."""
        return (
            f"{self.kind} {self.confidence:.4g}: "
            f"[{self.lower:.6e}, {self.upper:.6e}] ({self.rationale})"
        )


def binomial_interval(
    est: RareEventEstimate,
    confidence: float = 0.95,
    method: str = "clopper-pearson",
    side: str = "two-sided",
) -> ProportionInterval:
    """Binomial interval for a crude Monte-Carlo estimate.

    Raises
    ------
    ValueError
        If ``est.is_binomial`` is False.  A weighted estimator has no binomial
        count to invert, and substituting its raw failure count produces an
        interval for the wrong quantity.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if not est.is_binomial:
        raise ValueError(
            f"estimator {est.method!r} is not a binomial proportion, so a "
            f"{method} interval is not valid for it; use interval_for() or "
            "bootstrap_interval() instead"
        )
    return proportion_interval(
        est.n_failures, est.n_samples, confidence=confidence, method=method, side=side
    )


def normal_interval(
    estimate: float, standard_error: float, confidence: float = 0.95
) -> tuple[float, float]:
    """Central-limit interval ``estimate +- z * standard_error``, clipped to ``[0, 1]``."""
    if standard_error < 0.0:
        raise ValueError(f"standard_error must be non-negative, got {standard_error}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly in (0, 1), got {confidence}")
    z = float(stats.norm.isf((1.0 - confidence) / 2.0))
    return (
        max(0.0, estimate - z * standard_error),
        min(1.0, estimate + z * standard_error),
    )


def bootstrap_interval(
    est: RareEventEstimate,
    confidence: float = 0.95,
    n_bootstrap: int = 4000,
    rng: np.random.Generator | None = None,
) -> EstimateInterval:
    """Percentile bootstrap interval for a weighted estimator.

    The bootstrap resamples the ``n_samples`` per-sample contributions with
    replacement.  Because the zero contributions are not stored, the resample
    is generated exactly in two stages: the number of non-zero draws is
    binomial with success probability ``m / n``, and those draws are taken
    uniformly with replacement from the stored non-zero contributions.  This is
    identical in distribution to resampling the full contribution vector.

    Parameters
    ----------
    est
        Estimate carrying its non-zero contributions.
    confidence
        Nominal confidence level in ``(0, 1)``.
    n_bootstrap
        Number of bootstrap replicates, ``>= 100``.
    rng
        Random generator; a default-seeded one is created if omitted.

    Returns
    -------
    EstimateInterval
    """
    if n_bootstrap < 100:
        raise ValueError(f"n_bootstrap must be at least 100, got {n_bootstrap}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly in (0, 1), got {confidence}")
    generator = np.random.default_rng(0) if rng is None else rng
    values = np.asarray(est.contributions, dtype=float)
    n = est.n_samples
    m = values.size
    if m == 0:
        return EstimateInterval(
            lower=0.0,
            upper=0.0,
            confidence=confidence,
            kind="bootstrap-weighted",
            rationale="no sample reached the failure region; the bootstrap is degenerate",
        )
    counts = generator.binomial(n, m / n, size=n_bootstrap)
    replicates = np.empty(n_bootstrap, dtype=float)
    for i, count in enumerate(counts):
        if count == 0:
            replicates[i] = 0.0
        else:
            picks = generator.integers(0, m, size=int(count))
            replicates[i] = values[picks].sum() / n
    alpha = 1.0 - confidence
    lower, upper = np.quantile(replicates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return EstimateInterval(
        lower=float(max(0.0, lower)),
        upper=float(min(1.0, upper)),
        confidence=confidence,
        kind="bootstrap-weighted",
        rationale=(
            f"percentile bootstrap over {n_bootstrap} replicates of the weighted "
            f"contributions ({m} non-zero of {n})"
        ),
    )


def interval_for(
    est: RareEventEstimate,
    confidence: float = 0.95,
    binomial_method: str = "clopper-pearson",
) -> EstimateInterval:
    """Return the interval that is valid for this estimator.

    Binomial (Clopper-Pearson or Wilson) when the estimator is a crude count,
    central-limit on the weighted mean otherwise.  The returned
    :attr:`EstimateInterval.rationale` records which case applied.
    """
    if est.is_binomial:
        interval = binomial_interval(est, confidence=confidence, method=binomial_method)
        return EstimateInterval(
            lower=interval.lower,
            upper=interval.upper,
            confidence=confidence,
            kind=binomial_method,
            rationale=f"crude estimator: binomial count {est.n_failures}/{est.n_samples}",
        )
    lower, upper = normal_interval(est.estimate, est.standard_error, confidence=confidence)
    if est.standard_error_kind == "subset-independence-lower-bound":
        rationale = (
            "subset simulation: central-limit interval on a standard error that "
            "assumes independence within levels and is therefore a LOWER BOUND; "
            "this interval is narrower than the truth"
        )
    else:
        rationale = (
            f"weighted estimator: central-limit interval, effective sample size "
            f"{est.effective_sample_size:.2f}"
        )
    return EstimateInterval(
        lower=lower,
        upper=upper,
        confidence=confidence,
        kind="normal-weighted",
        rationale=rationale,
    )
