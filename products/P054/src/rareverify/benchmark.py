"""Measured comparison of estimators by independent replication.

Everything reported here is measured, not derived from a formula.  Each
estimator is run ``n_replications`` times with independent seeds, and the
variance reduction factor is the ratio of empirical variances, put on an equal
*true-evaluation* footing:

    VRF = Var_ref_empirical * (E_ref / E_cand) / Var_cand_empirical

The ``E_ref / E_cand`` factor rescales the reference's variance to the
candidate's evaluation budget.  It uses the fact that the crude estimator's
variance is exactly inversely proportional to its sample count, which is true
by construction and is the only place the scaling is applied.  A surrogate
method's ``E_cand`` includes its training evaluations, which is what makes the
comparison honest: training a surrogate is not free, it is paid for in runs of
the simulator.

``vrf_against_exact_crude`` is the same quantity with the reference variance
taken as the exact ``p (1 - p) / E_cand`` instead of an empirical estimate.  It
is reported as a cross-check because the empirical crude variance is itself
noisy at these probabilities; the two should agree within the replication
noise, and validation/validate_variance_reduction.py reports both.

A variance reduction factor below 1 means the method is **worse than plain
Monte Carlo at the same cost**.  That is a normal outcome for a badly chosen
tilt and the code labels it rather than hiding it.

A variance reduction factor on its own is not a quality metric
---------------------------------------------------------------
This was found the hard way during the build and is recorded in
validation/VALIDATION.md.  A grossly over-tilted importance sampler produces
estimates that are wrong by many orders of magnitude *and* have tiny
variance, because almost every sample that would carry weight is never drawn.
The first version of the tilt sweep reported such a tilt as "better" with a
variance reduction factor of 8.5e11 while its estimate was seven orders of
magnitude below the truth.

:class:`VarianceReduction` therefore also carries a **mean-squared-error**
reduction factor,

    MSERF = [p (1 - p) / E_cand] / [ (mean_cand - p)^2 + Var_cand ]

against the exactly unbiased crude estimator, and
:attr:`VarianceReduction.worse_in_mse` is the field a reader should believe
when the two disagree.  The squared-bias term is itself estimated from ``R``
replications, so it is inflated by about ``Var / R`` even for an unbiased
estimator; with ``R = 30`` that is a 3 % overstatement of the MSE and is not
corrected for.

Two cautions that apply to every number here:

- With ``n_replications`` of order 20-40 the empirical standard deviation
  itself has a relative uncertainty of roughly ``1 / sqrt(2 (R - 1))``, i.e.
  about 11-16 %, so a variance reduction factor is good to roughly a factor of
  1.3 and small differences between methods are not resolved.
- The crude estimator at a probability of ``1e-4`` and a budget of ``1e5`` runs
  sees about ten failures per replication, so its empirical variance is a
  coarse estimate.  This is reported, not corrected.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .estimate import RareEventEstimate

__all__ = [
    "ReplicationSummary",
    "VarianceReduction",
    "replicate",
    "summarise",
    "variance_reduction",
]

EstimatorFn = Callable[[np.random.Generator], RareEventEstimate]


@dataclass(frozen=True)
class ReplicationSummary:
    """Summary of independent replications of one estimator.

    Attributes
    ----------
    label
        Human-readable estimator label, including the tilt where applicable.
    method
        The estimator's own ``method`` field.
    n_replications
        Number of independent runs, each with its own seed.
    mean_estimate, median_estimate
        Across replications, dimensionless probabilities.
    empirical_std
        Sample standard deviation of the estimates across replications
        (``ddof=1``).  This is the honest measure of the estimator's spread.
    empirical_cov
        ``empirical_std / mean_estimate``, dimensionless.
    relative_bias
        ``mean_estimate / reference_probability - 1`` when a reference is
        supplied, else NaN.
    mean_reported_se
        Mean of the estimators' own reported standard errors.
    se_ratio
        ``mean_reported_se / empirical_std``.  A value well below 1 means the
        estimator understates its own uncertainty, which is the failure mode
        that matters most in a verification campaign.
    mean_true_evaluations
        Mean number of true limit-state evaluations per replication.
    mean_wall_seconds
        Mean measured wall-clock time per replication on this container.  A
        machine property, not a method property.
    zero_estimate_count
        Replications that returned exactly zero, where no sample reached the
        failure region.
    estimates
        The raw replication estimates, kept so plots and tests use the same
        numbers as the summary.
    """

    label: str
    method: str
    n_replications: int
    mean_estimate: float
    median_estimate: float
    empirical_std: float
    empirical_cov: float
    relative_bias: float
    mean_reported_se: float
    se_ratio: float
    mean_true_evaluations: float
    mean_wall_seconds: float
    zero_estimate_count: int
    estimates: np.ndarray

    def describe(self) -> str:
        """One-line summary. Contains no I/O."""
        return (
            f"{self.label}: mean={self.mean_estimate:.6e} "
            f"std={self.empirical_std:.6e} cov={self.empirical_cov:.4f} "
            f"relbias={self.relative_bias:+.4f} se/std={self.se_ratio:.3f} "
            f"evals={self.mean_true_evaluations:.0f} R={self.n_replications}"
        )


def replicate(
    estimator: EstimatorFn, n_replications: int, seed: int = 0
) -> list[RareEventEstimate]:
    """Run an estimator ``n_replications`` times with independent seeds.

    Parameters
    ----------
    estimator
        Callable taking a ``numpy.random.Generator`` and returning a
        :class:`~rareverify.estimate.RareEventEstimate`.
    n_replications
        Number of replications, ``>= 2`` (a standard deviation needs two).
    seed
        Base seed; replication ``i`` uses ``numpy.random.default_rng([seed, i])``,
        which gives independent streams and exact reproducibility.

    Returns
    -------
    list of RareEventEstimate
    """
    if n_replications < 2:
        raise ValueError(
            f"n_replications must be at least 2 to estimate a spread, got "
            f"{n_replications}"
        )
    return [
        estimator(np.random.default_rng([int(seed), int(i)]))
        for i in range(int(n_replications))
    ]


def summarise(
    label: str,
    results: list[RareEventEstimate],
    reference_probability: float | None = None,
) -> ReplicationSummary:
    """Reduce a list of replications to a :class:`ReplicationSummary`."""
    if len(results) < 2:
        raise ValueError(f"need at least 2 replications to summarise, got {len(results)}")
    estimates = np.array([r.estimate for r in results], dtype=float)
    reported = np.array([r.standard_error for r in results], dtype=float)
    evaluations = np.array([r.true_evaluations for r in results], dtype=float)
    wall = np.array([r.wall_seconds for r in results], dtype=float)
    std = float(estimates.std(ddof=1))
    mean = float(estimates.mean())
    mean_reported = float(np.nanmean(reported))
    return ReplicationSummary(
        label=label,
        method=results[0].method,
        n_replications=len(results),
        mean_estimate=mean,
        median_estimate=float(np.median(estimates)),
        empirical_std=std,
        empirical_cov=(std / mean) if mean > 0.0 else math.inf,
        relative_bias=(
            (mean / reference_probability - 1.0)
            if reference_probability
            else math.nan
        ),
        mean_reported_se=mean_reported,
        se_ratio=(mean_reported / std) if std > 0.0 else math.inf,
        mean_true_evaluations=float(evaluations.mean()),
        mean_wall_seconds=float(wall.mean()),
        zero_estimate_count=int(np.count_nonzero(estimates == 0.0)),
        estimates=estimates,
    )


@dataclass(frozen=True)
class VarianceReduction:
    """Measured variance reduction of a candidate against a reference.

    Attributes
    ----------
    candidate, reference
        Labels of the two summaries.
    vrf_measured
        ``Var_ref * (E_ref / E_cand) / Var_cand`` from the empirical variances.
        Above 1 the candidate is better at equal evaluation cost; below 1 it is
        worse than the reference.
    vrf_against_exact_crude
        The same with the reference variance replaced by the exact crude
        variance ``p (1 - p) / E_cand``.  Requires the true probability.
    equal_evaluation_basis
        The evaluation count both variances were put on, i.e. ``E_cand``.
    worse_than_reference
        ``vrf_measured < 1``: worse in **variance** at equal cost.
    candidate_mse
        Measured mean squared error ``(mean - p)^2 + Var``, requires the true
        probability; NaN otherwise.
    reference_mse_exact
        ``p (1 - p) / E_cand``, the exact mean squared error of the crude
        estimator at the candidate's evaluation budget, which is also its
        variance because the crude estimator is unbiased.
    mse_reduction_factor
        ``reference_mse_exact / candidate_mse``.  Below 1 the candidate is
        worse than plain Monte Carlo **once its bias is counted**, which is the
        verdict to believe.
    worse_in_mse
        ``mse_reduction_factor < 1``.
    candidate_relative_bias
        Carried through so a large variance reduction cannot be quoted without
        its bias.
    """

    candidate: str
    reference: str
    vrf_measured: float
    vrf_against_exact_crude: float
    equal_evaluation_basis: float
    worse_than_reference: bool
    candidate_mse: float
    reference_mse_exact: float
    mse_reduction_factor: float
    worse_in_mse: bool
    candidate_relative_bias: float

    def describe(self) -> str:
        """One-line summary. Contains no I/O."""
        verdict = "WORSE than" if self.worse_than_reference else "better than"
        mse_verdict = "WORSE in MSE" if self.worse_in_mse else "better in MSE"
        return (
            f"{self.candidate} vs {self.reference}: VRF={self.vrf_measured:.4g} "
            f"({verdict} reference at {self.equal_evaluation_basis:.0f} evaluations), "
            f"VRF_vs_exact_crude={self.vrf_against_exact_crude:.4g}, "
            f"MSERF={self.mse_reduction_factor:.4g} ({mse_verdict}), "
            f"relbias={self.candidate_relative_bias:+.4f}"
        )


def variance_reduction(
    candidate: ReplicationSummary,
    reference: ReplicationSummary,
    true_probability: float | None = None,
) -> VarianceReduction:
    """Compute the measured variance reduction factor of ``candidate``.

    Parameters
    ----------
    candidate
        The method under test.
    reference
        The baseline, normally the crude Monte-Carlo summary.
    true_probability
        Known failure probability, used for ``vrf_against_exact_crude``.  NaN
        is returned for that field if omitted.

    Returns
    -------
    VarianceReduction

    Raises
    ------
    ValueError
        If the candidate's empirical variance is zero, where no ratio exists.
    """
    if candidate.empirical_std <= 0.0:
        raise ValueError(
            f"candidate {candidate.label!r} has zero empirical spread across "
            "replications; a variance ratio is undefined. This happens when "
            "every replication returned the same value, usually zero."
        )
    basis = candidate.mean_true_evaluations
    reference_variance = (
        reference.empirical_std**2 * reference.mean_true_evaluations / basis
    )
    vrf = reference_variance / candidate.empirical_std**2
    if true_probability is None:
        exact = math.nan
        candidate_mse = math.nan
        reference_mse = math.nan
        mserf = math.nan
    else:
        reference_mse = true_probability * (1.0 - true_probability) / basis
        exact = reference_mse / candidate.empirical_std**2
        candidate_mse = (
            candidate.mean_estimate - true_probability
        ) ** 2 + candidate.empirical_std**2
        mserf = reference_mse / candidate_mse if candidate_mse > 0.0 else math.inf
    return VarianceReduction(
        candidate=candidate.label,
        reference=reference.label,
        vrf_measured=float(vrf),
        vrf_against_exact_crude=float(exact),
        equal_evaluation_basis=float(basis),
        worse_than_reference=bool(vrf < 1.0),
        candidate_mse=float(candidate_mse),
        reference_mse_exact=float(reference_mse),
        mse_reduction_factor=float(mserf),
        worse_in_mse=bool(mserf < 1.0) if math.isfinite(mserf) else False,
        candidate_relative_bias=candidate.relative_bias,
    )
