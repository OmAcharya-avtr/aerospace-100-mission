"""The three baselines, in the order the specification requires them built.

1. :func:`mean_plus_margin` -- the mean-plus-margin rule. The thing being
   refuted, implemented deliberately because it is what the practice does.
2. :func:`gpd_tail` with :func:`threshold_stability` and
   :func:`choose_threshold` -- extreme value theory, peaks over threshold.
3. :func:`counted_rate` -- the directly counted miss rate. The ground truth.

References
----------
Pickands 1975, "Statistical Inference Using Extreme Order Statistics", Annals of
Statistics 3(1):119-131 -- the limiting generalised Pareto form for exceedances
over a high threshold.

Balkema & de Haan 1974, "Residual Life Time at Great Age", Annals of Probability
2(5):792-804 -- the companion result.

Coles 2001, "An Introduction to Statistical Modeling of Extreme Values",
Springer, chapter 4 -- the peaks-over-threshold model, the return-level formula
used in :func:`gpd_tail`, and the threshold-stability diagnostic in
:func:`threshold_stability`, including the **modified scale**
``beta* = beta - xi*u``, which is the quantity that is constant in ``u`` above a
threshold at which the GPD approximation holds (Coles 2001 eq. 4.10).

Validity range. The GPD limit is asymptotic in the threshold. Below a threshold
high enough for the limit to bite, the fit is biased; above it, the exceedance
count collapses and the variance explodes. That trade-off is the reason the
stability diagnostic is part of the deliverable and not an optional extra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .accounting import DeadlineAccount, account
from .trace import JobTrace

#: Candidate threshold quantiles for :func:`threshold_stability`, declared here
#: rather than chosen after looking at a fit.
THRESHOLD_QUANTILES: tuple[float, ...] = tuple(round(0.80 + 0.01 * i, 2) for i in range(20))

#: Minimum exceedance count for a candidate threshold to be admissible. Below
#: this the shape estimate's standard error is of the same order as the estimate.
MIN_EXCEEDANCES = 30

#: Number of consecutive candidate thresholds the declared choice rule averages
#: the shape estimate's variability over.
STABILITY_WINDOW = 5


@dataclass(frozen=True)
class MeanMarginResult:
    """The mean-plus-margin rule's own output.

    Attributes
    ----------
    mean_s, sd_s
        Sample mean and sample standard deviation (``ddof=1``) of the measured
        response times, seconds.
    k_to_deadline
        The ``k`` for which ``mean + k*sd = D``, dimensionless. This is the
        sigma level a practitioner's own margin rule corresponds to on this
        data. It is negative when the mean response already exceeds the
        deadline.
    probability
        ``P(resp > D)`` under a Gaussian with that mean and that standard
        deviation, dimensionless. Computed as ``norm.sf(k_to_deadline)``, which
        is accurate in the far tail where ``1 - norm.cdf`` is not.
    n
        Number of response-time samples the mean and sd came from.

    Notes
    -----
    The input is the response times of the **executed** jobs. A dropped release
    has no response time, so the rule never sees one. Its probability is
    therefore a statement about the executed subset and is compared against a
    counted rate over **all** releases, which is the comparison the practice
    makes and the one this product measures the error of.
    """

    mean_s: float
    sd_s: float
    k_to_deadline: float
    probability: float
    deadline_s: float
    n: int


def mean_plus_margin(response_s: np.ndarray, deadline_s: float) -> MeanMarginResult:
    """Baseline 1: the Gaussian mean-plus-margin rule.

    Parameters
    ----------
    response_s
        Measured response times, seconds. At least two samples, all finite and
        non-negative.
    deadline_s
        Implicit deadline ``D``, seconds, positive and finite.

    Returns
    -------
    MeanMarginResult

    Raises
    ------
    ValueError
        On fewer than two samples, a non-finite or negative response time, a
        non-positive deadline, or a zero sample standard deviation (for which
        ``k`` is not defined).
    """
    values = np.asarray(response_s, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"response_s must be one-dimensional, got shape {values.shape}")
    if values.size < 2:
        raise ValueError(
            f"mean_plus_margin needs at least 2 response samples, got {values.size}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("response_s contains a non-finite value")
    if np.any(values < 0.0):
        raise ValueError("response_s contains a negative value")
    if not np.isfinite(deadline_s) or deadline_s <= 0.0:
        raise ValueError(f"deadline_s must be finite and > 0, got {deadline_s!r}")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    if sd <= 0.0:
        raise ValueError(
            "response_s has zero sample standard deviation, so the k for which "
            "mean + k*sd = D is not defined"
        )
    k = (deadline_s - mean) / sd
    return MeanMarginResult(
        mean_s=mean,
        sd_s=sd,
        k_to_deadline=float(k),
        probability=float(stats.norm.sf(k)),
        deadline_s=float(deadline_s),
        n=int(values.size),
    )


@dataclass(frozen=True)
class GpdFit:
    """One peaks-over-threshold fit at a fixed threshold.

    Attributes
    ----------
    threshold_s
        ``u``, seconds.
    quantile
        The quantile of the response-time sample ``u`` was taken at.
    n_total, n_exceedances
        Sample size and the number of values strictly above ``u``.
    zeta
        Measured exceedance rate ``n_exceedances / n_total``, dimensionless.
    shape, scale
        GPD shape ``xi`` (dimensionless) and scale ``beta`` (seconds), fitted by
        maximum likelihood with the location fixed at zero.
    modified_scale
        ``beta - xi*u``, seconds. Coles 2001 eq. 4.10: constant in ``u`` above a
        threshold at which the GPD approximation holds.
    probability
        ``zeta * (1 + xi*(D-u)/beta)**(-1/xi)``, dimensionless, clipped to
        ``[0, 1]``. Zero when the fitted upper endpoint ``u - beta/xi`` lies
        below ``D`` for ``xi < 0``, which is the model stating that ``D`` cannot
        be exceeded; that is a model statement, not a measurement, and
        ``endpoint_below_deadline`` flags it.
    endpoint_below_deadline
        True when ``xi < 0`` and the fitted finite upper endpoint is below
        ``D``.
    """

    threshold_s: float
    quantile: float
    n_total: int
    n_exceedances: int
    zeta: float
    shape: float
    scale: float
    modified_scale: float
    probability: float
    endpoint_below_deadline: bool


def _gpd_tail_probability(
    zeta: float, shape: float, scale: float, threshold: float, deadline: float
) -> tuple[float, bool]:
    """``zeta * (1 + xi*(D-u)/beta)**(-1/xi)``, with the xi -> 0 limit handled."""
    excess = deadline - threshold
    if excess <= 0.0:
        return min(1.0, max(0.0, zeta)), False
    if abs(shape) < 1e-12:
        return float(np.clip(zeta * np.exp(-excess / scale), 0.0, 1.0)), False
    argument = 1.0 + shape * excess / scale
    if argument <= 0.0:
        # xi < 0 with a finite upper endpoint u - beta/xi that lies below D.
        return 0.0, True
    return float(np.clip(zeta * argument ** (-1.0 / shape), 0.0, 1.0)), False


def gpd_tail(response_s: np.ndarray, deadline_s: float, threshold_s: float) -> GpdFit:
    """Baseline 2 at a fixed threshold: GPD peaks-over-threshold tail estimate.

    Parameters
    ----------
    response_s
        Measured response times, seconds.
    deadline_s
        Implicit deadline, seconds.
    threshold_s
        Threshold ``u``, seconds, strictly below the sample maximum.

    Raises
    ------
    ValueError
        On fewer than :data:`MIN_EXCEEDANCES` exceedances, a non-positive
        deadline, or a threshold at or above the sample maximum.
    """
    values = np.asarray(response_s, dtype=np.float64)
    if values.size < 2:
        raise ValueError(f"gpd_tail needs at least 2 samples, got {values.size}")
    if not np.isfinite(deadline_s) or deadline_s <= 0.0:
        raise ValueError(f"deadline_s must be finite and > 0, got {deadline_s!r}")
    if not np.isfinite(threshold_s):
        raise ValueError(f"threshold_s must be finite, got {threshold_s!r}")
    if threshold_s >= values.max():
        raise ValueError(
            f"threshold_s ({threshold_s!r}) is at or above the sample maximum "
            f"({float(values.max())!r}); there are no exceedances to fit"
        )
    excesses = values[values > threshold_s] - threshold_s
    if excesses.size < MIN_EXCEEDANCES:
        raise ValueError(
            f"threshold {threshold_s!r} leaves {excesses.size} exceedance(s), fewer than "
            f"the declared minimum of {MIN_EXCEEDANCES}"
        )
    shape, _, scale = stats.genpareto.fit(excesses, floc=0.0)
    zeta = excesses.size / values.size
    probability, endpoint_below = _gpd_tail_probability(
        zeta, float(shape), float(scale), float(threshold_s), float(deadline_s)
    )
    quantile = float((values <= threshold_s).mean())
    return GpdFit(
        threshold_s=float(threshold_s),
        quantile=quantile,
        n_total=int(values.size),
        n_exceedances=int(excesses.size),
        zeta=float(zeta),
        shape=float(shape),
        scale=float(scale),
        modified_scale=float(scale) - float(shape) * float(threshold_s),
        probability=probability,
        endpoint_below_deadline=endpoint_below,
    )


def threshold_stability(
    response_s: np.ndarray,
    deadline_s: float,
    quantiles: tuple[float, ...] = THRESHOLD_QUANTILES,
) -> list[GpdFit]:
    """Fit the GPD at every admissible candidate threshold.

    The declared candidate grid is :data:`THRESHOLD_QUANTILES`. A candidate is
    admissible when it leaves at least :data:`MIN_EXCEEDANCES` exceedances and
    lies strictly below the sample maximum. Inadmissible candidates are dropped
    silently -- the count of fits returned is itself reported by the caller.

    Returns
    -------
    list[GpdFit]
        In increasing threshold order. Possibly empty.
    """
    values = np.asarray(response_s, dtype=np.float64)
    fits: list[GpdFit] = []
    for quantile in quantiles:
        threshold = float(np.quantile(values, quantile))
        try:
            fits.append(gpd_tail(values, deadline_s, threshold))
        except ValueError:
            continue
    return fits


@dataclass(frozen=True)
class ThresholdChoice:
    """The chosen threshold and the diagnostic it was chosen from.

    Attributes
    ----------
    chosen
        The selected :class:`GpdFit`.
    window
        Number of consecutive candidates the variability was measured over.
    shape_spread
        For each admissible candidate, the standard deviation of the shape
        estimate over the forward window starting there. The chosen candidate is
        the one minimising it.
    rule
        The declared rule, as a string, so a report can print the rule beside
        the number it produced.
    """

    chosen: GpdFit
    window: int
    shape_spread: tuple[float, ...]
    rule: str


#: The declared threshold-choice rule, fixed before any data existed.
THRESHOLD_RULE = (
    "Among candidate thresholds at sample quantiles 0.80..0.99 leaving at least 30 "
    "exceedances, take the one minimising the standard deviation of the GPD shape "
    "estimate over the forward window of 5 consecutive candidates starting there; "
    "ties and short windows go to the lowest threshold, which keeps the exceedance "
    "count highest. The rule is a stability-window rule in the sense of Coles 2001 "
    "section 4.3.1 and is applied by code, not by eye."
)


def choose_threshold(
    fits: list[GpdFit], window: int = STABILITY_WINDOW
) -> ThresholdChoice:
    """Apply the declared threshold-choice rule to a stability diagnostic.

    Raises
    ------
    ValueError
        On an empty ``fits`` list or ``window < 1``.
    """
    if not fits:
        raise ValueError(
            "no admissible candidate threshold: every candidate left fewer than "
            f"{MIN_EXCEEDANCES} exceedances"
        )
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    shapes = np.asarray([fit.shape for fit in fits], dtype=np.float64)
    spread = np.empty(shapes.size, dtype=np.float64)
    for index in range(shapes.size):
        block = shapes[index : index + window]
        spread[index] = float(block.std(ddof=0)) if block.size > 1 else np.inf
    if not np.any(np.isfinite(spread)):
        spread[:] = 0.0
    best = int(np.argmin(spread))
    return ThresholdChoice(
        chosen=fits[best],
        window=window,
        shape_spread=tuple(float(value) for value in spread),
        rule=THRESHOLD_RULE,
    )


def counted_rate(jobs: JobTrace) -> DeadlineAccount:
    """Baseline 3: the directly counted miss rate over all releases.

    A thin alias for :func:`deadlinerisk.accounting.account`, named for the
    baseline it is, so that a reader of the baseline comparison sees three
    functions and not two plus a detour.
    """
    return account(jobs)
