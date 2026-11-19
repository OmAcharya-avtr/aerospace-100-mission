"""Scores for a distributional forecast, because the output is a distribution.

WHY CALIBRATION AND NOT ONLY ERROR
----------------------------------
The product's claim is that a scheduler could book against a quantile of the
shortfall. A forecast whose point estimate is excellent and whose 80 per cent
interval covers 40 per cent of outcomes is useless for that, and a forecast
whose point estimate is mediocre and whose quantiles are honest is usable. So
the primary score here is a proper score over the whole predictive
distribution, and an interval-coverage measure beside it; the point error is
reported too, but it is not the figure the comparison turns on.

THE SCORES
----------
**Continuous ranked probability score (CRPS).** For a predictive distribution
represented by an equally weighted ensemble :math:`x_1,\\dots,x_n` and an
observation :math:`y`,

.. math::
   \\text{CRPS} = \\frac{1}{n}\\sum_i |x_i - y|
                 - \\frac{1}{2n^2}\\sum_i\\sum_j |x_i - x_j|

Source: Gneiting, T. and Raftery, A. E. (2007), "Strictly Proper Scoring
Rules, Prediction, and Estimation", Journal of the American Statistical
Association 102(477), 359-378, equation 21 (the energy-score form of the CRPS
for a sample); the same estimator appears as the decomposed form in Hersbach,
H. (2000), "Decomposition of the Continuous Ranked Probability Score for
Ensemble Prediction Systems", Weather and Forecasting 15(5), 559-570.

Units: the units of the forecast quantity, here **Mbit**. Lower is better.
It is **strictly proper**, so a forecaster cannot improve it by misreporting
its own uncertainty, which is the property that makes it the right score for
this product.

A deterministic forecast is the ensemble :math:`n=1`, for which the second
term vanishes and the CRPS **is** the absolute error. That is why baseline 1,
which is a point, can be scored on the same axis as a distribution without
any special pleading: see Gneiting & Raftery (2007) section 4.2.

**Probability integral transform (PIT) and its calibration error.** The PIT of
an observation under its own predictive distribution is uniform on ``[0, 1]``
if and only if the forecast is probabilistically calibrated (Dawid, A. P.
(1984), "Statistical Theory: The Prequential Approach", JRSS A 147(2),
278-292; Gneiting, T., Balabdaoui, F. and Raftery, A. E. (2007),
"Probabilistic Forecasts, Calibration and Sharpness", JRSS B 69(2), 243-268).
:func:`pit_calibration_error` returns the Kolmogorov distance between the
empirical PIT distribution and the uniform -- the supremum absolute difference
of the two cumulative distribution functions. Dimensionless, in ``[0, 1]``,
lower is better, 0 is perfect calibration of the PIT.

**Interval coverage.** The empirical fraction of observations inside a nominal
interval. Dimensionless. The target is the nominal level, and *either*
direction away from it is a defect: under-coverage is over-confidence, and
over-coverage is a forecast too wide to schedule against.

TIES
----
``pit_value`` resolves ties by counting half the equal members, which is the
mid-rank convention. Without it a forecast with atoms -- and the analytic
baseline has one, at zero loss -- produces a PIT pinned at 0 or 1 and the
calibration error becomes an artefact of the atom.
"""

from __future__ import annotations

import numpy as np


def crps_ensemble(ensemble_mbit: np.ndarray, observation_mbit: float) -> float:
    """CRPS of an equally weighted ensemble against one observation, Mbit.

    See the module docstring for the estimator and its source. An ensemble of
    one member returns the absolute error exactly.
    """
    members = np.asarray(ensemble_mbit, dtype=float).ravel()
    if members.size == 0:
        raise ValueError("ensemble_mbit must contain at least one member")
    if not np.all(np.isfinite(members)):
        raise ValueError("ensemble_mbit must be finite")
    if not np.isfinite(observation_mbit):
        raise ValueError(f"observation_mbit must be finite, got {observation_mbit!r}")
    count = members.size
    first = float(np.mean(np.abs(members - observation_mbit)))
    if count == 1:
        return first
    # The pairwise term is computed from the sorted members in O(n log n):
    # sum_i sum_j |x_i - x_j| = 2 * sum_k (2k - n + 1) x_(k) for sorted x.
    ordered = np.sort(members)
    weights = 2.0 * np.arange(count) - count + 1.0
    pairwise = 2.0 * float(np.dot(weights, ordered))
    return first - pairwise / (2.0 * count**2)


def crps_mean(ensembles_mbit: np.ndarray, observations_mbit: np.ndarray) -> float:
    """Mean CRPS over rows of an ``(n_observations, n_members)`` ensemble, Mbit."""
    ensembles = np.atleast_2d(np.asarray(ensembles_mbit, dtype=float))
    observations = np.asarray(observations_mbit, dtype=float).ravel()
    if ensembles.shape[0] != observations.size:
        raise ValueError(
            f"ensembles has {ensembles.shape[0]} rows but there are {observations.size} "
            "observations"
        )
    scores = [
        crps_ensemble(ensembles[index], float(observations[index]))
        for index in range(len(ensembles))
    ]
    return float(np.mean(scores))


def pit_value(ensemble_mbit: np.ndarray, observation_mbit: float) -> float:
    """PIT of one observation under an ensemble forecast, dimensionless.

    Mid-rank convention: ``(below + 0.5 * equal) / n``. See the module
    docstring for why the tie handling is not optional here.
    """
    members = np.asarray(ensemble_mbit, dtype=float).ravel()
    if members.size == 0:
        raise ValueError("ensemble_mbit must contain at least one member")
    below = float(np.count_nonzero(members < observation_mbit))
    equal = float(np.count_nonzero(members == observation_mbit))
    return (below + 0.5 * equal) / members.size


def pit_values(ensembles_mbit: np.ndarray, observations_mbit: np.ndarray) -> np.ndarray:
    """PIT of every observation under its own ensemble, dimensionless."""
    ensembles = np.atleast_2d(np.asarray(ensembles_mbit, dtype=float))
    observations = np.asarray(observations_mbit, dtype=float).ravel()
    if ensembles.shape[0] != observations.size:
        raise ValueError(
            f"ensembles has {ensembles.shape[0]} rows but there are {observations.size} "
            "observations"
        )
    return np.asarray(
        [pit_value(ensembles[i], float(observations[i])) for i in range(len(ensembles))],
        dtype=float,
    )


def pit_calibration_error(pit: np.ndarray) -> float:
    """Kolmogorov distance of the empirical PIT from the uniform, dimensionless.

    ``sup_u |F_hat(u) - u|`` over the observed PIT values, computed exactly at
    the order statistics as ``max_k max(k/n - u_(k), u_(k) - (k-1)/n)``, which
    is where the supremum of a step function against a line is attained.
    """
    values = np.sort(np.asarray(pit, dtype=float).ravel())
    if values.size == 0:
        raise ValueError("pit must contain at least one value")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("pit values must lie in [0, 1]")
    count = values.size
    upper = np.arange(1, count + 1) / count - values
    lower = values - np.arange(0, count) / count
    return float(max(np.max(upper), np.max(lower)))


def interval_coverage(
    lower_mbit: np.ndarray, upper_mbit: np.ndarray, observations_mbit: np.ndarray
) -> float:
    """Fraction of observations inside ``[lower, upper]``, dimensionless."""
    low = np.asarray(lower_mbit, dtype=float).ravel()
    high = np.asarray(upper_mbit, dtype=float).ravel()
    values = np.asarray(observations_mbit, dtype=float).ravel()
    if not (low.size == high.size == values.size):
        raise ValueError(
            f"lower, upper and observations must align, got {low.size}, {high.size}, {values.size}"
        )
    if np.any(high < low):
        raise ValueError("upper_mbit must not fall below lower_mbit anywhere")
    return float(np.mean((values >= low) & (values <= high)))


def mean_absolute_error(prediction_mbit: np.ndarray, observation_mbit: np.ndarray) -> float:
    """Mean absolute error, Mbit."""
    prediction = np.asarray(prediction_mbit, dtype=float).ravel()
    observation = np.asarray(observation_mbit, dtype=float).ravel()
    if prediction.size != observation.size:
        raise ValueError(
            f"prediction and observation must align, got {prediction.size} and {observation.size}"
        )
    return float(np.mean(np.abs(prediction - observation)))


def root_mean_square_error(prediction_mbit: np.ndarray, observation_mbit: np.ndarray) -> float:
    """Root mean square error, Mbit."""
    prediction = np.asarray(prediction_mbit, dtype=float).ravel()
    observation = np.asarray(observation_mbit, dtype=float).ravel()
    if prediction.size != observation.size:
        raise ValueError(
            f"prediction and observation must align, got {prediction.size} and {observation.size}"
        )
    return float(np.sqrt(np.mean((prediction - observation) ** 2)))
