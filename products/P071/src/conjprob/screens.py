"""Analytic screens, implemented before the learned model and scored against it.

Three closed-form screens, in increasing order of the information they use.
Each returns a *score* in which larger means more likely to exceed the Pc
threshold, so that all screens and the learned classifier can be compared on one
convention and one receiver-operating curve.

* :func:`miss_distance_score` -- the screen everyone starts with: flag anything
  closer than a fixed distance. Uses the miss distance and nothing else.
* :func:`mahalanobis_score` -- the obvious better screen: flag anything whose
  miss is improbable given the covariance. Uses the encounter-plane projection.
* :func:`small_body_score` -- the closed-form leading-order Pc itself,
  ``log10(R^2 / (2 sigma_x sigma_y) exp(-M^2 / 2))``. One line, and it uses
  exactly the features the learned classifier is given, which is what makes it
  the honest comparison.

The third screen is not in the product specification's list of baselines. It was
added because leaving it out would have made the learned model look better than
it is: a screen that is a monotone function of an approximation to the target is
the strongest cheap screen available, and a learned model that cannot beat it
has not earned its place.

Units: metres in; scores dimensionless.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .features import FEATURE_NAMES

_LOG10_MISS = FEATURE_NAMES.index("log10_miss_distance_m")
_LOG10_RADIUS = FEATURE_NAMES.index("log10_radius_m")
_LOG10_SIGMA_X = FEATURE_NAMES.index("log10_sigma_x_m")
_LOG10_SIGMA_Y = FEATURE_NAMES.index("log10_sigma_y_m")
_MAHALANOBIS = FEATURE_NAMES.index("mahalanobis")

SCREEN_NAMES: tuple[str, ...] = ("miss_distance", "mahalanobis", "small_body")
"""Analytic screen keys, in increasing order of information used."""


def _as_matrix(features: ArrayLike) -> NDArray[np.float64]:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"features must have shape (n, {len(FEATURE_NAMES)}), got {matrix.shape}"
        )
    return matrix


def miss_distance_score(features: ArrayLike) -> NDArray[np.float64]:
    """``-log10(miss distance / m)``. Larger means closer, hence more suspicious.

    Thresholding this score at ``-log10(d0)`` is exactly the operational
    practice of flagging every conjunction inside ``d0`` metres.
    """
    return -_as_matrix(features)[:, _LOG10_MISS]


def mahalanobis_score(features: ArrayLike) -> NDArray[np.float64]:
    """``-M``, the negated Mahalanobis distance in the encounter plane.

    Thresholding this score is the standard covariance-aware screen. It is
    invariant to the hard-body radius and to the absolute covariance scale,
    which is precisely its weakness and is what the measurement in
    ``validation/validate_screening_output.txt`` quantifies.
    """
    return -_as_matrix(features)[:, _MAHALANOBIS]


def small_body_score(features: ArrayLike) -> NDArray[np.float64]:
    """``log10`` of the leading-order Pc: a closed-form screen in one line.

    ``log10(R^2 / (2 sigma_x sigma_y)) - M^2 / (2 ln 10)``, computed from the
    same feature vector the learned classifier receives. Monotone in the
    leading-order Pc, so thresholding it is thresholding that approximation.
    """
    matrix = _as_matrix(features)
    log_area = (
        2.0 * matrix[:, _LOG10_RADIUS]
        - matrix[:, _LOG10_SIGMA_X]
        - matrix[:, _LOG10_SIGMA_Y]
        - np.log10(2.0)
    )
    return log_area - 0.5 * matrix[:, _MAHALANOBIS] ** 2 / np.log(10.0)


def all_screen_scores(features: ArrayLike) -> dict[str, NDArray[np.float64]]:
    """Every analytic screen's score for the same feature matrix."""
    return {
        "miss_distance": miss_distance_score(features),
        "mahalanobis": mahalanobis_score(features),
        "small_body": small_body_score(features),
    }
