"""Cheap features for conjunction screening, and what each one is.

The screening problem is to decide, without evaluating the collision-probability
integral, whether a conjunction will exceed a declared Pc threshold. Every
feature here is a closed-form function of the encounter geometry and the two
covariances, so the feature vector costs one eigendecomposition of a 2x2 matrix
and no quadrature.

The feature set deliberately includes everything the analytic screens in
:mod:`conjprob.screens` use, so that the learned classifier in
:mod:`conjprob.screening` is never beaten for want of information. It
deliberately excludes any quantity derived from the Pc integral itself. Two of
the ten features -- the relative speed and the out-of-plane standard deviation
-- have no effect whatever on Pc under the short-encounter assumptions, and are
included on purpose: a learned model that lets them change its decision is
being misled by them, and the feature importances reported in MODEL_CARD.md
measure whether it does.

Units are given per feature in :data:`FEATURE_DESCRIPTIONS`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import EncounterGeometry

FEATURE_NAMES: tuple[str, ...] = (
    "log10_miss_distance_m",
    "log10_radius_m",
    "log10_sigma_x_m",
    "log10_sigma_y_m",
    "log10_aspect_ratio",
    "mahalanobis",
    "log10_radius_over_sigma_eff",
    "log10_relative_speed_mps",
    "abs_cos_psi",
    "log10_sigma_isotropic_3d_m",
)
"""Feature order, fixed. Index positions are part of the model's contract."""

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "log10_miss_distance_m": "base-10 log of the in-plane miss distance, metres",
    "log10_radius_m": "base-10 log of the combined hard-body radius, metres",
    "log10_sigma_x_m": "base-10 log of the larger principal standard deviation, metres",
    "log10_sigma_y_m": "base-10 log of the smaller principal standard deviation, metres",
    "log10_aspect_ratio": "base-10 log of sigma_x / sigma_y, dimensionless",
    "mahalanobis": "Mahalanobis distance of the miss vector in the plane, dimensionless",
    "log10_radius_over_sigma_eff": (
        "base-10 log of R / sqrt(sigma_x sigma_y), dimensionless"
    ),
    "log10_relative_speed_mps": (
        "base-10 log of the relative speed, metres per second; has no effect on Pc "
        "under the short-encounter assumptions and is included as a nuisance feature"
    ),
    "abs_cos_psi": (
        "absolute cosine of the miss-vector direction in the principal frame, "
        "dimensionless; zero on the minor axis, one on the major axis"
    ),
    "log10_sigma_isotropic_3d_m": (
        "base-10 log of sqrt(trace(C_3d) / 3), metres; carries out-of-plane "
        "uncertainty, which the 2-D formulation marginalises away and which "
        "therefore does not affect Pc"
    ),
}

_EPS = 1e-300


def features_from_arrays(
    sigma_x_m: ArrayLike,
    sigma_y_m: ArrayLike,
    xbar_m: ArrayLike,
    ybar_m: ArrayLike,
    radius_m: ArrayLike,
    relative_speed_mps: ArrayLike,
    sigma_out_of_plane_m: ArrayLike,
) -> NDArray[np.float64]:
    """Feature matrix of shape ``(n, 10)`` from broadcastable encounter parameters.

    All length inputs in metres, speed in metres per second. ``sigma_x_m`` must
    be the larger principal standard deviation.

    Raises
    ------
    ValueError
        If any standard deviation, radius or speed is not strictly positive,
        since every one of them enters a logarithm.
    """
    sx, sy, xb, yb, r, speed, sz = (
        np.asarray(value, dtype=float)
        for value in (
            sigma_x_m,
            sigma_y_m,
            xbar_m,
            ybar_m,
            radius_m,
            relative_speed_mps,
            sigma_out_of_plane_m,
        )
    )
    sx, sy, xb, yb, r, speed, sz = np.broadcast_arrays(sx, sy, xb, yb, r, speed, sz)
    for name, value in (
        ("sigma_x_m", sx),
        ("sigma_y_m", sy),
        ("radius_m", r),
        ("relative_speed_mps", speed),
        ("sigma_out_of_plane_m", sz),
    ):
        if np.any(value <= 0.0):
            raise ValueError(f"{name} must be strictly positive everywhere")
    miss = np.hypot(xb, yb)
    sigma_eff = np.sqrt(sx * sy)
    mahalanobis = np.hypot(xb / sx, yb / sy)
    cos_psi = np.where(mahalanobis > 0.0, np.abs(xb / sx) / np.maximum(mahalanobis, _EPS), 1.0)
    isotropic = np.sqrt((sx**2 + sy**2 + sz**2) / 3.0)
    return np.column_stack(
        (
            np.log10(np.maximum(miss, _EPS)),
            np.log10(r),
            np.log10(sx),
            np.log10(sy),
            np.log10(sx / sy),
            mahalanobis,
            np.log10(r / sigma_eff),
            np.log10(speed),
            np.clip(cos_psi, 0.0, 1.0),
            np.log10(isotropic),
        )
    )


def features_from_geometry(
    geometry: EncounterGeometry, radius_m: float
) -> NDArray[np.float64]:
    """Feature vector of shape ``(1, 10)`` for one assessed encounter.

    The out-of-plane standard deviation is taken from the combined 3-D
    covariance along the relative-velocity direction, which is the direction the
    encounter-plane projection marginalises over.
    """
    frame = geometry.frame
    unit = geometry.relative_velocity_mps / np.linalg.norm(geometry.relative_velocity_mps)
    variance_out = float(unit @ geometry.combined_covariance_m2 @ unit)
    return features_from_arrays(
        frame.sigma_x_m,
        frame.sigma_y_m,
        frame.x_bar_m,
        frame.y_bar_m,
        radius_m,
        geometry.relative_speed_mps,
        max(np.sqrt(variance_out), _EPS),
    )
