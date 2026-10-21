"""Cheap screening features."""

from __future__ import annotations

import numpy as np
import pytest

from conjprob.features import (
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    features_from_arrays,
    features_from_geometry,
)
from conjprob.geometry import encounter_geometry


def test_every_feature_has_a_description_with_units():
    assert set(FEATURE_DESCRIPTIONS) == set(FEATURE_NAMES)
    for name, text in FEATURE_DESCRIPTIONS.items():
        assert text, name
        assert any(unit in text for unit in ("metres", "dimensionless", "second"))


def test_feature_values_are_hand_calculable():
    matrix = features_from_arrays(300.0, 54.0, 120.0, 30.0, 12.0, 5740.0, 80.0)
    assert matrix.shape == (1, 10)
    index = {name: position for position, name in enumerate(FEATURE_NAMES)}
    row = matrix[0]
    assert row[index["log10_miss_distance_m"]] == pytest.approx(
        np.log10(np.hypot(120.0, 30.0)), rel=1e-13
    )
    assert row[index["log10_radius_m"]] == pytest.approx(np.log10(12.0), rel=1e-13)
    assert row[index["log10_aspect_ratio"]] == pytest.approx(
        np.log10(300.0 / 54.0), rel=1e-13
    )
    assert row[index["mahalanobis"]] == pytest.approx(
        np.hypot(120.0 / 300.0, 30.0 / 54.0), rel=1e-13
    )
    assert row[index["log10_radius_over_sigma_eff"]] == pytest.approx(
        np.log10(12.0 / np.sqrt(300.0 * 54.0)), rel=1e-13
    )
    assert row[index["log10_sigma_isotropic_3d_m"]] == pytest.approx(
        np.log10(np.sqrt((300.0**2 + 54.0**2 + 80.0**2) / 3.0)), rel=1e-13
    )


def test_abs_cos_psi_is_one_on_the_major_axis_and_zero_on_the_minor():
    index = FEATURE_NAMES.index("abs_cos_psi")
    on_major = features_from_arrays(300.0, 54.0, 120.0, 0.0, 12.0, 5740.0, 80.0)
    on_minor = features_from_arrays(300.0, 54.0, 0.0, 30.0, 12.0, 5740.0, 80.0)
    assert on_major[0, index] == pytest.approx(1.0, rel=1e-13)
    assert on_minor[0, index] == pytest.approx(0.0, abs=1e-13)


def test_features_broadcast_over_arrays():
    matrix = features_from_arrays(
        [300.0, 100.0], [54.0, 100.0], [120.0, 10.0], [30.0, 0.0], 12.0, 5740.0, 80.0
    )
    assert matrix.shape == (2, 10)
    assert np.all(np.isfinite(matrix))


def test_features_from_geometry_matches_features_from_arrays(
    leo_primary, leo_secondary, leo_radius_m
):
    geometry = encounter_geometry(leo_primary, leo_secondary)
    frame = geometry.frame
    unit = geometry.relative_velocity_mps / np.linalg.norm(geometry.relative_velocity_mps)
    sigma_out = np.sqrt(unit @ geometry.combined_covariance_m2 @ unit)
    expected = features_from_arrays(
        frame.sigma_x_m,
        frame.sigma_y_m,
        frame.x_bar_m,
        frame.y_bar_m,
        leo_radius_m,
        geometry.relative_speed_mps,
        sigma_out,
    )
    np.testing.assert_allclose(
        features_from_geometry(geometry, leo_radius_m), expected, rtol=1e-13
    )


@pytest.mark.parametrize(
    "args, message",
    [
        ((0.0, 54.0, 1.0, 1.0, 1.0, 1.0, 1.0), "sigma_x_m"),
        ((300.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0), "sigma_y_m"),
        ((300.0, 54.0, 1.0, 1.0, 0.0, 1.0, 1.0), "radius_m"),
        ((300.0, 54.0, 1.0, 1.0, 1.0, 0.0, 1.0), "relative_speed_mps"),
        ((300.0, 54.0, 1.0, 1.0, 1.0, 1.0, 0.0), "sigma_out_of_plane_m"),
    ],
)
def test_rejects_non_positive_inputs(args, message):
    with pytest.raises(ValueError, match=message):
        features_from_arrays(*args)
