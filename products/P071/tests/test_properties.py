"""Property-based tests for the algebraic identities the Pc integral satisfies.

``.hypothesis/`` is not committed, so any falsifying example found here is lost
on a fresh clone. Every counterexample found during the build is pinned with its
literal inputs in ``tests/test_regressions.py``.

Deadlines are disabled: the adaptive quadrature inside the reference path has a
wide spread of run times and a deadline turns that into a flaky failure rather
than information.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conjprob.analytic import (
    chan_recommended_order,
    pc_chan,
    pc_foster,
    pc_small_body_approximation,
)
from conjprob.dilution import max_probability
from conjprob.features import FEATURE_NAMES, features_from_arrays
from conjprob.geometry import (
    ObjectState,
    encounter_geometry,
    encounter_plane_basis,
    mahalanobis_distance,
    miss_vector,
    principal_frame,
    project_covariance,
)
from conjprob.quadrature import pc_quadrature
from conjprob.screening import recall_at_fpr

SETTINGS = settings(max_examples=60, deadline=None)

sigmas = st.floats(min_value=1.0, max_value=1.0e4, allow_nan=False, allow_infinity=False)
offsets = st.floats(min_value=-5.0e4, max_value=5.0e4, allow_nan=False, allow_infinity=False)
radii = st.floats(min_value=0.1, max_value=200.0, allow_nan=False, allow_infinity=False)
factors = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)
angles = st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False, allow_infinity=False)


@SETTINGS
@given(sigmas, sigmas, offsets, offsets, radii)
def test_pc_is_always_a_probability(sigma_a, sigma_b, xbar, ybar, radius):
    sigma_x, sigma_y = max(sigma_a, sigma_b), min(sigma_a, sigma_b)
    for value in (
        pc_foster(sigma_x, sigma_y, xbar, ybar, radius),
        pc_chan(sigma_x, sigma_y, xbar, ybar, radius),
        pc_small_body_approximation(sigma_x, sigma_y, xbar, ybar, radius),
    ):
        assert 0.0 <= value <= 1.0


@SETTINGS
@given(sigmas, sigmas, offsets, offsets, radii, factors)
def test_foster_is_scale_invariant(sigma_a, sigma_b, xbar, ybar, radius, factor):
    sigma_x, sigma_y = max(sigma_a, sigma_b), min(sigma_a, sigma_b)
    base = pc_foster(sigma_x, sigma_y, xbar, ybar, radius)
    scaled = pc_foster(
        sigma_x * factor, sigma_y * factor, xbar * factor, ybar * factor, radius * factor
    )
    assert scaled == pytest.approx(base, rel=1e-7, abs=1e-300)


@SETTINGS
@given(sigmas, sigmas, offsets, offsets, radii)
def test_foster_matches_the_independent_quadrature(sigma_a, sigma_b, xbar, ybar, radius):
    sigma_x, sigma_y = max(sigma_a, sigma_b), min(sigma_a, sigma_b)
    reference, abserr = pc_quadrature(
        np.diag([sigma_x**2, sigma_y**2]), (xbar, ybar), radius
    )
    value = pc_foster(sigma_x, sigma_y, xbar, ybar, radius)
    assert value == pytest.approx(reference, rel=1e-6, abs=max(10.0 * abserr, 1e-300))


@SETTINGS
@given(sigmas, offsets, offsets, radii)
def test_chan_is_exact_for_a_circular_covariance(sigma, xbar, ybar, radius):
    # Exact only at the recommended order. A fixed order of 60 fails at
    # M = 9: the series weights are Poisson with mean M^2 / 2 and a fixed
    # truncation is adequate only up to a fixed M. Pinned in
    # tests/test_regressions.py.
    mahalanobis = math.hypot(xbar, ybar) / sigma
    if mahalanobis > 40.0:
        # Pc has underflowed to zero; there is nothing left to compare.
        return
    reference = pc_foster(sigma, sigma, xbar, ybar, radius)
    order = chan_recommended_order(mahalanobis)
    value = pc_chan(sigma, sigma, xbar, ybar, radius, order=order)
    assert value == pytest.approx(reference, rel=1e-6, abs=1e-300)


@SETTINGS
@given(sigmas, sigmas, offsets, offsets, radii, radii)
def test_pc_is_non_decreasing_in_the_hard_body_radius(
    sigma_a, sigma_b, xbar, ybar, radius_a, radius_b
):
    sigma_x, sigma_y = max(sigma_a, sigma_b), min(sigma_a, sigma_b)
    small, large = min(radius_a, radius_b), max(radius_a, radius_b)
    assert pc_foster(sigma_x, sigma_y, xbar, ybar, large) >= pc_foster(
        sigma_x, sigma_y, xbar, ybar, small
    ) - 1e-15


@SETTINGS
@given(sigmas, sigmas, offsets, offsets, radii)
def test_small_body_maximum_bounds_the_small_body_value(
    sigma_a, sigma_b, xbar, ybar, radius
):
    sigma_x, sigma_y = max(sigma_a, sigma_b), min(sigma_a, sigma_b)
    # The guard uses the squared Mahalanobis distance, because that is what
    # max_probability divides by and it can underflow to zero while the
    # unsquared hypotenuse is still a positive subnormal. Hypothesis found
    # exactly that at xbar = 2.2250738585072014e-308, and the case is pinned in
    # tests/test_regressions.py.
    if math.hypot(xbar / sigma_x, ybar / sigma_y) ** 2 <= 0.0:
        return
    pc_max, _ = max_probability(sigma_x, sigma_y, xbar, ybar, radius)
    here = pc_small_body_approximation(sigma_x, sigma_y, xbar, ybar, radius)
    assert here <= pc_max * (1.0 + 1e-9) + 1e-300


@SETTINGS
@given(offsets, offsets, offsets, angles)
def test_miss_vector_is_perpendicular_to_the_relative_velocity(rx, ry, rz, angle):
    velocity = np.array([math.cos(angle), math.sin(angle), 0.3]) * 5000.0
    position = np.array([rx, ry, rz])
    vector = miss_vector(position, velocity)
    scale = max(float(np.linalg.norm(vector)) * float(np.linalg.norm(velocity)), 1.0)
    assert abs(float(vector @ velocity)) <= 1e-9 * scale


@SETTINGS
@given(sigmas, sigmas, offsets, offsets, angles)
def test_mahalanobis_is_invariant_under_an_in_plane_rotation(
    sigma_a, sigma_b, xbar, ybar, angle
):
    covariance = np.diag([max(sigma_a, sigma_b) ** 2, min(sigma_a, sigma_b) ** 2])
    miss = np.array([xbar, ybar])
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    base = mahalanobis_distance(covariance, miss)
    rotated = mahalanobis_distance(rotation @ covariance @ rotation.T, rotation @ miss)
    assert rotated == pytest.approx(base, rel=1e-8, abs=1e-12)


@SETTINGS
@given(angles, angles, radii)
def test_pc_is_invariant_under_a_rigid_rotation_of_the_whole_problem(
    yaw, pitch, radius
):
    position_a = np.array([7078137.0, 0.0, 0.0])
    velocity_a = np.array([0.0, 7504.0, 0.0])
    position_b = np.array([7078167.0, 110.0, 40.0])
    velocity_b = np.array([0.0, 5306.934315511, 5306.934315511])
    covariance_a = np.diag([400.0, 10000.0, 400.0])
    covariance_b = np.diag([2500.0, 90000.0, 2500.0])
    cz, sz = math.cos(yaw), math.sin(yaw)
    cy, sy = math.cos(pitch), math.sin(pitch)
    rotation = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]) @ np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    )

    def pc(matrix):
        primary = ObjectState(
            position_m=matrix @ position_a,
            velocity_mps=matrix @ velocity_a,
            covariance_m2=matrix @ covariance_a @ matrix.T,
        )
        secondary = ObjectState(
            position_m=matrix @ position_b,
            velocity_mps=matrix @ velocity_b,
            covariance_m2=matrix @ covariance_b @ matrix.T,
        )
        geometry = encounter_geometry(primary, secondary)
        frame = geometry.frame
        return pc_foster(
            frame.sigma_x_m, frame.sigma_y_m, frame.x_bar_m, frame.y_bar_m, radius
        )

    assert pc(rotation) == pytest.approx(pc(np.eye(3)), rel=1e-6, abs=1e-300)


@SETTINGS
@given(sigmas, sigmas, angles)
def test_projection_preserves_the_trace_of_the_in_plane_part(sigma_a, sigma_b, angle):
    velocity = np.array([math.cos(angle), math.sin(angle), 0.5]) * 7000.0
    covariance = np.eye(3) * sigma_a**2
    basis = encounter_plane_basis(velocity)
    projected = project_covariance(covariance, basis)
    assert float(np.trace(projected)) == pytest.approx(2.0 * sigma_a**2, rel=1e-9)
    frame = principal_frame(projected, [sigma_b, 0.0])
    assert frame.aspect_ratio == pytest.approx(1.0, rel=1e-7)


@SETTINGS
@given(sigmas, sigmas, offsets, offsets, radii)
def test_mahalanobis_feature_matches_its_definition(sigma_a, sigma_b, xbar, ybar, radius):
    sigma_x, sigma_y = max(sigma_a, sigma_b), min(sigma_a, sigma_b)
    matrix = features_from_arrays(
        sigma_x, sigma_y, xbar, ybar, radius, 5000.0, sigma_x
    )
    index = FEATURE_NAMES.index("mahalanobis")
    assert matrix[0, index] == pytest.approx(
        math.hypot(xbar / sigma_x, ybar / sigma_y), rel=1e-12, abs=1e-12
    )


@SETTINGS
@given(
    st.integers(min_value=40, max_value=400),
    st.floats(min_value=0.01, max_value=0.5),
    st.integers(min_value=0, max_value=10_000),
)
def test_achieved_false_positive_rate_never_exceeds_the_target(n, target, seed):
    rng = np.random.default_rng(seed)
    labels = rng.random(n) < 0.4
    if labels.all() or not labels.any():
        return
    scores = labels * 0.8 + rng.normal(size=n)
    point = recall_at_fpr(labels, scores, target)
    assert point.fpr_achieved <= target + 1e-12
    assert 0.0 <= point.recall <= 1.0
