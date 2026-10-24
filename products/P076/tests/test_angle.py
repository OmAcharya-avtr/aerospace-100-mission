"""Point-ahead angle: known answers, attribution, invariants and the baseline score."""

from __future__ import annotations

import numpy as np
import pytest

from pointahead.angle import (
    closed_form_point_ahead,
    decompose_point_ahead,
    light_time_pair,
    offset_difference,
    reference_point_ahead,
    score_closed_form,
)
from pointahead.budget import direction_noise_floor_rad
from pointahead.geometry import C_LIGHT, angle_between, rotation_matrix
from pointahead.terminal import Crosslink, TerminalState


def test_closed_form_hand_calculated_known_answer(hand_link, hand_point_ahead_rad) -> None:
    # Local at rest, remote 2000 km away along +y moving at 7500 m/s along +x,
    # which is entirely transverse. theta = 2 * 7500 / 299792458
    #                                    = 5.003461427972281e-05 rad.
    # The literal below was wrong by 2.84e-12 on the first writing and the test
    # caught it; it is the value 15000 / 299792458 actually evaluates to.
    assert hand_point_ahead_rad == pytest.approx(5.003461427972281e-05, rel=1e-15, abs=0.0)
    result = closed_form_point_ahead(hand_link)
    assert result.method == "closed_form"

    # The OFFSET is exactly 2 v_perp / c.
    assert float(np.linalg.norm(result.offset_rad)) == pytest.approx(
        hand_point_ahead_rad, rel=1e-15, abs=0.0
    )

    # total_rad is the angle between two unit vectors built by applying half the
    # offset either side of the line of sight, so it is 2 arctan(theta / 2),
    # which is BELOW theta by theta^3 / 12 -- a relative deficit of
    # theta^2 / 12 = 2.0862e-10 here. Every comparison in this test passes
    # abs=0.0, because pytest.approx carries a default absolute tolerance of
    # 1e-12 that silently dominates any relative tolerance on a quantity of
    # order 1e-05: with the default, rel=1e-12 on this assertion is in fact a
    # 2e-08 relative comparison, and it passed while being wrong by 2.09e-10.
    # Pinned in tests/test_regressions.py.
    assert result.total_rad == pytest.approx(
        2.0 * float(np.arctan(0.5 * hand_point_ahead_rad)), rel=1e-15, abs=0.0
    )
    deficit = (hand_point_ahead_rad - result.total_rad) / hand_point_ahead_rad
    assert deficit == pytest.approx(hand_point_ahead_rad**2 / 12.0, rel=1e-6, abs=0.0)


def test_reference_reproduces_the_hand_calculation(hand_link, hand_point_ahead_rad) -> None:
    result = reference_point_ahead(hand_link)
    assert result.total_rad == pytest.approx(hand_point_ahead_rad, rel=1e-9)
    assert result.method == "reference"


def test_offset_norm_equals_the_total_angle(hand_link) -> None:
    result = reference_point_ahead(hand_link)
    assert float(np.linalg.norm(result.offset_rad)) == pytest.approx(result.total_rad, rel=1e-9)
    assert float(np.linalg.norm(result.offset_vector)) == pytest.approx(
        result.total_rad, rel=1e-9
    )


def test_point_ahead_vanishes_for_purely_radial_relative_motion(radial_link) -> None:
    # Both the apparent source direction and the required transmit direction
    # are parallel to the separation vector whatever the light times are, so
    # the angle is zero -- exactly in exact arithmetic, and to rounding here.
    assert closed_form_point_ahead(radial_link).total_rad == pytest.approx(0.0, abs=1e-18)
    assert reference_point_ahead(radial_link).total_rad < 1.0e-16


def test_point_ahead_is_invariant_under_a_global_rotation(crossing_link) -> None:
    # The criterion is ABSOLUTE and is the geometry's own float64 direction
    # resolution. A relative criterion here would be a statement about float64
    # rather than about the code; see validation/VALIDATION.md section 6.
    matrix = rotation_matrix([0.3, -0.7, 0.2], 1.1)
    rotated = Crosslink(
        local=TerminalState(
            matrix @ crossing_link.local.position_m,
            matrix @ crossing_link.local.velocity_mps,
            0.0,
            "local",
        ),
        remote=TerminalState(
            matrix @ crossing_link.remote.position_m,
            matrix @ crossing_link.remote.velocity_mps,
            0.0,
            "remote",
        ),
        trajectory_model=crossing_link.trajectory_model,
        mu=crossing_link.mu,
        label="rotated",
        frame=crossing_link.frame,
    )
    floor = direction_noise_floor_rad(crossing_link)
    for compute in (closed_form_point_ahead, reference_point_ahead):
        before = compute(crossing_link).total_rad
        after = compute(rotated).total_rad
        assert abs(after - before) <= floor


def test_reversing_the_relative_velocity_reverses_the_offset(hand_link) -> None:
    # Keeping the geometry and negating the relative velocity must negate the
    # point-ahead offset vector and leave its magnitude alone.
    reversed_link = Crosslink(
        local=hand_link.local,
        remote=TerminalState(
            hand_link.remote.position_m,
            2.0 * hand_link.local.velocity_mps - hand_link.remote.velocity_mps,
            0.0,
            "remote",
        ),
        trajectory_model="linear",
        label="reversed velocity",
        frame=hand_link.frame,
    )
    forward = closed_form_point_ahead(hand_link)
    backward = closed_form_point_ahead(reversed_link)
    assert backward.total_rad == pytest.approx(forward.total_rad, rel=1e-12)
    assert np.allclose(backward.offset_rad, -forward.offset_rad, atol=1e-18)


def test_swapping_the_terminals_preserves_the_magnitude(crossing_link) -> None:
    # The reverse link sees the same transverse relative speed, so the
    # point-ahead magnitude is the same. The attribution is not, because the
    # local velocity that drives the aberration pair has changed.
    forward = reference_point_ahead(crossing_link)
    backward = reference_point_ahead(crossing_link.reversed())
    assert backward.total_rad == pytest.approx(forward.total_rad, rel=1e-4)


def test_attribution_sums_to_the_reference_within_its_closure(crossing_link) -> None:
    decomposition = decompose_point_ahead(crossing_link)
    reference = reference_point_ahead(crossing_link)
    assert decomposition.total_rad == pytest.approx(reference.total_rad, rel=1e-8)
    assert decomposition.closure_rad < 1.0e-12
    assert decomposition.closure_rad / reference.total_rad < 1.0e-8


def test_the_two_aberration_lines_are_equal(crossing_link) -> None:
    # Both are the same -v_local_perp / c; the received beam and the
    # transmitted beam are aberrated by the same amount in opposite senses, and
    # both add into the point-ahead angle.
    decomposition = decompose_point_ahead(crossing_link)
    assert decomposition.stellar_aberration_magnitude_rad == pytest.approx(
        decomposition.velocity_aberration_magnitude_rad, rel=1e-15
    )
    assert np.allclose(
        decomposition.stellar_aberration, decomposition.velocity_aberration, atol=0.0
    )


def test_aberration_lines_vanish_for_a_stationary_local_terminal(hand_link) -> None:
    decomposition = decompose_point_ahead(hand_link)
    assert decomposition.stellar_aberration_magnitude_rad == pytest.approx(0.0, abs=1e-20)
    assert decomposition.light_time_lead_magnitude_rad == pytest.approx(
        2.0 * 7500.0 / C_LIGHT, rel=1e-12
    )


def test_trailing_geometry_has_a_smaller_angle_than_crossing(
    trailing_link, crossing_link
) -> None:
    assert (
        closed_form_point_ahead(trailing_link).total_rad
        < closed_form_point_ahead(crossing_link).total_rad
    )


def test_baseline_score_orders_its_three_metrics(crossing_link) -> None:
    score = score_closed_form(crossing_link)
    assert score.closed_form_rad > 0.0
    assert score.reference_rad > 0.0
    # The open-loop transmit-direction error contains the common-mode part that
    # a received-beam tracker removes, so it is the larger of the two.
    assert score.transmit_direction_rad > score.offset_difference_rad
    assert score.magnitude_difference_rad <= score.offset_difference_rad * 1.001


def test_truncating_the_iteration_moves_the_offset(crossing_link) -> None:
    converged = reference_point_ahead(crossing_link)
    truncated = reference_point_ahead(crossing_link, max_iter=0)
    assert offset_difference(truncated, converged) > 0.0
    assert offset_difference(converged, converged) == 0.0


def test_light_time_pair_returns_receive_then_transmit(crossing_link) -> None:
    receive, transmit = light_time_pair(crossing_link)
    assert receive.direction == "receive"
    assert transmit.direction == "transmit"
    assert receive.converged and transmit.converged


def test_first_order_aberration_route_changes_the_reference_slightly(crossing_link) -> None:
    exact = reference_point_ahead(crossing_link)
    approximate = reference_point_ahead(crossing_link, aberration_route="first_order")
    difference = offset_difference(exact, approximate)
    assert 0.0 < difference < 1.0e-12


def test_geo_link_light_time_lead_dominates_the_attribution(geo_link) -> None:
    # A geostationary-altitude terminal moves at about 3075 m/s and a 600 km
    # orbiter at about 7558 m/s, so for this pair the remote motion during
    # transit is the larger contribution. In the crossing pair it is not. The
    # ordering is the point; the numbers are measured.
    decomposition = decompose_point_ahead(geo_link)
    assert (
        decomposition.light_time_lead_magnitude_rad
        > decomposition.stellar_aberration_magnitude_rad
    )


def test_angle_between_source_and_transmit_matches_the_total(crossing_link) -> None:
    result = reference_point_ahead(crossing_link)
    assert angle_between(result.source_direction, result.transmit_direction) == pytest.approx(
        result.total_rad, rel=1e-15
    )
