"""The attributed budget: lines, totals, sensitivities and the refused inputs."""

from __future__ import annotations

import numpy as np
import pytest

from pointahead.budget import (
    BudgetInputs,
    build_budget,
    direction_noise_floor_rad,
    relative_acceleration,
    relative_jerk,
    staleness_position_error_m,
    staleness_velocity_error_mps,
    two_body_jerk,
    verify_sensitivities,
)
from pointahead.geometry import C_LIGHT, norm, transverse
from pointahead.terminal import BeamContext

EXPECTED_TERMS = (
    "relative_velocity_knowledge",
    "ephemeris_staleness_velocity",
    "ephemeris_staleness_geometry",
    "light_time_iteration_residual",
    "aberration_model_error",
    "angle_model_error",
)


def test_every_expected_line_is_present(crossing_link, default_inputs) -> None:
    budget = build_budget(crossing_link, default_inputs)
    assert tuple(term.name for term in budget.terms) == EXPECTED_TERMS
    for name in EXPECTED_TERMS:
        assert budget.term(name).name == name


def test_unknown_term_lookup_names_what_is_available(crossing_link, default_inputs) -> None:
    budget = build_budget(crossing_link, default_inputs)
    with pytest.raises(KeyError, match="no budget term named"):
        budget.term("jitter")


def test_total_is_between_the_largest_line_and_the_linear_sum(
    crossing_link, default_inputs
) -> None:
    budget = build_budget(crossing_link, default_inputs)
    values = [term.value_rad for term in budget.terms]
    assert budget.total_rss_rad >= max(values) * (1.0 - 1e-15)
    assert budget.total_rss_rad <= budget.total_linear_rad * (1.0 + 1e-15)


def test_dominant_term_is_the_largest_line(crossing_link, default_inputs) -> None:
    budget = build_budget(crossing_link, default_inputs)
    largest = max(budget.terms, key=lambda term: term.value_rad)
    assert budget.dominant_term == largest.name
    assert budget.dominant_fraction == pytest.approx(
        largest.value_rad / budget.total_rss_rad, rel=1e-13
    )


def test_velocity_knowledge_line_is_the_hand_calculated_sensitivity(
    crossing_link, default_beam
) -> None:
    # sigma per axis 0.05 m/s, transverse sigma 0.05 sqrt(2) = 0.0707106781...
    # theta = 2 * 0.0707106781 / 299792458 = 4.717308673...e-10 rad.
    inputs = BudgetInputs(0.05, 0.0, 3, default_beam)
    budget = build_budget(crossing_link, inputs)
    expected = 2.0 * 0.05 * float(np.sqrt(2.0)) / C_LIGHT
    assert budget.term("relative_velocity_knowledge").value_rad == pytest.approx(
        expected, rel=1e-15
    )
    assert expected == pytest.approx(4.717308673499369e-10, rel=1e-12)


def test_staleness_lines_vanish_at_zero_age(crossing_link, default_beam) -> None:
    inputs = BudgetInputs(0.05, 0.0, 3, default_beam)
    budget = build_budget(crossing_link, inputs)
    assert budget.term("ephemeris_staleness_velocity").value_rad == 0.0
    assert budget.term("ephemeris_staleness_geometry").value_rad == 0.0


def test_staleness_lines_grow_with_age(crossing_link, default_beam) -> None:
    previous = -1.0
    for age in (1.0, 10.0, 100.0, 1000.0):
        inputs = BudgetInputs(0.05, age, 3, default_beam)
        value = build_budget(crossing_link, inputs).term("ephemeris_staleness_velocity").value_rad
        assert value > previous
        previous = value


def test_relative_acceleration_is_radial_for_an_equal_radius_pair(crossing_link) -> None:
    # a_rel = -(mu / r^3) d for two terminals at the same radius, so it is
    # parallel to the separation vector and its transverse part is zero. This is
    # the geometry that made a first-order-only staleness model predict zero.
    acceleration = relative_acceleration(crossing_link)
    perpendicular = norm(transverse(acceleration, crossing_link.line_of_sight))
    assert norm(acceleration) > 1.0
    assert perpendicular / norm(acceleration) < 1.0e-14


def test_relative_jerk_has_a_transverse_part_where_acceleration_does_not(
    crossing_link,
) -> None:
    jerk = relative_jerk(crossing_link)
    perpendicular = norm(transverse(jerk, crossing_link.line_of_sight))
    assert perpendicular > 0.0
    assert perpendicular / norm(jerk) > 0.1


def test_two_body_jerk_known_answer(mu: float) -> None:
    # Circular orbit at a = 7000 km with r . v = 0:
    #   adot = -mu v / r^3 = -(3.986004418e14 / 3.43e20) * [0, 7546.0533, 0]
    #        = [0, -0.0087709..., 0] m/s^3
    a = 7.0e6
    speed = float(np.sqrt(mu / a))
    jerk = two_body_jerk([a, 0.0, 0.0], [0.0, speed, 0.0], mu)
    expected = -mu * speed / a**3
    assert jerk[1] == pytest.approx(expected, rel=1e-14)
    assert jerk[0] == pytest.approx(0.0, abs=1e-18)


def test_linear_model_has_no_acceleration_or_jerk(hand_link) -> None:
    assert np.allclose(relative_acceleration(hand_link), np.zeros(3))
    assert np.allclose(relative_jerk(hand_link), np.zeros(3))
    assert staleness_velocity_error_mps(hand_link, 600.0) == 0.0
    assert staleness_position_error_m(hand_link, 600.0) == 0.0


def test_staleness_helpers_refuse_a_negative_age(crossing_link) -> None:
    with pytest.raises(ValueError, match="age_s must be non-negative"):
        staleness_velocity_error_mps(crossing_link, -1.0)
    with pytest.raises(ValueError, match="age_s must be non-negative"):
        staleness_position_error_m(crossing_link, -1.0)


def test_staleness_position_error_uses_the_cubic_over_three_coefficient(
    crossing_link,
) -> None:
    # r_stale - r = -a dt^2 / 2 + adot dt^3 / 3. With a_rel_perp zero for this
    # geometry, the transverse position error is adot_perp dt^3 / 3 exactly, and
    # writing dt^3 / 6 would halve it.
    age = 30.0
    jerk_perp = norm(transverse(relative_jerk(crossing_link), crossing_link.line_of_sight))
    assert staleness_position_error_m(crossing_link, age) == pytest.approx(
        jerk_perp * age**3 / 3.0, rel=1e-12
    )


def test_aberration_line_is_only_paid_when_the_first_order_route_is_fielded(
    crossing_link, default_beam
) -> None:
    exact = build_budget(crossing_link, BudgetInputs(0.0, 0.0, 3, default_beam, "momentum_boost"))
    approximate = build_budget(
        crossing_link, BudgetInputs(0.0, 0.0, 3, default_beam, "first_order")
    )
    assert exact.term("aberration_model_error").value_rad == 0.0
    assert exact.term("aberration_model_error").measured_rad > 0.0
    assert approximate.term("aberration_model_error").value_rad == pytest.approx(
        exact.term("aberration_model_error").measured_rad, rel=1e-12
    )


def test_angle_model_line_is_only_paid_for_the_closed_form(
    crossing_link, default_beam
) -> None:
    closed = build_budget(
        crossing_link, BudgetInputs(0.0, 0.0, 3, default_beam, angle_method="closed_form")
    )
    reference = build_budget(
        crossing_link, BudgetInputs(0.0, 0.0, 3, default_beam, angle_method="reference")
    )
    assert closed.term("angle_model_error").value_rad > 0.0
    assert reference.term("angle_model_error").value_rad == 0.0
    assert reference.term("angle_model_error").measured_rad > 0.0


def test_light_time_line_shrinks_as_the_iteration_cap_rises(
    crossing_link, default_beam
) -> None:
    values = [
        build_budget(crossing_link, BudgetInputs(0.0, 0.0, k, default_beam))
        .term("light_time_iteration_residual")
        .value_rad
        for k in (0, 1, 2, 3)
    ]
    assert values[0] >= values[1] >= values[2]
    assert values[3] == 0.0


def test_beam_judgement_is_consistent(crossing_link, default_inputs) -> None:
    budget = build_budget(crossing_link, default_inputs)
    beam = budget.beam
    assert beam.in_far_field
    assert beam.error_fraction_of_beam == pytest.approx(
        budget.total_rss_rad / beam.divergence_half_angle_rad, rel=1e-13
    )
    assert beam.headroom == pytest.approx(
        beam.affordable_error_rad / budget.total_rss_rad, rel=1e-13
    )


def test_narrow_beam_has_less_headroom_for_the_same_error(crossing_link) -> None:
    wide = build_budget(
        crossing_link, BudgetInputs(0.05, 30.0, 2, BeamContext(1550.0e-9, 0.025))
    )
    narrow = build_budget(
        crossing_link, BudgetInputs(0.05, 30.0, 2, BeamContext(1064.0e-9, 0.150))
    )
    assert narrow.beam.divergence_half_angle_rad < wide.beam.divergence_half_angle_rad
    assert narrow.beam.headroom < wide.beam.headroom
    assert narrow.beam.loss_db > wide.beam.loss_db


def test_direction_noise_floor_is_tiny_but_not_zero(crossing_link) -> None:
    floor = direction_noise_floor_rad(crossing_link)
    assert 0.0 < floor < 1.0e-13
    expected = (
        8.0
        * float(np.sqrt(3.0))
        * float(np.finfo(np.float64).eps)
        * max(norm(crossing_link.local.position_m), norm(crossing_link.remote.position_m))
        / crossing_link.range_m
    )
    assert floor == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sigma_velocity_per_axis_mps": -1.0}, "sigma_velocity_per_axis_mps"),
        ({"ephemeris_age_s": -1.0}, "ephemeris_age_s"),
        ({"light_time_iterations": -1}, "light_time_iterations"),
        ({"angle_method": "magic"}, "angle_method must be one of"),
        ({"loss_budget_db": 0.0}, "loss_budget_db must be positive"),
    ],
)
def test_bad_budget_inputs_are_refused(kwargs: dict, match: str, default_beam) -> None:
    arguments = {
        "sigma_velocity_per_axis_mps": 0.05,
        "ephemeris_age_s": 30.0,
        "light_time_iterations": 2,
        "beam": default_beam,
        **kwargs,
    }
    with pytest.raises(ValueError, match=match):
        BudgetInputs(**arguments)


def test_transverse_sigma_is_root_two_times_per_axis(default_beam) -> None:
    inputs = BudgetInputs(0.1, 0.0, 2, default_beam)
    assert inputs.sigma_velocity_transverse_mps == pytest.approx(
        0.1 * float(np.sqrt(2.0)), rel=1e-15
    )


def test_sensitivity_checks_pass_on_the_named_geometries(
    crossing_link, geo_link, default_inputs
) -> None:
    for link in (crossing_link, geo_link):
        checks = verify_sensitivities(link, default_inputs)
        assert len(checks) == 4
        failures = [check.name for check in checks if not check.passed]
        assert not failures, f"{link.label}: {failures}"


def test_sensitivity_checks_are_all_zero_at_zero_age_and_full_iteration(
    crossing_link, default_beam
) -> None:
    inputs = BudgetInputs(0.0, 0.0, 30, default_beam)
    for check in verify_sensitivities(crossing_link, inputs):
        if check.name == "light_time_iteration_residual":
            continue
        assert check.analytic_rad == 0.0
        assert check.numeric_rad == 0.0
        assert check.passed


def test_unknown_comparison_is_refused(crossing_link, default_inputs) -> None:
    from pointahead.budget import _check

    with pytest.raises(ValueError, match="comparison must be"):
        _check("x", "roughly", 1.0, 1.0, 0.1)


def test_budget_as_dict_matches_the_terms(crossing_link, default_inputs) -> None:
    budget = build_budget(crossing_link, default_inputs)
    mapping = budget.as_dict()
    assert set(mapping) == set(EXPECTED_TERMS)
    for term in budget.terms:
        assert mapping[term.name] == term.value_rad
