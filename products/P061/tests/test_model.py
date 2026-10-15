"""Declared inputs: validation, closed forms and hand-calculated known answers."""

from __future__ import annotations

import math

import pytest
from scipy.integrate import quad

from activityplan.model import (
    WH_PER_J,
    Activity,
    Contact,
    PowerSegment,
    Problem,
    ResourceModel,
    SolarProfile,
)


def test_power_segment_energy_known_answer() -> None:
    # 36 W for 100 s = 3600 J = exactly 1 W*h.
    segment = PowerSegment(100.0, 36.0)
    assert segment.energy_wh == pytest.approx(1.0, abs=1e-15)


def test_wh_per_j_is_one_over_3600() -> None:
    assert WH_PER_J == pytest.approx(1.0 / 3600.0, rel=1e-15)


@pytest.mark.parametrize(
    ("duration", "power"), [(0.0, 10.0), (-1.0, 10.0), (math.inf, 10.0), (10.0, -1.0)]
)
def test_power_segment_rejects_bad_input(duration: float, power: float) -> None:
    with pytest.raises(ValueError):
        PowerSegment(duration, power)


def test_activity_energy_and_duration_known_answer() -> None:
    # Segments 100 s at 36 W (1 W*h) and 200 s at 18 W (3600 J = 1 W*h).
    # Duration 300 s, energy 2 W*h, mean power 2 * 3600 / 300 = 24 W, peak 36 W.
    activity = Activity(
        name="a", power_profile=(PowerSegment(100.0, 36.0), PowerSegment(200.0, 18.0))
    )
    assert activity.duration_s == pytest.approx(300.0, abs=1e-12)
    assert activity.energy_wh == pytest.approx(2.0, abs=1e-14)
    assert activity.mean_power_w == pytest.approx(24.0, rel=1e-13)
    assert activity.peak_power_w == pytest.approx(36.0, abs=1e-15)


def test_activity_power_at_boundaries_belongs_to_later_segment() -> None:
    activity = Activity(
        name="a", power_profile=(PowerSegment(100.0, 36.0), PowerSegment(200.0, 18.0))
    )
    assert activity.power_at(-1.0) == pytest.approx(0.0, abs=1e-15)
    assert activity.power_at(0.0) == pytest.approx(36.0, abs=1e-15)
    assert activity.power_at(99.999) == pytest.approx(36.0, abs=1e-15)
    assert activity.power_at(100.0) == pytest.approx(18.0, abs=1e-15)
    assert activity.power_at(300.0) == pytest.approx(0.0, abs=1e-15)


def test_activity_energy_between_known_answer() -> None:
    # [50, 150) covers 50 s at 36 W (1800 J) and 50 s at 18 W (900 J) = 2700 J
    # = 0.75 W*h.
    activity = Activity(
        name="a", power_profile=(PowerSegment(100.0, 36.0), PowerSegment(200.0, 18.0))
    )
    assert activity.energy_between_wh(50.0, 150.0) == pytest.approx(0.75, abs=1e-14)
    assert activity.energy_between_wh(0.0, 1000.0) == pytest.approx(2.0, abs=1e-14)
    assert activity.energy_between_wh(5.0, 5.0) == pytest.approx(0.0, abs=1e-15)


def test_activity_energy_between_rejects_reversed_interval() -> None:
    activity = Activity.uniform("a", 100.0, 10.0)
    with pytest.raises(ValueError, match="must not be less than"):
        activity.energy_between_wh(10.0, 5.0)


def test_activity_rejects_empty_profile_and_narrow_window() -> None:
    with pytest.raises(ValueError, match="empty power profile"):
        Activity(name="a", power_profile=())
    with pytest.raises(ValueError, match="window is too narrow"):
        Activity.uniform("a", 100.0, 10.0, earliest_start_s=0.0, latest_finish_s=99.0)


def test_activity_rejects_non_segment_profile() -> None:
    with pytest.raises(TypeError, match="PowerSegment"):
        Activity(name="a", power_profile=(1.0,))  # type: ignore[arg-type]


def test_activity_data_rate_known_answer() -> None:
    activity = Activity.uniform("a", 200.0, 10.0, data_mbit=50.0)
    assert activity.data_rate_mbit_per_s == pytest.approx(0.25, rel=1e-14)


def test_solar_full_arc_energy_is_two_over_pi_form_factor() -> None:
    # No eclipse: the whole 1000 s period is a half sine of peak 100 W.
    # Integral = 2 * P * T / pi joules = 2 * 100 * 1000 / pi = 63661.9772 J,
    # divided by 3600 = 17.6838826 W*h.
    solar = SolarProfile(peak_power_w=100.0, orbit_period_s=1000.0, eclipse_fraction=0.0)
    expected = 2.0 * 100.0 * 1000.0 / math.pi / 3600.0
    assert expected == pytest.approx(17.68388256576615, rel=1e-12)
    assert solar.energy_wh(0.0, 1000.0) == pytest.approx(expected, rel=1e-13)


def test_solar_half_arc_is_half_the_full_arc() -> None:
    solar = SolarProfile(peak_power_w=100.0, orbit_period_s=1000.0, eclipse_fraction=0.0)
    full = solar.energy_wh(0.0, 1000.0)
    assert solar.energy_wh(0.0, 500.0) == pytest.approx(0.5 * full, rel=1e-13)


def test_solar_closed_form_matches_numerical_quadrature() -> None:
    solar = SolarProfile(
        peak_power_w=165.0, orbit_period_s=5400.0, eclipse_fraction=0.34, phase_s=900.0
    )
    closed = solar.energy_wh(0.0, 16_200.0)
    numerical = quad(solar.power_at, 0.0, 16_200.0, limit=400)[0] / 3600.0
    assert closed == pytest.approx(numerical, rel=1e-9)


def test_solar_is_zero_in_eclipse() -> None:
    solar = SolarProfile(peak_power_w=100.0, orbit_period_s=1000.0, eclipse_fraction=0.5)
    assert solar.sunlit_s == pytest.approx(500.0, rel=1e-14)
    assert solar.power_at(600.0) == pytest.approx(0.0, abs=1e-15)
    assert solar.energy_wh(500.0, 1000.0) == pytest.approx(0.0, abs=1e-15)


def test_solar_handles_negative_phase_and_many_orbits() -> None:
    solar = SolarProfile(
        peak_power_w=50.0, orbit_period_s=600.0, eclipse_fraction=0.25, phase_s=-250.0
    )
    one = solar.energy_wh(0.0, 600.0)
    ten = solar.energy_wh(0.0, 6000.0)
    assert ten == pytest.approx(10.0 * one, rel=1e-12)


@pytest.mark.parametrize(
    ("peak", "period", "fraction"),
    [(-1.0, 100.0, 0.0), (1.0, 0.0, 0.0), (1.0, 100.0, 1.0), (1.0, 100.0, -0.1)],
)
def test_solar_rejects_bad_input(peak: float, period: float, fraction: float) -> None:
    with pytest.raises(ValueError):
        SolarProfile(peak_power_w=peak, orbit_period_s=period, eclipse_fraction=fraction)


def test_solar_rejects_reversed_interval() -> None:
    solar = SolarProfile(peak_power_w=1.0, orbit_period_s=100.0)
    with pytest.raises(ValueError, match="must not be less than"):
        solar.energy_wh(10.0, 1.0)


def test_contact_capacity_known_answer() -> None:
    # 2 Mbit/s for 450 s = 900 Mbit.
    contact = Contact(100.0, 550.0, 2.0, 25.0, "pass")
    assert contact.duration_s == pytest.approx(450.0, abs=1e-12)
    assert contact.capacity_mbit == pytest.approx(900.0, rel=1e-14)


def test_contact_rejects_reversed_window() -> None:
    with pytest.raises(ValueError, match="end_s > start_s"):
        Contact(100.0, 100.0, 1.0)


def _solar() -> SolarProfile:
    return SolarProfile(peak_power_w=100.0, orbit_period_s=1000.0, eclipse_fraction=0.3)


def test_resource_model_floor_and_usable_known_answer() -> None:
    # 10 W*h nameplate at a 0.5 depth-of-discharge limit: floor 5, usable 5.
    resources = ResourceModel(
        battery_capacity_wh=10.0,
        max_depth_of_discharge=0.5,
        solar=_solar(),
        recorder_capacity_mbit=100.0,
    )
    assert resources.floor_wh == pytest.approx(5.0, rel=1e-14)
    assert resources.usable_wh == pytest.approx(5.0, rel=1e-14)
    assert resources.initial_energy_wh == pytest.approx(10.0, rel=1e-14)


def test_resource_model_rejects_initial_state_below_the_floor() -> None:
    with pytest.raises(ValueError, match="already violates"):
        ResourceModel(
            battery_capacity_wh=10.0,
            max_depth_of_discharge=0.2,
            solar=_solar(),
            recorder_capacity_mbit=100.0,
            initial_state_of_charge=0.5,
        )


def test_resource_model_rejects_overlapping_contacts() -> None:
    with pytest.raises(ValueError, match="overlap"):
        ResourceModel(
            battery_capacity_wh=10.0,
            max_depth_of_discharge=0.5,
            solar=_solar(),
            recorder_capacity_mbit=100.0,
            contacts=(Contact(0.0, 100.0, 1.0, name="a"), Contact(50.0, 150.0, 1.0, name="b")),
        )


def test_resource_model_sorts_contacts_and_reports_rate_and_power() -> None:
    resources = ResourceModel(
        battery_capacity_wh=10.0,
        max_depth_of_discharge=0.5,
        solar=_solar(),
        recorder_capacity_mbit=100.0,
        contacts=(Contact(200.0, 300.0, 3.0, 9.0, "b"), Contact(0.0, 100.0, 1.0, 5.0, "a")),
    )
    assert tuple(contact.name for contact in resources.contacts) == ("a", "b")
    assert resources.downlink_rate_at(50.0) == pytest.approx(1.0, rel=1e-14)
    assert resources.contact_power_at(250.0) == pytest.approx(9.0, rel=1e-14)
    assert resources.downlink_rate_at(150.0) == pytest.approx(0.0, abs=1e-15)
    assert resources.contact_at(100.0) is None
    assert resources.total_contact_capacity_mbit == pytest.approx(400.0, rel=1e-14)


def _problem(**kwargs: object) -> Problem:
    resources = ResourceModel(
        battery_capacity_wh=10.0,
        max_depth_of_discharge=0.5,
        solar=_solar(),
        recorder_capacity_mbit=100.0,
    )
    activities = (
        Activity.uniform("a", 100.0, 10.0, priority=2.0, data_mbit=5.0),
        Activity.uniform("b", 100.0, 10.0, priority=3.0, data_mbit=7.0),
    )
    defaults: dict[str, object] = {
        "activities": activities,
        "resources": resources,
        "horizon_s": 1000.0,
    }
    defaults.update(kwargs)
    return Problem(**defaults)  # type: ignore[arg-type]


def test_problem_totals_known_answer() -> None:
    problem = _problem()
    assert problem.total_priority == pytest.approx(5.0, rel=1e-14)
    assert problem.total_data_mbit == pytest.approx(12.0, rel=1e-14)
    # 10 W for 100 s = 1000 J = 0.2777... W*h per activity, two of them.
    assert problem.total_activity_energy_wh == pytest.approx(2000.0 / 3600.0, rel=1e-13)


def test_problem_rejects_duplicates_unknown_names_and_cycles() -> None:
    with pytest.raises(ValueError, match="duplicate activity names"):
        _problem(
            activities=(
                Activity.uniform("a", 100.0, 10.0),
                Activity.uniform("a", 100.0, 10.0),
            )
        )
    with pytest.raises(ValueError, match="unknown activity"):
        _problem(precedences=(("a", "zzz"),))
    with pytest.raises(ValueError, match="self-referential"):
        _problem(exclusions=(("a", "a"),))
    with pytest.raises(ValueError, match="cycle"):
        _problem(precedences=(("a", "b"), ("b", "a")))


def test_problem_deduplicates_exclusions_regardless_of_order() -> None:
    problem = _problem(exclusions=(("a", "b"), ("b", "a")))
    assert problem.exclusions == (("a", "b"),)


def test_problem_rejects_activity_that_cannot_fit_before_the_horizon() -> None:
    with pytest.raises(ValueError, match="cannot fit before the horizon"):
        _problem(
            activities=(Activity.uniform("a", 100.0, 10.0, earliest_start_s=950.0),),
            horizon_s=1000.0,
        )


def test_problem_baseline_energy_balance_known_answer() -> None:
    # Bus 36 W over 1000 s = 36000 J = 10 W*h, plus a 100 s contact at 36 W
    # = 1 W*h, so 11 W*h of fixed consumption.
    resources = ResourceModel(
        battery_capacity_wh=10.0,
        max_depth_of_discharge=0.5,
        solar=SolarProfile(peak_power_w=0.0, orbit_period_s=1000.0),
        recorder_capacity_mbit=100.0,
        bus_load_w=36.0,
        contacts=(Contact(0.0, 100.0, 1.0, 36.0, "a"),),
    )
    problem = Problem(
        activities=(Activity.uniform("a", 100.0, 10.0),),
        resources=resources,
        horizon_s=1000.0,
    )
    generated, fixed = problem.baseline_energy_balance_wh()
    assert generated == pytest.approx(0.0, abs=1e-15)
    assert fixed == pytest.approx(11.0, rel=1e-13)


def test_problem_index_and_activity_lookup() -> None:
    problem = _problem()
    assert problem.index("b") == 1
    assert problem.activity("b").priority == pytest.approx(3.0, rel=1e-14)
    with pytest.raises(KeyError, match="unknown activity"):
        problem.index("zzz")


def test_problem_fixed_load_includes_contact_power() -> None:
    resources = ResourceModel(
        battery_capacity_wh=10.0,
        max_depth_of_discharge=0.5,
        solar=_solar(),
        recorder_capacity_mbit=100.0,
        bus_load_w=20.0,
        contacts=(Contact(0.0, 100.0, 1.0, 5.0, "a"),),
    )
    problem = Problem(
        activities=(Activity.uniform("a", 100.0, 10.0),),
        resources=resources,
        horizon_s=1000.0,
    )
    assert problem.fixed_load_w(50.0) == pytest.approx(25.0, rel=1e-14)
    assert problem.fixed_load_w(500.0) == pytest.approx(20.0, rel=1e-14)
