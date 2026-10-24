"""Scenario builders: elements to states, named geometries and seeded families."""

from __future__ import annotations

import numpy as np
import pytest

from pointahead.kepler import MU_EARTH, R_EARTH
from pointahead.scenario import (
    DEFAULT_BEAM,
    FRAME_LABEL,
    NAMED_SCENARIOS,
    NARROW_BEAM,
    CircularOrbit,
    _chord_perigee_m,
    crosslink_from_orbits,
    random_crosslink,
    scenario_family,
)


def test_circular_orbit_radius_and_speed(circular_orbit: CircularOrbit, mu: float) -> None:
    # 600 km altitude on the WGS-84 equatorial radius: a = 6978137 m,
    # v = sqrt(mu / a) = 7557.8652... m/s, T = 5801.2318... s.
    assert circular_orbit.semi_major_axis_m == pytest.approx(R_EARTH + 600.0e3, rel=1e-15)
    assert circular_orbit.speed_mps(mu) == pytest.approx(
        float(np.sqrt(mu / (R_EARTH + 600.0e3))), rel=1e-15
    )
    assert circular_orbit.period_s(mu) == pytest.approx(5801.231764, rel=1e-8)


def test_circular_orbit_state_is_at_the_declared_radius(circular_orbit: CircularOrbit) -> None:
    state = circular_orbit.state("t")
    assert float(np.linalg.norm(state.position_m)) == pytest.approx(
        circular_orbit.semi_major_axis_m, rel=1e-13
    )
    assert state.speed_mps == pytest.approx(circular_orbit.speed_mps(), rel=1e-13)
    # A circular orbit has r . v = 0.
    assert abs(float(np.dot(state.position_m, state.velocity_mps))) < 1.0e-5


def test_inclination_tilts_the_orbit_plane() -> None:
    equatorial = CircularOrbit(600.0e3, 0.0, 0.0, 0.5).state("a")
    inclined = CircularOrbit(600.0e3, float(np.pi / 2), 0.0, 0.5).state("b")
    assert abs(equatorial.position_m[2]) < 1.0e-6
    assert abs(inclined.position_m[2]) > 1.0e6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"altitude_m": 0.0},
        {"altitude_m": -1.0},
        {"inclination_rad": np.nan},
        {"raan_rad": np.inf},
    ],
)
def test_bad_orbit_elements_are_refused(kwargs: dict) -> None:
    arguments = {
        "altitude_m": 600.0e3,
        "inclination_rad": 0.0,
        "raan_rad": 0.0,
        "argument_of_latitude_rad": 0.0,
        **kwargs,
    }
    with pytest.raises(ValueError):
        CircularOrbit(**arguments)


def test_named_scenarios_are_usable_and_carry_the_frame_label() -> None:
    for name, builder in NAMED_SCENARIOS.items():
        link = builder()
        assert link.label == name
        assert link.frame == FRAME_LABEL
        assert link.range_m > 1.0e5
        assert link.trajectory_model == "kepler"


def test_geo_link_is_far_longer_than_the_leo_links() -> None:
    links = {name: builder() for name, builder in NAMED_SCENARIOS.items()}
    assert links["geo_relay_to_leo"].range_m > 10.0 * links["leo_crossing"].range_m


def test_crossing_and_trailing_pairs_share_a_range() -> None:
    # Both are the same arc separation at the same altitude, so the chord length
    # is the same; only the plane differs.
    links = {name: builder() for name, builder in NAMED_SCENARIOS.items()}
    assert links["leo_crossing"].range_m == pytest.approx(
        links["leo_trailing"].range_m, rel=1e-12
    )


def test_scenario_family_is_deterministic_in_the_seed() -> None:
    first = scenario_family(1234, 5)
    second = scenario_family(1234, 5)
    third = scenario_family(1235, 5)
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a.local.position_m, b.local.position_m)
        assert np.array_equal(a.remote.velocity_mps, b.remote.velocity_mps)
    assert not np.array_equal(first[0].local.position_m, third[0].local.position_m)


def test_scenario_family_refuses_a_non_positive_count() -> None:
    with pytest.raises(ValueError, match="count must be positive"):
        scenario_family(1, 0)


def test_random_crosslinks_are_unoccluded_and_not_too_close() -> None:
    for link in scenario_family(99, 80):
        assert link.range_m >= 50.0e3
        assert _chord_perigee_m(link) > R_EARTH


def test_random_crosslink_labels_are_passed_through() -> None:
    rng = np.random.default_rng(7)
    link = random_crosslink(rng, "mine")
    assert link.label == "mine"


def test_crosslink_from_orbits_accepts_the_linear_model() -> None:
    link = crosslink_from_orbits(
        CircularOrbit(600.0e3, 0.0, 0.0, 0.0),
        CircularOrbit(600.0e3, 0.5, 0.0, 0.3),
        "pair",
        trajectory_model="linear",
        mu=MU_EARTH,
    )
    assert link.trajectory_model == "linear"


def test_declared_beams_have_the_expected_divergences() -> None:
    from pointahead.divergence import gaussian_divergence_half_angle

    assert gaussian_divergence_half_angle(
        DEFAULT_BEAM.wavelength_m, DEFAULT_BEAM.waist_m
    ) == pytest.approx(1.9735212943395024e-05, rel=1e-12)
    assert gaussian_divergence_half_angle(
        NARROW_BEAM.wavelength_m, NARROW_BEAM.waist_m
    ) < gaussian_divergence_half_angle(DEFAULT_BEAM.wavelength_m, DEFAULT_BEAM.waist_m)
