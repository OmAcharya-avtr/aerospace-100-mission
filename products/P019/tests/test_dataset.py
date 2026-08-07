"""Tests for the seeded synthetic dataset generator."""

from __future__ import annotations

import numpy as np
import pytest

from cncast.dataset import (
    ALTITUDE_MAX_M,
    ALTITUDE_MIN_M,
    FEATURE_NAMES,
    build_table,
    default_altitude_grid,
    generate_scenarios,
    met_features,
    profile_cn2,
    split_scenarios,
)

# ---------------------------------------------------------------- reproducibility


def test_scenarios_are_reproducible_for_a_fixed_seed() -> None:
    """Same seed -> bit-identical scenarios (mission requirement)."""
    a = generate_scenarios(25, seed=123)
    b = generate_scenarios(25, seed=123)
    assert a == b


def test_scenarios_differ_for_different_seeds() -> None:
    assert generate_scenarios(25, seed=123) != generate_scenarios(25, seed=124)


def test_profile_is_deterministic_given_a_scenario() -> None:
    """profile_cn2 is a pure function of (scenario, altitude) - no hidden rng."""
    sc = generate_scenarios(1, seed=5)[0]
    h = default_altitude_grid(30)
    assert np.array_equal(profile_cn2(sc, h), profile_cn2(sc, h))


def test_build_table_is_reproducible() -> None:
    sc = generate_scenarios(10, seed=1)
    x1, y1, g1 = build_table(sc, n_altitudes=8, seed=77)
    x2, y2, g2 = build_table(sc, n_altitudes=8, seed=77)
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2) and np.array_equal(g1, g2)


def test_split_is_reproducible_and_disjoint() -> None:
    tr, te = split_scenarios(100, 0.25, seed=42)
    tr2, te2 = split_scenarios(100, 0.25, seed=42)
    assert np.array_equal(tr, tr2) and np.array_equal(te, te2)
    assert set(tr).isdisjoint(set(te))
    assert len(tr) + len(te) == 100
    assert len(te) == 25


# ---------------------------------------------------------------- physical plausibility


def test_generated_profiles_span_a_plausible_cn2_range() -> None:
    """Surface Cn^2 in 1e-16..1e-13 and 20 km Cn^2 below 1e-16 for every scenario.

    Those brackets are the range spanned by the published baselines themselves
    (HV 5/7 ground 1.7e-14; SLC-Night ground 8.4e-15; all models < 1e-17 at
    20 km), widened for the perturbations.
    """
    scenarios = generate_scenarios(120, seed=9)
    ground = np.array([float(profile_cn2(s, np.array([5.0]))[0]) for s in scenarios])
    top = np.array([float(profile_cn2(s, np.array([20_000.0]))[0]) for s in scenarios])
    assert np.all(ground > 1e-16) and np.all(ground < 1e-13)
    assert np.all(top < 1e-16)


def test_generated_profiles_decrease_through_the_lower_troposphere() -> None:
    """Cn^2 must fall from 5 m to 2 km in every scenario (physical plausibility).

    The elevated-layer perturbation is confined to 800-8000 m and can be strong,
    so the check uses the decade-scale ratio rather than pointwise monotonicity.
    """
    scenarios = generate_scenarios(120, seed=11)
    for sc in scenarios:
        c = profile_cn2(sc, np.array([5.0, 2000.0]))
        assert c[0] > c[1]


def test_daytime_ground_turbulence_exceeds_night_on_average() -> None:
    """The generator encodes a diurnal cycle: noon > midnight at the surface.

    This is the single strongest signal a learned model can pick up, and it is
    the reason a fixed climatology such as HV 5/7 cannot compete on this data.
    """
    scenarios = generate_scenarios(400, seed=13)
    noon = [s.ground_cn2 for s in scenarios if 11.0 <= s.hour_of_day <= 13.0]
    night = [s.ground_cn2 for s in scenarios if s.hour_of_day <= 3.0 or s.hour_of_day >= 21.0]
    assert len(noon) > 5 and len(night) > 5
    assert np.median(noon) > 3.0 * np.median(night)


def test_altitude_grid_endpoints() -> None:
    g = default_altitude_grid(24)
    assert g[0] == pytest.approx(ALTITUDE_MIN_M)
    assert g[-1] == pytest.approx(ALTITUDE_MAX_M)
    assert np.all(np.diff(g) > 0.0)


def test_feature_matrix_shape_and_order() -> None:
    h = np.array([10.0, 100.0, 1000.0])
    x = met_features(20.0, 5.0, 50.0, 12.0, 180, h)
    assert x.shape == (3, len(FEATURE_NAMES))
    # column 7 is log10 altitude
    assert np.allclose(x[:, 7], np.log10(h))
    # cyclic encodings at 12:00 -> sin = 0, cos = -1
    assert x[0, 3] == pytest.approx(0.0, abs=1e-12)
    assert x[0, 4] == pytest.approx(-1.0, abs=1e-12)


# ---------------------------------------------------------------- input validation


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"surface_temp_c": 200.0}, "surface_temp_c"),
        ({"surface_wind_m_s": -1.0}, "surface_wind_m_s"),
        ({"relative_humidity_pct": 150.0}, "relative_humidity_pct"),
        ({"hour_of_day": 24.0}, "hour_of_day"),
        ({"hour_of_day": -0.5}, "hour_of_day"),
        ({"day_of_year": 0}, "day_of_year"),
        ({"day_of_year": 400}, "day_of_year"),
    ],
)
def test_met_features_rejects_impossible_inputs(kwargs: dict, match: str) -> None:
    good = {
        "surface_temp_c": 15.0,
        "surface_wind_m_s": 5.0,
        "relative_humidity_pct": 50.0,
        "hour_of_day": 12.0,
        "day_of_year": 180,
    }
    good.update(kwargs)
    with pytest.raises(ValueError, match=match):
        met_features(h_m=np.array([100.0]), **good)


def test_met_features_rejects_bad_altitudes() -> None:
    with pytest.raises(ValueError, match="h_m"):
        met_features(15.0, 5.0, 50.0, 12.0, 180, np.array([0.0]))
    with pytest.raises(ValueError, match="h_m"):
        met_features(15.0, 5.0, 50.0, 12.0, 180, np.array([]))


def test_generate_scenarios_rejects_zero() -> None:
    with pytest.raises(ValueError, match="n_scenarios"):
        generate_scenarios(0)


def test_split_rejects_degenerate_fractions() -> None:
    with pytest.raises(ValueError, match="test_fraction"):
        split_scenarios(10, 0.0)
    with pytest.raises(ValueError, match="test_fraction"):
        split_scenarios(10, 1.0)
    with pytest.raises(ValueError, match="at least 2"):
        split_scenarios(1, 0.5)


def test_build_table_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_table([], n_altitudes=4)


def test_profile_rejects_zero_altitude() -> None:
    sc = generate_scenarios(1, seed=3)[0]
    with pytest.raises(ValueError, match="h_m"):
        profile_cn2(sc, np.array([0.0]))
