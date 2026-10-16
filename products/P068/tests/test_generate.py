"""Synthetic trajectory generation: determinism, labels and selection."""

from __future__ import annotations

import numpy as np
import pytest

from degradepredict.generate import (
    DEFAULT_LIMIT,
    FAMILIES,
    GeneratorConfig,
    generate_population,
    mean_curve,
)
from degradepredict.limits import LimitSpec, first_crossing_time


def test_mean_curve_linear_known_answer():
    # y = 100 - 0.02 t: at t = 500 d the metric is 100 - 10 = 90 percent of BOL.
    y = mean_curve(np.array([0.0, 500.0]), "linear",
                   {"y0_percent": 100.0, "b_percent_per_day": 0.02})
    assert y[0] == pytest.approx(100.0)
    assert y[1] == pytest.approx(90.0)


def test_mean_curve_exponential_known_answer():
    # y = 100 exp(-k t) with k = ln(2)/1000 halves in 1000 d.
    k = np.log(2.0) / 1000.0
    y = mean_curve(np.array([1000.0]), "exponential", {"y0_percent": 100.0, "k_per_day": k})
    assert y[0] == pytest.approx(50.0)


def test_mean_curve_powerlaw_known_answer():
    # y = 100 (1 - B t**0.5) with B = 0.01 gives 100 (1 - 0.01*10) = 90 at t = 100 d.
    y = mean_curve(np.array([100.0]), "powerlaw",
                   {"y0_percent": 100.0, "B_per_day_z": 0.01, "z": 0.5})
    assert y[0] == pytest.approx(90.0)


def test_mean_curve_rejects_unknown_family():
    with pytest.raises(ValueError, match="unknown family"):
        mean_curve(np.array([0.0]), "quadratic", {"y0_percent": 100.0})  # type: ignore[arg-type]


def test_mean_curve_rejects_negative_times():
    with pytest.raises(ValueError, match="non-negative"):
        mean_curve(np.array([-1.0]), "linear",
                   {"y0_percent": 100.0, "b_percent_per_day": 0.01})


@pytest.mark.parametrize("family", FAMILIES)
def test_solved_shape_puts_the_mean_crossing_where_the_label_says(family):
    """The crossing label must be the crossing of the noise-free mean curve.

    Checked by locating the crossing numerically on a dense grid and comparing
    with the recorded label. Agreement is to the linear-interpolation error of
    the grid, not bit-exact.
    """
    cfg = GeneratorConfig(family_weights=tuple(1.0 if f == family else 0.0 for f in FAMILIES))
    pop = generate_population(12, seed=4242, config=cfg)
    for tr in pop:
        assert tr.family == family
        grid = np.linspace(0.0, 1.5 * tr.crossing_time_days, 40001)
        located = first_crossing_time(grid, mean_curve(grid, tr.family, tr.params), DEFAULT_LIMIT)
        assert located is not None
        assert located == pytest.approx(tr.crossing_time_days, rel=1e-5)


def test_population_is_deterministic_for_a_seed():
    a = generate_population(25, seed=7)
    b = generate_population(25, seed=7)
    for x, y in zip(a, b, strict=True):
        assert x.family == y.family
        assert x.crossing_time_days == y.crossing_time_days
        np.testing.assert_array_equal(x.values_percent, y.values_percent)


def test_prefix_of_a_larger_population_is_unchanged():
    """Spawned seed sequences make unit k identical for any n >= k."""
    small = generate_population(10, seed=11)
    large = generate_population(60, seed=11)
    for x, y in zip(small, large[:10], strict=True):
        assert x.family == y.family
        np.testing.assert_array_equal(x.values_percent, y.values_percent)


def test_different_seeds_give_different_populations():
    a = generate_population(20, seed=1)
    b = generate_population(20, seed=2)
    assert any(x.crossing_time_days != y.crossing_time_days for x, y in zip(a, b, strict=True))


def test_every_unit_is_still_inside_the_limit_at_the_forecast_origin():
    for tr in generate_population(300, seed=31):
        assert tr.remaining_true_days > 0.0
        assert tr.origin_days < tr.crossing_time_days


def test_observed_duration_and_event_flag_are_consistent():
    for tr in generate_population(300, seed=32):
        expected = min(tr.remaining_true_days, tr.followup_days)
        assert tr.remaining_observed_days == pytest.approx(expected)
        assert tr.event == (tr.remaining_true_days <= tr.followup_days)


def test_default_configuration_leaves_most_units_censored():
    pop = generate_population(600, seed=20261016)
    censored = float(np.mean([not tr.event for tr in pop]))
    assert 0.45 < censored < 0.70, f"censored fraction {censored:.4f} outside the design range"


def test_sample_times_are_on_the_declared_cadence():
    cfg = GeneratorConfig(sample_interval_days=7.0)
    for tr in generate_population(20, seed=5, config=cfg):
        np.testing.assert_allclose(np.diff(tr.times_days), 7.0)
        assert tr.times_days[0] == 0.0
        assert len(tr.times_days) >= 3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sample_interval_days": 0.0}, "sample_interval_days"),
        ({"t0_range": (1.0, 2.0)}, "at least two samples"),
        ({"family_weights": (0.5, 0.2, 0.2)}, "family_weights"),
        ({"noise_sd_range": (-1.0, 1.0)}, "non-negative"),
        ({"powerlaw_z_range": (0.0, 0.5)}, "strictly positive"),
        ({"y0_range": (10.0, 1.0)}, "y0_range"),
    ],
)
def test_generator_config_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GeneratorConfig(**kwargs)


def test_generate_population_rejects_non_positive_n():
    with pytest.raises(ValueError, match="positive integer"):
        generate_population(0, seed=1)


def test_generate_population_rejects_an_upward_limit():
    with pytest.raises(ValueError, match="downward-degrading"):
        generate_population(3, seed=1, limit=LimitSpec(value=1.0, direction="above"))


def test_generate_population_reports_when_selection_starves_it():
    # Crossing times near e**4 = 54.6 d are almost always before the earliest
    # forecast origin, so nearly every candidate is rejected.
    cfg = GeneratorConfig(log_crossing_mean=4.0, log_crossing_sd=0.1)
    with pytest.raises(ValueError, match="survived selection"):
        generate_population(50, seed=3, config=cfg)
