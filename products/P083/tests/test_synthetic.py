"""The generator: its declared equations, its determinism and its validation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hilverdict.plant import ALPHA_VALID_RAD, LoopModel
from hilverdict.synthetic import (
    CAMPAIGNS,
    CLOCK_CLEAN,
    SWITCH_FRACTIONS,
    TIMING_PROFILES,
    ClockSpec,
    ExecutionModel,
    doublet_command,
    generate_population,
    generate_run,
)
from hilverdict.timeline import miss_events, utilisation


def test_three_timing_profiles_are_declared():
    assert set(TIMING_PROFILES) == {"nominal", "loaded", "bursty"}


def test_every_profile_names_itself():
    for key, profile in TIMING_PROFILES.items():
        assert profile.name == key


def test_the_loaded_profile_runs_hotter_than_the_nominal_one():
    assert (
        TIMING_PROFILES["loaded"].mean_fraction
        > TIMING_PROFILES["nominal"].mean_fraction
    )


def test_only_the_bursty_profile_correlates_its_stalls():
    assert TIMING_PROFILES["bursty"].burst_continuation > 0.0
    assert TIMING_PROFILES["nominal"].burst_continuation == 0.0
    assert TIMING_PROFILES["loaded"].burst_continuation == 0.0


@pytest.mark.parametrize("value", [0.0, 2.0, -1.0])
def test_execution_model_rejects_an_out_of_range_mean_fraction(value):
    with pytest.raises(ValueError, match="mean_fraction must lie"):
        ExecutionModel(value, 0.1, 0.01, 1.0, 0.0, "t")


def test_execution_model_rejects_a_negative_jitter():
    with pytest.raises(ValueError, match="log_jitter must be non-negative"):
        ExecutionModel(0.5, -0.1, 0.01, 1.0, 0.0, "t")


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_execution_model_rejects_an_out_of_range_stall_rate(value):
    with pytest.raises(ValueError, match="stall_rate must lie"):
        ExecutionModel(0.5, 0.1, value, 1.0, 0.0, "t")


def test_execution_model_rejects_negative_stall_periods():
    with pytest.raises(ValueError, match="stall_periods must be non-negative"):
        ExecutionModel(0.5, 0.1, 0.01, -1.0, 0.0, "t")


@pytest.mark.parametrize("value", [-0.1, 1.0, 1.5])
def test_execution_model_rejects_an_out_of_range_burst_continuation(value):
    with pytest.raises(ValueError, match="burst_continuation must lie"):
        ExecutionModel(0.5, 0.1, 0.01, 1.0, value, "t")


def test_draw_returns_positive_durations():
    durations = TIMING_PROFILES["loaded"].draw(
        500, 0.010, np.random.default_rng(0)
    )
    assert durations.shape == (500,)
    assert float(np.min(durations)) > 0.0


def test_draw_is_deterministic_for_a_seed():
    first = TIMING_PROFILES["loaded"].draw(100, 0.010, np.random.default_rng(3))
    second = TIMING_PROFILES["loaded"].draw(100, 0.010, np.random.default_rng(3))
    assert np.array_equal(first, second)


def test_draw_median_is_near_the_declared_mean_fraction():
    profile = ExecutionModel(0.6, 0.2, 0.0, 0.0, 0.0, "t")
    durations = profile.draw(20000, 0.010, np.random.default_rng(1))
    assert float(np.median(durations)) == pytest.approx(
        0.006, rel=0.02, abs=0.0
    )


def test_a_zero_stall_rate_removes_the_tail():
    profile = ExecutionModel(0.5, 0.05, 0.0, 2.0, 0.0, "t")
    durations = profile.draw(5000, 0.010, np.random.default_rng(2))
    assert float(np.max(durations)) < 0.010


def test_switch_fractions_are_increasing_and_inside_the_run():
    assert list(SWITCH_FRACTIONS) == sorted(SWITCH_FRACTIONS)
    assert 0.0 < SWITCH_FRACTIONS[0] and SWITCH_FRACTIONS[-1] < 1.0


def test_doublet_command_has_the_declared_number_of_switches():
    command = doublet_command(200, math.radians(6.0))
    changes = int(np.count_nonzero(np.diff(command) != 0.0))
    assert changes == len(SWITCH_FRACTIONS)


def test_doublet_command_starts_and_ends_at_trim():
    command = doublet_command(200, math.radians(6.0))
    assert float(command[0]) == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert float(command[-1]) == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_doublet_command_amplitude_is_respected():
    amplitude = math.radians(6.0)
    command = doublet_command(200, amplitude)
    assert float(np.max(np.abs(command))) == pytest.approx(
        amplitude, rel=1e-15, abs=0.0
    )


def test_doublet_command_changes_sign():
    command = doublet_command(200, math.radians(6.0))
    assert float(np.min(command)) < 0.0 < float(np.max(command))


def test_doublet_command_scales_linearly():
    first = doublet_command(100, 0.1)
    second = doublet_command(100, 0.2)
    assert second == pytest.approx(2.0 * first, rel=1e-15, abs=0.0)


def test_generate_run_rejects_an_unknown_profile():
    with pytest.raises(ValueError, match="unknown timing profile"):
        generate_run(seed=1, profile="turbo")


def test_generate_run_rejects_too_few_steps():
    with pytest.raises(ValueError, match="steps must be at least 2"):
        generate_run(seed=1, steps=1)


def test_generate_run_is_deterministic():
    first = generate_run(seed=21, profile="bursty", amplitude_deg=6.0, steps=80)
    second = generate_run(seed=21, profile="bursty", amplitude_deg=6.0, steps=80)
    assert np.array_equal(first.durations_s, second.durations_s)
    assert np.array_equal(first.states, second.states)
    assert np.array_equal(first.device_time_s, second.device_time_s)


def test_generate_run_label_names_its_declaration():
    log = generate_run(seed=21, profile="bursty", amplitude_deg=6.0, steps=80)
    assert "bursty" in log.label
    assert "21" in log.label


def test_generate_run_host_time_is_the_exact_sample_grid():
    log = generate_run(seed=1, steps=50)
    expected = np.arange(51) * log.model.sampling.period_s
    assert np.asarray(log.host_time_s) == pytest.approx(expected, rel=1e-15, abs=0.0)


def test_generate_run_reference_is_miss_free():
    from hilverdict.resimulate import simulate

    log = generate_run(seed=1, profile="loaded", amplitude_deg=6.0, steps=80)
    clean = simulate(log.model, log.command_rad, None)
    assert np.array_equal(np.asarray(log.states), clean.states)


def test_generate_run_scales_the_durations_by_the_clock_drift():
    clean = generate_run(seed=33, profile="loaded", amplitude_deg=6.0, steps=80)
    spec = ClockSpec(drift_ppm=10000.0, jitter_s=2e-7, tick_s=1e-7,
                     reset_fractions=(), name="t")
    drifted = generate_run(seed=33, profile="loaded", amplitude_deg=6.0, steps=80,
                           clock=spec)
    assert np.asarray(drifted.durations_s) == pytest.approx(
        np.asarray(clean.durations_s) * 1.01, rel=1e-12, abs=0.0
    )


def test_the_clean_clock_spec_has_no_drift_and_no_reset():
    assert CLOCK_CLEAN.drift_ppm == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert CLOCK_CLEAN.reset_fractions == ()


def test_nine_campaigns_are_declared():
    assert len(CAMPAIGNS) == 9
    assert {profile for profile, _amplitude in CAMPAIGNS} == set(TIMING_PROFILES)


def test_every_campaign_stays_inside_the_declared_validity_region():
    for profile, amplitude in CAMPAIGNS:
        log = generate_run(seed=1, profile=profile, amplitude_deg=amplitude, steps=200)
        assert log.peak_alpha_rad <= ALPHA_VALID_RAD
        assert log.saturated_steps == 0


def test_every_campaign_has_a_utilisation_below_one():
    for profile, amplitude in CAMPAIGNS:
        log = generate_run(seed=2, profile=profile, amplitude_deg=amplitude, steps=200)
        assert utilisation(log) < 1.0


def test_generate_population_cycles_the_campaigns():
    runs = generate_population(tuple(range(9)), steps=40)
    labels = [run.label.split("-seed")[0] for run in runs]
    assert labels == [f"{profile}-{amplitude:g}deg" for profile, amplitude in CAMPAIGNS]


def test_generate_population_rejects_an_empty_seed_tuple():
    with pytest.raises(ValueError, match="seeds must not be empty"):
        generate_population(())


def test_generate_population_accepts_an_explicit_model():
    model = LoopModel()
    runs = generate_population((1, 2), steps=30, model=model)
    assert all(run.model is model for run in runs)


def test_a_loaded_population_has_more_misses_than_a_nominal_one():
    nominal = generate_population(
        (1, 2, 3), steps=200, campaigns=(("nominal", 6.0),) * 3
    )
    loaded = generate_population(
        (1, 2, 3), steps=200, campaigns=(("loaded", 6.0),) * 3
    )
    assert sum(len(miss_events(run)) for run in loaded) > sum(
        len(miss_events(run)) for run in nominal
    )
