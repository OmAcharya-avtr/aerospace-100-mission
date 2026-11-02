"""The run-level verdict: its order of operations and its four outcomes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hilverdict.clock import CLOCK_CLEAN_VERDICT, CLOCK_DRIFTING, CLOCK_RESET
from hilverdict.synthetic import ClockSpec, generate_run
from hilverdict.timeline import RunLog, miss_events
from hilverdict.verdict import (
    NOT_TIMING_FAITHFUL,
    TIMING_FAITHFUL,
    UNDECIDABLE,
    UNDECIDED_RUN,
    run_verdict,
)


def _clean(seed: int = 51, steps: int = 100) -> RunLog:
    log = generate_run(seed=seed, profile="nominal", amplitude_deg=3.0, steps=steps)
    return RunLog(
        model=log.model,
        command_rad=log.command_rad,
        durations_s=np.full(log.steps, 0.004),
        states=log.states,
        computed_rad=log.computed_rad,
        saturated=log.saturated,
        host_time_s=log.host_time_s,
        device_time_s=log.device_time_s,
        label="clean",
    )


def test_a_run_with_no_overrun_is_timing_faithful():
    verdict = run_verdict(_clean(), 1.0)
    assert verdict.verdict == TIMING_FAITHFUL
    assert verdict.faithful is True
    assert verdict.misses == 0
    assert verdict.resimulation_saving == pytest.approx(1.0, rel=0.0, abs=0.0)


def test_a_clean_run_reports_a_clean_clock():
    assert run_verdict(_clean(), 1.0).clock.verdict == CLOCK_CLEAN_VERDICT


def test_the_verdict_rejects_a_non_positive_tolerance():
    with pytest.raises(ValueError, match="tolerance must be positive"):
        run_verdict(_clean(), 0.0)


def test_a_reset_makes_the_run_undecidable():
    spec = ClockSpec(drift_ppm=0.0, jitter_s=2e-7, tick_s=1e-7,
                     reset_fractions=(0.5,), name="t")
    log = generate_run(seed=52, profile="loaded", amplitude_deg=6.0, steps=100,
                       clock=spec)
    verdict = run_verdict(log, 1.0)
    assert verdict.verdict == UNDECIDABLE
    assert verdict.clock.verdict == CLOCK_RESET
    assert verdict.certificate is None
    assert math.isnan(verdict.aggregate_bound)
    assert any("reset" in reason for reason in verdict.reasons)


def test_a_reset_still_reports_the_miss_count():
    spec = ClockSpec(drift_ppm=0.0, jitter_s=2e-7, tick_s=1e-7,
                     reset_fractions=(0.5,), name="t")
    log = generate_run(seed=52, profile="loaded", amplitude_deg=6.0, steps=100,
                       clock=spec)
    verdict = run_verdict(log, 1.0)
    assert verdict.misses == len(miss_events(log))


def test_a_drift_beyond_the_allowance_is_corrected_and_recorded():
    spec = ClockSpec(drift_ppm=20000.0, jitter_s=2e-7, tick_s=1e-7,
                     reset_fractions=(), name="t")
    log = generate_run(seed=53, profile="loaded", amplitude_deg=6.0, steps=200,
                       clock=spec)
    verdict = run_verdict(log, 1.0)
    assert verdict.clock.verdict == CLOCK_DRIFTING
    assert any("divided by 1 + rho" in reason for reason in verdict.reasons)
    assert verdict.verdict in {TIMING_FAITHFUL, NOT_TIMING_FAITHFUL}


def test_the_drift_correction_can_be_switched_off():
    spec = ClockSpec(drift_ppm=20000.0, jitter_s=2e-7, tick_s=1e-7,
                     reset_fractions=(), name="t")
    log = generate_run(seed=53, profile="loaded", amplitude_deg=6.0, steps=200,
                       clock=spec)
    corrected = run_verdict(log, 1.0, correct_drift=True)
    raw = run_verdict(log, 1.0, correct_drift=False)
    assert corrected.misses != raw.misses or corrected.reasons != raw.reasons


def test_a_saturating_run_is_undecidable():
    log = generate_run(seed=54, profile="loaded", amplitude_deg=14.0, steps=200)
    verdict = run_verdict(log, 1.0)
    assert verdict.verdict == UNDECIDABLE
    assert verdict.certificate is not None
    assert any("saturat" in reason for reason in verdict.reasons)


def test_an_overloaded_run_is_undecidable():
    log = generate_run(seed=55, profile="loaded", amplitude_deg=6.0, steps=200)
    overloaded = RunLog(
        model=log.model,
        command_rad=log.command_rad,
        durations_s=np.asarray(log.durations_s) * 2.0,
        states=log.states,
        computed_rad=log.computed_rad,
        saturated=log.saturated,
        host_time_s=log.host_time_s,
        device_time_s=log.device_time_s,
    )
    assert run_verdict(overloaded, 1.0).verdict == UNDECIDABLE


def test_not_resimulating_the_residual_leaves_the_run_undecided(long_run):
    verdict = run_verdict(long_run, 1.0, resimulate_undecided=False)
    assert verdict.verdict in {TIMING_FAITHFUL, UNDECIDED_RUN}
    if verdict.verdict == UNDECIDED_RUN:
        assert verdict.resimulated == 0
        assert any("left undecided" in reason for reason in verdict.reasons)


def test_the_verdict_only_resimulates_the_residual(long_run):
    verdict = run_verdict(long_run, 1.0)
    assert verdict.resimulated == verdict.misses - verdict.dismissed


def test_the_resimulation_saving_is_the_dismissal_rate(long_run):
    verdict = run_verdict(long_run, 1.0)
    assert verdict.resimulation_saving == pytest.approx(
        verdict.dismissed / verdict.misses, rel=1e-15, abs=0.0
    )


def test_a_consequential_miss_makes_the_run_not_faithful():
    found = None
    for seed in range(60, 90):
        verdict = run_verdict(
            generate_run(seed=seed, profile="loaded", amplitude_deg=8.0, steps=200),
            1.0,
        )
        if verdict.verdict == NOT_TIMING_FAITHFUL:
            found = verdict
            break
    assert found is not None
    assert found.consequential
    assert found.faithful is False
    assert any("moved the trajectory past" in reason for reason in found.reasons)


def test_a_looser_tolerance_never_makes_a_run_less_faithful():
    log = generate_run(seed=61, profile="loaded", amplitude_deg=8.0, steps=200)
    tight = run_verdict(log, 0.2)
    loose = run_verdict(log, 5.0)
    assert len(loose.consequential) <= len(tight.consequential)


def test_the_aggregate_bound_matches_the_certificate(long_run):
    verdict = run_verdict(long_run, 1.0)
    assert verdict.certificate is not None
    assert verdict.aggregate_bound == pytest.approx(
        verdict.certificate.aggregate_bound, rel=0.0, abs=0.0
    )


def test_the_aggregate_dismissed_flag_is_the_comparison(long_run):
    verdict = run_verdict(long_run, 1.0)
    assert verdict.aggregate_dismissed == (verdict.aggregate_bound < 1.0)


def test_a_clean_run_has_an_aggregate_bound_of_zero():
    verdict = run_verdict(_clean(), 1.0)
    assert verdict.aggregate_bound == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert verdict.aggregate_dismissed is True


def test_the_cascade_ratio_is_reported(long_run):
    verdict = run_verdict(long_run, 1.0)
    assert verdict.cascade_ratio >= 1.0


def test_a_clean_run_has_an_infinite_cascade_ratio():
    assert math.isinf(run_verdict(_clean(), 1.0).cascade_ratio)


def test_the_verdict_reports_the_utilisation(long_run):
    verdict = run_verdict(long_run, 1.0)
    assert 0.0 < verdict.utilisation < 1.0


def test_a_faithful_verdict_still_explains_itself(long_run):
    for seed in range(70, 110):
        verdict = run_verdict(
            generate_run(seed=seed, profile="nominal", amplitude_deg=3.0, steps=200),
            1.0,
        )
        if verdict.verdict == TIMING_FAITHFUL:
            from hilverdict.report import render_verdict

            assert "dismissed in closed form" in render_verdict(verdict)
            return
    raise AssertionError("no faithful run found in the declared seed range")
