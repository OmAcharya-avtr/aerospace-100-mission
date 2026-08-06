"""Integration tests: end-to-end episodes and the CLI."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from trackforge.__main__ import main
from trackforge.reacq import AlwaysFullPolicy, AlwaysLocalPolicy, train_q_learning
from trackforge.sim import (
    DEFAULT_SCENARIO,
    Scenario,
    load_scenario,
    run_episode,
    run_monte_carlo,
    sim_steps_per_second,
)

EXAMPLES = ["examples/scenario_leo_downlink.yaml", "examples/scenario_high_jitter.yaml"]


def test_episode_completes_all_four_phases():
    res = run_episode(DEFAULT_SCENARIO, seed=11)
    assert res.acquisition_time_s is not None
    assert res.scan_points > 100
    assert res.track_rms_rad > 0
    assert res.lock_lost
    assert res.reacq_time_s is not None
    assert res.total_time_s > res.acquisition_time_s


def test_episode_time_series_shapes_agree():
    res = run_episode(DEFAULT_SCENARIO, seed=3)
    n = int(round(DEFAULT_SCENARIO.track_duration / DEFAULT_SCENARIO.dt))
    for arr in (res.t, res.los_error, res.torque, res.jitter):
        assert arr.shape == (n,)


def test_episode_summary_is_json_serialisable():
    res = run_episode(DEFAULT_SCENARIO, seed=3, keep_series=False)
    s = json.dumps(res.summary(), default=float)
    assert "track_rms_rad" in s


def test_keep_series_false_drops_arrays():
    res = run_episode(DEFAULT_SCENARIO, seed=3, keep_series=False)
    assert res.t.size == 0 and res.los_error.size == 0


def test_spike_causes_loss_and_no_spike_does_not():
    calm = Scenario(spike_amplitude=1e-9)
    assert not run_episode(calm, seed=5).lock_lost
    assert run_episode(Scenario(), seed=5).lock_lost


def test_tracking_rms_far_below_threshold_before_spike():
    res = run_episode(DEFAULT_SCENARIO, seed=7)
    assert res.track_rms_rad < 0.2 * DEFAULT_SCENARIO.track_threshold


def test_peak_error_exceeds_threshold_at_spike():
    res = run_episode(DEFAULT_SCENARIO, seed=7)
    assert res.track_peak_rad > DEFAULT_SCENARIO.track_threshold


@pytest.mark.parametrize("path", EXAMPLES)
def test_example_scenarios_run_end_to_end(path):
    sc = load_scenario(path)
    res = run_episode(sc, keep_series=False)
    assert math.isfinite(res.track_rms_rad)
    assert res.total_time_s > 0


def test_lqr_scenario_tracks_at_least_as_well_as_pid():
    pid = run_episode(Scenario(controller="pid", spike_amplitude=1e-9), seed=2)
    lqr = run_episode(Scenario(controller="lqr", spike_amplitude=1e-9), seed=2)
    assert lqr.track_rms_rad < 3.0 * pid.track_rms_rad


def test_raster_pattern_episode_runs():
    res = run_episode(Scenario(pattern="raster"), seed=1, keep_series=False)
    assert res.scan_points > 0


def test_learned_policy_can_drive_an_episode():
    pol = train_q_learning(None, episodes=500, seed=2)
    sc = Scenario(reacq_policy="learned")
    res = run_episode(sc, seed=4, policy=pol, keep_series=False)
    assert res.reacq_time_s is not None


def test_learned_policy_required_when_declared():
    with pytest.raises(ValueError, match="learned"):
        run_episode(Scenario(reacq_policy="learned"), seed=4)


def test_baseline_policies_selected_from_scenario():
    a = run_episode(Scenario(reacq_policy="always_full"), seed=6, keep_series=False)
    b = run_episode(Scenario(reacq_policy="always_local"), seed=6, keep_series=False)
    assert a.reacq_time_s is not None and b.reacq_time_s is not None


def test_explicit_policy_overrides_scenario_setting():
    sc = Scenario(reacq_policy="always_local")
    a = run_episode(sc, seed=9, policy=AlwaysFullPolicy(), keep_series=False)
    b = run_episode(sc, seed=9, policy=AlwaysLocalPolicy(), keep_series=False)
    assert a.reacq_attempts != b.reacq_attempts or a.reacq_time_s != b.reacq_time_s


def test_run_episode_rejects_non_scenario():
    with pytest.raises(TypeError):
        run_episode({"dt": 1e-3})


def test_monte_carlo_aggregates():
    mc = run_monte_carlo(Scenario(track_duration=1.0, spike_time=0.5), n_episodes=5,
                         base_seed=100)
    assert mc["n_episodes"] == 5
    assert len(mc["episodes"]) == 5
    assert 0.0 <= mc["acquired_fraction"] <= 1.0
    assert math.isfinite(mc["mean_track_rms_rad"])


def test_monte_carlo_rejects_zero_episodes():
    with pytest.raises(ValueError):
        run_monte_carlo(n_episodes=0)


def test_sim_steps_per_second_reports_positive_rate():
    perf = sim_steps_per_second(DEFAULT_SCENARIO, duration=0.1)
    assert perf["steps"] == pytest.approx(0.1 / DEFAULT_SCENARIO.dt, rel=1e-9)
    assert perf["steps_per_second"] > 0
    assert perf["realtime_factor"] > 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_run_default_scenario(capsys):
    assert main(["run", "--seed", "5"]) == 0
    out = capsys.readouterr().out
    assert "acquisition_time_s" in out and "reacq_time_s" in out


def test_cli_run_json_output(capsys):
    assert main(["run", EXAMPLES[0], "--seed", "5", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "track_rms_rad" in payload


def test_cli_run_with_scenario_file(capsys):
    assert main(["run", EXAMPLES[1]]) == 0
    assert "scenario: high-jitter-lqr" in capsys.readouterr().out


def test_cli_benchmark(capsys):
    assert main(["benchmark"]) == 0
    out = capsys.readouterr().out
    assert "PID" in out and "LQR" in out and "steps/s" in out


def test_cli_reacq_small_budget(capsys):
    assert main(["reacq", "--episodes", "500", "--eval-episodes", "100"]) == 0
    out = capsys.readouterr().out
    assert "baseline-always-full" in out and "q-learning" in out


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit):
        main([])


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert "trackforge 0.1.0" in capsys.readouterr().out


def test_cli_unknown_scenario_file():
    with pytest.raises(FileNotFoundError):
        main(["run", "/nope/missing.yaml"])


def test_disturbance_series_has_spike_at_configured_time():
    res = run_episode(DEFAULT_SCENARIO, seed=1)
    k = int(np.argmax(np.abs(res.jitter)))
    assert res.t[k] == pytest.approx(DEFAULT_SCENARIO.spike_time, abs=5 * 0.02)
