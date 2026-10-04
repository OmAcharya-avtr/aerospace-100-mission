"""The benchmark harness and its record."""

from __future__ import annotations

import json

import pytest

from hilforge.backends import make_backend_pair
from hilforge.bench import run_benchmark, write_record
from hilforge.errors import ConfigurationError

NOTE = "unit test, shared single-core cloud container"


def test_benchmark_runs_and_records_the_method():
    sim, _ = make_backend_pair()
    record, run = run_benchmark(
        sim, n_iterations=200, environment_note=NOTE, label="unit-test"
    )
    assert record.n_completed == 200
    assert run.n_completed == 200
    assert record.is_hardware is False
    assert record.environment_note == NOTE
    assert record.method["clock"].startswith("time.perf_counter_ns")
    assert record.method["pacing"] == "none; the loop is not slept to the period"
    assert record.throughput_iter_s > 0.0
    assert record.clock_measured_resolution_s > 0.0
    assert record.duration_resolution_uncertainty_s > 0.0


def test_benchmark_record_states_the_level_4_caveat():
    sim, _ = make_backend_pair()
    record, _ = run_benchmark(sim, n_iterations=50, environment_note=NOTE)
    assert "Level 4 validation is NOT claimed" in record.caveat
    assert "Jetson Orin Nano" in record.caveat
    assert "single-core cloud container" in record.caveat
    assert "hardware numbers" in record.caveat


def test_benchmark_text_contains_every_stage_and_the_uncertainty_budget():
    sim, _ = make_backend_pair()
    record, _ = run_benchmark(sim, n_iterations=100, environment_note=NOTE)
    text = record.as_text()
    for stage in ("sense", "estimate", "control", "actuate", "total"):
        assert stage in text
    assert "u_resolution_s" in text
    assert "expanded_k2_s" in text
    assert "is_hardware           : False" in text


def test_benchmark_requires_an_environment_note():
    sim, _ = make_backend_pair()
    with pytest.raises(ConfigurationError, match="environment_note"):
        run_benchmark(sim, n_iterations=10, environment_note="   ")


def test_benchmark_validates_its_arguments():
    sim, _ = make_backend_pair()
    with pytest.raises(ConfigurationError):
        run_benchmark(sim, n_iterations=1, environment_note=NOTE)
    with pytest.raises(ConfigurationError):
        run_benchmark(sim, period_s=0.0, n_iterations=10, environment_note=NOTE)


def test_benchmark_dry_run_path_issues_no_write():
    sim, _ = make_backend_pair()
    sim.open()
    actuator = sim.actuators["torque"]
    record, run = run_benchmark(
        sim, n_iterations=100, environment_note=NOTE, dry_run=True
    )
    assert record.dry_run is True
    assert run.rehearsed_writes == 100
    assert actuator.write_count == 0
    sim.close()


def test_write_record_emits_text_and_json(tmp_path):
    sim, _ = make_backend_pair()
    record, _ = run_benchmark(sim, n_iterations=50, environment_note=NOTE)
    txt, js = write_record(record, tmp_path / "bench")
    assert txt.exists() and js.exists()
    payload = json.loads(js.read_text())
    assert payload["is_hardware"] is False
    assert payload["environment_note"] == NOTE
    assert "caveat" in payload
    assert "HilForge benchmark record" in txt.read_text()


def test_device_backend_benchmark_is_still_not_hardware():
    _, dev = make_backend_pair()
    record, _ = run_benchmark(
        dev, n_iterations=50, environment_note=NOTE, label="loopback"
    )
    assert record.backend_kind == "device"
    assert record.backend_driver == "loopback"
    assert record.is_hardware is False


def test_maxrss_note_matches_the_platform():
    sim, _ = make_backend_pair()
    record, _ = run_benchmark(sim, n_iterations=20, environment_note=NOTE)
    assert "ru_maxrss interpreted as" in record.maxrss_note
    assert record.maxrss_bytes > 0.0
    assert record.tracemalloc_peak_bytes > 0.0
