"""Benchmark harness: shape, determinism of structure, and honest labelling."""

from __future__ import annotations

import time

import pytest

from constellink.benchmark import (
    BenchmarkResult,
    Environment,
    benchmark,
    clock_resolution_s,
    environment,
)


def test_clock_resolution_is_positive_and_small():
    res = clock_resolution_s(500)
    assert 0.0 < res < 1e-3


def test_clock_resolution_rejects_tiny_probe():
    with pytest.raises(ValueError):
        clock_resolution_s(5)


def test_environment_records_the_container_caveat():
    env = environment()
    assert isinstance(env, Environment)
    assert env.clock_resolution_s > 0.0
    assert "not a measurement on target" in env.note
    block = env.format_block()
    assert "python_version" in block
    assert "numpy_version" in block


def test_benchmark_statistics_are_ordered():
    r = benchmark("sleep", lambda: time.sleep(0.005), repeats=3)
    assert r.repeats == 3
    assert len(r.samples_s) == 3
    assert r.min_s <= r.median_s <= r.max_s
    assert r.min_s >= 0.004      # the sleep really happened
    assert "sleep" in r.format_row()
    assert "min[ms]" in BenchmarkResult.header()


def test_benchmark_measures_memory_when_asked():
    def allocate():
        return [0] * 200_000

    with_mem = benchmark("alloc", allocate, repeats=1, measure_memory=True)
    without = benchmark("alloc", allocate, repeats=1, measure_memory=False)
    assert with_mem.peak_python_bytes > 100_000
    assert without.peak_python_bytes == 0


def test_benchmark_rejects_zero_repeats():
    with pytest.raises(ValueError):
        benchmark("x", lambda: None, repeats=0)


def test_benchmark_runs_the_function_exactly_repeats_times_for_timing():
    calls = []
    benchmark("count", lambda: calls.append(1), repeats=4,
              measure_memory=False)
    assert len(calls) == 4


def test_memory_pass_is_an_extra_call_not_a_timed_one():
    calls = []
    r = benchmark("count", lambda: calls.append(1), repeats=2,
                  measure_memory=True)
    assert len(calls) == 3        # two timed plus one memory pass
    assert len(r.samples_s) == 2
