"""Harness statistics, measurement-method propagation and timing recovery."""

from __future__ import annotations

import time

import numpy as np
import pytest

from edgeinfer.backends import PipelineStage, SimulatedBackend, spin_wait
from edgeinfer.harness import LatencyProfile, benchmark, time_calls


class TestTimeCalls:
    def test_returns_one_sample_per_repeat(self) -> None:
        samples = time_calls(lambda: None, repeats=17, warmup=3)
        assert samples.shape == (17,)
        assert np.all(samples >= 0.0)

    def test_warmup_calls_are_made_but_not_retained(self) -> None:
        calls = []
        time_calls(lambda: calls.append(1), repeats=5, warmup=4)
        assert len(calls) == 9
        assert time_calls(lambda: None, repeats=5, warmup=4).shape == (5,)

    def test_arguments_are_forwarded(self) -> None:
        seen = []
        time_calls(lambda a, b=0: seen.append((a, b)), 7, repeats=2, b=9)
        assert seen == [(7, 9), (7, 9)]

    def test_gc_is_restored_afterwards(self) -> None:
        import gc

        was_enabled = gc.isenabled()
        time_calls(lambda: None, repeats=3, disable_gc=True)
        assert gc.isenabled() == was_enabled

    @pytest.mark.parametrize(
        ("kwargs", "match"), [({"repeats": 0}, "repeats"), ({"warmup": -1}, "warmup")]
    )
    def test_invalid_counts_are_rejected(self, kwargs: dict, match: str) -> None:
        base = {"repeats": 3}
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            time_calls(lambda: None, **base)


class TestSpinWait:
    def test_a_spin_consumes_at_least_the_requested_time(self) -> None:
        """A busy-wait on perf_counter may overshoot by one loop iteration but
        must never return early, or an injected cost model could not be
        recovered from the measurement."""
        for target in (50e-6, 200e-6):
            t0 = time.perf_counter()
            spin_wait(target)
            assert time.perf_counter() - t0 >= target

    def test_zero_returns_immediately(self) -> None:
        t0 = time.perf_counter()
        spin_wait(0.0)
        assert time.perf_counter() - t0 < 1e-3

    def test_negative_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            spin_wait(-1e-6)


class TestLatencyProfile:
    def _profile(self, samples: np.ndarray) -> LatencyProfile:
        from edgeinfer.environment import environment_record

        return LatencyProfile(
            label="synthetic",
            samples_s=samples,
            n_repeats=samples.size,
            n_warmup=0,
            clock="perf_counter",
            clock_resolution_s=1e-9,
            timer_bias_s=4e-8,
            gc_disabled=True,
            environment=environment_record(),
        )

    def test_statistics_are_the_numpy_ones(self) -> None:
        samples = np.arange(1, 101, dtype=float) * 1e-6
        profile = self._profile(samples)
        assert profile.mean_s == pytest.approx(np.mean(samples))
        assert profile.p50_s == pytest.approx(np.quantile(samples, 0.50))
        assert profile.p99_s == pytest.approx(np.quantile(samples, 0.99))
        assert profile.max_s == pytest.approx(samples.max())
        assert profile.min_s == pytest.approx(samples.min())
        assert profile.std_s == pytest.approx(np.std(samples, ddof=1))

    def test_tail_ratio_is_p99_over_p50(self) -> None:
        samples = np.arange(1, 101, dtype=float) * 1e-6
        profile = self._profile(samples)
        assert profile.tail_ratio == pytest.approx(profile.p99_s / profile.p50_s)

    def test_a_heavy_tail_shows_up_as_a_large_tail_ratio(self) -> None:
        """The whole reason worst case is reported separately: these samples
        have a median of 10 us and a 5 % chance of 1 ms."""
        samples = np.concatenate([np.full(190, 10e-6), np.full(10, 1e-3)])
        profile = self._profile(samples)
        assert profile.p50_s == pytest.approx(10e-6)
        assert profile.max_s == pytest.approx(1e-3)
        assert profile.tail_ratio > 10.0

    def test_p99_cannot_see_a_one_in_two_hundred_outlier_but_max_can(self) -> None:
        """A stated limitation of the p99 figure, asserted rather than left to
        be discovered: with n repeats, the 99th percentile is blind to an
        event rarer than 1 in 100, so the maximum is reported alongside it and
        the repeat count travels with both."""
        samples = np.concatenate([np.full(199, 10e-6), [1e-3]])
        profile = self._profile(samples)
        assert profile.p99_s == pytest.approx(10e-6, rel=0.2)
        assert profile.max_s == pytest.approx(1e-3)

    def test_std_is_nan_for_a_single_sample(self) -> None:
        assert np.isnan(self._profile(np.array([1e-6])).std_s)

    def test_out_of_range_quantile_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="quantile"):
            self._profile(np.arange(1, 11, dtype=float)).quantile_s(1.5)

    def test_method_line_names_the_clock_and_the_repeat_count(self) -> None:
        line = self._profile(np.arange(1, 11, dtype=float)).method_line()
        assert "perf_counter" in line
        assert "n=10" in line

    def test_summary_reports_the_tail_before_the_median(self) -> None:
        text = "\n".join(self._profile(np.arange(1, 101, dtype=float)).summary_lines())
        assert text.index("p99") < text.index("p50")

    def test_uncertainty_for_the_mean_uses_the_standard_error(self) -> None:
        samples = np.arange(1, 101, dtype=float) * 1e-6
        budget = self._profile(samples).uncertainty("mean")
        expected = np.std(samples, ddof=1) / np.sqrt(samples.size)
        assert budget.type_a_s == pytest.approx(expected)
        assert budget.n_samples == 100
        assert "GUM" in budget.method

    def test_uncertainty_for_a_quantile_is_bootstrapped(self) -> None:
        samples = np.arange(1, 101, dtype=float) * 1e-6
        budget = self._profile(samples).uncertainty("p99", n_resamples=200, seed=1)
        assert budget.type_a_s > 0
        assert "bootstrap" in budget.method
        assert "200 resamples" in budget.method

    def test_an_unknown_statistic_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="mean' or 'pNN"):
            self._profile(np.arange(1, 11, dtype=float)).uncertainty("median")

    def test_a_malformed_quantile_label_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="mean' or 'pNN"):
            self._profile(np.arange(1, 11, dtype=float)).uncertainty("pXX")


class TestBenchmark:
    def test_profile_carries_the_environment_and_extras(self) -> None:
        profile = benchmark(
            lambda: None,
            label="noop",
            repeats=20,
            warmup=2,
            timer_bias_samples=50,
            extra={"backend": "none"},
        )
        assert profile.label == "noop"
        assert profile.n_repeats == 20
        assert profile.n_warmup == 2
        assert profile.extra["backend"] == "none"
        assert profile.environment.cpu_count_available is not None
        assert profile.timer_bias_s >= 0.0

    def test_environment_note_reaches_the_environment_line(self) -> None:
        profile = benchmark(
            lambda: None,
            label="noop",
            repeats=5,
            warmup=0,
            timer_bias_samples=20,
            environment_note="CONTAINER NUMBERS ONLY",
        )
        assert "CONTAINER NUMBERS ONLY" in profile.environment.one_line()

    def test_measured_mean_recovers_an_injected_cost_model(self) -> None:
        """The instrument check: inject a known per-stage cost, measure it
        through the harness, and require the measured mean back.

        The spin-wait overshoots by at most one loop iteration per stage and
        the host is shared, so the measured mean is expected to be *above* the
        injected mean. The assertion is therefore one-sided on the low side
        and loose on the high side, and the measured bias is printed by
        ``validation/validate_backend_cost_model.py`` rather than asserted
        tightly here.
        """
        stages = (
            PipelineStage("preprocess", 120e-6, 0.0, "constant"),
            PipelineStage("infer", 240e-6, 0.0, "constant"),
        )
        backend = SimulatedBackend(stages, seed=4, consume_time=True)
        backend.prepare()
        profile = benchmark(
            backend.infer, label="injected", repeats=60, warmup=5, timer_bias_samples=100
        )
        injected = backend.injected_mean_s
        assert profile.p50_s >= injected
        assert profile.p50_s < 1.6 * injected
