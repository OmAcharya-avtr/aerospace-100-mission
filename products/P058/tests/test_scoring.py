"""Scoring tests: run lengths, ARL estimators, censoring bookkeeping, intervals."""

from __future__ import annotations

import numpy as np
import pytest

from telemdrift.detectors import ADWIN, CUSUM, Detector, WindowedKS
from telemdrift.scoring import (
    blind_fraction,
    bootstrap_mean_ci,
    measure_arl0,
    measure_arl1,
    run_lengths_on_stream,
    wilson_interval,
)
from telemdrift.streams import ChangeSpec, change_stream, stationary


class EveryNth(Detector):
    """A deterministic detector that alarms on every ``n``-th sample.

    Used so the ARL machinery can be tested against arithmetic rather than
    against another estimate: with ``n = 10`` on 100 samples the run lengths are
    exactly ten tens and ARL0 is exactly 10.0.
    """

    name = "EveryNth"

    def __init__(self, n: int = 10, warmup: int = 0):
        self.n = n
        self.warmup = warmup
        self.count = 0

    @property
    def threshold(self) -> float:
        return float(self.n)

    @threshold.setter
    def threshold(self, value: float) -> None:
        self.n = int(value)

    @staticmethod
    def default_threshold() -> float:
        return 10.0

    def alarm_ratio(self) -> float:
        return self.count / self.n

    def is_armed(self) -> bool:
        return self.count >= self.warmup

    def reset(self) -> None:
        self.count = 0

    def update(self, x: float) -> bool:
        self.count += 1
        if self.count >= self.n:
            self.count = 0
            return True
        return False


def test_run_lengths_on_a_deterministic_detector_are_exact():
    lengths, tail = run_lengths_on_stream(EveryNth(10), np.zeros(100))
    assert lengths == [10] * 10
    assert tail == 0


def test_censored_tail_is_reported_not_absorbed():
    lengths, tail = run_lengths_on_stream(EveryNth(10), np.zeros(107))
    assert lengths == [10] * 10
    assert tail == 7


def test_arl0_of_the_deterministic_detector_is_exactly_the_period():
    res = measure_arl0(lambda: EveryNth(10), lambda L, s: np.zeros(L), [1, 2, 3], 100)
    assert res.arl0 == 10.0
    assert res.n_runs == 30
    assert res.sem == 0.0
    assert res.total_samples == 300
    assert res.censored_fraction == 0.0


def test_arl0_sem_is_the_sample_standard_error():
    res = measure_arl0(
        lambda: CUSUM(h=3.0), lambda L, s: stationary(L, s), [41, 42], 20_000
    )
    expected = res.run_lengths.std(ddof=1) / np.sqrt(res.run_lengths.size)
    assert res.sem == pytest.approx(expected)
    assert res.relative_sem == pytest.approx(res.sem / res.arl0)


def test_arl0_returns_the_budget_as_a_lower_bound_when_nothing_alarms():
    """A detector that never fires must not produce a fabricated ARL0."""
    res = measure_arl0(
        lambda: CUSUM(h=1e9), lambda L, s: stationary(L, s), [1, 2], 5_000
    )
    assert res.n_runs == 0
    assert res.arl0 == 10_000.0  # the whole budget
    assert np.isnan(res.sem)
    assert np.isnan(res.relative_sem)


def test_arl0_summary_mentions_the_relative_standard_error():
    res = measure_arl0(lambda: EveryNth(10), lambda L, s: np.zeros(L), [1], 100)
    assert "rel. SEM" in res.summary()


@pytest.mark.parametrize("kwargs,msg", [
    ({"seeds": [], "stream_length": 10}, "at least one seed"),
    ({"seeds": [1], "stream_length": 0}, "stream_length must be >= 1"),
])
def test_arl0_validates_its_inputs(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        measure_arl0(lambda: EveryNth(10), lambda L, s: np.zeros(L), **kwargs)


def test_arl1_uses_every_replicate_and_drops_none():
    res = measure_arl1(
        lambda: CUSUM(h=4.0),
        lambda pre, post, s: change_stream(pre, post, ChangeSpec("mean_step", 2.0), s),
        range(100, 160),
        300,
        400,
    )
    assert res.n_used == res.n_replicates == 60
    assert res.delays.size == 60


def test_arl1_on_the_deterministic_detector_is_exact():
    """EveryNth(10) with pre_length 100 is reset at sample 99, so the first alarm
    at or after the change index 100 lands at 109: a delay of exactly 9."""
    res = measure_arl1(
        lambda: EveryNth(10),
        lambda pre, post, s: (np.zeros(pre + post), pre),
        [1, 2, 3],
        100,
        50,
    )
    assert res.arl1 == 9.0
    assert res.n_censored == 0
    assert res.n_pre_change_alarms == 3


def test_arl1_censors_at_the_budget_and_flags_a_lower_bound():
    res = measure_arl1(
        lambda: CUSUM(h=1e9),
        lambda pre, post, s: change_stream(pre, post, ChangeSpec("mean_step", 1.0), s),
        [1, 2, 3, 4],
        100,
        200,
    )
    assert res.arl1 == 200.0
    assert res.n_censored == 4
    assert res.censored_rate == 1.0
    assert res.is_lower_bound
    assert "LOWER BOUND" in res.summary()


def test_arl1_reports_unarmed_at_change_for_a_windowed_detector():
    """The windowed KS test resets to a 300-sample warm-up after every false
    alarm, so with an ARL0 well under the pre-change length it is frequently
    blind when the change arrives. The rate is a measurement, not a guess."""
    res = measure_arl1(
        lambda: WindowedKS(c=0.1),
        lambda pre, post, s: change_stream(pre, post, ChangeSpec("mean_step", 1.0), s),
        range(200, 230),
        1_000,
        600,
    )
    assert 0.0 < res.blind_at_change_rate <= 1.0
    assert res.n_reset_at_change > 0


def test_arl1_on_a_detector_with_no_warmup_is_never_unarmed():
    res = measure_arl1(
        lambda: CUSUM(h=4.0),
        lambda pre, post, s: change_stream(pre, post, ChangeSpec("mean_step", 1.0), s),
        range(300, 320),
        500,
        400,
    )
    assert res.blind_at_change_rate == 0.0


@pytest.mark.parametrize("kwargs,msg", [
    ({"seeds": [], "pre_length": 10, "budget": 10}, "at least one seed"),
    ({"seeds": [1], "pre_length": 10, "budget": 0}, "budget must be >= 1"),
])
def test_arl1_validates_its_inputs(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        measure_arl1(
            lambda: EveryNth(10), lambda pre, post, s: (np.zeros(pre + post), pre), **kwargs
        )


def test_self_resetting_detector_is_not_reset_by_the_harness():
    """ADWIN in shrink mode manages its own post-alarm state. If the harness
    reset it anyway, the shrink convention could not be measured at all."""
    shrink = ADWIN(delta=0.002, shrink_on_detect=True)
    reset = ADWIN(delta=0.002, shrink_on_detect=False)
    x = stationary(20_000, 51)
    a, _ = run_lengths_on_stream(shrink, x)
    b, _ = run_lengths_on_stream(reset, x)
    assert a != b
    assert shrink.n > 0  # window survived its own alarms


def test_blind_fraction_is_zero_for_a_detector_without_warmup():
    assert blind_fraction(CUSUM(h=4.0), stationary(5_000, 52)) == 0.0


def test_blind_fraction_is_substantial_for_the_windowed_ks_test():
    """300-sample warm-up against an ARL0 of a few hundred: most of its life."""
    frac = blind_fraction(WindowedKS(c=0.135), stationary(20_000, 53))
    assert 0.3 < frac < 0.95


def test_blind_fraction_of_an_empty_stream_is_zero():
    assert blind_fraction(CUSUM(h=4.0), np.asarray([])) == 0.0


def test_bootstrap_ci_brackets_the_mean_of_a_tight_sample():
    values = np.full(500, 7.0)
    lo, hi = bootstrap_mean_ci(values, seed=1)
    assert lo == pytest.approx(7.0)
    assert hi == pytest.approx(7.0)


def test_bootstrap_ci_contains_the_sample_mean_for_a_skewed_sample():
    rng = np.random.default_rng(54)
    values = rng.exponential(30.0, size=400)
    lo, hi = bootstrap_mean_ci(values, seed=2)
    assert lo < values.mean() < hi


def test_bootstrap_ci_is_reproducible_given_a_seed():
    rng = np.random.default_rng(55)
    v = rng.exponential(5.0, 200)
    assert bootstrap_mean_ci(v, seed=3) == bootstrap_mean_ci(v, seed=3)


def test_bootstrap_ci_of_an_empty_sample_is_nan():
    lo, hi = bootstrap_mean_ci(np.asarray([]))
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_ci_rejects_an_invalid_level():
    with pytest.raises(ValueError, match="0 < level < 1"):
        bootstrap_mean_ci(np.ones(10), level=1.0)


def test_wilson_interval_hand_computed_at_a_half():
    """n = 4, successes = 2, z = 1.959964.

    denom  = 1 + z^2/n = 1 + 3.841459/4 = 1.960365
    centre = (0.5 + 3.841459/8) / 1.960365 = (0.5 + 0.480182) / 1.960365
           = 0.980182 / 1.960365 = 0.5
    half   = (1.959964 / 1.960365) * sqrt(0.25/4 + 3.841459/64)
           = 0.999795 * sqrt(0.0625 + 0.060023) = 0.999795 * 0.350033
           = 0.349961
    => (0.150039, 0.849961)
    """
    lo, hi = wilson_interval(2, 4)
    assert lo == pytest.approx(0.150039, abs=1e-5)
    assert hi == pytest.approx(0.849961, abs=1e-5)


def test_wilson_interval_is_non_degenerate_at_the_boundaries():
    lo0, hi0 = wilson_interval(0, 300)
    lo1, hi1 = wilson_interval(300, 300)
    # Exactly 0 and exactly 1: the Wilson bounds are analytically degenerate at
    # the endpoints and are set exactly, so a measured rate of 1.0 is never
    # outside its own interval. See wilson_interval.
    assert lo0 == 0.0 and 0.0 < hi0 < 0.02
    assert hi1 == 1.0 and 0.98 < lo1 < 1.0


@pytest.mark.parametrize("s,n", [(-1, 10), (11, 10)])
def test_wilson_interval_rejects_impossible_counts(s, n):
    with pytest.raises(ValueError, match="successes must satisfy"):
        wilson_interval(s, n)


def test_wilson_interval_rejects_zero_trials():
    with pytest.raises(ValueError, match="trials must be >= 1"):
        wilson_interval(0, 0)
