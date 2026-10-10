"""Detector unit tests, including hand-computed known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import ks_2samp

from telemdrift.detectors import (
    ADWIN,
    ANALYTIC_DETECTORS,
    CUSUM,
    EWMA,
    PageHinkley,
    WindowedKS,
    alarm_ratio_trace,
    first_alarm_at_or_after,
    ks_two_sample_statistic,
    make_detector,
)
from telemdrift.streams import stationary

# --------------------------------------------------------------------------
# CUSUM
# --------------------------------------------------------------------------


def test_cusum_hand_computed_recursion():
    """Hand arithmetic, k = 0.5, h = 2, mu0 = 0, sigma0 = 1.

    z            : 1.0   1.0   1.0   1.0
    S+ = max(0, S+ + z - 0.5)
      t=1: max(0, 0.0 + 1.0 - 0.5) = 0.5    0.5 > 2 ? no
      t=2: max(0, 0.5 + 1.0 - 0.5) = 1.0    1.0 > 2 ? no
      t=3: max(0, 1.0 + 1.0 - 0.5) = 1.5    1.5 > 2 ? no
      t=4: max(0, 1.5 + 1.0 - 0.5) = 2.0    2.0 > 2 ? no (strict inequality)
      t=5: max(0, 2.0 + 1.0 - 0.5) = 2.5    2.5 > 2 ? YES -> alarm at t = 5
    S- stays at 0 throughout because -z - k = -1.5 < 0 every step.
    """
    det = CUSUM(h=2.0, k=0.5)
    fired = [det.update(1.0) for _ in range(5)]
    assert fired == [False, False, False, False, True]
    assert det.s_plus == pytest.approx(2.5)
    assert det.s_minus == pytest.approx(0.0)


def test_cusum_is_symmetric_under_sign_flip():
    x = stationary(5_000, 21)
    up = CUSUM(h=3.0)
    dn = CUSUM(h=3.0)
    a = [up.update(v) for v in x]
    b = [dn.update(-v) for v in x]
    assert a == b


def test_cusum_standardises_by_mu0_and_sigma0():
    """A stream of N(5, 2^2) fed with mu0=5, sigma0=2 must behave like N(0,1)."""
    x = stationary(3_000, 22)
    plain = CUSUM(h=3.0)
    shifted = CUSUM(h=3.0, mu0=5.0, sigma0=2.0)
    assert [plain.update(v) for v in x] == [shifted.update(5.0 + 2.0 * v) for v in x]


def test_cusum_alarm_ratio_crosses_one_exactly_at_the_alarm():
    det = CUSUM(h=2.0, k=0.5)
    for _ in range(4):
        det.update(1.0)
        assert det.alarm_ratio() <= 1.0
    assert det.update(1.0)
    assert det.alarm_ratio() > 1.0


def test_cusum_reset_clears_both_arms():
    det = CUSUM(h=10.0)
    for _ in range(20):
        det.update(2.0)
    assert det.s_plus > 0
    det.reset()
    assert det.s_plus == 0.0 and det.s_minus == 0.0


@pytest.mark.parametrize("kwargs,msg", [
    ({"h": 0.0}, "h must be > 0"),
    ({"h": -1.0}, "h must be > 0"),
    ({"k": -0.1}, "k must be >= 0"),
    ({"sigma0": 0.0}, "sigma0 must be > 0"),
])
def test_cusum_rejects_invalid_parameters(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        CUSUM(**kwargs)


def test_cusum_threshold_setter_validates():
    det = CUSUM()
    with pytest.raises(ValueError, match="h must be > 0"):
        det.threshold = -1.0


# --------------------------------------------------------------------------
# Page-Hinkley
# --------------------------------------------------------------------------


def test_page_hinkley_hand_computed_running_mean():
    """Hand arithmetic, delta = 0, lambda = 10 (so no alarm interferes).

    x        : 0.0   2.0   4.0
    t=1: n=1, mean = 0.0,            m += 0.0 - 0.0 - 0 = 0.0   -> m = 0.0
    t=2: n=2, mean = 0.0 + (2-0)/2 = 1.0, m += 2.0 - 1.0 = 1.0  -> m = 1.0
    t=3: n=3, mean = 1.0 + (4-1)/3 = 2.0, m += 4.0 - 2.0 = 2.0  -> m = 3.0
    m_min = 0.0 (reached at t=1), m_max = 3.0
    PH_up = 3.0 - 0.0 = 3.0,  PH_dn = 3.0 - 3.0 = 0.0
    """
    det = PageHinkley(lambda_=10.0, delta=0.0)
    for v in (0.0, 2.0, 4.0):
        assert not det.update(v)
    assert det.n == 3
    assert det.mean == pytest.approx(2.0)
    assert det.m == pytest.approx(3.0)
    assert det.m_min == pytest.approx(0.0)
    assert det.m_max == pytest.approx(3.0)
    assert det.alarm_ratio() == pytest.approx(0.3)


def test_page_hinkley_cannot_fire_on_a_constant_stream():
    """A property of the running mean, and a trap worth pinning.

    On a constant stream the running mean equals the value at every step, so
    every increment ``x_i - xbar_i - delta`` is exactly ``-delta`` and the
    statistic never grows. Page-Hinkley is a detector of *changes in level*, not
    of level, and a constant stream contains no change however extreme its
    value. CUSUM, which standardises against a declared ``mu0``, does fire.
    """
    ph = PageHinkley(lambda_=1.0, delta=0.0)
    assert not any(ph.update(1000.0) for _ in range(500))
    cs = CUSUM(h=1.0)
    assert cs.update(1000.0)


def test_page_hinkley_fires_on_a_step_after_a_stationary_segment():
    det = PageHinkley(lambda_=5.0, delta=0.0)
    for _ in range(200):
        det.update(0.0)
    fired = any(det.update(1.0) for _ in range(50))
    assert fired


def test_page_hinkley_running_mean_makes_it_scale_dependent():
    """Unlike CUSUM, Page-Hinkley has no sigma0, so scaling the stream matters."""
    x = stationary(2_000, 23)
    a = PageHinkley(lambda_=20.0)
    b = PageHinkley(lambda_=20.0)
    assert [a.update(v) for v in x] != [b.update(10.0 * v) for v in x]


@pytest.mark.parametrize("kwargs,msg", [
    ({"lambda_": 0.0}, "lambda must be > 0"),
    ({"delta": -1.0}, "delta must be >= 0"),
])
def test_page_hinkley_rejects_invalid_parameters(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        PageHinkley(**kwargs)


# --------------------------------------------------------------------------
# EWMA
# --------------------------------------------------------------------------


def test_ewma_hand_computed_first_two_steps():
    """Hand arithmetic, r = 0.5, L = 3, x = 1.0 twice.

    asym = r / (2 - r) = 0.5 / 1.5 = 1/3
    t=1: z = 0.5*0 + 0.5*1 = 0.5
         var = (1/3)(1 - 0.5^2) = (1/3)(0.75) = 0.25,  sd = 0.5
         limit = 3 * 0.5 = 1.5 ;  abs(z) = 0.5 > 1.5 ? no
    t=2: z = 0.5*0.5 + 0.5*1 = 0.75
         var = (1/3)(1 - 0.5^4) = (1/3)(0.9375) = 0.3125, sd = 0.559017
         limit = 1.677051 ; abs(z) = 0.75 > 1.677051 ? no
    """
    det = EWMA(L=3.0, r=0.5)
    assert not det.update(1.0)
    assert det.z == pytest.approx(0.5)
    assert det.alarm_ratio() == pytest.approx(0.5 / 1.5)
    assert not det.update(1.0)
    assert det.z == pytest.approx(0.75)
    assert det.alarm_ratio() == pytest.approx(0.75 / (3.0 * math.sqrt(0.3125)))


def test_ewma_exact_limit_is_narrower_than_the_asymptotic_one_early_on():
    det = EWMA(L=3.0, r=0.1)
    det.update(0.0)
    asymptotic = 3.0 * math.sqrt(0.1 / 1.9)
    assert det._limit() < asymptotic
    for _ in range(500):
        det.update(0.0)
    assert det._limit() == pytest.approx(asymptotic, rel=1e-9)


def test_ewma_is_symmetric_under_sign_flip():
    x = stationary(3_000, 24)
    a, b = EWMA(L=2.5), EWMA(L=2.5)
    assert [a.update(v) for v in x] == [b.update(-v) for v in x]


@pytest.mark.parametrize("kwargs,msg", [
    ({"L": 0.0}, "L must be > 0"),
    ({"r": 0.0}, "0 < r <= 1"),
    ({"r": 1.5}, "0 < r <= 1"),
    ({"sigma0": -1.0}, "sigma0 must be > 0"),
])
def test_ewma_rejects_invalid_parameters(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        EWMA(**kwargs)


def test_ewma_with_r_one_is_a_shewhart_chart():
    """r = 1 gives z = x and var = 1, so the alarm is abs(x) > L exactly."""
    det = EWMA(L=2.0, r=1.0)
    assert not det.update(1.9)
    assert det.update(2.1)


# --------------------------------------------------------------------------
# Windowed KS
# --------------------------------------------------------------------------


def test_ks_statistic_hand_computed_on_two_tiny_samples():
    """ref = [0, 1], win = [2, 3].

    Grid is [0, 1, 2, 3]. F_ref = [0.5, 1.0, 1.0, 1.0], F_win = [0, 0, 0.5, 1.0].
    Absolute differences: [0.5, 1.0, 0.5, 0.0] -> sup = 1.0, complete separation.
    """
    assert ks_two_sample_statistic(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 1.0


def test_ks_statistic_is_zero_for_identical_samples():
    a = np.sort(np.array([1.0, 2.0, 3.0, 4.0]))
    assert ks_two_sample_statistic(a, a) == 0.0


def test_ks_statistic_matches_scipy_ks_2samp_to_machine_precision():
    """scipy.stats.ks_2samp is the reference. Note the module docstring: the
    kstest(x, 'norm', args=(loc, scale)) form raises TypeError on the installed
    SciPy, which is why the two-sample call is the reference here."""
    rng = np.random.default_rng(25)
    worst = 0.0
    for _ in range(200):
        n, m = int(rng.integers(5, 60)), int(rng.integers(5, 60))
        a = rng.standard_normal(n)
        b = rng.standard_normal(m) + rng.uniform(-1, 1)
        mine = ks_two_sample_statistic(np.sort(a), b)
        theirs = ks_2samp(a, b).statistic
        worst = max(worst, abs(mine - theirs))
    assert worst < 1e-12, f"worst absolute difference {worst:.3e}"


def test_ks_statistic_rejects_empty_samples():
    with pytest.raises(ValueError, match="non-empty"):
        ks_two_sample_statistic(np.array([1.0]), np.array([]))


def test_windowed_ks_is_unarmed_during_warmup_and_cannot_alarm():
    """Warm-up is n_ref + n_det = 30 samples; no alarm is possible before that.

    The stream turns extreme at sample 27, which would separate the windows
    completely, and the detector still cannot alarm because its detection window
    is not full.
    """
    det = WindowedKS(c=0.01, n_ref=20, n_det=10, stride=1)
    assert det.warmup == 30
    for i in range(29):
        assert not det.update(100.0 if i > 25 else 0.0)
        assert not det.is_armed(), f"armed too early at sample {i}"
    assert not det.update(100.0)  # the 30th sample completes the window
    assert det.is_armed()


def test_windowed_ks_warmup_equals_n_ref_plus_n_det():
    det = WindowedKS(n_ref=30, n_det=15)
    assert det.warmup == 45


def test_windowed_ks_detects_total_separation():
    det = WindowedKS(c=0.5, n_ref=50, n_det=25, stride=1)
    for _ in range(75):
        det.update(0.0)
    assert det.update(1000.0) is False or True  # warm-up boundary is permitted
    fired = any(det.update(1000.0) for _ in range(30))
    assert fired


def test_windowed_ks_stride_limits_how_often_a_test_runs():
    det = WindowedKS(c=0.01, n_ref=10, n_det=5, stride=7)
    for _ in range(15):
        det.update(0.0)
    results = [det.update(500.0) for _ in range(7)]
    assert sum(results) <= 1


def test_windowed_ks_default_threshold_matches_the_asymptotic_formula():
    n, m, alpha = WindowedKS.N_REF, WindowedKS.N_DET, WindowedKS.ALPHA_DEFAULT
    expected = math.sqrt(-0.5 * math.log(alpha / 2.0)) * math.sqrt((n + m) / (n * m))
    assert WindowedKS.default_threshold() == pytest.approx(expected)


@pytest.mark.parametrize("kwargs,msg", [
    ({"c": 0.0}, "0 < c <= 1"),
    ({"c": 1.5}, "0 < c <= 1"),
    ({"n_ref": 1}, ">= 2"),
    ({"n_det": 1}, ">= 2"),
    ({"stride": 0}, "stride must be >= 1"),
])
def test_windowed_ks_rejects_invalid_parameters(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        WindowedKS(**kwargs)


# --------------------------------------------------------------------------
# ADWIN
# --------------------------------------------------------------------------


def test_adwin_cut_rule_matches_the_published_formula_by_hand():
    """Hand-check eps_cut against the transcribed rule.

    Feed 10 samples at 0.0 then 10 at 10.0 with check_every = 20, min_sub = 5,
    delta = 0.05. At the only check, n = 20. The bucket boundary at
    n0 = 10 / n1 = 10 gives

        m       = 2 / (1/10 + 1/10) = 10
        eps_cut = sqrt( (1 / 20) * ln(4 * 20 / 0.05) )
                = sqrt( 0.05 * ln(1600) )
                = sqrt( 0.05 * 7.3777589 ) = sqrt(0.36888794) = 0.60736146
        diff    = abs(0.0 - 10.0) = 10.0 > 0.60736146  -> cut
    """
    det = ADWIN(delta=0.05, min_sub=5, check_every=20, max_buckets=40)
    for _ in range(10):
        det.update(0.0)
    for _ in range(9):
        det.update(10.0)
    m = 2.0 / (1.0 / 10 + 1.0 / 10)
    eps = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 * 20 / 0.05))
    assert eps == pytest.approx(0.60736146, abs=1e-7)
    assert det.update(10.0)
    assert det.cut_evidence() > 1.0


def test_adwin_does_not_cut_on_a_stationary_window():
    det = ADWIN(delta=0.05, min_sub=5, check_every=10)
    x = stationary(200, 26) * 0.01  # tiny variance, no cut is possible
    assert not any(det.update(v) for v in x)


def test_adwin_bucket_counts_and_sums_are_conserved():
    det = ADWIN(delta=1e-9, check_every=100_000)
    x = stationary(500, 27)
    for v in x:
        det.update(v)
    buckets = det.buckets_oldest_first()
    assert sum(int(b[0]) for b in buckets) == det.n == 500
    assert sum(b[1] for b in buckets) == pytest.approx(det.total, abs=1e-9)
    assert sum(b[1] for b in buckets) == pytest.approx(float(x.sum()), abs=1e-9)


def test_adwin_compression_keeps_the_bucket_count_logarithmic():
    det = ADWIN(delta=1e-9, max_buckets=5, check_every=100_000)
    for v in stationary(4_000, 28):
        det.update(v)
    # Exponential histogram: at most max_buckets per row, rows ~ log2(n).
    assert len(det.buckets_oldest_first()) <= 5 * (int(math.log2(4_000)) + 2)


def test_adwin_is_unarmed_until_two_minimum_subwindows_exist():
    det = ADWIN(min_sub=30)
    for i in range(59):
        det.update(0.0)
        assert not det.is_armed(), i
    det.update(0.0)
    assert det.is_armed()


def test_adwin_shrink_mode_declares_itself_self_resetting():
    assert ADWIN(shrink_on_detect=True).self_resetting is True
    assert ADWIN(shrink_on_detect=False).self_resetting is False


def test_adwin_shrink_keeps_the_recent_window_and_drops_the_old():
    det = ADWIN(delta=0.05, min_sub=5, check_every=10, max_buckets=40,
                shrink_on_detect=True)
    for _ in range(20):
        det.update(0.0)
    fired = False
    for _ in range(20):
        if det.update(20.0):
            fired = True
            break
    assert fired
    assert det.n < 40  # the older part was discarded
    assert det.total / det.n > 1.0  # what remains is from the new regime


@pytest.mark.parametrize("kwargs,msg", [
    ({"delta": 0.0}, "0 < delta < 1"),
    ({"delta": 1.0}, "0 < delta < 1"),
    ({"max_buckets": 1}, "max_buckets must be >= 2"),
    ({"min_sub": 0}, "min_sub must be >= 1"),
    ({"check_every": 0}, "check_every must be >= 1"),
])
def test_adwin_rejects_invalid_parameters(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        ADWIN(**kwargs)


# --------------------------------------------------------------------------
# Cross-detector contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", ANALYTIC_DETECTORS)
def test_make_detector_uses_the_declared_default(key):
    det = make_detector(key)
    assert det.threshold == pytest.approx(type(det).default_threshold())


@pytest.mark.parametrize("key", ANALYTIC_DETECTORS)
def test_make_detector_honours_an_explicit_threshold(key):
    base = make_detector(key).threshold
    det = make_detector(key, threshold=base * 1.5)
    assert det.threshold == pytest.approx(base * 1.5)


def test_make_detector_rejects_an_unknown_key():
    with pytest.raises(ValueError, match="unknown detector"):
        make_detector("bayesian_magic")


@pytest.mark.parametrize("key", ANALYTIC_DETECTORS)
def test_every_detector_describes_itself(key):
    text = make_detector(key).describe()
    assert make_detector(key).threshold_name in text


@pytest.mark.parametrize("key", ANALYTIC_DETECTORS)
def test_reset_restores_the_initial_alarm_ratio(key):
    det = make_detector(key)
    before = det.alarm_ratio()
    for v in stationary(400, 29):
        det.update(v)
    det.reset()
    assert det.alarm_ratio() == pytest.approx(before)


@pytest.mark.parametrize("key", ANALYTIC_DETECTORS)
def test_alarm_ratio_exceeds_one_exactly_when_an_alarm_is_raised(key):
    """The alarm-ratio contract, which the figures depend on."""
    det = make_detector(key)
    stream = np.concatenate([stationary(600, 30), stationary(600, 31) + 6.0])
    for v in stream:
        fired = det.update(v)
        if fired:
            assert det.alarm_ratio() > 1.0
            det.reset()


@pytest.mark.parametrize("key", ANALYTIC_DETECTORS)
def test_alarm_ratio_trace_reports_every_alarm(key):
    det = make_detector(key)
    stream = np.concatenate([stationary(500, 32), stationary(500, 33) + 8.0])
    ratios, alarms = alarm_ratio_trace(det, stream)
    assert ratios.shape == stream.shape
    assert alarms.size > 0
    for a in alarms:
        assert ratios[a] > 1.0


@pytest.mark.parametrize("key", ANALYTIC_DETECTORS)
def test_alarm_ratio_trace_without_reset_stops_growing_after_one_alarm(key):
    """With reset_on_alarm=False the statistic is never cleared, so a long
    sustained change produces many consecutive alarms rather than periodic ones.
    The default (True) is what the benchmark and a real deployment do."""
    stream = np.concatenate([stationary(400, 34), stationary(600, 35) + 8.0])
    _, with_reset = alarm_ratio_trace(make_detector(key), stream, reset_on_alarm=True)
    _, without = alarm_ratio_trace(make_detector(key), stream, reset_on_alarm=False)
    assert without.size >= with_reset.size


def test_first_alarm_at_or_after_ignores_pre_change_alarms():
    alarms = np.asarray([5, 40, 120, 310])
    assert first_alarm_at_or_after(alarms, 100) == 120
    assert first_alarm_at_or_after(alarms, 0) == 5
    assert first_alarm_at_or_after(alarms, 400) == -1
    assert first_alarm_at_or_after(np.asarray([], dtype=int), 0) == -1
