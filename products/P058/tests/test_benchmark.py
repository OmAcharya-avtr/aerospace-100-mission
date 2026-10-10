"""Benchmark harness tests, including the batch/online identity for run lengths."""

from __future__ import annotations

import numpy as np
import pytest

from telemdrift.benchmark import (
    DETECTOR_LABELS,
    SCORED_CHANGES,
    STANDARD,
    BenchmarkConfig,
    _run_lengths_from_scores,
    analytic_factory,
    ar1_stream_fn,
    blind_fraction_from_scores,
    blind_fraction_table,
    calibrate_learned_threshold,
    change_stream_fn,
    default_threshold_operating_points,
    detector_instance,
    learned_arl0,
    learned_arl1,
    measure_change_response,
    stationary_stream_fn,
    timed,
    tradeoff_curve,
    transient_response,
)
from telemdrift.detectors import ANALYTIC_DETECTORS
from telemdrift.learned import LearnedDetector, score_stream
from telemdrift.scoring import run_lengths_on_stream
from telemdrift.streams import ChangeSpec, stationary


def test_standard_config_is_the_documented_one():
    assert STANDARD.target_arl0 == 500.0
    assert STANDARD.pre_length == 1_000
    assert STANDARD.arl1_budget == 1_500
    assert STANDARD.window == 50
    assert len(STANDARD.cal_seeds) == 6
    assert len(STANDARD.eval_seeds) == 8


def test_pre_length_exceeds_the_longest_detector_warmup():
    """The windowed KS test needs 300 samples; the config must clear it."""
    from telemdrift.detectors import WindowedKS

    assert STANDARD.pre_length > WindowedKS().warmup


def test_arl1_seeds_are_distinct_between_experiments():
    a = set(STANDARD.arl1_seeds(0))
    b = set(STANDARD.arl1_seeds(1))
    c = set(STANDARD.arl1_seeds(2))
    assert len(a) == STANDARD.arl1_replicates
    assert not (a & b) and not (a & c) and not (b & c)


def test_detector_labels_cover_every_key():
    for key in ANALYTIC_DETECTORS:
        assert key in DETECTOR_LABELS
    assert "learned" in DETECTOR_LABELS


def test_scored_changes_include_the_transient_negative_control():
    kinds = {spec.kind for _, spec in SCORED_CHANGES}
    assert kinds == {"mean_step", "variance_step", "drift_ramp", "transient"}


@pytest.mark.parametrize("key", ANALYTIC_DETECTORS)
def test_analytic_factory_sets_the_threshold(key):
    value = 0.123 if key in ("ks", "adwin") else 7.0
    assert analytic_factory(key, value)().threshold == pytest.approx(value)
    assert detector_instance(key, value).threshold == pytest.approx(value)


def test_detector_instance_requires_a_model_for_the_learned_detector():
    with pytest.raises(ValueError, match="model is required"):
        detector_instance("learned", 0.5)


def test_stationary_stream_fn_is_deterministic():
    assert np.array_equal(stationary_stream_fn(100, 5), stationary_stream_fn(100, 5))


def test_ar1_stream_fn_produces_the_declared_autocorrelation():
    x = ar1_stream_fn(0.7)(50_000, 6)
    assert abs(np.corrcoef(x[:-1], x[1:])[0, 1] - 0.7) < 0.03


def test_change_stream_fn_places_the_change_at_pre_length():
    stream, idx = change_stream_fn(ChangeSpec("mean_step", 1.0))(50, 60, 7)
    assert idx == 50
    assert stream.size == 110


def test_timed_returns_the_result_and_a_positive_duration():
    out, seconds = timed(lambda a, b: a + b, 2, 3)
    assert out == 5
    assert seconds >= 0.0


def test_default_threshold_operating_points_spread_over_decades():
    """The product's opening claim, measured on a reduced budget here.

    The five shipped defaults are not one operating point. If this ever came out
    within a factor of two, the premise of the package would be wrong and the
    README would need rewriting rather than the test relaxing.
    """
    cfg = BenchmarkConfig(eval_seeds=(58_201, 58_202), eval_length=20_000)
    res = default_threshold_operating_points(cfg)
    arl0s = [r.arl0 for r in res.values()]
    assert max(arl0s) / min(arl0s) > 20.0


def test_learned_run_length_batch_path_equals_the_online_detector(small_model):
    """The identity that licenses every learned ARL0 figure.

    _run_lengths_from_scores simulates the alarm logic over a precomputed score
    series; run_lengths_on_stream runs the real detector. Equality must be exact.
    """
    x = stationary(6_000, 11)
    p = 0.3
    scores = score_stream(small_model, x, 50)
    batch_lengths, batch_tail = _run_lengths_from_scores(scores, p, 50)
    online_lengths, online_tail = run_lengths_on_stream(
        LearnedDetector(small_model, p, 50), x
    )
    assert batch_lengths == online_lengths
    assert batch_tail == online_tail


def test_blind_fraction_from_scores_matches_the_online_definition(small_model):
    x = stationary(3_000, 12)
    scores = score_stream(small_model, x, 50)
    batch = blind_fraction_from_scores(scores, 0.3, 50)
    from telemdrift.scoring import blind_fraction

    online = blind_fraction(LearnedDetector(small_model, 0.3, 50), x)
    assert batch == pytest.approx(online)


def test_learned_arl0_reports_runs_and_a_standard_error(small_model):
    cfg = BenchmarkConfig(eval_seeds=(58_201, 58_202), eval_length=20_000)
    res = learned_arl0(small_model, 0.4, cfg)
    assert res.n_runs > 0
    assert res.sem > 0.0
    assert res.total_samples == 40_000


def test_learned_arl0_on_an_unreachable_threshold_returns_the_budget(small_model):
    cfg = BenchmarkConfig(eval_seeds=(58_201,), eval_length=5_000)
    res = learned_arl0(small_model, 0.999_999, cfg)
    assert res.n_runs == 0
    assert res.arl0 == 5_000.0


def test_calibrate_learned_threshold_lands_near_the_target(small_model):
    cfg = BenchmarkConfig(
        cal_seeds=(58_101, 58_102),
        eval_seeds=(58_201, 58_202),
        cal_length=20_000,
        eval_length=20_000,
    )
    res = calibrate_learned_threshold(small_model, cfg, target_arl0=400.0)
    assert 0.0 < res.threshold < 1.0
    assert abs(res.target_error) < 0.6, res.summary()


def test_learned_arl1_uses_every_replicate(small_model):
    res = learned_arl1(
        small_model, 0.4, ChangeSpec("mean_step", 2.0), STANDARD, replicates=20
    )
    assert res.n_used == res.n_replicates == 20


def test_measure_change_response_requires_a_model_for_the_learned_detector():
    with pytest.raises(ValueError, match="model is required"):
        measure_change_response("learned", 0.5, ChangeSpec("mean_step", 1.0))


def test_tradeoff_curve_is_monotone_in_arl0_over_the_threshold_sweep():
    """ARL0 increases with the CUSUM decision interval. The curve is the
    product's headline figure and a non-monotone ARL0 axis would make it
    unreadable, so the monotonicity is pinned rather than assumed."""
    cfg = BenchmarkConfig(sweep_seeds=(58_301, 58_302), sweep_length=20_000)
    pts = tradeoff_curve(
        "cusum", [3.0, 4.0, 5.0], ChangeSpec("mean_step", 1.0), cfg, replicates=20
    )
    arl0s = [p.arl0.arl0 for p in pts]
    assert arl0s == sorted(arl0s)
    arl1s = [p.arl1.arl1 for p in pts]
    assert arl1s[0] <= arl1s[-1]  # a wider threshold cannot detect faster


def test_tradeoff_curve_requires_a_model_for_the_learned_detector():
    with pytest.raises(ValueError, match="model is required"):
        tradeoff_curve("learned", [0.5], ChangeSpec("mean_step", 1.0))


def test_transient_response_reports_both_rates_and_their_difference():
    cfg = BenchmarkConfig()
    out = transient_response("cusum", 5.0, 4.0, 20, cfg, replicates=25)
    assert out["replicates"] == 25
    assert out["window"] == 70
    assert 0.0 <= out["baseline_rate"] <= 1.0
    assert out["excess"] == pytest.approx(out["transient_rate"] - out["baseline_rate"])
    assert out["transient_lo"] <= out["transient_rate"] <= out["transient_hi"]


def test_transient_response_requires_a_model_for_the_learned_detector():
    with pytest.raises(ValueError, match="model is required"):
        transient_response("learned", 0.5, 4.0, 20, replicates=2)


def test_blind_fraction_table_reports_zero_for_memoryless_detectors():
    out = blind_fraction_table(
        {"cusum": 5.0, "ewma": 2.8, "ks": 0.135}, seeds=[58_201], stream_length=10_000
    )
    assert out["cusum"] == 0.0
    assert out["ewma"] == 0.0
    assert out["ks"] > 0.2


def test_blind_fraction_table_requires_a_model_for_the_learned_detector():
    with pytest.raises(ValueError, match="model is required"):
        blind_fraction_table({"learned": 0.5}, seeds=[1], stream_length=100)
