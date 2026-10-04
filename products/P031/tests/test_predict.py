"""Trace generation, causal features, the two baselines and the learned model.

Sizes here are kept small on purpose: the whole file must finish in well under
a minute on one core. The full-size comparison lives in
``validation/predictor_benchmark.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hilforge.errors import ConfigurationError
from hilforge.predict import (
    FEATURE_NAMES,
    FixedThresholdPredictor,
    LearnedOverrunPredictor,
    OverrunMetrics,
    QueueingOverrunPredictor,
    TraceConfig,
    build_dataset,
    evaluate_predictor,
    generate_trace,
    lead_time_stats,
    make_features,
    split_trace,
    wilson_interval,
)
from hilforge.predict.baselines import md1_mean_wait_s
from hilforge.predict.features import make_labels
from hilforge.predict.metrics import cutoff_for_flag_rate

HORIZON = 3
N = 6000
SEED = 4242


def _dataset(preset="bursty", n=N, seed=SEED):
    cfg = TraceConfig.preset(preset, n_iterations=n)
    trace = generate_trace(cfg, seed=seed)
    train, test = split_trace(trace, train_fraction=0.6)
    deadline = cfg.period_s
    x_tr, y_tr, _ = build_dataset(train.stage_s, deadline, horizon=HORIZON)
    x_te, y_te, idx = build_dataset(test.stage_s, deadline, horizon=HORIZON)
    over = test.overruns()
    window = np.column_stack([over[idx + h] for h in range(1, HORIZON + 1)])
    return deadline, x_tr, y_tr, x_te, y_te, window, trace


# -- trace generation -------------------------------------------------------


def test_trace_is_reproducible_from_the_seed():
    cfg = TraceConfig.preset("bursty", n_iterations=2000)
    a = generate_trace(cfg, seed=11)
    b = generate_trace(cfg, seed=11)
    assert a.stage_s.tobytes() == b.stage_s.tobytes()
    assert a.regime.tobytes() == b.regime.tobytes()
    c = generate_trace(cfg, seed=12)
    assert c.stage_s.tobytes() != a.stage_s.tobytes()


def test_trace_shapes_and_totals():
    cfg = TraceConfig.preset("jittery", n_iterations=1000)
    trace = generate_trace(cfg, seed=1)
    assert trace.stage_s.shape == (1000, 4)
    assert trace.totals_s.shape == (1000,)
    assert np.allclose(trace.totals_s, trace.stage_s.sum(axis=1))
    assert np.all(trace.stage_s > 0.0)
    assert trace.n == 1000


def test_bursty_overruns_are_autocorrelated_and_jittery_ones_are_not():
    def acf1(flags: np.ndarray) -> float:
        x = flags.astype(np.float64)
        x = x - x.mean()
        return float(np.dot(x[:-1], x[1:]) / np.dot(x, x))

    jit = generate_trace(TraceConfig.preset("jittery", n_iterations=20000), seed=11)
    bur = generate_trace(TraceConfig.preset("bursty", n_iterations=20000), seed=11)
    assert abs(acf1(jit.overruns())) < 0.10
    assert acf1(bur.overruns()) > 0.60


def test_trace_statistics_match_the_markov_closed_forms():
    cfg = TraceConfig.preset("bursty", n_iterations=60000)
    trace = generate_trace(cfg, seed=5)
    # Stationary burst probability p / (p + q) for a two-state chain.
    expected = cfg.p_calm_to_burst / (cfg.p_calm_to_burst + cfg.p_burst_to_calm)
    assert cfg.stationary_burst_probability == pytest.approx(expected)
    assert float(trace.regime.mean()) == pytest.approx(expected, abs=0.03)
    assert cfg.expected_burst_length == pytest.approx(1.0 / cfg.p_burst_to_calm)


def test_utilisation_is_mean_over_period():
    trace = generate_trace(TraceConfig.preset("bursty", n_iterations=2000), seed=2)
    assert trace.utilisation == pytest.approx(
        float(trace.totals_s.mean()) / trace.config.period_s
    )
    assert 0.0 < trace.utilisation < 1.0


def test_trace_config_validation():
    with pytest.raises(ConfigurationError):
        TraceConfig(period_s=0.0)
    with pytest.raises(ConfigurationError):
        TraceConfig(n_iterations=10)
    with pytest.raises(ConfigurationError):
        TraceConfig(stage_mean_s=(1.0, 1.0, 1.0, 1.0))
    with pytest.raises(ConfigurationError):
        TraceConfig(stage_shape=(1.0, 1.0, 1.0, -1.0))
    with pytest.raises(ConfigurationError):
        TraceConfig(burst_scale=1.0)
    with pytest.raises(ConfigurationError):
        TraceConfig(p_calm_to_burst=0.0)
    with pytest.raises(ConfigurationError):
        TraceConfig(p_burst_to_calm=1.0)
    with pytest.raises(ConfigurationError):
        TraceConfig(interrupt_prob=1.0)
    with pytest.raises(ConfigurationError):
        TraceConfig(interrupt_mean_s=-1.0)
    with pytest.raises(ConfigurationError):
        TraceConfig.preset("quiet")


def test_split_is_chronological_and_validated():
    trace = generate_trace(TraceConfig.preset("bursty", n_iterations=1000), seed=3)
    train, test = split_trace(trace, train_fraction=0.6)
    assert train.n == 600 and test.n == 400
    assert np.array_equal(train.stage_s, trace.stage_s[:600])
    assert np.array_equal(test.stage_s, trace.stage_s[600:])
    with pytest.raises(ConfigurationError):
        split_trace(trace, train_fraction=0.0)
    with pytest.raises(ConfigurationError):
        split_trace(trace, train_fraction=0.999)


# -- features ---------------------------------------------------------------


def test_feature_matrix_shape_and_names():
    trace = generate_trace(TraceConfig.preset("bursty", n_iterations=500), seed=1)
    x = make_features(trace.stage_s, trace.config.period_s)
    assert x.shape == (500, len(FEATURE_NAMES))
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_features_are_causal():
    """Changing iteration k must not change any feature row before k."""
    trace = generate_trace(TraceConfig.preset("bursty", n_iterations=400), seed=9)
    base = make_features(trace.stage_s, trace.config.period_s)
    perturbed = trace.stage_s.copy()
    perturbed[250, :] *= 7.0
    after = make_features(perturbed, trace.config.period_s)
    assert np.array_equal(base[:250], after[:250])
    assert not np.array_equal(base[250], after[250])


def test_feature_known_answers_on_a_hand_built_trace():
    # Four stages, constant 1, 2, 3, 4 s -> total 10 s every iteration.
    stage = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (40, 1))
    x = make_features(stage, deadline_s=20.0)
    cols = {name: i for i, name in enumerate(FEATURE_NAMES)}
    row = x[39]
    assert row[cols["total_last_s"]] == 10.0
    assert row[cols["total_lag1_s"]] == 10.0
    assert row[cols["total_mean8_s"]] == 10.0
    assert row[cols["total_max8_s"]] == 10.0
    assert row[cols["total_std8_s"]] == pytest.approx(0.0)
    assert row[cols["total_slope4_s"]] == pytest.approx(0.0)
    assert row[cols["overruns_in_8"]] == 0.0
    # headroom = (D - d) / D = (20 - 10) / 20 = 0.5; util = 10/20 = 0.5
    assert row[cols["headroom_frac"]] == pytest.approx(0.5)
    assert row[cols["util_last"]] == pytest.approx(0.5)
    assert row[cols["sense_last_s"]] == 1.0
    assert row[cols["actuate_last_s"]] == 4.0


def test_ewma_known_answer():
    # alpha = 0.30, constant input 10 -> the EWMA stays at 10 exactly.
    stage = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (10, 1))
    x = make_features(stage, deadline_s=20.0)
    col = FEATURE_NAMES.index("total_ewma_s")
    assert x[0, col] == pytest.approx(10.0)
    assert x[-1, col] == pytest.approx(10.0)


def test_labels_known_answer():
    # d = [1, 9, 1, 1, 9, 1], D = 5, H = 2 -> overruns at 1 and 4.
    #   y[0] = any(d[1], d[2]) > 5 -> True
    #   y[1] = any(d[2], d[3])     -> False
    #   y[2] = any(d[3], d[4])     -> True
    #   y[3] = any(d[4], d[5])     -> True
    #   y[4], y[5] = False (inside the trailing horizon)
    y = make_labels(np.array([1.0, 9.0, 1.0, 1.0, 9.0, 1.0]), 5.0, horizon=2)
    assert y.tolist() == [True, False, True, True, False, False]


def test_labels_validation():
    with pytest.raises(ConfigurationError):
        make_labels(np.ones(5), 0.0)
    with pytest.raises(ConfigurationError):
        make_labels(np.ones(5), 1.0, horizon=0)


def test_build_dataset_drops_warmup_and_horizon():
    trace = generate_trace(TraceConfig.preset("bursty", n_iterations=200), seed=4)
    x, y, idx = build_dataset(trace.stage_s, trace.config.period_s, horizon=HORIZON)
    assert idx[0] == 32
    assert idx[-1] == 200 - HORIZON - 1
    assert x.shape[0] == y.size == idx.size == 200 - 32 - HORIZON


def test_build_dataset_validation():
    trace = generate_trace(TraceConfig.preset("bursty", n_iterations=60), seed=4)
    with pytest.raises(ConfigurationError):
        build_dataset(trace.stage_s, trace.config.period_s, warmup=0)
    with pytest.raises(ConfigurationError):
        build_dataset(trace.stage_s, trace.config.period_s, warmup=100)
    with pytest.raises(ConfigurationError):
        make_features(np.ones((10, 3)), 0.01)
    with pytest.raises(ValueError):
        make_features(-np.ones((40, 4)), 0.01)
    with pytest.raises(ConfigurationError):
        make_features(np.ones((40, 4)), 0.0)


# -- baselines --------------------------------------------------------------


def test_fixed_threshold_fits_and_describes():
    deadline, x_tr, y_tr, x_te, _y_te, _, _ = _dataset()
    model = FixedThresholdPredictor(deadline_s=deadline).fit(x_tr, y_tr)
    assert model.sweep_ is not None and model.sweep_.shape[1] == 2
    assert model.threshold_s > 0.0
    assert 0.0 <= model.negative_rate_ <= model.positive_rate_ <= 1.0
    assert "threshold" in model.describe()
    proba = model.predict_proba(x_te)
    assert set(np.unique(proba)).issubset({model.positive_rate_, model.negative_rate_})
    assert model.predict(x_te).dtype == bool


def test_fixed_threshold_requires_fit():
    model = FixedThresholdPredictor(deadline_s=0.01)
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((2, len(FEATURE_NAMES))))


def test_fixed_threshold_validation():
    with pytest.raises(ConfigurationError):
        FixedThresholdPredictor(deadline_s=0.0)
    with pytest.raises(ConfigurationError):
        FixedThresholdPredictor(deadline_s=0.01, n_candidates=1)
    model = FixedThresholdPredictor(deadline_s=0.01)
    with pytest.raises(ConfigurationError):
        model.fit(np.zeros((3, len(FEATURE_NAMES))), np.zeros(4, dtype=bool))
    with pytest.raises(ConfigurationError):
        model.calibrate_flag_rate(np.zeros((3, len(FEATURE_NAMES))), 0.0)


def test_queueing_baseline_fits_a_two_state_chain():
    deadline, x_tr, y_tr, x_te, _y_te, _, _ = _dataset()
    model = QueueingOverrunPredictor(deadline_s=deadline, horizon=HORIZON).fit(x_tr, y_tr)
    assert model.transition_.shape == (2, 2)
    assert np.allclose(model.transition_.sum(axis=1), 1.0)
    pi = model.stationary_distribution()
    assert pi.sum() == pytest.approx(1.0)
    assert len(model.fits_) == 2
    assert model.fits_[1].mean_s > model.fits_[0].mean_s  # busy is slower
    proba = model.predict_proba(x_te)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    text = model.describe()
    assert "transition matrix" in text and "Otsu" in text


def test_queueing_baseline_validation():
    with pytest.raises(ConfigurationError):
        QueueingOverrunPredictor(deadline_s=0.0)
    with pytest.raises(ConfigurationError):
        QueueingOverrunPredictor(deadline_s=0.01, horizon=0)
    with pytest.raises(ConfigurationError):
        QueueingOverrunPredictor(deadline_s=0.01, n_prob_candidates=1)
    model = QueueingOverrunPredictor(deadline_s=0.01)
    with pytest.raises(RuntimeError):
        model.predict_proba(np.zeros((2, len(FEATURE_NAMES))))


def test_md1_pollaczek_khinchine_known_answer():
    # rho = 0.5, S = 0.01 s -> Wq = 0.5*0.01/(2*0.5) = 0.005 s.
    assert md1_mean_wait_s(0.5, 0.01) == pytest.approx(0.005, rel=1e-12)
    # rho = 0.9 -> Wq = 0.9*0.01/(2*0.1) = 0.045 s, nine times the rho=0.5 value.
    assert md1_mean_wait_s(0.9, 0.01) == pytest.approx(0.045, rel=1e-12)
    assert md1_mean_wait_s(0.0, 0.01) == 0.0
    with pytest.raises(ConfigurationError):
        md1_mean_wait_s(1.0, 0.01)
    with pytest.raises(ConfigurationError):
        md1_mean_wait_s(0.5, 0.0)


# -- learned model ----------------------------------------------------------


def test_learned_model_fits_and_produces_calibrated_probabilities():
    _deadline, x_tr, y_tr, x_te, y_te, _window, _ = _dataset()
    model = LearnedOverrunPredictor(max_iter=60, max_depth=3).fit(x_tr, y_tr)
    proba = model.predict_proba(x_te)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert 0.0 < model.cutoff_ < 1.0
    assert model.n_fit_ + model.n_calib_ == y_tr.size
    assert math.isfinite(model.train_time_s)
    assert "HistGradientBoosting" in model.describe()
    table = model.reliability_table(x_te, y_te, n_bins=5)
    assert table.shape == (5, 4)
    populated = table[np.isfinite(table[:, 2])]
    assert populated.size > 0


def test_learned_model_beats_a_constant_on_the_bursty_trace():
    """A Brier score better than the base-rate constant is the minimum bar."""
    _deadline, x_tr, y_tr, x_te, y_te, window, _ = _dataset()
    model = LearnedOverrunPredictor(max_iter=60, max_depth=3).fit(x_tr, y_tr)
    base_rate = float(y_tr.mean())
    constant_brier = float(np.mean((base_rate - y_te.astype(float)) ** 2))
    metrics = evaluate_predictor(model, x_te, y_te, window, horizon=HORIZON)
    assert metrics.brier < constant_brier
    assert metrics.roc_auc > 0.6


def test_learned_model_validation():
    with pytest.raises(ConfigurationError):
        LearnedOverrunPredictor(max_iter=0)
    with pytest.raises(ConfigurationError):
        LearnedOverrunPredictor(learning_rate=0.0)
    with pytest.raises(ConfigurationError):
        LearnedOverrunPredictor(l2_regularization=-1.0)
    with pytest.raises(ConfigurationError):
        LearnedOverrunPredictor(calibration_fraction=0.9)
    model = LearnedOverrunPredictor()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((2, len(FEATURE_NAMES))))
    with pytest.raises(ConfigurationError):
        model.fit(np.zeros((10, len(FEATURE_NAMES))), np.zeros(10, dtype=bool))
    with pytest.raises(ConfigurationError):
        model.fit(np.zeros((100, 3)), np.zeros(100, dtype=bool))


def test_learned_model_rejects_a_single_class_fit_slice():
    model = LearnedOverrunPredictor()
    x = np.zeros((200, len(FEATURE_NAMES)))
    y = np.zeros(200, dtype=bool)
    with pytest.raises(ConfigurationError, match="single class"):
        model.fit(x, y)


def test_permutation_importance_is_non_negative_for_the_strongest_feature():
    _deadline, x_tr, y_tr, x_te, y_te, _, _ = _dataset()
    model = LearnedOverrunPredictor(max_iter=40, max_depth=3).fit(x_tr, y_tr)
    imp = model.feature_importance_permutation(x_te[:2000], y_te[:2000], n_repeats=1)
    assert imp.shape == (len(FEATURE_NAMES),)
    assert float(imp.max()) > 0.0
    with pytest.raises(ConfigurationError):
        model.feature_importance_permutation(x_te[:10], y_te[:10], n_repeats=0)


# -- metrics ----------------------------------------------------------------


def test_wilson_interval_known_answer():
    # 50 of 100 at z = 1.959964. Hand evaluation of the Wilson formula:
    #   denom  = 1 + z^2/n          = 1 + 3.84145/100     = 1.0384145
    #   centre = (p + z^2/2n)/denom = (0.5 + 0.0192073)/1.0384145 = 0.5
    #   half   = (z/denom) sqrt(p(1-p)/n + z^2/4n^2)
    #          = 1.8874766 * sqrt(0.0025 + 9.603626e-05)  = 0.09616847
    #   -> [0.40383153, 0.59616847]
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.40383153, abs=1e-8)
    assert hi == pytest.approx(0.59616847, abs=1e-8)
    # 0 of 100: the interval stays inside [0, 1] where the normal one would not.
    lo0, hi0 = wilson_interval(0, 100)
    assert lo0 == 0.0
    assert hi0 == pytest.approx(0.03699350, abs=1e-8)
    assert all(math.isnan(v) for v in wilson_interval(0, 0))
    with pytest.raises(ConfigurationError):
        wilson_interval(5, 2)


def test_lead_time_known_answer():
    # Three rows, horizon 3. Row 0: overrun at offset 1 -> lead 1.
    # Row 1: overrun at offset 3 -> lead 3. Row 2: flagged but no label.
    y_true = np.array([True, True, False])
    y_pred = np.array([True, True, True])
    window = np.array(
        [[True, False, False], [False, False, True], [False, False, False]]
    )
    stats = lead_time_stats(y_true, y_pred, window, horizon=3)
    assert stats["n_true_positive"] == 2.0
    assert stats["mean_iters"] == pytest.approx(2.0)
    assert stats["min_iters"] == 1.0
    assert stats["max_iters"] == 3.0


def test_lead_time_with_no_true_positives_is_nan():
    stats = lead_time_stats(
        np.array([True]), np.array([False]), np.array([[True, False]]), horizon=2
    )
    assert math.isnan(stats["mean_iters"])
    assert stats["n_true_positive"] == 0.0


def test_lead_time_validation():
    with pytest.raises(ConfigurationError):
        lead_time_stats(np.array([True]), np.array([True]), np.zeros((1, 2)), horizon=0)
    with pytest.raises(ConfigurationError):
        lead_time_stats(np.array([True]), np.array([True]), np.zeros((1, 3)), horizon=2)
    with pytest.raises(ConfigurationError):
        lead_time_stats(
            np.array([True, True]), np.array([True]), np.zeros((2, 2)), horizon=2
        )


def test_evaluate_predictor_confusion_matrix_sums():
    deadline, x_tr, y_tr, x_te, y_te, window, _ = _dataset()
    model = FixedThresholdPredictor(deadline_s=deadline).fit(x_tr, y_tr)
    metrics = evaluate_predictor(model, x_te, y_te, window, horizon=HORIZON)
    assert metrics.tp + metrics.fp + metrics.tn + metrics.fn == metrics.n
    assert metrics.n_positive == metrics.tp + metrics.fn
    assert metrics.false_alarm_rate == pytest.approx(
        metrics.fp / (metrics.fp + metrics.tn)
    )
    assert metrics.missed_overrun_rate == pytest.approx(1.0 - metrics.recall)
    assert metrics.false_alarm_ci[0] <= metrics.false_alarm_rate <= metrics.false_alarm_ci[1]
    assert OverrunMetrics.header().startswith("predictor")
    assert model.name in metrics.as_row()


def test_cutoff_for_flag_rate_hits_a_continuous_target():
    scores = np.linspace(0.0, 1.0, 10001)
    cut = cutoff_for_flag_rate(scores, 0.20)
    assert float(np.mean(scores > cut)) == pytest.approx(0.20, abs=0.002)


def test_cutoff_for_flag_rate_is_honest_about_a_two_level_score():
    # 30 % of the rows are at the high level, so no cut-off gives 15 %, and the
    # nearest achievable rate is either 0 % or 30 %.
    scores = np.concatenate([np.zeros(700), np.ones(300)])
    cut = cutoff_for_flag_rate(scores, 0.15)
    achieved = float(np.mean(scores > cut))
    assert achieved in (0.0, 0.3)


def test_cutoff_for_flag_rate_validation():
    with pytest.raises(ConfigurationError):
        cutoff_for_flag_rate(np.array([]), 0.1)
    with pytest.raises(ValueError):
        cutoff_for_flag_rate(np.array([np.nan, 1.0]), 0.1)
    with pytest.raises(ConfigurationError):
        cutoff_for_flag_rate(np.array([0.0, 1.0]), 1.0)
