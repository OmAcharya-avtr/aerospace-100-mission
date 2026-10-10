"""Learned-detector tests: labelling, the batch/online identity, and the contract."""

from __future__ import annotations

import numpy as np
import pytest

from telemdrift.features import WINDOW
from telemdrift.learned import (
    TRAIN_CHANGES,
    LearnedDetector,
    build_training_set,
    score_stream,
    train_learned_detector,
)
from telemdrift.streams import ChangeSpec, change_stream, stationary


def test_training_set_shape_and_label_provenance(small_training_set):
    ts = small_training_set
    assert ts.features.shape[1] == 8
    assert ts.features.shape[0] == ts.labels.size
    assert ts.window == WINDOW
    assert ts.horizon == WINDOW
    assert ts.include_transients is False
    assert ts.change_specs == TRAIN_CHANGES
    assert set(np.unique(ts.labels)) == {0, 1}


def test_training_set_summary_states_the_positive_fraction(small_training_set):
    text = small_training_set.summary()
    assert "positive" in text
    assert "transients_in_training=False" in text


def test_positive_labels_lie_exactly_in_the_declared_horizon():
    """Label 1 iff the window's right edge is in [change_index, change_index+horizon).

    With pre_length 400 and window 50, row i covers samples [i, i+50), so its
    right edge is i + 49. The first positive row therefore has i + 49 = 400,
    i.e. i = 351, and the last has i + 49 = 449, i.e. i = 400: exactly 50 rows.
    """
    ts = build_training_set(seeds=[1], n_stationary=0, pre_length=400, post_length=400)
    # The first block of rows comes from the first change spec.
    first_block = ts.labels[:751]
    positives = np.flatnonzero(first_block == 1)
    assert positives.size == 50
    assert positives[0] == 351
    assert positives[-1] == 400


def test_training_set_with_transients_adds_only_negatives():
    without = build_training_set(seeds=[2], n_stationary=2, include_transients=False)
    with_tr = build_training_set(seeds=[2], n_stationary=2, include_transients=True)
    assert with_tr.features.shape[0] > without.features.shape[0]
    assert int(with_tr.labels.sum()) == int(without.labels.sum())
    assert with_tr.include_transients is True


def test_build_training_set_rejects_an_empty_seed_list():
    with pytest.raises(ValueError, match="at least one seed"):
        build_training_set(seeds=[])


def test_build_training_set_rejects_a_nonpositive_horizon():
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        build_training_set(seeds=[1], horizon=0)


def test_training_refuses_a_set_with_no_positive_windows(small_training_set):
    ts = small_training_set
    broken = type(ts)(
        features=ts.features,
        labels=np.zeros_like(ts.labels),
        seeds=ts.seeds,
        window=ts.window,
        horizon=ts.horizon,
        change_specs=ts.change_specs,
        include_transients=ts.include_transients,
    )
    with pytest.raises(ValueError, match="no positive windows"):
        train_learned_detector(broken)


def test_training_refuses_mismatched_feature_and_label_lengths(small_training_set):
    ts = small_training_set
    broken = type(ts)(
        features=ts.features[:-5],
        labels=ts.labels,
        seeds=ts.seeds,
        window=ts.window,
        horizon=ts.horizon,
        change_specs=ts.change_specs,
        include_transients=ts.include_transients,
    )
    with pytest.raises(ValueError, match="feature/label length mismatch"):
        train_learned_detector(broken)


def test_fitted_model_runs_single_threaded(small_model):
    """n_jobs > 1 is 5.7-8.9x slower at single-row inference on this container,
    and the per-sample online cost is a number this product publishes."""
    assert small_model.n_jobs == 1


def test_score_stream_is_aligned_and_zero_before_the_first_full_window(small_model):
    x = stationary(300, 71)
    s = score_stream(small_model, x, WINDOW)
    assert s.shape == x.shape
    assert np.all(s[: WINDOW - 1] == 0.0)
    assert np.all((s[WINDOW - 1 :] >= 0.0) & (s[WINDOW - 1 :] <= 1.0))


def test_score_stream_on_a_stream_shorter_than_the_window_is_all_zero(small_model):
    s = score_stream(small_model, stationary(10, 72), WINDOW)
    assert np.all(s == 0.0)


def test_online_update_and_batch_score_path_agree_exactly(small_model):
    """The identity the Monte Carlo rests on.

    LearnedDetector.update computes features from its own ring buffer one sample
    at a time; score_stream computes them for the whole stream at once. The
    features are causal functions of the trailing window, so the two must give
    the same score at every index. If they diverge, every learned ARL figure in
    this repository is wrong.
    """
    x = stationary(400, 73)
    det = LearnedDetector(small_model, 0.99, WINDOW)  # threshold high: no refractory
    det.reset()
    online = []
    for v in x:
        det.update(v)
        online.append(det.last_score)
    batch = score_stream(small_model, x, WINDOW)
    worst = float(np.max(np.abs(np.asarray(online)[WINDOW - 1 :] - batch[WINDOW - 1 :])))
    assert worst == 0.0, f"worst absolute difference {worst:.3e}"


def test_alarms_from_scores_matches_the_online_alarm_sequence(small_model):
    """Same identity, now including the refractory logic."""
    stream, _ = change_stream(300, 600, ChangeSpec("mean_step", 1.5), 74)
    det = LearnedDetector(small_model, 0.4, WINDOW)
    det.reset()
    online = [i for i, v in enumerate(stream) if det.update(v)]
    batch = LearnedDetector(small_model, 0.4, WINDOW).alarms_from_scores(
        score_stream(small_model, stream, WINDOW)
    )
    assert online == batch.tolist()


def test_refractory_period_suppresses_repeat_alarms(small_model):
    """A sustained 6-sigma step would otherwise alarm on every sample."""
    stream = np.concatenate([stationary(200, 75), stationary(600, 76) + 6.0])
    det = LearnedDetector(small_model, 0.3, WINDOW)
    alarms = det.alarms_from_scores(score_stream(small_model, stream, WINDOW))
    assert alarms.size > 0
    gaps = np.diff(alarms)
    # The refractory period is window - 1 samples, so consecutive alarms are at
    # least window apart. That equals a full reset: see LearnedDetector.update.
    assert gaps.size == 0 or gaps.min() >= WINDOW


def test_last_score_is_the_confidence_output_and_lies_in_the_unit_interval(small_model):
    det = LearnedDetector(small_model, 0.5, WINDOW)
    det.reset()
    for v in stationary(200, 77):
        det.update(v)
    assert 0.0 <= det.last_score <= 1.0
    assert det.alarm_ratio() == pytest.approx(det.last_score / 0.5)


def test_is_armed_is_false_during_warmup_and_during_the_refractory_period(small_model):
    """Warm-up is window - 1 samples, so the window-th sample is the first scorable
    one -- the same index at which score_stream places its first score."""
    det = LearnedDetector(small_model, 0.001, WINDOW)  # fires on almost anything
    det.reset()
    assert not det.is_armed()
    for i in range(WINDOW - 1):
        assert not det.update(0.0)
        if i < WINDOW - 2:
            assert not det.is_armed(), f"armed too early after {i + 1} samples"
    assert det.is_armed()
    assert det.update(20.0)
    assert not det.is_armed()


def test_reset_clears_the_refractory_state(small_model):
    det = LearnedDetector(small_model, 0.001, WINDOW)
    det.reset()
    for _ in range(WINDOW + 1):
        det.update(20.0)
    det.reset()
    assert det.last_score == 0.0
    assert not det.is_armed()
    assert det._refractory == WINDOW - 1


def test_default_threshold_is_the_argmax_rule():
    assert LearnedDetector.default_threshold() == 0.5


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_learned_detector_rejects_an_out_of_range_probability(small_model, bad):
    with pytest.raises(ValueError, match="0 < p\\* < 1"):
        LearnedDetector(small_model, bad)


def test_learned_detector_rejects_a_tiny_window(small_model):
    with pytest.raises(ValueError, match="window must be >= 4"):
        LearnedDetector(small_model, 0.5, window=2)


def test_threshold_setter_validates(small_model):
    det = LearnedDetector(small_model, 0.5)
    with pytest.raises(ValueError, match="0 < p\\* < 1"):
        det.threshold = 1.0
