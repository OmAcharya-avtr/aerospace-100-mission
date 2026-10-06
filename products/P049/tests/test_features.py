"""Feature extraction, labelling and split tests, including leakage checks."""

from __future__ import annotations

import numpy as np
import pytest

from linkoutage.features import (
    FEATURE_NAMES,
    build_outage_dataset,
    decision_indices,
    extract_features,
    onset_labels,
    temporal_split,
)


def test_feature_names_are_unique_and_fourteen():
    assert len(FEATURE_NAMES) == 14
    assert len(set(FEATURE_NAMES)) == 14


def test_decision_indices_hand_calculation():
    # N = 20, W = 5, H = 3, stride = 4: first t = 4, last t = 20 - 1 - 3 = 16
    assert decision_indices(
        20, window_samples=5, horizon_samples=3, stride_samples=4
    ).tolist() == [4, 8, 12, 16]


def test_decision_indices_stride_one():
    idx = decision_indices(10, window_samples=3, horizon_samples=2, stride_samples=1)
    assert idx[0] == 2 and idx[-1] == 7


def test_decision_indices_too_short_raises():
    with pytest.raises(ValueError, match="too short"):
        decision_indices(5, window_samples=4, horizon_samples=4, stride_samples=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_samples": 1, "horizon_samples": 2, "stride_samples": 1},
        {"window_samples": 3, "horizon_samples": 0, "stride_samples": 1},
        {"window_samples": 3, "horizon_samples": 2, "stride_samples": 0},
    ],
)
def test_decision_indices_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        decision_indices(100, **kwargs)


def test_extract_features_shape(short_channel):
    idx = decision_indices(
        short_channel.n_samples,
        window_samples=100,
        horizon_samples=50,
        stride_samples=500,
    )
    x = extract_features(short_channel.amplitude, index=idx, window_samples=100, threshold=0.5)
    assert x.shape == (idx.size, 14)
    assert np.all(np.isfinite(x))


def test_extract_features_hand_calculation():
    # Window of 4 samples ending at t = 4, amplitude [_, 1.0, 0.4, 1.0, 1.0]
    # window = a[1..4] = [1.0, 0.4, 1.0, 1.0], threshold 0.6
    a = np.array([1.0, 1.0, 0.4, 1.0, 1.0])
    x = extract_features(a, index=[4], window_samples=4, threshold=0.6)[0]
    names = list(FEATURE_NAMES)
    assert x[names.index("log_amp_last")] == pytest.approx(0.0)
    assert x[names.index("log_amp_min")] == pytest.approx(np.log(0.4))
    assert x[names.index("log_amp_max")] == pytest.approx(0.0)
    assert x[names.index("frac_below_threshold")] == pytest.approx(0.25)
    assert x[names.index("n_down_crossings_in_window")] == pytest.approx(1.0)
    assert x[names.index("in_fade_now")] == pytest.approx(0.0)
    # the only in-window down-crossing is at window position 1, current is 3
    assert x[names.index("samples_since_down_crossing")] == pytest.approx(2.0)


def test_samples_since_down_crossing_is_capped_when_none():
    a = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    x = extract_features(a, index=[4], window_samples=4, threshold=0.6)[0]
    assert x[list(FEATURE_NAMES).index("samples_since_down_crossing")] == pytest.approx(4.0)


def test_in_fade_now_flag():
    a = np.array([1.0, 1.0, 1.0, 0.4])
    x = extract_features(a, index=[3], window_samples=4, threshold=0.6)[0]
    assert x[list(FEATURE_NAMES).index("in_fade_now")] == pytest.approx(1.0)


def test_slope_of_a_geometric_ramp_is_constant():
    # log amplitude rising by 0.1 per sample -> OLS slope exactly 0.1
    a = np.exp(0.1 * np.arange(20, dtype=float))
    x = extract_features(a, index=[19], window_samples=16, threshold=1e-9)[0]
    assert x[list(FEATURE_NAMES).index("log_amp_slope_full")] == pytest.approx(0.1)


def test_diff_std_is_zero_on_a_ramp():
    a = np.exp(0.1 * np.arange(20, dtype=float))
    x = extract_features(a, index=[19], window_samples=16, threshold=1e-9)[0]
    assert x[list(FEATURE_NAMES).index("log_amp_diff_std")] == pytest.approx(0.0, abs=1e-12)


def test_chunking_does_not_change_the_answer(short_channel):
    idx = decision_indices(
        20_000, window_samples=64, horizon_samples=32, stride_samples=97
    )
    a = short_channel.amplitude[:20_000]
    one = extract_features(a, index=idx, window_samples=64, threshold=0.6, chunk=10_000)
    many = extract_features(a, index=idx, window_samples=64, threshold=0.6, chunk=7)
    # Not bit-identical: the slope columns are matrix-vector products, and BLAS
    # chooses a different blocking for a 206-row block than for a 7-row one.
    # The disagreement is at the last bit of a float64, not at the algorithm.
    assert np.allclose(one, many, rtol=0.0, atol=1e-15)


def test_extract_features_rejects_non_positive_amplitude():
    with pytest.raises(ValueError, match="strictly positive"):
        extract_features(np.array([1.0, 0.0, 1.0]), index=[2], window_samples=2, threshold=0.5)


def test_extract_features_rejects_insufficient_history():
    with pytest.raises(ValueError, match="history"):
        extract_features(np.ones(10), index=[1], window_samples=5, threshold=0.5)


def test_extract_features_rejects_index_past_the_end():
    with pytest.raises(ValueError, match="n_samples"):
        extract_features(np.ones(10), index=[10], window_samples=5, threshold=0.5)


def test_extract_features_empty_index():
    x = extract_features(np.ones(10), index=[], window_samples=5, threshold=0.5)
    assert x.shape == (0, 14)


def test_onset_labels_hand_calculation():
    # a = [1, 1, 0.4, 1, 1, 0.4, 1]; down-crossings at 2 and 5.
    # t = 1, H = 2 -> window (1, 3] contains index 2 -> label 1
    # t = 3, H = 1 -> window (3, 4] contains nothing -> label 0
    # t = 3, H = 2 -> window (3, 5] contains index 5 -> label 1
    a = np.array([1.0, 1.0, 0.4, 1.0, 1.0, 0.4, 1.0])
    assert onset_labels(a, index=[1], threshold=0.6, horizon_samples=2).tolist() == [1]
    assert onset_labels(a, index=[3], threshold=0.6, horizon_samples=1).tolist() == [0]
    assert onset_labels(a, index=[3], threshold=0.6, horizon_samples=2).tolist() == [1]


def test_onset_label_excludes_the_present_sample():
    # A down-crossing exactly at t is in the past, not the horizon.
    a = np.array([1.0, 0.4, 1.0, 1.0, 1.0])
    assert onset_labels(a, index=[1], threshold=0.6, horizon_samples=3).tolist() == [0]


def test_onset_label_is_zero_when_no_crossing_exists():
    a = np.ones(50)
    assert int(onset_labels(a, index=[10, 20], threshold=0.6, horizon_samples=5).sum()) == 0


def test_onset_labels_reject_overrunning_horizon():
    with pytest.raises(ValueError, match="runs past"):
        onset_labels(np.ones(10), index=[9], threshold=0.6, horizon_samples=5)


def test_labels_do_not_depend_on_samples_beyond_the_horizon(short_channel):
    a = short_channel.amplitude[:50_000].copy()
    idx = decision_indices(
        40_000, window_samples=200, horizon_samples=100, stride_samples=50
    )
    base = onset_labels(a, index=idx, threshold=0.4, horizon_samples=100)
    tampered = a.copy()
    tampered[40_100:] = 0.01  # far beyond the last horizon
    after = onset_labels(tampered, index=idx, threshold=0.4, horizon_samples=100)
    assert np.array_equal(base, after)


def test_features_do_not_depend_on_the_future(short_channel):
    a = short_channel.amplitude[:20_000].copy()
    idx = decision_indices(
        10_000, window_samples=200, horizon_samples=100, stride_samples=50
    )
    base = extract_features(a, index=idx, window_samples=200, threshold=0.4)
    tampered = a.copy()
    tampered[int(idx[-1]) + 1 :] = 0.01
    after = extract_features(tampered, index=idx, window_samples=200, threshold=0.4)
    assert np.array_equal(base, after)


def test_temporal_split_is_ordered_and_disjoint():
    y = np.zeros(1000, dtype=np.int64)
    y[::37] = 1
    split = temporal_split(y, gap_rows=10)
    assert split.train[-1] < split.calibration[0]
    assert split.calibration[-1] < split.test[0]
    assert split.calibration[0] - split.train[-1] == 11
    assert split.test[0] - split.calibration[-1] == 11
    assert set(split.train.tolist()).isdisjoint(split.calibration.tolist())


def test_temporal_split_fraction_sizes():
    y = np.zeros(1000, dtype=np.int64)
    split = temporal_split(y, fractions=(0.5, 0.2, 0.3), gap_rows=0)
    assert split.train.size == 500
    assert split.calibration.size == 200
    assert split.test.size == 300


def test_temporal_split_positive_counts():
    y = np.zeros(1000, dtype=np.int64)
    y[:600] = 1
    split = temporal_split(y, gap_rows=0)
    assert split.n_positive[0] == 600
    assert split.n_positive[1] == 0


def test_temporal_split_report():
    y = np.zeros(1000, dtype=np.int64)
    assert "gap" in temporal_split(y, gap_rows=5).report()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fractions": (0.5, 0.5, 0.5)},
        {"fractions": (0.5, 0.0, 0.5)},
        {"gap_rows": -1},
    ],
)
def test_temporal_split_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        temporal_split(np.zeros(1000, dtype=np.int64), **kwargs)


def test_temporal_split_too_few_rows():
    with pytest.raises(ValueError, match="at least 10"):
        temporal_split(np.zeros(5, dtype=np.int64))


def test_build_dataset_gap_covers_window_plus_horizon(short_channel):
    data = build_outage_dataset(
        short_channel.amplitude,
        threshold=0.4,
        window_samples=400,
        horizon_samples=200,
        stride_samples=50,
        gaussian=short_channel.gaussian,
    )
    assert data.configuration["gap_rows"] == 12  # ceil((400 + 200) / 50)
    assert data.split.gap_rows == 12
    # The gap in samples must cover window + horizon.
    assert data.split.gap_rows * 50 >= 400 + 200


def test_build_dataset_consistency(short_channel):
    data = build_outage_dataset(
        short_channel.amplitude,
        threshold=0.4,
        window_samples=200,
        horizon_samples=100,
        stride_samples=100,
        gaussian=short_channel.gaussian,
    )
    assert data.x.shape == (data.n_rows, 14)
    assert data.y.size == data.n_rows
    assert data.index.size == data.n_rows
    assert data.gaussian is not None and data.gaussian.size == data.n_rows
    assert data.base_rate == pytest.approx(float(data.y.mean()))
    assert "base rate" in data.report()


def test_build_dataset_without_gaussian(short_channel):
    data = build_outage_dataset(
        short_channel.amplitude,
        threshold=0.4,
        window_samples=200,
        horizon_samples=100,
        stride_samples=100,
    )
    assert data.gaussian is None


def test_build_dataset_rejects_mismatched_gaussian(short_channel):
    with pytest.raises(ValueError, match="same record"):
        build_outage_dataset(
            short_channel.amplitude,
            threshold=0.4,
            window_samples=200,
            horizon_samples=100,
            stride_samples=100,
            gaussian=short_channel.gaussian[:100],
        )


def test_build_dataset_gaussian_is_aligned(short_channel):
    data = build_outage_dataset(
        short_channel.amplitude,
        threshold=0.4,
        window_samples=200,
        horizon_samples=100,
        stride_samples=100,
        gaussian=short_channel.gaussian,
    )
    assert np.allclose(data.gaussian, short_channel.gaussian[data.index])
