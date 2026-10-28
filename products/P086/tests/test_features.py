"""Feature extraction: shape, bounds, determinism and the exclusions it declares."""

from __future__ import annotations

import numpy as np
import pytest

from telemgap import features
from telemgap.features import COUNTER_CAP, FEATURE_NAMES, INTERVAL_CAP, Z_CAP, build
from telemgap.record import TelemetryRecord


def test_feature_names_are_fifteen_and_unique() -> None:
    assert len(FEATURE_NAMES) == 15
    assert len(set(FEATURE_NAMES)) == 15


def test_no_channel_identity_feature() -> None:
    # Deliberate exclusion: the generator injects each misparse on a named
    # channel, so a channel-identity feature would let a model recover the
    # construction rather than the physics.
    for name in FEATURE_NAMES:
        assert "channel_index" not in name
        assert "channel_id" not in name
        assert name != "channel"


def test_build_shapes_and_order(generated_small) -> None:
    matrix, pairs = build(generated_small.record)
    n = len(generated_small.record)
    channels = generated_small.record.n_channels
    assert matrix.shape == (n * channels, len(FEATURE_NAMES))
    assert pairs.shape == (n * channels, 2)
    assert list(pairs[:channels, 1]) == list(range(channels))
    assert int(pairs[0, 0]) == 0
    assert int(pairs[-1, 0]) == n - 1


def test_build_is_finite_and_bounded(generated_small) -> None:
    matrix, _ = build(generated_small.record)
    assert np.all(np.isfinite(matrix))
    assert float(matrix.min()) >= -1.0
    assert float(matrix.max()) <= 1.0


def test_build_is_deterministic(generated_small) -> None:
    first, _ = build(generated_small.record)
    second, _ = build(generated_small.record)
    assert np.array_equal(first, second)


def test_build_requires_raw_words(generated_small) -> None:
    with pytest.raises(ValueError, match="raw_words"):
        build(generated_small.record.without_raw_words())


def test_validity_absent_gives_the_neutral_value(nominal_record: TelemetryRecord) -> None:
    matrix, _ = build(nominal_record.without_validity())
    column = FEATURE_NAMES.index("validity_ok")
    assert np.all(matrix[:, column] == 0.5)


def test_inside_range_feature_known_answer(small_schema, record_builder) -> None:
    # raw 40 -> 30.0 inside [12, 60]; raw 200 -> 110.0 outside.
    raws = [40] * 21
    raws[10] = 200
    record = record_builder(small_schema, raws)
    matrix, _ = build(record)
    inside = FEATURE_NAMES.index("inside_range")
    excess = FEATURE_NAMES.index("range_excess_norm")
    assert matrix[10, inside] == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert matrix[0, inside] == pytest.approx(1.0, rel=0.0, abs=0.0)
    # Excess is (110.0 - 60.0) / 48.0 = 1.0416..., clipped to 1.0.
    assert matrix[10, excess] == pytest.approx(1.0, rel=0.0, abs=0.0)
    # Inside the range the excess is negative and is the larger of the two
    # signed distances: max(12 - 30, 30 - 60) = -18, over the span of 48.
    assert matrix[0, excess] == pytest.approx(-18.0 / 48.0, rel=1e-12, abs=0.0)


def test_frame_out_of_range_fraction_known_answer(generated_small) -> None:
    matrix, pairs = build(generated_small.record)
    column = FEATURE_NAMES.index("frame_out_of_range_fraction")
    channels = generated_small.record.n_channels
    # The feature is a frame property, so every row of one frame carries it.
    for frame in (0, 5, 50):
        rows = matrix[frame * channels : (frame + 1) * channels, column]
        assert float(rows.std()) == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert float(matrix[:, column].max()) <= 1.0


def test_byteswap_closer_fires_on_a_byteswap(small_schema, record_builder) -> None:
    raws = [40] * 21
    raws[10] = 0x2800  # byteswaps to 40, the neighbours' value
    record = record_builder(small_schema, raws)
    matrix, _ = build(record)
    column = FEATURE_NAMES.index("byteswap_closer")
    assert matrix[10, column] == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert matrix[0, column] == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_any_rescue_fires_on_a_byteswap(small_schema, record_builder) -> None:
    raws = [40] * 21
    raws[10] = 0x2800
    record = record_builder(small_schema, raws)
    matrix, _ = build(record)
    column = FEATURE_NAMES.index("any_rescue")
    assert matrix[10, column] == pytest.approx(1.0, rel=0.0, abs=0.0)


def test_hamming_feature_known_answer(small_schema, record_builder) -> None:
    # Neighbours all read raw 40 = 0b101000. Flipping one bit to 41 gives a
    # Hamming distance of 1 over a 16-bit word: 1 / 16 = 0.0625.
    raws = [40] * 21
    raws[10] = 41
    record = record_builder(small_schema, raws)
    matrix, _ = build(record)
    column = FEATURE_NAMES.index("hamming_word_norm")
    assert matrix[10, column] == pytest.approx(1.0 / 16.0, rel=1e-12, abs=0.0)
    assert matrix[0, column] == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_caps_are_the_documented_values() -> None:
    assert Z_CAP == pytest.approx(50.0, rel=0.0, abs=0.0)
    assert INTERVAL_CAP == pytest.approx(8.0, rel=0.0, abs=0.0)
    assert COUNTER_CAP == pytest.approx(8.0, rel=0.0, abs=0.0)


def test_truth_labels_flattens_in_feature_order(generated_small) -> None:
    labels = features.truth_labels(generated_small.value_truth)
    assert labels.shape == (generated_small.value_truth.size,)
    assert labels[0] == generated_small.value_truth[0, 0]
    assert labels[1] == generated_small.value_truth[0, 1]


def test_counter_delta_feature_known_answer(small_schema, record_builder) -> None:
    # Counter deltas 1, 1, 4 -> normalised by the cap of 8: 0.125 and 0.5.
    record = record_builder(
        small_schema, [40] * 4, timestamps=[0.0, 1.0, 2.0, 6.0], counters=[0, 1, 2, 6]
    )
    matrix, _ = build(record)
    column = FEATURE_NAMES.index("counter_delta_norm")
    assert matrix[1, column] == pytest.approx(1.0 / 8.0, rel=1e-12, abs=0.0)
    assert matrix[3, column] == pytest.approx(4.0 / 8.0, rel=1e-12, abs=0.0)
