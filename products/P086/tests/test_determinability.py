"""The seven determinability limits, including the exact enumeration."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from telemgap import dataset
from telemgap.determinability import (
    EXACT_ENUMERATION_BITS,
    LIMIT_KEYS,
    Limit,
    assess,
    in_range_single_bit_flip_fraction,
    modal_interval,
)
from telemgap.record import ChannelSpec, RecordSchema, TelemetryRecord


def test_all_seven_limits_present(nominal_record: TelemetryRecord) -> None:
    verdict = assess(nominal_record)
    assert tuple(limit.key for limit in verdict.limits) == LIMIT_KEYS
    assert len(LIMIT_KEYS) == 7


def test_structural_limits_are_never_determinable(nominal_record: TelemetryRecord) -> None:
    verdict = assess(nominal_record)
    for key in ("counter_horizon", "in_range_corruption", "gap_interior"):
        assert verdict[key].determinable is False, key


def test_validity_and_raw_words_track_the_record(nominal_record: TelemetryRecord) -> None:
    verdict = assess(nominal_record)
    assert verdict["frame_validity"].determinable
    assert verdict["raw_words"].determinable
    assert verdict.can_separate_corrupt_from_misparsed
    stripped = assess(nominal_record.without_validity())
    assert stripped["frame_validity"].determinable is False
    assert stripped.can_separate_corrupt_from_misparsed is False
    bare = assess(nominal_record.without_raw_words())
    assert bare["raw_words"].determinable is False
    assert bare.can_separate_corrupt_from_misparsed is False


def test_unknown_limit_key_raises(nominal_record: TelemetryRecord) -> None:
    verdict = assess(nominal_record)
    with pytest.raises(KeyError, match="no determinability limit"):
        verdict["nonsense"]


def test_reasons_only_returns_undeterminable(nominal_record: TelemetryRecord) -> None:
    verdict = assess(nominal_record.without_validity())
    reasons = verdict.reasons(("frame_validity", "raw_words", "nonsense"))
    assert len(reasons) == 1
    assert "error-control" in reasons[0]


def test_undeterminable_keys_is_a_subset(nominal_record: TelemetryRecord) -> None:
    verdict = assess(nominal_record)
    assert set(verdict.undeterminable_keys) <= set(LIMIT_KEYS)
    assert "gap_interior" in verdict.undeterminable_keys


def test_counter_horizon_quantity_known_answer(nominal_record: TelemetryRecord) -> None:
    # modulus 256, default horizon 128, cadence 1 s -> the shortest gap the
    # counter domain cannot represent is 128 frames, that is 128 s.
    limit = assess(nominal_record)["counter_horizon"]
    assert limit.quantity == pytest.approx(128.0, rel=0.0, abs=0.0)


def test_in_range_fraction_known_answer_full_range() -> None:
    # A channel whose declared range covers every word code: every single-bit
    # flip of an in-range word is still in range, so the fraction is exactly 1
    # and the range screen can catch nothing.
    channel = ChannelSpec(
        name="wide",
        unit="count",
        scale=1.0,
        offset=0.0,
        low=0.0,
        high=65535.0,
        max_slew=1.0,
        max_curvature=1.0,
        word_bits=16,
    )
    fraction, method, codes = in_range_single_bit_flip_fraction(channel)
    assert method == "enumerated"
    assert fraction == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert codes == pytest.approx(1.0, rel=0.0, abs=0.0)


def test_in_range_fraction_known_answer_single_code() -> None:
    # A declared range admitting exactly one word code (raw 0 -> 0.0): every
    # single-bit flip leaves it, so the fraction is exactly 0 and the code
    # share is 1 / 65536.
    channel = ChannelSpec(
        name="narrow",
        unit="count",
        scale=1.0,
        offset=0.0,
        low=-0.5,
        high=0.5,
        max_slew=1.0,
        max_curvature=1.0,
        word_bits=16,
    )
    fraction, method, codes = in_range_single_bit_flip_fraction(channel)
    assert method == "enumerated"
    assert fraction == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert codes == pytest.approx(1.0 / 65536.0, rel=1e-15, abs=0.0)


def test_in_range_fraction_empty_range_returns_zero() -> None:
    # No word code decodes into the declared range at all.
    channel = ChannelSpec(
        name="impossible",
        unit="count",
        scale=1.0,
        offset=0.0,
        low=-10.0,
        high=-5.0,
        max_slew=1.0,
        max_curvature=1.0,
        word_bits=16,
    )
    fraction, _, codes = in_range_single_bit_flip_fraction(channel)
    assert fraction == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert codes == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_thirty_two_bit_channel_uses_monte_carlo() -> None:
    channel = ChannelSpec(
        name="wide32",
        unit="count",
        scale=1.0,
        offset=0.0,
        low=0.0,
        high=2.0e9,
        max_slew=1.0,
        max_curvature=1.0,
        word_bits=32,
    )
    assert channel.word_bits > EXACT_ENUMERATION_BITS
    fraction, method, codes = in_range_single_bit_flip_fraction(channel, seed=7)
    assert method == "monte_carlo"
    assert 0.0 <= fraction <= 1.0
    assert 0.0 < codes < 1.0
    # Deterministic in the seed.
    again, _, _ = in_range_single_bit_flip_fraction(channel, seed=7)
    assert again == pytest.approx(fraction, rel=0.0, abs=0.0)


def test_generated_channels_have_high_in_range_flip_fraction(generated_small) -> None:
    # The honest measurement behind the claim that a range screen cannot see
    # most bit errors on these channels.
    limit = assess(generated_small.record)["in_range_corruption"]
    assert limit.quantity > 0.5
    assert limit.method == "enumerated"


def test_modal_interval_known_answer() -> None:
    # Eleven frames at exactly 2 s spacing, one 10 s gap: the mode is 2 s.
    times = np.array([0, 2, 4, 6, 8, 18, 20, 22, 24, 26, 28], dtype=float)
    assert modal_interval(times, 2.0) == pytest.approx(2.0, rel=1e-12, abs=1e-15)


def test_modal_interval_of_single_frame_is_nan() -> None:
    assert np.isnan(modal_interval(np.array([3.0]), 1.0))


def test_cadence_declaration_fires_when_wrong(small_channel: ChannelSpec) -> None:
    # Declared cadence 1 s, actual spacing 5 s: the limit must say so.
    schema = RecordSchema(
        cadence_s=1.0,
        jitter_s=0.1,
        counter_modulus=256,
        channels=(small_channel,),
        has_validity_flag=True,
        has_raw_words=True,
    )
    record = TelemetryRecord(
        schema=schema,
        timestamps=5.0 * np.arange(20, dtype=float),
        counters=np.arange(20, dtype=np.int64),
        values=np.full((20, 1), 30.0),
        raw_words=np.full((20, 1), 40, dtype=np.uint64),
        validity=np.ones(20, dtype=bool),
    )
    limit = assess(record)["cadence_declaration"]
    assert limit.determinable is False
    assert limit.quantity == pytest.approx(4.0, rel=1e-12, abs=1e-15)


def test_time_source_tracks_provenance(nominal_record: TelemetryRecord) -> None:
    assert assess(nominal_record)["time_source"].determinable
    stripped = dataclasses.replace(nominal_record, provenance={})
    assert assess(stripped)["time_source"].determinable is False


def test_generator_declares_a_time_source(generated_small) -> None:
    assert assess(generated_small.record)["time_source"].determinable


def test_gap_interior_counts_declared_curvature() -> None:
    generated = dataset.generate(samples=200, seed=11)
    limit = assess(generated.record)["gap_interior"]
    # telemgap.dataset declares a curvature bound on two of its four channels.
    assert limit.quantity == pytest.approx(2.0, rel=0.0, abs=0.0)
    undeclared = dataset.generate(samples=200, seed=11, declare_curvature=False)
    assert assess(undeclared.record)["gap_interior"].quantity == pytest.approx(
        0.0, rel=0.0, abs=0.0
    )


def test_assess_rejects_non_record() -> None:
    with pytest.raises(TypeError, match="TelemetryRecord"):
        assess({"not": "a record"})


def test_limit_defaults() -> None:
    limit = Limit(key="frame_validity", determinable=True, statement="s", consequence="c")
    assert np.isnan(limit.quantity)
    assert limit.unit == ""
    assert limit.method == "declared"
