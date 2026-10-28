"""Data model: validation, known-answer decodes and the withholding views."""

from __future__ import annotations

import numpy as np
import pytest

from telemgap.record import SUPPORTED_WORD_BITS, ChannelSpec, RecordSchema, TelemetryRecord


def test_decode_known_answer(small_channel: ChannelSpec) -> None:
    # Hand-calculated from equation (1) with scale 0.5 and offset 10.0:
    # raw 0 -> 10.0, raw 1 -> 10.5, raw 100 -> 60.0, raw 65535 -> 32777.5.
    assert small_channel.decode(0) == pytest.approx(10.0, rel=0.0, abs=0.0)
    assert small_channel.decode(1) == pytest.approx(10.5, rel=0.0, abs=0.0)
    assert small_channel.decode(100) == pytest.approx(60.0, rel=0.0, abs=0.0)
    assert small_channel.decode(65535) == pytest.approx(32777.5, rel=1e-15, abs=0.0)


def test_span_and_mid_known_answer(small_channel: ChannelSpec) -> None:
    # low 12.0, high 60.0 -> span 48.0, mid 36.0.
    assert small_channel.span == pytest.approx(48.0, rel=0.0, abs=0.0)
    assert small_channel.mid == pytest.approx(36.0, rel=0.0, abs=0.0)


def test_in_range_is_inclusive(small_channel: ChannelSpec) -> None:
    assert bool(small_channel.in_range(12.0))
    assert bool(small_channel.in_range(60.0))
    assert not bool(small_channel.in_range(11.999))
    assert not bool(small_channel.in_range(60.001))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", ""),
        ("name", "   "),
        ("scale", 0.0),
        ("scale", float("nan")),
        ("offset", float("inf")),
        ("max_slew", -1.0),
        ("max_curvature", -1e-9),
        ("noise_sigma", -1.0),
        ("word_bits", 8),
        ("word_bits", 12),
    ],
)
def test_channel_spec_rejects_bad_field(field: str, value: object) -> None:
    kwargs = {
        "name": "probe",
        "unit": "K",
        "scale": 0.5,
        "offset": 10.0,
        "low": 12.0,
        "high": 60.0,
        "max_slew": 1.0,
        "max_curvature": 0.25,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        ChannelSpec(**kwargs)


def test_channel_spec_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="strictly below"):
        ChannelSpec(
            name="probe",
            unit="K",
            scale=1.0,
            offset=0.0,
            low=10.0,
            high=10.0,
            max_slew=1.0,
            max_curvature=1.0,
        )


def test_supported_word_bits_excludes_eight() -> None:
    # Byte-swap reinterpretation is undefined for a single byte, which is why
    # 8 is not supported.
    assert 8 not in SUPPORTED_WORD_BITS
    assert set(SUPPORTED_WORD_BITS) == {16, 32}


def test_schema_rejects_jitter_at_half_cadence(small_channel: ChannelSpec) -> None:
    with pytest.raises(ValueError, match="half the cadence"):
        RecordSchema(
            cadence_s=1.0, jitter_s=0.5, counter_modulus=256, channels=(small_channel,)
        )


def test_schema_rejects_duplicate_channel_names(small_channel: ChannelSpec) -> None:
    with pytest.raises(ValueError, match="unique"):
        RecordSchema(
            cadence_s=1.0,
            jitter_s=0.1,
            counter_modulus=256,
            channels=(small_channel, small_channel),
        )


@pytest.mark.parametrize("modulus", [-1, 0, 1, 1.5])
def test_schema_rejects_bad_modulus(small_channel: ChannelSpec, modulus: object) -> None:
    with pytest.raises(ValueError):
        RecordSchema(
            cadence_s=1.0, jitter_s=0.1, counter_modulus=modulus, channels=(small_channel,)
        )


def test_schema_rejects_empty_channels() -> None:
    with pytest.raises(ValueError, match="at least one"):
        RecordSchema(cadence_s=1.0, jitter_s=0.1, counter_modulus=256, channels=())


def test_effective_reorder_horizon_defaults_to_half(small_channel: ChannelSpec) -> None:
    schema = RecordSchema(
        cadence_s=1.0, jitter_s=0.1, counter_modulus=256, channels=(small_channel,)
    )
    assert schema.effective_reorder_horizon == 128
    explicit = RecordSchema(
        cadence_s=1.0,
        jitter_s=0.1,
        counter_modulus=256,
        channels=(small_channel,),
        reorder_horizon=16,
    )
    assert explicit.effective_reorder_horizon == 16


def test_reorder_horizon_out_of_range(small_channel: ChannelSpec) -> None:
    with pytest.raises(ValueError, match="reorder_horizon"):
        RecordSchema(
            cadence_s=1.0,
            jitter_s=0.1,
            counter_modulus=256,
            channels=(small_channel,),
            reorder_horizon=200,
        )


def test_channel_index_and_names(small_schema: RecordSchema) -> None:
    assert small_schema.channel_names == ("probe",)
    assert small_schema.channel_index("probe") == 0
    with pytest.raises(KeyError, match="no channel named"):
        small_schema.channel_index("absent")


def test_record_shape_validation(small_schema: RecordSchema) -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        TelemetryRecord(
            schema=small_schema,
            timestamps=np.empty(0),
            counters=np.empty(0, dtype=np.int64),
            values=np.empty((0, 1)),
            raw_words=np.empty((0, 1), dtype=np.uint64),
            validity=np.empty(0, dtype=bool),
        )
    with pytest.raises(ValueError, match="counters must have shape"):
        TelemetryRecord(
            schema=small_schema,
            timestamps=np.arange(3, dtype=float),
            counters=np.arange(2, dtype=np.int64),
            values=np.full((3, 1), 20.0),
            raw_words=np.zeros((3, 1), dtype=np.uint64),
            validity=np.ones(3, dtype=bool),
        )


def test_record_rejects_non_finite_timestamps(small_schema: RecordSchema) -> None:
    with pytest.raises(ValueError, match="finite"):
        TelemetryRecord(
            schema=small_schema,
            timestamps=np.array([0.0, float("nan"), 2.0]),
            counters=np.arange(3, dtype=np.int64),
            values=np.full((3, 1), 20.0),
            raw_words=np.zeros((3, 1), dtype=np.uint64),
            validity=np.ones(3, dtype=bool),
        )


def test_record_rejects_counter_out_of_modulus(small_schema: RecordSchema) -> None:
    with pytest.raises(ValueError, match="counters must lie"):
        TelemetryRecord(
            schema=small_schema,
            timestamps=np.arange(2, dtype=float),
            counters=np.array([0, 256], dtype=np.int64),
            values=np.full((2, 1), 20.0),
            raw_words=np.zeros((2, 1), dtype=np.uint64),
            validity=np.ones(2, dtype=bool),
        )


def test_record_rejects_oversized_raw_word(small_schema: RecordSchema) -> None:
    with pytest.raises(ValueError, match="16-bit range"):
        TelemetryRecord(
            schema=small_schema,
            timestamps=np.arange(2, dtype=float),
            counters=np.arange(2, dtype=np.int64),
            values=np.full((2, 1), 20.0),
            raw_words=np.array([[0], [1 << 16]], dtype=np.uint64),
            validity=np.ones(2, dtype=bool),
        )


def test_record_requires_declared_arrays(small_schema: RecordSchema) -> None:
    with pytest.raises(ValueError, match="has_raw_words"):
        TelemetryRecord(
            schema=small_schema,
            timestamps=np.arange(2, dtype=float),
            counters=np.arange(2, dtype=np.int64),
            values=np.full((2, 1), 20.0),
            raw_words=None,
            validity=np.ones(2, dtype=bool),
        )
    with pytest.raises(ValueError, match="has_validity_flag"):
        TelemetryRecord(
            schema=small_schema,
            timestamps=np.arange(2, dtype=float),
            counters=np.arange(2, dtype=np.int64),
            values=np.full((2, 1), 20.0),
            raw_words=np.zeros((2, 1), dtype=np.uint64),
            validity=None,
        )


def test_record_reshapes_one_dimensional_values(small_schema: RecordSchema) -> None:
    record = TelemetryRecord(
        schema=small_schema,
        timestamps=np.arange(3, dtype=float),
        counters=np.arange(3, dtype=np.int64),
        values=np.array([20.0, 21.0, 22.0]),
        raw_words=np.array([20, 22, 24], dtype=np.uint64),
        validity=np.ones(3, dtype=bool),
    )
    assert record.values.shape == (3, 1)
    assert record.raw_words.shape == (3, 1)
    assert len(record) == 3
    assert record.n_channels == 1


def test_channel_view(nominal_record: TelemetryRecord) -> None:
    view = nominal_record.channel("probe")
    assert view.shape == (40,)
    assert view[0] == pytest.approx(30.0, rel=0.0, abs=0.0)


def test_without_validity_and_raw_words(nominal_record: TelemetryRecord) -> None:
    stripped = nominal_record.without_validity()
    assert stripped.validity is None
    assert stripped.schema.has_validity_flag is False
    assert stripped.raw_words is not None
    bare = nominal_record.without_raw_words()
    assert bare.raw_words is None
    assert bare.schema.has_raw_words is False
    assert bare.validity is not None
    # The views are copies: mutating one must not touch the original.
    bare.values[0, 0] = -999.0
    assert nominal_record.values[0, 0] != -999.0


def test_build_record_helper_is_self_consistent(nominal_record: TelemetryRecord) -> None:
    channel = nominal_record.schema.channels[0]
    decoded = channel.decode(nominal_record.raw_words[:, 0].astype(np.int64))
    assert np.max(np.abs(decoded - nominal_record.values[:, 0])) == 0.0


def test_thirty_two_bit_channel_round_trip(record_builder) -> None:
    channel = ChannelSpec(
        name="wide",
        unit="count",
        scale=1.0,
        offset=0.0,
        low=0.0,
        high=4.0e9,
        max_slew=1.0,
        max_curvature=1.0,
        word_bits=32,
    )
    schema = RecordSchema(
        cadence_s=1.0,
        jitter_s=0.1,
        counter_modulus=256,
        channels=(channel,),
        has_validity_flag=True,
        has_raw_words=True,
    )
    record = record_builder(schema, [0, 1, 1 << 31, (1 << 32) - 1])
    assert record.values[3, 0] == pytest.approx(float((1 << 32) - 1), rel=1e-15, abs=0.0)
