"""The generator: determinism, self-consistency and the ground truth it carries."""

from __future__ import annotations

import numpy as np
import pytest

from telemgap import dataset
from telemgap.dataset import (
    CADENCE_S,
    COUNTER_MODULUS,
    FRAME_EVENTS,
    GAP_RUN_LENGTHS,
    JITTER_S,
    MODELS,
    VALUE_CLASSES,
    SignalModel,
    channel_specs,
    generate,
    long_outage_record,
    schema,
)


def test_signal_model_bounds_known_answer() -> None:
    # One term, A = 2, f = 1 / (2 pi): omega = 1, so the curvature bound is
    # A omega^2 = 2 and the slew bound is A omega = 2.
    model = SignalModel(
        name="unit", constant=0.0, ramp=0.0, terms=((2.0, 1.0 / (2.0 * np.pi), 0.0),),
        noise_sigma=0.0,
    )
    assert model.curvature_bound == pytest.approx(2.0, rel=1e-12, abs=0.0)
    assert model.slew_bound == pytest.approx(2.0, rel=1e-12, abs=0.0)
    assert model.amplitude_bound == pytest.approx(2.0, rel=0.0, abs=0.0)
    assert model.bandwidth_hz == pytest.approx(1.0 / (2.0 * np.pi), rel=0.0, abs=0.0)


def test_signal_model_ramp_enters_the_slew_bound() -> None:
    model = SignalModel(name="r", constant=1.0, ramp=-0.5, terms=(), noise_sigma=0.0)
    assert model.slew_bound == pytest.approx(0.5, rel=0.0, abs=0.0)
    assert model.curvature_bound == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert model.bandwidth_hz == pytest.approx(0.0, rel=0.0, abs=0.0)
    times = np.array([0.0, 2.0])
    assert np.allclose(model.evaluate(times), [1.0, 0.0], rtol=1e-12, atol=0.0)
    assert np.allclose(model.second_derivative(times), [0.0, 0.0], rtol=0.0, atol=0.0)


def test_second_derivative_matches_a_finite_difference() -> None:
    # Equation (12) differentiated term by term against a central difference of
    # the closed form. The step is chosen so that both error terms are below
    # the tolerance: the truncation error of a central second difference is
    # step^2 / 12 times the fourth derivative, about 3.2e-10 here, and the
    # cancellation error is of order 8 eps |v| / step^2, about 2.1e-11. The
    # absolute tolerance of 1e-9 covers both, and is needed because the
    # analytic second derivative passes through zero, where no relative
    # tolerance can hold.
    model = MODELS[0]
    times = np.linspace(10.0, 500.0, 37)
    step = 0.05
    numerical = (
        model.evaluate(times + step) - 2.0 * model.evaluate(times) + model.evaluate(times - step)
    ) / step**2
    analytic = model.second_derivative(times)
    assert np.allclose(analytic, numerical, rtol=1e-6, atol=1e-9)


def test_analytic_curvature_bound_dominates_the_realised_curvature() -> None:
    times = np.linspace(0.0, 4000.0, 20_001)
    for model in MODELS:
        realised = float(np.max(np.abs(model.second_derivative(times))))
        assert realised <= model.curvature_bound * (1.0 + 1e-12), model.name


def test_schema_constants() -> None:
    assert CADENCE_S == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert JITTER_S == pytest.approx(0.25, rel=0.0, abs=0.0)
    assert COUNTER_MODULUS == 16384
    assert 600 in GAP_RUN_LENGTHS
    assert VALUE_CLASSES == ("clean", "corrupt", "misparsed")
    assert FRAME_EVENTS == ("duplicate", "reorder", "clock_reset")


def test_channel_specs_declare_what_the_docstring_says() -> None:
    specs = channel_specs()
    declared = {spec.name for spec in specs if spec.max_curvature > 0.0}
    assert declared == {"battery_voltage", "tank_pressure"}
    assert all(spec.noise_sigma > 0.0 for spec in specs)
    assert [spec.name for spec in specs] == [model.name for model in MODELS]
    bare = channel_specs(declare_curvature=False)
    assert all(spec.max_curvature == 0.0 for spec in bare)


def test_schema_withholding_flags() -> None:
    assert schema().has_validity_flag
    assert schema().has_raw_words
    assert not schema(has_validity_flag=False).has_validity_flag
    assert not schema(has_raw_words=False).has_raw_words


def test_generation_is_deterministic_in_the_seed() -> None:
    first = generate(samples=400, seed=1234)
    second = generate(samples=400, seed=1234)
    assert np.array_equal(first.record.timestamps, second.record.timestamps)
    assert np.array_equal(first.record.raw_words, second.record.raw_words)
    assert np.array_equal(first.value_truth, second.value_truth)
    third = generate(samples=400, seed=1235)
    assert not np.array_equal(third.record.raw_words, first.record.raw_words)


def test_record_is_self_consistent(generated_small) -> None:
    # values == decode(raw_words) for every frame and channel. Without this a
    # classifier could separate the classes by re-decoding the words, which no
    # real archive permits, and every measured score would be meaningless.
    record = generated_small.record
    for column, channel in enumerate(record.schema.channels):
        raw = record.raw_words[:, column].astype(np.int64)
        if channel.signed:
            span = 1 << channel.word_bits
            raw = np.where(raw >= span // 2, raw - span, raw)
        decoded = raw.astype(np.float64) * channel.scale + channel.offset
        assert np.max(np.abs(decoded - record.values[:, column])) == 0.0, channel.name


def test_truth_counts_cover_every_class(generated_small) -> None:
    counts = generated_small.truth_counts()
    for name in VALUE_CLASSES:
        assert counts[name] > 0, name
    for event in FRAME_EVENTS:
        assert counts[event] > 0, event
    assert counts["missing_frames"] > 0
    assert counts["corrupt"] == counts["corrupt_in_range"] + counts["corrupt_out_of_range"]
    assert counts["frames"] == len(generated_small.record)


def test_most_corruptions_stay_inside_the_declared_range(generated_test) -> None:
    # The measured reason the specified range screen cannot find them. The
    # split is a consequence of the bit flips, not a generator parameter.
    counts = generated_test.truth_counts()
    assert counts["corrupt_in_range"] > counts["corrupt_out_of_range"]


def test_misparse_kinds_are_the_three_constructed(generated_test) -> None:
    kinds = {
        kind for kind in generated_test.misparse_kind.reshape(-1).tolist() if kind
    }
    assert kinds == {"signedness", "byteswap", "scale_decade_down"}


def test_misparse_is_confined_to_its_channel(generated_test) -> None:
    record = generated_test.record
    for name, expected in (
        ("valve_current", "signedness"),
        ("tank_pressure", "byteswap"),
        ("gyro_rate", "scale_decade_down"),
    ):
        column = record.schema.channel_index(name)
        kinds = {kind for kind in generated_test.misparse_kind[:, column].tolist() if kind}
        assert kinds == {expected}, name
    column = record.schema.channel_index("battery_voltage")
    assert not {kind for kind in generated_test.misparse_kind[:, column].tolist() if kind}


def test_truth_arrays_have_the_record_shape(generated_small) -> None:
    record = generated_small.record
    shape = (len(record), record.n_channels)
    assert generated_small.value_truth.shape == shape
    assert generated_small.misparse_kind.shape == shape
    assert generated_small.corruption_in_range.shape == shape
    assert generated_small.frame_truth.shape == (len(record),)


def test_deleted_frames_are_recorded(generated_small) -> None:
    assert generated_small.missing_slots.size > 0
    assert np.all(np.diff(generated_small.missing_slots) > 0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("samples", 16),
        ("jitter_fraction", 0.25),
        ("jitter_fraction", -0.1),
        ("flagged_probability", 1.5),
        ("flagged_probability", -0.1),
    ],
)
def test_generate_rejects_bad_arguments(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        generate(**{field: value})


def test_long_outage_record_has_one_long_gap() -> None:
    generated = long_outage_record(samples=200, outage_frames=1000)
    record = generated.record
    deltas = np.diff(np.sort(record.timestamps))
    # The frame-to-frame interval is one cadence longer than the unobserved
    # interval: 1000 missing frames sit between two frames 1001 s apart.
    assert float(deltas.max()) == pytest.approx(1001.0, rel=1e-9, abs=0.0)
    assert generated.parameters["outage_frames"] == pytest.approx(1000.0, rel=0.0, abs=0.0)
    with pytest.raises(ValueError, match="outage_frames"):
        long_outage_record(outage_frames=1)


def test_split_seeds_are_distinct() -> None:
    assert len({dataset.TRAINING_SEED, dataset.TEST_SEED, dataset.CALIBRATION_SEED}) == 3
    train = dataset.training_record(samples=300)
    test = dataset.test_record(samples=300)
    calibration = dataset.calibration_record(samples=300)
    assert not np.array_equal(train.record.raw_words, test.record.raw_words)
    assert not np.array_equal(train.record.raw_words, calibration.record.raw_words)


def test_withheld_fields_propagate_to_the_record() -> None:
    generated = generate(samples=300, seed=8, has_validity_flag=False)
    assert generated.record.validity is None
    generated = generate(samples=300, seed=8, has_raw_words=False)
    assert generated.record.raw_words is None


def test_signedness_misparse_matches_negative_truth() -> None:
    # The standing fault on valve_current is exactly the set of samples whose
    # noisy true value is negative, so the label is derivable rather than
    # asserted.
    generated = generate(samples=1200, seed=606, n_corruptions=0, n_byteswap=0, n_scale=0)
    column = generated.record.schema.channel_index("valve_current")
    flagged = generated.misparse_kind[:, column] == "signedness"
    assert flagged.sum() > 0
    assert np.all(generated.record.values[flagged, column] > 0.5)
