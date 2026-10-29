"""The record: construction, validation and the residual of equation (7)."""

from __future__ import annotations

import numpy as np
import pytest

from aligncal.geometry import rotation_from_vector
from aligncal.record import AlignmentRecord, SensorSpec, residuals


def _minimal(samples: int = 20, sigma: float = 1.0e-4) -> AlignmentRecord:
    times = np.arange(samples, dtype=float) / 10.0
    rates = np.column_stack(
        [np.sin(times), np.cos(times), np.sin(2.0 * times)]
    ) * 0.01
    return AlignmentRecord(
        times=times,
        reference=SensorSpec("reference", sigma),
        reference_rates=rates,
        secondaries={"s": SensorSpec("s", sigma)},
        secondary_rates={"s": rates},
        provenance="tests/test_record.py _minimal",
    )


def test_sensor_spec_defaults_to_identity_mounting():
    spec = SensorSpec("a", 1.0e-4)
    assert np.array_equal(spec.mounting, np.eye(3))


def test_sensor_spec_rejects_empty_name():
    with pytest.raises(ValueError, match="non-empty string"):
        SensorSpec("", 1.0e-4)


def test_sensor_spec_rejects_non_string_name():
    with pytest.raises(ValueError, match="non-empty string"):
        SensorSpec(3, 1.0e-4)  # type: ignore[arg-type]


@pytest.mark.parametrize("sigma", [0.0, -1.0e-4, np.nan, np.inf])
def test_sensor_spec_rejects_unusable_sigma(sigma):
    with pytest.raises(ValueError, match="noise_sigma"):
        SensorSpec("a", sigma)


def test_sensor_spec_rejects_non_rotation_mounting():
    with pytest.raises(ValueError, match="not a rotation matrix"):
        SensorSpec("a", 1.0e-4, 2.0 * np.eye(3))


def test_sensor_spec_rejects_reflection_mounting():
    with pytest.raises(ValueError, match="not a rotation matrix"):
        SensorSpec("a", 1.0e-4, np.diag([1.0, 1.0, -1.0]))


def test_sensor_spec_rejects_wrong_shaped_mounting():
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        SensorSpec("a", 1.0e-4, np.eye(4))


def test_record_basics():
    record = _minimal(30)
    assert len(record) == 30
    assert record.names == ("s",)
    assert record.duration_s == pytest.approx(2.9, rel=1.0e-12, abs=0.0)
    assert record.mean_interval_s == pytest.approx(0.1, rel=1.0e-12, abs=0.0)


def test_pair_sigma_is_the_quadrature_sum():
    times = np.arange(10, dtype=float)
    rates = np.ones((10, 3)) * 0.01
    record = AlignmentRecord(
        times=times,
        reference=SensorSpec("reference", 3.0e-4),
        reference_rates=rates,
        secondaries={"s": SensorSpec("s", 4.0e-4)},
        secondary_rates={"s": rates},
    )
    # 3-4-5 triangle, by hand.
    assert record.pair_sigma("s") == pytest.approx(5.0e-4, rel=1.0e-14, abs=0.0)


def test_pair_sigma_rejects_unknown_name():
    with pytest.raises(KeyError, match="unknown secondary"):
        _minimal().pair_sigma("nope")


def test_derotated_rejects_unknown_name():
    with pytest.raises(KeyError, match="unknown secondary"):
        _minimal().derotated("nope")


def test_record_rejects_short_series():
    with pytest.raises(ValueError, match="at least 2 samples"):
        AlignmentRecord(
            times=np.array([0.0]),
            reference=SensorSpec("reference", 1.0e-4),
            reference_rates=np.zeros((1, 3)),
            secondaries={"s": SensorSpec("s", 1.0e-4)},
            secondary_rates={"s": np.zeros((1, 3))},
        )


def test_record_rejects_non_increasing_times():
    times = np.array([0.0, 0.1, 0.1, 0.2])
    rates = np.zeros((4, 3))
    with pytest.raises(ValueError, match="strictly increasing"):
        AlignmentRecord(
            times=times,
            reference=SensorSpec("reference", 1.0e-4),
            reference_rates=rates,
            secondaries={"s": SensorSpec("s", 1.0e-4)},
            secondary_rates={"s": rates},
        )


def test_record_rejects_non_finite_times():
    times = np.array([0.0, 0.1, np.nan, 0.3])
    rates = np.zeros((4, 3))
    with pytest.raises(ValueError, match="must be finite"):
        AlignmentRecord(
            times=times,
            reference=SensorSpec("reference", 1.0e-4),
            reference_rates=rates,
            secondaries={"s": SensorSpec("s", 1.0e-4)},
            secondary_rates={"s": rates},
        )


def test_record_rejects_non_identity_reference_mounting():
    times = np.arange(5, dtype=float)
    rates = np.zeros((5, 3))
    rotated = SensorSpec("reference", 1.0e-4, rotation_from_vector(np.array([0.0, 0.0, 0.1])))
    with pytest.raises(ValueError, match="must be the identity"):
        AlignmentRecord(
            times=times,
            reference=rotated,
            reference_rates=rates,
            secondaries={"s": SensorSpec("s", 1.0e-4)},
            secondary_rates={"s": rates},
        )


def test_record_rejects_no_secondaries():
    times = np.arange(5, dtype=float)
    with pytest.raises(ValueError, match="at least one secondary"):
        AlignmentRecord(
            times=times,
            reference=SensorSpec("reference", 1.0e-4),
            reference_rates=np.zeros((5, 3)),
            secondaries={},
            secondary_rates={},
        )


def test_record_rejects_mismatched_names():
    times = np.arange(5, dtype=float)
    rates = np.zeros((5, 3))
    with pytest.raises(ValueError, match="same names"):
        AlignmentRecord(
            times=times,
            reference=SensorSpec("reference", 1.0e-4),
            reference_rates=rates,
            secondaries={"a": SensorSpec("a", 1.0e-4)},
            secondary_rates={"b": rates},
        )


def test_record_rejects_duplicate_reference_name():
    times = np.arange(5, dtype=float)
    rates = np.zeros((5, 3))
    with pytest.raises(ValueError, match="every name must be unique"):
        AlignmentRecord(
            times=times,
            reference=SensorSpec("reference", 1.0e-4),
            reference_rates=rates,
            secondaries={"reference": SensorSpec("reference", 1.0e-4)},
            secondary_rates={"reference": rates},
        )


def test_record_rejects_wrong_shaped_rates():
    times = np.arange(5, dtype=float)
    with pytest.raises(ValueError, match=r"shape \(5, 3\)"):
        AlignmentRecord(
            times=times,
            reference=SensorSpec("reference", 1.0e-4),
            reference_rates=np.zeros((5, 2)),
            secondaries={"s": SensorSpec("s", 1.0e-4)},
            secondary_rates={"s": np.zeros((5, 3))},
        )


def test_record_rejects_nan_rates_with_an_explanation():
    times = np.arange(5, dtype=float)
    rates = np.zeros((5, 3))
    rates[2, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite values"):
        AlignmentRecord(
            times=times,
            reference=SensorSpec("reference", 1.0e-4),
            reference_rates=rates,
            secondaries={"s": SensorSpec("s", 1.0e-4)},
            secondary_rates={"s": np.zeros((5, 3))},
        )


def test_derotation_undoes_a_declared_mounting_exactly():
    """With no misalignment, the de-rotated secondary equals the reference."""
    times = np.arange(50, dtype=float) / 10.0
    rates = np.column_stack([np.sin(times), np.cos(times), times * 0.01]) * 0.02
    mounting = rotation_from_vector(np.array([0.3, -0.7, 1.1]))
    record = AlignmentRecord(
        times=times,
        reference=SensorSpec("reference", 1.0e-4),
        reference_rates=rates,
        secondaries={"s": SensorSpec("s", 1.0e-4, mounting)},
        secondary_rates={"s": rates @ mounting.T},
    )
    assert np.max(np.abs(record.derotated("s") - rates)) < 1.0e-16


def test_residual_matches_equation_seven_to_second_order():
    """r = [w]x theta with an absolute error O(|theta|^2 |w|).

    The residual is itself O(|theta| |w|), so the RELATIVE error of the
    linearisation is O(|theta|), not O(|theta|^2). At the |theta| = 1.21 mrad
    used here the predicted relative bound is |theta| / 2 = 6.0e-4 and the
    measured value is 5.4e-4, so the tolerance below is 1e-3. Writing 1e-5 here
    -- the relative truncation of the rotation MATRIX at this angle rather than
    of the residual -- was the first draft of this test and it failed; the
    distinction is pinned in tests/test_regressions.py.
    """
    times = np.arange(200, dtype=float) / 20.0
    rates = np.column_stack(
        [0.05 * np.sin(times), 0.04 * np.cos(0.7 * times), 0.03 * np.sin(1.3 * times)]
    )
    theta = np.array([1.0, -0.6, 0.3]) * 1.0e-3
    mounting = rotation_from_vector(np.array([0.2, 0.9, -0.4]))
    true_mounting = mounting @ rotation_from_vector(-theta)
    record = AlignmentRecord(
        times=times,
        reference=SensorSpec("reference", 1.0e-6),
        reference_rates=rates,
        secondaries={"s": SensorSpec("s", 1.0e-6, mounting)},
        secondary_rates={"s": rates @ true_mounting.T},
    )
    measured = residuals(record, "s")
    from aligncal.geometry import skew

    predicted = skew(rates) @ theta
    scale = float(np.max(np.abs(predicted)))
    assert np.max(np.abs(measured - predicted)) < 1.0e-3 * scale


def test_subrecord_slices_every_block():
    record = _minimal(40)
    window = record.subrecord(30, 10)
    assert len(window) == 20
    assert np.array_equal(window.times, record.times[10:30])
    assert np.array_equal(window.reference_rates, record.reference_rates[10:30])
    assert np.array_equal(window.secondary_rates["s"], record.secondary_rates["s"][10:30])
    assert "samples 10:30" in window.provenance


def test_subrecord_rejects_empty_window():
    with pytest.raises(ValueError, match="stop > start"):
        _minimal().subrecord(5, 5)
