"""Rate-distortion sweep tests. Matched distortion, real bytes, honest monotonicity."""

from __future__ import annotations

import numpy as np
import pytest

from boundcompress import baselines, ratedistortion
from boundcompress.predictors import CLASSICAL_NAMES


def test_geometric_bounds_span_the_requested_fractions() -> None:
    values = np.linspace(0.0, 10.0, 100)
    bounds = ratedistortion.geometric_bounds(values, count=5)
    assert len(bounds) == 5
    assert bounds[0] == pytest.approx(1e-2 * 10.0, rel=1e-12)
    assert bounds[-1] == pytest.approx(1e-6 * 10.0, rel=1e-12)
    assert all(bounds[index] > bounds[index + 1] for index in range(4))


def test_geometric_bounds_refuses_a_constant_array() -> None:
    with pytest.raises(ValueError, match="constant array"):
        ratedistortion.geometric_bounds(np.full(10, 2.0))


def test_geometric_bounds_refuses_a_single_point() -> None:
    with pytest.raises(ValueError, match="at least two bounds"):
        ratedistortion.geometric_bounds(np.linspace(0.0, 1.0, 10), count=1)


def test_distortion_is_identical_across_predictors(signals: dict[str, np.ndarray]) -> None:
    """The point of the architecture, asserted exactly rather than to a tolerance.

    Every predictor shares the quantiser, so the realised maximum error is the
    same number -- not a close number -- for every coder at a given bound.
    """
    values = signals["battery_voltage"]
    bounds = ratedistortion.geometric_bounds(values, count=3)
    rows = ratedistortion.sweep_signal("battery_voltage", values, bounds)
    for bound in bounds:
        errors = {row.max_error for row in rows if row.bound == bound
                  and row.coder not in baselines.LOSSLESS_METHODS}
        assert len(errors) == 1, f"distortion is not matched at bound {bound}"


def test_every_row_round_trips_and_holds_the_bound(signals: dict[str, np.ndarray]) -> None:
    values = signals["panel_temperature"]
    bounds = ratedistortion.geometric_bounds(values, count=3)
    rows = ratedistortion.sweep_signal("panel_temperature", values, bounds)
    assert ratedistortion.verify_roundtrip(rows) == []
    assert all(row.bound_holds for row in rows)
    assert all(row.encoded_bytes > 0 for row in rows)


def test_sweep_covers_every_predictor_and_baseline(signals: dict[str, np.ndarray]) -> None:
    values = signals["magnetometer"]
    rows = ratedistortion.sweep_signal("magnetometer", values, [0.1])
    coders = {row.coder for row in rows}
    assert set(CLASSICAL_NAMES) <= coders
    assert set(baselines.LOSSY_METHODS) <= coders
    assert set(baselines.LOSSLESS_METHODS) <= coders


def test_rate_grows_as_the_bound_tightens(signals: dict[str, np.ndarray]) -> None:
    """Monotone in the bound for a fixed predictor. A measured property, not a theorem.

    The relation is monotone for every signal and predictor measured in
    validation/validate_rate_distortion.py. It is not a theorem -- the
    frequency table and the escape stream both change discontinuously with the
    bound -- so the assertion is restricted to the smooth signal where the
    margin is large.
    """
    values = signals["battery_voltage"]
    bounds = ratedistortion.geometric_bounds(values, count=4)
    rows = ratedistortion.sweep_signal(
        "battery_voltage", values, bounds, predictors=("linear",), include_baselines=False
    )
    ordered = sorted(rows, key=lambda row: -row.bound)
    sizes = [row.encoded_bytes for row in ordered]
    assert sizes == sorted(sizes)


def test_lossless_rows_report_zero_distortion(signals: dict[str, np.ndarray]) -> None:
    rows = ratedistortion.sweep_signal("gyro_rate", signals["gyro_rate"], [1e-4])
    for row in rows:
        if row.coder in baselines.LOSSLESS_METHODS:
            assert row.max_error == 0.0


def test_best_rows_excludes_the_lossless_references(
    signals: dict[str, np.ndarray]
) -> None:
    values = signals["white_noise"]
    rows = ratedistortion.sweep_signal("white_noise", values, [0.1, 0.01])
    best = ratedistortion.best_rows(rows)
    assert set(best) == {("white_noise", 0.1), ("white_noise", 0.01)}
    for row in best.values():
        assert row.coder not in baselines.LOSSLESS_METHODS


def test_entropy_column_is_nan_for_baselines(signals: dict[str, np.ndarray]) -> None:
    rows = ratedistortion.sweep_signal("gyro_rate", signals["gyro_rate"], [1e-4])
    for row in rows:
        if row.coder in baselines.LOSSY_METHODS + baselines.LOSSLESS_METHODS:
            assert np.isnan(row.entropy_bits)
        else:
            assert np.isfinite(row.entropy_bits)
