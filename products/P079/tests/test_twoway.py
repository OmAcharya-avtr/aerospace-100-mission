"""Baseline 1: the symmetric-delay estimator and its closed-form asymmetry bias.

The central claims of this module are testable arithmetic, so they are tested
as arithmetic: a hand-calculated exchange, exactness under symmetry, and the
slope of the bias against a declared asymmetry.
"""

from __future__ import annotations

import numpy as np
import pytest

from timetransfer.acquire import acquire
from timetransfer.backends import SimulatedBackend
from timetransfer.errors import ConfigurationError, InsufficientDataError
from timetransfer.exchange import ExchangeSet
from timetransfer.twoway import (
    asymmetry_bias_s,
    asymmetry_from_bias_s,
    offset_skew_correction_s,
    symmetric_offsets,
    two_way_estimate,
)

from conftest import make_spec


def _one(t1: int, t2: int, t3: int, t4: int) -> ExchangeSet:
    return ExchangeSet(
        t1_ns=np.asarray([t1], dtype=np.int64),
        t2_ns=np.asarray([t2], dtype=np.int64),
        t3_ns=np.asarray([t3], dtype=np.int64),
        t4_ns=np.asarray([t4], dtype=np.int64),
        seq=np.asarray([0], dtype=np.int64),
    )


def test_offset_and_delay_known_answer() -> None:
    # Hand calculation. Let theta = +2 ms, d_ab = d_ba = 5 ms, turnaround 1 ms.
    #   t1 = 0
    #   t2 = t1 + d_ab + theta        = 0 + 5e6 + 2e6  = 7_000_000 ns
    #   t3 = t2 + turnaround          = 7e6 + 1e6      = 8_000_000 ns
    #   t4 = t1 + d_ab + turn + d_ba  = 5e6 + 1e6 + 5e6 = 11_000_000 ns
    # offset = ((t2-t1) - (t4-t3))/2 = (7e6 - 3e6)/2 = 2e6 ns = 2 ms  -> theta
    # delay  = ((t2-t1) + (t4-t3))/2 = (7e6 + 3e6)/2 = 5e6 ns = 5 ms  -> d
    offsets, delays = symmetric_offsets(_one(0, 7_000_000, 8_000_000, 11_000_000))
    assert offsets[0] == pytest.approx(2.0e-3, rel=1e-15)
    assert delays[0] == pytest.approx(5.0e-3, rel=1e-15)


def test_exact_under_symmetry_to_within_rounding() -> None:
    # With d_ab == d_ba exactly and integer timestamps, the estimator returns
    # theta with no error term in exact arithmetic. The claim implemented is
    # agreement to within the float division, which is what is asserted.
    theta_ns = 2_000_000
    d_ns = 5_000_000
    turn_ns = 1_000_000
    t1 = 0
    t2 = t1 + d_ns + theta_ns
    t3 = t2 + turn_ns
    t4 = t1 + d_ns + turn_ns + d_ns
    offsets, _ = symmetric_offsets(_one(t1, t2, t3, t4))
    assert offsets[0] == pytest.approx(theta_ns * 1e-9, abs=1e-15)


def test_bias_is_half_the_asymmetry_exactly() -> None:
    assert asymmetry_bias_s(8.0e-4) == pytest.approx(4.0e-4, rel=1e-15)
    assert asymmetry_bias_s(-8.0e-4) == pytest.approx(-4.0e-4, rel=1e-15)
    assert asymmetry_bias_s(0.0) == 0.0


def test_bias_and_its_inverse_round_trip() -> None:
    for value in (-1e-3, -1e-9, 0.0, 1e-9, 1e-3, 1.0):
        assert asymmetry_from_bias_s(asymmetry_bias_s(value)) == pytest.approx(value, rel=1e-15)


def test_bias_vectorises() -> None:
    out = asymmetry_bias_s(np.asarray([-2.0, 0.0, 4.0]))
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [-1.0, 0.0, 2.0]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_bias_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ConfigurationError, match="asymmetry_s"):
        asymmetry_bias_s(bad)


def test_asymmetry_injected_into_an_exchange_appears_as_half_of_it() -> None:
    # Construct the exchange directly so that the asymmetry is exact.
    theta_ns = 0
    turn_ns = 1_000_000
    for asym_ns in (-2_000_000, -500_000, 0, 500_000, 2_000_000):
        d_ab = 5_000_000 + asym_ns // 2
        d_ba = 5_000_000 - asym_ns // 2
        t1 = 0
        t2 = t1 + d_ab + theta_ns
        t3 = t2 + turn_ns
        t4 = t1 + d_ab + turn_ns + d_ba
        offsets, _ = symmetric_offsets(_one(t1, t2, t3, t4))
        expected = asymmetry_bias_s(float(d_ab - d_ba) * 1e-9)
        assert offsets[0] == pytest.approx(expected, abs=1e-15)


def test_declared_asymmetry_bias_is_recovered_from_a_simulated_run() -> None:
    # A noiseless, jitter-free symmetric-in-queue link with a declared static
    # asymmetry: the mean estimation error must equal the closed-form bias.
    asym = 8.0e-4
    spec = make_spec(asymmetry_s=asym, jitter_scale_s=0.0, skew_s_per_s=0.0, seed=31)
    backend = SimulatedBackend(spec)
    backend.open()
    report = acquire(backend, 120, min_exchanges=8)
    backend.close()
    estimate = two_way_estimate(report.exchanges)
    truth = report.exchanges.true_offset_s
    assert truth is not None
    error = float(np.mean(estimate.offset_s - truth))
    assert error == pytest.approx(asymmetry_bias_s(asym), abs=2e-9)


def test_bias_slope_against_declared_asymmetry_is_one_half() -> None:
    asymmetries = np.asarray([-1.6e-3, -8.0e-4, 0.0, 8.0e-4, 1.6e-3])
    errors = []
    for asym in asymmetries:
        spec = make_spec(
            asymmetry_s=float(asym), jitter_scale_s=0.0, skew_s_per_s=0.0, seed=77
        )
        backend = SimulatedBackend(spec)
        backend.open()
        report = acquire(backend, 60, min_exchanges=8)
        backend.close()
        estimate = two_way_estimate(report.exchanges)
        truth = report.exchanges.true_offset_s
        assert truth is not None
        errors.append(float(np.mean(estimate.offset_s - truth)))
    slope = float(np.polyfit(asymmetries, np.asarray(errors), 1)[0])
    assert slope == pytest.approx(0.5, abs=1e-6)


def test_skew_correction_bound_known_answer() -> None:
    # |a| * R / 2 = 1e-6 * 1e-2 / 2 = 5e-9 s
    assert offset_skew_correction_s(1.0e-2, 1.0e-6) == pytest.approx(5.0e-9, rel=1e-12)
    assert offset_skew_correction_s(1.0e-2, -1.0e-6) == pytest.approx(5.0e-9, rel=1e-12)


def test_skew_correction_rejects_negative_round_trip() -> None:
    with pytest.raises(ConfigurationError, match="round_trip_s"):
        offset_skew_correction_s(-1.0, 1e-6)


def test_two_way_estimate_summary_fields(exchanges: ExchangeSet) -> None:
    estimate = two_way_estimate(exchanges, assumed_skew_s_per_s=2.0e-6)
    assert estimate.n == exchanges.n
    assert estimate.offset_mean_s == pytest.approx(float(np.mean(estimate.offset_s)))
    assert estimate.offset_median_s == pytest.approx(float(np.median(estimate.offset_s)))
    assert estimate.delay_min_s == pytest.approx(float(np.min(estimate.delay_s)))
    assert estimate.skew_correction_bound_s > 0.0
    assert "skew correction bound" in estimate.as_text()


def test_corrected_offset_subtracts_the_bias(exchanges: ExchangeSet) -> None:
    estimate = two_way_estimate(exchanges)
    corrected = estimate.corrected_offset_s(1.0e-4)
    assert corrected == pytest.approx(estimate.offset_s - 1.0e-4)
    per_row = np.full(estimate.n, 2.0e-4)
    assert estimate.corrected_offset_s(per_row) == pytest.approx(estimate.offset_s - 2.0e-4)


def test_corrected_offset_rejects_a_wrong_length_bias(exchanges: ExchangeSet) -> None:
    estimate = two_way_estimate(exchanges)
    with pytest.raises(ConfigurationError, match="bias_s has"):
        estimate.corrected_offset_s(np.zeros(estimate.n + 1))


def test_symmetric_offsets_rejects_an_empty_set() -> None:
    empty = ExchangeSet(
        t1_ns=np.zeros(0, dtype=np.int64),
        t2_ns=np.zeros(0, dtype=np.int64),
        t3_ns=np.zeros(0, dtype=np.int64),
        t4_ns=np.zeros(0, dtype=np.int64),
        seq=np.zeros(0, dtype=np.int64),
    )
    with pytest.raises(InsufficientDataError):
        symmetric_offsets(empty)
