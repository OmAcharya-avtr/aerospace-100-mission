"""The attribution report: the residual, the decided/undecided distinction, validation."""

from __future__ import annotations

import numpy as np
import pytest

from handoverloss.accounting import (
    BITS_PER_GBIT,
    AttributionReport,
    MechanismShare,
    attribute,
    residual_tolerance_bits,
)
from handoverloss.bootstrap import RESAMPLING_UNIT
from handoverloss.sequence import MECHANISMS, HandoverSequence
from handoverloss.trace import EventParameters, simulate


@pytest.fixture
def trace(rng):
    sequences = [
        HandoverSequence(
            source_rate_bps=2.0e8,
            downlink_rate_bps=1.0e9,
            buffer_capacity_bits=3.2e10,
            initial_occupancy_bits=2.9e10,
            acquisition_s=float(duration),
            restart_s=12.0,
            ramp_s=60.0,
            horizon_s=700.0,
        )
        for duration in np.linspace(5.0, 110.0, 40)
    ]
    return simulate(EventParameters.from_sequences(sequences), rng, slot_seconds=0.5)


def test_report_shape_and_order(trace):
    report = attribute(trace, resamples=200)
    assert tuple(share.name for share in report.mechanisms) == MECHANISMS
    assert report.n_events == 40
    assert report.n_resamples == 200
    assert report.resampling_unit == RESAMPLING_UNIT
    assert report.dominant in MECHANISMS


def test_mechanism_shares_sum_to_one(trace):
    report = attribute(trace, resamples=200)
    total_share = sum(share.share for share in report.mechanisms)
    assert total_share == pytest.approx(1.0, rel=1e-12, abs=0.0)


def test_decomposition_residual_is_within_the_conditioning_tolerance(trace):
    # The residual compares the bucketed sum against mass conservation, which
    # never touches the buckets. Asserted against the conditioning-scaled
    # tolerance, and both numbers are reported by the object.
    report = attribute(trace, resamples=200)
    assert abs(report.residual_bits) <= report.residual_tolerance_bits
    assert abs(report.residual_relative) < 1e-9
    assert report.residual_within_tolerance is True


def test_residual_tolerance_scales_with_slots_and_magnitude():
    small = float(residual_tolerance_bits(1.0e9, 1.0e9, 1.0e8, 100))
    more_slots = float(residual_tolerance_bits(1.0e9, 1.0e9, 1.0e8, 1000))
    bigger = float(residual_tolerance_bits(1.0e12, 1.0e12, 1.0e11, 100))
    assert more_slots == pytest.approx(10.0 * small, rel=1e-12, abs=0.0)
    assert bigger == pytest.approx(1000.0 * small, rel=1e-12, abs=0.0)


def test_residual_tolerance_rejects_zero_slots():
    with pytest.raises(ValueError, match="at least 1"):
        residual_tolerance_bits(1.0, 1.0, 1.0, 0)


def test_dominance_decided_only_when_intervals_are_disjoint(trace):
    report = attribute(trace, resamples=400)
    leader, runner_up = report.ranked()[0], report.ranked()[1]
    # The reported boolean must agree with the bounds it is derived from.
    disjoint = leader.low_bits > runner_up.high_bits or runner_up.low_bits > leader.high_bits
    assert report.dominance_is_decided == disjoint


def test_undecided_dominance_when_two_mechanisms_are_equal(rng):
    # A hand-built trace in which the acquisition and restart buckets are drawn
    # from the same distribution: the leading interval must overlap the
    # runner-up's, so the dominance is NOT decided.
    count = 60
    dropped = np.empty((count, 3), dtype=float)
    dropped[:, 0] = rng.normal(1.0e9, 2.0e8, count)
    dropped[:, 1] = rng.normal(1.0e9, 2.0e8, count)
    dropped[:, 2] = rng.normal(1.0e7, 1.0e6, count)
    dropped = np.abs(dropped)
    from handoverloss.trace import TraceResult

    fake = TraceResult(
        dropped_bits=dropped,
        produced_bits=np.full(count, 1.0e11),
        delivered_bits=np.full(count, 1.0e11) - dropped.sum(axis=1),
        final_occupancy_bits=np.zeros(count),
        initial_occupancy_bits=np.zeros(count),
        peak_occupancy_bits=np.zeros(count),
        slots_full=np.zeros(count, dtype=np.int64),
        slot_seconds=0.25,
        slot_count=1000,
    )
    report = attribute(fake, resamples=400)
    assert report.dominance_is_decided is False
    assert report.dominance_margin < 0.1


def test_lines_render_every_field(trace):
    report = attribute(trace, resamples=200)
    text = "\n".join(report.lines())
    for name in MECHANISMS:
        assert name in text
    assert "resampling unit        : handover event" in text
    assert "decomposition residual" in text
    assert "dominance decided" in text


def test_gigabit_helpers(trace):
    report = attribute(trace, resamples=200)
    assert report.total_loss_gbit == pytest.approx(
        report.total_loss_bits / BITS_PER_GBIT, rel=1e-15, abs=0.0
    )
    for share in report.mechanisms:
        assert share.loss_gbit == pytest.approx(
            share.loss_bits / 1.0e9, rel=1e-15, abs=0.0
        )
        assert share.low_gbit == pytest.approx(
            share.low_bits / 1.0e9, rel=1e-15, abs=0.0
        )
        assert share.high_gbit == pytest.approx(
            share.high_bits / 1.0e9, rel=1e-15, abs=0.0
        )
        assert share.low_gbit <= share.high_gbit


def test_share_lookup(trace):
    report = attribute(trace, resamples=200)
    assert report.share("drain").name == "drain"
    with pytest.raises(KeyError, match="no mechanism named"):
        report.share("thermal")


def test_mechanism_share_rejects_unknown_name():
    with pytest.raises(ValueError, match="must be one of"):
        MechanismShare(name="thermal", loss_bits=1.0, low_bits=0.0, high_bits=2.0, share=1.0)


def test_mechanism_share_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="exceeds"):
        MechanismShare(
            name="drain", loss_bits=1.0, low_bits=2.0, high_bits=1.0, share=1.0
        )


def _share(name: str) -> MechanismShare:
    return MechanismShare(name=name, loss_bits=1.0, low_bits=0.0, high_bits=2.0, share=1 / 3)


def test_report_rejects_a_foreign_resampling_unit():
    with pytest.raises(ValueError, match="resampling_unit must be"):
        AttributionReport(
            mechanisms=tuple(_share(name) for name in MECHANISMS),  # type: ignore[arg-type]
            dominant="drain",
            dominance_margin=0.0,
            dominance_is_decided=False,
            total_loss_bits=3.0,
            conservation_total_bits=3.0,
            residual_bits=0.0,
            residual_relative=0.0,
            residual_tolerance_bits=1.0,
            resampling_unit="accounting slot",
            n_events=10,
            n_resamples=10,
            confidence=0.95,
        )


def test_report_rejects_mechanisms_out_of_order():
    with pytest.raises(ValueError, match="must be in the order"):
        AttributionReport(
            mechanisms=(_share("drain"), _share("restart"), _share("acquisition")),
            dominant="drain",
            dominance_margin=0.0,
            dominance_is_decided=False,
            total_loss_bits=3.0,
            conservation_total_bits=3.0,
            residual_bits=0.0,
            residual_relative=0.0,
            residual_tolerance_bits=1.0,
            resampling_unit=RESAMPLING_UNIT,
            n_events=10,
            n_resamples=10,
            confidence=0.95,
        )


def test_report_rejects_an_unknown_dominant():
    with pytest.raises(ValueError, match="dominant must be one of"):
        AttributionReport(
            mechanisms=tuple(_share(name) for name in MECHANISMS),  # type: ignore[arg-type]
            dominant="thermal",
            dominance_margin=0.0,
            dominance_is_decided=False,
            total_loss_bits=3.0,
            conservation_total_bits=3.0,
            residual_bits=0.0,
            residual_relative=0.0,
            residual_tolerance_bits=1.0,
            resampling_unit=RESAMPLING_UNIT,
            n_events=10,
            n_resamples=10,
            confidence=0.95,
        )


def test_attribute_rejects_a_wrong_shaped_trace(trace):
    from handoverloss.trace import TraceResult

    broken = TraceResult(
        dropped_bits=np.zeros((10, 2)),
        produced_bits=np.zeros(10),
        delivered_bits=np.zeros(10),
        final_occupancy_bits=np.zeros(10),
        initial_occupancy_bits=np.zeros(10),
        peak_occupancy_bits=np.zeros(10),
        slots_full=np.zeros(10, dtype=np.int64),
        slot_seconds=0.25,
        slot_count=100,
    )
    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        attribute(broken)


def test_attribute_rejects_too_few_events(rng):
    sequences = [
        HandoverSequence(
            source_rate_bps=2.0e8,
            downlink_rate_bps=1.0e9,
            buffer_capacity_bits=3.2e10,
            initial_occupancy_bits=2.9e10,
            acquisition_s=40.0,
            restart_s=12.0,
            ramp_s=60.0,
            horizon_s=700.0,
        )
    ] * 3
    trace = simulate(EventParameters.from_sequences(sequences), rng, slot_seconds=1.0)
    with pytest.raises(ValueError, match="at least 5 events"):
        attribute(trace)
