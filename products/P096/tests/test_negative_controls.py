"""Negative controls: every one anchored on the TRUE value, and asserted to fire.

A check that never fails is not a check. Each control here injects a violation
anchored on the GROUND TRUTH rather than on an output, and asserts that the
guard it exercises FIRES. Two controls are declared must-NOT-fire, and both
directions are asserted, because a control that fires when it must not is
equally wrong.

The anchoring matters. Offsetting an output that already sits inside a band does
not guarantee leaving the band: only an offset strictly greater than twice the
bound does. An earlier product of this mission shipped two of eleven controls
silently too weak for exactly that reason, and the certificate read as evidence
while the control was wrong.

``validation/validate_negative_controls.py`` runs the same controls and reports
how many of each kind fired, as its own line.
"""

from __future__ import annotations

import numpy as np
import pytest

from handoverloss.accounting import attribute, residual_tolerance_bits
from handoverloss.baselines import fluid_decomposition
from handoverloss.bootstrap import event_bootstrap
from handoverloss.bounds import loss_bracket
from handoverloss.sequence import HandoverSequence
from handoverloss.trace import EventParameters, TraceResult, simulate


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


def _perturbed(trace: TraceResult, column: int, factor: float) -> TraceResult:
    """A trace whose one mechanism bucket is multiplied, leaving conservation alone.

    The perturbation is applied to the BUCKETS and not to the conservation
    totals, so the decomposition residual must detect it. That is the point: the
    residual compares two independent accountings, and breaking one of them must
    move it.
    """
    dropped = trace.dropped_bits.copy()
    dropped[:, column] *= factor
    return TraceResult(
        dropped_bits=dropped,
        produced_bits=trace.produced_bits,
        delivered_bits=trace.delivered_bits,
        final_occupancy_bits=trace.final_occupancy_bits,
        initial_occupancy_bits=trace.initial_occupancy_bits,
        peak_occupancy_bits=trace.peak_occupancy_bits,
        slots_full=trace.slots_full,
        slot_seconds=trace.slot_seconds,
        slot_count=trace.slot_count,
    )


def test_control_residual_fires_on_a_one_per_cent_bucket_error(trace):
    """MUST FIRE. A 1 per cent inflation of the acquisition bucket.

    Anchored on the true bucket value: the injected error is 0.01 times the
    TRUE acquisition volume, which is many orders of magnitude above the
    conditioning-scaled tolerance, so the violation is guaranteed rather than
    hoped for.
    """
    clean = attribute(trace, resamples=100)
    assert clean.residual_within_tolerance is True
    dirty = attribute(_perturbed(trace, 0, 1.01), resamples=100)
    injected = 0.01 * float(trace.dropped_bits[:, 0].mean())
    assert injected > 100.0 * dirty.residual_tolerance_bits
    assert dirty.residual_within_tolerance is False
    assert abs(dirty.residual_bits) > dirty.residual_tolerance_bits


def test_control_residual_fires_on_a_dropped_mechanism(trace):
    """MUST FIRE. Zeroing the restart bucket entirely."""
    dirty = attribute(_perturbed(trace, 1, 0.0), resamples=100)
    true_restart = float(trace.dropped_bits[:, 1].mean())
    assert true_restart > 100.0 * dirty.residual_tolerance_bits
    assert dirty.residual_within_tolerance is False


def test_control_residual_does_not_fire_on_the_unperturbed_trace(trace):
    """MUST NOT FIRE. The clean trace's residual stays inside its tolerance.

    Asserted as its own control because a residual check that fires on correct
    input is as useless as one that never fires.
    """
    report = attribute(trace, resamples=100)
    assert report.residual_within_tolerance is True
    assert abs(report.residual_bits) <= report.residual_tolerance_bits


def test_control_independent_bracket_fires_on_an_inflated_total():
    """MUST FIRE. A total above the offered volume.

    Anchored on the TRUE offered volume, not on a model output: the injected
    value is twice the offered volume plus twice the slack, so it is guaranteed
    to leave the bracket from above whatever the model returns.
    """
    offered = 1.0e11
    bracket = loss_bracket(
        offered_bits=offered,
        downlink_rate_bps=1.0e9,
        outage_s=90.0,
        ramp_s=10.0,
        horizon_s=100.0,
        buffer_capacity_bits=1.0e10,
        initial_occupancy_bits=0.0,
    )
    injected = 2.0 * offered + 2.0 * bracket.slack_bits
    assert bracket.brackets(injected) is False
    assert bracket.gap_above_bits(injected) > bracket.slack_bits


def test_control_independent_bracket_fires_on_a_total_below_the_pigeonhole_bound():
    """MUST FIRE. A total below what conservation forces.

    Anchored on the TRUE lower bound: the injected value is the lower bound
    minus twice the slack, which is guaranteed to be outside. An offset applied
    to a model output that already sits above the bound would not be.
    """
    bracket = loss_bracket(
        offered_bits=1.0e11,
        downlink_rate_bps=1.0e9,
        outage_s=90.0,
        ramp_s=10.0,
        horizon_s=100.0,
        buffer_capacity_bits=1.0e10,
        initial_occupancy_bits=0.0,
    )
    assert bracket.lower_bits > 0.0
    injected = bracket.lower_bits - 2.0 * bracket.slack_bits - 1.0e6
    assert bracket.brackets(injected) is False
    assert bracket.gap_below_bits(injected) > bracket.slack_bits


def test_control_independent_bracket_does_not_fire_on_the_closed_form(worked_sequence):
    """MUST NOT FIRE. The honest closed-form total stays inside the bracket."""
    decomposition = fluid_decomposition(worked_sequence)
    bracket = loss_bracket(
        offered_bits=worked_sequence.source_rate_bps * worked_sequence.horizon_s,
        downlink_rate_bps=worked_sequence.downlink_rate_bps,
        outage_s=worked_sequence.outage_s,
        ramp_s=worked_sequence.ramp_s,
        horizon_s=worked_sequence.horizon_s,
        buffer_capacity_bits=worked_sequence.buffer_capacity_bits,
        initial_occupancy_bits=worked_sequence.initial_occupancy_bits,
    )
    assert bracket.brackets(decomposition.total_bits) is True


def test_control_slot_level_bootstrap_would_report_a_narrower_interval(trace):
    """MUST FIRE. Resampling below the unit of independence narrows the interval.

    The reason the resampling unit is declared and asserted. A per-event
    bootstrap over 40 handovers and a bootstrap over the same total volume split
    into 40 x 10 pseudo-independent pieces give intervals of different widths,
    and the pseudo-independent one is narrower. Asserted as a RATIO of two
    interval widths, which are both statistics of the same committed data and
    not wall-clock readings.
    """
    per_event = event_bootstrap(
        trace.dropped_bits,
        lambda rows: float(rows[:, 0].mean()),
        resamples=2000,
        rng=np.random.default_rng(1),
    )
    # Split each event's volume into 10 equal pieces and treat them as units.
    # This is what resampling slots amounts to, and it is wrong.
    pieces = np.repeat(trace.dropped_bits[:, 0] / 10.0, 10).reshape(-1, 1)
    pseudo = event_bootstrap(
        pieces,
        lambda rows: float(rows[:, 0].mean() * 10.0),
        resamples=2000,
        rng=np.random.default_rng(1),
    )
    assert pseudo.width < 0.9 * per_event.width


def test_control_a_foreign_resampling_unit_is_refused():
    """MUST FIRE. The report refuses to carry any unit but the handover event."""
    from handoverloss.accounting import AttributionReport, MechanismShare
    from handoverloss.sequence import MECHANISMS

    shares = tuple(
        MechanismShare(name=name, loss_bits=1.0, low_bits=0.0, high_bits=2.0, share=1 / 3)
        for name in MECHANISMS
    )
    with pytest.raises(ValueError, match="resampling_unit must be"):
        AttributionReport(
            mechanisms=shares,  # type: ignore[arg-type]
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


def test_control_residual_tolerance_cannot_be_made_vacuous():
    """MUST FIRE. A tolerance scaled to zero magnitude admits nothing.

    The conditioning-scaled tolerance must collapse when the magnitudes do;
    a tolerance floored at an absolute constant would admit the very error the
    check exists to catch, which is the defect class this mission recorded.
    """
    tiny = float(residual_tolerance_bits(0.0, 0.0, 0.0, 1))
    assert tiny == 0.0
    # And a real error against zero magnitude is therefore detected.
    assert abs(1.0) > tiny
