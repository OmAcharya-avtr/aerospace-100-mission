"""The declared handover sequence: validation, derived quantities, phase lookup."""

from __future__ import annotations

import pytest

from handoverloss.sequence import MECHANISMS, HandoverSequence, Phase


def test_mechanism_order_is_fixed():
    # Every decomposition array in the package uses this column order, and the
    # Phase enum values are the column indices.
    assert MECHANISMS == ("acquisition", "restart", "drain")
    assert (Phase.ACQUISITION, Phase.RESTART, Phase.DRAIN) == (0, 1, 2)


def test_derived_quantities_hand_calculated(worked_sequence):
    # Hand arithmetic for the worked handover:
    #   outage        = 40 + 12               = 52 s
    #   headroom      = 3.2e10 - 2.9e10       = 3.0e9 bits
    #   headroom_s    = 3.0e9 / 2.0e8         = 15 s
    #   utilisation   = 2.0e8 / 1.0e9         = 0.2
    assert worked_sequence.outage_s == pytest.approx(52.0, rel=1e-15, abs=0.0)
    assert worked_sequence.headroom_bits == pytest.approx(3.0e9, rel=1e-15, abs=0.0)
    assert worked_sequence.headroom_s == pytest.approx(15.0, rel=1e-15, abs=0.0)
    assert worked_sequence.utilisation == pytest.approx(0.2, rel=1e-15, abs=0.0)


def test_phase_of_boundaries(worked_sequence):
    assert worked_sequence.phase_of(0.0) is Phase.ACQUISITION
    assert worked_sequence.phase_of(39.999) is Phase.ACQUISITION
    # The boundary belongs to the LATER phase: [0, T_acq) then [T_acq, T_o).
    assert worked_sequence.phase_of(40.0) is Phase.RESTART
    assert worked_sequence.phase_of(51.999) is Phase.RESTART
    assert worked_sequence.phase_of(52.0) is Phase.DRAIN
    # Everything after the ramp finishes is still the drain phase, so the three
    # phases partition the whole accounting window.
    assert worked_sequence.phase_of(699.0) is Phase.DRAIN


def test_egress_rate_schedule_hand_calculated(worked_sequence):
    # Zero during acquisition and restart; linear 0 -> 1e9 over 60 s from t=52;
    # at t = 52 + 30 the rate is half of 1e9.
    assert worked_sequence.egress_rate_bps(10.0) == 0.0
    assert worked_sequence.egress_rate_bps(51.9) == 0.0
    assert worked_sequence.egress_rate_bps(52.0) == pytest.approx(0.0, rel=1e-15, abs=0.0)
    assert worked_sequence.egress_rate_bps(82.0) == pytest.approx(5.0e8, rel=1e-14, abs=0.0)
    assert worked_sequence.egress_rate_bps(112.0) == pytest.approx(1.0e9, rel=1e-15, abs=0.0)
    assert worked_sequence.egress_rate_bps(600.0) == pytest.approx(1.0e9, rel=1e-15, abs=0.0)


def test_zero_ramp_steps_to_full_rate():
    sequence = HandoverSequence(
        source_rate_bps=1.0e8,
        downlink_rate_bps=5.0e8,
        buffer_capacity_bits=1.0e10,
        initial_occupancy_bits=0.0,
        acquisition_s=10.0,
        restart_s=5.0,
        ramp_s=0.0,
        horizon_s=100.0,
    )
    assert sequence.egress_rate_bps(14.9) == 0.0
    assert sequence.egress_rate_bps(15.0) == pytest.approx(5.0e8, rel=1e-15, abs=0.0)


def test_make_before_break_is_accepted():
    sequence = HandoverSequence(
        source_rate_bps=1.0e8,
        downlink_rate_bps=5.0e8,
        buffer_capacity_bits=1.0e10,
        initial_occupancy_bits=1.0e9,
        acquisition_s=0.0,
        restart_s=2.0,
        ramp_s=1.0,
        horizon_s=50.0,
    )
    assert sequence.phase_of(0.0) is Phase.RESTART
    assert sequence.outage_s == pytest.approx(2.0, rel=1e-15, abs=0.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_rate_bps", 0.0),
        ("source_rate_bps", -1.0),
        ("downlink_rate_bps", 0.0),
        ("buffer_capacity_bits", 0.0),
        ("initial_occupancy_bits", -1.0),
        ("acquisition_s", -0.1),
        ("restart_s", -0.1),
        ("ramp_s", -0.1),
    ],
)
def test_out_of_range_fields_raise(field, value):
    kwargs = dict(
        source_rate_bps=1.0e8,
        downlink_rate_bps=5.0e8,
        buffer_capacity_bits=1.0e10,
        initial_occupancy_bits=0.0,
        acquisition_s=10.0,
        restart_s=5.0,
        ramp_s=5.0,
        horizon_s=100.0,
    )
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        HandoverSequence(**kwargs)


def test_overfull_buffer_raises():
    with pytest.raises(ValueError, match="exceeds buffer_capacity_bits"):
        HandoverSequence(
            source_rate_bps=1.0e8,
            downlink_rate_bps=5.0e8,
            buffer_capacity_bits=1.0e10,
            initial_occupancy_bits=1.1e10,
            acquisition_s=10.0,
            restart_s=5.0,
            ramp_s=5.0,
            horizon_s=100.0,
        )


def test_short_horizon_raises():
    with pytest.raises(ValueError, match="shorter than"):
        HandoverSequence(
            source_rate_bps=1.0e8,
            downlink_rate_bps=5.0e8,
            buffer_capacity_bits=1.0e10,
            initial_occupancy_bits=0.0,
            acquisition_s=40.0,
            restart_s=20.0,
            ramp_s=30.0,
            horizon_s=89.0,
        )


def test_non_numeric_field_raises_type_error():
    with pytest.raises(TypeError, match="must be a real number"):
        HandoverSequence(
            source_rate_bps="fast",  # type: ignore[arg-type]
            downlink_rate_bps=5.0e8,
            buffer_capacity_bits=1.0e10,
            initial_occupancy_bits=0.0,
            acquisition_s=10.0,
            restart_s=5.0,
            ramp_s=5.0,
            horizon_s=100.0,
        )


def test_boolean_is_rejected_as_a_rate():
    # bool is a subclass of int; a True rate is a programming error, not a rate.
    with pytest.raises(TypeError, match="must be a real number"):
        HandoverSequence(
            source_rate_bps=True,  # type: ignore[arg-type]
            downlink_rate_bps=5.0e8,
            buffer_capacity_bits=1.0e10,
            initial_occupancy_bits=0.0,
            acquisition_s=10.0,
            restart_s=5.0,
            ramp_s=5.0,
            horizon_s=100.0,
        )


def test_non_finite_field_raises():
    with pytest.raises(ValueError, match="must be finite"):
        HandoverSequence(
            source_rate_bps=float("inf"),
            downlink_rate_bps=5.0e8,
            buffer_capacity_bits=1.0e10,
            initial_occupancy_bits=0.0,
            acquisition_s=10.0,
            restart_s=5.0,
            ramp_s=5.0,
            horizon_s=100.0,
        )


def test_negative_time_lookups_raise(worked_sequence):
    with pytest.raises(ValueError, match="non-negative"):
        worked_sequence.phase_of(-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        worked_sequence.egress_rate_bps(-1.0)
