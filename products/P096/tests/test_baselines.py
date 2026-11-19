"""Baseline 1 and baseline 2: known answers, the shared term, and the branches."""

from __future__ import annotations

import numpy as np
import pytest

from handoverloss.baselines import (
    acquisition_only_loss_bits,
    fluid_decomposition,
    fluid_decomposition_array,
    ramp_drive_bits,
)
from handoverloss.sequence import MECHANISMS, HandoverSequence


def test_fluid_decomposition_known_answer(worked_sequence):
    # HAND ARITHMETIC, worked handover (R_s = 2.0e8, R_d = 1.0e9, B = 3.2e10,
    # q0 = 2.9e10, T_acq = 40, T_rst = 12, T_ramp = 60):
    #
    #   headroom       = 3.2e10 - 2.9e10                   = 3.0e9 bits
    #   acquisition fill = 2.0e8 * 40                       = 8.0e9 bits
    #   L_acq          = 8.0e9 - 3.0e9                      = 5.0e9 bits
    #   buffer now full, so the restart fill is all lost:
    #   L_rst          = 2.0e8 * 12                         = 2.4e9 bits
    #   ramp drive     = R_s^2 * T_ramp / (2 R_d)
    #                  = (2.0e8)^2 * 60 / (2 * 1.0e9)
    #                  = 4.0e16 * 60 / 2.0e9               = 1.2e9 bits
    #   L_drain        = 1.2e9 bits (buffer already full)
    #   total          = 5.0e9 + 2.4e9 + 1.2e9             = 8.6e9 bits
    decomposition = fluid_decomposition(worked_sequence)
    assert decomposition.acquisition_bits == pytest.approx(5.0e9, rel=1e-14, abs=0.0)
    assert decomposition.restart_bits == pytest.approx(2.4e9, rel=1e-14, abs=0.0)
    assert decomposition.drain_bits == pytest.approx(1.2e9, rel=1e-14, abs=0.0)
    assert decomposition.total_bits == pytest.approx(8.6e9, rel=1e-14, abs=0.0)
    assert decomposition.dominant == "acquisition"
    assert decomposition.drains is True


def test_acquisition_only_known_answer(worked_sequence):
    # HAND ARITHMETIC: max(0, 2.0e8 * 40 - 3.0e9) = max(0, 5.0e9) = 5.0e9 bits.
    assert acquisition_only_loss_bits(worked_sequence) == pytest.approx(
        5.0e9, rel=1e-14, abs=0.0
    )


def test_acquisition_only_equals_fluid_acquisition_term(worked_sequence):
    # Baseline 1 is exactly the restriction of baseline 2 to its first phase.
    # This is asserted because it is what makes baseline 1 full strength: every
    # difference between the two is attributable to the two omitted phases and
    # to nothing else. An under-strength baseline 1 would break this equality.
    decomposition = fluid_decomposition(worked_sequence)
    assert acquisition_only_loss_bits(worked_sequence) == pytest.approx(
        decomposition.acquisition_bits, rel=1e-15, abs=0.0
    )


def test_acquisition_only_is_a_lower_bound_on_the_fluid_total(rng):
    # Adding phases can only add loss, so baseline 1 can never exceed baseline 2
    # evaluated at the same acquisition time. Checked over random handovers.
    for _ in range(200):
        source = float(rng.uniform(1e7, 5e8))
        downlink = float(rng.uniform(2e8, 3e9))
        capacity = float(rng.uniform(1e9, 8e10))
        occupancy = float(rng.uniform(0.0, capacity))
        sequence = HandoverSequence(
            source_rate_bps=source,
            downlink_rate_bps=downlink,
            buffer_capacity_bits=capacity,
            initial_occupancy_bits=occupancy,
            acquisition_s=float(rng.uniform(0.0, 120.0)),
            restart_s=float(rng.uniform(0.0, 80.0)),
            ramp_s=float(rng.uniform(0.0, 180.0)),
            horizon_s=800.0,
        )
        assert acquisition_only_loss_bits(sequence) <= (
            fluid_decomposition(sequence).total_bits * (1.0 + 1e-12) + 1.0
        )


def test_ramp_drive_closed_form_hand_calculated():
    # R_s = 3.0e8, R_d = 1.2e9, T_ramp = 40 s:
    #   drive = (3.0e8)^2 * 40 / (2 * 1.2e9) = 9.0e16 * 40 / 2.4e9 = 1.5e9 bits
    sequence = HandoverSequence(
        source_rate_bps=3.0e8,
        downlink_rate_bps=1.2e9,
        buffer_capacity_bits=1.0e11,
        initial_occupancy_bits=0.0,
        acquisition_s=5.0,
        restart_s=5.0,
        ramp_s=40.0,
        horizon_s=400.0,
    )
    assert ramp_drive_bits(sequence) == pytest.approx(1.5e9, rel=1e-14, abs=0.0)


def test_ramp_drive_is_continuous_at_unit_utilisation():
    # The two branches of step 3 agree at R_s == R_d: the draining form gives
    # R_s^2 T / (2 R_d) = 0.5 R_s T, and the saturated form gives
    # R_s T - 0.5 R_d T + 0 = 0.5 R_s T. Asserted to within rounding rather than
    # exactly, because the two expressions round differently.
    common = dict(
        buffer_capacity_bits=1.0e11,
        initial_occupancy_bits=0.0,
        acquisition_s=1.0,
        restart_s=1.0,
        ramp_s=50.0,
        horizon_s=600.0,
    )
    at_unit = ramp_drive_bits(
        HandoverSequence(source_rate_bps=1.0e9, downlink_rate_bps=1.0e9, **common)
    )
    just_below = ramp_drive_bits(
        HandoverSequence(
            source_rate_bps=1.0e9 * (1.0 - 1e-9), downlink_rate_bps=1.0e9, **common
        )
    )
    assert at_unit == pytest.approx(0.5 * 1.0e9 * 50.0, rel=1e-12, abs=0.0)
    assert at_unit == pytest.approx(just_below, rel=1e-8, abs=0.0)


def test_saturated_branch_is_horizon_dependent_and_flagged():
    # R_s > R_d: the backlog never drains, the drain term depends on the window,
    # and the object says so rather than returning a number that looks like a
    # property of the handover alone.
    short = HandoverSequence(
        source_rate_bps=1.0e9,
        downlink_rate_bps=4.0e8,
        buffer_capacity_bits=1.0e10,
        initial_occupancy_bits=0.0,
        acquisition_s=2.0,
        restart_s=2.0,
        ramp_s=10.0,
        horizon_s=100.0,
    )
    long = HandoverSequence(
        source_rate_bps=1.0e9,
        downlink_rate_bps=4.0e8,
        buffer_capacity_bits=1.0e10,
        initial_occupancy_bits=0.0,
        acquisition_s=2.0,
        restart_s=2.0,
        ramp_s=10.0,
        horizon_s=400.0,
    )
    assert fluid_decomposition(short).drains is False
    assert fluid_decomposition(long).drain_bits > fluid_decomposition(short).drain_bits


def test_no_loss_when_the_buffer_absorbs_everything():
    # Headroom 1e10 bits at 1e8 bit/s absorbs 100 s. Outage is 20 s and the ramp
    # drive is 1e8^2 * 10 / (2e9) = 5e7 bits, well inside the headroom.
    sequence = HandoverSequence(
        source_rate_bps=1.0e8,
        downlink_rate_bps=1.0e9,
        buffer_capacity_bits=2.0e10,
        initial_occupancy_bits=1.0e10,
        acquisition_s=10.0,
        restart_s=10.0,
        ramp_s=10.0,
        horizon_s=400.0,
    )
    decomposition = fluid_decomposition(sequence)
    assert decomposition.total_bits == 0.0
    assert decomposition.mechanisms == dict.fromkeys(MECHANISMS, 0.0)
    assert acquisition_only_loss_bits(sequence) == 0.0


def test_sum_residual_is_reported_and_tiny(worked_sequence):
    decomposition = fluid_decomposition(worked_sequence)
    # Exact in exact arithmetic; the value is the rounding of a three-term sum.
    assert abs(decomposition.sum_residual_bits) <= 1e-9 * decomposition.total_bits


def test_vectorised_agrees_with_scalar(rng):
    count = 300
    source = rng.uniform(1e7, 1e9, count)
    downlink = rng.uniform(2e8, 3e9, count)
    capacity = rng.uniform(1e9, 8e10, count)
    occupancy = rng.uniform(0.0, 1.0, count) * capacity
    acquisition = rng.uniform(0.0, 150.0, count)
    restart = rng.uniform(0.0, 90.0, count)
    ramp = rng.uniform(0.0, 200.0, count)
    horizon = np.full(count, 900.0)
    batch = fluid_decomposition_array(
        source, downlink, capacity, occupancy, acquisition, restart, ramp, horizon
    )
    for row in range(count):
        scalar = fluid_decomposition(
            HandoverSequence(
                source_rate_bps=float(source[row]),
                downlink_rate_bps=float(downlink[row]),
                buffer_capacity_bits=float(capacity[row]),
                initial_occupancy_bits=float(occupancy[row]),
                acquisition_s=float(acquisition[row]),
                restart_s=float(restart[row]),
                ramp_s=float(ramp[row]),
                horizon_s=900.0,
            )
        )
        expected = (scalar.acquisition_bits, scalar.restart_bits, scalar.drain_bits)
        for column in range(3):
            # Agreeing to within rounding, not bit-exactly: the two evaluation
            # orders are not guaranteed to round identically.
            assert batch[row, column] == pytest.approx(
                expected[column], rel=1e-12, abs=1.0
            )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (0, -1.0, "strictly positive"),
        (1, 0.0, "strictly positive"),
        (2, 0.0, "strictly positive"),
        (3, -1.0, "non-negative"),
        (4, -1.0, "non-negative"),
    ],
)
def test_vectorised_rejects_out_of_range(field, value, message):
    arguments = [
        np.array([1e8]),
        np.array([1e9]),
        np.array([1e10]),
        np.array([0.0]),
        np.array([10.0]),
        np.array([5.0]),
        np.array([5.0]),
        np.array([400.0]),
    ]
    arguments[field] = np.array([value])
    with pytest.raises(ValueError, match=message):
        fluid_decomposition_array(*arguments)


def test_vectorised_rejects_short_horizon():
    with pytest.raises(ValueError, match="horizon_s must be at least"):
        fluid_decomposition_array(
            np.array([1e8]),
            np.array([1e9]),
            np.array([1e10]),
            np.array([0.0]),
            np.array([40.0]),
            np.array([20.0]),
            np.array([30.0]),
            np.array([50.0]),
        )


def test_vectorised_rejects_overfull_buffer():
    with pytest.raises(ValueError, match="must not exceed"):
        fluid_decomposition_array(
            np.array([1e8]),
            np.array([1e9]),
            np.array([1e10]),
            np.array([2e10]),
            np.array([10.0]),
            np.array([5.0]),
            np.array([5.0]),
            np.array([400.0]),
        )


def test_acquisition_only_rejects_negative_duration(worked_sequence):
    with pytest.raises(ValueError, match="non-negative"):
        acquisition_only_loss_bits(worked_sequence, acquisition_s=-1.0)


def test_acquisition_only_at_an_upper_quantile_is_larger(worked_sequence):
    # The conservative engineer's version of baseline 1: evaluate at an upper
    # quantile of the acquisition distribution rather than at its mean.
    assert acquisition_only_loss_bits(
        worked_sequence, acquisition_s=90.0
    ) > acquisition_only_loss_bits(worked_sequence)
