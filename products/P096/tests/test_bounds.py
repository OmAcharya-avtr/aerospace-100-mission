"""The independently derived bracket: known answers, independence, conditioning."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from handoverloss.baselines import fluid_decomposition
from handoverloss.bounds import (
    UNIT_ROUNDOFF,
    bracket_slack_bits,
    loss_bracket,
    max_service_bits,
)
from handoverloss.sequence import HandoverSequence


def test_bounds_module_imports_no_model():
    # The bracket is only independent evidence if it does not call the models it
    # checks. Asserted structurally, by parsing this package's own source, so a
    # future import cannot quietly make the check circular.
    source = Path("src/handoverloss/bounds.py").read_text()
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    forbidden = {"baselines", "trace", "accounting", "predictor", "corpus"}
    assert not (imported & forbidden), f"bounds.py imports a model: {imported}"
    assert not any(name.startswith(".") for name in imported)


def test_max_service_integral_hand_calculated():
    # R_d = 1.0e9 bit/s, T_o = 52 s, T_ramp = 60 s, H = 700 s.
    #   ramp area = 0.5 * 1.0e9 * 60                    = 3.0e10 bits
    #   flat area = 1.0e9 * (700 - 52 - 60) = 1.0e9*588 = 5.88e11 bits
    #   S         = 3.0e10 + 5.88e11                    = 6.18e11 bits
    service = max_service_bits(1.0e9, 52.0, 60.0, 700.0)
    assert service == pytest.approx(6.18e11, rel=1e-14, abs=0.0)


def test_max_service_integral_window_ending_inside_the_ramp():
    # Window ends 30 s into a 60 s ramp. The ramp's area over its first half is
    # 0.5 * R_d * (30^2 / 60) = 0.5 * 1.0e9 * 15 = 7.5e9 bits, and there is no
    # flat part.
    service = max_service_bits(1.0e9, 52.0, 60.0, 82.0)
    assert service == pytest.approx(7.5e9, rel=1e-14, abs=0.0)


def test_max_service_integral_window_ending_in_the_outage():
    assert max_service_bits(1.0e9, 52.0, 60.0, 52.0) == pytest.approx(
        0.0, rel=0.0, abs=0.0
    )


def test_bracket_hand_calculated(worked_sequence):
    # Fluid case: A = R_s * H = 2.0e8 * 700 = 1.4e11 bits.
    #   S        = 6.18e11 bits (above)
    #   headroom = 3.0e9 bits
    #   lower    = max(0, 1.4e11 - 6.18e11 - 3.0e9) = 0  (service dominates)
    #   upper    = A + q0 = 1.4e11 + 2.9e10 = 1.69e11 bits
    #   tight    = A = 1.4e11 bits
    bracket = loss_bracket(
        offered_bits=2.0e8 * 700.0,
        downlink_rate_bps=1.0e9,
        outage_s=52.0,
        ramp_s=60.0,
        horizon_s=700.0,
        buffer_capacity_bits=3.2e10,
        initial_occupancy_bits=2.9e10,
    )
    assert bracket.lower_bits == 0.0
    assert bracket.upper_tight_bits == pytest.approx(1.4e11, rel=1e-14, abs=0.0)
    assert bracket.upper_bits == pytest.approx(1.69e11, rel=1e-14, abs=0.0)
    assert bracket.brackets(fluid_decomposition(worked_sequence).total_bits)


def test_lower_bound_is_tight_when_service_cannot_keep_up():
    # A short window with a long outage and a small buffer: the pigeonhole bound
    # binds. R_s = 1.0e9, H = 100 s -> A = 1.0e11 bits. T_o = 90 s, T_ramp = 10 s,
    # R_d = 1.0e9 -> S = 0.5*1e9*10 = 5.0e9 bits. B = 1.0e10, q0 = 0 ->
    # headroom 1.0e10. lower = 1.0e11 - 5.0e9 - 1.0e10 = 8.5e10 bits.
    bracket = loss_bracket(
        offered_bits=1.0e11,
        downlink_rate_bps=1.0e9,
        outage_s=90.0,
        ramp_s=10.0,
        horizon_s=100.0,
        buffer_capacity_bits=1.0e10,
        initial_occupancy_bits=0.0,
    )
    assert bracket.lower_bits == pytest.approx(8.5e10, rel=1e-14, abs=0.0)


def test_bracket_contains_the_fluid_total_over_random_handovers(rng):
    # The headline property: a bound from conservation and capacity counting
    # alone brackets the closed form's total. 500 random handovers, with the
    # numeric gaps asserted against the conditioning-scaled slack.
    outside = 0
    for _ in range(500):
        downlink = float(rng.uniform(2e8, 3e9))
        source = float(rng.uniform(0.02, 1.4)) * downlink
        capacity = float(rng.uniform(1e9, 8e10))
        occupancy = float(rng.uniform(0.0, 1.0)) * capacity
        acquisition = float(rng.uniform(0.0, 150.0))
        restart = float(rng.uniform(0.0, 90.0))
        ramp = float(rng.uniform(0.0, 200.0))
        horizon = acquisition + restart + ramp + float(rng.uniform(0.0, 400.0))
        sequence = HandoverSequence(
            source_rate_bps=source,
            downlink_rate_bps=downlink,
            buffer_capacity_bits=capacity,
            initial_occupancy_bits=occupancy,
            acquisition_s=acquisition,
            restart_s=restart,
            ramp_s=ramp,
            horizon_s=horizon,
        )
        total = fluid_decomposition(sequence).total_bits
        bracket = loss_bracket(
            offered_bits=source * horizon,
            downlink_rate_bps=downlink,
            outage_s=sequence.outage_s,
            ramp_s=ramp,
            horizon_s=horizon,
            buffer_capacity_bits=capacity,
            initial_occupancy_bits=occupancy,
        )
        if bracket.gap_below_bits(total) > bracket.slack_bits:
            outside += 1
        if bracket.gap_above_bits(total) > bracket.slack_bits:
            outside += 1
    assert outside == 0


def test_slack_follows_the_conditioning_not_a_constant():
    # Two brackets whose terms differ by a factor of 1e6 must have slacks
    # differing by the same factor. A constant slack would be vacuous for the
    # large one and falsely tight for the small one, which is exactly the defect
    # this mission recorded.
    small = bracket_slack_bits(1.0e6, 2.0e6, 1.0e6, 5.0e5)
    large = bracket_slack_bits(1.0e12, 2.0e12, 1.0e12, 5.0e11)
    assert float(large) / float(small) == pytest.approx(1.0e6, rel=1e-12, abs=0.0)
    assert float(small) == pytest.approx(
        8.0 * UNIT_ROUNDOFF * (1.0e6 + 2.0e6 + 1.0e6 + 5.0e5), rel=1e-14, abs=0.0
    )


def test_slack_is_vectorised():
    slack = bracket_slack_bits(
        np.array([1.0e9, 1.0e10]),
        np.array([2.0e9, 2.0e10]),
        np.array([1.0e9, 1.0e10]),
        np.array([0.0, 0.0]),
    )
    assert slack.shape == (2,)
    assert slack[1] == pytest.approx(10.0 * slack[0], rel=1e-12, abs=0.0)


def test_gap_helpers_sign_convention():
    bracket = loss_bracket(
        offered_bits=1.0e11,
        downlink_rate_bps=1.0e9,
        outage_s=90.0,
        ramp_s=10.0,
        horizon_s=100.0,
        buffer_capacity_bits=1.0e10,
        initial_occupancy_bits=0.0,
    )
    # Inside: both gaps negative.
    inside = 0.5 * (bracket.lower_bits + bracket.upper_tight_bits)
    assert bracket.gap_below_bits(inside) < 0.0
    assert bracket.gap_above_bits(inside) < 0.0
    # Below the lower bound: gap_below positive.
    assert bracket.gap_below_bits(0.0) > 0.0
    # Above the upper bound: gap_above positive.
    assert bracket.gap_above_bits(2.0 * bracket.upper_bits) > 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(offered_bits=-1.0), "non-negative"),
        (dict(buffer_capacity_bits=0.0), "strictly positive"),
        (dict(initial_occupancy_bits=-1.0), r"\[0, buffer_capacity_bits\]"),
        (dict(initial_occupancy_bits=1.0e12), r"\[0, buffer_capacity_bits\]"),
    ],
)
def test_bracket_rejects_out_of_range(kwargs, message):
    base = dict(
        offered_bits=1.0e11,
        downlink_rate_bps=1.0e9,
        outage_s=90.0,
        ramp_s=10.0,
        horizon_s=100.0,
        buffer_capacity_bits=1.0e10,
        initial_occupancy_bits=0.0,
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        loss_bracket(**base)


def test_max_service_rejects_short_window():
    with pytest.raises(ValueError, match="at least outage_s"):
        max_service_bits(1.0e9, 90.0, 10.0, 50.0)


def test_max_service_rejects_negative_inputs():
    with pytest.raises(ValueError, match="non-negative"):
        max_service_bits(-1.0e9, 90.0, 10.0, 200.0)
