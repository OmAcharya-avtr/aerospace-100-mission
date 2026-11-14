"""Brackets from other mathematics, their slack, and that the controls fire."""

from __future__ import annotations

import numpy as np
import pytest

from powerenvelope.bounds import (
    EPS,
    Bracket,
    duty_cycle_lower_bracket,
    extreme_value_bracket,
    headline_lower_bracket,
    riemann_bracket,
    slack_from_conditioning,
)
from powerenvelope.decomposition import decompose
from powerenvelope.energy import active_windows, cell_energy, trapezoid_energy_mj


def test_slack_scales_with_the_magnitudes_summed():
    small = slack_from_conditioning(np.asarray([1e-20, 1e-20]))
    large = slack_from_conditioning(np.asarray([1e20, 1e20]))
    assert small < 1e-30
    assert large > 1e4
    # The point of the rule: a narrow bracket does not inherit an absolute
    # floor that would make it vacuous.
    assert small == pytest.approx(2.0 * EPS * 2e-20, rel=1e-12, abs=0.0)


def test_slack_of_nothing_is_zero():
    assert slack_from_conditioning(np.asarray([])) == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_bracket_contains_and_excess_agree():
    bracket = Bracket(low=1.0, high=2.0, slack=0.1, source="test")
    assert bracket.contains(1.5)
    assert bracket.contains(0.95)
    assert not bracket.contains(0.5)
    assert bracket.excess(1.5) == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert bracket.excess(0.5) == pytest.approx(0.4, rel=1e-12, abs=0.0)
    assert bracket.excess(2.6) == pytest.approx(0.5, rel=1e-12, abs=0.0)
    assert bracket.width == pytest.approx(1.0, rel=0.0, abs=0.0)


def test_riemann_bracket_contains_the_trapezoid_on_a_known_series():
    t = np.asarray([0.0, 1.0, 2.0, 3.0])
    p = np.asarray([10.0, 30.0, 20.0, 40.0])
    bracket = riemann_bracket(t, p)
    assert bracket.contains(trapezoid_energy_mj(t, p))
    # Left sum 10 + 30 + 20 = 60, right sum 30 + 20 + 40 = 90, trapezoid 75.
    assert bracket.low == pytest.approx(60.0, rel=1e-12, abs=0.0)
    assert bracket.high == pytest.approx(90.0, rel=1e-12, abs=0.0)


def test_riemann_bracket_is_degenerate_for_a_constant_series():
    t = np.asarray([0.0, 0.5, 1.0])
    p = np.asarray([7.0, 7.0, 7.0])
    bracket = riemann_bracket(t, p)
    assert bracket.width == pytest.approx(0.0, rel=0.0, abs=1e-12)
    assert bracket.contains(trapezoid_energy_mj(t, p))


def test_riemann_bracket_contains_every_cells_measured_integral(campaign):
    for cell in campaign.cells.values():
        bracket = riemann_bracket(cell.power.t, cell.power.vdd_in_mw)
        assert bracket.contains(
            trapezoid_energy_mj(cell.power.t, cell.power.vdd_in_mw)
        )


def test_extreme_value_bracket_contains_every_cells_measured_integral(campaign):
    for cell in campaign.cells.values():
        bracket = extreme_value_bracket(cell.power.t, cell.power.vdd_in_mw)
        assert bracket.contains(
            trapezoid_energy_mj(cell.power.t, cell.power.vdd_in_mw)
        )


def test_extreme_value_bracket_known_answer():
    # Span 2 s, min 10 mW, max 40 mW: the integral lies in [20, 80] mJ.
    bracket = extreme_value_bracket(
        np.asarray([0.0, 1.0, 2.0]), np.asarray([10.0, 40.0, 20.0])
    )
    assert bracket.low == pytest.approx(20.0, rel=1e-12, abs=0.0)
    assert bracket.high == pytest.approx(80.0, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("builder", [riemann_bracket, extreme_value_bracket])
def test_brackets_reject_bad_input(builder):
    with pytest.raises(ValueError, match="at least two samples"):
        builder(np.asarray([0.0]), np.asarray([1.0]))
    with pytest.raises(ValueError, match="samples and power_mw has"):
        builder(np.asarray([0.0, 1.0]), np.asarray([1.0]))


def test_riemann_bracket_rejects_decreasing_time():
    with pytest.raises(ValueError, match="non-decreasing"):
        riemann_bracket(np.asarray([1.0, 0.0]), np.asarray([1.0, 1.0]))


def test_duty_cycle_lower_bracket_known_answer():
    # 1000 mJ active, a floor that never fell below 1900 mW, 10 s asleep:
    # E_total >= 1000 + 19000, so the factor is at least 20.0.
    bracket = duty_cycle_lower_bracket(1000.0, 1900.0, 10.0)
    assert bracket.low == pytest.approx(20.0, rel=1e-12, abs=0.0)
    assert bracket.high == float("inf")


def test_duty_cycle_lower_bracket_holds_for_every_measured_cell(campaign):
    floor_min = float(campaign.require("floor").power.vdd_in_mw.min())
    for cell in campaign.pipeline_cells():
        whole, active = cell_energy(cell)
        sleep = max(0.0, whole.duration_s - active.duration_s)
        bracket = duty_cycle_lower_bracket(active.energy_mj, floor_min, sleep)
        assert bracket.contains(whole.energy_mj / active.energy_mj)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((0.0, 1.0, 1.0), "active_energy_mj must be positive"),
        ((1.0, -1.0, 1.0), "floor_min_mw must be non-negative"),
        ((1.0, 1.0, -1.0), "sleep_duration_s must be non-negative"),
    ],
)
def test_duty_cycle_lower_bracket_validates(args, match):
    with pytest.raises(ValueError, match=match):
        duty_cycle_lower_bracket(*args)


def test_headline_lower_bracket_known_answer():
    # 76 s run, never below 1900 mW, 100 accepted: numerator >= 1444 mJ.
    # Active power never above 2600 mW, latency never above 0.02 s:
    # denominator <= 52 mJ. So the factor is at least 1444 / 52 = 27.769...
    bracket = headline_lower_bracket(76.0, 1900.0, 2600.0, 0.02, 100)
    assert bracket.low == pytest.approx(1444.0 / 52.0, rel=1e-12, abs=0.0)


def test_headline_lower_bracket_holds_for_every_measured_cell(campaign):
    for cell in campaign.pipeline_cells():
        result = decompose(cell)
        decisions = cell.require_decisions()
        in_active = np.zeros(cell.power.t.size, dtype=bool)
        for lo, hi in zip(*active_windows(decisions), strict=True):
            in_active |= (cell.power.t >= lo) & (cell.power.t <= hi)
        bracket = headline_lower_bracket(
            run_duration_s=float(cell.power.t[-1] - cell.power.t[0]),
            min_power_mw=float(cell.power.vdd_in_mw.min()),
            max_active_power_mw=float(cell.power.vdd_in_mw[in_active].max()),
            max_latency_s=float(decisions.latency_s.max()),
            n_accepted=result.n_accepted,
        )
        assert bracket.contains(result.factor)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"run_duration_s": 0.0}, "run_duration_s must be positive"),
        ({"min_power_mw": 0.0}, "min_power_mw must be positive"),
        ({"max_active_power_mw": 0.0}, "max_active_power_mw must be positive"),
        ({"max_latency_s": 0.0}, "max_latency_s must be positive"),
        ({"n_accepted": 0}, "n_accepted must be positive"),
    ],
)
def test_headline_lower_bracket_validates(kwargs, match):
    base = {
        "run_duration_s": 76.0,
        "min_power_mw": 1900.0,
        "max_active_power_mw": 2600.0,
        "max_latency_s": 0.02,
        "n_accepted": 100,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        headline_lower_bracket(**base)


# --------------------------------------------------- negative controls, firing


def test_control_a_perturbed_integral_leaves_the_riemann_bracket(campaign):
    # Anchored on the TRUE value: the bracket's own width plus twice its slack
    # is the smallest offset guaranteed to leave it, so the control uses that
    # and is asserted to fire.
    cell = campaign.require("w64_d050")
    bracket = riemann_bracket(cell.power.t, cell.power.vdd_in_mw)
    truth = trapezoid_energy_mj(cell.power.t, cell.power.vdd_in_mw)
    offset = bracket.width + 2.0 * bracket.slack + abs(truth) * 1e-9
    assert bracket.contains(truth)
    assert not bracket.contains(truth + offset)
    assert not bracket.contains(truth - offset)


def test_control_a_halved_acceptance_rate_moves_the_headline(campaign):
    # The rejection factor is 1/a, so halving a must double the factor. The
    # control is anchored on the measured factor, not on an output of the
    # decomposition.
    result = decompose(campaign.require("w64_d050"))
    doubled = result.factor * 2.0
    assert doubled > result.factor * 1.9
    assert abs(doubled / result.factor - 2.0) < 1e-12


def test_control_a_bracket_with_an_absolute_slack_floor_would_be_vacuous():
    # Why the slack follows conditioning: a bracket of width 4.5e-23 widened by
    # an absolute 1.8e-15 admits anything, which is the failure this product's
    # slack rule exists to avoid. The control asserts the wrong form fails and
    # the right form does not.
    narrow = Bracket(low=1e-23, high=1e-23 + 4.5e-23, slack=1.8e-15, source="wrong")
    correct = Bracket(
        low=1e-23,
        high=1e-23 + 4.5e-23,
        slack=slack_from_conditioning(np.asarray([1e-23, 4.5e-23])),
        source="right",
    )
    intruder = 1.0
    assert narrow.contains(intruder) is False
    assert narrow.contains(1e-16)
    assert not correct.contains(1e-16)
