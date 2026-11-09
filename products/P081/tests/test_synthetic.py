"""Synthetic cases: determinism, the four kinds, and the equivalence pair."""

from __future__ import annotations

import numpy as np
import pytest

from twindiverge.plant import NOMINAL, SCALE_DIRECTION, simulate
from twindiverge.synthetic import (
    KINDS,
    MAX_MAGNITUDE,
    MIN_MAGNITUDE,
    MULTI,
    UNIDENTIFIABLE,
    WEAK_MAX_MAGNITUDE,
    WEAK_MIN_MAGNITUDE,
    generate_cases,
    observe,
)

SEEDS = tuple(range(7000, 7030))


@pytest.mark.parametrize("kind", KINDS)
def test_cases_are_a_function_of_the_seed_list_alone(kind: str):
    first = generate_cases(SEEDS, kind)
    second = generate_cases(SEEDS, kind)
    assert len(first) == len(SEEDS)
    for left, right in zip(first, second, strict=True):
        assert left.seed == right.seed
        assert np.array_equal(left.delta_log, right.delta_log)


def test_single_cases_move_exactly_one_parameter_in_the_declared_range():
    for case in generate_cases(SEEDS, "single"):
        moved = np.flatnonzero(case.delta_log)
        assert moved.size == 1
        assert int(moved[0]) == case.label
        assert MIN_MAGNITUDE <= abs(float(case.delta_log[moved[0]])) <= MAX_MAGNITUDE


def test_weak_cases_lie_wholly_below_the_ordinary_range():
    for case in generate_cases(SEEDS, "weak"):
        moved = np.flatnonzero(case.delta_log)
        assert moved.size == 1
        magnitude = abs(float(case.delta_log[moved[0]]))
        assert WEAK_MIN_MAGNITUDE <= magnitude <= WEAK_MAX_MAGNITUDE
        assert magnitude <= MIN_MAGNITUDE


def test_scale_cases_lie_along_the_scale_direction():
    for case in generate_cases(SEEDS, "scale"):
        assert case.label == UNIDENTIFIABLE
        unit = case.delta_log / np.linalg.norm(case.delta_log)
        assert np.abs(unit) == pytest.approx(SCALE_DIRECTION, rel=1e-12, abs=0.0)


def test_none_cases_move_nothing():
    for case in generate_cases(SEEDS, "none"):
        assert case.label == UNIDENTIFIABLE
        assert case.delta_log == pytest.approx(np.zeros(5), rel=0.0, abs=0.0)
        assert case.magnitude == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_a_pair_differs_from_its_partner_by_exactly_a_scale_shift():
    singles = generate_cases(SEEDS, "single")
    partners = generate_cases(SEEDS, "equivalent")
    for single, partner in zip(singles, partners, strict=True):
        assert partner.label == MULTI
        assert partner.partner_seed == single.seed
        difference = single.delta_log - partner.delta_log
        assert difference == pytest.approx(
            np.full(5, float(difference[0])), rel=1e-13, abs=0.0
        )


def test_a_pair_produces_identical_noiseless_telemetry():
    """Equation (6) in its sharpest form: two different physical faults, one
    trajectory. Exact in exact arithmetic; the bound below is the measured
    rounding over thirty cases."""
    singles = generate_cases(SEEDS, "single")
    partners = generate_cases(SEEDS, "equivalent")
    worst = 0.0
    for single, partner in zip(singles, partners, strict=True):
        left = simulate(single.parameters, "short").rate_rad_s
        right = simulate(partner.parameters, "short").rate_rad_s
        worst = max(worst, float(np.abs(left - right).max() / np.abs(left).max()))
    assert worst < 1e-14, worst


def test_a_pair_shares_its_noise_draw(short_set):
    """Same seed, so the two observations are identical to the last bit and any
    classifier necessarily returns the same answer for both."""
    single = generate_cases((7001,), "single")[0]
    partner = generate_cases((7001,), "equivalent")[0]
    assert np.array_equal(observe(short_set, single), observe(short_set, partner))


def test_a_scale_case_and_a_none_case_are_indistinguishable(short_set):
    """The same statement from the other side: a pure scale change leaves the
    rate telemetry bit-identical to no change at all."""
    scale = generate_cases((7002,), "scale")[0]
    nothing = generate_cases((7002,), "none")[0]
    assert np.array_equal(observe(short_set, scale), observe(short_set, nothing))


def test_label_names_cover_every_label():
    for kind in KINDS:
        for case in generate_cases((7003,), kind):
            assert isinstance(case.label_name, str)
            assert case.label_name


def test_generate_cases_validates_its_arguments():
    with pytest.raises(ValueError, match="unknown case kind"):
        generate_cases(SEEDS, "whatever")
    with pytest.raises(ValueError, match="at least one seed"):
        generate_cases((), "single")
    with pytest.raises(ValueError, match="must not be negative"):
        generate_cases((-1,), "single")


def test_the_true_parameters_are_the_nominal_ones_moved():
    case = generate_cases((7004,), "single")[0]
    assert case.parameters.as_array() == pytest.approx(
        NOMINAL.as_array() * np.exp(case.delta_log), rel=1e-15, abs=0.0
    )
