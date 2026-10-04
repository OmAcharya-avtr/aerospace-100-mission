"""Timing-budget composition tests, hand-computed."""

from __future__ import annotations

import math

import pytest

from rtclock.budget import Stage, compose_budget

# A four-stage 400 Hz loop (period 2.5 ms). WCETs in us:
#   sense 200, estimate 450, control 300, actuate 150  -> total 1100 us
# Means in us: 150, 380, 260, 120                      -> total  910 us
# Sigmas in us: 20, 60, 25, 10
FOUR_STAGE = [
    Stage("sense", 200e-6, 150e-6, 20e-6),
    Stage("estimate", 450e-6, 380e-6, 60e-6),
    Stage("control", 300e-6, 260e-6, 25e-6),
    Stage("actuate", 150e-6, 120e-6, 10e-6),
]


# Hand computation:
#   WCET total = (200 + 450 + 300 + 150) us = 1100 us
#   mean total = (150 + 380 + 260 + 120) us =  910 us
#   sigma independent = sqrt(20^2 + 60^2 + 25^2 + 10^2) us
#                     = sqrt(400 + 3600 + 625 + 100) = sqrt(4725)
#                     = 68.7386354243 us
#   sigma fully correlated = 20 + 60 + 25 + 10 = 115 us
#   margin = 2500 - 1100 = 1400 us ; utilization = 1100/2500 = 0.44
def test_four_stage_composition_hand_values():
    r = compose_budget(FOUR_STAGE, budget_s=2.5e-3)
    assert r.wcet_total_s == pytest.approx(1100e-6, rel=1e-12)
    assert r.mean_total_s == pytest.approx(910e-6, rel=1e-12)
    assert r.stdev_independent_s == pytest.approx(math.sqrt(4725) * 1e-6, rel=1e-12)
    assert r.stdev_independent_s == pytest.approx(68.7386354243e-6, rel=1e-10)
    assert r.stdev_fully_correlated_s == pytest.approx(115e-6, rel=1e-12)
    assert r.margin_s == pytest.approx(1400e-6, rel=1e-12)
    assert r.utilization == pytest.approx(0.44, rel=1e-12)
    assert r.fits_worst_case is True
    assert r.n_stages == 4


# Hand computation: the correlated bound is never below the independent one
#   (115 us >= 68.74 us), because sum(sigma) >= sqrt(sum(sigma^2)).
def test_correlated_bound_dominates_independent():
    r = compose_budget(FOUR_STAGE, budget_s=2.5e-3)
    assert r.stdev_fully_correlated_s >= r.stdev_independent_s


# Hand computation: estimate's share of the worst case = 450/1100 = 0.409090909
def test_stage_fractions_hand_value():
    r = compose_budget(FOUR_STAGE, budget_s=2.5e-3)
    assert r.stage_fractions["estimate"] == pytest.approx(450.0 / 1100.0, rel=1e-12)
    assert math.fsum(r.stage_fractions.values()) == pytest.approx(1.0, abs=1e-12)


# Hand computation: sigma headroom (independent) = (2500 - 910)/68.7386354243
#   = 1590/68.7386354243 = 23.1310...
def test_sigma_headroom_hand_value():
    r = compose_budget(FOUR_STAGE, budget_s=2.5e-3)
    assert r.sigma_headroom("independent") == pytest.approx(1590.0 / 68.7386354243, rel=1e-9)
    assert r.sigma_headroom("correlated") == pytest.approx(1590.0 / 115.0, rel=1e-12)
    with pytest.raises(ValueError, match="which must be"):
        r.sigma_headroom("gaussian")


def test_sigma_headroom_is_infinite_without_spread():
    r = compose_budget([Stage("only", 1e-3)], budget_s=2e-3)
    assert r.sigma_headroom() == math.inf


# Hand computation of the quantization term: q * sqrt(n+1)/sqrt(12) with
#   q = 1e-9 s and n = 4 stages (5 boundaries):
#   1e-9 * sqrt(5)/sqrt(12) = 1e-9 * 2.2360679775 / 3.4641016151
#                           = 6.4549722437e-10 s
def test_quantization_term_hand_value():
    r = compose_budget(FOUR_STAGE, budget_s=2.5e-3, clock_quantum_s=1e-9)
    assert r.quantization_uncertainty_s == pytest.approx(6.4549722437e-10, rel=1e-9)
    # It is tiny next to the stage spread, which is the point of reporting it
    # separately rather than folding it in.
    assert r.quantization_uncertainty_s < 1e-5 * r.stdev_independent_s


def test_zero_quantum_gives_zero_quantization_term():
    r = compose_budget(FOUR_STAGE, budget_s=2.5e-3, clock_quantum_s=0.0)
    assert r.quantization_uncertainty_s == 0.0


def test_overrunning_budget_is_reported_not_raised():
    r = compose_budget(FOUR_STAGE, budget_s=1e-3)
    assert r.fits_worst_case is False
    assert r.margin_s == pytest.approx(-100e-6, rel=1e-12)
    assert r.utilization > 1.0


def test_stage_defaults_mean_to_wcet_and_sigma_to_zero():
    s = Stage("x", 1e-3)
    assert s.mean_s == 1e-3
    assert s.stdev_s == 0.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "", "wcet_s": 1e-3}, "non-empty string"),
        ({"name": "a", "wcet_s": 0.0}, "wcet_s must be a finite value > 0"),
        ({"name": "a", "wcet_s": float("nan")}, "wcet_s must be a finite value > 0"),
        ({"name": "a", "wcet_s": 1e-3, "mean_s": 2e-3}, "must not exceed wcet_s"),
        ({"name": "a", "wcet_s": 1e-3, "mean_s": 0.0}, "mean_s must be a finite value > 0"),
        ({"name": "a", "wcet_s": 1e-3, "stdev_s": -1e-9}, "stdev_s must be a finite value >= 0"),
    ],
)
def test_stage_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Stage(**kwargs)


@pytest.mark.parametrize(
    ("stages", "budget", "quantum", "match"),
    [
        ([], 1e-3, 0.0, "must be non-empty"),
        ([Stage("a", 1e-4), Stage("a", 1e-4)], 1e-3, 0.0, "duplicated"),
        ([Stage("a", 1e-4)], 0.0, 0.0, "budget_s must be a finite value > 0"),
        ([Stage("a", 1e-4)], 1e-3, -1.0, "clock_quantum_s must be a finite value >= 0"),
    ],
)
def test_compose_budget_validation(stages, budget, quantum, match):
    with pytest.raises(ValueError, match=match):
        compose_budget(stages, budget_s=budget, clock_quantum_s=quantum)
