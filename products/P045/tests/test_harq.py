"""HARQ: known answers, the three computation paths, and the crossover."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from arqlonghaul.channel import bpsk_ber
from arqlonghaul.harq import (
    HarqConfig,
    block_fer,
    crossover_rtt,
    harq_goodput,
    harq_goodput_exact,
    harq_simulate,
    optimal_first_rate,
    schedule_metrics,
    singleton_t,
)


def test_block_fer_hand_calculated() -> None:
    # t = 0: the frame fails unless every symbol is right.
    # n = 10, p = 0.1  =>  1 - 0.9**10 = 0.6513215599
    assert block_fer(10, 0, 0.1) == pytest.approx(1.0 - 0.9**10, rel=1e-12)
    # n = 3, t = 1, p = 0.5: 1 - [C(3,0) + C(3,1)] / 8 = 1 - 4/8 = 0.5
    assert block_fer(3, 1, 0.5) == pytest.approx(0.5, rel=1e-12)
    # A perfect channel never fails; a dead channel with t < n always does.
    assert block_fer(100, 5, 0.0) == pytest.approx(0.0)
    assert block_fer(100, 5, 1.0) == pytest.approx(1.0)
    # t = n cannot fail.
    assert block_fer(10, 10, 0.5) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "args", [(0, 1, 0.1), (10, -1, 0.1), (10, 11, 0.1), (10, 1, -0.1), (10, 1, 1.1)]
)
def test_block_fer_validation(args: tuple) -> None:
    with pytest.raises(ValueError):
        block_fer(*args)


def test_singleton_t_hand_calculated() -> None:
    # alpha = 1: t = floor((n-k)/2)
    assert singleton_t(300, 200, 1.0) == 50
    assert singleton_t(301, 200, 1.0) == 50
    assert singleton_t(200, 200, 1.0) == 0
    # alpha = 0.5: t = floor(0.5 * 100 / 2) = 25
    assert singleton_t(300, 200, 0.5) == 25


@pytest.mark.parametrize(
    "args", [(300, 0, 1.0), (100, 200, 1.0), (300, 200, 0.0), (300, 200, 1.5)]
)
def test_singleton_t_validation(args: tuple) -> None:
    with pytest.raises(ValueError):
        singleton_t(*args)


def test_config_derived_quantities() -> None:
    cfg = HarqConfig(k=200, n1=250, delta=40, max_rounds=3, scheme="type_ii")
    assert cfg.first_rate == pytest.approx(0.8)
    assert cfg.length_after(1) == 250
    assert cfg.length_after(3) == 330
    assert cfg.increment(1) == 250
    assert cfg.increment(2) == 40
    assert cfg.total_symbols() == 330
    assert cfg.t_after(1) == singleton_t(250, 200, 1.0)
    # Type-I repeats the same word and gains energy instead.
    c1 = HarqConfig(k=200, n1=250, max_rounds=3, esn0_db=2.0, scheme="type_i")
    assert c1.length_after(3) == 250
    assert c1.increment(3) == 250
    assert c1.symbol_ber(1) == pytest.approx(bpsk_ber(2.0))
    assert c1.symbol_ber(4) == pytest.approx(bpsk_ber(2.0 + 10 * math.log10(4)))
    assert c1.symbol_ber(4) < c1.symbol_ber(1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scheme": "type_iii"},
        {"k": 0},
        {"n1": 100},
        {"max_rounds": 0},
        {"rtt_symbols": -1.0},
        {"delta": -5},
        {"alpha": 0.0},
        {"alpha": 1.5},
    ],
)
def test_config_validation(kwargs: dict) -> None:
    base = {"k": 200, "n1": 250, "delta": 40}
    base.update(kwargs)
    with pytest.raises(ValueError):
        HarqConfig(**base)  # type: ignore[arg-type]


def test_config_round_index_validation() -> None:
    cfg = HarqConfig(k=200, n1=250, delta=40)
    for fn in (cfg.length_after, cfg.increment, cfg.t_after, cfg.symbol_ber):
        with pytest.raises(ValueError, match="m must be"):
            fn(0)


def test_single_round_goodput_is_closed_form_by_hand() -> None:
    # One round only: elapsed = n1 + D, success = 1 - P_F, goodput = k(1-P_F)/(n1+D)
    cfg = HarqConfig(
        k=200, n1=300, delta=0, max_rounds=1, esn0_db=1.0, rtt_symbols=100.0, alpha=0.5
    )
    pf = block_fer(300, singleton_t(300, 200, 0.5), bpsk_ber(1.0))
    res = harq_goodput_exact(cfg)
    assert res.residual_fer == pytest.approx(pf, rel=1e-9)
    assert res.elapsed_symbols == pytest.approx(400.0)
    assert res.goodput == pytest.approx(200.0 * (1.0 - pf) / 400.0, rel=1e-9)
    assert res.mean_rounds == pytest.approx(1.0)


def test_schedule_metrics_agrees_with_exact_dp_on_a_uniform_schedule() -> None:
    for n1 in (240, 300, 360):
        for snr in (0.0, 1.0, 2.0):
            cfg = HarqConfig(
                k=200, n1=n1, delta=40, max_rounds=4, esn0_db=snr,
                rtt_symbols=250.0, alpha=0.5,
            )
            e = harq_goodput_exact(cfg)
            s = schedule_metrics(200, [n1 - 200] + [40] * 3, bpsk_ber(snr), 250.0, 0.5)
            assert s["goodput"] == pytest.approx(e.goodput, rel=1e-12)
            assert s["residual_fer"] == pytest.approx(e.residual_fer, rel=1e-9)
            assert s["elapsed_symbols"] == pytest.approx(e.elapsed_symbols, rel=1e-12)


def test_independent_round_approximation_is_an_upper_bound_on_residual() -> None:
    for n1 in (240, 280, 340):
        cfg = HarqConfig(
            k=200, n1=n1, delta=40, max_rounds=4, esn0_db=0.5,
            rtt_symbols=200.0, alpha=0.5,
        )
        assert harq_goodput(cfg).residual_fer >= harq_goodput_exact(cfg).residual_fer


def test_type_i_exact_equals_closed_form_by_construction() -> None:
    cfg = HarqConfig(
        k=200, n1=300, max_rounds=4, esn0_db=0.0, rtt_symbols=200.0,
        alpha=0.5, scheme="type_i",
    )
    a = harq_goodput(cfg)
    b = harq_goodput_exact(cfg)
    assert b.goodput == pytest.approx(a.goodput, rel=1e-15)
    assert b.method.startswith("exact_type_i")


def test_monte_carlo_agrees_with_the_exact_dp() -> None:
    cfg = HarqConfig(
        k=200, n1=240, delta=40, max_rounds=4, esn0_db=0.0,
        rtt_symbols=200.0, alpha=0.5,
    )
    exact = harq_goodput_exact(cfg)
    mc = harq_simulate(cfg, 30_000, np.random.default_rng(7))
    se = math.sqrt(exact.residual_fer * (1 - exact.residual_fer) / 30_000)
    assert abs(mc.residual_fer - exact.residual_fer) < 5 * se
    assert mc.goodput == pytest.approx(exact.goodput, rel=0.02)


def test_harq_simulate_validation() -> None:
    cfg = HarqConfig(k=200, n1=250, delta=40)
    with pytest.raises(ValueError, match="n_frames"):
        harq_simulate(cfg, 0, np.random.default_rng(0))


def test_more_rounds_never_raise_the_residual() -> None:
    prev = 1.0
    for m in (1, 2, 3, 4, 6):
        cfg = HarqConfig(
            k=200, n1=240, delta=40, max_rounds=m, esn0_db=0.0,
            rtt_symbols=100.0, alpha=0.5,
        )
        res = harq_goodput_exact(cfg).residual_fer
        assert res <= prev + 1e-15
        prev = res


def test_crossover_moves_the_right_way_with_rtt() -> None:
    cx = crossover_rtt(
        k=200, esn0_db=1.0, delta=40, retransmit_rate=0.90,
        upfront_rate=0.50, max_rounds=4, alpha=0.5,
    )
    d = cx["d"]
    diff = cx["goodput_upfront"] - cx["goodput_retransmit"]
    assert math.isfinite(cx["crossover_d"])
    assert diff[0] < 0.0  # retransmission wins at small D
    assert diff[-1] > 0.0  # up-front redundancy wins at large D
    # the reported crossover lies inside the grid
    assert d[0] < cx["crossover_d"] < d[-1]


def test_crossover_validation() -> None:
    with pytest.raises(ValueError, match="must be below"):
        crossover_rtt(
            k=200, esn0_db=1.0, delta=40, retransmit_rate=0.5, upfront_rate=0.9
        )
    with pytest.raises(ValueError, match="retransmit_rate"):
        crossover_rtt(
            k=200, esn0_db=1.0, delta=40, retransmit_rate=1.5, upfront_rate=0.5
        )


def test_optimal_first_rate_falls_as_rtt_grows() -> None:
    rates = [
        optimal_first_rate(200, d, 1.0, 40, 4, 0.5)[0]
        for d in (1.0, 100.0, 1e4, 1e6)
    ]
    assert rates == sorted(rates, reverse=True)
    assert rates[0] > rates[-1]


def test_schedule_metrics_validation() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        schedule_metrics(200, [], 0.05, 100.0)
    with pytest.raises(ValueError, match=">= 0"):
        schedule_metrics(200, [-1, 40], 0.05, 100.0)
    with pytest.raises(ValueError, match="first increment"):
        schedule_metrics(200, [0, 40], 0.05, 100.0)
    with pytest.raises(ValueError, match="k must be"):
        schedule_metrics(0, [40], 0.05, 100.0)
    with pytest.raises(ValueError, match="p must be"):
        schedule_metrics(200, [40], 1.5, 100.0)
    with pytest.raises(ValueError, match="rtt_symbols"):
        schedule_metrics(200, [40], 0.05, -1.0)
    with pytest.raises(ValueError, match="drop_penalty"):
        schedule_metrics(200, [40], 0.05, 100.0, drop_penalty=-1.0)


def test_drop_penalty_enters_expected_cost_linearly() -> None:
    a = schedule_metrics(200, [40, 40], 0.1, 100.0, 0.5, drop_penalty=0.0)
    b = schedule_metrics(200, [40, 40], 0.1, 100.0, 0.5, drop_penalty=1000.0)
    assert b["expected_cost"] == pytest.approx(
        a["expected_cost"] + 1000.0 * a["residual_fer"], rel=1e-12
    )


@given(
    n=st.integers(min_value=4, max_value=400),
    frac=st.floats(min_value=0.0, max_value=1.0),
    p=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=250, deadline=None)
def test_block_fer_is_a_probability_and_monotone_in_p(
    n: int, frac: float, p: float
) -> None:
    t = int(frac * n)
    value = block_fer(n, t, p)
    assert 0.0 - 1e-12 <= value <= 1.0 + 1e-12
    if p < 1.0 and t < n:
        assert block_fer(n, t, min(1.0, p + 0.01)) >= value - 1e-12
    if t + 1 <= n:
        assert block_fer(n, t + 1, p) <= value + 1e-12
