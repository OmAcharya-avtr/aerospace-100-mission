"""Protocol state machines: agreement with the closed forms, and behaviour."""

from __future__ import annotations

import math

import numpy as np
import pytest

from arqlonghaul import closedform as cf
from arqlonghaul.channel import GilbertElliottChannel, IndependentFrameChannel
from arqlonghaul.protocols import (
    PROTOCOL_NAMES,
    simulate,
    simulate_go_back_n,
    simulate_selective_repeat,
    simulate_stop_and_wait,
)

SLOTS = 200_000


def _errors(p: float, seed: int = 0, slots: int = SLOTS) -> np.ndarray:
    return IndependentFrameChannel(p).errors(slots, np.random.default_rng(seed))


def test_error_free_channel_known_answers() -> None:
    clean = np.zeros(10_000, dtype=bool)
    assert simulate_stop_and_wait(clean, 10).goodput == pytest.approx(0.1, rel=1e-9)
    assert simulate_go_back_n(clean, 10, 10).goodput == pytest.approx(1.0, rel=1e-3)
    assert simulate_selective_repeat(clean, 10, 10).goodput == pytest.approx(
        1.0, rel=1e-3
    )


def test_all_errors_channel_delivers_nothing() -> None:
    dead = np.ones(10_000, dtype=bool)
    for name in PROTOCOL_NAMES:
        res = simulate(name, dead, 10, 10)
        assert res.delivered == 0
        assert res.goodput == 0.0
        assert math.isinf(res.mean_transmissions_per_frame)


@pytest.mark.parametrize("n", [2, 10, 50])
@pytest.mark.parametrize("p", [0.01, 0.1, 0.3])
def test_stop_and_wait_matches_closed_form(n: int, p: float) -> None:
    res = simulate_stop_and_wait(_errors(p, seed=n), n)
    ref = cf.sw_throughput(p, n)
    # Four batch-means standard errors.
    assert abs(res.goodput - ref) < 4.0 * res.goodput_stderr + 1e-6


@pytest.mark.parametrize("n", [5, 20])
@pytest.mark.parametrize("p", [0.01, 0.1, 0.3])
def test_go_back_n_matches_closed_form(n: int, p: float) -> None:
    res = simulate_go_back_n(_errors(p, seed=n + 1), n, n)
    ref = cf.gbn_throughput(p, n, n)
    assert abs(res.goodput - ref) < 4.0 * res.goodput_stderr + 1e-6


@pytest.mark.parametrize("n", [5, 20])
@pytest.mark.parametrize("p", [0.01, 0.1, 0.3])
def test_selective_repeat_large_window_matches_closed_form(n: int, p: float) -> None:
    res = simulate_selective_repeat(_errors(p, seed=n + 2), n, 60 * n)
    ref = cf.sr_throughput(p, n)
    assert abs(res.goodput - ref) < 4.0 * res.goodput_stderr + 1e-6


@pytest.mark.parametrize("n", [5, 20, 50])
def test_selective_repeat_window_one_is_stop_and_wait(n: int) -> None:
    errors = _errors(0.1, seed=n + 3)
    sr = simulate_selective_repeat(errors, n, 1)
    ref = cf.sw_throughput(0.1, n)
    assert abs(sr.goodput - ref) < 4.0 * sr.goodput_stderr + 1e-6


def test_closed_form_is_an_upper_bound_for_window_limited_sr() -> None:
    # Documented finding: (1-p) W/N assumes a window that never blocks, so it
    # over-predicts for 1 < W < N. The simulator must come out below it.
    errors = _errors(0.1, seed=11)
    n = 60
    for w in (10, 20, 30):
        sim = simulate_selective_repeat(errors, n, w).goodput
        assert sim < cf.sr_throughput(0.1, n, w)


def test_correlation_helps_go_back_n() -> None:
    # Documented finding: at a fixed marginal rate, bursts raise go-back-N
    # goodput, because one burst costs one go-back.
    rng = np.random.default_rng(17)
    iid = IndependentFrameChannel(0.05).errors(SLOTS, rng)
    ge = GilbertElliottChannel.from_mean_and_burst(0.05, 25.0).errors(SLOTS, rng)
    n = 40
    assert (
        simulate_go_back_n(ge, n, n).goodput
        > 1.5 * simulate_go_back_n(iid, n, n).goodput
    )


def test_correlation_does_not_move_stop_and_wait() -> None:
    rng = np.random.default_rng(19)
    iid = IndependentFrameChannel(0.05).errors(SLOTS, rng)
    ge = GilbertElliottChannel.from_mean_and_burst(0.05, 25.0).errors(SLOTS, rng)
    n = 40
    a = simulate_stop_and_wait(iid, n).goodput
    b = simulate_stop_and_wait(ge, n).goodput
    assert abs(a - b) / a < 0.05


def test_results_are_deterministic_for_a_given_realisation() -> None:
    errors = _errors(0.1, seed=23, slots=50_000)
    for name in PROTOCOL_NAMES:
        a = simulate(name, errors, 10, 10)
        b = simulate(name, errors, 10, 10)
        assert a.goodput == b.goodput
        assert a.delivered == b.delivered
        assert a.transmissions == b.transmissions


def test_accounting_is_consistent() -> None:
    errors = _errors(0.15, seed=29, slots=50_000)
    for name in PROTOCOL_NAMES:
        res = simulate(name, errors, 12, 12)
        assert res.transmissions + res.idle_slots <= res.slots
        assert res.delivered <= res.transmissions
        assert 0.0 <= res.utilisation <= 1.0
        assert res.errored_transmissions <= res.transmissions
        d = res.as_dict()
        assert d["protocol"] == name


def test_goodput_bps_scaling() -> None:
    errors = _errors(0.0, seed=31, slots=20_000)
    res = simulate_selective_repeat(errors, 10, 400)
    # goodput ~ 1, 8792 payload bits in 4.46 ms  =>  about 1.97 Mbit/s
    bps = res.goodput_bps(8792, 4.46e-3)
    assert bps == pytest.approx(res.goodput * 8792 / 4.46e-3, rel=1e-12)
    with pytest.raises(ValueError):
        res.goodput_bps(0, 1e-3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 0},
        {"n": 1.5},
        {"window": 0},
        {"window": -4},
        {"warmup": -1},
    ],
)
def test_invalid_arguments_raise(kwargs: dict) -> None:
    errors = _errors(0.1, seed=37, slots=5000)
    args = {"n": 10, "window": 10, "warmup": None}
    args.update(kwargs)
    with pytest.raises(ValueError):
        simulate_selective_repeat(
            errors, args["n"], args["window"], args["warmup"]
        )


def test_bad_error_array_raises() -> None:
    with pytest.raises(ValueError, match="1-D"):
        simulate_stop_and_wait(np.zeros((4, 4), dtype=bool), 2)
    with pytest.raises(ValueError, match="at least 2 slots"):
        simulate_stop_and_wait(np.zeros(1, dtype=bool), 2)


def test_unknown_protocol_raises() -> None:
    with pytest.raises(ValueError, match="unknown protocol"):
        simulate("alternating_bit", _errors(0.1, slots=5000), 10, 10)


def test_stderr_is_nan_without_enough_batches() -> None:
    errors = _errors(0.1, seed=41, slots=5000)
    res = simulate_stop_and_wait(errors, 10, warmup=0, n_batches=1)
    assert math.isnan(res.goodput_stderr)
