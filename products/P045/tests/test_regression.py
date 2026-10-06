"""Regression and benchmark: pinned numbers that must not drift silently.

Every value below was produced by this repository's own code and is recorded
here so that a refactor that changes a result fails a test instead of quietly
changing a README.  Where a value is also hand-derivable it is derived in the
comment; where it is not, the script that produced it is named.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from arqlonghaul.channel import GilbertElliottChannel, IndependentFrameChannel, bpsk_ber
from arqlonghaul.closedform import gbn_throughput, sr_throughput, sw_throughput
from arqlonghaul.crc import CATALOGUE, crc
from arqlonghaul.harq import HarqConfig, crossover_rtt, harq_goodput_exact
from arqlonghaul.link import preset
from arqlonghaul.protocols import simulate_go_back_n, simulate_selective_repeat
from arqlonghaul.window import size_window

# Pinned CRC catalogue values; reference crcmod 1.7 predefined.py.
PINNED_CRC = {
    "crc-32": 0xCBF43926,
    "crc-32c": 0xE3069283,
    "crc-ccitt-false": 0x29B1,
    "crc-16": 0xBB3D,
    "crc-8": 0xF4,
}


def test_pinned_crc_check_values() -> None:
    for name, expected in PINNED_CRC.items():
        assert crc(b"123456789", CATALOGUE[name]) == expected


def test_pinned_closed_forms() -> None:
    # Hand arithmetic: N = 60, p = 0.05.
    # SW  = 0.95/60                       = 0.01583333...
    # GBN = 0.95/(0.95 + 60*0.05)         = 0.95/3.95 = 0.240506...
    # SR  = 0.95
    assert sw_throughput(0.05, 60) == pytest.approx(0.95 / 60.0, rel=1e-12)
    assert gbn_throughput(0.05, 60) == pytest.approx(0.95 / 3.95, rel=1e-12)
    assert sr_throughput(0.05, 60) == pytest.approx(0.95, rel=1e-12)


def test_pinned_link_presets() -> None:
    # validation/validate_window.py section 1.
    assert preset("geo").slots_per_cycle == pytest.approx(59.0131, abs=1e-3)
    assert preset("lunar").slots_per_cycle == pytest.approx(294.0987, abs=1e-3)
    assert preset("lunar").bdp_bytes == pytest.approx(326_805.0, abs=1.0)
    s = size_window(preset("lunar"), 0.05)
    assert s.knee_frames == 295
    assert s.gbn_goodput_at_knee == pytest.approx(0.0607, abs=5e-4)


def test_pinned_bpsk_ber() -> None:
    # 0.5*erfc(sqrt(10**(x/10)))
    assert bpsk_ber(0.0) == pytest.approx(0.0786496035, abs=1e-9)
    assert bpsk_ber(1.0) == pytest.approx(0.0562819519, abs=1e-9)
    assert bpsk_ber(2.5) == pytest.approx(0.0296552876, abs=1e-9)
    assert bpsk_ber(-1.5) == pytest.approx(0.1170404081, abs=1e-9)


def test_pinned_ge_construction() -> None:
    ch = GilbertElliottChannel.from_mean_and_burst(0.05, 25.0)
    assert ch.p_bg == pytest.approx(0.04)
    assert ch.p_gb == pytest.approx(0.04 * 0.05 / 0.95, rel=1e-12)
    assert ch.lam == pytest.approx(1.0 - ch.p_gb - ch.p_bg, rel=1e-12)
    assert ch.mean_good_slots == pytest.approx(475.0, rel=1e-9)


def test_pinned_harq_exact_dp() -> None:
    # validation/validate_harq.py section 3 and validation/worked_example.py.
    cfg = HarqConfig(
        k=200, n1=300, delta=40, max_rounds=4, esn0_db=1.0,
        rtt_symbols=500.0, alpha=0.5,
    )
    res = harq_goodput_exact(cfg)
    assert res.goodput == pytest.approx(0.2466, abs=5e-4)
    assert res.residual_fer == pytest.approx(2.804e-09, rel=1e-2)
    assert res.mean_rounds == pytest.approx(1.0205, abs=1e-3)


def test_pinned_crossover() -> None:
    # validation/validate_harq.py section 4.
    cx = crossover_rtt(
        k=200, esn0_db=1.0, delta=40, retransmit_rate=0.90,
        upfront_rate=0.50, max_rounds=4, alpha=0.5,
    )
    assert cx["crossover_d"] == pytest.approx(85.87, abs=0.5)


def test_pinned_burst_effect_on_go_back_n() -> None:
    """validation/validate_burst_goodput.py: bursts of 25 raise go-back-N."""
    rng = np.random.default_rng(45_045_2)
    n = 60
    iid = IndependentFrameChannel(0.05).errors(300_000, rng)
    ge = GilbertElliottChannel.from_mean_and_burst(0.05, 25.0).errors(300_000, rng)
    gain = simulate_go_back_n(ge, n, n).goodput / simulate_go_back_n(iid, n, n).goodput
    # The validation run measured +259 % at 800000 slots; allow a wide band
    # because this is a 300000-slot Monte Carlo with a different seed.
    assert 2.5 < gain < 4.5


def test_pinned_window_shortfall_at_the_textbook_window() -> None:
    """validation/validate_window.py section 4: W = N is not enough."""
    rng = np.random.default_rng(45_045_3)
    n = 60
    errors = IndependentFrameChannel(0.05).errors(300_000, rng)
    frac = simulate_selective_repeat(errors, n, n).goodput / 0.95
    assert 0.50 < frac < 0.62


def test_benchmark_simulator_throughput() -> None:
    """A performance regression guard, not a correctness check.

    The selective-repeat state machine must sustain at least 100000 slots per
    second on one core; the build machine managed roughly 400000. A tenfold
    slowdown would push the validation scripts past their stated runtimes.
    """
    errors = IndependentFrameChannel(0.05).errors(200_000, np.random.default_rng(0))
    start = time.perf_counter()
    simulate_selective_repeat(errors, 60, 600)
    elapsed = time.perf_counter() - start
    assert elapsed > 0.0
    assert 200_000 / elapsed > 100_000
