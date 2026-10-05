"""Channel conventions, Eq. (1), and the P010 BERBench consistency anchor."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from framesync.channel import awgn_bpsk_samples, bpsk_ber, bsc_flip, ebn0_to_esn0_db, qfunc


def test_qfunc_known_answer():
    # Q(1) = 0.158655253931457... (Abramowitz & Stegun Table 26.1).
    assert float(qfunc(1.0)) == pytest.approx(0.1586552539, abs=1e-10)
    # Q(0) = 1/2 exactly.
    assert float(qfunc(0.0)) == pytest.approx(0.5, abs=1e-15)


def test_bpsk_ber_matches_scipy_independently():
    # Pb = Q(sqrt(2 gamma)); at 10 dB, gamma = 10 so sqrt(2*10) = sqrt(20).
    got = float(bpsk_ber(10.0)[0])
    assert got == pytest.approx(float(norm.sf(math.sqrt(20.0))), rel=1e-12)


def test_bpsk_ber_matches_p010_published_values():
    """Hand-check against P010 BERBench's own published validation output.

    P010 validation/bpsk_textbook_output.txt records
        Pb at Eb/N0 = 10 dB = 3.872108216e-06
    and P010 validation/VALIDATION.md records
        bpsk [awgn] at 8 dB -> BER analytic 1.9091e-04.
    Both are reproduced here from Eq. (1) with R = 1, which is the whole
    point of the convention: the two products must not disagree.
    """
    assert float(bpsk_ber(10.0)[0]) == pytest.approx(3.872108216e-06, rel=1e-9)
    assert float(bpsk_ber(8.0)[0]) == pytest.approx(1.9091e-04, rel=1e-4)


def test_rate_loss_is_a_pure_db_offset():
    # Es/N0 = R Eb/N0 -> esn0_db = ebn0_db + 10 log10 R.
    assert float(ebn0_to_esn0_db(10.0, 0.5)[0]) == pytest.approx(10.0 - 3.010299957, abs=1e-9)
    # and bpsk_ber(x, R) == bpsk_ber(x + 10log10 R, 1)
    r = 223 / 255
    left = float(bpsk_ber(7.0, r)[0])
    right = float(bpsk_ber(7.0 + 10 * math.log10(r), 1.0)[0])
    assert left == pytest.approx(right, rel=1e-12)


def test_ber_is_monotone_decreasing_in_ebn0():
    snr = np.arange(-2.0, 14.0, 0.5)
    ber = bpsk_ber(snr)
    assert np.all(np.diff(ber) < 0.0)
    assert np.all((ber > 0.0) & (ber <= 0.5))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_ebn0_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        bpsk_ber(bad)


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
def test_bad_code_rate_rejected(bad):
    with pytest.raises(ValueError, match="code_rate"):
        bpsk_ber(5.0, bad)


def test_bsc_flip_rate_matches_p():
    rng = np.random.default_rng(4242)
    bits = np.zeros(400_000, dtype=np.uint8)
    out = bsc_flip(bits, 0.01, rng)
    frac = out.mean()
    # binomial standard error = sqrt(0.01*0.99/4e5) = 1.57e-4; allow 5 sigma.
    assert abs(frac - 0.01) < 5 * math.sqrt(0.01 * 0.99 / bits.size)


def test_bsc_flip_endpoints_and_validation():
    rng = np.random.default_rng(1)
    bits = np.array([0, 1, 0, 1], dtype=np.uint8)
    assert np.array_equal(bsc_flip(bits, 0.0, rng), bits)
    assert np.array_equal(bsc_flip(bits, 1.0, rng), 1 - bits)
    with pytest.raises(ValueError, match="crossover"):
        bsc_flip(bits, 1.5, rng)
    with pytest.raises(ValueError, match="0/1 bits"):
        bsc_flip(np.array([0, 7], dtype=np.uint8), 0.1, rng)


def test_awgn_samples_hard_decision_rate_matches_eq1():
    rng = np.random.default_rng(99)
    n = 300_000
    bits = rng.integers(0, 2, n, dtype=np.uint8)
    y = awgn_bpsk_samples(bits, 4.0, 1.0, rng)
    hard = (y < 0).astype(np.uint8)
    measured = float(np.count_nonzero(hard != bits)) / n
    expected = float(bpsk_ber(4.0)[0])
    se = math.sqrt(expected * (1 - expected) / n)
    assert abs(measured - expected) < 5 * se
