"""Frame error rate: Eq. (11), binomial statistics, and coding gain."""

from __future__ import annotations

import math

import numpy as np
import pytest

from framesync.channel import bpsk_ber
from framesync.fer import (
    binomial_stderr,
    coding_gain_db,
    ebn0_for_target,
    measure_conv_fer,
    measure_rs_fer,
    measure_uncoded_fer,
    n_frames_for_target,
    uncoded_fer,
)
from framesync.frames import FrameGeometry


def test_uncoded_fer_is_the_analytic_expression():
    """FER = 1 - (1-p)^n, checked against a direct power at 10 dB, n = 8936."""
    p = float(bpsk_ber(10.0)[0])
    direct = 1.0 - (1.0 - p) ** 8936
    assert float(uncoded_fer(10.0, 8936)[0]) == pytest.approx(direct, rel=1e-12)


def test_uncoded_fer_hand_values():
    # n = 1 reduces to the bit error rate itself.
    assert float(uncoded_fer(7.0, 1)[0]) == pytest.approx(float(bpsk_ber(7.0)[0]), rel=1e-14)
    # Small n*p: FER ~ n*p to first order.
    p = float(bpsk_ber(14.0)[0])
    assert float(uncoded_fer(14.0, 100)[0]) == pytest.approx(100 * p, rel=1e-3)


def test_uncoded_fer_monotone_and_bounded():
    snr = np.arange(0.0, 16.0, 0.5)
    fer = uncoded_fer(snr, 8936)
    assert np.all(np.diff(fer) <= 0.0)
    assert np.all((fer >= 0.0) & (fer <= 1.0))


def test_uncoded_fer_validation():
    with pytest.raises(ValueError, match="frame_bits must be positive"):
        uncoded_fer(5.0, 0)


def test_binomial_stderr_hand_values():
    # f = 0.25, N = 400 -> sqrt(0.25*0.75/400) = sqrt(4.6875e-4) = 0.0216506...
    assert binomial_stderr(100, 400) == pytest.approx(math.sqrt(0.25 * 0.75 / 400), rel=1e-14)
    assert binomial_stderr(0, 100) == 0.0
    assert binomial_stderr(100, 100) == 0.0
    with pytest.raises(ValueError, match="n_trials"):
        binomial_stderr(0, 0)
    with pytest.raises(ValueError, match="n_errors"):
        binomial_stderr(5, 4)


def test_n_frames_for_target_inverts_the_standard_error():
    n = n_frames_for_target(0.25, 0.05)
    # (1-f)/(f r^2) = 0.75 / (0.25 * 0.0025) = 1200
    assert n == 1200
    assert binomial_stderr(int(0.25 * n), n) / 0.25 == pytest.approx(0.05, rel=0.02)
    with pytest.raises(ValueError, match="fer_expected"):
        n_frames_for_target(0.0)
    with pytest.raises(ValueError, match="rel_stderr"):
        n_frames_for_target(0.1, 1.5)


def test_measured_uncoded_fer_agrees_with_eq11():
    """Short-sample consistency check; the sized sweep is in validation/."""
    geom = FrameGeometry(data_octets=120, fecf=True)
    rng = np.random.default_rng(606)
    pt = measure_uncoded_fer(8.0, geom, 600, rng)
    expected = float(uncoded_fer(8.0, geom.frame_bits)[0])
    se = max(pt.stderr, 1e-6)
    assert abs(pt.fer - expected) < 4 * se
    assert pt.n_frames == 600 and pt.link == "uncoded"
    assert pt.extra["n_undetected"] >= 0
    assert "FER=" in pt.as_row()


def test_measure_uncoded_requires_a_fecf():
    with pytest.raises(ValueError, match="FECF"):
        measure_uncoded_fer(
            8.0, FrameGeometry(data_octets=8, fecf=False), 2, np.random.default_rng(1)
        )


def test_zero_error_point_reports_a_rule_of_three_limit():
    geom = FrameGeometry(data_octets=16, fecf=True)
    rng = np.random.default_rng(1)
    pt = measure_uncoded_fer(20.0, geom, 40, rng)
    assert pt.n_frame_errors == 0
    assert pt.rule_of_three_upper == pytest.approx(3.0 / 40)
    assert "95% one-sided" in pt.as_row()


def test_measured_rs_fer_small_sample():
    rng = np.random.default_rng(31)
    pt = measure_rs_fer(5.0, 6, rng, interleave=1)
    assert pt.n_frames == 6 and 0.0 <= pt.fer <= 1.0
    assert pt.extra["code_rate"] == pytest.approx(223 / 255)
    assert pt.extra["analytic_fer"] >= 0.0


def test_measured_conv_fer_small_sample():
    rng = np.random.default_rng(32)
    pt = measure_conv_fer(4.0, 128, 20, rng, batch=20)
    assert pt.n_frames == 20 and 0.0 <= pt.fer <= 1.0
    assert pt.extra["code_rate"] == 0.5
    assert 0.0 <= pt.extra["decoded_ber"] <= 1.0


def test_ebn0_for_target_reproduces_the_bpsk_textbook_point():
    """BER = 1e-5 for uncoded BPSK at Eb/N0 = 9.5879 dB.

    P010 BERBench's validation output records the same inverted value,
    9.5879 dB, against the textbook-quoted "~9.6 dB". Reproduced here from
    an independent inversion of Eq. (1).
    """
    assert ebn0_for_target("uncoded_ber", 1e-5) == pytest.approx(9.5879, abs=5e-4)


def test_ebn0_for_target_round_trips():
    x = ebn0_for_target("uncoded_fer", 1e-3, frame_bits=8936)
    assert float(uncoded_fer(x, 8936)[0]) == pytest.approx(1e-3, rel=1e-5)


def test_ebn0_for_target_validation():
    with pytest.raises(ValueError, match="curve must be one of"):
        ebn0_for_target("nonsense", 1e-5)
    with pytest.raises(ValueError, match="target must lie"):
        ebn0_for_target("uncoded_ber", 0.0)
    with pytest.raises(ValueError, match="not bracketed"):
        ebn0_for_target("uncoded_ber", 1e-5, bracket=(0.0, 2.0))


def test_coding_gain_is_positive_and_below_the_asymptotic_bound():
    """Real coding gain must be positive and below 10 log10(R (E+1)).

    Asymptotic coding gain of a hard-decision bounded-distance decoder is
    G_inf = 10 log10(R (t+1)) dB [standard result, e.g. Lin & Costello 2004,
    "Error Control Coding" 2nd ed., coding-gain discussion]. For
    RS(255,223), R = 223/255 and t = 16 give G_inf = 11.7221 dB. Real gain at
    a finite error rate is strictly smaller.
    """
    g_inf = 10 * math.log10((223 / 255) * 17)
    assert g_inf == pytest.approx(11.7221, abs=1e-3)
    for metric in ("ber", "fer"):
        g = coding_gain_db(1e-5, metric=metric)
        assert 0.0 < g["gain_db"] < g_inf
        assert g["metric"] == metric


def test_coding_gain_validation():
    with pytest.raises(ValueError, match="metric must be"):
        coding_gain_db(1e-5, metric="nope")
