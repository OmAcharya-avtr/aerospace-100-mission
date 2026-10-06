"""Tests for the BPSK BER routines, sample and quadrature."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from aperturediv.ber import (
    MAX_HERMGAUSS_NODES,
    bpsk_ber_awgn,
    bpsk_ber_from_snr,
    bpsk_ber_lognormal_gauss_hermite,
    bpsk_ber_lognormal_sample,
    q_function,
)


class TestQFunction:
    def test_known_answers(self):
        assert float(q_function(0.0)) == pytest.approx(0.5, rel=1e-14)
        # Q(1) = 0.158655254, Q(2) = 0.022750132, Q(3) = 0.001349898.
        assert float(q_function(1.0)) == pytest.approx(0.15865525393, rel=1e-9)
        assert float(q_function(2.0)) == pytest.approx(0.02275013195, rel=1e-9)
        assert float(q_function(3.0)) == pytest.approx(0.0013498980316, rel=1e-9)

    def test_symmetry(self):
        for x in (0.3, 1.0, 2.5):
            assert float(q_function(-x)) == pytest.approx(1.0 - float(q_function(x)), rel=1e-12)

    def test_decreasing(self):
        x = np.linspace(-4.0, 6.0, 300)
        assert np.all(np.diff(q_function(x)) < 0.0)


class TestAwgnBer:
    def test_known_answer_zero_db(self):
        # Eb/N0 = 1: Q(sqrt 2) = Q(1.4142136) = 0.0786496035.
        assert float(bpsk_ber_awgn(0.0)) == pytest.approx(0.07864960352, rel=1e-9)

    def test_known_answer_ten_db(self):
        # Eb/N0 = 10: Q(sqrt 20) = Q(4.4721360) = 3.872108e-6.
        assert float(bpsk_ber_awgn(10.0)) == pytest.approx(3.8721082e-6, rel=1e-6)

    def test_half_at_minus_infinity_limit(self):
        assert float(bpsk_ber_awgn(-200.0)) == pytest.approx(0.5, abs=1e-6)

    def test_decreasing_in_ebn0(self):
        vals = bpsk_ber_awgn(np.arange(-5.0, 15.0, 0.5))
        assert np.all(np.diff(vals) < 0.0)

    def test_from_snr_rejects_negative(self):
        with pytest.raises(ValueError, match="snr_linear"):
            bpsk_ber_from_snr(-1.0)

    def test_from_snr_zero_is_one_half(self):
        assert float(bpsk_ber_from_snr(0.0)) == pytest.approx(0.5, rel=1e-14)


class TestGaussHermite:
    def test_converged_in_node_count(self):
        a = bpsk_ber_lognormal_gauss_hermite(10.0, 0.3, 120)
        b = bpsk_ber_lognormal_gauss_hermite(10.0, 0.3, 300)
        assert a == pytest.approx(b, rel=1e-9)

    def test_fading_is_worse_than_awgn_at_high_snr(self):
        # Deep fades dominate once the AWGN BER is small.
        assert bpsk_ber_lognormal_gauss_hermite(10.0, 0.3) > float(bpsk_ber_awgn(10.0))

    def test_increases_with_scintillation_index(self):
        vals = [bpsk_ber_lognormal_gauss_hermite(10.0, si) for si in (0.05, 0.1, 0.3, 0.6)]
        assert all(y > x for x, y in zip(vals, vals[1:], strict=False))

    def test_decreases_with_ebn0(self):
        vals = [bpsk_ber_lognormal_gauss_hermite(e, 0.3) for e in (0.0, 5.0, 10.0, 15.0)]
        assert all(y < x for x, y in zip(vals, vals[1:], strict=False))

    def test_rejects_too_few_nodes(self):
        with pytest.raises(ValueError, match="n_nodes must be >= 2"):
            bpsk_ber_lognormal_gauss_hermite(10.0, 0.3, 1)

    def test_rejects_node_count_above_guard(self):
        with pytest.raises(ValueError, match="hermgauss overflows"):
            bpsk_ber_lognormal_gauss_hermite(10.0, 0.3, MAX_HERMGAUSS_NODES + 1)

    def test_guard_value_is_usable(self):
        got = bpsk_ber_lognormal_gauss_hermite(10.0, 0.3, MAX_HERMGAUSS_NODES)
        assert np.isfinite(got)
        assert got > 0.0


class TestSampleBer:
    def test_matches_quadrature(self):
        res = bpsk_ber_lognormal_sample(10.0, 0.3, 2_000_000, 44044)
        want = bpsk_ber_lognormal_gauss_hermite(10.0, 0.3, 300)
        z = (res.ber - want) / res.binomial_se
        assert abs(z) < 4.0

    def test_reproducible_under_the_same_seed(self):
        a = bpsk_ber_lognormal_sample(8.0, 0.3, 200_000, 11)
        b = bpsk_ber_lognormal_sample(8.0, 0.3, 200_000, 11)
        assert a.n_errors == b.n_errors
        assert a.ber == b.ber

    def test_different_seed_gives_a_different_sample(self):
        a = bpsk_ber_lognormal_sample(8.0, 0.3, 200_000, 11)
        b = bpsk_ber_lognormal_sample(8.0, 0.3, 200_000, 12)
        assert a.n_errors != b.n_errors

    def test_chunking_is_part_of_the_configuration(self):
        # Changing the chunk size repartitions the single RNG stream, so the
        # exact sample changes. Documented, and asserted here so it cannot be
        # quietly changed and still called the same number.
        a = bpsk_ber_lognormal_sample(8.0, 0.3, 200_000, 11, chunk_size=200_000)
        b = bpsk_ber_lognormal_sample(8.0, 0.3, 200_000, 11, chunk_size=50_000)
        assert a.n_bits == b.n_bits
        assert a.n_errors != b.n_errors

    def test_binomial_se_formula(self):
        res = bpsk_ber_lognormal_sample(8.0, 0.3, 200_000, 11)
        want = np.sqrt(res.ber * (1.0 - res.ber) / res.n_bits)
        assert res.binomial_se == pytest.approx(want, rel=1e-14)

    def test_relative_se(self):
        res = bpsk_ber_lognormal_sample(8.0, 0.3, 200_000, 11)
        assert res.relative_se == pytest.approx(res.binomial_se / res.ber, rel=1e-14)

    def test_relative_se_infinite_with_no_errors(self):
        res = bpsk_ber_lognormal_sample(40.0, 0.01, 1000, 5)
        assert res.n_errors == 0
        assert res.relative_se == float("inf")

    def test_error_count_consistent_with_ber(self):
        res = bpsk_ber_lognormal_sample(8.0, 0.3, 100_000, 3)
        assert res.ber == pytest.approx(res.n_errors / res.n_bits, rel=1e-15)

    @given(st.integers(min_value=1, max_value=5000))
    def test_bit_count_is_exact(self, n):
        res = bpsk_ber_lognormal_sample(6.0, 0.3, n, 1, chunk_size=1000)
        assert res.n_bits == n

    @pytest.mark.parametrize("n,chunk", [(0, 100), (-5, 100), (100, 0), (100, -1)])
    def test_rejects_bad_sizes(self, n, chunk):
        with pytest.raises(ValueError):
            bpsk_ber_lognormal_sample(10.0, 0.3, n, 1, chunk_size=chunk)

    def test_rejects_bad_si(self):
        with pytest.raises(ValueError):
            bpsk_ber_lognormal_sample(10.0, 0.0, 100, 1)
