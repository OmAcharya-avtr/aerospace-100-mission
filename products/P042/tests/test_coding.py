"""Tests for the Reed-Solomon bounded-distance accounting."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from acmpilot.coding import SHIPPED_CODES, ReedSolomonCode


class TestConstruction:
    @pytest.mark.parametrize("code", SHIPPED_CODES, ids=lambda c: c.label)
    def test_mds_properties(self, code):
        assert code.d_min == code.n - code.k + 1
        assert code.t == (code.n - code.k) // 2

    def test_known_answer_rs_255_223(self):
        # n - k = 32, so d_min = 33 and t = 16, both by hand from the MDS property.
        code = ReedSolomonCode(255, 223)
        assert code.d_min == 33
        assert code.t == 16
        assert code.rate == pytest.approx(223 / 255, rel=1e-15)
        assert code.label == "RS(255,223)"

    def test_known_answer_rs_7_3_over_gf8(self):
        # GF(8): n <= 7. n - k = 4, so d_min = 5 and t = 2.
        code = ReedSolomonCode(7, 3, bits_per_symbol=3)
        assert code.d_min == 5
        assert code.t == 2

    def test_shipped_codes_have_increasing_rate(self):
        rates = [c.rate for c in SHIPPED_CODES]
        assert rates == sorted(rates)

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            ((256, 200, 8), "n must be in"),
            ((0, 0, 8), "n must be in"),
            ((255, 255, 8), "k must satisfy"),
            ((255, 0, 8), "k must satisfy"),
            ((255, 300, 8), "k must satisfy"),
            ((255, 200, 0), "bits_per_symbol"),
        ],
    )
    def test_rejects_invalid_parameters(self, args, message):
        with pytest.raises(ValueError, match=message):
            ReedSolomonCode(*args)


class TestSymbolErrorRate:
    def test_known_answer_m_eight_p_point_one(self):
        # p_s = 1 - 0.9**8 = 1 - 0.43046721 = 0.56953279 by hand.
        code = ReedSolomonCode(255, 223)
        assert float(code.symbol_error_rate(0.1)) == pytest.approx(
            0.56953279, rel=1e-12
        )

    def test_zero_and_one_endpoints(self):
        code = ReedSolomonCode(255, 223)
        assert float(code.symbol_error_rate(0.0)) == 0.0
        assert float(code.symbol_error_rate(1.0)) == 1.0

    def test_known_answer_m_two(self):
        # m = 2: p_s = 1 - (1 - p_b)**2. At p_b = 0.5 this is 1 - 0.25 = 0.75.
        code = ReedSolomonCode(3, 1, bits_per_symbol=2)
        assert float(code.symbol_error_rate(0.5)) == pytest.approx(0.75, rel=1e-14)

    def test_n_must_fit_the_field(self):
        # An RS code over GF(2**m) has n <= 2**m - 1; GF(4) cannot carry n = 4.
        with pytest.raises(ValueError, match="n must be in"):
            ReedSolomonCode(4, 2, bits_per_symbol=2)

    def test_monotone(self):
        code = ReedSolomonCode(255, 223)
        values = code.symbol_error_rate(np.linspace(0.0, 1.0, 40))
        assert np.all(np.diff(values) > 0)

    @pytest.mark.parametrize("bad", [-0.01, 1.01])
    def test_rejects_out_of_range(self, bad):
        with pytest.raises(ValueError, match=r"p_b"):
            ReedSolomonCode(255, 223).symbol_error_rate(bad)


class TestPostDecodingRates:
    def test_known_answer_rs_3_1_over_gf4(self):
        # RS(3,1) over GF(4): n - k = 2 so t = 1, and m = 2 so
        # p_s = 1 - (1 - p_b)**2. Choose p_b = 1 - sqrt(0.5) = 0.29289321881345254,
        # which makes p_s exactly 0.5. Then, all by hand:
        #   FER     = P(S > 1) = 3 p_s^2 (1-p_s) + p_s^3
        #           = 3 * 0.25 * 0.5 + 0.125 = 0.375 + 0.125 = 0.5
        #   SER_out = E[S 1{S>1}]/3 = (2 * 0.375 + 3 * 0.125)/3 = 1.125/3 = 0.375
        #   BER_out = SER_out * p_b/p_s = 0.375 * 0.5857864376269049
        #           = 0.21966991410508934
        code = ReedSolomonCode(3, 1, bits_per_symbol=2)
        p_b = 1.0 - np.sqrt(0.5)
        assert code.t == 1
        assert float(code.symbol_error_rate(p_b)) == pytest.approx(0.5, rel=1e-12)
        assert float(code.frame_error_rate(p_b)) == pytest.approx(0.5, rel=1e-12)
        assert float(code.output_symbol_error_rate(p_b)) == pytest.approx(
            0.375, rel=1e-12
        )
        assert float(code.output_bit_error_rate(p_b)) == pytest.approx(
            0.21966991410508934, rel=1e-9
        )

    def test_known_answer_all_bits_wrong(self):
        # Every bit wrong: all 3 symbols in error, so FER = 1 and SER_out = 1.
        code = ReedSolomonCode(3, 1, bits_per_symbol=2)
        assert float(code.frame_error_rate(1.0)) == pytest.approx(1.0, rel=1e-12)
        assert float(code.output_symbol_error_rate(1.0)) == pytest.approx(1.0, rel=1e-12)

    def test_zero_channel_ber_gives_zero_everything(self):
        for code in SHIPPED_CODES:
            assert float(code.frame_error_rate(0.0)) == pytest.approx(0.0, abs=1e-300)
            assert float(code.output_symbol_error_rate(0.0)) == pytest.approx(
                0.0, abs=1e-300
            )
            assert float(code.output_bit_error_rate(0.0)) == 0.0

    def test_output_ber_never_exceeds_channel_ber(self):
        grid = np.linspace(1e-4, 0.5, 60)
        for code in SHIPPED_CODES:
            assert np.all(code.output_bit_error_rate(grid) <= grid + 1e-12)

    def test_monotone_in_channel_ber(self):
        grid = np.linspace(1e-3, 0.4, 80)
        for code in SHIPPED_CODES:
            for metric in (
                code.frame_error_rate, code.output_symbol_error_rate,
                code.output_bit_error_rate,
            ):
                values = np.asarray(metric(grid))
                assert np.all(np.diff(values) >= -1e-15)

    def test_lower_rate_code_is_stronger(self):
        p_b = 0.01
        values = [float(c.output_bit_error_rate(p_b)) for c in SHIPPED_CODES]
        assert values == sorted(values)

    def test_frame_error_rate_at_least_output_symbol_error_rate(self):
        grid = np.linspace(1e-3, 0.3, 40)
        for code in SHIPPED_CODES:
            assert np.all(
                np.asarray(code.frame_error_rate(grid))
                >= np.asarray(code.output_symbol_error_rate(grid)) - 1e-15
            )

    def test_scalar_input_gives_scalar_shape(self):
        code = ReedSolomonCode(255, 223)
        assert np.shape(code.output_symbol_error_rate(0.01)) == ()
        assert np.shape(code.output_bit_error_rate(0.01)) == ()

    def test_array_input_preserves_shape(self):
        code = ReedSolomonCode(255, 223)
        grid = np.linspace(0.001, 0.02, 7)
        assert code.output_symbol_error_rate(grid).shape == (7,)
        assert code.output_bit_error_rate(grid).shape == (7,)

    @given(p_b=st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=40, deadline=None)
    def test_property_rates_are_probabilities(self, p_b):
        code = ReedSolomonCode(255, 223)
        for metric in (
            code.frame_error_rate, code.output_symbol_error_rate,
            code.output_bit_error_rate,
        ):
            value = float(metric(p_b))
            assert 0.0 <= value <= 1.0 + 1e-12

    @given(
        n_k=st.integers(min_value=2, max_value=120),
        p_b=st.floats(min_value=1e-5, max_value=0.2),
    )
    @settings(max_examples=30, deadline=None)
    def test_property_more_redundancy_never_hurts(self, n_k, p_b):
        weak = ReedSolomonCode(255, 255 - n_k)
        strong = ReedSolomonCode(255, 255 - n_k - 1)
        assert float(strong.frame_error_rate(p_b)) <= float(
            weak.frame_error_rate(p_b)
        ) + 1e-15


class TestDirectMonteCarloAgreement:
    def test_formula_matches_direct_simulation_for_bpsk(self):
        """Known-answer style check against a simulated bit stream.

        BPSK carries one bit per modulation symbol, so channel bit errors really
        are i.i.d. and equations (2)-(5) of ``acmpilot.coding`` hold exactly. This
        is the integration-level check that the combinatorics are right; the
        validation script repeats it at more operating points.
        """
        from acmpilot.modulation import simulate_bit_errors

        code = ReedSolomonCode(255, 223)
        rng = np.random.default_rng(99)
        n_words = 4000
        bits = simulate_bit_errors("BPSK", 4.5, n_words * 255 * 8, rng=rng)
        grid = bits.reshape(n_words, 255, 8)
        counts = grid.any(axis=2).sum(axis=1)
        measured_fer = float((counts > code.t).mean())
        p_b = float(bits.mean())
        assert measured_fer == pytest.approx(float(code.frame_error_rate(p_b)), rel=0.08)
