"""Tests for the constellations and the Monte Carlo BER measurement."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from acmpilot.modulation import (
    CONSTELLATION_NAMES,
    constellation,
    measure_ber,
    qfunc,
    simulate_bit_errors,
    theoretical_ber,
)


class TestConstellations:
    @pytest.mark.parametrize("name", CONSTELLATION_NAMES)
    def test_unit_average_energy(self, name):
        con = constellation(name)
        assert float(np.mean(np.abs(con.points) ** 2)) == pytest.approx(1.0, rel=1e-12)

    @pytest.mark.parametrize(
        ("name", "order", "bits"),
        [("BPSK", 2, 1), ("QPSK", 4, 2), ("8PSK", 8, 3), ("16QAM", 16, 4)],
    )
    def test_order_and_bits_per_symbol(self, name, order, bits):
        con = constellation(name)
        assert con.order == order
        assert con.bits_per_symbol == bits
        assert con.bit_map.shape == (order, bits)

    @pytest.mark.parametrize("name", CONSTELLATION_NAMES)
    def test_bit_labels_are_distinct(self, name):
        con = constellation(name)
        labels = {tuple(row) for row in con.bit_map}
        assert len(labels) == con.order

    @pytest.mark.parametrize("name", CONSTELLATION_NAMES)
    def test_points_are_distinct(self, name):
        con = constellation(name)
        assert np.unique(np.round(con.points, 9)).size == con.order

    def test_bpsk_known_answer(self):
        # Unit energy BPSK is exactly {-1, +1} with labels 0 and 1.
        con = constellation("BPSK")
        np.testing.assert_allclose(con.points, np.array([-1.0, 1.0]), atol=1e-12)
        np.testing.assert_array_equal(con.bit_map, np.array([[0], [1]], dtype=np.uint8))

    def test_qpsk_gray_adjacent_points_differ_in_one_bit(self):
        con = constellation("QPSK")
        order = np.argsort(np.angle(con.points))
        for i in range(4):
            a = con.bit_map[order[i]]
            b = con.bit_map[order[(i + 1) % 4]]
            assert int(np.count_nonzero(a != b)) == 1

    def test_8psk_gray_adjacent_points_differ_in_one_bit(self):
        con = constellation("8PSK")
        order = np.argsort(np.angle(con.points))
        for i in range(8):
            a = con.bit_map[order[i]]
            b = con.bit_map[order[(i + 1) % 8]]
            assert int(np.count_nonzero(a != b)) == 1

    def test_16qam_is_a_four_by_four_square(self):
        con = constellation("16QAM")
        reals = np.unique(np.round(con.points.real, 9))
        imags = np.unique(np.round(con.points.imag, 9))
        assert reals.size == 4
        assert imags.size == 4
        # Equally spaced levels.
        assert np.allclose(np.diff(reals), np.diff(reals)[0])

    def test_16qam_nearest_neighbours_differ_in_one_bit(self):
        con = constellation("16QAM")
        step = float(np.min(np.abs(np.diff(np.unique(np.round(con.points.real, 9))))))
        count = 0
        for i in range(16):
            for j in range(16):
                if i == j:
                    continue
                if abs(abs(con.points[i] - con.points[j]) - step) < 1e-9:
                    assert int(np.count_nonzero(con.bit_map[i] != con.bit_map[j])) == 1
                    count += 1
        assert count == 48  # 2 * 24 ordered adjacent pairs in a 4x4 grid

    def test_unknown_name_rejected(self):
        with pytest.raises(ValueError, match="unknown constellation"):
            constellation("64QAM")


class TestQFunction:
    def test_known_answers(self):
        # Q(0) = 0.5 exactly; Q(1) = 0.15865525393145705; Q(2) = 0.022750131948179195.
        assert float(qfunc(0.0)) == pytest.approx(0.5, rel=1e-14)
        assert float(qfunc(1.0)) == pytest.approx(0.15865525393145705, rel=1e-12)
        assert float(qfunc(2.0)) == pytest.approx(0.022750131948179195, rel=1e-12)

    def test_monotone_decreasing(self):
        x = np.linspace(-4, 4, 50)
        assert np.all(np.diff(qfunc(x)) < 0)


class TestTheoreticalBer:
    def test_bpsk_known_answer_at_0_db(self):
        # Es/N0 = 1, so BER = Q(sqrt(2)) = 0.5*erfc(1) = 0.07864960352514251.
        assert float(theoretical_ber("BPSK", 0.0)) == pytest.approx(
            0.07864960352514251, rel=1e-10
        )

    def test_qpsk_known_answer_at_0_db(self):
        # Es/N0 = 1, so BER = Q(1) = 0.15865525393145705.
        assert float(theoretical_ber("QPSK", 0.0)) == pytest.approx(
            0.15865525393145705, rel=1e-10
        )

    def test_qpsk_is_bpsk_shifted_by_three_db(self):
        # Gray QPSK has the same BER as BPSK at twice the symbol energy, so the
        # curve is the BPSK curve translated by 10*log10(2) = 3.0103 dB.
        snr = np.linspace(0.0, 12.0, 25)
        np.testing.assert_allclose(
            theoretical_ber("QPSK", snr + 10.0 * np.log10(2.0)),
            theoretical_ber("BPSK", snr),
            rtol=1e-10,
        )

    @pytest.mark.parametrize("name", CONSTELLATION_NAMES)
    def test_monotone_decreasing_in_snr(self, name):
        values = theoretical_ber(name, np.linspace(0.0, 20.0, 40))
        assert np.all(np.diff(values) < 0)

    @pytest.mark.parametrize("name", CONSTELLATION_NAMES)
    def test_bounded(self, name):
        values = theoretical_ber(name, np.linspace(-6.0, 26.0, 60))
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)

    def test_higher_order_is_worse_at_fixed_es_n0(self):
        snr = 12.0
        order = [float(theoretical_ber(n, snr)) for n in CONSTELLATION_NAMES]
        assert order == sorted(order)

    def test_no_closed_form_raises(self):
        with pytest.raises(ValueError, match="no closed form"):
            theoretical_ber("64QAM", 10.0)


class TestMeasureBer:
    @pytest.mark.parametrize(("name", "snr"), [("BPSK", 3.0), ("QPSK", 6.0)])
    def test_matches_exact_closed_form_within_four_sigma(self, name, snr):
        rng = np.random.default_rng(2)
        ber, n_err, n_bits = measure_ber(
            name, snr, rng=rng, target_errors=5000, max_symbols=600_000
        )
        form = float(theoretical_ber(name, snr))
        se = float(np.sqrt(form * (1.0 - form) / n_bits))
        assert n_err > 100
        assert abs(ber - form) < 4.0 * se

    @pytest.mark.parametrize("name", CONSTELLATION_NAMES)
    def test_rate_in_unit_interval(self, name, rng):
        ber, _, n_bits = measure_ber(name, 4.0, rng=rng, target_errors=200)
        assert 0.0 <= ber <= 1.0
        assert n_bits > 0

    def test_falls_with_snr(self, rng):
        values = [
            measure_ber("QPSK", snr, rng=rng, target_errors=1500)[0]
            for snr in (2.0, 5.0, 8.0)
        ]
        assert values[0] > values[1] > values[2]

    def test_reproducible_from_generator_state(self):
        a = measure_ber("QPSK", 5.0, rng=np.random.default_rng(8), target_errors=500)
        b = measure_ber("QPSK", 5.0, rng=np.random.default_rng(8), target_errors=500)
        assert a == b

    def test_stops_at_max_symbols(self, rng):
        _, _, n_bits = measure_ber(
            "BPSK", 30.0, rng=rng, target_errors=10**9, min_symbols=1000,
            max_symbols=5000,
        )
        assert n_bits == 5000

    def test_rejects_max_below_min(self, rng):
        with pytest.raises(ValueError, match="max_symbols"):
            measure_ber("BPSK", 5.0, rng=rng, min_symbols=1000, max_symbols=10)

    @given(snr=st.floats(min_value=-2.0, max_value=14.0))
    @settings(max_examples=12, deadline=None)
    def test_property_rate_is_a_probability(self, snr):
        ber, _, _ = measure_ber(
            "8PSK", snr, rng=np.random.default_rng(4), target_errors=120,
            min_symbols=2000, max_symbols=20_000,
        )
        assert 0.0 <= ber <= 1.0


class TestSimulateBitErrors:
    @pytest.mark.parametrize("name", CONSTELLATION_NAMES)
    def test_length_and_dtype(self, name, rng):
        errors = simulate_bit_errors(name, 6.0, 1000, rng=rng)
        assert errors.dtype == np.bool_
        assert errors.size == 1000 * constellation(name).bits_per_symbol

    def test_rate_agrees_with_measure_ber(self):
        rate_a = simulate_bit_errors(
            "QPSK", 5.0, 200_000, rng=np.random.default_rng(6)
        ).mean()
        rate_b, _, _ = measure_ber(
            "QPSK", 5.0, rng=np.random.default_rng(6), target_errors=10**9,
            min_symbols=200_000, max_symbols=200_000,
        )
        assert float(rate_a) == pytest.approx(rate_b, rel=1e-12)

    def test_no_errors_at_very_high_snr(self, rng):
        assert not simulate_bit_errors("BPSK", 40.0, 20_000, rng=rng).any()

    def test_rejects_zero_symbols(self, rng):
        with pytest.raises(ValueError, match="n_symbols"):
            simulate_bit_errors("BPSK", 5.0, 0, rng=rng)
