"""Tests for the goodput, outage and two-kind mis-selection accounting."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from acmpilot.accounting import Accounting, account


class TestKnownAnswers:
    def test_hand_computed_four_slot_case(self, tiny_table):
        """Four slots, worked by hand.

        Thresholds 3, 5, 7 dB; efficiencies e = [1, 2, 3] * 253/255.

        slot  true  best  chosen  outcome
        0     8.0   2     2       exact,            delivers e[2]
        1     8.0   2     0       conservative,     delivers e[0], wastes e[2]-e[0]
        2     4.0   0     2       too aggressive,   delivers 0,    loses  e[0]
        3     1.0  -1     0       unavoidable,      delivers 0

        goodput             = (e[2] + e[0] + 0 + 0) / 4
        outage_fraction     = 2/4 = 0.5
        aggressive          = 2/4 = 0.5   (slots 2 and 3)
        avoidable outage    = 1/4 = 0.25  (slot 2 only)
        unavoidable         = 1/4 = 0.25  (slot 3)
        conservative        = 1/4 = 0.25  (slot 1)
        exact               = 1/4 = 0.25  (slot 0)
        wasted              = (e[2] - e[0]) / 4
        lost                = e[0] / 4
        clairvoyant goodput = (e[2] + e[2] + e[0] + 0) / 4
        switches over 3 gaps: 2->0, 0->2, 2->0 = 3, so switch rate 3/3 = 1.0
        """
        eta = tiny_table.spectral_efficiencies
        chosen = np.array([2, 0, 2, 0])
        true = np.array([8.0, 8.0, 4.0, 1.0])
        acc = account(tiny_table, chosen, true, warmup=0)
        assert acc.n_slots == 4
        assert acc.goodput_bit_per_symbol == pytest.approx((eta[2] + eta[0]) / 4)
        assert acc.outage_fraction == pytest.approx(0.5)
        assert acc.aggressive_fraction == pytest.approx(0.5)
        assert acc.avoidable_outage_fraction == pytest.approx(0.25)
        assert acc.unavoidable_outage_fraction == pytest.approx(0.25)
        assert acc.conservative_fraction == pytest.approx(0.25)
        assert acc.exact_fraction == pytest.approx(0.25)
        assert acc.wasted_bit_per_symbol == pytest.approx((eta[2] - eta[0]) / 4)
        assert acc.lost_bit_per_symbol == pytest.approx(eta[0] / 4)
        assert acc.clairvoyant_goodput_bit_per_symbol == pytest.approx(
            (2 * eta[2] + eta[0]) / 4
        )
        assert acc.switch_rate_per_slot == pytest.approx(1.0)
        assert acc.mean_index == pytest.approx(1.0)
        assert acc.mean_best_index == pytest.approx((2 + 2 + 0 - 1) / 4)

    def test_perfect_policy_has_unit_efficiency(self, tiny_table):
        true = np.array([8.0, 6.0, 4.0, 8.0])
        chosen = np.atleast_1d(tiny_table.best_supported(true))
        acc = account(tiny_table, np.maximum(chosen, 0), true, warmup=0)
        assert acc.efficiency == pytest.approx(1.0)
        assert acc.wasted_bit_per_symbol == pytest.approx(0.0)
        assert acc.lost_bit_per_symbol == pytest.approx(0.0)
        assert acc.conservative_fraction == 0.0

    def test_always_lowest_mode_never_too_aggressive_above_floor(self, tiny_table):
        true = np.array([8.0, 6.0, 4.0, 8.0])
        acc = account(tiny_table, np.zeros(4, dtype=int), true, warmup=0)
        assert acc.avoidable_outage_fraction == 0.0
        assert acc.conservative_fraction == pytest.approx(0.75)
        assert acc.exact_fraction == pytest.approx(0.25)

    def test_warmup_excludes_leading_slots(self, tiny_table):
        chosen = np.array([2, 0, 0, 0])
        true = np.array([1.0, 8.0, 8.0, 8.0])
        acc = account(tiny_table, chosen, true, warmup=1)
        assert acc.n_slots == 3
        assert acc.outage_fraction == 0.0
        assert acc.conservative_fraction == pytest.approx(1.0)


class TestInvariants:
    def test_three_categories_partition_the_slots(self, tiny_table, rng):
        true = 5.0 + 3.0 * rng.standard_normal(5000)
        chosen = rng.integers(0, tiny_table.n_modes, size=5000)
        acc = account(tiny_table, chosen, true, warmup=0)
        total = (
            acc.exact_fraction + acc.conservative_fraction + acc.aggressive_fraction
        )
        assert total == pytest.approx(1.0)

    def test_outage_equals_aggressive_on_monotone_ladder(self, tiny_table, rng):
        true = 5.0 + 3.0 * rng.standard_normal(3000)
        chosen = rng.integers(0, tiny_table.n_modes, size=3000)
        acc = account(tiny_table, chosen, true, warmup=0)
        assert acc.outage_fraction == pytest.approx(acc.aggressive_fraction)

    def test_avoidable_plus_unavoidable_equals_aggressive(self, tiny_table, rng):
        true = 4.0 + 3.0 * rng.standard_normal(3000)
        chosen = rng.integers(0, tiny_table.n_modes, size=3000)
        acc = account(tiny_table, chosen, true, warmup=0)
        assert (
            acc.avoidable_outage_fraction + acc.unavoidable_outage_fraction
            == pytest.approx(acc.aggressive_fraction)
        )

    def test_goodput_never_exceeds_clairvoyant(self, tiny_table, rng):
        true = 5.0 + 4.0 * rng.standard_normal(4000)
        for _ in range(5):
            chosen = rng.integers(0, tiny_table.n_modes, size=4000)
            acc = account(tiny_table, chosen, true, warmup=0)
            assert (
                acc.goodput_bit_per_symbol
                <= acc.clairvoyant_goodput_bit_per_symbol + 1e-12
            )
            assert 0.0 <= acc.efficiency <= 1.0 + 1e-12

    def test_goodput_plus_wasted_plus_lost_equals_clairvoyant(self, tiny_table, rng):
        true = 5.0 + 3.0 * rng.standard_normal(4000)
        chosen = rng.integers(0, tiny_table.n_modes, size=4000)
        acc = account(tiny_table, chosen, true, warmup=0)
        assert (
            acc.goodput_bit_per_symbol
            + acc.wasted_bit_per_symbol
            + acc.lost_bit_per_symbol
        ) == pytest.approx(acc.clairvoyant_goodput_bit_per_symbol)

    def test_switch_rate_in_unit_interval(self, tiny_table, rng):
        true = np.full(1000, 8.0)
        chosen = rng.integers(0, tiny_table.n_modes, size=1000)
        acc = account(tiny_table, chosen, true, warmup=0)
        assert 0.0 <= acc.switch_rate_per_slot <= 1.0

    def test_constant_index_has_zero_switch_rate(self, tiny_table):
        acc = account(tiny_table, np.ones(50, dtype=int), np.full(50, 8.0), warmup=0)
        assert acc.switch_rate_per_slot == 0.0

    @given(
        offset=st.floats(min_value=-4.0, max_value=12.0),
        index=st.integers(min_value=0, max_value=2),
    )
    @settings(
        max_examples=40, deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_fixed_index_metrics_consistent(self, tiny_table, offset, index):
        true = offset + np.linspace(-3.0, 3.0, 200)
        acc = account(tiny_table, np.full(200, index), true, warmup=0)
        assert 0.0 <= acc.outage_fraction <= 1.0
        assert acc.goodput_bit_per_symbol >= 0.0
        assert acc.wasted_bit_per_symbol >= -1e-12
        assert acc.lost_bit_per_symbol >= -1e-12


class TestValidation:
    def test_rejects_shape_mismatch(self, tiny_table):
        with pytest.raises(ValueError, match="shape"):
            account(tiny_table, np.zeros(3, dtype=int), np.zeros(4))

    def test_rejects_negative_warmup(self, tiny_table):
        with pytest.raises(ValueError, match="warmup"):
            account(tiny_table, np.zeros(4, dtype=int), np.zeros(4), warmup=-1)

    def test_rejects_warmup_consuming_everything(self, tiny_table):
        with pytest.raises(ValueError, match="warmup"):
            account(tiny_table, np.zeros(4, dtype=int), np.zeros(4), warmup=4)

    def test_rejects_out_of_range_index(self, tiny_table):
        with pytest.raises(ValueError, match="chosen indices"):
            account(tiny_table, np.array([0, 3]), np.zeros(2))

    def test_rejects_negative_index(self, tiny_table):
        with pytest.raises(ValueError, match="chosen indices"):
            account(tiny_table, np.array([0, -1]), np.zeros(2))


class TestAccountingDataclass:
    def test_as_dict_round_trip(self, tiny_table):
        acc = account(tiny_table, np.zeros(10, dtype=int), np.full(10, 8.0), warmup=0)
        payload = acc.as_dict()
        assert isinstance(payload, dict)
        assert Accounting(**payload) == acc

    def test_as_dict_has_every_documented_field(self, tiny_table):
        acc = account(tiny_table, np.zeros(10, dtype=int), np.full(10, 8.0), warmup=0)
        for key in (
            "n_slots", "goodput_bit_per_symbol",
            "clairvoyant_goodput_bit_per_symbol", "efficiency", "outage_fraction",
            "aggressive_fraction", "avoidable_outage_fraction",
            "unavoidable_outage_fraction", "conservative_fraction", "exact_fraction",
            "wasted_bit_per_symbol", "lost_bit_per_symbol", "switch_rate_per_slot",
            "mean_index", "mean_best_index",
        ):
            assert key in acc.as_dict()

    def test_efficiency_is_nan_when_channel_never_supports_anything(self, tiny_table):
        acc = account(tiny_table, np.zeros(10, dtype=int), np.full(10, -50.0), warmup=0)
        assert np.isnan(acc.efficiency)
        assert acc.unavoidable_outage_fraction == 1.0
