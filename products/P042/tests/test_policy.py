"""Tests for the three non-learned policies."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from acmpilot.policy import (
    CLAIRVOYANT_LABEL,
    ClairvoyantUpperBound,
    FixedMargin,
    Policy,
    ThresholdHysteresis,
    baseline_policies,
)


class TestFixedMargin:
    def test_known_answer_selection(self, tiny_table):
        # Thresholds 3, 5, 7 dB; margin 3 dB. An observation of 10 dB gives a
        # prediction of 7 dB, which supports index 2. 8 dB gives 5 dB -> index 1.
        # 6 dB gives 3 dB -> index 0. 5 dB gives 2 dB -> nothing, floored at 0.
        policy = FixedMargin(margin_db=3.0)
        observed = np.array([10.0, 8.0, 6.0, 5.0])
        out = policy.select(tiny_table, observed, observed)
        np.testing.assert_array_equal(out, np.array([2, 1, 0, 0]))

    def test_zero_margin_matches_best_supported_floored(self, tiny_table):
        observed = np.array([0.0, 3.0, 5.0, 7.0, 20.0])
        out = FixedMargin(margin_db=0.0).select(tiny_table, observed, observed)
        np.testing.assert_array_equal(out, np.array([0, 0, 1, 2, 2]))

    def test_never_returns_out_of_range_index(self, tiny_table):
        observed = np.linspace(-50.0, 50.0, 200)
        out = FixedMargin(margin_db=2.0).select(tiny_table, observed, observed)
        assert out.min() >= 0
        assert out.max() <= tiny_table.n_modes - 1

    def test_larger_margin_never_selects_higher(self, tiny_table):
        observed = np.linspace(0.0, 20.0, 100)
        small = FixedMargin(margin_db=1.0).select(tiny_table, observed, observed)
        large = FixedMargin(margin_db=4.0).select(tiny_table, observed, observed)
        assert np.all(large <= small)

    def test_ignores_true_snr(self, tiny_table):
        observed = np.array([10.0, 10.0])
        a = FixedMargin().select(tiny_table, observed, np.array([0.0, 0.0]))
        b = FixedMargin().select(tiny_table, observed, np.array([99.0, 99.0]))
        np.testing.assert_array_equal(a, b)

    def test_is_causal_and_named(self):
        policy = FixedMargin(margin_db=2.5)
        assert policy.causal
        assert policy.name == "fixed margin 2.5 dB"
        assert isinstance(policy, Policy)

    def test_rejects_negative_margin(self):
        with pytest.raises(ValueError, match="margin_db"):
            FixedMargin(margin_db=-0.1)

    @given(
        margin=st.floats(min_value=0.0, max_value=12.0),
        snr=st.floats(min_value=-10.0, max_value=30.0),
    )
    @settings(
        max_examples=50, deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_monotone_in_observation(self, tiny_table, margin, snr):
        policy = FixedMargin(margin_db=margin)
        lower = policy.select(tiny_table, np.array([snr]), np.array([snr]))[0]
        upper = policy.select(tiny_table, np.array([snr + 2.0]), np.array([snr + 2.0]))[0]
        assert upper >= lower


class TestThresholdHysteresis:
    def test_known_answer_single_step_climb(self, tiny_table):
        # Thresholds 3, 5, 7; up margin 0, down margin 0. A constant 100 dB
        # observation starting from index 0 climbs exactly one step per slot:
        # 1, 2, 2, 2.
        policy = ThresholdHysteresis(up_margin_db=0.0, down_margin_db=0.0)
        observed = np.full(4, 100.0)
        np.testing.assert_array_equal(
            policy.select(tiny_table, observed, observed), np.array([1, 2, 2, 2])
        )

    def test_known_answer_dead_band_holds(self, tiny_table):
        # Starting at 0 with up margin 2 dB: upgrading to index 1 needs
        # x >= 5 + 2 = 7 dB. At 6.5 dB it holds at 0 forever.
        policy = ThresholdHysteresis(up_margin_db=2.0, down_margin_db=0.5)
        observed = np.full(5, 6.5)
        np.testing.assert_array_equal(
            policy.select(tiny_table, observed, observed), np.zeros(5, dtype=int)
        )

    def test_known_answer_downgrade(self, tiny_table):
        # Climb to index 2 on a high observation, then a 0 dB observation:
        # downgrading from k requires x < threshold[k] + down_margin. With
        # down margin 0.5 that is x < 7.5 at k=2 and x < 5.5 at k=1 and
        # x < 3.5 at k=0, so the index walks 2 -> 1 -> 0 and stays at the floor.
        policy = ThresholdHysteresis(up_margin_db=0.5, down_margin_db=0.5)
        observed = np.array([100.0, 100.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(
            policy.select(tiny_table, observed, observed), np.array([1, 2, 1, 0, 0])
        )

    def test_dead_band_property(self):
        policy = ThresholdHysteresis(up_margin_db=4.0, down_margin_db=1.0)
        assert policy.dead_band_db == pytest.approx(3.0)

    def test_switch_count_falls_as_dead_band_widens(self, tiny_table, rng):
        observed = 5.0 + 3.0 * rng.standard_normal(4000)
        counts = []
        for up in (0.5, 2.0, 5.0):
            out = ThresholdHysteresis(up_margin_db=up, down_margin_db=0.5).select(
                tiny_table, observed, observed
            )
            counts.append(int(np.count_nonzero(np.diff(out))))
        assert counts[0] > counts[1] > counts[2]

    def test_index_range(self, tiny_table, rng):
        observed = 5.0 + 8.0 * rng.standard_normal(2000)
        out = ThresholdHysteresis().select(tiny_table, observed, observed)
        assert out.min() >= 0
        assert out.max() <= tiny_table.n_modes - 1

    def test_index_changes_by_at_most_one_per_slot(self, tiny_table, rng):
        observed = 5.0 + 8.0 * rng.standard_normal(3000)
        out = ThresholdHysteresis().select(tiny_table, observed, observed)
        assert int(np.max(np.abs(np.diff(out)))) <= 1

    def test_ignores_true_snr(self, tiny_table):
        observed = np.full(6, 9.0)
        a = ThresholdHysteresis().select(tiny_table, observed, np.zeros(6))
        b = ThresholdHysteresis().select(tiny_table, observed, np.full(6, 99.0))
        np.testing.assert_array_equal(a, b)

    def test_is_causal_and_named(self):
        policy = ThresholdHysteresis(up_margin_db=2.0, down_margin_db=0.5)
        assert policy.causal
        assert policy.name == "hysteresis +2/-0.5 dB"

    def test_rejects_negative_dead_band(self):
        with pytest.raises(ValueError, match="dead band"):
            ThresholdHysteresis(up_margin_db=0.5, down_margin_db=2.0)

    def test_rejects_negative_down_margin(self):
        with pytest.raises(ValueError, match="down_margin_db"):
            ThresholdHysteresis(up_margin_db=1.0, down_margin_db=-1.0)


class TestClairvoyantUpperBound:
    def test_known_answer_uses_true_snr(self, tiny_table):
        policy = ClairvoyantUpperBound()
        true = np.array([0.0, 3.0, 5.0, 7.0])
        observed = np.full(4, -100.0)
        np.testing.assert_array_equal(
            policy.select(tiny_table, observed, true), np.array([0, 0, 1, 2])
        )

    def test_is_labelled_acausal(self):
        policy = ClairvoyantUpperBound()
        assert policy.causal is False
        assert policy.name == CLAIRVOYANT_LABEL
        assert "upper bound" in policy.name
        assert "no causal policy" in policy.name

    def test_label_constant_states_unachievability(self):
        assert "upper bound" in CLAIRVOYANT_LABEL
        assert "achieve" in CLAIRVOYANT_LABEL

    def test_ignores_observation(self, tiny_table):
        true = np.array([8.0, 8.0])
        a = ClairvoyantUpperBound().select(tiny_table, np.zeros(2), true)
        b = ClairvoyantUpperBound().select(tiny_table, np.full(2, 50.0), true)
        np.testing.assert_array_equal(a, b)

    def test_dominates_any_causal_choice(self, tiny_table, rng):
        true = 5.0 + 3.0 * rng.standard_normal(2000)
        observed = true + rng.standard_normal(2000)
        bound = ClairvoyantUpperBound().select(tiny_table, observed, true)
        eta = tiny_table.spectral_efficiencies
        for policy in (FixedMargin(1.0), ThresholdHysteresis()):
            chosen = policy.select(tiny_table, observed, true)
            best = np.atleast_1d(tiny_table.best_supported(true))
            delivered = eta[chosen] * (chosen <= best)
            bound_delivered = eta[bound] * (bound <= best)
            assert float(delivered.mean()) <= float(bound_delivered.mean()) + 1e-12


class TestBaselinePolicies:
    def test_returns_three_in_specified_order(self):
        policies = baseline_policies()
        assert len(policies) == 3
        assert isinstance(policies[0], FixedMargin)
        assert isinstance(policies[1], ThresholdHysteresis)
        assert isinstance(policies[2], ClairvoyantUpperBound)

    def test_exactly_one_is_acausal(self):
        assert sum(not p.causal for p in baseline_policies()) == 1

    def test_passes_through_parameters(self):
        fixed, hyst, _ = baseline_policies(
            margin_db=1.5, up_margin_db=3.0, down_margin_db=1.0
        )
        assert fixed.margin_db == 1.5
        assert hyst.up_margin_db == 3.0
        assert hyst.down_margin_db == 1.0

    def test_all_satisfy_the_protocol(self):
        for policy in baseline_policies():
            assert isinstance(policy, Policy)
            assert hasattr(policy, "name")
            assert hasattr(policy, "causal")
