"""Tests for the episode runner and the delay sweep."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from acmpilot.policy import ClairvoyantUpperBound, FixedMargin, baseline_policies
from acmpilot.simulate import delayed_observation, run_episode, run_policy, sweep_delay


class TestDelayedObservation:
    def test_known_answer_shift(self):
        # delay 2 slots: the first two entries hold at x[0] and the rest are x
        # shifted right by two.
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(
            delayed_observation(x, 2), np.array([1.0, 1.0, 1.0, 2.0, 3.0])
        )

    def test_zero_delay_is_identity(self, rng):
        x = rng.standard_normal(50)
        np.testing.assert_array_equal(delayed_observation(x, 0), x)

    def test_zero_delay_returns_a_copy(self):
        x = np.array([1.0, 2.0])
        out = delayed_observation(x, 0)
        out[0] = 99.0
        assert x[0] == 1.0

    def test_preserves_length(self, rng):
        x = rng.standard_normal(100)
        for d in (0, 1, 7, 99):
            assert delayed_observation(x, d).size == 100

    def test_rejects_negative_delay(self):
        with pytest.raises(ValueError, match="delay_slots"):
            delayed_observation(np.zeros(5), -1)

    def test_rejects_delay_at_or_beyond_length(self):
        with pytest.raises(ValueError, match="delay_slots"):
            delayed_observation(np.zeros(5), 5)

    @given(delay=st.integers(min_value=0, max_value=20))
    @settings(max_examples=25, deadline=None)
    def test_property_tail_matches_shifted_head(self, delay):
        x = np.arange(60.0)
        out = delayed_observation(x, delay)
        np.testing.assert_array_equal(out[delay:], x[: 60 - delay])


class TestRunPolicy:
    def test_result_fields(self, table, snr_path):
        res = run_policy(table, snr_path, FixedMargin(2.0), delay_slots=5)
        assert res.policy_name == "fixed margin 2 dB"
        assert res.causal is True
        assert res.delay_slots == 5
        assert res.chosen.shape == snr_path.shape
        assert res.accounting.n_slots == snr_path.size - 50

    def test_default_warmup_is_at_least_the_delay(self, table, snr_path):
        res = run_policy(table, snr_path, FixedMargin(2.0), delay_slots=200)
        assert res.accounting.n_slots == snr_path.size - 200

    def test_explicit_warmup_respected(self, table, snr_path):
        res = run_policy(table, snr_path, FixedMargin(2.0), delay_slots=5, warmup=123)
        assert res.accounting.n_slots == snr_path.size - 123

    def test_clairvoyant_is_independent_of_delay(self, table, snr_path):
        values = [
            run_policy(
                table, snr_path, ClairvoyantUpperBound(), delay_slots=d, warmup=100
            ).accounting.goodput_bit_per_symbol
            for d in (0, 5, 50, 500)
        ]
        assert len(set(values)) == 1

    def test_clairvoyant_dominates_causal_policies(self, table, snr_path):
        bound = run_policy(
            table, snr_path, ClairvoyantUpperBound(), delay_slots=10, warmup=100
        ).accounting.goodput_bit_per_symbol
        for policy in baseline_policies()[:2]:
            value = run_policy(
                table, snr_path, policy, delay_slots=10, warmup=100
            ).accounting.goodput_bit_per_symbol
            assert value <= bound + 1e-12

    def test_goodput_falls_as_delay_grows(self, table, snr_path):
        values = [
            run_policy(
                table, snr_path, FixedMargin(2.0), delay_slots=d, warmup=200
            ).accounting.goodput_bit_per_symbol
            for d in (0, 2, 5, 20, 100)
        ]
        assert values[0] > values[-1]
        assert values[0] > values[2] > values[-1]

    def test_outage_rises_as_delay_grows(self, table, snr_path):
        values = [
            run_policy(
                table, snr_path, FixedMargin(2.0), delay_slots=d, warmup=200
            ).accounting.outage_fraction
            for d in (0, 5, 40)
        ]
        assert values[0] < values[1] < values[2]

    def test_rejects_policy_returning_wrong_shape(self, table, snr_path):
        class Broken:
            name = "broken"
            causal = True

            def select(self, table, observed_db, true_db):
                return np.zeros(3, dtype=int)

        with pytest.raises(ValueError, match="returned shape"):
            run_policy(table, snr_path, Broken(), delay_slots=1)


class TestRunEpisode:
    def test_all_policies_see_the_same_path(self, table, config):
        snr, results = run_episode(
            table, config, baseline_policies(), n_slots=3000, seed=4, tau_s=5e-3
        )
        assert snr.size == 3000
        assert len(results) == 3
        assert {r.delay_slots for r in results} == {5}

    def test_reproducible(self, table, config):
        a, _ = run_episode(
            table, config, baseline_policies(), n_slots=1000, seed=9, tau_s=2e-3
        )
        b, _ = run_episode(
            table, config, baseline_policies(), n_slots=1000, seed=9, tau_s=2e-3
        )
        np.testing.assert_array_equal(a, b)

    def test_exactly_one_result_is_acausal(self, table, config):
        _, results = run_episode(
            table, config, baseline_policies(), n_slots=1000, seed=3, tau_s=2e-3
        )
        assert sum(not r.causal for r in results) == 1


class TestSweepDelay:
    def test_row_count_and_keys(self, table, config):
        rows = sweep_delay(
            table, config, baseline_policies(), tau_list_s=(0.0, 5e-3),
            n_slots=3000, seeds=(1, 2),
        )
        assert len(rows) == 6
        for key in (
            "tau_s", "delay_slots", "policy", "causal", "n_seeds",
            "goodput_bit_per_symbol", "goodput_sem", "outage_fraction",
            "conservative_fraction", "avoidable_outage_fraction",
        ):
            assert key in rows[0]

    def test_single_seed_has_zero_sem(self, table, config):
        rows = sweep_delay(
            table, config, (FixedMargin(2.0),), tau_list_s=(0.0,), n_slots=2000,
            seeds=(1,),
        )
        assert rows[0]["goodput_sem"] == 0.0

    def test_sem_positive_with_several_seeds(self, table, config):
        rows = sweep_delay(
            table, config, (FixedMargin(2.0),), tau_list_s=(5e-3,), n_slots=2000,
            seeds=(1, 2, 3, 4),
        )
        assert rows[0]["goodput_sem"] > 0.0

    def test_goodput_decreasing_in_delay(self, table, config):
        rows = sweep_delay(
            table, config, (FixedMargin(2.0),), tau_list_s=(0.0, 10e-3, 40e-3),
            n_slots=4000, seeds=(1, 2),
        )
        values = [float(r["goodput_bit_per_symbol"]) for r in rows]
        assert values[0] > values[1] > values[2]

    def test_delay_slots_recorded(self, table, config):
        rows = sweep_delay(
            table, config, (FixedMargin(2.0),), tau_list_s=(0.0, 7e-3),
            n_slots=2000, seeds=(1,),
        )
        assert [r["delay_slots"] for r in rows] == [0.0, 7.0]
