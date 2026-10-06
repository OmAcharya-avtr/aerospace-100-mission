"""Tests for the tune-then-test benchmark protocol."""

from __future__ import annotations

import numpy as np
import pytest

from acmpilot.benchmark import (
    GATE_GRID,
    HYSTERESIS_GRID,
    MARGIN_GRID,
    BenchmarkConfig,
    BenchmarkSplit,
    benchmark_at_delay,
    fit_predictors,
    score_policy,
    tune_fixed_margin,
    tune_gate,
    tune_hysteresis,
)
from acmpilot.policy import ClairvoyantUpperBound, FixedMargin, ThresholdHysteresis

SMALL = BenchmarkConfig(
    n_slots=3000,
    train_slots=1500,
    n_lags=3,
    n_estimators=8,
    split=BenchmarkSplit(train_seeds=(101,), tune_seeds=(201, 202), test_seeds=(301, 302)),
)


class TestBenchmarkSplit:
    def test_default_sets_are_disjoint(self):
        split = BenchmarkSplit()
        assert not set(split.train_seeds) & set(split.tune_seeds)
        assert not set(split.train_seeds) & set(split.test_seeds)
        assert not set(split.tune_seeds) & set(split.test_seeds)

    def test_default_test_set_is_the_largest(self):
        split = BenchmarkSplit()
        assert len(split.test_seeds) >= len(split.tune_seeds)
        assert len(split.test_seeds) >= len(split.train_seeds)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"train_seeds": (1,), "tune_seeds": (1,)},
            {"tune_seeds": (5,), "test_seeds": (5,)},
            {"train_seeds": (7,), "test_seeds": (7,)},
        ],
    )
    def test_rejects_overlap(self, kwargs):
        base = {"train_seeds": (1,), "tune_seeds": (2,), "test_seeds": (3,)}
        base.update(kwargs)
        with pytest.raises(ValueError, match="disjoint"):
            BenchmarkSplit(**base)

    def test_rejects_empty_set(self):
        with pytest.raises(ValueError, match="non-empty"):
            BenchmarkSplit(train_seeds=(), tune_seeds=(2,), test_seeds=(3,))


class TestGrids:
    def test_margin_grid_includes_zero(self):
        assert 0.0 in MARGIN_GRID
        assert list(MARGIN_GRID) == sorted(MARGIN_GRID)

    def test_gate_grid_includes_the_ablation(self):
        assert 0.0 in GATE_GRID

    def test_hysteresis_grid_has_valid_pairs(self):
        for up, down in HYSTERESIS_GRID:
            assert up >= down >= 0.0


class TestScorePolicy:
    def test_fields_and_seed_count(self, table, config):
        row = score_policy(
            table, config, FixedMargin(2.0), tau_s=5e-3, seeds=(1, 2, 3), n_slots=3000
        )
        assert row["n_seeds"] == 3.0
        assert row["goodput_sem"] > 0.0
        assert "goodput_bit_per_symbol" in row
        assert "avoidable_outage_fraction" in row

    def test_single_seed_zero_sem(self, table, config):
        row = score_policy(
            table, config, FixedMargin(2.0), tau_s=0.0, seeds=(1,), n_slots=2000
        )
        assert row["goodput_sem"] == 0.0

    def test_reproducible(self, table, config):
        kwargs = {"tau_s": 5e-3, "seeds": (1, 2), "n_slots": 2000}
        a = score_policy(table, config, FixedMargin(2.0), **kwargs)
        b = score_policy(table, config, FixedMargin(2.0), **kwargs)
        assert a == b

    def test_clairvoyant_scores_highest(self, table, config):
        kwargs = {"tau_s": 10e-3, "seeds": (1, 2), "n_slots": 3000}
        bound = score_policy(table, config, ClairvoyantUpperBound(), **kwargs)
        causal = score_policy(table, config, ThresholdHysteresis(), **kwargs)
        assert (
            bound["goodput_bit_per_symbol"] >= causal["goodput_bit_per_symbol"] - 1e-12
        )


class TestTuning:
    def test_tune_fixed_margin_returns_the_argmax_of_its_trace(self, table, config):
        policy, trace = tune_fixed_margin(
            table, config, tau_s=10e-3, seeds=(201, 202), n_slots=3000
        )
        assert len(trace) == len(MARGIN_GRID)
        best = max(trace, key=lambda row: row[1])
        assert policy.margin_db == best[0]

    def test_tuned_margin_is_non_zero_at_non_zero_delay(self, table, config):
        policy, _ = tune_fixed_margin(
            table, config, tau_s=20e-3, seeds=(201, 202), n_slots=4000
        )
        assert policy.margin_db > 0.0

    def test_tuned_margin_is_zero_at_zero_delay(self, table, config):
        policy, _ = tune_fixed_margin(
            table, config, tau_s=0.0, seeds=(201, 202), n_slots=4000
        )
        assert policy.margin_db == 0.0

    def test_tune_hysteresis_returns_the_argmax_of_its_trace(self, table, config):
        policy, trace = tune_hysteresis(
            table, config, tau_s=10e-3, seeds=(201, 202), n_slots=3000
        )
        assert len(trace) == len(HYSTERESIS_GRID)
        best = max(trace, key=lambda row: row[2])
        assert (policy.up_margin_db, policy.down_margin_db) == (best[0], best[1])

    def test_tune_gate_returns_the_argmax_of_its_trace(self, table, config):
        analytic, learned, _, _ = fit_predictors(config, tau_s=10e-3, bench=SMALL)
        policy, trace = tune_gate(
            table, config, analytic, tau_s=10e-3, seeds=(201, 202), n_slots=3000,
            n_lags=SMALL.n_lags,
        )
        assert len(trace) == len(GATE_GRID)
        best = max(trace, key=lambda row: row[1])
        assert policy.gate_k == best[0]

    def test_tuned_gate_is_positive_at_non_zero_delay(self, table, config):
        analytic, _, _, _ = fit_predictors(config, tau_s=20e-3, bench=SMALL)
        policy, _ = tune_gate(
            table, config, analytic, tau_s=20e-3, seeds=(201, 202), n_slots=4000,
            n_lags=SMALL.n_lags,
        )
        assert policy.gate_k > 0.0


class TestFitPredictors:
    def test_returns_both_predictors_and_the_design_matrix(self, config):
        analytic, learned, features, target = fit_predictors(
            config, tau_s=5e-3, bench=SMALL
        )
        assert analytic.learned is False
        assert learned.learned is True
        assert features.ndim == 2
        assert features.shape[0] == target.size
        assert features.shape[1] == 2 * SMALL.n_lags - 1

    def test_training_uses_only_the_training_seeds(self, config):
        _, _, features, _ = fit_predictors(config, tau_s=5e-3, bench=SMALL)
        expected_rows = len(SMALL.split.train_seeds) * SMALL.train_slots - (
            config.delay_slots(5e-3) + SMALL.n_lags - 1
        )
        assert features.shape[0] == expected_rows


@pytest.fixture(scope="module")
def result(table, config):
    """One full benchmark at tau = 10 ms, built once for the whole module."""
    return benchmark_at_delay(table, config, tau_s=10e-3, bench=SMALL)


class TestBenchmarkAtDelay:
    def test_top_level_keys(self, result):
        for key in (
            "tau_s", "delay_slots", "n_train_rows", "tuned", "traces", "predictors",
            "scores",
        ):
            assert key in result

    def test_seven_scored_policies(self, result):
        assert len(result["scores"]) == 7

    def test_variants_present(self, result):
        variants = [row["variant"] for row in result["scores"]]
        assert variants.count("default") == 2
        assert variants.count("tuned") == 2
        assert variants.count("predictive") == 2
        assert variants.count("bound") == 1

    def test_exactly_one_learned_entry(self, result):
        assert sum(bool(row["learned"]) for row in result["scores"]) == 1

    def test_exactly_one_acausal_entry(self, result):
        assert sum(not row["causal"] for row in result["scores"]) == 1

    def test_bound_dominates_every_causal_entry(self, result):
        bound = next(r for r in result["scores"] if not r["causal"])
        for row in result["scores"]:
            if row["causal"]:
                assert (
                    float(row["goodput_bit_per_symbol"])
                    <= float(bound["goodput_bit_per_symbol"]) + 1e-12
                )

    def test_tuned_beats_or_matches_default(self, result):
        """Tuning on held-out tuning seeds should not make a baseline worse.

        It can, in principle, because the tuning seeds are not the test seeds;
        that is the point of the split. A large regression would mean the tuning
        grid or the split is wrong, so a tolerance is allowed and stated.
        """
        scores = {(r["variant"], str(r["policy"]).split(" ")[0]): r for r in result["scores"]}
        for family in ("fixed", "hysteresis"):
            default = next(
                r for r in result["scores"]
                if r["variant"] == "default" and str(r["policy"]).startswith(family)
            )
            tuned = next(
                r for r in result["scores"]
                if r["variant"] == "tuned" and str(r["policy"]).startswith(family)
            )
            assert float(tuned["goodput_bit_per_symbol"]) >= float(
                default["goodput_bit_per_symbol"]
            ) - 0.02
        assert scores  # the mapping is built and non-empty

    def test_tuned_hyperparameters_recorded(self, result):
        for key in (
            "fixed_margin_db", "hysteresis_up_db", "hysteresis_down_db",
            "analytic_gate_k", "learned_gate_k",
        ):
            assert key in result["tuned"]
            assert np.isfinite(float(result["tuned"][key]))

    def test_traces_cover_the_whole_grid(self, result):
        assert len(result["traces"]["fixed_margin"]) == len(MARGIN_GRID)
        assert len(result["traces"]["hysteresis"]) == len(HYSTERESIS_GRID)
        assert len(result["traces"]["learned_gate"]) == len(GATE_GRID)
        assert len(result["traces"]["analytic_gate"]) == len(GATE_GRID)

    def test_delay_slots_recorded(self, result, config):
        assert result["delay_slots"] == config.delay_slots(10e-3)

    def test_every_score_row_carries_a_standard_error(self, result):
        for row in result["scores"]:
            assert row["goodput_sem"] >= 0.0
            assert row["n_seeds"] == float(len(SMALL.split.test_seeds))
