"""The seeded scenario generator and the feature/target table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from relfiledelivery.dataset import (
    FEATURE_COLUMNS,
    PREDICTOR_MODES,
    REGIMES,
    TARGET_COLUMN,
    ScenarioRanges,
    feature_matrix,
    fit_split,
    generate_scenario,
    generate_table,
    scenario_from_row,
    targets,
)
from relfiledelivery.modes import TransferMode


class TestScenarioGeneration:
    @pytest.mark.parametrize("regime", REGIMES)
    def test_scenario_is_deterministic_in_seed_and_regime(self, regime: str) -> None:
        first = generate_scenario(123, regime)
        second = generate_scenario(123, regime)
        assert first.spec == second.spec
        assert first.link == second.link
        assert first.mode is second.mode

    def test_different_seeds_give_different_scenarios(self) -> None:
        specs = {generate_scenario(seed, "no_outage").spec for seed in range(20)}
        assert len(specs) > 1

    def test_no_outage_regime_is_the_closed_form_regime(self) -> None:
        for seed in range(15):
            scenario = generate_scenario(seed, "no_outage")
            assert scenario.link.outage.is_outage_free
            assert scenario.link.is_continuous
            assert scenario.link.availability == pytest.approx(1.0)

    def test_correlated_outage_regime_declares_outages_and_windows(self) -> None:
        for seed in range(15):
            scenario = generate_scenario(seed, "correlated_outage")
            assert not scenario.link.outage.is_outage_free
            assert len(scenario.link.windows) >= 4
            assert scenario.link.availability < 1.0

    def test_mode_is_drawn_from_the_predictor_modes(self) -> None:
        modes = {generate_scenario(seed, "no_outage").mode for seed in range(40)}
        assert modes.issubset(set(PREDICTOR_MODES))
        assert TransferMode.UNACKNOWLEDGED not in modes

    def test_sampled_unit_count_stays_inside_the_declared_range(self) -> None:
        counts = [generate_scenario(seed, "no_outage").spec.pdu_count for seed in range(60)]
        assert min(counts) >= 2
        assert max(counts) <= 1000

    def test_last_unit_is_usually_short(self) -> None:
        shorts = sum(
            1
            for seed in range(60)
            if generate_scenario(seed, "no_outage").spec.last_payload_bytes
            < generate_scenario(seed, "no_outage").spec.pdu_payload_bytes
        )
        assert shorts > 30

    def test_unknown_regime_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="regime must be one of"):
            generate_scenario(1, "mystery")

    def test_ranges_can_be_overridden(self) -> None:
        narrow = ScenarioRanges(
            log10_pdu_count=(1.0, 1.0), payload_choices=(256,), loss_probability=(0.0, 0.0)
        )
        scenario = generate_scenario(5, "no_outage", narrow)
        assert scenario.spec.pdu_count == 10
        assert scenario.spec.pdu_payload_bytes == 256
        assert scenario.link.pdu_loss_probability == pytest.approx(0.0)


class TestFeatures:
    def test_feature_keys_match_the_declared_column_order(self) -> None:
        features = generate_scenario(1, "correlated_outage").features()
        assert tuple(features) == FEATURE_COLUMNS

    def test_features_are_all_finite(self) -> None:
        for regime in REGIMES:
            for seed in range(10):
                values = list(generate_scenario(seed, regime).features().values())
                assert all(np.isfinite(value) for value in values)

    def test_mode_indicator_is_zero_or_one(self) -> None:
        for seed in range(20):
            scenario = generate_scenario(seed, "no_outage")
            indicator = scenario.features()["mode_is_immediate_nak"]
            assert indicator in (0.0, 1.0)
            assert (indicator == 1.0) == (scenario.mode is TransferMode.IMMEDIATE_NAK)

    def test_the_analytic_prediction_is_not_a_feature(self) -> None:
        """Feeding the closed form in as a feature would make the comparison circular."""
        banned = ("analytic", "expected_delivery")
        assert not any(token in name for name in FEATURE_COLUMNS for token in banned)


class TestTable:
    def test_table_has_one_row_per_scenario_per_regime(self, small_table: pd.DataFrame) -> None:
        assert len(small_table) == 120 * len(REGIMES)
        assert set(small_table["regime"]) == set(REGIMES)

    def test_table_carries_every_feature_column_and_the_target(
        self, small_table: pd.DataFrame
    ) -> None:
        for name in (*FEATURE_COLUMNS, TARGET_COLUMN, "completed", "seed", "regime"):
            assert name in small_table.columns

    def test_incomplete_rows_carry_a_missing_target(self, small_table: pd.DataFrame) -> None:
        incomplete = small_table[~small_table["completed"]]
        assert np.all(~np.isfinite(incomplete[TARGET_COLUMN]))

    def test_fit_split_keeps_only_completed_rows(self, small_table: pd.DataFrame) -> None:
        usable = fit_split(small_table)
        assert len(usable) == int(small_table["completed"].sum())
        assert np.all(np.isfinite(targets(usable)))

    def test_table_is_reproducible(self) -> None:
        first = generate_table(12, 31337)
        second = generate_table(12, 31337)
        pd.testing.assert_frame_equal(first, second)

    def test_the_correlated_regime_censors_some_runs(self) -> None:
        table = generate_table(120, 4242)
        by_regime = table.groupby("regime")["completed"].mean()
        assert by_regime["no_outage"] == pytest.approx(1.0)
        assert by_regime["correlated_outage"] < 1.0

    def test_the_two_regimes_have_different_delivery_time_scales(self) -> None:
        table = fit_split(generate_table(120, 909))
        q95 = table.groupby("regime")[TARGET_COLUMN].quantile(0.95)
        assert q95["correlated_outage"] > q95["no_outage"]

    def test_feature_matrix_shape_and_column_order(self, small_table: pd.DataFrame) -> None:
        matrix = feature_matrix(small_table)
        assert matrix.shape == (len(small_table), len(FEATURE_COLUMNS))
        assert matrix[:, 0] == pytest.approx(
            np.asarray(small_table["log10_pdu_count"], dtype=float)
        )

    def test_feature_matrix_names_the_missing_columns(self) -> None:
        with pytest.raises(ValueError, match="missing feature columns"):
            feature_matrix(pd.DataFrame({"log10_pdu_count": [1.0]}))

    def test_non_positive_row_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            generate_table(0, 1)

    def test_scenario_from_row_rebuilds_the_generator_output(
        self, small_table: pd.DataFrame
    ) -> None:
        row = small_table.iloc[7]
        rebuilt = scenario_from_row(row)
        assert rebuilt.spec.pdu_count == int(row["pdu_count"])
        assert rebuilt.link.forward_rate_bps == pytest.approx(float(row["forward_rate_bps"]))
        assert rebuilt.mode.value == row["mode"]
