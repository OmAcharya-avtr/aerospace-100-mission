"""The measured frequency co-variation SPEC §6 requires.

SPEC §6 requires this measured rather than asserted. It comes out small, and a
small number here is a measurement with a stated limit, not a claim that the
clock did not matter: ``env.csv`` samples ``scaling_cur_freq`` six times per
cell as a point-in-time read, not as a time-average over the pass, so what is
bounded is what this sampling cadence can see.
"""

from __future__ import annotations

import numpy as np
import pytest

from hiltimingdiff.frequency import (
    PASS_SAMPLE_POINTS,
    frequency_covariation,
    pair_frequency_with_passes,
)

STAGES = ("ingest", "background", "threshold", "centroid", "gate", "track")


def test_pass_sample_points_are_the_campaign_cadence():
    assert PASS_SAMPLE_POINTS == {
        "pass0": 0,
        "pass10": 10,
        "pass20": 20,
        "pass30": 30,
        "pass40": 40,
        "pass50": 50,
    }


def test_pairing_yields_six_samples_per_cell(campaign, units):
    paired = pair_frequency_with_passes(campaign, units)
    assert len(paired) == 54
    assert set(paired.groupby("cell").size().unique()) == {6}


def test_pairing_carries_every_stage_column(campaign, units):
    paired = pair_frequency_with_passes(campaign, units)
    for stage in STAGES:
        assert stage in paired.columns


def test_pairing_carries_the_mean_frequency(campaign, units):
    paired = pair_frequency_with_passes(campaign, units)
    assert "freq_mean_khz" in paired.columns
    assert float(paired["freq_mean_khz"].min()) > 0.0


def test_mean_frequency_is_the_mean_of_the_six_cores(campaign, units):
    paired = pair_frequency_with_passes(campaign, units)
    columns = list(campaign.frequency_columns)
    row = paired.iloc[0]
    assert row["freq_mean_khz"] == pytest.approx(
        float(np.mean([row[name] for name in columns])), rel=1e-12, abs=0.0
    )


def test_governor_is_recorded_as_schedutil(covariation):
    assert covariation.governor == "schedutil"


def test_power_mode_is_recorded_as_maxn_super(covariation):
    assert "MAXN_SUPER" in covariation.power_mode


def test_passwordless_sudo_was_unavailable(covariation):
    assert covariation.sudo_passwordless is False


def test_frequency_span_matches_the_spec_measurement(covariation):
    # SPEC §6 recorded 729600 to 1190400 kHz at idle before the campaign. The
    # campaign itself saw the clock reach the full 1728000 kHz, which is a
    # wider span than the pre-campaign idle reading and is reported as measured.
    assert covariation.frequency_min_khz == 729600.0
    assert covariation.frequency_max_khz == 1728000.0


def test_frequency_relative_spread_is_substantial(covariation):
    assert covariation.frequency_relative_spread > 0.5


@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_has_a_covariation_entry(covariation, stage):
    assert stage in covariation.per_stage


@pytest.mark.parametrize("stage", STAGES)
def test_pooled_variance_share_is_a_fraction(covariation, stage):
    share = covariation.per_stage[stage].pooled_variance_share
    assert 0.0 <= share <= 1.0


@pytest.mark.parametrize("stage", STAGES)
def test_within_cell_variance_share_is_a_fraction(covariation, stage):
    share = covariation.per_stage[stage].within_cell_variance_share
    assert 0.0 <= share <= 1.0


@pytest.mark.parametrize("stage", STAGES)
def test_pooled_r_squared_is_the_squared_pooled_pearson(covariation, stage):
    item = covariation.per_stage[stage]
    assert item.pooled_variance_share == pytest.approx(
        item.pooled_pearson**2, rel=1e-9, abs=0.0
    )


@pytest.mark.parametrize("stage", STAGES)
def test_within_cell_r_squared_is_the_squared_within_cell_pearson(covariation, stage):
    item = covariation.per_stage[stage]
    assert item.within_cell_variance_share == pytest.approx(
        item.within_cell_pearson**2, rel=1e-9, abs=0.0
    )


@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_uses_all_fifty_four_paired_points(covariation, stage):
    assert covariation.per_stage[stage].n_points == 54


@pytest.mark.parametrize("stage", STAGES)
def test_measured_covariation_is_under_one_per_cent_of_variance(covariation, stage):
    # MEASURED, NOT ASSERTED. Less than one per cent of the per-stage variance
    # that this sampling can see moves with mean core frequency, pooled or
    # within-cell. The bound is on what six point-in-time reads per cell can
    # detect, not on what the unpinned clock did.
    item = covariation.per_stage[stage]
    assert item.pooled_variance_share < 0.01
    assert item.within_cell_variance_share < 0.01


def test_largest_within_cell_share_is_reported(covariation):
    assert covariation.max_within_cell_share == max(
        item.within_cell_variance_share for item in covariation.per_stage.values()
    )


def test_largest_within_cell_stage_is_named(covariation):
    assert covariation.max_within_cell_stage in STAGES


def test_largest_pooled_share_is_reported(covariation):
    assert covariation.max_pooled_share == max(
        item.pooled_variance_share for item in covariation.per_stage.values()
    )


def test_covariation_frame_is_in_pipeline_order(covariation):
    assert list(covariation.frame()["stage"]) == list(STAGES)


def test_frequency_columns_are_the_six_cores(covariation):
    assert len(covariation.frequency_columns) == 6


def test_pairing_refuses_a_campaign_whose_env_has_no_pass_points(campaign, units):
    class Shim:
        pass

    shim = Shim()
    shim.env = campaign.env.loc[campaign.env["point"].isin(("before", "after"))]
    shim.frequency_columns = campaign.frequency_columns
    with pytest.raises(ValueError, match="no env sample pairs"):
        pair_frequency_with_passes(shim, units)


def test_covariation_refuses_a_campaign_with_no_frequency_column(campaign, units):
    class Shim:
        pass

    shim = Shim()
    shim.env = campaign.env.drop(columns=list(campaign.frequency_columns))
    shim.frequency_columns = ()
    with pytest.raises(ValueError, match="no freq_cpu"):
        frequency_covariation(shim, units)
