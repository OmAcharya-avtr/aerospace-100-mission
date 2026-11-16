"""End-to-end: load, measure both modes, and reach every headline output.

One test here runs the whole product the way a reader will: ``build_headline``
with a one-pass simulation measurement at the declared stack length. The rest
assert cross-module consistency -- that the same number computed two ways in
two modules agrees.
"""

from __future__ import annotations

import numpy as np
import pytest

from hiltimingdiff import build_headline, headline_lines

STAGES = ("ingest", "background", "threshold", "centroid", "gate", "track")


@pytest.fixture(scope="module")
def headline():
    return build_headline(passes=1, resamples=200)


def test_headline_loads_the_committed_campaign(headline):
    assert headline.campaign.label == "data/traces"


def test_headline_measures_simulation_mode_at_the_declared_stack(headline):
    assert headline.simulation.stack_matches_campaign is True


def test_headline_covers_the_nine_cells_in_both_modes(headline):
    assert len(headline.campaign.runs) == 9
    assert headline.simulation.table["cell"].nunique() == 9


def test_headline_counts_the_hardware_frames(headline):
    assert headline.n_hardware_frames == 810000


def test_headline_ns_per_op_is_positive(headline):
    assert headline.ns_per_op > 0.0


def test_headline_source_md5_is_the_coordinator_table(headline):
    assert headline.source_md5["pat_pipeline.py"] == "80ba9015b54fa8e95eae96b2c93eb9c2"
    assert headline.source_md5["staged_exec.py"] == "0f654894ea7dc3c6ec5d3880e9150459"
    assert headline.source_md5["campaign.py"] == "8c137fc2c31c48c2e8f16d1ae1f206d3"
    assert headline.source_md5["verify_outputs.py"] == "ac6b6aa2b486ea7d9a227d1598dc8293"


def test_headline_spread_verdict_is_confirmed(headline):
    assert headline.spread_verdict.held is True


def test_headline_transfer_verdict_is_refuted(headline):
    assert headline.transfer_verdict.held is False


def test_headline_gate_reachability_holds(headline):
    assert headline.gate.all_branches_fired is True


@pytest.mark.parametrize("stage", STAGES)
def test_baseline3_median_matches_the_divergence_numerator(headline, stage):
    assert headline.baseline3.median_ns[stage] == pytest.approx(
        headline.divergence.hw_median_ns[stage], rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("stage", STAGES)
def test_baseline1_hardware_median_matches_the_divergence(headline, stage):
    assert headline.baseline1.hw_median_ns[stage] == pytest.approx(
        headline.divergence.hw_median_ns[stage], rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("stage", STAGES)
def test_baseline1_declared_time_matches_the_divergence(headline, stage):
    assert headline.baseline1.declared_ns[stage] == pytest.approx(
        headline.divergence.declared_ns[stage], rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("stage", STAGES)
def test_baseline2_hardware_median_matches_the_divergence(headline, stage):
    assert headline.baseline2.hw_median_ns[stage] == pytest.approx(
        headline.divergence.hw_median_ns[stage], rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("stage", STAGES)
def test_baseline1_relative_error_follows_from_the_divergence_ratio(headline, stage):
    # Two modules, one identity: |R - 1| is baseline 1's relative error when the
    # error is expressed against the measured time, because
    # declared/hw = 1/R. Checked to tolerance on both sides, never bit-exactly.
    ratio = headline.divergence.ratio[stage]
    assert headline.baseline1.relative_error[stage] == pytest.approx(
        abs(1.0 / ratio - 1.0), rel=1e-9, abs=0.0
    )


def test_drift_comparison_raw_is_the_reported_divergence(headline):
    assert headline.drift.raw.spread == pytest.approx(
        headline.divergence.spread, rel=1e-12, abs=0.0
    )


def test_confounding_uses_the_same_nine_cells(headline):
    assert set(headline.confounding.cells) == set(headline.campaign.runs["cell"])


def test_frequency_covariation_uses_the_same_stages(headline):
    assert headline.frequency.stages == headline.divergence.stages


def test_headline_lines_render_without_error(headline):
    lines = headline_lines(headline)
    assert len(lines) > 20
    assert all(isinstance(line, str) for line in lines)


def test_whole_pipeline_measured_total_is_the_sum_of_the_stage_medians(headline):
    total = sum(headline.divergence.hw_median_ns.values())
    assert total == pytest.approx(headline.baseline1.total_hw_ns, rel=1e-12, abs=0.0)


def test_argmax_stage_dominates_measured_frame_time(headline):
    # The least trustworthy stage as a schedulability input is also the most
    # expensive one, which is what makes the divergence consequential rather
    # than a curiosity about a cheap stage.
    assert headline.divergence.argmax_share > 0.5


def test_argmin_stage_is_a_negligible_share_of_frame_time(headline):
    assert headline.divergence.argmin_share < 0.01


def test_simulation_and_hardware_stage_orders_differ_from_declared(headline):
    simulated = headline.simulation.stage_medians_ns()
    simulated_order = tuple(sorted(STAGES, key=lambda name: simulated[name]))
    assert simulated_order != headline.divergence.declared_order


def test_the_platform_ratio_is_far_more_uniform_than_the_declared_model(headline):
    # The finding behind both verdicts, asserted as a measured comparison: the
    # hardware-to-simulation ratio spans well under one order of magnitude
    # across the six stages, while the declared model's divergence spans three.
    ratios = np.asarray(
        [headline.baseline2.per_stage_ratio[stage] for stage in STAGES], dtype=np.float64
    )
    platform_spread = float(ratios.max() / ratios.min())
    assert platform_spread < 3.0
    assert headline.divergence.spread > 100.0 * platform_spread
