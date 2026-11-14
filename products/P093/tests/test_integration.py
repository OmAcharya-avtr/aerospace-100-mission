"""End to end over a whole synthetic campaign directory on disk.

This is the integration test the medium class requires. It writes a complete
twenty-cell campaign to disk, loads it through the real loader, and runs every
published surface over it: the three baselines, the rail-accounting fit, the
between-run interval, the headline decomposition for every pipeline cell, the
window dataset, and the learned predictor against both baselines on both
grouped splits -- including the specification's pre-registered comparison.

Everything here is SYNTHETIC. The measured campaign is exercised by the scripts
under ``validation/``, which refuse and exit non-zero when it is absent.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from powerenvelope.baselines import (
    between_run_spread,
    duty_cycle_model,
    energy_per_inference,
    floor_power,
    ground_truth,
    rail_accounting_curve,
)
from powerenvelope.decomposition import decompose
from powerenvelope.predictor import (
    ActivePowerPredictor,
    AnalyticActivePowerModel,
    ConstantPowerPredictor,
    score,
)
from powerenvelope.report import (
    MEASUREMENT_MODEL,
    campaign_lines,
    cell_table,
    score_table,
)
from powerenvelope.trace import PLANNED_CELLS
from powerenvelope.windows import extract_windows, grouped_splits

from fixtures import synthetic_campaign


@pytest.fixture(scope="module")
def whole(tmp_path_factory):
    """A complete synthetic campaign, written once for this module."""
    return synthetic_campaign.campaign(tmp_path_factory.mktemp("integration"))


def test_every_planned_cell_loads_with_all_three_traces(whole):
    assert set(whole.cells) == set(PLANNED_CELLS)
    for cell in whole.cells.values():
        assert len(cell.power) > 0
        assert len(cell.monitor) > 0
        assert (cell.decisions is not None) == cell.has_pipeline


def test_all_three_baselines_run_on_every_pipeline_cell(whole):
    floor = floor_power(whole)
    for cell in whole.pipeline_cells():
        baseline1 = energy_per_inference(cell)
        baseline2 = duty_cycle_model(whole, cell.label, floor=floor)
        baseline3 = ground_truth(cell)
        assert baseline1.energy_mj > 0.0
        assert baseline2.energy_per_useful_decision_mj > 0.0
        assert baseline3.energy_per_useful_decision_mj > 0.0
        # Ground truth charges the sleep window; energy per inference does not.
        assert baseline3.energy_per_useful_decision_mj > baseline1.energy_mj


def test_baseline2_tracks_baseline3_across_the_whole_grid(whole):
    # The baseline is validated against the ground truth BEFORE the learned
    # component is benchmarked, which is the order the build guide requires.
    floor = floor_power(whole)
    ratios = []
    for cell in whole.grid_cells():
        model = duty_cycle_model(whole, cell.label, floor=floor)
        truth = ground_truth(cell)
        ratios.append(
            model.energy_per_useful_decision_mj / truth.energy_per_useful_decision_mj
        )
    assert max(abs(ratio - 1.0) for ratio in ratios) < 0.10


def test_rail_accounting_and_between_run_interval_run(whole):
    rail = rail_accounting_curve(whole)
    assert len(rail.cells) == 20
    spread = between_run_spread(whole)
    assert spread.n == 4


def test_headline_decomposes_for_every_pipeline_cell(whole):
    for cell in whole.pipeline_cells():
        for variant in ("measured-active", "nominal-rail"):
            result = decompose(cell, variant=variant)
            product = result.product_of_named_factors() * result.residual_factor
            assert product == pytest.approx(result.factor, rel=1e-12, abs=0.0)
            assert result.factor > 1.0


def test_learned_predictor_runs_on_both_splits_inside_the_budget(whole):
    windows = extract_windows(whole)
    floor = floor_power(whole)
    started = time.perf_counter()
    rows = []
    for grouping in ("duty", "frame_dim"):
        for label, train, test in grouped_splits(windows, grouping):
            learned = ActivePowerPredictor().fit(
                windows.features[train], windows.targets[train], windows.feature_names
            )
            analytic = AnalyticActivePowerModel().fit(
                whole, windows.cells[train], floor_mw=floor.mean_mw
            )
            constant = ConstantPowerPredictor().fit(
                windows.features[train], windows.targets[train]
            )
            truth = windows.targets[test]
            features = windows.features[test]
            rows.append(score(f"learned/{label}", truth, learned.predict(features)))
            rows.append(score(f"analytic/{label}", truth, analytic.predict(features)))
            rows.append(score(f"constant/{label}", truth, constant.predict(features)))
    elapsed = time.perf_counter() - started
    assert elapsed < 180.0, f"the whole benchmark took {elapsed:.1f} s"
    assert len(rows) == 3 * 8
    assert all(np.isfinite(row.mae_mw) for row in rows)
    assert len(score_table(rows)) == len(rows) + 2


def test_pre_registered_comparison_is_evaluated_and_reported_either_way(whole):
    """The specification's pre-registered expectation, evaluated on synthetic data.

    The expectation is that the analytic duty-cycle model is competitive and
    that the learned predictor's error is NOT smaller than the analytic model's
    by more than the between-run interval's half-width. On the SYNTHETIC
    campaign the four reference repeats are byte-identical, so the half-width is
    exactly zero and the comparison reduces to which error is smaller. The test
    asserts only that the comparison is computable and that both outcomes are
    representable, because the result that matters is the MEASURED one and this
    fixture is not a measurement of anything.
    """
    windows = extract_windows(whole)
    floor = floor_power(whole)
    learned_errors = []
    analytic_errors = []
    for grouping in ("duty", "frame_dim"):
        for label, train, test in grouped_splits(windows, grouping):
            learned = ActivePowerPredictor().fit(
                windows.features[train], windows.targets[train], windows.feature_names
            )
            analytic = AnalyticActivePowerModel().fit(
                whole, windows.cells[train], floor_mw=floor.mean_mw
            )
            truth = windows.targets[test]
            learned_errors.append(
                score(f"l/{label}", truth, learned.predict(windows.features[test])).mae_mw
            )
            analytic_errors.append(
                score(f"a/{label}", truth, analytic.predict(windows.features[test])).mae_mw
            )
    margin = float(np.mean(analytic_errors)) - float(np.mean(learned_errors))
    half_width = between_run_spread(whole).half_width_mj
    assert np.isfinite(margin)
    assert half_width >= 0.0
    verdict = "learned wins by more than the interval" if margin > half_width else "not refuted"
    assert verdict in {"learned wins by more than the interval", "not refuted"}


def test_report_lines_render_without_io(whole):
    text = "\n".join([*campaign_lines(whole), *cell_table(whole)])
    assert "synthetic" in text
    assert "w64_d050_permissive" in text
    assert "MEASUREMENT MODEL" in MEASUREMENT_MODEL


def test_an_incomplete_campaign_is_loadable_but_flagged(tmp_path):
    partial = synthetic_campaign.campaign(tmp_path / "partial", complete=False)
    assert not partial.complete
    assert len(partial.cells) == 20
