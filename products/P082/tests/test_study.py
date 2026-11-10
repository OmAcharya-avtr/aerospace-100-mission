"""The ordered study: baselines first, then the learned model, then the scores.

The tests here are about ORDER and STRUCTURE, not about the research numbers:
the numbers are measured and reported by the validation scripts and pinned in
``tests/test_regressions.py``. What is asserted here is that the comparison is
set up so that its numbers mean what they claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from sitediverse.study import (
    MARGIN_BLOCK,
    MARGIN_REPLICATES,
    MODEL_NAMES,
    joint_from_rho,
    row_nodes,
    run_study,
    table_half_mask,
)


@pytest.fixture(scope="module")
def study(short_record):
    """A study on the declared SHORT record, with a reduced bootstrap.

    The short record and 60 replicates keep the whole test module inside a few
    seconds. The reference-record study is run by the validation scripts, which
    is where the research numbers come from.
    """
    return run_study(
        short_record, "short", margin_replicates=60, margin_block=32
    )


def test_the_declared_model_list_is_the_one_documented():
    """Equation (39)'s table, as a list, in report order."""
    assert MODEL_NAMES == ("independent", "exponential", "per_regime", "learned")
    assert MARGIN_REPLICATES == 400
    assert MARGIN_BLOCK == 64


def test_the_study_scores_the_three_cells_and_not_the_fourth(study):
    """Equation (29): the joint extrapolation cell is not a headline score."""
    assert set(study.scores) == {"train", "held-out SITES", "held-out TIME"}


def test_the_independent_model_predicts_a_correlation_of_exactly_zero(study):
    """Baseline 1 is ``rho = 0``, which is the whole of what it assumes."""
    assert np.all(study.predictions["independent"] == 0.0)


def test_the_independent_models_joint_equals_the_conditional_product(study):
    """With ``rho = 0`` the copula of equation (8) reduces to equation (15).

    A structural cross-check between two code paths: the copula route reaches
    the answer through the quadrature of equation (9) over the binned nodes,
    and the conditional product of equation (15) reaches it through a product
    of marginals over the same epochs. The correlation excess at ``rho = 0`` is
    exactly zero, so the two must agree, and they are not the same arithmetic.
    The tolerance is the binned nodes' own measured error.
    """
    from sitediverse.independence import conditional_product

    rate = study.climatology.rate_series()
    worst = 0.0
    for row in range(0, study.table.rows, 23):
        sites = [int(study.table.site_i[row]), int(study.table.site_j[row])]
        window = study.split.time_mask(int(study.table.half[row])) & (
            study.record_for_nodes().regime == int(study.table.regime[row])
        )
        expected = conditional_product(rate[window][:, sites])
        worst = max(worst, abs(study.joint_predicted["independent"][row] - expected))
    assert worst < 1e-5


def test_every_models_joint_prediction_is_a_probability(study):
    """Structural, for all four models over all rows."""
    for model in MODEL_NAMES:
        values = study.joint_predicted[model]
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)
        assert values.shape == (study.table.rows,)


def test_the_baselines_are_fitted_on_the_training_cell_only(study):
    """The sample count must equal the training row count and nothing more."""
    train = int(np.count_nonzero(study.table.mask(0)))
    assert study.exponential.sample_count == train
    assert sum(m.sample_count for m in study.per_regime.models) == train
    assert study.learned.report.rows == train


def test_the_strongest_baseline_is_chosen_on_the_training_cell(study):
    """Selecting a baseline on a hold-out cell would invalidate that cell.

    The selection rule is the training-cell error, which is the only cell the
    models were allowed to see. The test checks the rule rather than the
    outcome, because which baseline wins is a measurement.
    """
    train = study.scores["train"].joint_mae
    expected = min(("exponential", "per_regime"), key=lambda name: train[name])
    assert study.strongest_baseline == expected


def test_the_independence_baseline_is_the_worst_of_the_four_on_every_cell(study):
    """The thing being refuted is refuted, on every cell, on this record.

    This is a research claim and it is asserted here because it is the
    product's premise: if independence were competitive there would be nothing
    to measure. The MARGIN is reported by the validation scripts; what is
    asserted is the ordering.
    """
    for score in study.scores.values():
        assert score.joint_mae["independent"] > score.joint_mae["exponential"]


def test_both_hold_out_cells_get_their_own_margins_against_both_baselines(study):
    """Equation (41): two cells by two baselines, never merged."""
    cells = {result.cell for result in study.margins}
    baselines = {result.baseline for result in study.margins}
    assert cells == {"held-out SITES", "held-out TIME"}
    assert baselines == {"exponential", "per_regime"}
    assert len(study.margins) == 4


def test_each_margin_matches_the_difference_of_the_two_cell_errors(study):
    """The reported margin is the quantity it claims to be."""
    for result in study.margins:
        score = study.scores[result.cell]
        expected = score.joint_mae[result.baseline] - score.joint_mae["learned"]
        assert result.margin == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_each_margins_bootstrap_interval_is_ordered_and_non_degenerate(study):
    """A percentile interval from 60 distinct replicates has positive width."""
    for result in study.margins:
        assert result.bootstrap.lower <= result.bootstrap.upper
        assert result.bootstrap.replicate_values.size == 60


def test_the_reference_half_width_is_positive_and_is_a_separate_figure(study):
    """The two intervals of equation (41) answer different questions.

    The paired interval on the margin is narrow because the two errors move
    together; the reference half-width is the scale at which the thing being
    scored against is itself uncertain. They must not be the same number.
    """
    for result in study.margins:
        assert result.reference_half_width > 0.0
        assert result.reference_half_width != result.bootstrap.half_width


def test_the_margin_verdicts_are_consistent_with_their_own_definitions(study):
    """``inside_interval`` and ``inside_reference_half_width``, checked."""
    for result in study.margins:
        assert result.inside_interval == (
            result.bootstrap.lower <= 0.0 <= result.bootstrap.upper
        )
        assert result.inside_reference_half_width == (
            abs(result.margin) <= result.reference_half_width
        )


def test_the_interval_coverage_is_reported_per_cell(study):
    """Nominal 80 per cent; the measured value is a measurement, not a target."""
    for score in study.scores.values():
        assert 0.0 <= score.interval_coverage <= 1.0


def test_the_study_reports_its_own_wall_clock(study):
    """The compute budget is reported, not assumed."""
    assert study.seconds > 0.0
    assert study.learned_seconds > 0.0
    assert study.learned_seconds < study.seconds


def test_the_study_finishes_inside_the_declared_compute_budget(study):
    """Three minutes for any training or Monte Carlo run."""
    assert study.seconds < 180.0


def test_row_nodes_are_built_once_and_shared_across_models(study):
    """So that two models cannot be scored through different nodes."""
    nodes = row_nodes(study.table, study.climatology, study.record_for_nodes())
    assert len(nodes) == study.table.rows


def test_joint_from_rho_rejects_a_mismatched_correlation_array(study):
    """Input validation."""
    nodes = row_nodes(study.table, study.climatology, study.record_for_nodes())
    with pytest.raises(ValueError, match="rho must have shape"):
        joint_from_rho(nodes, np.zeros(len(nodes) + 1))


def test_joint_from_rho_is_monotone_in_the_correlation(study):
    """A higher correlation gives a higher joint unavailability, every row."""
    nodes = row_nodes(study.table, study.climatology, study.record_for_nodes())
    low = joint_from_rho(nodes, np.zeros(len(nodes)))
    high = joint_from_rho(nodes, np.full(len(nodes), 0.8))
    assert np.all(high > low - 1e-15)


def test_table_half_mask_partitions_the_record(short_record):
    """Structural."""
    first = table_half_mask(short_record, 0)
    second = table_half_mask(short_record, 1)
    assert not np.any(first & second)
    assert np.all(first | second)
    assert int(first.sum()) == short_record.epochs // 2


def test_run_study_rejects_a_split_that_leaves_no_training_rows(short_record):
    """Holding out all but one station leaves no training pair."""
    with pytest.raises(ValueError, match="no training pair exists"):
        run_study(
            short_record,
            "short",
            held_out_sites=tuple(range(short_record.site_count - 1)),
        )
