"""Report assembly. No printing happens in library code, so these are strings."""

from __future__ import annotations

import pytest

from deadlinerisk.accounting import account
from deadlinerisk.baselines import choose_threshold, mean_plus_margin, threshold_stability
from deadlinerisk.bootstrap import counted_rate_interval, understatement_factor
from deadlinerisk.predictor import Score
from deadlinerisk.report import (
    account_lines,
    campaign_lines,
    evt_lines,
    headline_lines,
    interval_lines,
    margin_lines,
    score_table,
)


def test_campaign_lines_name_the_origin_and_the_model(campaign) -> None:
    lines = campaign_lines(campaign)
    assert any("INPUT ORIGIN: synthetic" in line for line in lines)
    assert any("synthetic fixture" in line for line in lines)
    assert any("implicit, D = T" in line for line in lines)


def test_account_lines_state_both_rates(ground_truth) -> None:
    lines = account_lines(account(ground_truth))
    assert any("counted miss rate" in line for line in lines)
    assert any("executed-only miss rate" in line for line in lines)


def test_margin_lines_state_k(ground_truth) -> None:
    lines = margin_lines(mean_plus_margin(ground_truth.response_s, ground_truth.deadline_s))
    assert any("k with mean + k*sd = D" in line for line in lines)


def test_evt_lines_carry_the_declared_rule(ground_truth) -> None:
    fits = threshold_stability(ground_truth.response_s, ground_truth.deadline_s)
    lines = evt_lines(choose_threshold(fits))
    assert any("declared rule" in line for line in lines)
    assert any("modified scale" in line for line in lines)


def test_interval_lines_report_the_block_selection(ground_truth) -> None:
    lines = interval_lines(counted_rate_interval(ground_truth, resamples=100, seed=3))
    assert any("block length (releases)" in line for line in lines)
    assert any("integrated tau" in line for line in lines)


def test_headline_lines_state_a_verdict_matching_the_interval(ground_truth) -> None:
    headline = understatement_factor(ground_truth, resamples=200, seed=3)
    lines = headline_lines(headline)
    verdict = next(line for line in lines if line.startswith("verdict"))
    if headline.interval_contains_one:
        assert "CONTAINS 1" in verdict
    else:
        assert "EXCLUDES 1" in verdict


def test_score_table_renders_a_row_per_score() -> None:
    scores = [
        Score("a", 10, 0.1, 0.2, 0.01, 0.8),
        Score("b", 10, 0.3, 0.4, -0.02, float("nan")),
    ]
    lines = score_table(scores)
    # Header, rule, two rows.
    assert len(lines) == 4
    assert "predictor" in lines[0]
    assert "n/a" in lines[3]


def test_score_table_is_empty_apart_from_its_header() -> None:
    lines = score_table([])
    assert len(lines) == 2


def test_report_functions_return_lists_not_none(ground_truth, campaign) -> None:
    for lines in (
        campaign_lines(campaign),
        account_lines(account(ground_truth)),
        margin_lines(mean_plus_margin(ground_truth.response_s, ground_truth.deadline_s)),
    ):
        assert isinstance(lines, list)
        assert all(isinstance(line, str) for line in lines)


def test_provenance_rejects_a_non_origin() -> None:
    from deadlinerisk.provenance import Provenance

    with pytest.raises(TypeError, match="origin must be an Origin"):
        Provenance(origin="measured", directory="x", runs=())  # type: ignore[arg-type]


def test_provenance_banner_counts_the_runs() -> None:
    from deadlinerisk.provenance import Origin, Provenance

    provenance = Provenance(origin=Origin.MEASURED, directory="data/traces", runs=("a", "b"))
    assert provenance.is_measured is True
    assert "2 run(s)" in provenance.banner()
