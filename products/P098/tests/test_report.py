"""The headline report and its rendering."""

from __future__ import annotations

import pytest

from evidencestale.report import build_report, render_report


def test_headline_stale_fraction_matches_the_committed_labels(layered_corpus) -> None:
    report = build_report(layered_corpus, "complete", resamples=400)
    assert report.truth.stale_count == 194
    assert report.truth.total == 480
    # 194 / 480 = 0.40416666...
    assert report.truth.stale_fraction == pytest.approx(194 / 480, rel=1e-15, abs=0.0)
    assert report.closure_hash.stale_fraction == pytest.approx(
        report.truth.stale_fraction, rel=0.0, abs=0.0
    )


def test_closure_comparison_makes_no_error_in_either_direction(layered_corpus) -> None:
    report = build_report(layered_corpus, "complete", resamples=400)
    errors = report.errors("baseline2_closure_hash")
    assert (errors.false_negative, errors.false_positive) == (0, 0)
    assert errors.agreement == 1.0


def test_timestamp_baseline_makes_errors_in_both_directions(layered_corpus) -> None:
    report = build_report(layered_corpus, "complete", resamples=400)
    errors = report.errors("baseline1_timestamp")
    assert errors.false_negative > 0
    assert errors.false_positive > 0
    stronger = report.errors("baseline1b_timestamp_depth1")
    # Reading one level of the graph trades missed staleness for false alarms.
    assert stronger.false_negative < errors.false_negative
    assert stronger.false_positive > errors.false_positive


def test_unknown_method_rejected(layered_corpus) -> None:
    report = build_report(layered_corpus, "complete", resamples=200)
    with pytest.raises(ValueError, match="not scored in this report"):
        report.errors("baseline4")


def test_goal_weighted_fraction_is_at_least_the_item_fraction(layered_corpus) -> None:
    # A leaf goal is stale if ANY of its supporting solutions is, so the
    # goal-weighted figure cannot be below the item figure when every goal has
    # the same number of solutions.
    report = build_report(layered_corpus, "complete", resamples=200)
    assert report.goal_count == 120
    assert report.goal_stale_fraction >= report.truth.stale_fraction


def test_partial_declarations_lower_the_reported_fraction(layered_corpus) -> None:
    complete = build_report(layered_corpus, "complete", resamples=200)
    partial = build_report(layered_corpus, "partial", resamples=200)
    assert partial.closure_hash.stale_fraction < complete.closure_hash.stale_fraction
    # The truth does not move: it is always computed over the true graph.
    assert partial.truth.as_array() == complete.truth.as_array()


def test_both_bootstrap_units_are_reported(layered_corpus) -> None:
    report = build_report(layered_corpus, "complete", resamples=400)
    assert report.cluster_interval.unit == "cluster"
    assert report.item_interval.unit == "item"
    assert report.cluster_interval.n_units == 240
    assert report.item_interval.n_units == 480


def test_rendered_report_carries_the_required_statements(layered_corpus) -> None:
    text = render_report(build_report(layered_corpus, "complete", resamples=200))
    assert "research-grade" in text
    assert "not flight-qualified" in text
    assert "HEADLINE -- stale fraction" in text
    assert "missed staleness" in text
    assert "false alarms" in text


def test_report_on_the_cyclic_corpus_runs_and_is_exact(cyclic_corpus) -> None:
    report = build_report(cyclic_corpus, "complete", resamples=200)
    errors = report.errors("baseline2_closure_hash")
    assert (errors.false_negative, errors.false_positive) == (0, 0)
    assert report.truth.total == 12
