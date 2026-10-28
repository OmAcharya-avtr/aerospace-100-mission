"""Rendering: the table primitive and each renderer, including the ordering rule."""

from __future__ import annotations

import numpy as np
import pytest

from telemgap import classify, determinability, policy
from telemgap.benchmark import score
from telemgap.report import (
    _number,
    format_table,
    render_determinability,
    render_integrity,
    render_policy,
    render_scores,
)


def test_format_table_known_answer() -> None:
    text = format_table(["a", "bb"], [["x", "yyy"], ["zz", "w"]])
    lines = text.splitlines()
    assert lines[0] == "a   bb "
    assert lines[1] == "--  ---"
    assert lines[2] == "x   yyy"
    assert lines[3] == "zz  w  "


def test_format_table_rejects_a_ragged_row() -> None:
    with pytest.raises(ValueError, match="row 1 has 1 cells"):
        format_table(["a", "b"], [["x", "y"], ["z"]])


def test_format_table_of_no_rows() -> None:
    text = format_table(["only"], [])
    assert text.splitlines() == ["only", "----"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0, "1"),
        (0.5, "0.5"),
        (float("nan"), "nan"),
        (float("inf"), "inf"),
        (float("-inf"), "-inf"),
        (None, "-"),
    ],
)
def test_number_rendering(value: object, expected: str) -> None:
    assert _number(value) == expected


def test_render_determinability_leads_with_the_heading(nominal_record) -> None:
    text = render_determinability(determinability.assess(nominal_record))
    assert text.startswith("WHAT THIS RECORD CANNOT DETERMINE")
    for key in determinability.LIMIT_KEYS:
        assert key in text
    assert "consequence:" in text


def test_render_integrity_includes_counts_and_the_mode(generated_small) -> None:
    report = classify.classify(generated_small.record)
    text = render_integrity(report, max_anomalies=5)
    assert "INTEGRITY REPORT (mode=strengthened)" in text
    assert "declared ordinal tier, not a probability" in text
    assert "range_violations" in text
    assert "undecided" in text


def test_render_integrity_of_a_clean_record(nominal_record) -> None:
    report = classify.classify(nominal_record)
    text = render_integrity(report)
    assert "anomalies" not in text.split("INTEGRITY REPORT")[1].split("verdict")[0]
    assert "missing" in text


def test_render_policy_labels_the_coverage_statement(generated_small) -> None:
    record = generated_small.record
    report = classify.classify(record)
    plan = policy.plan(record, report, dict.fromkeys(record.schema.channel_names, 1.0))
    text = render_policy(plan, max_rows=6)
    assert text.startswith("GAP POLICY")
    assert "COVERAGE" in text
    assert "estimator bandwidth Hz" in text
    assert "coverage_k = 3" in text


def test_render_policy_lists_refusals(generated_small) -> None:
    record = generated_small.record
    report = classify.classify(record)
    plan = policy.plan(record, report, dict.fromkeys(record.schema.channel_names, 1e-9))
    text = render_policy(plan, max_rows=4)
    assert plan.refusals
    assert "refusals, with the gate that refused them" in text


def test_render_scores_shows_the_confusion_matrix() -> None:
    truth = np.array([0, 1, 2, 2])
    predicted = np.array([0, 1, 2, -1])
    text = render_scores(score(truth, predicted, "label"))
    assert "label  population=4" in text
    assert "confusion (rows true, columns predicted)" in text
    assert "undecided" in text
