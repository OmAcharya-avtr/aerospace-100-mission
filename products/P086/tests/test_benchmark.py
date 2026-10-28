"""Scoring: equations (13) to (15) against hand-computed confusion matrices."""

from __future__ import annotations

import numpy as np
import pytest

from telemgap import classify
from telemgap.benchmark import (
    UNDECIDED,
    VALUE_OBSERVATIONS,
    VERDICT_TO_CODE,
    BenchmarkResult,
    confusion_matrix,
    rule_labels,
    score,
)
from telemgap.learned import CLASS_NAMES


def test_verdict_map_excludes_missing() -> None:
    assert set(VERDICT_TO_CODE) == {"corrupt", "misparsed", "undecided"}
    assert VERDICT_TO_CODE["undecided"] == UNDECIDED
    assert VALUE_OBSERVATIONS == (
        "value_out_of_range",
        "value_local_outlier",
        "value_in_failed_frame",
    )


def test_confusion_matrix_known_answer() -> None:
    # Hand-built: truth 0,0,1,1,2,2; predicted 0,1,1,-1,2,0.
    truth = np.array([0, 0, 1, 1, 2, 2])
    predicted = np.array([0, 1, 1, UNDECIDED, 2, 0])
    matrix = confusion_matrix(truth, predicted)
    assert matrix.shape == (3, 4)
    assert matrix.tolist() == [[1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0]]
    assert int(matrix.sum()) == truth.size


def test_confusion_matrix_validates_shapes() -> None:
    with pytest.raises(ValueError, match="must match"):
        confusion_matrix(np.zeros(3, dtype=np.int64), np.zeros(4, dtype=np.int64))


def test_score_known_answers() -> None:
    # Same arrays as above.
    # clean  : support 2, predicted 2 (rows 0 and 5), tp 1 -> P 0.5, R 0.5
    # corrupt: support 2, predicted 2 (rows 1 and 2), tp 1 -> P 0.5, R 0.5,
    #          undecided 1 -> rate 0.5
    # misparsed: support 2, predicted 1, tp 1 -> P 1.0, R 0.5
    truth = np.array([0, 0, 1, 1, 2, 2])
    predicted = np.array([0, 1, 1, UNDECIDED, 2, 0])
    table = score(truth, predicted, "hand")
    assert table.label == "hand"
    assert table.population == 6
    assert table["clean"].precision == pytest.approx(0.5, rel=1e-12, abs=0.0)
    assert table["clean"].recall == pytest.approx(0.5, rel=1e-12, abs=0.0)
    assert table["corrupt"].precision == pytest.approx(0.5, rel=1e-12, abs=0.0)
    assert table["corrupt"].recall == pytest.approx(0.5, rel=1e-12, abs=0.0)
    assert table["corrupt"].undecided == 1
    assert table["corrupt"].undecided_rate == pytest.approx(0.5, rel=1e-12, abs=0.0)
    assert table["misparsed"].precision == pytest.approx(1.0, rel=1e-12, abs=0.0)
    assert table["misparsed"].recall == pytest.approx(0.5, rel=1e-12, abs=0.0)
    # f1 of P = R = 0.5 is 0.5; of P = 1, R = 0.5 is 2 / 3.
    assert table["clean"].f1 == pytest.approx(0.5, rel=1e-12, abs=0.0)
    assert table["misparsed"].f1 == pytest.approx(2.0 / 3.0, rel=1e-12, abs=0.0)
    # 5 of 6 rows decided; 3 of those 5 correct.
    assert table.decided_fraction == pytest.approx(5.0 / 6.0, rel=1e-12, abs=0.0)
    assert table.accuracy_decided == pytest.approx(3.0 / 5.0, rel=1e-12, abs=0.0)


def test_score_reports_nan_for_an_empty_class() -> None:
    truth = np.array([0, 0, 0])
    predicted = np.array([0, 0, 0])
    table = score(truth, predicted, "empty")
    assert np.isnan(table["corrupt"].precision)
    assert np.isnan(table["corrupt"].recall)
    assert np.isnan(table["corrupt"].f1)
    assert table["clean"].recall == pytest.approx(1.0, rel=0.0, abs=0.0)


def test_score_of_an_all_undecided_prediction() -> None:
    truth = np.array([0, 1, 2])
    predicted = np.full(3, UNDECIDED)
    table = score(truth, predicted, "abstain")
    assert table.decided_fraction == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert np.isnan(table.accuracy_decided)
    for name in CLASS_NAMES:
        assert np.isnan(table[name].precision)
        assert table[name].undecided == 1


def test_score_lookup_raises_on_unknown_class() -> None:
    table = score(np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64), "x")
    with pytest.raises(KeyError, match="no class named"):
        table["nonsense"]
    assert table.precision("clean") == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert table.recall("clean") == pytest.approx(1.0, rel=0.0, abs=0.0)


def test_rule_labels_shape_and_clean_default(nominal_record) -> None:
    report = classify.classify(nominal_record)
    labels = rule_labels(nominal_record, report)
    assert labels.shape == (len(nominal_record), nominal_record.n_channels)
    assert np.all(labels == 0)


def test_rule_labels_pick_up_each_verdict(small_schema, record_builder) -> None:
    raws = [40] * 31
    raws[10] = 0x2800  # rescued by byteswap -> misparsed
    raws[20] = 0xFFFF  # nothing rescues -> undecided
    validity = [True] * 31
    raws[25] = 50000
    validity[25] = False  # out of range with failed validity -> corrupt
    record = record_builder(small_schema, raws, validity=validity)
    report = classify.classify(record)
    labels = rule_labels(record, report)
    assert labels[10, 0] == 2
    assert labels[20, 0] == UNDECIDED
    assert labels[25, 0] == 1
    assert labels[0, 0] == 0


def test_rule_labels_prefer_a_decided_verdict(small_schema, record_builder) -> None:
    # A frame can trip both the range screen and the residual screen. The
    # decided verdict must win over the undecided one.
    raws = [40] * 31
    raws[15] = 0x2800
    record = record_builder(small_schema, raws)
    report = classify.classify(record)
    positions = [
        anomaly.observation for anomaly in report.anomalies if anomaly.index == 15
    ]
    assert "value_out_of_range" in positions
    assert rule_labels(record, report)[15, 0] == 2


def test_benchmark_result_verdict_lines_report_both_directions() -> None:
    result = BenchmarkResult()
    truth = np.array([0, 1, 1, 2, 2])
    rules = np.array([0, 1, UNDECIDED, 2, 2])
    learned = np.array([0, 1, 1, 2, 0])
    result.add("all_rules_strengthened", score(truth, rules, "rules"))
    result.add("all_learned", score(truth, learned, "learned"))
    lines = result.verdict_lines()
    assert len(lines) == 3 * len(CLASS_NAMES)
    joined = "\n".join(lines)
    assert "winner rules" in joined
    assert "winner learned" in joined


def test_benchmark_result_reports_a_missing_table() -> None:
    result = BenchmarkResult()
    assert "unavailable" in result.verdict_lines()[0]


def test_benchmark_result_stores_timings_and_populations() -> None:
    result = BenchmarkResult()
    result.timings_s["fit"] = 1.25
    result.populations["test"] = 100
    result.notes.append("a note")
    assert result.timings_s["fit"] == pytest.approx(1.25, rel=0.0, abs=0.0)
    assert result.populations["test"] == 100
    assert result.notes == ["a note"]


def test_surfaced_rate_known_answers() -> None:
    # Equation (15b). truth 1,1,1,1; predicted 1,-1,-1,0.
    # corrupt: tp 1, undecided 2 -> surfaced 3 / 4 = 0.75, recall 0.25.
    truth = np.array([1, 1, 1, 1])
    predicted = np.array([1, UNDECIDED, UNDECIDED, 0])
    table = score(truth, predicted, "surfaced")
    assert table["corrupt"].recall == pytest.approx(0.25, rel=1e-12, abs=0.0)
    assert table["corrupt"].surfaced_rate == pytest.approx(0.75, rel=1e-12, abs=0.0)
    # An empty class has no surfaced rate.
    assert np.isnan(table["misparsed"].surfaced_rate)
    # A predictor that never abstains has surfaced == recall.
    never = score(truth, np.array([1, 1, 0, 0]), "never")
    assert never["corrupt"].surfaced_rate == pytest.approx(
        never["corrupt"].recall, rel=0.0, abs=0.0
    )
