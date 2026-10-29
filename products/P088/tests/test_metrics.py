"""Detection metrics and the exact binomial interval, with hand-calculated answers."""

from __future__ import annotations

import math

import pytest

from tmtcdiff.changes import ChangeClass
from tmtcdiff.metrics import ClassMetrics, aggregate, clopper_pearson, per_class_metrics

RESCALED = ChangeClass.RESCALED
REMOVED = ChangeClass.REMOVED


class TestClassMetrics:
    def test_known_answer(self) -> None:
        # 3 true positives, 1 false positive, 1 false negative.
        # precision 3/4 = 0.75, recall 3/4 = 0.75, F1 = 0.75.
        metrics = ClassMetrics(RESCALED, true_positives=3, false_positives=1, false_negatives=1)
        assert metrics.precision == pytest.approx(0.75, rel=1e-12, abs=0.0)
        assert metrics.recall == pytest.approx(0.75, rel=1e-12, abs=0.0)
        assert metrics.f1 == pytest.approx(0.75, rel=1e-12, abs=0.0)

    def test_asymmetric_known_answer(self) -> None:
        # precision 2/5 = 0.4, recall 2/3; F1 = 2*0.4*(2/3)/(0.4+2/3) = 0.5.
        metrics = ClassMetrics(RESCALED, true_positives=2, false_positives=3, false_negatives=1)
        assert metrics.precision == pytest.approx(0.4, rel=1e-12, abs=0.0)
        assert metrics.recall == pytest.approx(2 / 3, rel=1e-12, abs=0.0)
        assert metrics.f1 == pytest.approx(0.5, rel=1e-12, abs=0.0)

    def test_no_detections_gives_nan_precision_not_one(self) -> None:
        metrics = ClassMetrics(RESCALED, true_positives=0, false_positives=0, false_negatives=2)
        assert math.isnan(metrics.precision)

    def test_no_ground_truth_gives_nan_recall_not_one(self) -> None:
        metrics = ClassMetrics(RESCALED, true_positives=0, false_positives=2, false_negatives=0)
        assert math.isnan(metrics.recall)

    def test_f1_is_nan_when_either_metric_is_nan(self) -> None:
        metrics = ClassMetrics(RESCALED, true_positives=0, false_positives=0, false_negatives=0)
        assert math.isnan(metrics.f1)

    def test_f1_is_nan_when_both_metrics_are_zero(self) -> None:
        metrics = ClassMetrics(RESCALED, true_positives=0, false_positives=1, false_negatives=1)
        assert math.isnan(metrics.f1)

    def test_counts(self) -> None:
        metrics = ClassMetrics(RESCALED, true_positives=3, false_positives=1, false_negatives=2)
        assert metrics.detections == 4
        assert metrics.injected == 5


class TestPerClassMetrics:
    def test_a_perfect_detection(self) -> None:
        truth = [(RESCALED, "A"), (REMOVED, "B")]
        metrics = per_class_metrics(truth, truth)
        assert metrics[RESCALED].recall == 1.0
        assert metrics[REMOVED].precision == 1.0

    def test_a_wrong_class_is_a_false_positive_and_a_false_negative(self) -> None:
        metrics = per_class_metrics([(RESCALED, "A")], [(REMOVED, "A")])
        assert metrics[RESCALED].false_negatives == 1
        assert metrics[REMOVED].false_positives == 1

    def test_a_wrong_item_is_counted_both_ways(self) -> None:
        metrics = per_class_metrics([(RESCALED, "A")], [(RESCALED, "B")])
        assert metrics[RESCALED].true_positives == 0
        assert metrics[RESCALED].false_positives == 1
        assert metrics[RESCALED].false_negatives == 1

    def test_duplicates_are_counted_as_multisets(self) -> None:
        metrics = per_class_metrics([(RESCALED, "A"), (RESCALED, "A")], [(RESCALED, "A")])
        assert metrics[RESCALED].true_positives == 1
        assert metrics[RESCALED].false_negatives == 1

    def test_classes_present_only_in_the_detections_appear(self) -> None:
        metrics = per_class_metrics([], [(RESCALED, "A")])
        assert metrics[RESCALED].false_positives == 1

    def test_empty_inputs_give_an_empty_mapping(self) -> None:
        assert per_class_metrics([], []) == {}


class TestAggregate:
    def test_micro_average_sums_the_counts(self) -> None:
        entries = [
            ClassMetrics(RESCALED, 3, 1, 1),
            ClassMetrics(REMOVED, 2, 0, 2),
        ]
        total = aggregate(entries)
        assert (total.true_positives, total.false_positives, total.false_negatives) == (5, 1, 3)

    def test_micro_averaged_precision(self) -> None:
        entries = [ClassMetrics(RESCALED, 3, 1, 1), ClassMetrics(REMOVED, 2, 0, 2)]
        assert aggregate(entries).precision == pytest.approx(5 / 6, rel=1e-12, abs=0.0)

    def test_empty_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            aggregate([])


class TestClopperPearson:
    def test_zero_successes_known_answer(self) -> None:
        # With k = 0 the lower bound is 0 and the upper bound is
        # 1 - (alpha/2)**(1/n) = 1 - 0.025**0.1 = 0.30849...
        low, high = clopper_pearson(0, 10)
        assert low == 0.0
        assert high == pytest.approx(1.0 - 0.025 ** (1 / 10), rel=1e-9, abs=0.0)

    def test_all_successes_known_answer(self) -> None:
        # With k = n the upper bound is 1 and the lower bound is
        # (alpha/2)**(1/n) = 0.025**0.1 = 0.69150...
        low, high = clopper_pearson(10, 10)
        assert high == 1.0
        assert low == pytest.approx(0.025 ** (1 / 10), rel=1e-9, abs=0.0)

    def test_the_interval_brackets_the_point_estimate(self) -> None:
        low, high = clopper_pearson(3, 10)
        assert low < 0.3 < high

    def test_the_interval_narrows_with_more_trials(self) -> None:
        narrow = clopper_pearson(50, 100)
        wide = clopper_pearson(5, 10)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_zero_trials_gives_a_degenerate_interval(self) -> None:
        assert clopper_pearson(0, 0) == (0.0, 0.0)

    def test_more_successes_than_trials_rejected(self) -> None:
        with pytest.raises(ValueError, match="exceeds trials"):
            clopper_pearson(11, 10)

    def test_negative_counts_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            clopper_pearson(-1, 10)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_alpha_outside_the_open_unit_interval_rejected(self, alpha: float) -> None:
        with pytest.raises(ValueError, match=r"alpha must lie in \(0, 1\)"):
            clopper_pearson(1, 10, alpha)

    def test_a_wider_alpha_gives_a_narrower_interval(self) -> None:
        tight = clopper_pearson(5, 20, 0.5)
        loose = clopper_pearson(5, 20, 0.01)
        assert (tight[1] - tight[0]) < (loose[1] - loose[0])
