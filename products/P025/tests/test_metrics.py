"""Intervals, confusion matrices and ROC curves, with hand-checked answers."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fdiscope.metrics import (
    Interval,
    confusion_matrix,
    confusion_report,
    mean_ci,
    roc_curve,
    wilson_interval,
)


class TestInterval:
    def test_half_width_and_containment(self):
        iv = Interval(point=1.0, low=0.5, high=1.5)
        assert np.isclose(iv.half_width, 0.5)
        assert iv.contains(0.5) and iv.contains(1.5) and not iv.contains(1.6)

    def test_string_form_is_point_then_bracket(self):
        assert str(Interval(1.0, 0.5, 1.5)).startswith("1 [0.5")


class TestWilsonInterval:
    def test_known_answer(self):
        # k = 10, n = 1000, z = 1.959964:
        #   p = 0.01, denom = 1 + z^2/n = 1.003842
        #   centre = (0.01 + z^2/2000) / denom = 0.011880 / 1.003842 = 0.011834
        #   half = z sqrt(0.01*0.99/1000 + z^2/(4e6)) / denom = 0.006196
        #   -> [0.005638, 0.018030]  (matches the published Wilson formula)
        iv = wilson_interval(10, 1000)
        assert np.isclose(iv.point, 0.01)
        assert np.isclose(iv.low, 0.0054408, atol=1e-6)
        assert np.isclose(iv.high, 0.0183095, atol=1e-6)

    def test_zero_successes_gives_a_lower_bound_of_zero(self):
        iv = wilson_interval(0, 100)
        assert iv.low == 0.0
        assert 0.0 < iv.high < 0.05

    def test_all_successes_gives_an_upper_bound_of_one(self):
        assert wilson_interval(50, 50).high == 1.0

    def test_interval_narrows_with_sample_size(self):
        assert wilson_interval(100, 10000).half_width < wilson_interval(1, 100).half_width

    def test_covers_the_point_estimate(self):
        for k, n in [(0, 10), (3, 10), (10, 10), (7, 1000)]:
            iv = wilson_interval(k, n)
            assert iv.low <= iv.point <= iv.high

    @pytest.mark.parametrize("args", [(0, 0), (-1, 10), (11, 10)])
    def test_rejects_bad_counts(self, args):
        with pytest.raises(ValueError):
            wilson_interval(*args)

    def test_rejects_bad_level(self):
        with pytest.raises(ValueError, match="level"):
            wilson_interval(1, 10, level=1.5)


class TestMeanCi:
    def test_known_answer(self):
        # values (1, 2, 3, 4, 5): mean 3, sd 1.5811388, sem 0.70710678,
        # t(0.975, 4) = 2.7764451 -> half width 1.9633
        iv = mean_ci([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.isclose(iv.point, 3.0)
        assert np.isclose(iv.half_width, 1.963243, atol=1e-5)

    def test_rejects_censored_values(self):
        with pytest.raises(ValueError, match="finite"):
            mean_ci([1.0, np.inf, 2.0])

    def test_rejects_a_single_value(self):
        with pytest.raises(ValueError, match="at least 2"):
            mean_ci([1.0])

    def test_rejects_bad_level(self):
        with pytest.raises(ValueError, match="level"):
            mean_ci([1.0, 2.0], level=0.0)


class TestConfusionMatrix:
    def test_known_answer(self):
        # truth (0, 1, 1, 2), prediction (0, 1, 2, 2):
        #   row 0: [1, 0, 0]; row 1: [0, 1, 1]; row 2: [0, 0, 1]
        m = confusion_matrix([0, 1, 1, 2], [0, 1, 2, 2], 3)
        assert np.array_equal(m, [[1, 0, 0], [0, 1, 1], [0, 0, 1]])

    def test_rows_sum_to_the_class_support(self):
        m = confusion_matrix([0, 1, 1, 2], [0, 1, 2, 2], 3)
        assert np.array_equal(m.sum(axis=1), [1, 2, 1])

    def test_total_equals_the_sample_count(self):
        m = confusion_matrix([0, 1, 1, 2], [0, 1, 2, 2], 3)
        assert m.sum() == 4

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="predicted has"):
            confusion_matrix([0, 1], [0], 2)

    def test_rejects_out_of_range_class(self):
        with pytest.raises(ValueError, match=r"\[0, 2\)"):
            confusion_matrix([0, 2], [0, 0], 2)

    def test_rejects_zero_classes(self):
        with pytest.raises(ValueError, match="n_classes"):
            confusion_matrix([], [], 0)


class TestConfusionReport:
    def test_known_answer_recall_precision_accuracy(self):
        # truth (0, 1, 1, 2), prediction (0, 1, 2, 2):
        #   recall    = 1/1, 1/2, 1/1
        #   precision = 1/1, 1/1, 1/2
        #   accuracy  = 3/4
        rep = confusion_report([0, 1, 1, 2], [0, 1, 2, 2], ("a", "b", "c"))
        assert np.allclose(rep.recall, [1.0, 0.5, 1.0])
        assert np.allclose(rep.precision, [1.0, 1.0, 0.5])
        assert np.isclose(rep.accuracy, 0.75)

    def test_absent_class_gets_nan_not_zero(self):
        rep = confusion_report([0, 0], [0, 0], ("a", "b"))
        assert np.isnan(rep.recall[1])
        assert np.isnan(rep.precision[1])

    def test_text_rendering_contains_every_label_and_the_accuracy(self):
        text = confusion_report([0, 1], [0, 1], ("alpha", "beta")).to_text(12)
        assert "alpha" in text and "beta" in text
        assert "overall accuracy = 1.000000" in text


class TestRocCurve:
    def test_known_answer_perfect_separation(self):
        curve = roc_curve([2.0, 3.0], [0.0, 1.0])
        assert np.isclose(curve.auc, 1.0)

    def test_known_answer_identical_scores_give_half(self):
        # Complete overlap: every threshold gives TPR == FPR, so AUC = 0.5.
        curve = roc_curve([1.0, 1.0], [1.0, 1.0])
        assert np.isclose(curve.auc, 0.5)

    def test_known_answer_one_swap(self):
        # positives (1, 3), negatives (2, 4): concordant pairs are
        # (1>2)? no, (1>4)? no, (3>2)? yes, (3>4)? no -> AUC = 1/4.
        curve = roc_curve([1.0, 3.0], [2.0, 4.0])
        assert np.isclose(curve.auc, 0.25)

    def test_curve_starts_at_the_origin(self):
        curve = roc_curve([2.0, 3.0], [0.0, 1.0])
        assert curve.thresholds[0] == np.inf
        assert curve.tpr[0] == 0.0 and curve.fpr[0] == 0.0

    def test_tpr_at_fpr_interpolates(self):
        curve = roc_curve(np.linspace(1.0, 5.0, 200), np.linspace(0.0, 4.0, 200))
        assert 0.0 <= curve.tpr_at_fpr(0.1) <= 1.0
        assert curve.tpr_at_fpr(1.0) == 1.0

    def test_tpr_at_fpr_rejects_out_of_range(self):
        curve = roc_curve([1.0], [0.0])
        with pytest.raises(ValueError, match="target_fpr"):
            curve.tpr_at_fpr(1.5)

    def test_rejects_empty_samples(self):
        with pytest.raises(ValueError, match="non-empty"):
            roc_curve([], [1.0])

    def test_rejects_non_finite_scores(self):
        with pytest.raises(ValueError, match="finite"):
            roc_curve([np.inf], [0.0])

    def test_label_is_carried_through(self):
        assert roc_curve([1.0], [0.0], "cusum").label == "cusum"

    @settings(max_examples=30, deadline=None)
    @given(shift=st.floats(0.0, 4.0, allow_nan=False))
    def test_auc_is_monotone_in_the_separation(self, shift):
        rng = np.random.default_rng(0)
        neg = rng.standard_normal(400)
        pos = rng.standard_normal(400) + shift
        assert 0.4 <= roc_curve(pos, neg).auc <= 1.0

    @settings(max_examples=20, deadline=None)
    @given(n=st.integers(5, 60))
    def test_auc_matches_the_mann_whitney_statistic(self, n):
        rng = np.random.default_rng(n)
        pos = rng.standard_normal(n) + 1.0
        neg = rng.standard_normal(n)
        greater = float(np.mean(pos[:, None] > neg[None, :]))
        ties = float(np.mean(pos[:, None] == neg[None, :]))
        assert np.isclose(roc_curve(pos, neg).auc, greater + 0.5 * ties, atol=1e-9)
