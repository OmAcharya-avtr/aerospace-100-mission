"""Chi-squared and CUSUM detectors: hand-checked statistics and validation."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fdiscope.analytic import chi2_threshold
from fdiscope.detectors import (
    ChiSquaredDetector,
    CusumBank,
    CusumDetector,
    detection_delay,
    first_alarm_index,
)


def unit_residual(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, 2))


class TestChiSquaredDetector:
    def test_known_answer_statistic(self):
        # residual rows (1, 0), (0, 2), (3, 4) -> NIS 1, 4, 25.
        # window 2: statistic at index 1 is 1 + 4 = 5, at index 2 is 4 + 25 = 29.
        det = ChiSquaredDetector(window=2, dim=2, alpha=0.5)
        out = det.run([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
        assert np.isnan(out.statistic[0])
        assert np.isclose(out.statistic[1], 5.0)
        assert np.isclose(out.statistic[2], 29.0)

    def test_threshold_matches_the_design_formula(self):
        det = ChiSquaredDetector(window=25, dim=2, alpha=1e-3)
        assert det.dof == 50
        assert np.isclose(det.threshold, chi2_threshold(1e-3, 50))

    def test_leading_samples_never_alarm(self):
        det = ChiSquaredDetector(window=5, dim=2, alpha=0.5)
        out = det.run(1e6 * np.ones((10, 2)))
        assert not out.alarm[:4].any()
        assert out.alarm[4:].all()

    def test_alarm_fraction_ignores_nan_statistics(self):
        det = ChiSquaredDetector(window=5, dim=2, alpha=0.5)
        out = det.run(1e6 * np.ones((10, 2)))
        assert np.isclose(out.alarm_fraction, 1.0)

    def test_shorter_than_window_gives_all_nan(self):
        out = ChiSquaredDetector(window=50, dim=2, alpha=0.5).run(unit_residual(10))
        assert np.all(np.isnan(out.statistic))
        assert not out.alarm.any()
        assert np.isnan(out.alarm_fraction)

    @pytest.mark.parametrize("kwargs", [{"window": 0}, {"dim": 0}, {"alpha": 0.0}, {"alpha": 1.0}])
    def test_rejects_bad_parameters(self, kwargs):
        with pytest.raises(ValueError):
            ChiSquaredDetector(**kwargs)

    def test_rejects_wrong_residual_width(self):
        with pytest.raises(ValueError, match="expected 2"):
            ChiSquaredDetector(window=2, dim=2).run(np.zeros((5, 3)))

    def test_rejects_non_finite_residual(self):
        with pytest.raises(ValueError, match="finite"):
            ChiSquaredDetector(window=2, dim=2).run([[1.0, np.nan], [0.0, 0.0]])

    @settings(max_examples=30, deadline=None)
    @given(scale=st.floats(1.0, 20.0, allow_nan=False))
    def test_statistic_is_monotone_in_residual_scale(self, scale):
        det = ChiSquaredDetector(window=4, dim=2, alpha=0.1)
        r = unit_residual(20, seed=3)
        a = det.run(r).statistic[3:]
        b = det.run(scale * r).statistic[3:]
        assert np.all(b >= a - 1e-9)


class TestCusumDetector:
    def test_known_answer_increments(self):
        # mu = 2, direction e0: s_k = 2 * r_k0 - 2.
        # r = (1, 0), (2, 0), (0, 0) -> s = 0, 2, -2
        det = CusumDetector(direction=[1.0, 0.0], mu=2.0, threshold=1.0)
        assert np.allclose(det.increments([[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]]), [0.0, 2.0, -2.0])

    def test_known_answer_statistic(self):
        # continuing the example: g = max(0, 0+0)=0, max(0, 0+2)=2, max(0, 2-2)=0
        det = CusumDetector(direction=[1.0, 0.0], mu=2.0, threshold=1.0)
        out = det.run([[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]])
        assert np.allclose(out.statistic, [0.0, 2.0, 0.0])
        assert list(out.alarm) == [False, True, False]

    def test_statistic_never_negative(self):
        det = CusumDetector(direction=[0.0, 1.0], mu=1.0, threshold=5.0)
        assert np.all(det.run(unit_residual(500, 1)).statistic >= 0.0)

    def test_closed_form_matches_the_explicit_recursion(self):
        det = CusumDetector(direction=[1.0, 1.0], mu=0.7, threshold=1e9)
        r = unit_residual(400, 5)
        fast = det.run(r).statistic
        incr = det.increments(r)
        slow = np.empty(incr.size)
        acc = 0.0
        for i, s in enumerate(incr):
            acc = max(0.0, acc + s)
            slow[i] = acc
        assert np.allclose(fast, slow, atol=1e-12)

    def test_reset_on_alarm_zeroes_after_a_crossing(self):
        # constant projection 5 with mu = 2: increment 8 per sample, h = 10.
        # g = 8, 16 -> alarm, reset -> 8, 16 -> alarm ...
        r = np.zeros((6, 2))
        r[:, 0] = 5.0
        det = CusumDetector(direction=[1.0, 0.0], mu=2.0, threshold=10.0)
        out = det.run(r, reset_on_alarm=True)
        assert np.allclose(out.statistic, [8.0, 16.0, 8.0, 16.0, 8.0, 16.0])
        assert int(np.count_nonzero(out.alarm)) == 3

    def test_direction_is_normalised(self):
        det = CusumDetector(direction=[3.0, 4.0], mu=1.0, threshold=1.0)
        assert np.isclose(np.linalg.norm(det.direction), 1.0)
        assert np.allclose(det.direction, [0.6, 0.8])

    def test_projection_known_answer(self):
        det = CusumDetector(direction=[3.0, 4.0], mu=1.0, threshold=1.0)
        # (1, 1) . (0.6, 0.8) = 1.4
        assert np.allclose(det.project([[1.0, 1.0]]), [1.4])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"direction": [0.0, 0.0]},
            {"mu": 0.0},
            {"mu": -1.0},
            {"threshold": 0.0},
            {"direction": [np.nan, 1.0]},
        ],
    )
    def test_rejects_bad_parameters(self, kwargs):
        base = {"direction": [1.0, 0.0], "mu": 1.0, "threshold": 1.0}
        base.update(kwargs)
        with pytest.raises(ValueError):
            CusumDetector(**base)

    def test_rejects_mismatched_residual_width(self):
        det = CusumDetector(direction=[1.0, 0.0, 0.0], mu=1.0, threshold=1.0)
        with pytest.raises(ValueError, match="direction has 3"):
            det.project(np.zeros((4, 2)))

    @settings(max_examples=30, deadline=None)
    @given(mu=st.floats(0.1, 3.0, allow_nan=False))
    def test_statistic_is_invariant_to_direction_scaling(self, mu):
        r = unit_residual(200, 7)
        a = CusumDetector(direction=[1.0, 2.0], mu=mu, threshold=1.0).run(r).statistic
        b = CusumDetector(direction=[5.0, 10.0], mu=mu, threshold=1.0).run(r).statistic
        assert np.allclose(a, b)


class TestCusumBank:
    def make(self, threshold: float = 5.0) -> CusumBank:
        return CusumBank(
            detectors={
                "pos": CusumDetector(direction=[1.0, 0.0], mu=1.0, threshold=threshold),
                "neg": CusumDetector(direction=[-1.0, 0.0], mu=1.0, threshold=threshold),
            }
        )

    def test_statistics_shape_and_names(self):
        bank = self.make()
        stats = bank.statistics(unit_residual(50, 2))
        assert stats.shape == (50, 2)
        assert bank.names == ("pos", "neg")

    def test_max_over_bank_is_the_detector_statistic(self):
        bank = self.make()
        r = unit_residual(80, 3)
        assert np.allclose(bank.run(r).statistic, np.max(bank.statistics(r), axis=1))

    def test_isolate_returns_the_argmax(self):
        bank = self.make()
        r = unit_residual(80, 4)
        idx, value = bank.isolate(r)
        stats = bank.statistics(r)
        assert np.allclose(value, stats[np.arange(80), idx])

    def test_a_persistent_positive_shift_fires_the_positive_member(self):
        r = np.zeros((60, 2))
        r[:, 0] = 3.0
        bank = self.make(threshold=5.0)
        idx, _ = bank.isolate(r)
        assert idx[-1] == 0

    def test_run_lengths_are_positive_and_sum_below_the_sample_count(self):
        bank = self.make(threshold=4.0)
        lengths = bank.run_lengths(unit_residual(4000, 11))
        assert np.all(lengths >= 1)
        assert int(lengths.sum()) <= 4000

    def test_rejects_empty_bank(self):
        with pytest.raises(ValueError, match="at least one detector"):
            CusumBank(detectors={})

    def test_rejects_wrong_member_type(self):
        with pytest.raises(TypeError, match="not a CusumDetector"):
            CusumBank(detectors={"x": "not a detector"})


class TestAlarmTiming:
    def test_first_alarm_index_known_answer(self):
        alarm = [False, True, False, True, True, True]
        assert first_alarm_index(alarm) == 1
        assert first_alarm_index(alarm, start=2) == 3
        assert first_alarm_index(alarm, persistence=2) == 3
        assert first_alarm_index(alarm, persistence=3) == 3
        assert first_alarm_index(alarm, persistence=4) == -1

    def test_detection_delay_known_answer(self):
        # onset 2, first alarm at index 5 -> delay 3 samples.
        alarm = [True, True, False, False, False, True]
        assert detection_delay(alarm, 2) == 3.0

    def test_alarms_before_onset_are_not_counted_as_detections(self):
        alarm = [True, True, False, False]
        assert detection_delay(alarm, 2) == float("inf")

    def test_alarm_on_the_onset_sample_is_zero_delay(self):
        assert detection_delay([False, True, True], 1) == 0.0

    def test_rejects_bad_persistence(self):
        with pytest.raises(ValueError, match="persistence"):
            first_alarm_index([True], persistence=0)
