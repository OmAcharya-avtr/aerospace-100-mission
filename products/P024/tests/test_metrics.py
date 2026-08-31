"""Confidence intervals."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from detumblesim.metrics import format_interval, mean_ci, paired_difference_ci


class TestMeanCI:
    def test_known_answer(self):
        # values 1..5: mean 3, s = sqrt(2.5) = 1.5811388, n = 5,
        # t(0.975, 4) = 2.7764451; half-width = 2.7764451*1.5811388/sqrt(5)
        #             = 1.9632432
        i = mean_ci([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.isclose(i.mean, 3.0)
        assert np.isclose(i.std, np.sqrt(2.5))
        assert np.isclose(i.half_width, 1.9632432, atol=1e-6)
        assert np.isclose(i.ci_low, 3.0 - 1.9632432, atol=1e-6)

    def test_single_sample_collapses(self):
        i = mean_ci([4.2])
        assert i.mean == i.ci_low == i.ci_high == 4.2
        assert np.isnan(i.std)
        assert i.n == 1

    def test_zero_variance_collapses(self):
        i = mean_ci([2.0, 2.0, 2.0])
        assert i.ci_low == i.ci_high == 2.0
        assert i.std == 0.0

    def test_excludes_zero(self):
        assert mean_ci([1.0, 1.1, 0.9, 1.05]).excludes_zero
        assert not mean_ci([-1.0, 1.0, 0.1, -0.2]).excludes_zero

    @pytest.mark.parametrize("bad", [[], [1.0, np.nan]])
    def test_rejects_bad_values(self, bad):
        with pytest.raises(ValueError):
            mean_ci(bad)

    @pytest.mark.parametrize("lvl", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_bad_level(self, lvl):
        with pytest.raises(ValueError, match="ci_level"):
            mean_ci([1.0, 2.0], ci_level=lvl)

    @given(
        vals=st.lists(st.floats(-100.0, 100.0), min_size=2, max_size=40),
        lvl=st.floats(0.5, 0.99),
    )
    @settings(max_examples=60, deadline=None)
    def test_interval_brackets_the_mean(self, vals, lvl):
        i = mean_ci(vals, ci_level=lvl)
        assert i.ci_low <= i.mean + 1e-9
        assert i.ci_high >= i.mean - 1e-9
        assert i.n == len(vals)

    def test_wider_level_gives_wider_interval(self):
        v = [1.0, 3.0, 2.0, 5.0, 4.0, 2.5]
        assert mean_ci(v, 0.99).half_width > mean_ci(v, 0.90).half_width

    def test_matches_scipy_t_quantile(self):
        v = np.array([0.3, 1.2, -0.5, 2.2, 0.9])
        i = mean_ci(v, 0.95)
        expected = float(stats.t.ppf(0.975, 4)) * v.std(ddof=1) / np.sqrt(5)
        assert np.isclose(i.half_width, expected)


class TestPairedCI:
    def test_known_answer(self):
        # a - b = (-0.1, -0.2, -0.3); mean -0.2, s = 0.1, n = 3,
        # t(0.975, 2) = 4.302653; half-width = 4.302653*0.1/sqrt(3) = 0.248414
        i = paired_difference_ci([1.0, 2.0, 3.0], [1.1, 2.2, 3.3])
        assert np.isclose(i.mean, -0.2)
        assert np.isclose(i.half_width, 0.2484137, atol=1e-6)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="equal length"):
            paired_difference_ci([1.0, 2.0], [1.0])

    def test_pairing_narrows_the_interval_for_correlated_samples(self):
        rng = np.random.default_rng(0)
        common = rng.normal(0.0, 10.0, 30)
        a = common + rng.normal(0.0, 0.1, 30)
        b = common + 1.0 + rng.normal(0.0, 0.1, 30)
        paired = paired_difference_ci(a, b)
        unpaired_hw = np.hypot(mean_ci(a).half_width, mean_ci(b).half_width)
        assert paired.half_width < 0.1 * unpaired_hw
        assert paired.excludes_zero


class TestFormatting:
    def test_format(self):
        s = format_interval(mean_ci([1.0, 2.0, 3.0]), unit="s", digits=2)
        assert s.startswith("2.00 [")
        assert s.endswith(" s")
