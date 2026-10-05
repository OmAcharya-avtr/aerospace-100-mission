"""Hand-computed EWMA and CUSUM recursions, plus the designed-threshold link."""

from __future__ import annotations

import numpy as np
import pytest

from telemetryool.arl import design_cusum_h, design_ewma_L, ewma_sigma_z
from telemetryool.charts import CusumChart, EwmaChart


def test_ewma_recursion_hand_values() -> None:
    """lam = 0.5, u = [2, 0, 0, 0], z_0 = 0:

    z_1 = 0.5 * 2                 = 1.0
    z_2 = 0.5 * 0 + 0.5 * 1.0     = 0.5
    z_3 = 0.5 * 0 + 0.5 * 0.5     = 0.25
    z_4 = 0.5 * 0 + 0.5 * 0.25    = 0.125
    """
    chart = EwmaChart(lam=0.5, limit_mult=3.0)
    z = chart.run(np.array([2.0, 0.0, 0.0, 0.0])).statistic
    assert z.tolist() == [1.0, 0.5, 0.25, 0.125]


def test_ewma_steady_state_limits_are_constant() -> None:
    chart = EwmaChart(lam=0.2, limit_mult=3.0)
    lim = chart.limits(5)
    assert np.allclose(lim, 3.0 * ewma_sigma_z(0.2))
    assert chart.sigma_z == pytest.approx(np.sqrt(0.2 / 1.8), abs=1e-15)


def test_ewma_time_varying_limits_hand_values() -> None:
    """lam = 0.5, L = 3: the limit at sample t (1-based) is
    3 * sqrt(1/3) * sqrt(1 - 0.5**(2 t)), so

    t = 1 -> 3 * sqrt(1/3) * sqrt(0.75)
    t = 2 -> 3 * sqrt(1/3) * sqrt(0.9375)
    and the limits increase towards the steady-state value.
    """
    chart = EwmaChart(lam=0.5, limit_mult=3.0, time_varying_limits=True)
    lim = chart.limits(3)
    base = 3.0 * np.sqrt(1.0 / 3.0)
    assert lim[0] == pytest.approx(base * np.sqrt(1.0 - 0.5**2), abs=1e-14)
    assert lim[1] == pytest.approx(base * np.sqrt(1.0 - 0.5**4), abs=1e-14)
    assert lim[0] < lim[1] < lim[2] < base


def test_cusum_recursion_hand_values() -> None:
    """k = 0.5, u = [1, 1, 1, -3]:

    C+  max(0, 0 + 1 - 0.5) = 0.5
        max(0, 0.5 + 1 - 0.5) = 1.0
        max(0, 1.0 + 1 - 0.5) = 1.5
        max(0, 1.5 - 3 - 0.5) = 0.0
    C-  max(0, 0 - 1 - 0.5) = 0.0
        0.0, 0.0
        max(0, 0 + 3 - 0.5) = 2.5
    """
    chart = CusumChart(k=0.5, h=5.0)
    run = chart.run(np.array([1.0, 1.0, 1.0, -3.0]))
    assert run.statistic[0].tolist() == [0.5, 1.0, 1.5, 0.0]
    assert run.statistic[1].tolist() == [0.0, 0.0, 0.0, 2.5]


def test_cusum_breach_and_alarm_index() -> None:
    """k = 0.5, h = 1.2, u = [1, 1, 1]: C+ = 0.5, 1.0, 1.5, so only the third
    sample exceeds h and the alarm index is 2 with persistence 1."""
    chart = CusumChart(k=0.5, h=1.2)
    run = chart.run(np.array([1.0, 1.0, 1.0]))
    assert run.breach.tolist() == [False, False, True]
    assert run.alarm_index == 2


def test_cusum_persistence_delays_the_alarm() -> None:
    """Same series extended: C+ = 0.5, 1.0, 1.5, 2.0 breaches at indices 2 and 3,
    so persistence 2 raises at index 3 rather than 2."""
    chart = CusumChart(k=0.5, h=1.2, persistence=2)
    run = chart.run(np.array([1.0, 1.0, 1.0, 1.0]))
    assert run.breach.tolist() == [False, False, True, True]
    assert run.alarm_index == 3


def test_no_alarm_returns_minus_one() -> None:
    chart = CusumChart(k=0.5, h=50.0)
    assert chart.run(np.zeros(20)).alarm_index == -1


def test_ewma_score_sign_symmetry() -> None:
    """The chart is two-sided, so negating the series leaves |z| unchanged."""
    chart = EwmaChart(lam=0.3, limit_mult=3.0)
    u = np.array([0.4, -1.2, 2.0, -0.7, 0.1])
    assert np.allclose(np.abs(chart.run(u).statistic), np.abs(chart.run(-u).statistic))


def test_run_windows_matches_per_row_run() -> None:
    rng = np.random.default_rng(11)
    block = rng.standard_normal((40, 60))
    for chart in (EwmaChart(lam=0.2, limit_mult=1.5), CusumChart(k=0.5, h=2.0)):
        batch = chart.run_windows(block)
        single = [chart.run(block[i]).alarm_index for i in range(block.shape[0])]
        assert batch.tolist() == single


def test_designed_cusum_threshold_feeds_the_chart() -> None:
    design = design_cusum_h(0.05, 0.5, 100)
    chart = CusumChart(k=0.5, h=design.threshold)
    assert chart.h == design.threshold
    rng = np.random.default_rng(3)
    alarms = chart.run_windows(rng.standard_normal((4000, 100))) >= 0
    # 4000 windows: binomial SE at alpha = 0.05 is 0.00345, so a 4-sigma band is
    # [0.036, 0.064].  This is a smoke check that the design is in the right
    # place, not the measurement -- that is validate_far_design.py.
    assert 0.036 < alarms.mean() < 0.064


def test_designed_ewma_threshold_feeds_the_chart() -> None:
    design = design_ewma_L(0.05, 0.2, 100)
    chart = EwmaChart(lam=0.2, limit_mult=design.threshold)
    rng = np.random.default_rng(4)
    alarms = chart.run_windows(rng.standard_normal((4000, 100))) >= 0
    assert 0.036 < alarms.mean() < 0.064


def test_chart_parameter_validation() -> None:
    with pytest.raises(ValueError, match=r"lam must lie in \(0, 1\]"):
        EwmaChart(lam=0.0, limit_mult=3.0)
    with pytest.raises(ValueError, match="limit_mult must be > 0"):
        EwmaChart(lam=0.2, limit_mult=0.0)
    with pytest.raises(ValueError, match="persistence must be an integer >= 1"):
        EwmaChart(lam=0.2, limit_mult=3.0, persistence=0)
    with pytest.raises(ValueError, match="k must be > 0"):
        CusumChart(k=0.0, h=5.0)
    with pytest.raises(ValueError, match="h must be > 0"):
        CusumChart(k=0.5, h=-1.0)
    with pytest.raises(ValueError, match="persistence must be an integer >= 1"):
        CusumChart(k=0.5, h=5.0, persistence=0)


def test_input_shape_validation() -> None:
    chart = CusumChart(k=0.5, h=5.0)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        chart.run_windows(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="is empty"):
        chart.run_windows(np.zeros((0, 5)))
    with pytest.raises(ValueError, match="takes a single sequence"):
        chart.run(np.zeros((3, 5)))
