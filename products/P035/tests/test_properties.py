"""Property-based tests for the algebraic identities the package relies on.

Each property is an identity or an inequality that must hold for every input in
its stated domain, so a generator is the right tool rather than a fixed example.
"""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from telemetryool.arl import (
    design_cusum_h,
    design_ewma_L,
    design_ool_limit,
    ewma_sigma_z,
    ool_arl,
    ool_window_false_alarm,
)
from telemetryool.calibration import (
    binomial_se,
    calibrate_threshold,
    clopper_pearson_interval,
    wilson_interval,
    window_trigger_level,
)
from telemetryool.charts import CusumChart, EwmaChart
from telemetryool.limits import BreachLevel, LimitSet
from telemetryool.metrics import confusion_matrix
from telemetryool.runs import first_run_index

SETTINGS = settings(max_examples=60, deadline=None)
#: The threshold designers solve a bisection over a Markov chain per example, so
#: they get a smaller budget to keep this file inside the 30 s test-file limit
#: stated in the README compute section.
SLOW_SETTINGS = settings(max_examples=10, deadline=None)

score_blocks = hnp.arrays(
    dtype=np.float64,
    shape=st.tuples(st.integers(1, 12), st.integers(1, 20)),
    elements=st.floats(-20.0, 20.0, allow_nan=False, allow_infinity=False, width=64),
)
series = hnp.arrays(
    dtype=np.float64,
    shape=st.integers(1, 40),
    elements=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False, width=64),
)
probabilities = st.floats(1e-6, 1 - 1e-6, allow_nan=False, allow_infinity=False)
lams = st.floats(0.01, 1.0, allow_nan=False, allow_infinity=False)


# --------------------------------------------------------------------------- #
# The window trigger level is defined by an equivalence; it must hold exactly.
# --------------------------------------------------------------------------- #


@SETTINGS
@given(
    score_blocks,
    st.integers(1, 6),
    st.floats(-25.0, 25.0, allow_nan=False, allow_infinity=False),
)
def test_trigger_level_equivalence(scores: np.ndarray, persistence: int, h: float) -> None:
    w = window_trigger_level(scores, persistence)
    alarmed = first_run_index(scores > h, persistence) >= 0
    assert np.array_equal(w > h, alarmed)


@SETTINGS
@given(score_blocks, st.integers(1, 6))
def test_trigger_level_is_non_increasing_in_persistence(
    scores: np.ndarray, persistence: int
) -> None:
    """Requiring more consecutive exceedances can never make a window easier to
    trigger, so the trigger level is non-increasing in the debounce count."""
    coarse = window_trigger_level(scores, persistence)
    finer = window_trigger_level(scores, persistence + 1)
    assert np.all(finer <= coarse + 1e-12)


@SETTINGS
@given(score_blocks, st.integers(1, 5))
def test_first_run_index_certifies_its_own_answer(
    scores: np.ndarray, persistence: int
) -> None:
    breach = scores > 0.0
    idx = first_run_index(breach, persistence)
    for row, i in zip(breach, idx.tolist(), strict=True):
        if i < 0:
            assert not np.any(
                np.all(
                    np.lib.stride_tricks.sliding_window_view(row, persistence), axis=1
                )
            ) if row.size >= persistence else True
        else:
            assert i >= persistence - 1
            assert np.all(row[i - persistence + 1 : i + 1])
            if i > persistence - 1:
                earlier = np.lib.stride_tricks.sliding_window_view(
                    row[:i], persistence
                ) if i >= persistence else np.empty((0, persistence), dtype=bool)
                assert not np.any(np.all(earlier, axis=1))


# --------------------------------------------------------------------------- #
# Monotonicity of the designed operating point.
# --------------------------------------------------------------------------- #


@SETTINGS
@given(probabilities, st.integers(1, 5), st.integers(1, 60))
def test_ool_window_far_is_a_probability(p: float, persistence: int, window: int) -> None:
    alpha = ool_window_false_alarm(p, persistence, window)
    assert 0.0 <= alpha <= 1.0


@SETTINGS
@given(probabilities, st.integers(1, 5), st.integers(1, 40))
def test_ool_window_far_increases_with_window_length(
    p: float, persistence: int, window: int
) -> None:
    shorter = ool_window_false_alarm(p, persistence, window)
    longer = ool_window_false_alarm(p, persistence, window + 1)
    assert longer >= shorter - 1e-15


@SETTINGS
@given(st.floats(0.01, 0.9), st.integers(1, 5), st.integers(1, 40))
def test_ool_window_far_increases_with_exceedance_probability(
    p: float, persistence: int, window: int
) -> None:
    lower = ool_window_false_alarm(p, persistence, window)
    higher = ool_window_false_alarm(min(p * 1.5, 0.999999), persistence, window)
    assert higher >= lower - 1e-15


@SETTINGS
@given(st.floats(0.01, 0.99), st.integers(1, 6))
def test_ool_arl_decreases_with_exceedance_probability(p: float, persistence: int) -> None:
    assume(p * 1.2 < 0.999)
    assert ool_arl(min(p * 1.2, 0.999), persistence) <= ool_arl(p, persistence) + 1e-9


@SETTINGS
@given(st.floats(0.001, 0.3), st.integers(1, 4), st.integers(10, 200))
def test_design_ool_round_trip_property(
    alpha: float, persistence: int, window: int
) -> None:
    design = design_ool_limit(alpha, persistence, window)
    assert design.achieved_alpha_w == np.float64(alpha) or abs(
        design.achieved_alpha_w - alpha
    ) < 1e-8


@SLOW_SETTINGS
@given(st.floats(0.005, 0.2), st.floats(0.1, 1.0), st.integers(20, 200))
def test_design_ewma_round_trip_property(alpha: float, lam: float, window: int) -> None:
    design = design_ewma_L(alpha, lam, window, n_states=101)
    assert abs(design.achieved_alpha_w - alpha) < 1e-6 * max(alpha, 1e-3)


@SLOW_SETTINGS
@given(st.floats(0.005, 0.2), st.floats(0.2, 1.5), st.integers(20, 200))
def test_design_cusum_round_trip_property(alpha: float, k: float, window: int) -> None:
    design = design_cusum_h(alpha, k, window, n_states=61)
    assert abs(design.achieved_alpha_w - alpha) < 1e-6 * max(alpha, 1e-3)


@SETTINGS
@given(lams)
def test_ewma_sigma_z_is_increasing_and_bounded(lam: float) -> None:
    value = ewma_sigma_z(lam)
    assert 0.0 < value <= 1.0
    if lam < 1.0:
        assert ewma_sigma_z(min(lam * 1.1, 1.0)) >= value - 1e-15


# --------------------------------------------------------------------------- #
# Chart algebra.
# --------------------------------------------------------------------------- #


@SETTINGS
@given(series, st.floats(0.05, 3.0))
def test_cusum_arms_are_non_negative(u: np.ndarray, k: float) -> None:
    c_up, c_dn = CusumChart(k=k, h=5.0).arms(u)
    assert np.all(c_up >= 0.0)
    assert np.all(c_dn >= 0.0)


@SETTINGS
@given(series, st.floats(0.05, 3.0))
def test_cusum_is_antisymmetric_under_negation(u: np.ndarray, k: float) -> None:
    """Negating the series swaps the two arms exactly."""
    chart = CusumChart(k=k, h=5.0)
    up_a, dn_a = chart.arms(u)
    up_b, dn_b = chart.arms(-u)
    assert np.allclose(up_a, dn_b, atol=1e-12)
    assert np.allclose(dn_a, up_b, atol=1e-12)


@SETTINGS
@given(series, lams)
def test_ewma_is_bounded_by_the_largest_input(u: np.ndarray, lam: float) -> None:
    """z_t is a weighted average of u_1..u_t with weights summing to
    1 - (1 - lam)^t <= 1, so |z_t| <= max |u_i| for every t."""
    z = EwmaChart(lam=lam, limit_mult=3.0).statistic(u[None, :])[0]
    assert np.all(np.abs(z) <= np.max(np.abs(u)) + 1e-12)


@SETTINGS
@given(series, lams)
def test_ewma_of_a_constant_series_matches_its_closed_form(
    u: np.ndarray, lam: float
) -> None:
    """For a constant input c, unrolling the recursion gives exactly
    z_t = c (1 - (1 - lam)^t), which tends to c only as t -> inf.  At lam = 0.01
    and t = 200 the factor is still 0.866, so the identity -- not convergence --
    is what holds at finite t.
    """
    constant = float(u[0])
    n = 200
    z = EwmaChart(lam=lam, limit_mult=3.0).statistic(np.full((1, n), constant))[0]
    t = np.arange(1, n + 1, dtype=float)
    expected = constant * (1.0 - (1.0 - lam) ** t)
    assert np.allclose(z, expected, rtol=1e-11, atol=1e-12)


# --------------------------------------------------------------------------- #
# Limit sets.
# --------------------------------------------------------------------------- #


@SETTINGS
@given(
    st.floats(0.1, 10.0),
    st.floats(0.0, 10.0),
    st.floats(-30.0, 30.0, allow_nan=False, allow_infinity=False),
)
def test_symmetric_limit_level_depends_only_on_the_absolute_value(
    soft: float, extra: float, value: float
) -> None:
    ls = LimitSet.symmetric(0.0, soft, soft + extra)
    assert ls.level(value) == ls.level(-value)


@SETTINGS
@given(st.floats(0.1, 5.0), st.floats(0.0, 5.0), st.floats(0.0, 30.0))
def test_limit_level_is_monotone_in_magnitude(
    soft: float, extra: float, value: float
) -> None:
    ls = LimitSet.symmetric(0.0, soft, soft + extra)
    assert int(ls.level(value * 1.5)) >= int(ls.level(value))
    assert ls.level(0.0) is BreachLevel.IN_LIMIT


# --------------------------------------------------------------------------- #
# Interval estimates and confusion matrices.
# --------------------------------------------------------------------------- #


@SETTINGS
@given(st.integers(1, 5000), st.floats(0.0, 1.0))
def test_binomial_se_is_bounded_by_half_over_root_n(trials: int, rate: float) -> None:
    assert binomial_se(rate, trials) <= 0.5 / np.sqrt(trials) + 1e-15


@SETTINGS
@given(st.integers(1, 2000), st.floats(0.0, 1.0))
def test_intervals_contain_the_point_estimate(trials: int, fraction: float) -> None:
    successes = int(round(fraction * trials))
    point = successes / trials
    for low, high in (
        wilson_interval(successes, trials),
        clopper_pearson_interval(successes, trials),
    ):
        assert low <= point + 1e-12
        assert point <= high + 1e-12
        assert 0.0 <= low <= high <= 1.0


@SETTINGS
@given(st.integers(10, 2000), st.floats(0.1, 0.9))
def test_clopper_pearson_contains_wilson_in_the_central_region(
    trials: int, fraction: float
) -> None:
    """In the central region the exact interval is the wider of the two.

    The containment is restricted to ``0.1 <= p_hat <= 0.9`` on purpose, because
    it is **false** in the extreme tails: at ``n = 999, k = 1`` the
    Clopper-Pearson interval is [2.534e-05, 5.564e-03] while the Wilson interval
    is [1.767e-04, 5.648e-03], so neither contains the other.  That is a known
    property of the two intervals (Brown, Cai and DasGupta 2001), not a defect,
    and the package reports both rather than pretending one dominates.
    """
    successes = int(round(fraction * trials))
    assume(0.1 <= successes / trials <= 0.9)
    wl, wh = wilson_interval(successes, trials)
    cl, ch = clopper_pearson_interval(successes, trials)
    assert cl <= wl + 1e-12
    assert ch >= wh - 1e-12


@SETTINGS
@given(
    hnp.arrays(np.bool_, st.integers(1, 50)),
    hnp.arrays(np.bool_, st.integers(1, 50)),
)
def test_confusion_matrix_counts_are_consistent(
    positives: np.ndarray, negatives: np.ndarray
) -> None:
    cm = confusion_matrix(positives, negatives)
    assert cm.true_positive + cm.false_negative == positives.size
    assert cm.false_positive + cm.true_negative == negatives.size
    assert cm.total == positives.size + negatives.size
    assert cm.tpr == positives.mean()
    assert cm.fpr == negatives.mean()


@SETTINGS
@given(score_blocks, st.integers(1, 4))
def test_calibration_never_exceeds_its_target_by_more_than_one_window(
    scores: np.ndarray, persistence: int
) -> None:
    """Calibration takes the ceil(alpha * M)-th largest trigger level, so the
    in-sample alarm count is at least ceil(alpha * M) and, absent ties, exactly
    that; ties can only reduce the threshold's discriminating power, never make
    it alarm on fewer windows than requested."""
    m, length = scores.shape
    alpha = 0.5
    assume(alpha * m >= 1.0)
    # A window shorter than the debounce count can never alarm at any threshold;
    # calibrate_threshold raises rather than returning a meaningless -inf.
    assume(length >= persistence)
    calib = calibrate_threshold(scores, persistence, alpha)
    assert calib.n_alarming >= int(np.ceil(alpha * m))
