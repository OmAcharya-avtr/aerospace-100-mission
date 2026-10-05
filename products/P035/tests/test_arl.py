"""Known-answer tests for the ARL and window-false-alarm machinery.

Each test states where its expected value comes from: a closed form derived in
the test comment, an exhaustive enumeration, or a limiting case where two
independent derivations must agree.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.stats import norm

from telemetryool import arl

# --------------------------------------------------------------------------- #
# Out-of-limit persistence chain: exact, so it is checked against exhaustive
# enumeration over every pattern of W Bernoulli samples.
# --------------------------------------------------------------------------- #


def _brute_force_window_far(p: float, persistence: int, window: int) -> float:
    """Exhaustive enumeration of all 2**window breach patterns."""
    total = 0.0
    for pattern in itertools.product([0, 1], repeat=window):
        prob = 1.0
        for bit in pattern:
            prob *= p if bit else (1.0 - p)
        run = 0
        for bit in pattern:
            run = run + 1 if bit else 0
            if run >= persistence:
                total += prob
                break
    return total


@pytest.mark.parametrize(
    ("p", "persistence", "window"),
    [(0.5, 2, 2), (0.5, 2, 3), (0.3, 3, 8), (0.1, 2, 10), (0.25, 1, 5), (0.4, 4, 9)],
)
def test_ool_window_far_matches_exhaustive_enumeration(
    p: float, persistence: int, window: int
) -> None:
    expected = _brute_force_window_far(p, persistence, window)
    got = arl.ool_window_false_alarm(p, persistence, window)
    assert got == pytest.approx(expected, rel=0, abs=1e-14)


def test_ool_window_far_hand_values() -> None:
    """Two runs of 2 heads with p = 1/2.

    W = 2: only HH qualifies, probability 1/4.
    W = 3: NHH, HHN, HHH qualify, 3 * 1/8 = 3/8.
    """
    assert arl.ool_window_false_alarm(0.5, 2, 2) == pytest.approx(0.25, abs=1e-15)
    assert arl.ool_window_false_alarm(0.5, 2, 3) == pytest.approx(0.375, abs=1e-15)


def test_ool_arl_hand_value() -> None:
    """Expected tosses to see 2 consecutive heads with a fair coin is 6.

    The closed form (1 - p**r) / (p**r (1 - p)) gives (1 - 1/4) / (1/4 * 1/2) = 6.
    """
    assert arl.ool_arl(0.5, 2) == pytest.approx(6.0, abs=1e-12)
    # r = 1 reduces to the geometric mean 1 / p.
    assert arl.ool_arl(0.25, 1) == pytest.approx(4.0, abs=1e-12)


def test_design_ool_limit_single_sample_window_is_the_normal_quantile() -> None:
    """With persistence = 1 and W = 1 the window FAR is just the per-sample tail,
    so the designed two-sided limit must be the exact normal quantile
    Phi^-1(1 - alpha / 2) = 1.959963984540054 at alpha = 0.05."""
    design = arl.design_ool_limit(0.05, persistence=1, window_length=1)
    assert design.threshold == pytest.approx(float(norm.ppf(0.975)), abs=1e-12)
    assert design.achieved_alpha_w == pytest.approx(0.05, abs=1e-12)


def test_design_ool_limit_one_sided_quantile() -> None:
    design = arl.design_ool_limit(0.05, 1, 1, two_sided=False)
    assert design.threshold == pytest.approx(float(norm.ppf(0.95)), abs=1e-12)


# --------------------------------------------------------------------------- #
# EWMA Markov chain: at lam = 1 the EWMA statistic is the raw deviate, so the
# chart is a Shewhart chart with ARL0 = 1 / (2 (1 - Phi(L))), an exact value the
# discretisation must reproduce.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("limit_mult", [2.0, 2.5, 3.0, 3.5])
def test_ewma_markov_reduces_to_shewhart_at_lam_one(limit_mult: float) -> None:
    exact = 1.0 / (2.0 * (1.0 - float(norm.cdf(limit_mult))))
    got = arl.ewma_arl_markov(0.0, 1.0, limit_mult, 801)
    assert got == pytest.approx(exact, rel=1e-10)


def test_ewma_sigma_z_known_values() -> None:
    """sigma_z = sqrt(lam / (2 - lam)): 1 at lam = 1, sqrt(1/3) at lam = 0.5."""
    assert arl.ewma_sigma_z(1.0) == pytest.approx(1.0, abs=1e-15)
    assert arl.ewma_sigma_z(0.5) == pytest.approx(np.sqrt(1.0 / 3.0), abs=1e-15)
    assert arl.ewma_sigma_z(0.2) == pytest.approx(np.sqrt(0.2 / 1.8), abs=1e-15)


def test_ewma_window_far_at_lam_one_is_the_independent_product() -> None:
    """At lam = 1 successive samples are independent, so
    alpha_W = 1 - (1 - p)**W with p = 2 (1 - Phi(L))."""
    limit_mult, window = 3.0, 25
    p = 2.0 * (1.0 - float(norm.cdf(limit_mult)))
    expected = 1.0 - (1.0 - p) ** window
    got = arl.ewma_window_false_alarm(1.0, limit_mult, window, n_states=801)
    assert got == pytest.approx(expected, rel=1e-8)


def test_ewma_arl_decreases_with_shift() -> None:
    base = arl.ewma_arl_markov(0.0, 0.2, 3.0, 301)
    for delta in (0.5, 1.0, 2.0):
        assert arl.ewma_arl_markov(delta, 0.2, 3.0, 301) < base


# --------------------------------------------------------------------------- #
# CUSUM: the Siegmund closed form and the Brook-Evans chain are independent
# derivations, so their agreement is the test.
# --------------------------------------------------------------------------- #


def test_siegmund_in_control_limit_is_b_squared() -> None:
    """At delta = k the numerator and denominator both vanish; the limit of
    (exp(-2 d b) + 2 d b - 1) / (2 d^2) as d -> 0 is b^2 with b = h + 1.166."""
    h, k = 5.0, 0.5
    b = h + arl.SIEGMUND_B_OFFSET
    assert arl.cusum_arl_siegmund(k, k, h) == pytest.approx(b * b, abs=1e-12)
    # It is continuous there.  The series branch covers |2 d b| < 1e-3, so
    # d = 1e-5 lands inside it and d = 1e-3 outside; both must agree with the
    # analytic expansion b^2 (1 - 2 d b / 3 + (d b)^2 / 3).
    for d in (1e-5, 1e-3):
        expansion = b * b * (1.0 - 2.0 * d * b / 3.0 + (d * b) ** 2 / 3.0)
        assert arl.cusum_arl_siegmund(k + d, k, h) == pytest.approx(expansion, rel=1e-6)


@pytest.mark.parametrize("h", [4.0, 5.0, 6.351522848680384, 8.0])
@pytest.mark.parametrize("delta", [0.0, 0.5, 1.0, 2.0])
def test_siegmund_agrees_with_brook_evans_within_five_percent(h: float, delta: float) -> None:
    markov = arl.cusum_arl_markov(delta, 0.5, h, 400)
    sieg = arl.cusum_arl_siegmund_two_sided(delta, 0.5, h)
    rel = abs(sieg - markov) / markov
    assert rel < 0.05, f"delta={delta} h={h}: markov={markov:.4f} siegmund={sieg:.4f} rel={rel:.4f}"


def test_cusum_markov_discretisation_converges() -> None:
    """Brook-Evans error is O(1 / n_states), so doubling the states must at least
    halve the gap to a very fine reference."""
    reference = arl.cusum_arl_markov(0.0, 0.5, 5.0, 3200)
    coarse = abs(arl.cusum_arl_markov(0.0, 0.5, 5.0, 100) - reference)
    fine = abs(arl.cusum_arl_markov(0.0, 0.5, 5.0, 400) - reference)
    assert fine < coarse / 3.0


def test_cusum_arl_decreases_with_shift() -> None:
    previous = np.inf
    for delta in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        value = arl.cusum_arl_markov(delta, 0.5, 5.0, 400)
        assert value < previous
        previous = value


# --------------------------------------------------------------------------- #
# Design round-trips: the designer must deliver what it promises, through the
# same numerical method, to the bisection tolerance.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
def test_design_cusum_round_trip(alpha: float) -> None:
    design = arl.design_cusum_h(alpha, 0.5, 100)
    assert design.achieved_alpha_w == pytest.approx(alpha, rel=1e-6)
    assert design.target_alpha_w == alpha
    assert design.arl0 > 0.0
    assert "brook-evans" in design.method


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
def test_design_ewma_round_trip(alpha: float) -> None:
    design = arl.design_ewma_L(alpha, 0.2, 100)
    assert design.achieved_alpha_w == pytest.approx(alpha, rel=1e-6)
    assert "lucas-saccucci" in design.method


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
def test_design_ool_round_trip(alpha: float) -> None:
    design = arl.design_ool_limit(alpha, 3, 100)
    assert design.achieved_alpha_w == pytest.approx(alpha, rel=1e-9)


def test_tighter_target_needs_a_larger_threshold() -> None:
    loose = arl.design_cusum_h(0.1, 0.5, 100).threshold
    tight = arl.design_cusum_h(0.001, 0.5, 100).threshold
    assert tight > loose


def test_window_far_is_monotone_in_window_length() -> None:
    previous = -1.0
    for window in (1, 5, 20, 100, 400):
        value = arl.cusum_window_false_alarm(0.5, 5.0, window)
        assert value > previous
        previous = value
    assert previous < 1.0


# --------------------------------------------------------------------------- #
# Input validation.
# --------------------------------------------------------------------------- #


def test_rejects_non_probability_targets() -> None:
    with pytest.raises(ValueError, match="strictly in"):
        arl.design_cusum_h(0.0, 0.5, 100)
    with pytest.raises(ValueError, match="strictly in"):
        arl.design_ewma_L(1.0, 0.2, 100)
    with pytest.raises(ValueError, match="strictly in"):
        arl.design_ool_limit(-0.1, 2, 100)


def test_rejects_bad_chart_parameters() -> None:
    with pytest.raises(ValueError, match="k must be > 0"):
        arl.cusum_arl_siegmund(0.0, 0.0, 5.0)
    with pytest.raises(ValueError, match="h must be > 0"):
        arl.cusum_arl_siegmund(0.0, 0.5, 0.0)
    with pytest.raises(ValueError, match=r"lam must lie in \(0, 1\]"):
        arl.ewma_sigma_z(0.0)
    with pytest.raises(ValueError, match=r"lam must lie in \(0, 1\]"):
        arl.ewma_sigma_z(1.5)


def test_rejects_bad_window_and_states() -> None:
    with pytest.raises(ValueError, match="window_length must be"):
        arl.cusum_window_false_alarm(0.5, 5.0, 0)
    with pytest.raises(ValueError, match="n_states must be >= 2"):
        arl.cusum_arl_markov(0.0, 0.5, 5.0, 1)
    with pytest.raises(ValueError, match="n_states must be >= 3"):
        arl.ewma_arl_markov(0.0, 0.2, 3.0, 2)
    with pytest.raises(ValueError, match="persistence must be >= 1"):
        arl.ool_arl(0.1, 0)


def test_unreachable_target_is_reported_not_silently_clipped() -> None:
    with pytest.raises(ValueError, match="not reachable"):
        arl.design_cusum_h(0.999999999, 0.5, 2, bracket=(1.0, 60.0))
