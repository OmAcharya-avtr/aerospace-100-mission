"""Known-answer tests: measured ARL against published and closed-form values.

These are the tests that make the rest of the measurements trustworthy. Every
other number in this repository comes out of the same ARL machinery, so if the
machinery reproduces a published ARL for the one detector that has one, the
machinery is measuring what it says it is.

The reference chain, with the arithmetic in the open
----------------------------------------------------
The NIST/SEMATECH e-Handbook section 6.3.2.3.1 tabulates, for ``k = 0.5``, the
in-control ARL of a **one-sided** CUSUM as 336 at ``h = 4`` and 930 at ``h = 5``.
This package's CUSUM runs two one-sided charts against the same ``h``, so it
raises false alarms at twice the rate and

    h = 4:  ARL0 = 336 / 2 = 168
    h = 5:  ARL0 = 930 / 2 = 465

Independently, Siegmund's closed form for the one-sided ARL with
``b = h + 1.166`` and ``D = delta - k`` gives

    h = 4:  (exp(2 * 0.5 * 5.166) + 2 * (-0.5) * 5.166 - 1) / (2 * 0.25)
          = (exp(5.166) - 5.166 - 1) / 0.5
          = (175.195 - 6.166) / 0.5 = 338.06    (0.6 % above the handbook)
    h = 5:  (exp(6.166) - 6.166 - 1) / 0.5
          = (476.265 - 7.166) / 0.5 = 938.20    (0.9 % above the handbook)

so the two references agree to within 1 % and either can anchor the test.
"""

from __future__ import annotations

import pytest

from telemdrift.detectors import CUSUM
from telemdrift.reference import (
    NIST_CUSUM_ARL,
    ks_asymptotic_tail_probability,
    nist_two_sided_arl0,
    siegmund_one_sided_arl,
    siegmund_two_sided_arl0,
)
from telemdrift.scoring import measure_arl0
from telemdrift.streams import stationary


def _stationary(length, seed):
    return stationary(length, seed)


@pytest.mark.parametrize("h,expected", [(4, 338.06), (5, 938.20)])
def test_siegmund_formula_reproduces_the_hand_arithmetic(h, expected):
    assert siegmund_one_sided_arl(h, 0.5, 0.0) == pytest.approx(expected, abs=0.05)


@pytest.mark.parametrize("h", [4, 5])
def test_siegmund_agrees_with_the_nist_handbook_within_one_percent(h):
    nist = NIST_CUSUM_ARL[h]["arl0"]
    sieg = siegmund_one_sided_arl(h, 0.5, 0.0)
    assert abs(sieg - nist) / nist < 0.01


def test_two_sided_reference_is_half_the_one_sided_value():
    assert nist_two_sided_arl0(4) == pytest.approx(168.0)
    assert nist_two_sided_arl0(5) == pytest.approx(465.0)
    assert siegmund_two_sided_arl0(4) == pytest.approx(169.03, abs=0.05)


def test_nist_table_only_covers_the_two_tabulated_thresholds():
    with pytest.raises(ValueError, match="handbook table covers"):
        nist_two_sided_arl0(6)


def test_siegmund_removable_singularity_at_delta_equals_k():
    """D = 0 makes the expression 0/0; the limit is b^2 = (h + 1.166)^2."""
    h = 3.0
    assert siegmund_one_sided_arl(h, 0.5, 0.5) == pytest.approx((h + 1.166) ** 2)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_siegmund_rejects_nonpositive_h(bad):
    with pytest.raises(ValueError, match="h must be > 0"):
        siegmund_one_sided_arl(bad)


def test_siegmund_rejects_negative_k():
    with pytest.raises(ValueError, match="k must be >= 0"):
        siegmund_one_sided_arl(3.0, -0.1)


@pytest.mark.parametrize(
    "h,reference,tolerance_frac",
    [
        # Reference = NIST one-sided value halved for the two-sided chart.
        # Tolerance is 10 % and is set from the Monte Carlo standard error, not
        # from the result: 12 seeds x 60000 samples gives about 4400 run lengths
        # at h = 4 (rel. SEM ~ 1.5 %) and 1560 at h = 5 (rel. SEM ~ 2.6 %), so
        # 10 % is between four and seven standard errors. The halving is itself
        # an approximation (the two arms are driven by the same observations),
        # which is why the band is not tighter.
        (4, 168.0, 0.10),
        (5, 465.0, 0.10),
    ],
)
def test_measured_cusum_arl0_matches_the_published_value(h, reference, tolerance_frac):
    res = measure_arl0(
        lambda: CUSUM(h=h, k=0.5), _stationary, range(59_001, 59_013), 60_000
    )
    assert res.n_runs > 1000
    rel = abs(res.arl0 - reference) / reference
    assert rel < tolerance_frac, (
        f"h={h}: measured {res.arl0:.1f} +/- {res.sem:.1f}, reference {reference}, "
        f"relative difference {100 * rel:.1f} %"
    )


def test_measured_cusum_arl0_is_slightly_below_the_reference_as_censoring_predicts():
    """A directional check, not a magnitude one.

    Excluding the right-censored tail after the last alarm drops the longest
    partial run from every stream, so the estimator is biased **downwards**. The
    bias is small (the tail is under 1 % of samples here) but it should have a
    sign, and a measurement that came out systematically high would mean
    something else was wrong.
    """
    res = measure_arl0(
        lambda: CUSUM(h=4.0, k=0.5), _stationary, range(59_001, 59_013), 60_000
    )
    assert res.arl0 < 168.0
    assert res.censored_tail_samples > 0


def test_ks_asymptotic_tail_reproduces_the_declared_alpha():
    """The shipped default c was derived from alpha = 0.005; the inverse agrees."""
    from telemdrift.detectors import WindowedKS

    p = ks_asymptotic_tail_probability(WindowedKS.default_threshold(), 200, 100)
    assert p == pytest.approx(0.005, rel=1e-9)


def test_ks_asymptotic_tail_is_monotone_and_bounded():
    probs = [ks_asymptotic_tail_probability(c, 200, 100) for c in (0.05, 0.1, 0.2, 0.4)]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert ks_asymptotic_tail_probability(0.0, 200, 100) == 1.0


def test_ks_asymptotic_tail_rejects_empty_windows():
    with pytest.raises(ValueError, match=">= 1"):
        ks_asymptotic_tail_probability(0.2, 0, 100)
