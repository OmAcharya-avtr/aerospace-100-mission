"""Pinned counterexamples and benchmark regressions, with literal inputs.

``.hypothesis/`` is gitignored, so a falsifying example found by one run is lost
on a fresh clone. Every counterexample Hypothesis has found against this product
is pinned here as a named test with the literal inputs in the body.
"""

from __future__ import annotations

import numpy as np
import pytest

from deadlinerisk.accounting import account
from deadlinerisk.baselines import mean_plus_margin
from deadlinerisk.bootstrap import block_length_from_autocorrelation, understatement_factor
from deadlinerisk.bounds import bonferroni_pair_bracket, cantelli_slack, cantelli_upper_bound

from fixtures.synthetic_trace import load_synthetic_campaign


def test_regression_k_invariance_two_nearly_equal_responses() -> None:
    """Hypothesis, 2026-11-13, seed 1, 4000 examples.

    ``test_mean_plus_margin_k_is_invariant_under_a_common_rescaling`` failed at

        responses = [0.09479453252194603, 0.09496984880350386]
        deadline  = 0.03125
        scale     = 0.001

    with k = -513.2980590228497 unscaled against -513.2980590228335 scaled, a
    discrepancy of 1.62e-11 against a tolerance of 1.45e-11. The tolerance was
    wrong, not the implementation: it accounted for the conditioning of
    ``D - mean`` and not for that of the sample variance. Two values differing
    in their fifth significant figure give a sample standard deviation whose
    relative condition number is about 1.2e6, and k inherits half of that. The
    fixed tolerance includes both terms. This test pins the literal inputs.
    """
    values = np.asarray([0.09479453252194603, 0.09496984880350386])
    deadline = 0.03125
    scale = 0.001
    plain = mean_plus_margin(values, deadline)
    scaled = mean_plus_margin(values * scale, deadline * scale)
    condition_subtraction = (abs(deadline) + abs(plain.mean_s)) / abs(
        deadline - plain.mean_s
    )
    condition_variance = float((values**2).sum()) / (
        (values.size - 1) * plain.sd_s**2
    )
    tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * (condition_subtraction + 0.5 * condition_variance)
        * abs(plain.k_to_deadline)
    )
    # The discrepancy that falsified the old tolerance, 1.62e-11, is inside the
    # new one; the new tolerance is nonetheless far below |k| itself, so the
    # check has not been made vacuous.
    assert tolerance > 1.62e-11
    assert tolerance < 1e-6 * abs(plain.k_to_deadline)
    assert scaled.k_to_deadline == pytest.approx(
        plain.k_to_deadline, rel=0.0, abs=float(tolerance)
    )


def test_regression_cantelli_slack_is_not_vacuous_for_a_tiny_bound() -> None:
    """A slack floored at an absolute constant would admit anything.

    An earlier product of this mission floored a bound-check slack at an
    absolute 1.8e-15, which made a bracket of width 4.5e-23 vacuous. This pins
    the opposite behaviour for the Cantelli slack: with sd = 1e-14 and
    D - mean = 1, the bound is 1e-28 and the slack must be far below it.
    """
    bound = cantelli_upper_bound(0.0, 1e-14, 1.0)
    slack = cantelli_slack(0.0, 1e-14, 1.0)
    assert bound == pytest.approx(1e-28, rel=1e-9, abs=0.0)
    assert slack < 1e-3 * bound


def test_regression_bonferroni_bracket_on_a_single_isolated_miss() -> None:
    """HAND CALCULATION, pinned because it is the smallest non-trivial case.

    indicator = [0, 1, 0]; two overlapping pairs (0,1) and (1,0).
      left  = [0, 1] -> p1 = 0.5
      right = [1, 0] -> p2 = 0.5
      max   = [1, 1] -> observed = 1.0
      lower = 0.5 ; upper = min(1, 1.0) = 1.0
    The union bound is exactly attained, so the observation sits on the upper
    end of the bracket. A slack floored at a constant would hide a violation
    just outside it; the slack here is a few eps of 1.0.
    """
    bracket = bonferroni_pair_bracket(np.asarray([0, 1, 0], dtype=np.int8))
    assert bracket.observed == pytest.approx(1.0, rel=1e-12, abs=0.0)
    assert bracket.upper == pytest.approx(1.0, rel=1e-12, abs=0.0)
    assert bracket.slack < 1e-14
    assert bracket.inside is True


def test_regression_block_length_one_for_an_alternating_indicator() -> None:
    """An alternating 0/1 sequence has rho(1) = -1 + O(1/n), below the band.

    The rule's cut-off lag is therefore 1 and the block length 1. Pinned
    because a rule that keyed on |rho| rather than rho would select a long
    block here, and the resulting interval would be wrong in the direction that
    looks conservative.
    """
    indicator = np.tile([0, 1], 2000).astype(np.int8)
    choice = block_length_from_autocorrelation(indicator)
    assert choice.rho_head[1] < 0.0
    assert choice.lag_cut == 1
    assert choice.block_length == 1


def test_regression_synthetic_benchmark_totals_are_stable(tmp_path) -> None:
    """Benchmark regression: the committed generator's own numbers at seed 920092.

    These are SYNTHETIC figures and measure no board. They are pinned so that a
    change to the generator, to numpy's PCG64 stream, or to the loop logic is
    caught as a change rather than silently moving every other test's inputs.
    The values were produced by running this generator in this environment
    (numpy 2.5.3, Python 3.13.18).
    """
    campaign = load_synthetic_campaign(tmp_path)
    totals = account(campaign.jobs["gt_long"])
    assert totals.executed == 5978
    assert totals.releases == 6001
    assert totals.misses == 40
    assert totals.miss_rate == pytest.approx(40.0 / 6001.0, rel=1e-12, abs=0.0)
    margin = mean_plus_margin(
        campaign.jobs["gt_long"].response_s, campaign.jobs["gt_long"].deadline_s
    )
    assert margin.k_to_deadline == pytest.approx(2.7952700891756392, rel=1e-9, abs=0.0)
    headline = understatement_factor(campaign.jobs["gt_long"], resamples=200, seed=920092)
    assert headline.factor == pytest.approx(2.5707759791936393, rel=1e-9, abs=0.0)
    assert headline.block.block_length == 4
