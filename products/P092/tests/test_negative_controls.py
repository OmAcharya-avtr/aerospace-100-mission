"""Negative controls, anchored on the TRUE value and asserted to FIRE.

Every control here is anchored on a COUNTED quantity of the synthetic trace --
the trace's own executed-job miss rate, which for a committed deterministic
trace is the true value and not an output of anything under test -- and each
asserts the direction it was declared to produce. A control that does not fire
is a failure of the control, not a success of the product.

The declared detection criterion, fixed before the controls were run and not
adjusted afterwards:

    factor = executed_miss_rate / mean_plus_margin_probability

    the rule is DETECTED AS UNDERSTATING  when factor >= 3.0
    the rule is DETECTED AS OVERSTATING   when factor <= 1/3

The executed-job rate is used rather than the all-release rate because it is the
quantity the Gaussian rule actually estimates: a dropped release has no response
time, so the rule never sees one. Comparing the rule against the all-release
rate mixes the distributional error with the drop accounting, and control C
below isolates the second on its own.

Which direction each generator knob pushes the rule was MEASURED, not assumed. A
Pareto-spiked trace turns out to make the rule OVERSTATE, because the spikes
inflate the sample standard deviation more than they inflate the exceedance
count. Control D asserts that direction because it is what the data does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deadlinerisk.accounting import account
from deadlinerisk.baselines import mean_plus_margin

from fixtures.synthetic_trace import RunSpec, load_synthetic_campaign

#: Declared detection criterion. Not a seed, not tuned after the fact.
UNDERSTATEMENT_CRITERION = 3.0
OVERSTATEMENT_CRITERION = 1.0 / 3.0

#: Declared criterion for the drop-accounting control.
DROP_INFLATION_CRITERION = 1.5

#: Seed for every control trace. One seed, declared.
CONTROL_SEED = 5150


def _factor(directory: Path, spec: RunSpec) -> tuple[float, int, int]:
    """Return ``(factor, executed_misses, releases)`` for one control trace."""
    campaign = load_synthetic_campaign(directory, plan=(spec,), seed=CONTROL_SEED)
    jobs = campaign.jobs[spec.run]
    totals = account(jobs)
    margin = mean_plus_margin(jobs.response_s, jobs.deadline_s)
    assert margin.probability > 0.0, "control trace underflowed the Gaussian tail"
    return (
        totals.executed_miss_rate / margin.probability,
        int(jobs.missed.sum()),
        totals.releases,
    )


def test_control_a_right_skewed_response_must_fire(tmp_path: Path) -> None:
    """Control A, must FIRE: a lognormal response is detected as understated.

    A lognormal response with sigma = 0.09 and a median at 0.75 of the deadline
    has a right-skewed tail. The TRUE executed-job miss rate is counted from the
    trace; the Gaussian rule's probability is computed from the same trace's mean
    and standard deviation. The control fires when the ratio reaches the declared
    3.0.

    The trace is 3000 s long and carries 193 measured executed misses over
    300001 releases, so the counting standard error on the numerator is about
    1/sqrt(193) = 7.2 per cent and the absolute standard error on the factor is
    about 0.34. The measured factor is about 4.70, which is about 5.0 such
    standard errors above the criterion.

    The trace was LENGTHENED from 600 s to 3000 s to get that margin. The
    DECLARED CRITERION of 3.0 was not moved, and the seed was not changed: a
    first draft of this control ran 600 s, carried 38 events, measured 4.54 and
    sat only 2.1 standard errors above the criterion, which is close enough to
    an expected outcome to be weak evidence. Strengthening a control by
    collecting more of a synthetic trace is legitimate; moving the criterion
    after seeing the result would not be.
    """
    factor, misses, releases = _factor(
        tmp_path,
        RunSpec(
            "gt_long",
            0,
            0,
            3000.0,
            tail_probability=0.0,
            distribution="lognormal",
            idle_fraction=0.75,
            spread=0.09,
        ),
    )
    fired = factor >= UNDERSTATEMENT_CRITERION
    assert fired, (
        f"control A did NOT fire: factor {factor!r} below the declared criterion "
        f"{UNDERSTATEMENT_CRITERION} ({misses} executed misses in {releases} releases)"
    )
    # The control's strength is asserted, not assumed: at least 100 measured
    # events, and the measured factor at least four counting standard errors
    # above the declared criterion.
    assert misses >= 100, f"only {misses} measured misses; the control is too thin"
    standard_error = factor / misses**0.5
    assert (factor - UNDERSTATEMENT_CRITERION) / standard_error >= 4.0, (
        f"control A fires by only {(factor - UNDERSTATEMENT_CRITERION) / standard_error:.2f} "
        "counting standard errors; that is too close to an expected outcome to be "
        "evidence"
    )


def test_control_b_gaussian_response_must_not_fire(tmp_path: Path) -> None:
    """Control B, must NOT fire: a Gaussian response is not detected as understated.

    The response distribution IS Gaussian by construction, so the Gaussian rule
    is right and the factor must sit near 1. A control that fired here would mean
    the detection criterion flags an adequate rule, which would make every firing
    of control A worthless as evidence.
    """
    factor, misses, releases = _factor(
        tmp_path,
        RunSpec(
            "gt_long",
            0,
            0,
            120.0,
            tail_probability=0.0,
            distribution="normal",
            idle_fraction=0.80,
            spread=0.12,
        ),
    )
    fired = factor >= UNDERSTATEMENT_CRITERION
    assert not fired, (
        f"control B FIRED when it must not: factor {factor!r} reached the criterion "
        f"{UNDERSTATEMENT_CRITERION} on a Gaussian response "
        f"({misses} executed misses in {releases} releases)"
    )
    assert OVERSTATEMENT_CRITERION <= factor <= UNDERSTATEMENT_CRITERION, (
        f"factor {factor!r} outside the two-sided adequacy band on a Gaussian response"
    )


def test_control_c_drop_accounting_must_inflate_the_miss_count(tmp_path: Path) -> None:
    """Control C, must FIRE: dropped releases raise the counted rate.

    A trace with overruns has releases that never ran. Each is a miss by the
    measurement model, so the all-release rate must exceed the executed-only
    rate. The declared criterion is a ratio of at least 1.5, which a single
    dropped release per overrun already produces (one overrun contributes two
    misses and two releases where the executed-only view sees one of one).
    """
    campaign = load_synthetic_campaign(
        tmp_path,
        plan=(
            RunSpec(
                "gt_long",
                0,
                0,
                120.0,
                tail_probability=0.0,
                distribution="normal",
                idle_fraction=0.80,
                spread=0.12,
            ),
        ),
        seed=CONTROL_SEED,
    )
    totals = account(campaign.jobs["gt_long"])
    assert totals.dropped > 0, "control C has no dropped releases to detect"
    ratio = totals.miss_rate / totals.executed_miss_rate
    fired = ratio >= DROP_INFLATION_CRITERION
    assert fired, (
        f"control C did NOT fire: counted/executed ratio {ratio!r} below the declared "
        f"{DROP_INFLATION_CRITERION} with {totals.dropped} dropped releases"
    )


def test_control_d_pareto_spikes_must_fire_as_overstatement(tmp_path: Path) -> None:
    """Control D, must FIRE as OVERSTATEMENT: a fat tail inflates the sd.

    This is the direction that was measured rather than assumed. Firing a Pareto
    spike on 2 per cent of jobs with exponent 1.1 raises the sample standard
    deviation enough that the Gaussian rule's k collapses and its probability
    overshoots the counted rate by more than an order of magnitude. A heavy tail
    therefore does NOT always make the mean-plus-margin rule optimistic, and this
    control exists so that claim is tested rather than stated.
    """
    factor, misses, releases = _factor(
        tmp_path,
        RunSpec("gt_long", 4, 0, 120.0, tail_probability=0.02, tail_alpha=1.1),
    )
    fired = factor <= OVERSTATEMENT_CRITERION
    assert fired, (
        f"control D did NOT fire: factor {factor!r} above the declared overstatement "
        f"criterion {OVERSTATEMENT_CRITERION} ({misses} executed misses in "
        f"{releases} releases)"
    )


def test_all_four_controls_are_declared_with_a_direction() -> None:
    """The control inventory, asserted so the count cannot drift silently."""
    declared = {
        "A right-skewed response": "must fire (understatement)",
        "B Gaussian response": "must NOT fire",
        "C drop accounting": "must fire (inflation)",
        "D Pareto spikes": "must fire (overstatement)",
    }
    assert len(declared) == 4
    assert sum("must fire" in value for value in declared.values()) == 3
    assert sum("must NOT fire" in value for value in declared.values()) == 1
    assert UNDERSTATEMENT_CRITERION == pytest.approx(3.0, rel=0.0, abs=0.0)
    assert OVERSTATEMENT_CRITERION == pytest.approx(1.0 / 3.0, rel=1e-15, abs=0.0)
