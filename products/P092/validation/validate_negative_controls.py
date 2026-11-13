"""Negative controls, anchored on the TRUE value, each asserted to FIRE.

A control is a deliberately violated case that the product's own detection
criterion must catch. Four are declared, with their direction fixed before they
were run:

  A  right-skewed (lognormal) response   must FIRE as understatement
  B  Gaussian response                   must NOT fire
  C  dropped-release accounting          must FIRE as inflation
  D  Pareto-spiked response              must FIRE as OVERSTATEMENT

Each is anchored on a COUNTED quantity of a committed deterministic trace --
the trace's own executed-job miss rate -- and not on an output of anything under
test. The declared detection criterion is

    factor = executed_miss_rate / mean_plus_margin_probability
    understating when factor >= 3.0 ; overstating when factor <= 1/3

and it was not adjusted after the controls were run. Control A's criterion is
not a 95 per cent interval: it is a declared factor of three, and the margin
between it and the measured value is reported below in counting standard errors
so a reader can judge the strength rather than take the word "FIRED" on trust. A
first draft of control A ran a 600 s trace, carried 38 events and sat 2.1
standard errors above the criterion; the trace was LENGTHENED to 3000 s to get a
usable margin, and the declared criterion was not moved.

The executed-job rate is used rather than the all-release rate because it is the
quantity the Gaussian rule actually estimates: a dropped release has no response
time, so the rule never sees one. Control C isolates the drop accounting on its
own.

Direction D was MEASURED, not assumed. A Pareto spike fired on 2 per cent of
jobs inflates the sample standard deviation more than it inflates the exceedance
count, so the Gaussian rule OVERSTATES. A heavy tail therefore does not always
make the mean-plus-margin rule optimistic, and this control exists so that claim
is tested rather than stated.

INPUT ORIGIN: the committed SYNTHETIC generator only. No measured data is read,
and no number here measures any board.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from deadlinerisk.accounting import account  # noqa: E402
from deadlinerisk.baselines import mean_plus_margin  # noqa: E402

from fixtures.synthetic_trace import RunSpec, load_synthetic_campaign  # noqa: E402

UNDERSTATEMENT_CRITERION = 3.0
OVERSTATEMENT_CRITERION = 1.0 / 3.0
DROP_INFLATION_CRITERION = 1.5

CONTROLS: tuple[tuple[str, RunSpec, str], ...] = (
    (
        "A right-skewed lognormal response",
        RunSpec(
            "gt_long", 0, 0, 3000.0, tail_probability=0.0,
            distribution="lognormal", idle_fraction=0.75, spread=0.09,
        ),
        "understatement",
    ),
    (
        "B Gaussian response",
        RunSpec(
            "gt_long", 0, 0, 120.0, tail_probability=0.0,
            distribution="normal", idle_fraction=0.80, spread=0.12,
        ),
        "none",
    ),
    (
        "D Pareto-spiked response",
        RunSpec("gt_long", 4, 0, 120.0, tail_probability=0.02, tail_alpha=1.1),
        "overstatement",
    ),
)


def main() -> int:
    out = _bootstrap.Tee("validate_negative_controls_output.txt")
    out.rule("NEGATIVE CONTROLS -- anchored on the true value, verified to fire")
    out("INPUT ORIGIN: SYNTHETIC generator only. No measured data is read.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)
    out("")
    out(f"declared understatement criterion : factor >= {UNDERSTATEMENT_CRITERION}")
    out(f"declared overstatement criterion  : factor <= {OVERSTATEMENT_CRITERION:.6f}")
    out(f"declared drop-inflation criterion : ratio  >= {DROP_INFLATION_CRITERION}")
    out(f"declared control seed             : {_bootstrap.SEED_CONTROL}")

    out.rule("1  the three distributional controls")
    out(
        f"  {'control':<36}{'exec_misses':>12}{'releases':>10}{'exec_rate':>13}"
        f"{'gauss_P':>13}{'factor':>12}"
    )
    measured: dict[str, tuple[float, int, int]] = {}
    for name, spec, _direction in CONTROLS:
        campaign = load_synthetic_campaign(
            Path(tempfile.mkdtemp(prefix="deadlinerisk-control-")),
            plan=(spec,),
            seed=_bootstrap.SEED_CONTROL,
        )
        jobs = campaign.jobs[spec.run]
        totals = account(jobs)
        margin = mean_plus_margin(jobs.response_s, jobs.deadline_s)
        factor = (
            totals.executed_miss_rate / margin.probability
            if margin.probability > 0.0
            else float("inf")
        )
        measured[name] = (factor, int(jobs.missed.sum()), totals.releases)
        out(
            f"  {name:<36}{int(jobs.missed.sum()):>12}{totals.releases:>10}"
            f"{totals.executed_miss_rate:>13.6e}{margin.probability:>13.6e}"
            f"{factor:>12.4f}"
        )
    out("")
    for name, _spec, direction in CONTROLS:
        factor, misses, releases = measured[name]
        if direction == "understatement":
            out.control(
                name,
                fired=factor >= UNDERSTATEMENT_CRITERION,
                must_fire=True,
                detail=f"factor {factor:.4f} against the criterion "
                f"{UNDERSTATEMENT_CRITERION}; {misses} measured executed misses in "
                f"{releases} releases, so the counting standard error on the "
                f"numerator is about {100.0 / max(misses, 1) ** 0.5:.1f} per cent and "
                f"the criterion is cleared by "
                f"{(factor - UNDERSTATEMENT_CRITERION) / (factor / max(misses, 1) ** 0.5):.2f} "
                "counting standard errors",
            )
        elif direction == "overstatement":
            out.control(
                name,
                fired=factor <= OVERSTATEMENT_CRITERION,
                must_fire=True,
                detail=f"factor {factor:.4f} against the criterion "
                f"{OVERSTATEMENT_CRITERION:.6f}; {misses} measured executed misses "
                f"in {releases} releases",
            )
        else:
            out.control(
                name,
                fired=factor >= UNDERSTATEMENT_CRITERION,
                must_fire=False,
                detail=f"factor {factor:.4f} stayed below the criterion "
                f"{UNDERSTATEMENT_CRITERION} on a Gaussian response; "
                f"{misses} measured executed misses in {releases} releases",
            )
            out.check(
                f"{name}: factor inside the two-sided adequacy band",
                OVERSTATEMENT_CRITERION <= factor <= UNDERSTATEMENT_CRITERION,
                f"{OVERSTATEMENT_CRITERION:.4f} <= {factor:.4f} <= "
                f"{UNDERSTATEMENT_CRITERION}",
            )

    out.rule("2  control C -- the drop accounting on its own")
    campaign = load_synthetic_campaign(
        Path(tempfile.mkdtemp(prefix="deadlinerisk-control-c-")),
        plan=(CONTROLS[1][1],),
        seed=_bootstrap.SEED_CONTROL,
    )
    totals = account(campaign.jobs["gt_long"])
    ratio = (
        totals.miss_rate / totals.executed_miss_rate
        if totals.executed_miss_rate > 0.0
        else float("inf")
    )
    out(f"  executed rows      : {totals.executed}")
    out(f"  dropped releases   : {totals.dropped}")
    out(f"  executed-only rate : {totals.executed_miss_rate:.6e}")
    out(f"  all-release rate   : {totals.miss_rate:.6e}")
    out(f"  ratio              : {ratio:.6f}")
    out.check(
        "control C has dropped releases to detect",
        totals.dropped > 0,
        f"{totals.dropped} dropped",
    )
    out.control(
        "C dropped-release accounting",
        fired=ratio >= DROP_INFLATION_CRITERION,
        must_fire=True,
        detail=f"ratio {ratio:.6f} against the criterion {DROP_INFLATION_CRITERION} "
        f"with {totals.dropped} dropped releases",
    )

    out.rule("3  the control inventory")
    out("  A right-skewed lognormal response  must FIRE  (understatement)")
    out("  B Gaussian response                must NOT fire")
    out("  C dropped-release accounting       must FIRE  (inflation)")
    out("  D Pareto-spiked response           must FIRE  (overstatement)")
    out(
        "  A control that does not fire is a failure of the CONTROL, not a success of "
        "the product: it would mean the check it guards could not catch the violation "
        "it exists to catch."
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
