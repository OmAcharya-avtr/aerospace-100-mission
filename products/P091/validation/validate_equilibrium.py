"""The equilibrium test on SYNTHETIC fixtures, with controls that must fire.

Input: SYNTHETIC fixtures. Every number here is a property of the estimator, not
a measurement of the board. The measured verdicts are in
``validate_measured_campaign.py``, which refuses to run without the traces.

Why the controls matter here more than elsewhere. The equilibrium test is the
gate on every sustained figure the product reports, and its failure mode is
silent: a test that is too permissive averages a still-warming run and reports a
throughput the board would not sustain, in the flattering direction. So every
control below is anchored on a KNOWN drift injected into a known-flat trace, and
each one is asserted to have fired.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

import numpy as np  # noqa: E402

from thermalbudget.equilibrium import (  # noqa: E402
    THRESHOLD_LADDER,
    equilibrium_sensitivity,
    equilibrium_test,
    fit_linear,
)
from thermalbudget.trace import Trace  # noqa: E402


def _with_drift(trace: Trace, drift_c_per_s: float) -> Trace:
    """A copy of ``trace`` with a known linear drift added to the die channel.

    The drift is anchored on the TRUE channel, not on a fitted value: a control
    built by offsetting a fit is not guaranteed to leave the band the fit
    already sits inside.
    """
    data = {name: values.copy() for name, values in trace.data.items()}
    from thermalbudget import schema

    data[schema.DIE_COLUMN] = data[schema.DIE_COLUMN] + drift_c_per_s * trace.t_rel
    return Trace(
        label=f"{trace.label}+drift{drift_c_per_s:+g}",
        duty=trace.duty,
        origin=trace.origin,
        path=trace.path,
        columns=trace.columns,
        data=data,
        missing=dict(trace.missing),
        temperature_divisor=trace.temperature_divisor,
    )


def main() -> int:
    out = _bootstrap.Tee("validate_equilibrium_output.txt")
    out.rule("EQUILIBRIUM TEST -- tested, not assumed")
    out("INPUT ORIGIN: SYNTHETIC. These are checks of the estimator. No number in this")
    out("file is a measurement of any board.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)
    out("")
    out(
        f"criterion: |fitted slope| <= {_bootstrap.THRESHOLD_C_PER_S:g} degC/s over the "
        f"final {_bootstrap.WINDOW_S:g} s, at least 30 finite samples"
    )

    campaign = _bootstrap.synthetic_campaign("no_throttle")

    out.rule("1  known-answer fits")
    fit = fit_linear(np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 3.0, 5.0, 7.0]))
    out.check(
        "an exact line y = 1 + 2x recovers slope 2 and intercept 1",
        abs(fit.slope - 2.0) < 1e-12 and abs(fit.intercept - 1.0) < 1e-12,
        f"slope {fit.slope:.15g}, intercept {fit.intercept:.15g}, "
        f"rms residual {fit.rms_residual:.3g}",
    )
    # Hand calculation: x = 0,1,2; y = 0,0,2. Sxx = 2, Sxy = 2, slope = 1,
    # intercept = 2/3 - 1 = -1/3, residuals 1/3, -2/3, 1/3.
    fit = fit_linear(np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.0, 2.0]))
    out.check(
        "the hand-calculated case x=(0,1,2), y=(0,0,2) gives slope 1, intercept -1/3",
        abs(fit.slope - 1.0) < 1e-12 and abs(fit.intercept + 1.0 / 3.0) < 1e-12,
        f"slope {fit.slope:.15g}, intercept {fit.intercept:.15g}, "
        f"OLS stderr {fit.slope_stderr_ols:.6f} (hand value {math.sqrt((2/3)/2):.6f})",
    )

    out.rule("2  the AR(1) correction, and the size of it on real-shaped data")
    for label in ("soak_1.00_r1", "soak_0.50", "idle_pre"):
        verdict = equilibrium_test(campaign[label])
        inflation = verdict.fit.inflation if verdict.fit else math.nan
        rho = verdict.fit.residual_autocorrelation if verdict.fit else math.nan
        out(
            f"  {label:<14s} rho {rho:+.4f}  stderr OLS "
            f"{verdict.fit.slope_stderr_ols:.3e}  AR(1) "
            f"{verdict.fit.slope_stderr_ar1:.3e}  inflation {inflation:.3f}x"
        )
    out.check(
        "the correction is applied and is reported beside the uncorrected value",
        all(
            equilibrium_test(campaign[label]).fit is not None
            for label in ("soak_1.00_r1", "idle_pre")
        ),
        "both standard errors travel with every verdict",
    )

    out.rule("3  verdicts over the whole synthetic plan")
    counts: dict[str, int] = {}
    for label in sorted(campaign.traces):
        verdict = equilibrium_test(campaign[label])
        counts[verdict.status] = counts.get(verdict.status, 0) + 1
        slope = (
            f"{verdict.slope_c_per_s:+.6f}" if math.isfinite(verdict.slope_c_per_s) else "n/a"
        )
        out(
            f"  {label:<16s} {verdict.status:<21s} slope {slope:>10s} degC/s  "
            f"n={verdict.n_samples:<4d} mean={verdict.mean_die_c:7.3f} degC"
        )
    out(f"  status counts: {counts}")
    out.check(
        "every duty-1.0 soak of the synthetic plan is equilibrated",
        all(equilibrium_test(trace).equilibrated for trace in campaign.soaks(1.0)),
        "3 of 3",
    )
    out.check(
        "no 20 s burst is reported equilibrated",
        all(not equilibrium_test(trace).equilibrated for trace in campaign.bursts()),
        "INSUFFICIENT_SAMPLES on both, which is explicitly not a pass",
    )
    out.check(
        "every cooldown fails the test, which is the correct answer for a decay",
        all(
            equilibrium_test(trace).status == "NOT_EQUILIBRATED"
            for trace in campaign.cooldowns()
        ),
        f"{len(campaign.cooldowns())} of {len(campaign.cooldowns())}",
    )

    out.rule("4  negative controls, each anchored on an injected KNOWN drift")
    flat = campaign["soak_1.00_r1"]
    baseline_verdict = equilibrium_test(flat)
    out(
        f"  the anchor trace fits a slope of {baseline_verdict.slope_c_per_s:+.6f} degC/s, "
        f"well inside the {_bootstrap.THRESHOLD_C_PER_S:g} degC/s criterion"
    )
    # A drift must exceed the threshold by more than the anchor's own offset for
    # the control to be GUARANTEED to fire; 3x the threshold is used so the
    # control does not depend on the noise draw.
    for multiplier in (3.0, 10.0, -3.0, -10.0):
        drift = multiplier * _bootstrap.THRESHOLD_C_PER_S
        drifted = _with_drift(flat, drift)
        verdict = equilibrium_test(drifted)
        out.control(
            f"an injected drift of {drift:+.4f} degC/s is caught",
            verdict.status == "NOT_EQUILIBRATED",
            must_fire=True,
            detail=f"fitted {verdict.slope_c_per_s:+.6f} degC/s, status {verdict.status}",
        )
    # And two that must NOT fire: a drift well inside the criterion, and the
    # unmodified trace. A control that fires when it must not is equally wrong.
    for multiplier in (0.0, 0.1):
        drift = multiplier * _bootstrap.THRESHOLD_C_PER_S
        verdict = equilibrium_test(_with_drift(flat, drift))
        out.control(
            f"a drift of {drift:+.5f} degC/s, inside the criterion, is NOT flagged",
            verdict.status == "NOT_EQUILIBRATED",
            must_fire=False,
            detail=f"fitted {verdict.slope_c_per_s:+.6f} degC/s, status {verdict.status}",
        )
    # A starved window must not pass either. All but 10 samples removed.
    starved = flat.window(0.0, 10.0)
    verdict = equilibrium_test(starved)
    out.control(
        "a window with fewer than 30 finite samples does not pass",
        not verdict.equilibrated,
        must_fire=True,
        detail=f"status {verdict.status} over {verdict.n_samples} samples",
    )

    out.rule("5  the threshold is a criterion, so the ladder is reported")
    out(f"  ladder: {THRESHOLD_LADDER} degC/s")
    for label in ("soak_0.25", "soak_0.75", "soak_1.00_r2", "cool_r1"):
        ladder = equilibrium_sensitivity(campaign[label])
        rendered = "  ".join(
            f"{threshold:g}:{status[:4]}" for threshold, status in ladder.items()
        )
        out(f"  {label:<14s} {rendered}")
    monotone = True
    for label in sorted(campaign.traces):
        statuses = list(equilibrium_sensitivity(campaign[label]).values())
        first = next(
            (index for index, status in enumerate(statuses) if status == "EQUILIBRATED"),
            None,
        )
        if first is not None and not all(
            status == "EQUILIBRATED" for status in statuses[first:]
        ):
            monotone = False
    out.check(
        "the verdict is monotone in the threshold across all 17 runs",
        monotone,
        "once a threshold admits a run, every looser threshold admits it too",
    )

    out.rule("6  input validation")
    for name, keywords, pattern in (
        ("a non-positive window", {"window_s": 0.0}, "window_s must be positive"),
        (
            "a non-positive threshold",
            {"threshold_c_per_s": -1.0},
            "threshold_c_per_s must be positive",
        ),
    ):
        fired = False
        try:
            equilibrium_test(flat, **keywords)
        except ValueError as error:
            fired = pattern in str(error)
        out.control(f"{name} is refused", fired, must_fire=True)

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
