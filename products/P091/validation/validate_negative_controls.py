"""Controls across modules: every rule this product rests on, attacked on purpose.

Input: SYNTHETIC fixtures. Each control injects a violation anchored on a KNOWN
true value -- never on an output, because a reconstruction already sits inside
the band it is being tested against -- and each one is asserted to have fired.
Controls that must NOT fire are declared and asserted in that direction too,
because a control that fires on correct data reads as evidence and is not.

The rules under attack:

- a synthetic figure may not be labelled measured, and a figure over zero inputs
  may not be labelled measured either;
- an unequilibrated run is excluded from the sustained figure and named, not
  averaged;
- a headline interval containing zero is reported INDISTINGUISHABLE, in either
  direction, and a learned margin inside the run-to-run interval is reported NOT
  DISTINGUISHABLE whatever its sign;
- a clock channel that does not move yields no clock-versus-temperature relation
  and no clock-to-throughput exponent, rather than a fitted artefact;
- every claim the campaign cannot support is refused by name.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

import numpy as np  # noqa: E402

from thermalbudget import limits  # noqa: E402
from thermalbudget.baselines import (  # noqa: E402
    burst_baseline,
    overstatement,
    rc_baseline,
    sustained_truth,
)
from thermalbudget.bootstrap import BootstrapResult  # noqa: E402
from thermalbudget.clock_throughput import (  # noqa: E402
    clock_temperature_relation,
    proportionality_exponent,
)
from thermalbudget.predictor import (  # noqa: E402
    build_dataset,
    compare_against_rc,
    leave_one_duty_out,
)
from thermalbudget.provenance import NotMeasuredError, Origin, require_measured  # noqa: E402


def _reference(half_width: float, point: float = 60.0) -> BootstrapResult:
    return BootstrapResult(
        point=point,
        ci_low=point - half_width,
        ci_high=point + half_width,
        confidence=0.95,
        n_resamples=1000,
        standard_error=half_width / 2.0,
        bias=0.0,
        n_groups=3,
        n_samples=300,
        n_distinct_group_draws=27,
    )


def main() -> int:
    out = _bootstrap.Tee("validate_negative_controls_output.txt")
    out.rule("NEGATIVE CONTROLS -- every rule attacked, every control asserted to fire")
    out("INPUT ORIGIN: SYNTHETIC.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)

    pinned = _bootstrap.synthetic_campaign("no_throttle")
    rolling = _bootstrap.synthetic_campaign("throttled")

    out.rule("1  labelling")
    for name, origin, count, pattern in (
        ("a synthetic figure labelled measured", Origin.SYNTHETIC, 3, "synthetic input"),
        ("a figure over zero inputs", Origin.MEASURED, 0, "empty input set"),
        ("a figure over negative inputs", Origin.MEASURED, -1, "empty input set"),
    ):
        fired = False
        try:
            require_measured(origin, name, n_inputs=count)
        except NotMeasuredError as error:
            fired = pattern in str(error)
        out.control(name, fired, must_fire=True)
    out.control(
        "a measured figure over real inputs",
        require_measured(Origin.MEASURED, "x", n_inputs=120) is not Origin.MEASURED,
        must_fire=False,
        detail="passes through unchanged",
    )

    out.rule("2  an unequilibrated run is excluded and named")
    import tempfile
    from pathlib import Path

    from thermalbudget.trace import load_trace

    from fixtures import synthetic_trace

    scratch = Path(tempfile.mkdtemp(prefix="thermalbudget-controls-"))
    short = load_trace(
        synthetic_trace.write_run(
            scratch, "soak_1.00_short", 1.0, 150, regime="no_throttle"
        ),
        origin=Origin.SYNTHETIC,
    )
    truth = sustained_truth(
        [*pinned.soaks(1.0), short], n_resamples=_bootstrap.RESAMPLES
    )
    out.control(
        "a 150 s loaded run is excluded from the sustained figure",
        short.label not in truth.included,
        must_fire=True,
        detail=f"included {list(truth.included)}",
    )
    out.control(
        "the exclusion is NAMED with its reason",
        any(label == short.label and "NOT_EQUILIBRATED" in reason
            for label, reason in truth.excluded),
        must_fire=True,
        detail=dict(truth.excluded).get(short.label, "")[:90],
    )
    for trace in pinned.soaks(1.0):
        out.control(
            f"the equilibrated run {trace.label} is NOT excluded",
            trace.label not in truth.included,
            must_fire=False,
            detail="included in the sustained figure",
        )
    fired = False
    try:
        sustained_truth([short], n_resamples=200)
    except ValueError as error:
        fired = "at least 2 equilibrated runs" in str(error)
    out.control(
        "a campaign with no equilibrated run refuses rather than averaging anyway",
        fired,
        must_fire=True,
    )

    out.rule("3  the headline verdict rule, attacked from both directions")
    base1 = burst_baseline(pinned.bursts(), n_resamples=_bootstrap.RESAMPLES)
    base3 = sustained_truth(pinned.soaks(1.0), n_resamples=_bootstrap.RESAMPLES)
    head = overstatement(
        base1.throughput_groups, base3.throughput_groups, n_resamples=_bootstrap.RESAMPLES
    )
    out(
        f"  pinned-clock fixture: overstatement {100.0 * head.overstatement.point:+.4f} "
        f"per cent [{100.0 * head.overstatement.ci_low:+.4f}, "
        f"{100.0 * head.overstatement.ci_high:+.4f}] per cent, verdict {head.verdict}"
    )
    out.control(
        "an interval containing zero is reported INDISTINGUISHABLE, not as a small gap",
        head.verdict == "INDISTINGUISHABLE" and head.overstatement.contains(0.0),
        must_fire=True,
        detail="the zero-gap outcome is a first-class verdict",
    )
    # Inject a KNOWN 30 per cent reduction into the sustained side. Anchored on
    # the true samples, not on a fitted value.
    scaled = tuple(0.7 * group for group in base3.throughput_groups)
    injected = overstatement(
        base1.throughput_groups, scaled, n_resamples=_bootstrap.RESAMPLES
    )
    expected = 1.0 / 0.7 - 1.0
    out.control(
        "an injected 30 per cent sustained shortfall is detected as a gap",
        injected.verdict == "OVERSTATES" and not injected.overstatement.contains(0.0),
        must_fire=True,
        detail=f"overstatement {100.0 * injected.overstatement.point:+.3f} per cent "
        f"against the injected {100.0 * expected:+.3f} per cent",
    )
    out.check(
        "the detected gap matches the injected one to within the interval",
        abs(injected.overstatement.point - expected) <= injected.overstatement.half_width,
        f"|{injected.overstatement.point:.5f} - {expected:.5f}| = "
        f"{abs(injected.overstatement.point - expected):.5f} <= "
        f"{injected.overstatement.half_width:.5f}",
    )
    # And the other direction: a sustained figure ABOVE the burst must be reported
    # as an understatement, not clamped to zero.
    inflated = tuple(1.3 * group for group in base3.throughput_groups)
    other = overstatement(
        base1.throughput_groups, inflated, n_resamples=_bootstrap.RESAMPLES
    )
    out.control(
        "an injected sustained EXCESS is reported as an understatement, not clamped",
        other.verdict == "UNDERSTATES" and other.overstatement.point < 0.0,
        must_fire=True,
        detail=f"overstatement {100.0 * other.overstatement.point:+.3f} per cent",
    )
    fired = False
    try:
        overstatement(
            [pinned["idle_pre"].throughput_iters_per_s] * 2,
            base3.throughput_groups,
            n_resamples=200,
        )
    except ValueError as error:
        fired = "non-positive" in str(error)
    out.control("a zero-throughput side refuses rather than dividing", fired, must_fire=True)

    out.rule("4  a clock channel that does not move yields no relation")
    degenerate = clock_temperature_relation(pinned.soaks() + pinned.bursts())
    out.control(
        "a pinned clock gives a DEGENERATE relation, not a fitted slope",
        not degenerate.identifiable,
        must_fire=True,
        detail=degenerate.reason[:110],
    )
    identifiable = clock_temperature_relation(rolling.soaks() + rolling.bursts())
    out.control(
        "a rolling-off clock is NOT reported degenerate",
        not identifiable.identifiable,
        must_fire=False,
        detail=f"fitted slope {identifiable.slope_khz_per_c:+.1f} kHz/degC",
    )
    exponent = proportionality_exponent(pinned.soaks(1.0) + pinned.bursts(), duty=1.0)
    out.control(
        "an unidentifiable exponent returns nan, NOT the assumed 1.0",
        not exponent.identifiable and not np.isfinite(exponent.exponent),
        must_fire=True,
        detail="returning the assumed value from a measurement function would make an "
        "assumption indistinguishable from a measurement",
    )
    base2 = rc_baseline(pinned.soaks(1.0), pinned.cooldowns(), pinned.bursts())
    out.control(
        "baseline 2 reduces to baseline 1 when the relation is degenerate, and says so",
        base2.reduces_to_baseline_1
        and any("NOT IDENTIFIABLE" in note for note in base2.notes),
        must_fire=True,
        detail=f"predicted {base2.predicted_throughput_iters_per_s:.2f} iters/s against "
        f"baseline 1's {base1.prediction_iters_per_s:.2f} iters/s",
    )

    out.rule("5  the learned-margin rule, attacked from both directions")
    dataset = build_dataset(pinned.soaks())
    result = leave_one_duty_out(dataset, pinned.soaks(), kind="ridge")
    out(
        f"  ridge pooled MAE {result.learned_mae_c:.4f} degC against RC "
        f"{result.rc_mae_c:.4f} degC on the same warm-up windows, margin "
        f"{result.margin_c:+.4f} degC"
    )
    wide = compare_against_rc(result, _reference(half_width=1000.0))
    out.control(
        "ANY margin inside a wide run-to-run interval is NOT DISTINGUISHABLE",
        wide.verdict == "NOT_DISTINGUISHABLE",
        must_fire=True,
        detail=f"ratio {wide.ratio:.6f} of the half-width",
    )
    narrow = compare_against_rc(result, _reference(half_width=1e-6))
    out.control(
        "a margin far outside the interval is attributed to the side that won",
        narrow.verdict in ("LEARNED_BETTER", "BASELINE_BETTER"),
        must_fire=True,
        detail=f"{narrow.verdict}, ratio {narrow.ratio:.1f}",
    )

    class _PositiveMargin:
        margin_c = 1000.0

    class _NegativeMargin:
        margin_c = -1000.0

    for name, stub, expected_verdict in (
        ("a large POSITIVE margin", _PositiveMargin(), "LEARNED_BETTER"),
        ("a large NEGATIVE margin", _NegativeMargin(), "BASELINE_BETTER"),
    ):
        verdict = compare_against_rc(stub, _reference(half_width=1.0))
        out.control(
            f"{name} outside the interval is named in the right direction",
            verdict.verdict == expected_verdict,
            must_fire=True,
            detail=f"{verdict.verdict}",
        )
    for name, stub in (
        ("a small positive margin", type("S", (), {"margin_c": 0.5})()),
        ("a small negative margin", type("S", (), {"margin_c": -0.5})()),
    ):
        verdict = compare_against_rc(stub, _reference(half_width=1.0))
        out.control(
            f"{name} inside the interval is NOT reported as a win",
            verdict.verdict != "NOT_DISTINGUISHABLE",
            must_fire=False,
            detail=f"{verdict.verdict}, ratio {verdict.ratio:.3f}",
        )

    out.rule("6  every unmeasured claim is refused by name")
    for claim in limits.UNMEASURED_BY_NAME:
        fired = False
        try:
            limits.refuse_unmeasured(claim.key)
        except limits.UnmeasuredClaimError as error:
            fired = claim.key in str(error) and claim.reason in str(error)
        out.control(f"claim {claim.key!r}", fired, must_fire=True)
    for mode in ("10W", "15W", "25W", "MAXN"):
        fired = False
        try:
            limits.power_mode_figure(mode)
        except limits.UnmeasuredClaimError:
            fired = True
        out.control(f"a figure at power mode {mode}", fired, must_fire=True)
    out.control(
        f"a figure at the measured mode {limits.POWER_MODE}",
        limits.POWER_MODE not in limits.power_mode_figure(limits.POWER_MODE),
        must_fire=False,
        detail="labelled MEASURED",
    )
    for ambient in (18.0, 22.5, 35.0):
        fired = False
        try:
            limits.ambient_figure(ambient)
        except limits.UnmeasuredClaimError:
            fired = True
        out.control(f"a figure at an ambient of {ambient:g} degC", fired, must_fire=True)

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
