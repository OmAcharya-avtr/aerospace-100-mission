"""Timing constraint arithmetic, unit-checked against hand calculations.

The timing checks are the only part of this package whose answers can be
verified with a pencil, so they are. Each case below states the arithmetic, the
expected verdict and the expected shortfall in seconds, and the script compares
them against the executor.

The tolerance convention is checked too, in both directions: a gap short by
less than ``TIME_TOL_S`` must not be reported, and a gap short by a thousand
times the tolerance must be. No boolean produced by a strict threshold
comparison is asserted anywhere near the boundary; the two probes sit decades
away from it on either side.
"""

from __future__ import annotations

import _bootstrap
from _bootstrap import Tee, header

ROOT = _bootstrap.bootstrap()

from cmdguard.model import (  # noqa: E402
    Case,
    CommandDictionary,
    CommandSequence,
    CommandSpec,
    InitialState,
    ModeMachine,
    ResourceLimits,
    SequenceStep,
)
from cmdguard.trace import TIME_TOL_S, ViolationKind, forward_execute  # noqa: E402

MACHINE = ModeMachine(modes=("NOMINAL",), transitions={"NOMINAL": ()})


def build(commands: list[CommandSpec], steps: list[tuple[float, str]], name: str) -> Case:
    """One inline case over ``commands`` with no mode gating and no resources."""
    return Case(
        name=name,
        dictionary=CommandDictionary(
            commands={spec.identifier: spec for spec in commands},
            resource_limits=ResourceLimits({}),
        ),
        machine=MACHINE,
        initial=InitialState(mode="NOMINAL", predicates={}),
        sequence=CommandSequence(
            tuple(SequenceStep(t_s=t, command=c) for t, c in steps)
        ),
    )


def main() -> int:
    tee = Tee("validate_timing_output.txt")
    header(tee, "cmdguard -- timing constraint arithmetic, unit-checked")
    tee("")
    tee(f"  absolute tolerance on every timing comparison: TIME_TOL_S = {TIME_TOL_S:.3e} s")

    # ------------------------------------------------------------------
    tee.rule("1. minimum inter-command gap (min_gap_after_s)")
    tee("  FIRST declares min_gap_after_s = 12.5 s. The gap owed to the NEXT command,")
    tee("  whatever it is, is 12.5 s measured from FIRST's execution time.")
    commands = [
        CommandSpec(identifier="FIRST", min_gap_after_s=12.5),
        CommandSpec(identifier="SECOND"),
    ]
    rows = [
        (0.0, 20.0, None, "20.0 - 0.0 = 12.5 + 7.5, so 7.5 s of margin"),
        (0.0, 12.5, None, "12.5 - 0.0 = 12.5 exactly: met, not violated"),
        (0.0, 12.0, 0.5, "12.0 - 0.0 = 12.0, short by 12.5 - 12.0 = 0.5 s"),
        (0.0, 0.0, 12.5, "simultaneous: short by the whole 12.5 s"),
        (100.0, 105.25, 7.25, "105.25 - 100.0 = 5.25, short by 12.5 - 5.25 = 7.25 s"),
    ]
    for t_first, t_second, expected, derivation in rows:
        case = build(commands, [(t_first, "FIRST"), (t_second, "SECOND")], "gap")
        report = forward_execute(case)
        gaps = report.of_kind(ViolationKind.TIMING_GAP)
        tee("")
        tee(f"  FIRST at t={t_first:.3f} s, SECOND at t={t_second:.3f} s")
        tee(f"    {derivation}")
        if expected is None:
            tee.check("no gap violation reported", len(gaps) == 0,
                      f"{len(gaps)} reported")
        else:
            ok = len(gaps) == 1 and abs((gaps[0].shortfall or -1.0) - expected) < 1e-12
            tee.check(
                f"gap violation with shortfall {expected:.6f} s",
                ok,
                f"measured {gaps[0].shortfall if gaps else 'none'}",
            )

    # ------------------------------------------------------------------
    tee.rule("2. minimum same-command interval (min_interval_s)")
    tee("  REPEATER declares min_interval_s = 300.0 s, measured from its own previous")
    tee("  execution and not from the previous step.")
    commands = [
        CommandSpec(identifier="REPEATER", min_interval_s=300.0),
        CommandSpec(identifier="FILLER"),
    ]
    case = build(
        commands,
        [(0.0, "REPEATER"), (50.0, "FILLER"), (200.0, "REPEATER"), (600.0, "REPEATER")],
        "interval",
    )
    report = forward_execute(case)
    intervals = report.of_kind(ViolationKind.TIMING_INTERVAL)
    tee("")
    tee("  REPEATER t=0, FILLER t=50, REPEATER t=200, REPEATER t=600")
    tee("    step 2: 200 - 0 = 200 s against 300 s -> short by 100 s")
    tee("    step 3: 600 - 200 = 400 s against 300 s -> met")
    tee(f"  interval violations reported: {len(intervals)}")
    tee.check("exactly one interval violation", len(intervals) == 1)
    if intervals:
        tee.check("it is at step 2", intervals[0].step == 2, f"step {intervals[0].step}")
        tee.check("shortfall is 100.0 s",
                  abs((intervals[0].shortfall or -1.0) - 100.0) < 1e-12,
                  f"{intervals[0].shortfall}")
        tee.check("the trace names step 0 as the previous execution",
                  intervals[0].trace[0].step == 0)
    tee("")
    tee("  the interval does not apply across different commands:")
    cross = build(
        [
            CommandSpec(identifier="A", min_interval_s=300.0),
            CommandSpec(identifier="B", min_interval_s=300.0),
        ],
        [(0.0, "A"), (1.0, "B"), (2.0, "A")],
        "cross",
    )
    cross_report = forward_execute(cross)
    tee(f"    A t=0, B t=1, A t=2 -> {cross_report.kind_counts()}")
    tee.check("only A's repeat is reported",
              cross_report.kind_counts() == {"timing_interval": 1}
              and cross_report.violations[0].step == 2)

    # ------------------------------------------------------------------
    tee.rule("3. the hold interval is half-open: [t_start, t_start + hold_s)")
    tee("  HOLDER takes exclusion group 'g' at t=0 for 10 s, so it is held over")
    tee("  [0, 10). TAKER at t=10 is legal. TAKER at t=9.5 is not.")
    commands = [
        CommandSpec(identifier="HOLDER", hold_s=10.0, exclusion_group="g"),
        CommandSpec(identifier="TAKER", hold_s=1.0, exclusion_group="g"),
    ]
    for t_taker, expect_violation, note in (
        (10.0, False, "exactly at the end of the hold: expired"),
        (9.5, True, "0.5 s inside the hold: still held"),
        (10.5, False, "0.5 s after the hold: expired"),
    ):
        case = build(commands, [(0.0, "HOLDER"), (t_taker, "TAKER")], "hold")
        report = forward_execute(case)
        exclusions = report.kind_counts().get("mutual_exclusion", 0)
        tee("")
        tee(f"  TAKER at t={t_taker:.3f} s -- {note}")
        tee.check(
            f"mutual exclusion {'reported' if expect_violation else 'not reported'}",
            (exclusions == 1) == expect_violation,
            f"{exclusions} reported",
        )

    # ------------------------------------------------------------------
    tee.rule("4. the tolerance band, probed from both sides")
    tee("  FIRST declares min_gap_after_s = 10.0 s. Two probes, both far from the")
    tee("  boundary relative to the tolerance, so no boolean is asserted on a value")
    tee("  that floating-point rounding could move across the threshold.")
    commands = [
        CommandSpec(identifier="FIRST", min_gap_after_s=10.0),
        CommandSpec(identifier="SECOND"),
    ]
    for factor, expect_violation in ((0.1, False), (1000.0, True)):
        offset = factor * TIME_TOL_S
        case = build(commands, [(0.0, "FIRST"), (10.0 - offset, "SECOND")], "tol")
        report = forward_execute(case)
        count = report.kind_counts().get("timing_gap", 0)
        tee("")
        tee(f"  gap short by {offset:.3e} s ({factor:g} x the tolerance)")
        tee.check(
            f"timing_gap {'reported' if expect_violation else 'not reported'}",
            (count == 1) == expect_violation,
            f"{count} reported",
        )

    # ------------------------------------------------------------------
    tee.rule("5. the generator's own spacing arithmetic")
    tee("  cmdguard.dataset._required_dt returns the minimum advance that satisfies")
    tee("  both declared spacings. Hand derivation against the base dictionary:")
    from cmdguard.dataset import _required_dt, base_dictionary

    dictionary = base_dictionary()
    check = dictionary.commands["PWR_BUS_CHECK"]
    tee(f"    PWR_BUS_CHECK declares min_gap_after_s = {check.min_gap_after_s:.1f} s, "
        f"min_interval_s = {check.min_interval_s:.1f} s")
    tee("    after one PWR_BUS_CHECK at t=0, the gap owed to any next command is 2.0 s")
    tee("    and the interval owed to another PWR_BUS_CHECK is 30.0 - 0.0 = 30.0 s")
    one = [SequenceStep(t_s=0.0, command="PWR_BUS_CHECK")]
    tee.check("no advance owed on an empty sequence",
              _required_dt(dictionary, [], "PWR_BUS_CHECK") == 0.0)
    tee.check("2.0 s owed to an unrelated next command",
              abs(_required_dt(dictionary, one, "SSR_DISARM") - 2.0) < 1e-12,
              f"{_required_dt(dictionary, one, 'SSR_DISARM')}")
    tee.check("30.0 s owed to a repeat",
              abs(_required_dt(dictionary, one, "PWR_BUS_CHECK") - 30.0) < 1e-12,
              f"{_required_dt(dictionary, one, 'PWR_BUS_CHECK')}")
    two = [
        SequenceStep(t_s=0.0, command="PWR_BUS_CHECK"),
        SequenceStep(t_s=25.0, command="SSR_DISARM"),
    ]
    tee("    with the last step at t=25, the interval owed from there is 30.0 - 25.0 = 5.0 s")
    tee.check("5.0 s owed to a repeat measured from the last step",
              abs(_required_dt(dictionary, two, "PWR_BUS_CHECK") - 5.0) < 1e-12,
              f"{_required_dt(dictionary, two, 'PWR_BUS_CHECK')}")

    code = tee.summarise()
    tee(f"transcript written to {tee.save()}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
