"""Validation: the persistence/debounce logic reproduces two hand traces exactly.

Level 2 requirement: "persistence logic reproduces a hand-traced alarm sequence
exactly".  The traces are the ones written out in ``tests/test_persistence.py``;
this script prints them side by side with what the implementation produces, so a
reader can check every sample without running pytest.  There is no tolerance:
every quantity compared is an integer or an enumeration member.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from _reporting import Report  # noqa: E402

from telemetryool.limits import (  # noqa: E402
    AlarmState,
    BreachLevel,
    ChannelSpec,
    InvalidPolicy,
    LimitSet,
    OolChecker,
)

N = None
S, H, IN = BreachLevel.SOFT, BreachLevel.HARD, BreachLevel.IN_LIMIT
NOM, SA, HA = AlarmState.NOMINAL, AlarmState.SOFT_ALARM, AlarmState.HARD_ALARM

# Trace 1: soft limits +/- 2.0 K, hard limits +/- 4.0 K, persistence_soft 3,
# persistence_hard 2, clear_persistence 2, all samples valid, one mode.
TRACE1 = {
    "values": [0.0, 2.5, 2.5, 0.0, 2.5, 2.5, 2.5, 2.5, 5.0, 5.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0],
    "level": [IN, S, S, IN, S, S, S, S, H, H, S, S, IN, IN, IN, IN],
    "soft": [0, 1, 2, 0, 1, 2, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0],
    "hard": [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0],
    "below": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    "state": [NOM, NOM, NOM, NOM, NOM, NOM, SA, SA, SA, HA, HA, SA, SA, NOM, NOM, NOM],
    "event": ["", "", "", "", "", "", "RAISE", "", "", "RAISE", "", "CLEAR", "", "CLEAR",
              "", ""],
}

# Trace 2: mode-dependent limits, one invalid sample under InvalidPolicy.HOLD,
# latch surviving a mode change.  SAFE soft +/- 2.0 hard +/- 4.0;
# SCIENCE soft +/- 1.0 hard +/- 1.5.  persistence_soft 2, persistence_hard 1,
# clear_persistence 2.
TRACE2 = {
    "values": [0.0, 1.5, 1.5, 1.5, 9.9, 2.0, 0.0, 0.0, 0.0, 0.0],
    "valid": [True, True, True, True, False, True, True, True, True, True],
    "modes": ["SAFE", "SAFE", "SCIENCE", "SCIENCE", "SCIENCE", "SCIENCE", "SAFE", "SAFE",
              "SAFE", "SAFE"],
    "level": [IN, IN, S, S, N, H, IN, IN, IN, IN],
    "soft": [0, 0, 1, 2, 2, 2, 0, 0, 0, 0],
    "hard": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "below": [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    "state": [NOM, NOM, NOM, SA, SA, HA, HA, SA, SA, NOM],
    "event": ["", "", "", "RAISE", "", "RAISE", "", "CLEAR", "", "CLEAR"],
}


def run_trace(report: Report, label: str, checker: OolChecker, trace: dict) -> bool:
    samples = checker.update_series(
        trace["values"], trace.get("valid"), trace.get("modes")
    )
    report.line(f"{label}: {len(samples)} samples")
    report.line(
        f"{'idx':>4s} {'value':>8s} {'val':>4s} {'mode':>8s} "
        f"{'level(hand/got)':>18s} {'soft':>9s} {'hard':>9s} {'below':>9s} "
        f"{'state(hand/got)':>26s} {'event':>12s}"
    )
    mismatches = 0
    for i, got in enumerate(samples):
        event = "RAISE" if got.raised else ("CLEAR" if got.cleared else "")
        exp = (
            trace["level"][i], trace["soft"][i], trace["hard"][i],
            trace["below"][i], trace["state"][i], trace["event"][i],
        )
        obs = (got.level, got.soft_count, got.hard_count, got.below_count, got.state, event)
        ok = exp == obs
        mismatches += 0 if ok else 1
        lvl_h = "-" if exp[0] is None else exp[0].name
        lvl_g = "-" if obs[0] is None else obs[0].name
        valid = int(trace.get("valid", [True] * len(samples))[i])
        mode = trace.get("modes", [got.mode] * len(samples))[i]
        report.line(
            f"{i:4d} {trace['values'][i]:8.2f} {valid:4d} {mode:>8s} "
            f"{lvl_h + '/' + lvl_g:>18s} "
            f"{str(exp[1]) + '/' + str(obs[1]):>9s} "
            f"{str(exp[2]) + '/' + str(obs[2]):>9s} "
            f"{str(exp[3]) + '/' + str(obs[3]):>9s} "
            f"{exp[4].name + '/' + obs[4].name:>26s} "
            f"{(exp[5] or '-') + '/' + (obs[5] or '-'):>12s}"
            + ("" if ok else "   <-- MISMATCH")
        )
    return mismatches == 0


def main() -> int:
    report = Report(
        "validate_persistence_trace",
        "Persistence / debounce logic against hand-traced alarm sequences",
    )
    report.line("Tolerance: none.  Every compared quantity is an integer or an enum member.")
    report.line("Columns show hand-traced value / implementation value.")

    report.section("Trace 1 -- soft and hard escalation, two-step de-escalation")
    spec1 = ChannelSpec(
        name="TEMP_A",
        units="K",
        limits={"*": LimitSet(soft_low=-2.0, soft_high=2.0, hard_low=-4.0, hard_high=4.0)},
        persistence_soft=3,
        persistence_hard=2,
        clear_persistence=2,
    )
    ck1 = OolChecker(spec1)
    ok1 = run_trace(report, "trace 1", ck1, TRACE1)
    report.line("")
    report.check("trace 1 reproduced sample by sample", ok1)
    report.check(
        "trace 1 final latched state is NOMINAL with all counters zero",
        ck1.state is NOM and ck1.counters == (0, 0, 0),
        f"state={ck1.state.name} counters={ck1.counters}",
    )

    report.section("Trace 2 -- mode-dependent limits, invalid sample, latch across mode change")
    spec2 = ChannelSpec(
        name="TEMP_B",
        units="K",
        limits={
            "SAFE": LimitSet(soft_low=-2.0, soft_high=2.0, hard_low=-4.0, hard_high=4.0),
            "SCIENCE": LimitSet(soft_low=-1.0, soft_high=1.0, hard_low=-1.5, hard_high=1.5),
        },
        persistence_soft=2,
        persistence_hard=1,
        clear_persistence=2,
        invalid_policy=InvalidPolicy.HOLD,
        latch_across_mode_change=True,
    )
    ck2 = OolChecker(spec2, mode="SAFE")
    ok2 = run_trace(report, "trace 2", ck2, TRACE2)
    report.line("")
    report.check("trace 2 reproduced sample by sample", ok2)

    report.section("Invalid-sample policy, same three-sample breach run in each case")
    base = dict(units="-", limits={"*": LimitSet(-2.0, 2.0, -4.0, 4.0)}, persistence_soft=3)
    cases = {
        InvalidPolicy.HOLD: ([1, 2, 2, 3], SA),
        InvalidPolicy.RESET: ([1, 2, 0, 1], NOM),
    }
    report.line("values 2.5, 2.5, [invalid], 2.5 with persistence_soft = 3:")
    for policy, (expected_counts, expected_state) in cases.items():
        ck = OolChecker(ChannelSpec(name="X", invalid_policy=policy, **base))
        out = ck.update_series([2.5, 2.5, 0.0, 2.5], [True, True, False, True])
        counts = [s.soft_count for s in out]
        report.check(
            f"InvalidPolicy.{policy.name}: soft counts {counts}, final state {ck.state.name}",
            counts == expected_counts and ck.state is expected_state,
            f"expected counts {expected_counts} and state {expected_state.name}",
        )
    ck = OolChecker(
        ChannelSpec(
            name="X", units="-", limits={"*": LimitSet(-2.0, 2.0, -4.0, 4.0)},
            persistence_hard=1, invalid_policy=InvalidPolicy.BREACH,
        )
    )
    s = ck.update(0.0, valid=False)
    report.check(
        "InvalidPolicy.BREACH raises HARD_ALARM on a single invalid sample",
        s.level is H and s.state is HA,
        f"level={s.level.name} state={s.state.name}",
    )

    report.section("Clearing steps down exactly one level per satisfied clear count")
    ck = OolChecker(
        ChannelSpec(
            name="X", units="-", limits={"*": LimitSet(-2.0, 2.0, -4.0, 4.0)},
            persistence_hard=1, clear_persistence=3,
        )
    )
    ck.update(9.0)
    states = [ck.update(0.0).state.name for _ in range(9)]
    expected = ["HARD_ALARM"] * 2 + ["SOFT_ALARM"] * 3 + ["NOMINAL"] * 4
    report.line("  clear_persistence = 3, nine in-limit samples after a HARD_ALARM:")
    report.line(f"  hand: {expected}")
    report.line(f"  got : {states}")
    report.check("HARD -> SOFT -> NOMINAL, one level per three in-limit samples",
                 states == expected)
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
