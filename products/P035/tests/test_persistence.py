"""Hand-traced persistence/debounce sequences, reproduced exactly.

These are the Level 2 validation requirement "persistence logic reproduces a
hand-traced alarm sequence exactly".  The traces below were worked out on paper
from the update rule documented in :meth:`telemetryool.limits.OolChecker.update`
and are asserted sample by sample with no tolerance, because every quantity is
an integer or an enumeration member.
"""

from __future__ import annotations

import pytest

from telemetryool.limits import (
    AlarmState,
    BreachLevel,
    ChannelSpec,
    InvalidPolicy,
    LimitSet,
    OolChecker,
)

# --------------------------------------------------------------------------- #
# Trace 1: soft escalation, hard escalation, two-step de-escalation.
#
# Channel: soft limits +/- 2.0, hard limits +/- 4.0 (units: K).
# persistence_soft = 3, persistence_hard = 2, clear_persistence = 2.
# All samples valid, mode constant.
#
#  i  value  level      soft hard below  state after    event
#  0   0.0   IN_LIMIT     0    0     0   NOMINAL        -
#  1   2.5   SOFT         1    0     0   NOMINAL        -
#  2   2.5   SOFT         2    0     0   NOMINAL        -
#  3   0.0   IN_LIMIT     0    0     0   NOMINAL        -      (run broken)
#  4   2.5   SOFT         1    0     0   NOMINAL        -
#  5   2.5   SOFT         2    0     0   NOMINAL        -
#  6   2.5   SOFT         3    0     0   SOFT_ALARM     RAISE  (soft >= 3)
#  7   2.5   SOFT         3    0     0   SOFT_ALARM     -      (soft saturates)
#  8   5.0   HARD         3    1     0   SOFT_ALARM     -      (hard 1 < 2)
#  9   5.0   HARD         3    2     0   HARD_ALARM     RAISE  (hard >= 2)
# 10   2.5   SOFT         3    0     1   HARD_ALARM     -      (level 1 < state 2)
# 11   2.5   SOFT         3    0     0   SOFT_ALARM     CLEAR  (below reached 2)
# 12   0.0   IN_LIMIT     0    0     1   SOFT_ALARM     -
# 13   0.0   IN_LIMIT     0    0     0   NOMINAL        CLEAR  (below reached 2)
# 14   0.0   IN_LIMIT     0    0     0   NOMINAL        -
# 15   0.0   IN_LIMIT     0    0     0   NOMINAL        -
# --------------------------------------------------------------------------- #

TRACE1_VALUES = [0.0, 2.5, 2.5, 0.0, 2.5, 2.5, 2.5, 2.5, 5.0, 5.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0]
TRACE1_LEVEL = [
    BreachLevel.IN_LIMIT, BreachLevel.SOFT, BreachLevel.SOFT, BreachLevel.IN_LIMIT,
    BreachLevel.SOFT, BreachLevel.SOFT, BreachLevel.SOFT, BreachLevel.SOFT,
    BreachLevel.HARD, BreachLevel.HARD, BreachLevel.SOFT, BreachLevel.SOFT,
    BreachLevel.IN_LIMIT, BreachLevel.IN_LIMIT, BreachLevel.IN_LIMIT, BreachLevel.IN_LIMIT,
]
TRACE1_SOFT = [0, 1, 2, 0, 1, 2, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0]
TRACE1_HARD = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0]
TRACE1_BELOW = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
TRACE1_STATE = [
    AlarmState.NOMINAL, AlarmState.NOMINAL, AlarmState.NOMINAL, AlarmState.NOMINAL,
    AlarmState.NOMINAL, AlarmState.NOMINAL, AlarmState.SOFT_ALARM, AlarmState.SOFT_ALARM,
    AlarmState.SOFT_ALARM, AlarmState.HARD_ALARM, AlarmState.HARD_ALARM,
    AlarmState.SOFT_ALARM, AlarmState.SOFT_ALARM, AlarmState.NOMINAL,
    AlarmState.NOMINAL, AlarmState.NOMINAL,
]
TRACE1_EVENT = [
    "", "", "", "", "", "", "RAISE", "", "", "RAISE", "", "CLEAR", "", "CLEAR", "", "",
]


@pytest.fixture
def trace1_checker() -> OolChecker:
    spec = ChannelSpec(
        name="TEMP_A",
        units="K",
        limits={"*": LimitSet(soft_low=-2.0, soft_high=2.0, hard_low=-4.0, hard_high=4.0)},
        persistence_soft=3,
        persistence_hard=2,
        clear_persistence=2,
    )
    return OolChecker(spec)


def test_trace1_reproduces_hand_trace_exactly(trace1_checker: OolChecker) -> None:
    samples = trace1_checker.update_series(TRACE1_VALUES)
    assert len(samples) == len(TRACE1_VALUES)
    for i, s in enumerate(samples):
        event = "RAISE" if s.raised else ("CLEAR" if s.cleared else "")
        assert (s.level, s.soft_count, s.hard_count, s.below_count, s.state, event) == (
            TRACE1_LEVEL[i],
            TRACE1_SOFT[i],
            TRACE1_HARD[i],
            TRACE1_BELOW[i],
            TRACE1_STATE[i],
            TRACE1_EVENT[i],
        ), f"divergence at sample {i}"


def test_trace1_indices_are_sequential(trace1_checker: OolChecker) -> None:
    samples = trace1_checker.update_series(TRACE1_VALUES)
    assert [s.index for s in samples] == list(range(len(TRACE1_VALUES)))


def test_trace1_final_state_is_nominal(trace1_checker: OolChecker) -> None:
    trace1_checker.update_series(TRACE1_VALUES)
    assert trace1_checker.state is AlarmState.NOMINAL
    assert trace1_checker.counters == (0, 0, 0)


# --------------------------------------------------------------------------- #
# Trace 2: mode-dependent limits, an invalid sample under InvalidPolicy.HOLD,
# and a latch that survives a mode change.
#
# Modes: SAFE    -> soft +/- 2.0, hard +/- 4.0
#        SCIENCE -> soft +/- 1.0, hard +/- 1.5
# persistence_soft = 2, persistence_hard = 1, clear_persistence = 2,
# invalid_policy = HOLD, latch_across_mode_change = True.
#
#  i  value valid mode     level      soft hard below  state after  event
#  0   0.0   T    SAFE     IN_LIMIT     0    0     0   NOMINAL      -
#  1   1.5   T    SAFE     IN_LIMIT     0    0     0   NOMINAL      -   (1.5 < 2.0)
#  2   1.5   T    SCIENCE  SOFT         1    0     0   NOMINAL      -   (counters zeroed
#                                                                        by mode change,
#                                                                        then 1.5 > 1.0)
#  3   1.5   T    SCIENCE  SOFT         2    0     0   SOFT_ALARM   RAISE
#  4   9.9   F    SCIENCE  None         2    0     0   SOFT_ALARM   -   (HOLD: untouched)
#  5   2.0   T    SCIENCE  HARD         2    1     0   HARD_ALARM   RAISE (2.0 > 1.5)
#  6   0.0   T    SAFE     IN_LIMIT     0    0     1   HARD_ALARM   -   (mode change zeroes
#                                                                        counters, latch kept)
#  7   0.0   T    SAFE     IN_LIMIT     0    0     0   SOFT_ALARM   CLEAR
#  8   0.0   T    SAFE     IN_LIMIT     0    0     1   SOFT_ALARM   -
#  9   0.0   T    SAFE     IN_LIMIT     0    0     0   NOMINAL      CLEAR
# --------------------------------------------------------------------------- #

TRACE2_VALUES = [0.0, 1.5, 1.5, 1.5, 9.9, 2.0, 0.0, 0.0, 0.0, 0.0]
TRACE2_VALID = [True, True, True, True, False, True, True, True, True, True]
TRACE2_MODES = [
    "SAFE", "SAFE", "SCIENCE", "SCIENCE", "SCIENCE", "SCIENCE", "SAFE", "SAFE", "SAFE", "SAFE",
]
TRACE2_LEVEL = [
    BreachLevel.IN_LIMIT, BreachLevel.IN_LIMIT, BreachLevel.SOFT, BreachLevel.SOFT,
    None, BreachLevel.HARD, BreachLevel.IN_LIMIT, BreachLevel.IN_LIMIT,
    BreachLevel.IN_LIMIT, BreachLevel.IN_LIMIT,
]
TRACE2_SOFT = [0, 0, 1, 2, 2, 2, 0, 0, 0, 0]
TRACE2_HARD = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
TRACE2_BELOW = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0]
TRACE2_STATE = [
    AlarmState.NOMINAL, AlarmState.NOMINAL, AlarmState.NOMINAL, AlarmState.SOFT_ALARM,
    AlarmState.SOFT_ALARM, AlarmState.HARD_ALARM, AlarmState.HARD_ALARM,
    AlarmState.SOFT_ALARM, AlarmState.SOFT_ALARM, AlarmState.NOMINAL,
]
TRACE2_EVENT = ["", "", "", "RAISE", "", "RAISE", "", "CLEAR", "", "CLEAR"]

SAFE_LIMITS = LimitSet(soft_low=-2.0, soft_high=2.0, hard_low=-4.0, hard_high=4.0)
SCIENCE_LIMITS = LimitSet(soft_low=-1.0, soft_high=1.0, hard_low=-1.5, hard_high=1.5)


@pytest.fixture
def trace2_checker() -> OolChecker:
    spec = ChannelSpec(
        name="TEMP_B",
        units="K",
        limits={"SAFE": SAFE_LIMITS, "SCIENCE": SCIENCE_LIMITS},
        persistence_soft=2,
        persistence_hard=1,
        clear_persistence=2,
        invalid_policy=InvalidPolicy.HOLD,
        latch_across_mode_change=True,
    )
    return OolChecker(spec, mode="SAFE")


def test_trace2_reproduces_hand_trace_exactly(trace2_checker: OolChecker) -> None:
    samples = trace2_checker.update_series(TRACE2_VALUES, TRACE2_VALID, TRACE2_MODES)
    for i, s in enumerate(samples):
        event = "RAISE" if s.raised else ("CLEAR" if s.cleared else "")
        assert (s.level, s.soft_count, s.hard_count, s.below_count, s.state, event) == (
            TRACE2_LEVEL[i],
            TRACE2_SOFT[i],
            TRACE2_HARD[i],
            TRACE2_BELOW[i],
            TRACE2_STATE[i],
            TRACE2_EVENT[i],
        ), f"divergence at sample {i}"


def test_trace2_mode_is_recorded_per_sample(trace2_checker: OolChecker) -> None:
    samples = trace2_checker.update_series(TRACE2_VALUES, TRACE2_VALID, TRACE2_MODES)
    assert [s.mode for s in samples] == TRACE2_MODES


def test_latch_cleared_by_mode_change_when_configured() -> None:
    """With latch_across_mode_change False the mode change clears the latch too.

    Same first four samples as trace 2: the latch is SOFT_ALARM at i = 3, and the
    SAFE sample at i = 4 then finds NOMINAL instead of SOFT_ALARM.
    """
    spec = ChannelSpec(
        name="TEMP_C",
        units="K",
        limits={"SAFE": SAFE_LIMITS, "SCIENCE": SCIENCE_LIMITS},
        persistence_soft=2,
        persistence_hard=1,
        clear_persistence=2,
        latch_across_mode_change=False,
    )
    ck = OolChecker(spec, mode="SAFE")
    states = [
        ck.update(v, True, m).state
        for v, m in zip(
            [0.0, 1.5, 1.5, 1.5, 0.0],
            ["SAFE", "SAFE", "SCIENCE", "SCIENCE", "SAFE"],
            strict=True,
        )
    ]
    assert states == [
        AlarmState.NOMINAL,
        AlarmState.NOMINAL,
        AlarmState.NOMINAL,
        AlarmState.SOFT_ALARM,
        AlarmState.NOMINAL,
    ]


def test_invalid_policy_reset_breaks_the_run() -> None:
    """RESET zeroes the raise counters, so a breach run split by an invalid sample
    never reaches persistence_soft = 3.

    values  2.5 2.5 [invalid] 2.5 2.5 -> soft counts 1, 2, 0, 1, 2 and no alarm.
    """
    spec = ChannelSpec(
        "X", "-", {"*": LimitSet(-2.0, 2.0, -4.0, 4.0)},
        persistence_soft=3, invalid_policy=InvalidPolicy.RESET,
    )
    ck = OolChecker(spec)
    out = ck.update_series([2.5, 2.5, 0.0, 2.5, 2.5], [True, True, False, True, True])
    assert [s.soft_count for s in out] == [1, 2, 0, 1, 2]
    assert all(s.state is AlarmState.NOMINAL for s in out)


def test_invalid_policy_hold_lets_the_run_continue() -> None:
    """HOLD leaves the counter alone, so the same sequence does alarm at i = 4.

    values  2.5 2.5 [invalid] 2.5 -> soft counts 1, 2, 2, 3 -> SOFT_ALARM at i = 3.
    """
    spec = ChannelSpec(
        "X", "-", {"*": LimitSet(-2.0, 2.0, -4.0, 4.0)},
        persistence_soft=3, invalid_policy=InvalidPolicy.HOLD,
    )
    ck = OolChecker(spec)
    out = ck.update_series([2.5, 2.5, 0.0, 2.5], [True, True, False, True])
    assert [s.soft_count for s in out] == [1, 2, 2, 3]
    assert out[-1].state is AlarmState.SOFT_ALARM
    assert out[-1].raised


def test_invalid_policy_breach_is_fail_safe() -> None:
    """BREACH counts an invalid sample as HARD, so one invalid sample raises
    immediately when persistence_hard = 1."""
    spec = ChannelSpec(
        "X", "-", {"*": LimitSet(-2.0, 2.0, -4.0, 4.0)},
        persistence_hard=1, invalid_policy=InvalidPolicy.BREACH,
    )
    ck = OolChecker(spec)
    s = ck.update(0.0, valid=False)
    assert s.level is BreachLevel.HARD
    assert s.state is AlarmState.HARD_ALARM
    assert s.raised


def test_clearing_is_one_level_per_satisfied_count() -> None:
    """From HARD_ALARM with clear_persistence = 3, nine in-limit samples step the
    latch down exactly twice: HARD -> SOFT at i = 2 and SOFT -> NOMINAL at i = 5."""
    spec = ChannelSpec(
        "X", "-", {"*": LimitSet(-2.0, 2.0, -4.0, 4.0)},
        persistence_hard=1, clear_persistence=3,
    )
    ck = OolChecker(spec)
    ck.update(9.0)
    assert ck.state is AlarmState.HARD_ALARM
    states = [ck.update(0.0).state for _ in range(9)]
    assert states == [
        AlarmState.HARD_ALARM, AlarmState.HARD_ALARM, AlarmState.SOFT_ALARM,
        AlarmState.SOFT_ALARM, AlarmState.SOFT_ALARM, AlarmState.NOMINAL,
        AlarmState.NOMINAL, AlarmState.NOMINAL, AlarmState.NOMINAL,
    ]
