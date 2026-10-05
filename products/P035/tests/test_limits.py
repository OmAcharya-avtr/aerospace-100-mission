"""Tests for limit sets, channel specs, the checker and the monitor bank."""

from __future__ import annotations

import numpy as np
import pytest

from telemetryool.limits import (
    DEFAULT_MODE,
    AlarmState,
    BreachLevel,
    ChannelSpec,
    InvalidPolicy,
    LimitSet,
    OolChecker,
    TelemetryMonitor,
    scale_limits,
)


def test_limit_set_level_boundaries_are_exclusive() -> None:
    """A value exactly on a bound is IN_LIMIT: the comparisons are strict, so
    soft_high = 2.0 with value 2.0 is not a breach, and 2.0000001 is."""
    ls = LimitSet(soft_low=-2.0, soft_high=2.0, hard_low=-4.0, hard_high=4.0)
    assert ls.level(2.0) is BreachLevel.IN_LIMIT
    assert ls.level(-2.0) is BreachLevel.IN_LIMIT
    assert ls.level(2.0000001) is BreachLevel.SOFT
    assert ls.level(4.0) is BreachLevel.SOFT
    assert ls.level(4.0000001) is BreachLevel.HARD
    assert ls.level(-4.0000001) is BreachLevel.HARD


def test_hard_takes_precedence_over_soft() -> None:
    ls = LimitSet(soft_low=-1.0, soft_high=1.0, hard_low=-2.0, hard_high=2.0)
    assert ls.level(5.0) is BreachLevel.HARD


def test_one_sided_limits() -> None:
    """A limit set with only an upper hard bound never breaches below."""
    ls = LimitSet(hard_high=10.0)
    assert ls.level(-1e9) is BreachLevel.IN_LIMIT
    assert ls.level(10.5) is BreachLevel.HARD


def test_limit_set_ordering_is_enforced() -> None:
    with pytest.raises(ValueError, match="limits must satisfy"):
        LimitSet(soft_low=-1.0, soft_high=1.0, hard_low=0.5, hard_high=2.0)
    with pytest.raises(ValueError, match="limits must satisfy"):
        LimitSet(soft_low=3.0, soft_high=1.0)


def test_symmetric_constructor() -> None:
    ls = LimitSet.symmetric(100.0, 2.0, 5.0)
    assert (ls.soft_low, ls.soft_high, ls.hard_low, ls.hard_high) == (98.0, 102.0, 95.0, 105.0)
    with pytest.raises(ValueError, match="must be >="):
        LimitSet.symmetric(0.0, 5.0, 2.0)
    with pytest.raises(ValueError, match="soft half-width must be >= 0"):
        LimitSet.symmetric(0.0, -1.0, 2.0)


def test_scale_limits_preserves_none() -> None:
    scaled = scale_limits(LimitSet(soft_high=2.0, hard_high=4.0), 10.0)
    assert (scaled.soft_high, scaled.hard_high) == (20.0, 40.0)
    assert scaled.soft_low is None and scaled.hard_low is None
    with pytest.raises(ValueError, match="factor must be > 0"):
        scale_limits(LimitSet(soft_high=1.0), 0.0)


def test_channel_spec_validation() -> None:
    ls = LimitSet(soft_high=1.0)
    with pytest.raises(ValueError, match="non-empty string"):
        ChannelSpec("", "K", {"*": ls})
    with pytest.raises(ValueError, match="limits mapping is empty"):
        ChannelSpec("A", "K", {})
    with pytest.raises(ValueError, match="persistence_soft must be an integer >= 1"):
        ChannelSpec("A", "K", {"*": ls}, persistence_soft=0)
    with pytest.raises(ValueError, match="clear_persistence must be an integer >= 1"):
        ChannelSpec("A", "K", {"*": ls}, clear_persistence=-1)
    with pytest.raises(TypeError, match="must be a LimitSet"):
        ChannelSpec("A", "K", {"*": (1.0, 2.0)})  # type: ignore[dict-item]


def test_limits_for_falls_back_to_default_mode() -> None:
    cruise = LimitSet(soft_high=1.0)
    fallback = LimitSet(soft_high=9.0)
    spec = ChannelSpec("A", "K", {"CRUISE": cruise, DEFAULT_MODE: fallback})
    assert spec.limits_for("CRUISE") is cruise
    assert spec.limits_for("ANYTHING_ELSE") is fallback


def test_missing_mode_without_fallback_raises() -> None:
    spec = ChannelSpec("A", "K", {"CRUISE": LimitSet(soft_high=1.0)})
    with pytest.raises(KeyError, match="no limit set for mode"):
        spec.limits_for("SAFE")


def test_non_finite_value_is_rejected_rather_than_guessed() -> None:
    spec = ChannelSpec("A", "K", {"*": LimitSet(soft_high=1.0)})
    ck = OolChecker(spec)
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="is not finite"):
            ck.update(bad)


def test_non_finite_value_is_accepted_when_marked_invalid() -> None:
    spec = ChannelSpec("A", "K", {"*": LimitSet(soft_high=1.0)})
    ck = OolChecker(spec)
    sample = ck.update(np.nan, valid=False)
    assert sample.level is None
    assert sample.state is AlarmState.NOMINAL


def test_reset_clears_everything() -> None:
    spec = ChannelSpec("A", "K", {"*": LimitSet(soft_high=1.0, hard_high=2.0)})
    ck = OolChecker(spec)
    ck.update(5.0)
    assert ck.state is AlarmState.HARD_ALARM
    ck.reset(mode="SAFE")
    assert ck.state is AlarmState.NOMINAL
    assert ck.counters == (0, 0, 0)
    assert ck.mode == "SAFE"


def test_update_series_validation() -> None:
    spec = ChannelSpec("A", "K", {"*": LimitSet(soft_high=1.0)})
    ck = OolChecker(spec)
    with pytest.raises(ValueError, match="values must be 1-D"):
        ck.update_series(np.zeros((2, 2)))
    with pytest.raises(ValueError, match=r"valid must have shape"):
        ck.update_series([0.0, 0.0], [True])
    with pytest.raises(ValueError, match="modes must have length"):
        ck.update_series([0.0, 0.0], None, ["SAFE"])


def _bank() -> TelemetryMonitor:
    ls = LimitSet(soft_low=-1.0, soft_high=1.0, hard_low=-2.0, hard_high=2.0)
    return TelemetryMonitor(
        [ChannelSpec("BUS_V", "V", {"*": ls}), ChannelSpec("TEMP", "K", {"*": ls})]
    )


def test_monitor_reports_per_channel_states() -> None:
    mon = _bank()
    out = mon.update({"BUS_V": 0.0, "TEMP": 3.0})
    assert out["BUS_V"].state is AlarmState.NOMINAL
    assert out["TEMP"].state is AlarmState.HARD_ALARM
    assert mon.active_alarms() == {"TEMP": AlarmState.HARD_ALARM}


def test_monitor_rejects_incomplete_and_unknown_frames() -> None:
    mon = _bank()
    with pytest.raises(KeyError, match="missing channel"):
        mon.update({"BUS_V": 0.0})
    with pytest.raises(KeyError, match="unknown channel"):
        mon.update({"BUS_V": 0.0, "TEMP": 0.0, "GHOST": 1.0})


def test_monitor_rejects_duplicate_names_and_empty_spec_list() -> None:
    ls = LimitSet(soft_high=1.0)
    with pytest.raises(ValueError, match="duplicate channel name"):
        TelemetryMonitor([ChannelSpec("A", "K", {"*": ls}), ChannelSpec("A", "K", {"*": ls})])
    with pytest.raises(ValueError, match="at least one ChannelSpec"):
        TelemetryMonitor([])


def test_monitor_mode_propagates_to_every_channel() -> None:
    hot = LimitSet(soft_high=1.0, hard_high=2.0)
    cold = LimitSet(soft_high=10.0, hard_high=20.0)
    specs = [
        ChannelSpec(name, "K", {"HOT": hot, "COLD": cold})
        for name in ("A", "B")
    ]
    mon = TelemetryMonitor(specs, mode="COLD")
    assert all(s.level is BreachLevel.IN_LIMIT for s in mon.update({"A": 5.0, "B": 5.0}).values())
    mon.set_mode("HOT")
    assert all(s.level is BreachLevel.HARD for s in mon.update({"A": 5.0, "B": 5.0}).values())
    assert mon.mode == "HOT"
    mon.reset()
    assert mon.active_alarms() == {}


def test_monitor_validity_defaults_to_true() -> None:
    mon = _bank()
    out = mon.update({"BUS_V": 3.0, "TEMP": 0.0}, valid={"BUS_V": False})
    assert out["BUS_V"].level is None
    assert out["TEMP"].level is BreachLevel.IN_LIMIT


def test_invalid_policy_enum_is_total() -> None:
    assert set(InvalidPolicy) == {
        InvalidPolicy.HOLD,
        InvalidPolicy.RESET,
        InvalidPolicy.BREACH,
    }
