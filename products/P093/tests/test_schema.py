"""Header and enum validation. A column that moved is rejected, not absorbed."""

from __future__ import annotations

import pytest

from powerenvelope.schema import (
    CURRENT_LSB_MA,
    DECISION_COLUMNS,
    MONITOR_COLUMNS,
    NOMINAL_RAIL_MV,
    POWER_COLUMNS,
    REJECT_REASONS,
    SchemaError,
    validate_decision_header,
    validate_monitor_header,
    validate_power_header,
    validate_reject_reason,
)


def test_declared_power_header_is_the_campaign_header():
    # Hand-checked against the header literal in data/campaign.py role_sampler().
    assert POWER_COLUMNS == (
        "t_mono",
        "vdd_in_mv",
        "vdd_in_ma",
        "cpu_gpu_cv_mv",
        "cpu_gpu_cv_ma",
        "soc_mv",
        "soc_ma",
        "vdd_in_mw",
    )


def test_declared_monitor_header_is_the_campaign_header():
    assert MONITOR_COLUMNS == (
        "t_mono",
        "cpu_busy_fraction",
        "tj_c",
        "load1",
        "clk0_khz",
        "clk1_khz",
        "clk2_khz",
        "clk3_khz",
        "clk4_khz",
        "clk5_khz",
    )


def test_declared_decision_header_is_the_campaign_header():
    assert DECISION_COLUMNS == (
        "t_start",
        "t_end",
        "cycle",
        "frame_idx",
        "score",
        "n_candidates",
        "area",
        "touches_border",
        "accepted",
        "reject_reason",
    )


@pytest.mark.parametrize(
    ("validator", "columns"),
    [
        (validate_power_header, POWER_COLUMNS),
        (validate_monitor_header, MONITOR_COLUMNS),
        (validate_decision_header, DECISION_COLUMNS),
    ],
)
def test_exact_header_accepted(validator, columns):
    validator(columns)


@pytest.mark.parametrize(
    ("validator", "columns"),
    [
        (validate_power_header, POWER_COLUMNS),
        (validate_monitor_header, MONITOR_COLUMNS),
        (validate_decision_header, DECISION_COLUMNS),
    ],
)
def test_whitespace_is_stripped(validator, columns):
    validator(tuple(f" {name} " for name in columns))


@pytest.mark.parametrize(
    ("validator", "columns"),
    [
        (validate_power_header, POWER_COLUMNS),
        (validate_monitor_header, MONITOR_COLUMNS),
        (validate_decision_header, DECISION_COLUMNS),
    ],
)
def test_missing_column_rejected(validator, columns):
    with pytest.raises(SchemaError, match="missing="):
        validator(columns[:-1])


@pytest.mark.parametrize(
    ("validator", "columns"),
    [
        (validate_power_header, POWER_COLUMNS),
        (validate_monitor_header, MONITOR_COLUMNS),
        (validate_decision_header, DECISION_COLUMNS),
    ],
)
def test_extra_column_rejected(validator, columns):
    with pytest.raises(SchemaError, match="unexpected="):
        validator((*columns, "surprise"))


@pytest.mark.parametrize(
    ("validator", "columns"),
    [
        (validate_power_header, POWER_COLUMNS),
        (validate_monitor_header, MONITOR_COLUMNS),
        (validate_decision_header, DECISION_COLUMNS),
    ],
)
def test_reordered_header_rejected_with_its_own_message(validator, columns):
    swapped = (columns[1], columns[0], *columns[2:])
    with pytest.raises(SchemaError, match="right names in the wrong order"):
        validator(swapped)


@pytest.mark.parametrize("reason", REJECT_REASONS)
def test_declared_reject_reasons_accepted(reason):
    assert validate_reject_reason(reason) == reason


def test_unknown_reject_reason_rejected():
    with pytest.raises(SchemaError, match="cannot write"):
        validate_reject_reason("timeout")


def test_quantisation_constant_is_the_boards_eight_milliamp_lsb():
    # 8 mA at the nominal 5 V rail is a 40 mW least-significant bit. Hand
    # calculation: 5000 mV * 8 mA / 1000 = 40.0 mW.
    assert CURRENT_LSB_MA == 8.0
    assert NOMINAL_RAIL_MV * CURRENT_LSB_MA / 1000.0 == pytest.approx(40.0, rel=0.0, abs=0.0)
