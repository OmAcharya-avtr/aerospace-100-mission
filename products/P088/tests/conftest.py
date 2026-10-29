"""Shared fixtures. Every dictionary here is written out, not generated."""

from __future__ import annotations

import pytest

from tmtcdiff.model import Command, CommandArgument, Dictionary, Parameter


@pytest.fixture
def battery() -> Parameter:
    """A 12-bit signed battery temperature in kelvin, scale 1/16 K per count."""
    return Parameter(
        name="BATT_T_1",
        path="TM/PWR/BATT_T_1",
        description="battery one temperature",
        data_type="int",
        bit_offset=0,
        bit_width=12,
        scale=0.0625,
        unit="K",
        low_limit=250.0,
        high_limit=330.0,
        packet="PWR_HK",
    )


@pytest.fixture
def mode() -> Parameter:
    """A 4-bit enumerated attitude mode."""
    return Parameter(
        name="AOCS_MODE",
        path="TM/AOCS/AOCS_MODE",
        description="attitude control mode",
        data_type="enum",
        bit_offset=12,
        bit_width=4,
        enum=((0, "IDLE"), (1, "SAFE_MODE"), (2, "NOMINAL_MODE")),
        packet="AOCS_HK",
    )


@pytest.fixture
def heater() -> Command:
    """A two-argument heater command."""
    return Command(
        name="SET_HEATER",
        path="TC/PWR/SET_HEATER",
        description="set a heater duty cycle",
        arguments=(
            CommandArgument(name="HEATER_ID", position=0, bit_width=8),
            CommandArgument(name="DUTY", position=1, bit_width=8, unit="percent"),
        ),
    )


@pytest.fixture
def dictionary(battery: Parameter, mode: Parameter, heater: Command) -> Dictionary:
    """A two-parameter, one-command dictionary at version 1.0."""
    return Dictionary(version="1.0", parameters=(battery, mode), commands=(heater,))
