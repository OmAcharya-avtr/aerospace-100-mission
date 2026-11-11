"""The channel declaration table and its validation."""

from __future__ import annotations

import pytest

from boardprobe.channels import Access, BoardDeclaration, Channel, ChannelKind
from boardprobe.orin_nano import ORIN_NANO, PERF_EVENT_OPEN_SYSCALL


def good(**overrides):
    base = {
        "name": "die_temp_cpu",
        "kind": ChannelKind.DIE_TEMPERATURE,
        "unit": "degC",
        "access": Access.SYSFS_INTEGER,
        "locator": "sys/devices/virtual/thermal/thermal_zone0/temp",
        "scale": 1e-3,
        "raw_unit": "millidegC",
        "reference": "Linux kernel thermal sysfs ABI",
    }
    return Channel(**{**base, **overrides})


def test_the_declared_table_covers_all_four_source_classes():
    kinds = {channel.kind for channel in ORIN_NANO.channels}
    assert kinds == set(ChannelKind)
    for kind in ChannelKind:
        assert ORIN_NANO.of_kind(kind)


def test_every_channel_in_the_shipped_table_is_unverified():
    """The flag has to be flipped deliberately the day a board is reached."""
    assert ORIN_NANO.verified_count == 0
    assert all(not channel.verified_on_hardware for channel in ORIN_NANO.channels)


def test_every_channel_cites_a_reference_and_a_relative_locator():
    for channel in ORIN_NANO.channels:
        assert channel.reference
        assert not channel.locator.startswith("/")
        assert channel.scale > 0.0
        assert channel.raw_unit


def test_the_thermal_scale_is_the_millidegree_convention():
    """Linux thermal sysfs reports millidegrees Celsius; 1e-3 is the conversion."""
    for channel in ORIN_NANO.of_kind(ChannelKind.DIE_TEMPERATURE):
        assert channel.scale == pytest.approx(1e-3, rel=0.0, abs=0.0)
        assert channel.raw_unit == "millidegC"
        assert channel.unit == "degC"


def test_the_clock_scale_is_nanoseconds_to_seconds():
    clock = ORIN_NANO.channel("monotonic_clock")
    assert clock.scale == pytest.approx(1e-9, rel=0.0, abs=0.0)
    assert clock.access is Access.CLOCK_SYSCALL
    assert clock.locator == "CLOCK_MONOTONIC_RAW"


def test_the_cpufreq_scale_is_kilohertz_to_hertz():
    channel = ORIN_NANO.channel("cpu0_frequency")
    assert channel.scale == pytest.approx(1e3, rel=0.0, abs=0.0)
    assert channel.raw_unit == "kHz"


def test_both_perf_syscall_numbers_are_declared():
    """298 on x86_64 and 241 on aarch64, from the kernel syscall tables."""
    assert PERF_EVENT_OPEN_SYSCALL == {"x86_64": 298, "aarch64": 241}


def test_an_absolute_locator_is_refused():
    with pytest.raises(ValueError, match="must be relative"):
        good(locator="/sys/devices/virtual/thermal/thermal_zone0/temp")


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_a_non_positive_scale_is_refused(scale):
    with pytest.raises(ValueError, match="strictly positive"):
        good(scale=scale)


def test_an_unnamed_or_undocumented_channel_is_refused():
    with pytest.raises(ValueError, match="non-empty"):
        good(name="  ")
    with pytest.raises(ValueError, match="reference must name"):
        good(reference="")


def test_a_string_kind_or_access_is_refused():
    with pytest.raises(TypeError, match="ChannelKind member"):
        good(kind="die-temperature")
    with pytest.raises(TypeError, match="Access member"):
        good(access="sysfs-integer")


def test_a_duplicate_channel_name_is_refused():
    with pytest.raises(ValueError, match="duplicate channel name"):
        BoardDeclaration(model="board", channels=(good(), good()))


def test_a_board_with_no_channels_is_refused():
    with pytest.raises(ValueError, match="declares no channels"):
        BoardDeclaration(model="board", channels=())


def test_an_unknown_channel_name_lists_what_exists():
    with pytest.raises(KeyError, match="monotonic_clock"):
        ORIN_NANO.channel("die_temp_nonexistent")


def test_of_kind_refuses_a_string():
    with pytest.raises(TypeError, match="ChannelKind member"):
        ORIN_NANO.of_kind("die-temperature")
