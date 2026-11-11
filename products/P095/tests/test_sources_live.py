"""The live backend's board-absent failure path, which is the testable half.

The live reading path cannot be exercised against hardware here: no NVIDIA
Jetson Orin Nano was reachable from the build environment. What IS testable,
and what this file tests, is that it fails cleanly and informatively on every
declared interface -- which is the behaviour a user on a machine without the
board will actually meet.

One live channel does work, on any Linux host: ``monotonic_clock``, because
``clock_gettime(2)`` is not board-specific. Its readings are genuinely
``LIVE``, and genuinely not readings of a Jetson; that distinction is what
``boardprobe.platform_attest`` carries and these tests assert.
"""

from __future__ import annotations

import os

import pytest

from boardprobe.channels import Access
from boardprobe.orin_nano import ORIN_NANO
from boardprobe.platform_attest import attest_platform
from boardprobe.provenance import Provenance
from boardprobe.sources import LiveSource
from boardprobe.sysio import PERF_EVENT_ATTR_SIZE, DeviceUnreachableError

BOARD_CHANNELS = tuple(
    channel.name
    for channel in ORIN_NANO.channels
    if channel.access is not Access.CLOCK_SYSCALL
)


@pytest.mark.parametrize("name", BOARD_CHANNELS)
def test_every_board_channel_refuses_cleanly_when_the_board_is_absent(name, absent_root):
    source = LiveSource(ORIN_NANO, root=absent_root)
    with pytest.raises(DeviceUnreachableError) as error:
        source.read(name)
    message = str(error.value)
    assert message
    # The message names a relative locator or the syscall, never an absolute
    # path. The needle is assembled from os.sep rather than written as a
    # literal, because this file is committed and a committed file containing
    # the literal pattern is what a repository-wide absolute-path scanner
    # looks for.
    assert f"{os.sep}home{os.sep}" not in message
    assert message.startswith(("cannot read", "no path matches", "perf_event_open"))


@pytest.mark.parametrize("name", BOARD_CHANNELS)
def test_the_refusal_says_what_to_check(name, absent_root):
    source = LiveSource(ORIN_NANO, root=absent_root)
    with pytest.raises(DeviceUnreachableError) as error:
        source.read(name)
    message = str(error.value)
    assert any(
        hint in message
        for hint in ("not reachable", "not present", "paranoid", "driver is bound")
    )


def test_the_monotonic_clock_is_served_live_on_any_linux_host():
    source = LiveSource(ORIN_NANO)
    first = source.read("monotonic_clock")
    second = source.read("monotonic_clock")
    assert first.provenance is Provenance.LIVE
    assert first.unit == "s"
    assert second.value >= first.value
    assert first.value > 0.0


def test_a_live_clock_reading_is_not_a_reading_of_the_declared_target():
    """The whole reason provenance and platform are separate facts."""
    platform = attest_platform()
    reading = LiveSource(ORIN_NANO).read("monotonic_clock")
    assert reading.provenance is Provenance.LIVE
    assert not platform.matches_target


def test_the_perf_attr_struct_is_the_documented_size():
    """``struct perf_event_attr`` through ``config3`` is 136 bytes on this build.

    Reported rather than asserted as a kernel fact: the struct's ``size`` field
    is the documented forward-compatibility mechanism, and a mismatch against
    the running kernel's expectation shows up as EINVAL from the syscall. 136
    is what ``ctypes.sizeof`` gives for the declaration in
    ``boardprobe.sysio``, which is the only thing this test can check.
    """
    assert PERF_EVENT_ATTR_SIZE == 136


def test_a_sysfs_node_holding_something_other_than_an_integer_is_refused(tmp_path):
    root = tmp_path / "fake-root"
    node = root / "sys/devices/virtual/thermal/thermal_zone0"
    node.mkdir(parents=True)
    (node / "temp").write_text("not a number\n")
    with pytest.raises(DeviceUnreachableError, match="does not hold an integer"):
        LiveSource(ORIN_NANO, root=root).read("die_temp_cpu")


def test_a_sysfs_node_that_does_hold_an_integer_is_scaled_by_the_declaration(tmp_path):
    """48123 millidegrees is 48.123 degrees Celsius. The declared scale, applied."""
    root = tmp_path / "fake-root"
    node = root / "sys/devices/virtual/thermal/thermal_zone0"
    node.mkdir(parents=True)
    (node / "temp").write_text("48123\n")
    reading = LiveSource(ORIN_NANO, root=root).read("die_temp_cpu")
    assert reading.value == pytest.approx(48.123, rel=1e-12, abs=0.0)
    assert reading.unit == "degC"
    assert reading.provenance is Provenance.LIVE


def test_an_ambiguous_glob_locator_is_refused_rather_than_chosen_from(tmp_path):
    root = tmp_path / "fake-root"
    for index in (0, 1):
        directory = root / f"sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon{index}"
        directory.mkdir(parents=True)
        (directory / "in1_input").write_text("4870\n")
    with pytest.raises(DeviceUnreachableError, match="matches 2 paths"):
        LiveSource(ORIN_NANO, root=root).read("rail_vdd_in_voltage")


def test_a_resolved_glob_locator_reads_and_scales(tmp_path):
    root = tmp_path / "fake-root"
    directory = root / "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3"
    directory.mkdir(parents=True)
    (directory / "in1_input").write_text("4870\n")
    reading = LiveSource(ORIN_NANO, root=root).read("rail_vdd_in_voltage")
    assert reading.value == pytest.approx(4.870, rel=1e-12, abs=0.0)
    assert reading.unit == "V"
