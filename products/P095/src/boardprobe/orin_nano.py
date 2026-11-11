"""The NVIDIA Jetson Orin Nano channel table, declared from documentation.

EVERY PATH, COUNTER NAME, RAW UNIT AND SCALE IN THIS FILE IS DECLARED FROM
PUBLISHED DOCUMENTATION AND IS **UNVERIFIED ON HARDWARE**. No Jetson Orin Nano
was reachable from the environment this package was built in. Not one of these
files was opened; not one of these counters was read. ``verified_on_hardware``
is ``False`` on every channel in the table and
``validation/validate_backends.py`` asserts that it is, so that the day a
reading is actually taken the flag has to be flipped deliberately rather than
having been true all along.

The references, in full, so a reader on a board can check each line
-------------------------------------------------------------------
1. **Thermal zones.** Linux kernel ABI, ``Documentation/ABI/testing/sysfs-api-thermal``
   and ``Documentation/driver-api/thermal/sysfs-api.rst``: a thermal zone
   exposes ``/sys/class/thermal/thermal_zone<N>/temp``, an integer in
   **millidegrees Celsius**, and ``.../type``, the zone's name. The
   ``/sys/devices/virtual/thermal/thermal_zone<N>/`` form used here is the
   device path the ``/sys/class/thermal`` symlinks point at. Hence
   ``scale = 1e-3`` from millidegrees to degrees Celsius. Zone NAMES on Jetson
   Orin (``cpu-thermal``, ``gpu-thermal``, ``cv0-thermal`` ... ``tj-thermal``)
   are from the NVIDIA Jetson Linux Developer Guide, "Thermal Management"
   chapter, Orin series. The ZONE INDEX of a given name is not fixed by any
   document and must be discovered at run time by reading each ``type`` file;
   the indices below are the documented default ordering and are the single
   most likely thing in this table to be wrong on a particular L4T release.
2. **Rail power.** Linux kernel hwmon documentation,
   ``Documentation/hwmon/ina3221.rst``: the INA3221 driver exposes, per
   channel ``i`` in 1..3, ``in<i>_input`` (**bus voltage, millivolts**),
   ``curr<i>_input`` (**current, milliamperes**) and ``in<i>_label`` (the rail
   name from device tree). The driver binds at
   ``/sys/bus/i2c/drivers/ina3221/<bus>-<addr>/hwmon/hwmon<N>/``. The Jetson
   Linux Developer Guide, "Software-Based Power Consumption Modeling" chapter,
   gives the Orin Nano Developer Kit module as carrying a single INA3221 at
   I2C address ``0x40`` on bus 1, with the three rails labelled ``VDD_IN``,
   ``VDD_CPU_GPU_CV`` and ``VDD_SOC``. Power is **not** exposed by the driver
   on this part and is derived here as ``in<i>_input * curr<i>_input``, which
   in raw units is mV x mA = microwatts, hence ``scale = 1e-6`` to watts. The
   ``hwmon<N>`` index is dynamic; see ``POWER_DIRECTORY_GLOB``.
3. **Monotonic clock.** ``clock_gettime(2)``, ``CLOCK_MONOTONIC_RAW``: "a raw
   hardware-based time that is not subject to NTP adjustments or the
   incremental adjustments performed by adjtime(3)". Nanoseconds, hence
   ``scale = 1e-9`` to seconds. This is the one channel in the table that is
   not board-specific and the only one a live backend can serve off the board;
   it is reached through ``time.clock_gettime_ns`` and not through sysfs.
4. **Performance counters.** ``perf_event_open(2)``: ``PERF_TYPE_HARDWARE``
   is type 0, and the generalised counters used here are
   ``PERF_COUNT_HW_CPU_CYCLES`` (config 0), ``PERF_COUNT_HW_INSTRUCTIONS``
   (config 1), ``PERF_COUNT_HW_CACHE_MISSES`` (config 3) and
   ``PERF_COUNT_HW_BRANCH_MISSES`` (config 5), from ``linux/perf_event.h``.
   The syscall needs ``/proc/sys/kernel/perf_event_paranoid`` at 1 or below
   for unprivileged per-process counting. The Orin Nano's CPU complex is six
   Arm Cortex-A78AE cores (Jetson Orin Nano Developer Kit data sheet), whose
   PMU is Arm PMUv3, driven by the kernel's ``armv8_pmuv3`` driver; the
   generalised events above are mapped onto PMUv3 events by that driver, so
   this package deliberately uses the generalised names rather than raw PMUv3
   event numbers, which differ between Arm cores (Arm Cortex-A78AE Technical
   Reference Manual, "Performance Monitors Unit" chapter).
5. **CPU frequency.** ``Documentation/admin-guide/pm/cpufreq.rst``:
   ``scaling_cur_freq`` is in **kilohertz**, hence ``scale = 1e3`` to hertz.
   On Jetson this reads back the frequency the governor last requested; the
   Jetson Linux Developer Guide, "Clock Frequency and Power Management"
   chapter, documents the BPMP as the authority and ``tegrastats`` as the
   vendor reader. The clock a ``cpufreq`` node reports is a declared rate and
   not an independent measurement of the core's clock.
6. **The vendor utility.** ``tegrastats`` (Jetson Linux Developer Guide,
   "tegrastats Utility") prints RAM, CPU-per-core load and frequency, GPU
   load, thermal zone temperatures and INA3221 rail power in one line. This
   package does not shell out to it: parsing a human-readable line whose
   format changes between L4T releases is a worse adapter than reading the
   sysfs nodes it reads. It is named here because it is what a Jetson engineer
   will reach for first, and ``README.md`` says so.

Throttling is a first-class concern for a benchmark harness on this board.
``trip_point_<N>_temp`` under each thermal zone gives the declared trip
temperature in millidegrees Celsius (same ABI document as 1); the harness
reads the die-temperature channels and ``docs/RECOVERY.md`` has the procedure
for a run that throttles. The throttle trip itself is a kernel action this
package observes and never controls.
"""

from __future__ import annotations

from .channels import Access, BoardDeclaration, Channel, ChannelKind

#: The device-tree model string a Jetson Orin Nano Developer Kit reports.
#: Jetson Linux Developer Guide, "Platform Adaptation and Bring-Up"; read from
#: ``proc/device-tree/model``. Matched as a PREFIX, because the trailing text
#: varies between carrier boards and L4T releases.
TARGET_MODEL_PREFIX = "NVIDIA Jetson Orin Nano"

#: Where the INA3221 power monitor binds. The leaf ``hwmon<N>`` index is
#: assigned in probe order and is therefore a glob, not a path.
POWER_DIRECTORY_GLOB = "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*"

#: Where the thermal zones live. The zone INDEX for a given zone name is not
#: fixed by any document; a live backend discovers it by reading each
#: ``type`` file. See reference 1 in the module docstring.
THERMAL_TYPE_GLOB = "sys/devices/virtual/thermal/thermal_zone*/type"

#: ``perf_event_open(2)`` syscall numbers. 298 on x86_64 and 241 on aarch64,
#: from the kernel syscall tables (``arch/x86/entry/syscalls/syscall_64.tbl``
#: and the generic ``include/uapi/asm-generic/unistd.h`` that arm64 uses).
#: The target is aarch64; the build host was x86_64. Both are declared, the
#: x86_64 number is the one that was exercised, and the aarch64 number is
#: UNVERIFIED like the rest of this table.
PERF_EVENT_OPEN_SYSCALL = {"x86_64": 298, "aarch64": 241}

_THERMAL_REFERENCE = (
    "Linux kernel ABI Documentation/driver-api/thermal/sysfs-api.rst "
    "(temp in millidegrees Celsius); zone names from the NVIDIA Jetson Linux "
    "Developer Guide, Thermal Management, Orin series"
)
_POWER_REFERENCE = (
    "Linux kernel Documentation/hwmon/ina3221.rst (in<i>_input millivolts, "
    "curr<i>_input milliamperes, in<i>_label); rail names and the 1-0040 "
    "binding from the NVIDIA Jetson Linux Developer Guide, Software-Based "
    "Power Consumption Modeling"
)
_PERF_REFERENCE = (
    "perf_event_open(2) with PERF_TYPE_HARDWARE and the generalised configs "
    "of linux/perf_event.h; mapped to Arm PMUv3 events by the kernel "
    "armv8_pmuv3 driver on the Cortex-A78AE complex"
)


def _thermal(name: str, zone: int) -> Channel:
    """One die-temperature channel at a documented default zone index."""
    return Channel(
        name=name,
        kind=ChannelKind.DIE_TEMPERATURE,
        unit="degC",
        access=Access.SYSFS_INTEGER,
        locator=f"sys/devices/virtual/thermal/thermal_zone{zone}/temp",
        scale=1e-3,
        raw_unit="millidegC",
        reference=_THERMAL_REFERENCE,
    )


def _rail(name: str, index: int, quantity: str) -> Channel:
    """One rail voltage or current channel on the INA3221 at 1-0040."""
    stem, unit, raw_unit, scale = {
        "voltage": ("in", "V", "mV", 1e-3),
        "current": ("curr", "A", "mA", 1e-3),
    }[quantity]
    return Channel(
        name=name,
        kind=ChannelKind.RAIL_POWER,
        unit=unit,
        access=Access.SYSFS_INTEGER,
        locator=f"{POWER_DIRECTORY_GLOB}/{stem}{index}_input",
        scale=scale,
        raw_unit=raw_unit,
        reference=_POWER_REFERENCE,
    )


def _counter(name: str, config: int, event: str) -> Channel:
    """One generalised hardware performance counter."""
    return Channel(
        name=name,
        kind=ChannelKind.PERF_COUNTER,
        unit="events",
        access=Access.PERF_EVENT_OPEN,
        locator=f"0:{config}",
        scale=1.0,
        raw_unit="events",
        reference=f"{_PERF_REFERENCE}; {event} is PERF_TYPE_HARDWARE config {config}",
    )


#: The declared channel table. Fourteen channels across the four source
#: classes the specification names.
ORIN_NANO = BoardDeclaration(
    model=f"{TARGET_MODEL_PREFIX} Developer Kit",
    notes=(
        "Declared from documentation and UNVERIFIED on hardware; no board was "
        "reachable from the build environment. Thermal zone indices and the "
        "hwmon index are discovered at run time on a real board and the "
        "indices here are the documented defaults."
    ),
    channels=(
        Channel(
            name="monotonic_clock",
            kind=ChannelKind.CLOCK,
            unit="s",
            access=Access.CLOCK_SYSCALL,
            locator="CLOCK_MONOTONIC_RAW",
            scale=1e-9,
            raw_unit="ns",
            reference=(
                "clock_gettime(2), CLOCK_MONOTONIC_RAW: raw hardware-based time "
                "not subject to NTP or adjtime(3) adjustment"
            ),
        ),
        Channel(
            name="cpu0_frequency",
            kind=ChannelKind.CLOCK,
            unit="Hz",
            access=Access.SYSFS_INTEGER,
            locator="sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
            scale=1e3,
            raw_unit="kHz",
            reference=(
                "Linux kernel Documentation/admin-guide/pm/cpufreq.rst "
                "(scaling_cur_freq in kHz); on Jetson this is the rate the "
                "governor last requested of the BPMP, not an independent "
                "measurement of the core clock"
            ),
        ),
        _thermal("die_temp_cpu", 0),
        _thermal("die_temp_gpu", 1),
        _thermal("die_temp_cv0", 2),
        _thermal("die_temp_soc0", 5),
        _thermal("die_temp_junction", 8),
        _rail("rail_vdd_in_voltage", 1, "voltage"),
        _rail("rail_vdd_in_current", 1, "current"),
        _rail("rail_cpu_gpu_cv_voltage", 2, "voltage"),
        _rail("rail_cpu_gpu_cv_current", 2, "current"),
        _counter("perf_cpu_cycles", 0, "PERF_COUNT_HW_CPU_CYCLES"),
        _counter("perf_instructions", 1, "PERF_COUNT_HW_INSTRUCTIONS"),
        _counter("perf_cache_misses", 3, "PERF_COUNT_HW_CACHE_MISSES"),
    ),
)

#: Rail power is a DERIVED quantity on this part: the INA3221 driver exposes
#: voltage and current and not power, so the product of the two channels in
#: each pair is the rail's instantaneous power in watts. Named here as data so
#: the derivation is declared rather than buried in the harness.
RAIL_POWER_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("rail_vdd_in_power", "rail_vdd_in_voltage", "rail_vdd_in_current"),
    ("rail_cpu_gpu_cv_power", "rail_cpu_gpu_cv_voltage", "rail_cpu_gpu_cv_current"),
)
