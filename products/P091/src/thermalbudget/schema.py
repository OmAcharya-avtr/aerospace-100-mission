"""The binding trace schema, and the two things about it that are easy to get wrong.

The schema is fixed by ``products/P091/docs/SPEC.md`` and by the campaign script
``data/campaign.py`` that writes the files. One header row, one row per second:

``t_wall, t_rel, iters_cum, iters_per_s, achieved_duty, loadavg1,
foreign_cpu_pct, temp_<zone> for each readable zone, cpu<i>_khz for each core,
gpu_hz``

Two properties of the file are load-bearing and are implemented here rather than
assumed by the loader.

**The missing-value sentinel is ``-1`` and it is not a value.** The campaign's
``read_int`` returns ``-1`` for any sysfs field it could not read, and
``foreign_cpu_pct`` is ``-1.0`` when the process table could not be sampled. A
``-1`` milli-degree reading is ``-0.001 °C``, which is a plausible-looking
number that would drag a mean towards zero and bias a slope. Every sentinel is
converted to ``nan`` on load and every statistic in this package is computed
over the finite entries only, with the count of missing samples carried beside
the result. Three thermal zones on the measured board (``cv0/cv1/cv2-thermal``)
return no bytes at all and are absent from the header rather than present as
``-1``; both cases are handled.

**The temperature scale is inferred, not assumed.** The Linux thermal sysfs
interface reports milli-degrees Celsius, and the measured traces do. The
specification says the loader converts and must not assume the scale, so
:func:`infer_temperature_divisor` decides from the magnitude of the finite
samples against a declared rule and raises on anything outside both accepted
bands rather than guessing. A file that has already been converted to degrees by
some other tool is therefore read correctly, and a file in units nobody
recognises is rejected loudly.

Reference for the sysfs contract: Linux kernel documentation,
``Documentation/driver-api/thermal/sysfs-api.rst`` (``temp`` is in
millidegrees Celsius), and ``Documentation/ABI/testing/sysfs-devices-system-cpu``
(``cpufreq/scaling_cur_freq`` in kHz). The GPU devfreq ``cur_freq`` is in Hz,
per ``Documentation/ABI/testing/sysfs-class-devfreq``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

#: Columns every trace carries, in order, before the per-zone and per-core blocks.
FIXED_COLUMNS: tuple[str, ...] = (
    "t_wall",
    "t_rel",
    "iters_cum",
    "iters_per_s",
    "achieved_duty",
    "loadavg1",
    "foreign_cpu_pct",
)

#: The six thermal zones readable on the measured board, in the sorted order the
#: campaign writes them. ``tj-thermal`` is the junction temperature and is this
#: product's die-temperature channel.
READABLE_ZONES: tuple[str, ...] = (
    "cpu-thermal",
    "gpu-thermal",
    "soc0-thermal",
    "soc1-thermal",
    "soc2-thermal",
    "tj-thermal",
)

#: Zones present in sysfs on the measured board whose ``temp`` file returns no
#: bytes at all. Recorded here because their ABSENCE from a trace header is
#: expected and must not be read as a damaged file. SPEC.md limit 4.
UNREADABLE_ZONES: tuple[str, ...] = ("cv0-thermal", "cv1-thermal", "cv2-thermal")

#: The die-temperature channel. Every thermal figure in this product is this one.
DIE_ZONE: str = "tj-thermal"

#: Column name of the die temperature as it appears in a trace header.
DIE_COLUMN: str = f"temp_{DIE_ZONE}"

#: Cores on the measured board, hence ``cpu0_khz`` .. ``cpu5_khz``.
N_CORES: int = 6

#: Value written by the campaign for any field it could not read. NOT a value.
MISSING_SENTINEL: float = -1.0

def sentinel_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """Columns of ``columns`` in which ``-1`` means missing rather than minus one.

    ``t_wall``, ``t_rel``, ``iters_cum``, ``iters_per_s``, ``achieved_duty`` and
    ``loadavg1`` are computed by the campaign itself and are never
    sentinel-coded, so a negative value in one of those is a damaged file and is
    reported as one rather than quietly dropped.
    """
    sentinel_named = ("gpu_hz", "foreign_cpu_pct")
    return tuple(
        name
        for name in columns
        if name.startswith("temp_") or name.endswith("_khz") or name in sentinel_named
    )


#: Accepted temperature bands. A finite sample's median must land in exactly one.
#: The bands do not touch: a board between 100 and 1000 in raw units would be
#: either a 100--1000 °C die or a 0.1--1.0 °C die, and neither is a reading this
#: product will silently pick between.
DEGREE_BAND: tuple[float, float] = (-40.0, 100.0)
MILLIDEGREE_BAND: tuple[float, float] = (-40_000.0, 150_000.0)
MILLIDEGREE_FLOOR: float = 1000.0


class TraceSchemaError(ValueError):
    """Raised when a file does not satisfy the binding schema of SPEC.md."""


def expected_columns(
    zones: Sequence[str] = READABLE_ZONES, n_cores: int = N_CORES
) -> tuple[str, ...]:
    """Full expected header for the given zones and core count, in order."""
    return (
        *FIXED_COLUMNS,
        *(f"temp_{zone}" for zone in zones),
        *(f"cpu{index}_khz" for index in range(n_cores)),
        "gpu_hz",
    )


def temperature_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """The ``temp_*`` columns of a header, in header order."""
    return tuple(name for name in columns if name.startswith("temp_"))


def clock_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """The ``cpu<i>_khz`` columns of a header, in header order."""
    return tuple(name for name in columns if name.endswith("_khz"))


def validate_header(columns: Sequence[str]) -> None:
    """Check a header against the schema, naming every defect found.

    The check is deliberately not an equality test against
    :func:`expected_columns`: the campaign writes one ``temp_`` column per
    READABLE zone and one ``cpu<i>_khz`` per worker, both discovered at run time,
    so a trace from a board with a different number of readable zones is a valid
    trace and must load. What is binding is the fixed prefix, the presence of the
    die channel, at least one clock column, the trailing ``gpu_hz``, and that
    nothing is duplicated.

    Raises
    ------
    TraceSchemaError
        With every violation listed, not just the first.
    """
    problems: list[str] = []
    if list(columns[: len(FIXED_COLUMNS)]) != list(FIXED_COLUMNS):
        problems.append(
            f"fixed prefix is {list(columns[: len(FIXED_COLUMNS)])}, "
            f"expected {list(FIXED_COLUMNS)}"
        )
    duplicates = sorted({name for name in columns if list(columns).count(name) > 1})
    if duplicates:
        problems.append(f"duplicated columns: {duplicates}")
    if DIE_COLUMN not in columns:
        problems.append(f"die-temperature column {DIE_COLUMN!r} is absent")
    if not clock_columns(columns):
        problems.append("no cpu<i>_khz column present")
    if columns and columns[-1] != "gpu_hz":
        problems.append(f"last column is {columns[-1]!r}, expected 'gpu_hz'")
    for zone in UNREADABLE_ZONES:
        if f"temp_{zone}" in columns:
            problems.append(
                f"temp_{zone} is present; on the measured board that zone returns no "
                "bytes and cannot have produced a column"
            )
    if problems:
        raise TraceSchemaError(
            "trace header violates the schema of docs/SPEC.md: " + "; ".join(problems)
        )


def infer_temperature_divisor(samples: Sequence[float]) -> float:
    """Divisor converting raw temperature samples to degrees Celsius.

    The decision rule, stated so it can be argued with rather than discovered:
    take the median of the finite, non-sentinel samples; if its magnitude is at
    or above :data:`MILLIDEGREE_FLOOR` and it lies inside
    :data:`MILLIDEGREE_BAND`, the units are milli-degrees and the divisor is
    1000; if it lies inside :data:`DEGREE_BAND`, the units are degrees and the
    divisor is 1. Anything else raises.

    Parameters
    ----------
    samples:
        Raw temperature values as they appear in the file, sentinels included.

    Returns
    -------
    float
        ``1000.0`` or ``1.0``.

    Raises
    ------
    TraceSchemaError
        If no finite non-sentinel sample exists, or the median lands outside
        both bands. Guessing here would silently scale every thermal figure in
        the product by a thousand.
    """
    finite = sorted(
        value
        for value in samples
        if value is not None and math.isfinite(value) and value != MISSING_SENTINEL
    )
    if not finite:
        raise TraceSchemaError(
            "cannot infer the temperature scale: no finite non-sentinel sample. "
            "Every reading in this column is missing"
        )
    middle = len(finite) // 2
    median = finite[middle] if len(finite) % 2 else 0.5 * (finite[middle - 1] + finite[middle])
    if abs(median) >= MILLIDEGREE_FLOOR and MILLIDEGREE_BAND[0] <= median <= MILLIDEGREE_BAND[1]:
        return 1000.0
    if DEGREE_BAND[0] <= median <= DEGREE_BAND[1]:
        return 1.0
    raise TraceSchemaError(
        f"cannot infer the temperature scale: median raw sample {median} lies in "
        f"neither the degree band {DEGREE_BAND} nor the milli-degree band "
        f"{MILLIDEGREE_BAND}"
    )
