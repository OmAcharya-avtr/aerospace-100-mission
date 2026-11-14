"""Column schemas for the campaign's three CSV files, and the validation of them.

The writer is ``data/campaign.py``, the stdlib-only program that ran on the
board. Its three table shapes are declared here once, so a loader cannot
silently accept a file whose columns have moved.

A file with the right column names in the wrong order is rejected as well. The
loader indexes by name, so it could in principle cope; but a reordered header is
evidence that the file was not written by the committed campaign script, and
nothing downstream may then assume the semantics of any column.

Units, which are not in the headers and therefore have to be written down
somewhere:

===========================  ======  =========================================
column                       unit    meaning
===========================  ======  =========================================
``t_mono``                   s       ``CLOCK_MONOTONIC``, float
``vdd_in_mv``                mV      INA3221 channel 1 bus voltage
``vdd_in_ma``                mA      INA3221 channel 1 shunt current, 8 mA LSB
``cpu_gpu_cv_mv``            mV      channel 2 bus voltage
``cpu_gpu_cv_ma``            mA      channel 2 shunt current
``soc_mv``                   mV      channel 3 bus voltage
``soc_ma``                   mA      channel 3 shunt current
``vdd_in_mw``                mW      ``vdd_in_mv * vdd_in_ma / 1000.0``
``cpu_busy_fraction``        1       total non-idle fraction from /proc/stat
``tj_c``                     degC    thermal zone 8
``load1``                    1       1-minute load average
``clk<i>_khz``               kHz     per-core scaling_cur_freq, -1 if unreadable
``t_start`` / ``t_end``      s       ``CLOCK_MONOTONIC`` bounds of one inference
``cycle``                    1       cycle index within the run
``frame_idx``                1       frame index, the pipeline's only input
``score``                    1       best candidate's gate score
``n_candidates``             1       connected components above the gradient gate
``area``                     px      best candidate's pixel area after 2x2 decimation
``touches_border``           0/1     best candidate's bounding box touches the border
``accepted``                 0/1     passed the gate
``reject_reason``            --      accept, no_candidate, border, area_out, low_score
===========================  ======  =========================================
"""

from __future__ import annotations

from collections.abc import Sequence

#: Number of CPU cores the campaign monitor samples clocks from.
N_CORES = 6

#: Columns of ``power_<cell>.csv``, one row per 20 Hz rail sample.
POWER_COLUMNS: tuple[str, ...] = (
    "t_mono",
    "vdd_in_mv",
    "vdd_in_ma",
    "cpu_gpu_cv_mv",
    "cpu_gpu_cv_ma",
    "soc_mv",
    "soc_ma",
    "vdd_in_mw",
)

#: Columns of ``monitor_<cell>.csv``, one row per 1 Hz environment sample.
MONITOR_COLUMNS: tuple[str, ...] = (
    "t_mono",
    "cpu_busy_fraction",
    "tj_c",
    "load1",
    *(f"clk{core}_khz" for core in range(N_CORES)),
)

#: Columns of ``decisions_<cell>.csv``, one row per inference.
DECISION_COLUMNS: tuple[str, ...] = (
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

#: The reject reasons ``data/campaign.py`` can write. Any other value is a
#: schema error: it means the gate in the loader's assumed pipeline is not the
#: gate that produced the file.
REJECT_REASONS: tuple[str, ...] = (
    "accept",
    "no_candidate",
    "border",
    "area_out",
    "low_score",
)

#: Current quantum of the INA3221 shunt reading on this board, mA. Every power
#: figure carries ``vdd_in_mv * CURRENT_LSB_MA / 1000`` as its least-significant
#: bit, which at the nominal 5 V rail is 40 mW.
CURRENT_LSB_MA = 8.0

#: Nominal rail voltage a power budget quotes instead of measuring, mV.
NOMINAL_RAIL_MV = 5000.0

#: The gate's declared area bounds, in pixels after the 2x2 decimation, as
#: ``data/campaign.py`` declares them and ``manifest.json`` records them. They
#: are mirrored here so that the synthetic generator's gate is the campaign's
#: gate and not a second opinion about it.
AREA_MIN_DECLARED = 2
AREA_MAX_DECLARED = 60


class SchemaError(ValueError):
    """A CSV file's header or values do not match the declared schema."""


def _check_header(found: Sequence[str], expected: tuple[str, ...], what: str) -> None:
    found = tuple(name.strip() for name in found)
    if found == expected:
        return
    missing = [name for name in expected if name not in found]
    extra = [name for name in found if name not in expected]
    if missing or extra:
        raise SchemaError(
            f"{what} header does not match the declared schema. "
            f"missing={missing} unexpected={extra}. "
            f"Expected exactly, in order: {list(expected)}"
        )
    raise SchemaError(
        f"{what} header has the right names in the wrong order. "
        f"got={list(found)} expected={list(expected)}. The loader indexes by name, "
        "but a reordered header means the file was not written by the committed "
        "campaign script, so nothing downstream may assume its semantics."
    )


def validate_power_header(found: Sequence[str]) -> None:
    """Raise :class:`SchemaError` unless ``found`` is exactly :data:`POWER_COLUMNS`."""
    _check_header(found, POWER_COLUMNS, "power CSV")


def validate_monitor_header(found: Sequence[str]) -> None:
    """Raise :class:`SchemaError` unless ``found`` is exactly :data:`MONITOR_COLUMNS`."""
    _check_header(found, MONITOR_COLUMNS, "monitor CSV")


def validate_decision_header(found: Sequence[str]) -> None:
    """Raise :class:`SchemaError` unless ``found`` is exactly :data:`DECISION_COLUMNS`."""
    _check_header(found, DECISION_COLUMNS, "decisions CSV")


def validate_reject_reason(value: str) -> str:
    """Return ``value`` if it is a declared reject reason, else raise.

    Raises
    ------
    SchemaError
        Naming the value found and the set the campaign script can write.
    """
    cleaned = value.strip()
    if cleaned not in REJECT_REASONS:
        raise SchemaError(
            f"decisions CSV carries reject_reason={cleaned!r}, which "
            f"data/campaign.py cannot write. Declared set: {list(REJECT_REASONS)}"
        )
    return cleaned
