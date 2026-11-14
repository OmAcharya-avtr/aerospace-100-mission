"""The loader, and the measurement model every number in this product depends on.

MEASUREMENT MODEL, stated once
------------------------------

The platform rail is **VDD_IN**, measured by the board's on-chip **INA3221** at
``/sys/class/hwmon/hwmon1``, channel 1, as bus voltage (mV) times shunt current
(mA). Instantaneous power is ``P(t) = V(t) * I(t)``, in mW when the voltage is
in mV and the current in mA and the product is divided by 1000.

Energy over a window is the **trapezoidal integral of P(t) over the sampler's
timestamps**. Those timestamps are ``CLOCK_MONOTONIC``, taken in the sampler
process; the pipeline's per-inference timestamps are ``CLOCK_MONOTONIC`` taken
in a different process. The two are directly comparable because
``CLOCK_MONOTONIC`` is a system-wide clock, which is why intersecting them is
legitimate and not an approximation.

A **cycle** is ``CYCLE_S = 2.0`` seconds. The pipeline runs back-to-back
inferences for ``duty * CYCLE_S`` and then sleeps the remainder of the cycle. A
run is 38 cycles, 76 s.

An **inference** is one pass of the pipeline over one frame. It is one row of
``decisions_<cell>.csv`` and it carries its own ``t_start`` and ``t_end``.

A **useful decision** is an inference whose output passes the gate: a candidate
exists, its bounding box does not touch the frame border, its area is within the
declared bounds ``[area_min, area_max]``, and its score is at least ``tau``.
Every rejection is recorded with its reason, so the acceptance rate is
**reconstructible** from the committed decisions CSV and is never asserted:
``a = sum(accepted) / len(decisions)``.

**Energy per useful decision** over a run is the trapezoidal integral of
``P(t)`` across **the run window ``[t0_mono, t1_mono]`` exactly, sleep windows
included**, with ``P`` interpolated at both ends, divided by the number of
accepted decisions in that run. The sleep windows are charged to the decisions
because an on-board pipeline that duty-cycles still draws the platform's
background power while it sleeps, and the mission has no other purpose for that
energy.

The sampler brackets the run deliberately -- it starts before ``t0_mono`` and
stops after ``t1_mono`` -- so the sampler's span is **wider** than the run and
integrating the span instead charges the lead-in and tail-out to the decisions.
On the reference cell that mistake overstates the energy by 1.1041 per cent.
``E_active`` and ``E_sleep`` partition the run window using the declared
per-cycle phases, with ``P`` interpolated at every boundary, which is what makes
the decomposition exact in three factors with no residual term.

What the loader does and does not do
------------------------------------

It reads a campaign directory into typed records, validates every header
against :mod:`powerenvelope.schema` and raises :class:`SchemaError` on a
mismatch rather than silently accepting a column that moved. It checks the
``vdd_in_mw`` column against ``vdd_in_mv * vdd_in_ma / 1000`` recomputed from
the same row, within the rounding of the ``%.3f`` format the campaign writes, so
a file whose power column was produced by something other than the committed
campaign script is rejected.

It does **not** substitute anything. A campaign directory that is absent,
incomplete, or missing a required cell raises :class:`TraceDataAbsentError`
naming what it looked for. There is no synthetic fallback anywhere in this
module: the synthetic generator in :mod:`powerenvelope.synthetic` writes its
traces into a directory a caller names, and a caller that loads one gets a
campaign whose :class:`~powerenvelope.provenance.Origin` says SYNTHETIC.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .provenance import Origin, Provenance
from .schema import (
    CURRENT_LSB_MA,
    DECISION_COLUMNS,
    MONITOR_COLUMNS,
    N_CORES,
    NOMINAL_RAIL_MV,
    POWER_COLUMNS,
    SchemaError,
    validate_decision_header,
    validate_monitor_header,
    validate_power_header,
    validate_reject_reason,
)

#: Cycle length, seconds. Declared by the campaign and cross-checked against
#: ``manifest.json``; a manifest that disagrees is a load error.
CYCLE_S = 2.0

#: The marker the campaign writes LAST of all. A figure computed from a
#: directory without it would be a figure from a partial campaign.
COMPLETION_MARKER = "CAMPAIGN_COMPLETE"

#: The cell measured with the sampler and monitor running and the pipeline NOT
#: running. It is a BACKGROUND floor on a shared machine, not an idle floor.
FLOOR_CELL = "floor"

#: The reference cell, run four times, from whose spread the between-run
#: interval is taken.
REFERENCE_CELL = "w64_d050"

#: The repeats of the reference cell, in campaign order. The first element is
#: the grid cell itself; a between-run spread over fewer than two of these is
#: not a spread and the code that computes it says so.
REFERENCE_REPEATS: tuple[str, ...] = (
    "w64_d050",
    "w64_d050_rep2",
    "w64_d050_rep3",
    "w64_d050_rep4",
)

#: The permissive-gate control cell: the reference cell at ``tau_permissive``,
#: where the rejection factor should nearly vanish.
PERMISSIVE_CELL = "w64_d050_permissive"

#: Every cell the campaign plan names, in plan order.
PLANNED_CELLS: tuple[str, ...] = (
    FLOOR_CELL,
    "w48_d010",
    "w48_d025",
    "w48_d050",
    "w48_d075",
    "w48_d100",
    "w64_d010",
    "w64_d025",
    "w64_d050",
    "w64_d075",
    "w64_d100",
    "w96_d010",
    "w96_d025",
    "w96_d050",
    "w96_d075",
    "w96_d100",
    "w64_d050_rep2",
    "w64_d050_rep3",
    "w64_d050_rep4",
    PERMISSIVE_CELL,
)

#: Tolerance, mW, on the recomputed ``vdd_in_mw`` column. The campaign writes it
#: with ``%.3f``, so the rounding alone is up to 5e-4 mW; the allowance is twice
#: that and nothing more, because this check exists to catch a power column that
#: was not produced by ``vdd_in_mv * vdd_in_ma / 1000``.
POWER_COLUMN_TOLERANCE_MW = 1e-3


class TraceDataAbsentError(FileNotFoundError):
    """A required campaign directory, file or cell is not present.

    Raised instead of returning anything, and never caught internally in order
    to fall back to generated data. The message names every file looked for.
    """


@dataclass(frozen=True)
class PowerTrace:
    """One ``power_<cell>.csv``: the 20 Hz rail sampler's output.

    Attributes
    ----------
    t
        ``CLOCK_MONOTONIC`` seconds, non-decreasing.
    vdd_in_mv, vdd_in_ma
        Channel 1 bus voltage (mV) and shunt current (mA). The current
        quantises at :data:`~powerenvelope.schema.CURRENT_LSB_MA` = 8 mA.
    cpu_gpu_cv_mv, cpu_gpu_cv_ma, soc_mv, soc_ma
        Channels 2 and 3, the two downstream rails, same units.
    vdd_in_mw
        The power column as written by the campaign, mW.
    """

    t: np.ndarray
    vdd_in_mv: np.ndarray
    vdd_in_ma: np.ndarray
    cpu_gpu_cv_mv: np.ndarray
    cpu_gpu_cv_ma: np.ndarray
    soc_mv: np.ndarray
    soc_ma: np.ndarray
    vdd_in_mw: np.ndarray

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def duration_s(self) -> float:
        """Span of the sampler's timestamps, seconds."""
        return float(self.t[-1] - self.t[0]) if self.t.size else 0.0

    @property
    def cpu_gpu_cv_mw(self) -> np.ndarray:
        """Channel 2 power, mW."""
        return self.cpu_gpu_cv_mv * self.cpu_gpu_cv_ma / 1000.0

    @property
    def soc_mw(self) -> np.ndarray:
        """Channel 3 power, mW."""
        return self.soc_mv * self.soc_ma / 1000.0

    @property
    def nominal_mw(self) -> np.ndarray:
        """VDD_IN power with the NOMINAL 5.0 V rail substituted, mW.

        This is the specific simplification a power budget makes when it quotes
        a rail voltage rather than measuring one. It is not a measurement of
        anything; it is the baseline being refuted.
        """
        return NOMINAL_RAIL_MV * self.vdd_in_ma / 1000.0

    @property
    def quantisation_floor_mw(self) -> np.ndarray:
        """Per-sample least-significant bit of the power reading, mW.

        ``vdd_in_mv * 8 mA / 1000``. At the nominal 5 V rail that is 40 mW, and
        every power figure in this product carries it as a floor.
        """
        return self.vdd_in_mv * CURRENT_LSB_MA / 1000.0


@dataclass(frozen=True)
class MonitorTrace:
    """One ``monitor_<cell>.csv``: the 1 Hz environment monitor's output.

    ``cpu_busy_fraction`` is the TOTAL non-idle fraction from ``/proc/stat``
    over the preceding second, which includes the pipeline's own occupancy. It
    is not a foreign-occupancy measurement on its own; see
    :func:`powerenvelope.windows.foreign_busy_fraction`.
    """

    t: np.ndarray
    cpu_busy_fraction: np.ndarray
    tj_c: np.ndarray
    load1: np.ndarray
    clk_khz: np.ndarray

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def sample_interval_s(self) -> float:
        """Nominal interval each sample's occupancy figure covers, seconds."""
        return 1.0

    @property
    def clk_mean_khz(self) -> np.ndarray:
        """Mean over the readable cores of ``scaling_cur_freq``, kHz.

        A core whose ``scaling_cur_freq`` could not be read is written as -1 by
        the campaign and is excluded from the mean. A sample with no readable
        core gives ``nan``, which the window builder drops rather than imputes.
        """
        values = np.where(self.clk_khz >= 0, self.clk_khz, np.nan)
        with np.errstate(invalid="ignore"):
            return np.nanmean(values, axis=1)


@dataclass(frozen=True)
class DecisionTrace:
    """One ``decisions_<cell>.csv``: one row per inference.

    ``accepted`` is 0 or 1 and is the only definition of a useful decision used
    anywhere in this product. ``reject_reason`` is one of
    :data:`~powerenvelope.schema.REJECT_REASONS`.
    """

    t_start: np.ndarray
    t_end: np.ndarray
    cycle: np.ndarray
    frame_idx: np.ndarray
    score: np.ndarray
    n_candidates: np.ndarray
    area: np.ndarray
    touches_border: np.ndarray
    accepted: np.ndarray
    reject_reason: np.ndarray

    def __len__(self) -> int:
        return int(self.t_start.size)

    @property
    def latency_s(self) -> np.ndarray:
        """Per-inference latency ``t_end - t_start``, seconds."""
        return self.t_end - self.t_start

    @property
    def inferences(self) -> int:
        """Number of inferences, counted."""
        return len(self)

    @property
    def accepted_count(self) -> int:
        """Number of useful decisions, counted from the ``accepted`` column."""
        return int(self.accepted.sum())

    @property
    def acceptance_rate(self) -> float:
        """``accepted / inferences``, dimensionless. Counted, never asserted."""
        if len(self) == 0:
            raise ValueError("acceptance rate of a run with no inferences is undefined")
        return self.accepted_count / len(self)

    def reason_counts(self) -> dict[str, int]:
        """Count per reject reason, including ``accept``."""
        labels, counts = np.unique(self.reject_reason, return_counts=True)
        return {str(label): int(count) for label, count in zip(labels, counts, strict=True)}


@dataclass(frozen=True)
class Cell:
    """One run cell: its configuration and its three traces.

    Attributes
    ----------
    label
        Cell name as the campaign plan declares it, e.g. ``w64_d050``.
    frame_dim
        Frame dimension ``n``; the frame is ``n * n`` bytes.
    duty
        Declared duty cycle, dimensionless, from the manifest.
    tau
        Declared gate threshold for this cell, from the manifest.
    cycles
        Declared number of cycles in the run.
    has_pipeline
        False for the background floor cell, which has no decisions file.
    """

    label: str
    frame_dim: int
    duty: float
    tau: float
    cycles: int
    has_pipeline: bool
    power: PowerTrace
    monitor: MonitorTrace
    decisions: DecisionTrace | None
    record: dict

    @property
    def t0_mono(self) -> float:
        """Run start on ``CLOCK_MONOTONIC``, from the manifest run record."""
        return float(self.record["t0_mono"])

    @property
    def t1_mono(self) -> float:
        """Run end on ``CLOCK_MONOTONIC``, from the manifest run record."""
        return float(self.record["t1_mono"])

    @property
    def wall_s(self) -> float:
        """Run wall duration, seconds, from the manifest run record."""
        return float(self.record["wall_s"])

    @property
    def active_cpu_s(self) -> float:
        """The pipeline's own summed inference CPU-wall time, seconds.

        From the manifest. Recorded as measurement back-action, and excluded by
        specification from the learned predictor's descriptors.
        """
        return float(self.record.get("active_cpu_s", float("nan")))

    def require_decisions(self) -> DecisionTrace:
        """Return the decisions trace, or raise if this cell has no pipeline."""
        if self.decisions is None:
            raise TraceDataAbsentError(
                f"cell {self.label!r} has no decisions trace. It is the background "
                "floor cell: the sampler and the monitor ran, the pipeline did not, "
                "so there are no inferences and no useful decisions. A quantity per "
                "useful decision is undefined for it."
            )
        return self.decisions


@dataclass(frozen=True)
class Campaign:
    """A loaded campaign directory.

    Attributes
    ----------
    provenance
        Origin, relative directory label and the cell labels loaded.
    manifest
        The parsed ``manifest.json``, verbatim.
    cells
        Mapping from cell label to :class:`Cell`, in load order.
    complete
        Whether the directory carries the campaign's completion marker.
    """

    provenance: Provenance
    manifest: dict
    cells: dict[str, Cell]
    complete: bool

    def __len__(self) -> int:
        return len(self.cells)

    @property
    def cycle_s(self) -> float:
        """Cycle length, seconds, from the manifest."""
        return float(self.manifest.get("cycle_s", CYCLE_S))

    @property
    def sample_period_s(self) -> float:
        """Nominal rail sampler period, seconds, from the manifest."""
        return float(self.manifest.get("sample_period_s", 0.05))

    def require(self, label: str) -> Cell:
        """Return cell ``label`` or raise naming the cells that are present."""
        try:
            return self.cells[label]
        except KeyError:
            raise TraceDataAbsentError(
                f"cell {label!r} is not in {self.provenance.directory}/; cells "
                f"present: {', '.join(sorted(self.cells)) or 'none'}"
            ) from None

    def pipeline_cells(self) -> tuple[Cell, ...]:
        """Every cell that ran the pipeline, in load order."""
        return tuple(cell for cell in self.cells.values() if cell.has_pipeline)

    def grid_cells(self) -> tuple[Cell, ...]:
        """The 15 grid cells: dimension by duty, excluding repeats and controls."""
        return tuple(
            cell
            for cell in self.pipeline_cells()
            if cell.label.count("_") == 1 and cell.label.startswith("w")
        )


def _read_csv(
    path: Path, what: str, allow_empty: bool = False
) -> tuple[list[str], list[list[str]]]:
    if not path.exists():
        raise TraceDataAbsentError(
            f"{what} not found: expected {path.name} in the campaign directory"
        )
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise SchemaError(f"{what} {path.name} is empty: no header row") from None
        rows = [row for row in reader if row]
    if not rows and not allow_empty:
        raise SchemaError(f"{what} {path.name} has a header and no data rows")
    if not rows:
        return header, []
    width = len(header)
    for number, row in enumerate(rows, start=2):
        if len(row) != width:
            raise SchemaError(
                f"{what} {path.name} line {number} has {len(row)} fields, header has "
                f"{width}. The campaign writes a fixed-width row; a ragged file was "
                "truncated or interleaved."
            )
    return header, rows


def load_power(path: Path) -> PowerTrace:
    """Load one ``power_<cell>.csv``, validating its header and its power column.

    Raises
    ------
    SchemaError
        On a header mismatch, a non-numeric field, non-monotonic timestamps, or a
        ``vdd_in_mw`` column that is not ``vdd_in_mv * vdd_in_ma / 1000`` to
        within :data:`POWER_COLUMN_TOLERANCE_MW`.
    TraceDataAbsentError
        If the file is not there.
    """
    header, rows = _read_csv(path, "power CSV")
    validate_power_header(header)
    try:
        table = np.array(rows, dtype=np.float64)
    except ValueError as error:
        raise SchemaError(f"power CSV {path.name} has a non-numeric field: {error}") from None
    columns = {name: table[:, index] for index, name in enumerate(POWER_COLUMNS)}
    t = columns["t_mono"]
    if np.any(np.diff(t) < 0.0):
        bad = int(np.argmin(np.diff(t)))
        raise SchemaError(
            f"power CSV {path.name} has a decreasing t_mono at row {bad + 2}. "
            "CLOCK_MONOTONIC does not go backwards, so the file is not what the "
            "campaign sampler wrote."
        )
    recomputed = columns["vdd_in_mv"] * columns["vdd_in_ma"] / 1000.0
    residual = np.abs(recomputed - columns["vdd_in_mw"])
    if residual.size and float(residual.max()) > POWER_COLUMN_TOLERANCE_MW:
        worst = int(np.argmax(residual))
        raise SchemaError(
            f"power CSV {path.name} row {worst + 2}: vdd_in_mw="
            f"{columns['vdd_in_mw'][worst]!r} but vdd_in_mv * vdd_in_ma / 1000 = "
            f"{recomputed[worst]!r}, a residual of {float(residual.max()):.6g} mW "
            f"against a tolerance of {POWER_COLUMN_TOLERANCE_MW:g} mW. The power "
            "column was not produced by the committed campaign script."
        )
    return PowerTrace(
        t=t,
        vdd_in_mv=columns["vdd_in_mv"],
        vdd_in_ma=columns["vdd_in_ma"],
        cpu_gpu_cv_mv=columns["cpu_gpu_cv_mv"],
        cpu_gpu_cv_ma=columns["cpu_gpu_cv_ma"],
        soc_mv=columns["soc_mv"],
        soc_ma=columns["soc_ma"],
        vdd_in_mw=columns["vdd_in_mw"],
    )


def load_monitor(path: Path) -> MonitorTrace:
    """Load one ``monitor_<cell>.csv``, validating its header.

    Raises
    ------
    SchemaError
        On a header mismatch, a non-numeric field, or non-monotonic timestamps.
    """
    header, rows = _read_csv(path, "monitor CSV")
    validate_monitor_header(header)
    try:
        table = np.array(rows, dtype=np.float64)
    except ValueError as error:
        raise SchemaError(f"monitor CSV {path.name} has a non-numeric field: {error}") from None
    columns = {name: table[:, index] for index, name in enumerate(MONITOR_COLUMNS)}
    t = columns["t_mono"]
    if np.any(np.diff(t) < 0.0):
        raise SchemaError(f"monitor CSV {path.name} has a decreasing t_mono")
    clocks = np.column_stack([columns[f"clk{core}_khz"] for core in range(N_CORES)])
    return MonitorTrace(
        t=t,
        cpu_busy_fraction=columns["cpu_busy_fraction"],
        tj_c=columns["tj_c"],
        load1=columns["load1"],
        clk_khz=clocks,
    )


def load_decisions(path: Path, allow_empty: bool = False) -> DecisionTrace:
    """Load one ``decisions_<cell>.csv``, validating its header and its enums.

    Parameters
    ----------
    path
        The file.
    allow_empty
        Whether a file with a header and no data rows is acceptable. It is for
        exactly one cell: the background floor, which by construction runs no
        inferences, and for which the campaign still opens and closes the file
        so that a header-only ``decisions_floor.csv`` exists in the committed
        traces. For any cell that ran the pipeline an empty decisions file means
        the run produced nothing and is an error, so the default is False.

    Raises
    ------
    SchemaError
        On a header mismatch, a non-numeric numeric field, an ``accepted`` value
        that is not 0 or 1, an interval with ``t_end < t_start``, a
        ``reject_reason`` the campaign script cannot write, or a file with no
        data rows when ``allow_empty`` is False.
    """
    header, rows = _read_csv(path, "decisions CSV", allow_empty=allow_empty)
    if not rows:
        empty_float = np.empty(0, dtype=np.float64)
        empty_int = np.empty(0, dtype=np.int64)
        validate_decision_header(header)
        return DecisionTrace(
            t_start=empty_float,
            t_end=empty_float.copy(),
            cycle=empty_int,
            frame_idx=empty_int.copy(),
            score=empty_float.copy(),
            n_candidates=empty_int.copy(),
            area=empty_int.copy(),
            touches_border=empty_int.copy(),
            accepted=empty_int.copy(),
            reject_reason=np.empty(0, dtype=object),
        )
    validate_decision_header(header)
    reasons = np.array([validate_reject_reason(row[-1]) for row in rows], dtype=object)
    try:
        table = np.array([row[:-1] for row in rows], dtype=np.float64)
    except ValueError as error:
        raise SchemaError(f"decisions CSV {path.name} has a non-numeric field: {error}") from None
    columns = {name: table[:, index] for index, name in enumerate(DECISION_COLUMNS[:-1])}
    accepted = columns["accepted"]
    if not np.all(np.isin(accepted, (0.0, 1.0))):
        raise SchemaError(
            f"decisions CSV {path.name} has an accepted value that is not 0 or 1. "
            "A useful decision is a counted binary event."
        )
    agree = (accepted == 1.0) == (reasons == "accept")
    if not np.all(agree):
        bad = int(np.argmin(agree))
        raise SchemaError(
            f"decisions CSV {path.name} row {bad + 2}: accepted="
            f"{accepted[bad]:.0f} disagrees with reject_reason={reasons[bad]!r}. "
            "The gate writes accepted=1 exactly when the reason is 'accept'."
        )
    width = columns["t_end"] - columns["t_start"]
    if np.any(width < 0.0):
        bad = int(np.argmin(width))
        raise SchemaError(
            f"decisions CSV {path.name} row {bad + 2} has t_end < t_start, so the "
            "inference interval is negative."
        )
    return DecisionTrace(
        t_start=columns["t_start"],
        t_end=columns["t_end"],
        cycle=columns["cycle"].astype(np.int64),
        frame_idx=columns["frame_idx"].astype(np.int64),
        score=columns["score"],
        n_candidates=columns["n_candidates"].astype(np.int64),
        area=columns["area"].astype(np.int64),
        touches_border=columns["touches_border"].astype(np.int64),
        accepted=accepted.astype(np.int64),
        reject_reason=reasons,
    )


def campaign_is_complete(directory: Path | str) -> bool:
    """Whether ``directory`` carries the campaign's completion marker."""
    return (Path(directory) / COMPLETION_MARKER).exists()


def _manifest_records(manifest: dict) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for record in manifest.get("runs", []):
        label = record.get("cell")
        if isinstance(label, str):
            records[label] = record
    return records


def load_campaign(
    directory: Path | str,
    origin: Origin = Origin.MEASURED,
    require: tuple[str, ...] = (),
    directory_label: str | None = None,
) -> Campaign:
    """Load a campaign directory into a :class:`Campaign`.

    Parameters
    ----------
    directory
        Campaign directory holding ``manifest.json`` and the per-cell CSVs.
    origin
        Origin to record. Pass :attr:`~powerenvelope.provenance.Origin.SYNTHETIC`
        for a directory written by :mod:`powerenvelope.synthetic`.
    require
        Cell labels that must be present. Any absent one raises.
    directory_label
        Relative label to print in reports. Defaults to the directory's own
        name, never an absolute path.

    Raises
    ------
    TraceDataAbsentError
        If the directory, ``manifest.json``, a required cell or a required
        per-cell file is absent. The message names what was looked for.
    SchemaError
        If any file's header or values do not match the declared schema.
    """
    root = Path(directory)
    label = directory_label if directory_label is not None else root.name
    if not root.is_dir():
        raise TraceDataAbsentError(
            f"campaign directory {label}/ does not exist. Expected manifest.json, "
            "campaign.log, and power_<cell>.csv, monitor_<cell>.csv, "
            "decisions_<cell>.csv per cell. Nothing was substituted."
        )
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        present = sorted(path.name for path in root.glob("*.csv"))
        raise TraceDataAbsentError(
            f"{label}/manifest.json is absent. CSV files present: {present}. The "
            "manifest carries the declared duty, frame dimension and tau of every "
            "cell, so no cell can be interpreted without it."
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as error:
        raise SchemaError(f"{label}/manifest.json is not valid JSON: {error}") from None
    if not isinstance(manifest, dict):
        raise SchemaError(f"{label}/manifest.json is not a JSON object")

    records = _manifest_records(manifest)
    discovered = sorted(
        path.name[len("power_") : -len(".csv")] for path in root.glob("power_*.csv")
    )
    missing = [name for name in require if name not in discovered]
    if missing:
        raise TraceDataAbsentError(
            f"cells absent from {label}/: {missing}. Cells present: {discovered}. "
            "Expected power_<cell>.csv, monitor_<cell>.csv and, for every cell that "
            "ran the pipeline, decisions_<cell>.csv. Nothing was substituted."
        )

    cells: dict[str, Cell] = {}
    for name in discovered:
        record = records.get(name)
        if record is None:
            raise TraceDataAbsentError(
                f"{label}/manifest.json has no run record for cell {name!r}, so its "
                f"declared duty, frame dimension and tau are unknown. Records "
                f"present: {sorted(records)}"
            )
        power = load_power(root / f"power_{name}.csv")
        monitor = load_monitor(root / f"monitor_{name}.csv")
        decisions_path = root / f"decisions_{name}.csv"
        has_pipeline = bool(record.get("pipeline", True))
        if has_pipeline:
            decisions = load_decisions(decisions_path)
            if len(decisions) == 0:
                raise SchemaError(f"{label}/decisions_{name}.csv has no inference rows")
        else:
            decisions = None
            # The campaign opens and closes a decisions file for the background
            # floor cell too, so `decisions_floor.csv` exists with a header and
            # no rows. That is not an error and it is not special-cased away
            # either: the header is validated like any other, and a non-empty
            # file for a cell the manifest says ran no pipeline is a
            # contradiction between the manifest and the trace and is refused.
            if decisions_path.exists():
                empty = load_decisions(decisions_path, allow_empty=True)
                if len(empty) != 0:
                    raise SchemaError(
                        f"{label}/decisions_{name}.csv has {len(empty)} inference row(s) "
                        f"while {label}/manifest.json records pipeline=false for that "
                        "cell. The manifest and the trace disagree, so nothing is loaded."
                    )
        cells[name] = Cell(
            label=name,
            frame_dim=int(record["n"]),
            duty=float(record["duty"]),
            tau=float(record["tau"]),
            cycles=int(record["cycles"]),
            has_pipeline=has_pipeline,
            power=power,
            monitor=monitor,
            decisions=decisions,
            record=record,
        )

    declared_cycle = float(manifest.get("cycle_s", CYCLE_S))
    if abs(declared_cycle - CYCLE_S) > 1e-9:
        raise SchemaError(
            f"{label}/manifest.json declares cycle_s={declared_cycle}, and this "
            f"product's analysis is written for cycle_s={CYCLE_S}. Nothing here "
            "supports a statement about a different cycle period."
        )
    return Campaign(
        provenance=Provenance(origin=origin, directory=label, cells=tuple(sorted(cells))),
        manifest=manifest,
        cells=cells,
        complete=campaign_is_complete(root),
    )
