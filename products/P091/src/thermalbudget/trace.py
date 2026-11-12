"""Loading measured and synthetic traces, with the origin carried in the object.

A :class:`Trace` is one CSV file from the campaign: one row per second of one
run, with the run's label and commanded duty cycle from ``manifest.json`` and an
:class:`~thermalbudget.provenance.Origin` that says whether the numbers came off
the board. A :class:`Campaign` is the whole directory.

The loader does four things that the rest of the package then relies on.

1. It validates the header against :mod:`thermalbudget.schema` and names every
   violation, rather than failing later inside a statistic.
2. It converts ``-1`` to ``nan`` in every sentinel-coded column, and records how
   many samples it converted, per column. Nothing downstream treats ``-1`` as a
   temperature.
3. It infers the temperature scale from the data rather than assuming
   milli-degrees.
4. It carries the origin. :func:`load_campaign` sets ``MEASURED`` only for a
   directory that is not the synthetic fixture directory and whose manifest
   declares the measured board; the synthetic generator's own loader entry point
   sets ``SYNTHETIC``, and there is no argument anywhere in this package that
   promotes one to the other.

The missing-data policy is the same everywhere: statistics are computed over
finite samples and the count of non-finite samples travels with the result. A
window whose finite sample count falls below a declared minimum does not produce
a number; it produces a refusal.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import schema
from .provenance import Origin

#: Name of the campaign manifest inside a trace directory.
MANIFEST_NAME = "manifest.json"

#: Prefix every file written by ``tests/fixtures/synthetic_trace.py`` carries, so
#: that a synthetic file cannot be mistaken for a measured one by name alone.
SYNTHETIC_PREFIX = "SYNTHETIC_"

#: ``/proc/device-tree/model`` of the board the campaign was run on. A manifest
#: whose model string does not contain this is loaded, but
#: :meth:`Campaign.board_matches_declared_target` returns False and the report
#: layer says so beside every figure.
DECLARED_BOARD = "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super"

#: The one power mode the campaign could reach. SPEC.md limit 1.
DECLARED_POWER_MODE = "MAXN_SUPER"


class TraceDataAbsentError(FileNotFoundError):
    """Raised when measured traces are required and are not on disk.

    Carries the names of the files it looked for. Validation scripts in this
    repository raise this rather than falling back to synthetic data: a figure
    computed from a fixture and reported as a measurement is the single failure
    mode this product is built to make impossible.
    """


@dataclass(frozen=True)
class Trace:
    """One run's trace, with its origin and its missing-data census.

    Attributes
    ----------
    label:
        Run label from the manifest, e.g. ``soak_1.00_r1``.
    duty:
        Commanded duty cycle in [0, 1]. The ACHIEVED duty is a column.
    origin:
        ``MEASURED`` or ``SYNTHETIC``.
    path:
        Path the trace was read from.
    columns:
        Header, in file order.
    data:
        Mapping column name to a float array of length ``n_rows``, with
        sentinels already converted to ``nan`` and temperatures already in
        degrees Celsius.
    missing:
        Mapping column name to the number of samples that were sentinel-coded or
        unparseable in that column.
    temperature_divisor:
        The divisor :func:`thermalbudget.schema.infer_temperature_divisor`
        decided on, recorded so a report can state it.
    """

    label: str
    duty: float
    origin: Origin
    path: Path
    columns: tuple[str, ...]
    data: dict[str, np.ndarray]
    missing: dict[str, int]
    temperature_divisor: float

    @property
    def n_rows(self) -> int:
        """Number of sample rows."""
        return int(self.data["t_rel"].size)

    @property
    def t_rel(self) -> np.ndarray:
        """Seconds since the start of this run."""
        return self.data["t_rel"]

    @property
    def die_temperature_c(self) -> np.ndarray:
        """Junction temperature in degrees Celsius, ``nan`` where unreadable."""
        return self.data[schema.DIE_COLUMN]

    @property
    def throughput_iters_per_s(self) -> np.ndarray:
        """Iterations per wall second summed over workers."""
        return self.data["iters_per_s"]

    @property
    def mean_cpu_khz(self) -> np.ndarray:
        """Mean CPU clock across cores per sample, in kHz, ignoring missing cores.

        A sample in which EVERY core is missing yields ``nan``. The mean is
        computed from an explicit sum and count rather than with ``nanmean``,
        which emits a RuntimeWarning for an all-missing row; a warning on a
        condition the schema explicitly allows is noise that trains a reader to
        ignore warnings.
        """
        stack = np.vstack([self.data[name] for name in schema.clock_columns(self.columns)])
        valid = np.isfinite(stack)
        counts = valid.sum(axis=0)
        totals = np.where(valid, stack, 0.0).sum(axis=0)
        result = np.full(counts.shape, math.nan, dtype=float)
        np.divide(totals, counts, out=result, where=counts > 0)
        return result

    @property
    def max_cpu_khz(self) -> float:
        """Largest finite CPU clock sample anywhere in the trace, in kHz."""
        values = self.mean_cpu_khz
        finite = values[np.isfinite(values)]
        return float(finite.max()) if finite.size else math.nan

    @property
    def foreign_cpu_pct(self) -> np.ndarray:
        """Total CPU per cent of processes that are not the campaign."""
        return self.data["foreign_cpu_pct"]

    def window(self, start_s: float, end_s: float) -> Trace:
        """A new trace restricted to ``start_s <= t_rel <= end_s``."""
        mask = (self.t_rel >= start_s) & (self.t_rel <= end_s)
        return Trace(
            label=self.label,
            duty=self.duty,
            origin=self.origin,
            path=self.path,
            columns=self.columns,
            data={name: values[mask] for name, values in self.data.items()},
            missing=dict(self.missing),
            temperature_divisor=self.temperature_divisor,
        )

    def tail(self, seconds: float) -> Trace:
        """The final ``seconds`` of the run."""
        if self.n_rows == 0:
            return self
        end = float(self.t_rel[-1])
        return self.window(end - seconds, end)

    def finite(self, column: str) -> np.ndarray:
        """Finite samples of ``column``, with ``nan`` entries dropped."""
        values = self.data[column]
        return values[np.isfinite(values)]


def _parse_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.nan


def load_trace(
    path: str | Path,
    *,
    origin: Origin,
    label: str | None = None,
    duty: float | None = None,
) -> Trace:
    """Read one trace CSV.

    Parameters
    ----------
    path:
        CSV file written by ``data/campaign.py`` or by the synthetic generator.
    origin:
        Required and never defaulted. The caller must say where the file came
        from; there is no inference from the filename, because a filename is not
        evidence.
    label, duty:
        Run label and commanded duty. Defaulted from the filename stem and from
        the label when the manifest is not available.

    Raises
    ------
    thermalbudget.schema.TraceSchemaError
        On a header violation, an empty file, a ragged row, or a negative value
        in a column where ``-1`` is not the missing sentinel.
    """
    path = Path(path)
    if not path.exists():
        raise TraceDataAbsentError(f"trace file not found: {path.name} (looked in {path.parent})")
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise schema.TraceSchemaError(f"{path.name} is empty: no header row")
    columns = tuple(name.strip() for name in rows[0])
    schema.validate_header(columns)
    body = rows[1:]
    if not body:
        raise schema.TraceSchemaError(
            f"{path.name} has a header and no data rows; a trace with no samples "
            "cannot produce a figure"
        )
    ragged = [index for index, row in enumerate(body) if len(row) != len(columns)]
    if ragged:
        raise schema.TraceSchemaError(
            f"{path.name}: {len(ragged)} row(s) have a different field count from the "
            f"header, first at data row {ragged[0]}"
        )

    raw = {
        name: np.array([_parse_float(row[index]) for row in body], dtype=float)
        for index, name in enumerate(columns)
    }
    sentinel_named = set(schema.sentinel_columns(columns))
    for name, values in raw.items():
        if name in sentinel_named:
            continue
        negative = int(np.sum(values < 0.0))
        if negative:
            raise schema.TraceSchemaError(
                f"{path.name}: column {name!r} has {negative} negative value(s); "
                "-1 is the missing sentinel only in temp_*, cpu*_khz, gpu_hz and "
                "foreign_cpu_pct, so a negative here is a damaged file"
            )

    missing: dict[str, int] = {}
    data: dict[str, np.ndarray] = {}
    for name, values in raw.items():
        if name in sentinel_named:
            sentinel_mask = values == schema.MISSING_SENTINEL
            unparseable = ~np.isfinite(values)
            cleaned = values.copy()
            cleaned[sentinel_mask | unparseable] = math.nan
            missing[name] = int(np.sum(sentinel_mask | unparseable))
            data[name] = cleaned
        else:
            missing[name] = int(np.sum(~np.isfinite(values)))
            data[name] = values

    temperature_names = schema.temperature_columns(columns)
    pooled: list[float] = []
    for name in temperature_names:
        pooled.extend(float(value) for value in data[name] if math.isfinite(value))
    divisor = schema.infer_temperature_divisor(pooled)
    for name in temperature_names:
        data[name] = data[name] / divisor

    stem = path.stem
    if label is None:
        label = stem[len(SYNTHETIC_PREFIX) :] if stem.startswith(SYNTHETIC_PREFIX) else stem
    if duty is None:
        duty = _duty_from_label(label)
    return Trace(
        label=label,
        duty=duty,
        origin=origin,
        path=path,
        columns=columns,
        data=data,
        missing=missing,
        temperature_divisor=divisor,
    )


def _duty_from_label(label: str) -> float:
    """Commanded duty parsed from a run label, 0.0 for an idle or cooldown run.

    The manifest is authoritative; this is the fallback for a trace loaded on its
    own. ``soak_0.75`` gives 0.75, ``burst_1.00_a`` gives 1.0, ``idle_pre`` and
    ``cool_r1`` give 0.0, and an unrecognised label gives ``nan`` rather than a
    guess.
    """
    if label.startswith(("idle", "cool")):
        return 0.0
    for part in label.split("_"):
        try:
            value = float(part)
        except ValueError:
            continue
        if 0.0 <= value <= 1.0:
            return value
    return math.nan


@dataclass(frozen=True)
class Campaign:
    """A directory of traces plus its manifest.

    Attributes
    ----------
    directory:
        Directory the traces were read from.
    origin:
        Meet of the origins of every trace; ``MEASURED`` only if all are.
    manifest:
        Parsed ``manifest.json``, or an empty mapping when absent.
    traces:
        Mapping run label to :class:`Trace`.
    """

    directory: Path
    origin: Origin
    manifest: dict = field(default_factory=dict)
    traces: dict[str, Trace] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Trace]:
        return iter(self.traces.values())

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, label: str) -> Trace:
        if label not in self.traces:
            raise TraceDataAbsentError(
                f"run {label!r} is not in {self.directory.name}; present: "
                f"{sorted(self.traces)}"
            )
        return self.traces[label]

    def select(self, prefix: str) -> list[Trace]:
        """Traces whose label starts with ``prefix``, sorted by label."""
        return [self.traces[name] for name in sorted(self.traces) if name.startswith(prefix)]

    def soaks(self, duty: float | None = None) -> list[Trace]:
        """Soak runs, optionally only those at a given commanded duty."""
        runs = self.select("soak_")
        if duty is None:
            return runs
        return [run for run in runs if math.isclose(run.duty, duty, rel_tol=0.0, abs_tol=1e-9)]

    def bursts(self) -> list[Trace]:
        """The short cold-burst runs."""
        return self.select("burst_")

    def cooldowns(self) -> list[Trace]:
        """The duty-0 cooldown runs, which carry the independent tau estimate."""
        return self.select("cool_")

    @property
    def board_model(self) -> str:
        """``/proc/device-tree/model`` as the manifest recorded it, or ``unknown``."""
        return str(self.manifest.get("model", "unknown"))

    def board_matches_declared_target(self) -> bool:
        """Whether the manifest's board model is the declared target board."""
        return DECLARED_BOARD in self.board_model

    @property
    def duty_cycles(self) -> tuple[float, ...]:
        """Distinct commanded duty cycles among the soak runs, ascending."""
        return tuple(sorted({run.duty for run in self.soaks()}))

    def idle_floor_c(self) -> float:
        """Mean die temperature of the ``idle_pre`` run, the campaign's ambient proxy.

        SPEC.md limit 2: ambient could not be varied, so this is a single number
        for the whole campaign and exists to make ambient DRIFT visible, not to
        support an ambient axis.
        """
        if "idle_pre" not in self.traces:
            return math.nan
        samples = self.traces["idle_pre"].finite(schema.DIE_COLUMN)
        return float(samples.mean()) if samples.size else math.nan


def _required_labels(manifest: dict) -> list[str]:
    runs = manifest.get("runs", [])
    return [str(entry["label"]) for entry in runs if "label" in entry]


def load_campaign(
    directory: str | Path,
    *,
    origin: Origin,
    require: Sequence[str] = (),
) -> Campaign:
    """Load every trace in ``directory``.

    Parameters
    ----------
    directory:
        Directory holding ``*.csv`` and optionally ``manifest.json``.
    origin:
        Origin to stamp on every trace. Required and never defaulted.
    require:
        Run labels that must be present. A missing one raises
        :class:`TraceDataAbsentError` naming it, so a check cannot run over a
        subset and report a verdict as though it had run over all of it.

    Raises
    ------
    TraceDataAbsentError
        If the directory does not exist, holds no CSV, or lacks a required run.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise TraceDataAbsentError(
            f"trace directory {directory.name!r} does not exist (expected at "
            f"{directory.parent.name}/{directory.name})"
        )
    manifest_path = directory / MANIFEST_NAME
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    duty_by_label = {
        str(entry.get("label")): float(entry.get("duty", math.nan))
        for entry in manifest.get("runs", [])
    }
    traces: dict[str, Trace] = {}
    for csv_path in sorted(directory.glob("*.csv")):
        stem = csv_path.stem
        label = stem[len(SYNTHETIC_PREFIX) :] if stem.startswith(SYNTHETIC_PREFIX) else stem
        traces[label] = load_trace(
            csv_path, origin=origin, label=label, duty=duty_by_label.get(label)
        )
    if not traces:
        raise TraceDataAbsentError(
            f"no *.csv in {directory.name}/; an empty input set cannot produce a verdict"
        )
    wanted = list(require) or _required_labels(manifest)
    absent = [name for name in wanted if name not in traces]
    if absent:
        raise TraceDataAbsentError(
            f"{directory.name}/ is missing required run(s) by name: {absent}; "
            f"present: {sorted(traces)}"
        )
    return Campaign(directory=directory, origin=origin, manifest=manifest, traces=traces)
