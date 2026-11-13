"""Loading the campaign's traces, and the measurement model every number uses.

MEASUREMENT MODEL, stated here because this is the module every other module
gets its numbers through. It is restated verbatim in ``README.md`` and in
``validation/VALIDATION.md``.

- The loop has period ``T`` and an **implicit deadline** ``D = T``.
- Job ``i`` is **released** at ``t0 + i*T``. Its **response time** is
  ``end - release``, not ``end - start``: the time a job spends waiting to be
  scheduled is part of the response the deadline is stated against.
- A job whose response time exceeds ``D`` is a **miss**.
- If a job finishes after one or more later releases have already passed, those
  releases are never executed. They are **dropped**, and **each dropped release
  is also a miss** -- the work that release represented was not done by its
  deadline, which is the only thing the deadline says. Every executed row
  carries ``dropped_before``, the number of releases skipped between that row
  and the next executed row, so both totals are reconstructible from the
  committed traces:

  ``releases = sum(1 + dropped_before)`` and
  ``misses = sum(missed + dropped_before)``.

- The response times in a trace are therefore the response times of the
  **executed** jobs only. A dropped release has no response time, because
  nothing ran. This matters for the mean-plus-margin rule, which is computed
  from measured response times and so never sees the dropped releases at all;
  ``baselines.mean_plus_margin`` says so in its own docstring.

Units are SI seconds throughout, except where a column name says otherwise:
``tj_mdegC`` is millidegrees Celsius as the kernel reports it and is converted
to degrees Celsius on load, and clock columns are kHz as ``scaling_cur_freq``
reports them and are left in kHz.

Source of the model: the implicit-deadline periodic task model of Liu & Layland
1973, "Scheduling Algorithms for Multiprogramming in a Hard-Real-Time
Environment", JACM 20(1):46-61, with the dropped-release accounting that a
non-queueing periodic loop forces. Validity: one task, one period, no queueing
of missed releases. Nothing here models a server with a backlog.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .provenance import Origin, Provenance
from .schema import validate_job_header, validate_monitor_header


class TraceDataAbsentError(FileNotFoundError):
    """The traces a check requires are not on disk. There is no fallback.

    A check that computes a MEASURED figure fails loudly with the names of the
    files it looked for. Substituting the synthetic generator here would produce
    a figure that looks like a measurement and is not one.
    """


@dataclass(frozen=True)
class JobTrace:
    """One run's executed-job rows, plus the period and deadline they are against.

    All time fields are seconds. ``missed`` and ``dropped_before`` are integer
    counts. ``response_s`` is ``end - release`` as written by the campaign.
    """

    run: str
    period_s: float
    deadline_s: float
    job_index: np.ndarray
    release_s: np.ndarray
    start_delay_s: np.ndarray
    end_s: np.ndarray
    exec_s: np.ndarray
    response_s: np.ndarray
    missed: np.ndarray
    dropped_before: np.ndarray
    origin: Origin

    def __len__(self) -> int:
        return int(self.job_index.size)

    @property
    def duration_s(self) -> float:
        """Wall time from the first release to the last job end, in seconds."""
        if self.job_index.size == 0:
            return 0.0
        return float(self.end_s[-1] - self.release_s[0])


@dataclass(frozen=True)
class MonitorTrace:
    """One run's 1 Hz contention descriptors.

    ``foreign_pct``, ``task_pct`` and ``stress_pct`` are percentages of total
    CPU capacity over the sample second, from ``/proc/stat`` jiffy deltas.
    ``tj_degC`` is degrees Celsius, converted from the kernel's millidegrees.
    Clocks are kHz.
    """

    run: str
    t_s: np.ndarray
    foreign_pct: np.ndarray
    task_pct: np.ndarray
    stress_pct: np.ndarray
    clk_mean_khz: np.ndarray
    clk_min_khz: np.ndarray
    tj_degC: np.ndarray
    load1: np.ndarray

    def __len__(self) -> int:
        return int(self.t_s.size)


@dataclass(frozen=True)
class Campaign:
    """A whole campaign: manifest, per-run job traces, per-run monitor traces."""

    manifest: dict
    jobs: dict[str, JobTrace]
    monitors: dict[str, MonitorTrace]
    provenance: Provenance

    @property
    def period_s(self) -> float:
        """Loop period in seconds, from the manifest."""
        return float(self.manifest["period_s"])

    @property
    def deadline_s(self) -> float:
        """Implicit deadline in seconds, from the manifest."""
        return float(self.manifest["deadline_s"])

    @property
    def complete(self) -> bool:
        """Whether the campaign wrote its completion flag into the manifest."""
        return bool(self.manifest.get("complete", False))

    def run_meta(self, run: str) -> dict:
        """The manifest's metadata block for ``run``, or ``{}`` if it has none."""
        for entry in self.manifest.get("runs", []):
            if entry.get("run") == run:
                return entry
        return {}

    def stressors(self, run: str) -> tuple[int, int]:
        """``(n_cpu_stress, n_mem_stress)`` for ``run`` from the manifest."""
        meta = self.run_meta(run)
        if "n_cpu_stress" not in meta:
            raise KeyError(
                f"run {run!r} has no n_cpu_stress in the manifest; "
                f"runs present: {sorted(self.jobs)}"
            )
        return int(meta["n_cpu_stress"]), int(meta.get("n_mem_stress", 0))


def _read_rows(path: Path, validator) -> dict[str, list[str]]:
    if not path.is_file():
        raise TraceDataAbsentError(f"no such trace file: {path.name}")
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise TraceDataAbsentError(f"trace file {path.name} is empty") from exc
        validator(header)
        columns: dict[str, list[str]] = {name.strip(): [] for name in header}
        names = [name.strip() for name in header]
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(names):
                raise ValueError(
                    f"{path.name} line {line_number}: {len(row)} field(s), "
                    f"expected {len(names)}"
                )
            for name, value in zip(names, row, strict=True):
                columns[name].append(value)
    return columns


def load_job_trace(
    path: Path | str,
    run: str,
    period_s: float,
    deadline_s: float,
    origin: Origin = Origin.MEASURED,
) -> JobTrace:
    """Load one ``jobs_<run>.csv``.

    Parameters
    ----------
    path
        File to read.
    run
        Run label to attach.
    period_s, deadline_s
        Loop period and implicit deadline, seconds. Both must be positive and
        finite; ``deadline_s`` must not exceed ``period_s``, because a deadline
        later than the next release is not an implicit deadline and the drop
        accounting in this module would not describe it.
    origin
        :class:`Origin` of the file.

    Raises
    ------
    ValueError
        On a non-positive period or deadline, a deadline past the period, a
        negative time field, a ``missed`` flag that is not 0 or 1, a negative
        ``dropped_before``, or a non-increasing ``job_index``.
    SchemaError
        On a header that is not exactly the declared one.
    """
    if not np.isfinite(period_s) or period_s <= 0.0:
        raise ValueError(f"period_s must be finite and > 0, got {period_s!r}")
    if not np.isfinite(deadline_s) or deadline_s <= 0.0:
        raise ValueError(f"deadline_s must be finite and > 0, got {deadline_s!r}")
    if deadline_s > period_s:
        raise ValueError(
            f"deadline_s ({deadline_s}) exceeds period_s ({period_s}); this module's "
            "drop accounting assumes an implicit deadline D <= T"
        )
    columns = _read_rows(Path(path), validate_job_header)
    index = np.asarray(columns["job_index"], dtype=np.int64)
    release = np.asarray(columns["release_s"], dtype=np.float64)
    delay = np.asarray(columns["start_delay_s"], dtype=np.float64)
    end = np.asarray(columns["end_s"], dtype=np.float64)
    execution = np.asarray(columns["exec_s"], dtype=np.float64)
    response = np.asarray(columns["response_s"], dtype=np.float64)
    missed = np.asarray(columns["missed"], dtype=np.int64)
    dropped = np.asarray(columns["dropped_before"], dtype=np.int64)

    if index.size and np.any(np.diff(index) <= 0):
        bad = int(np.argmin(np.diff(index)))
        raise ValueError(
            f"{Path(path).name}: job_index must be strictly increasing; it is not at "
            f"row {bad + 1} (values {int(index[bad])} then {int(index[bad + 1])})"
        )
    for name, values in (("release_s", release), ("end_s", end), ("exec_s", execution)):
        if values.size and np.any(values < 0.0):
            raise ValueError(f"{Path(path).name}: {name} has a negative value")
    if not np.all(np.isin(missed, (0, 1))):
        raise ValueError(
            f"{Path(path).name}: missed must be 0 or 1, found "
            f"{sorted(set(missed.tolist()) - {0, 1})}"
        )
    if np.any(dropped < 0):
        raise ValueError(f"{Path(path).name}: dropped_before has a negative value")
    return JobTrace(
        run=run,
        period_s=float(period_s),
        deadline_s=float(deadline_s),
        job_index=index,
        release_s=release,
        start_delay_s=delay,
        end_s=end,
        exec_s=execution,
        response_s=response,
        missed=missed,
        dropped_before=dropped,
        origin=origin,
    )


def load_monitor_trace(path: Path | str, run: str) -> MonitorTrace:
    """Load one ``mon_<run>.csv``, converting ``tj_mdegC`` to degrees Celsius."""
    columns = _read_rows(Path(path), validate_monitor_header)

    def column(name: str) -> np.ndarray:
        return np.asarray(columns[name], dtype=np.float64)

    return MonitorTrace(
        run=run,
        t_s=column("t_s"),
        foreign_pct=column("foreign_pct"),
        task_pct=column("task_pct"),
        stress_pct=column("stress_pct"),
        clk_mean_khz=column("clk_mean_khz"),
        clk_min_khz=column("clk_min_khz"),
        tj_degC=column("tj_mdegC") / 1000.0,
        load1=column("load1"),
    )


def campaign_is_complete(directory: Path | str) -> bool:
    """Whether ``CAMPAIGN_COMPLETE`` exists in ``directory``.

    The campaign script writes that zero-byte marker last of all, after every
    per-run CSV and after the manifest. Its absence means the campaign had not
    finished when the directory was read, not that it failed.
    """
    return (Path(directory) / "CAMPAIGN_COMPLETE").exists()


def load_campaign(
    directory: Path | str,
    origin: Origin = Origin.MEASURED,
    require: tuple[str, ...] = (),
    directory_label: str | None = None,
) -> Campaign:
    """Load ``manifest.json`` and every ``jobs_<run>.csv`` beside it.

    Parameters
    ----------
    directory
        Directory holding the campaign's files.
    origin
        :class:`Origin` to record. Nothing about the files is inspected to guess
        it: the caller knows which generator wrote them.
    require
        Run labels that must be present. A missing one raises
        :class:`TraceDataAbsentError` naming every absent file.
    directory_label
        Label to record in the provenance instead of the directory's own path.
        Used so that a report never prints an absolute path.

    Raises
    ------
    TraceDataAbsentError
        If the directory, the manifest, or any required run is absent.
    """
    root = Path(directory)
    label = directory_label if directory_label is not None else root.name
    if not root.is_dir():
        raise TraceDataAbsentError(
            f"campaign directory {label}/ does not exist. Nothing was substituted: "
            "a measured figure has no synthetic fallback."
        )
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise TraceDataAbsentError(
            f"{label}/manifest.json is absent. The campaign writes it after each run, "
            "so its absence means no run has completed."
        )
    manifest = json.loads(manifest_path.read_text())
    for key in ("period_s", "deadline_s"):
        if key not in manifest:
            raise ValueError(f"{label}/manifest.json has no {key!r}")
    period = float(manifest["period_s"])
    deadline = float(manifest["deadline_s"])

    found = sorted(path.name[len("jobs_") : -len(".csv")] for path in root.glob("jobs_*.csv"))
    absent = [run for run in require if run not in found]
    if absent:
        raise TraceDataAbsentError(
            f"required run(s) absent from {label}/: "
            + ", ".join(f"jobs_{run}.csv" for run in absent)
            + f". Runs present: {found or 'none'}. Nothing was substituted."
        )
    jobs: dict[str, JobTrace] = {}
    monitors: dict[str, MonitorTrace] = {}
    for run in found:
        jobs[run] = load_job_trace(
            root / f"jobs_{run}.csv", run, period, deadline, origin=origin
        )
        monitor_path = root / f"mon_{run}.csv"
        if monitor_path.is_file():
            monitors[run] = load_monitor_trace(monitor_path, run)
    return Campaign(
        manifest=manifest,
        jobs=jobs,
        monitors=monitors,
        provenance=Provenance(origin=origin, directory=label, runs=tuple(found)),
    )
