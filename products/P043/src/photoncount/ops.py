"""Deployment and recovery as executable checks, not prose.

A procedure written in a document is not a procedure; it is a hope. This module
turns three things into runnable, tested functions:

**Preflight** (:func:`run_preflight`) --- the checks that must pass before an
acquisition is allowed to start. Each returns a value as well as a verdict, so
a failure says what the number was. ``FAIL`` blocks the run; ``WARN`` does not,
but is recorded.

**Capture** (:class:`RunRecord`, :func:`write_run_record`) --- what is kept from
a run: the backend description, the mode, the request, the preflight report, the
per-window counts, the wall-clock time, the seed and the package version. A run
whose record does not say which backend produced it, and whether that backend
was simulated, is not a record.

**Backout** (:class:`RunJournal`, :func:`backout_incomplete_runs`) --- what
happens to a run that stops half way. A journal file is written before the first
window and renamed on success; anything left in the in-progress state is a
half-finished run, and :func:`backout_incomplete_runs` moves each to an aborted
record and reports what it did. Nothing is deleted: a half-finished run is
evidence.

**Paths.** Every function here takes a directory and writes only inside it, by
bare filename. Records store **relative** paths only. That is a deliberate
constraint: an absolute path in a committed artefact is a defect in this
repository, and the easiest way not to write one is to have no code that can.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import __version__
from .deadtime import paralyzable_maximum
from .hal import Acquisition, AcquisitionRequest, BackendMode, PhotonCountingBackend

__all__ = [
    "IN_PROGRESS_SUFFIX",
    "CheckStatus",
    "PreflightCheck",
    "PreflightReport",
    "RunJournal",
    "RunRecord",
    "backout_incomplete_runs",
    "environment_summary",
    "read_run_record",
    "run_preflight",
    "write_run_record",
]

CheckStatus = Literal["PASS", "WARN", "FAIL"]

#: Journal files carrying this suffix mark a run that started and did not finish.
IN_PROGRESS_SUFFIX = ".inprogress.json"

_ABORTED_SUFFIX = ".aborted.json"
_COMPLETE_SUFFIX = ".json"


@dataclass(frozen=True)
class PreflightCheck:
    """One preflight verdict.

    Attributes
    ----------
    name:
        Stable identifier, safe to assert on in a test.
    status:
        ``"PASS"``, ``"WARN"`` or ``"FAIL"``.
    value:
        The number or string the check looked at, so a failure is diagnosable.
    detail:
        One sentence explaining the verdict.
    """

    name: str
    status: CheckStatus
    value: float | str
    detail: str


@dataclass(frozen=True)
class PreflightReport:
    """The whole preflight outcome."""

    checks: tuple[PreflightCheck, ...]

    @property
    def passed(self) -> bool:
        """True when no check is ``FAIL``. A ``WARN`` does not block a run."""
        return not any(c.status == "FAIL" for c in self.checks)

    @property
    def failures(self) -> tuple[PreflightCheck, ...]:
        """Only the ``FAIL`` checks."""
        return tuple(c for c in self.checks if c.status == "FAIL")

    @property
    def warnings(self) -> tuple[PreflightCheck, ...]:
        """Only the ``WARN`` checks."""
        return tuple(c for c in self.checks if c.status == "WARN")

    def names(self) -> tuple[str, ...]:
        """Check names in order."""
        return tuple(c.name for c in self.checks)

    def as_dicts(self) -> list[dict[str, Any]]:
        """JSON-ready list of check dicts."""
        return [asdict(c) for c in self.checks]


def environment_summary() -> dict[str, str]:
    """Where this ran, in terms safe to commit.

    Deliberately excludes every filesystem path. Includes the Python version,
    the platform string, the machine architecture and the processor count,
    which is what a reader needs to know that a timing number came from a
    shared cloud container rather than from a Jetson Orin Nano.
    """
    import os

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": str(os.cpu_count()),
        "photoncount_version": __version__,
    }


def run_preflight(
    backend: PhotonCountingBackend,
    request: AcquisitionRequest,
    expected_rate_hz: float,
    directory: str | Path = ".",
    window_dead_time_ratio: float = 1000.0,
) -> PreflightReport:
    """The checks that must pass before an acquisition starts.

    Checks, in order, with the identifiers the tests assert on:

    ``backend_described``
        :meth:`describe` returns every required key. FAIL otherwise.
    ``dead_time_declared``
        The declared dead time is finite and ``>= 0``. FAIL otherwise; a run
        whose dead time is unknown cannot be corrected afterwards.
    ``window_exceeds_dead_time``
        ``window_s >= window_dead_time_ratio * tau``. FAIL otherwise: a window
        comparable with the dead time makes the per-window count statistics
        meaningless.
    ``rate_below_saturation``
        For a paralyzable detector, ``expected_rate_hz < 1/tau`` (FAIL at or
        above, where the observed rate starts *falling* with illumination), and
        WARN above ``0.3/tau`` where the inversion is ill-conditioned. For a
        non-paralyzable detector, WARN above ``0.5/tau``.
    ``inversion_branch_unambiguous``
        For a paralyzable detector, WARN when the expected observed rate is
        within 10 % of the maximum ``1/(e tau)``, where the two inverse branches
        are close together and the lower branch cannot be assumed.
    ``afterpulsing_declared``
        WARN when the declared afterpulse probability is 0, which usually means
        unmeasured rather than absent.
    ``directory_writable``
        The capture directory exists (or can be created) and a probe file can be
        written and removed. FAIL otherwise.
    ``backend_self_test``
        :meth:`self_test` reports ``passed``. FAIL when it does not; recorded as
        WARN with the exception text when the backend raises
        ``NotImplementedError``, which is the expected state of a device backend
        before hardware exists.

    ``expected_rate_hz`` is the caller's *estimate* of the true rate, from a
    link budget. It is not measured here, and a wrong estimate makes the
    saturation checks wrong; that is stated in the record.
    """
    from .hal import REQUIRED_DESCRIPTION_KEYS

    checks: list[PreflightCheck] = []
    try:
        desc = backend.describe()
    except Exception as exc:  # pragma: no cover - a backend that cannot describe itself
        return PreflightReport(
            (
                PreflightCheck(
                    "backend_described", "FAIL", type(exc).__name__, f"describe() raised: {exc}"
                ),
            )
        )

    missing = [k for k in REQUIRED_DESCRIPTION_KEYS if k not in desc]
    checks.append(
        PreflightCheck(
            "backend_described",
            "PASS" if not missing else "FAIL",
            str(desc.get("name", "unknown")),
            "all required description keys present"
            if not missing
            else f"describe() is missing {missing}",
        )
    )

    tau = float(desc.get("dead_time_s", float("nan")))
    model = str(desc.get("dead_time_model", "unknown"))
    tau_ok = np.isfinite(tau) and tau >= 0.0
    checks.append(
        PreflightCheck(
            "dead_time_declared",
            "PASS" if tau_ok else "FAIL",
            tau,
            f"declared dead time {tau!r} s, model {model!r}"
            if tau_ok
            else "dead time is not a finite non-negative number",
        )
    )

    if tau_ok and tau > 0.0:
        ratio = request.window_s / tau
        checks.append(
            PreflightCheck(
                "window_exceeds_dead_time",
                "PASS" if ratio >= window_dead_time_ratio else "FAIL",
                ratio,
                f"window is {ratio:.3g} dead times long "
                f"(required >= {window_dead_time_ratio:g})",
            )
        )
        n_max, m_max = paralyzable_maximum(tau)
        rate = float(expected_rate_hz)
        if model == "paralyzable":
            if rate >= n_max:
                status: CheckStatus = "FAIL"
                detail = (
                    f"expected rate {rate:.4g} counts/s is at or above 1/tau = "
                    f"{n_max:.4g}, where the observed rate falls as illumination rises"
                )
            elif rate > 0.3 * n_max:
                status = "WARN"
                detail = (
                    f"expected rate {rate:.4g} counts/s exceeds 0.3/tau = "
                    f"{0.3 * n_max:.4g}; the paralyzable inversion is ill-conditioned here"
                )
            else:
                status = "PASS"
                detail = f"expected rate {rate:.4g} counts/s is {rate / n_max:.3f} of 1/tau"
            checks.append(PreflightCheck("rate_below_saturation", status, rate, detail))
            observed = rate * float(np.exp(-rate * tau))
            near = observed > 0.9 * m_max
            checks.append(
                PreflightCheck(
                    "inversion_branch_unambiguous",
                    "WARN" if near else "PASS",
                    observed / m_max,
                    f"expected observed rate is {observed / m_max:.3f} of the paralyzable "
                    f"maximum {m_max:.4g} counts/s"
                    + ("; the two inverse branches are close, do not assume the lower" if near
                       else ""),
                )
            )
        else:
            over = rate > 0.5 * n_max
            checks.append(
                PreflightCheck(
                    "rate_below_saturation",
                    "WARN" if over else "PASS",
                    rate,
                    f"expected rate {rate:.4g} counts/s is {rate / n_max:.3f} of 1/tau"
                    + ("; above 0.5/tau the correction amplifies count noise" if over else ""),
                )
            )
            checks.append(
                PreflightCheck(
                    "inversion_branch_unambiguous",
                    "PASS",
                    "nonparalyzable",
                    "the non-paralyzable inverse is single-valued",
                )
            )
    else:
        checks.append(
            PreflightCheck(
                "window_exceeds_dead_time", "PASS", float("inf"),
                "dead time is zero; no window constraint",
            )
        )
        checks.append(
            PreflightCheck(
                "rate_below_saturation", "PASS", float(expected_rate_hz),
                "dead time is zero; the detector cannot saturate",
            )
        )
        checks.append(
            PreflightCheck(
                "inversion_branch_unambiguous", "PASS", "ideal",
                "dead time is zero; the inverse is the identity",
            )
        )

    p_ap = float(desc.get("afterpulse_probability", 0.0))
    checks.append(
        PreflightCheck(
            "afterpulsing_declared",
            "PASS" if p_ap > 0.0 else "WARN",
            p_ap,
            f"declared afterpulse probability {p_ap:.4g}"
            + (
                "; zero usually means unmeasured rather than absent"
                if p_ap == 0.0
                else ""
            ),
        )
    )

    root = Path(directory)
    probe = root / f".preflight_probe_{request.label}"
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe.write_text("probe\n", encoding="utf-8")
        probe.unlink()
        checks.append(
            PreflightCheck(
                "directory_writable", "PASS", probe.name, "capture directory is writable"
            )
        )
    except OSError as exc:
        checks.append(
            PreflightCheck(
                "directory_writable", "FAIL", type(exc).__name__,
                f"capture directory is not writable: {exc.strerror}",
            )
        )

    try:
        st = backend.self_test()
        ok = bool(st.get("passed", False))
        checks.append(
            PreflightCheck(
                "backend_self_test",
                "PASS" if ok else "FAIL",
                str(st.get("detail", "")),
                "backend self-test passed" if ok else "backend self-test failed",
            )
        )
    except NotImplementedError as exc:
        checks.append(
            PreflightCheck(
                "backend_self_test", "WARN", "NotImplementedError",
                f"backend cannot self-test without hardware: {exc}",
            )
        )
    return PreflightReport(tuple(checks))


@dataclass
class RunRecord:
    """Everything kept from one run. Serialised by :func:`write_run_record`.

    ``relative_paths_only`` is not a field: it is a property of the design. No
    attribute of this record holds a filesystem path beyond a bare filename.
    """

    label: str
    backend_description: dict[str, Any]
    mode: str
    simulated: bool
    request: dict[str, Any]
    preflight: list[dict[str, Any]]
    preflight_passed: bool
    counts: list[int]
    observed_rate_hz: float
    fano_factor: float
    wall_seconds: float
    seed: int | None
    environment: dict[str, str] = field(default_factory=environment_summary)
    metadata: dict[str, Any] = field(default_factory=dict)
    started_unix: float = field(default_factory=time.time)

    @classmethod
    def from_acquisition(
        cls,
        acquisition: Acquisition,
        request: AcquisitionRequest,
        backend: PhotonCountingBackend,
        preflight: PreflightReport,
        seed: int | None = None,
    ) -> RunRecord:
        """Build a record from a finished acquisition."""
        return cls(
            label=request.label,
            backend_description={k: v for k, v in backend.describe().items()},
            mode=acquisition.mode.value,
            simulated=acquisition.simulated,
            request={
                "window_s": request.window_s,
                "n_windows": int(request.n_windows),
                "label": request.label,
            },
            preflight=preflight.as_dicts(),
            preflight_passed=preflight.passed,
            counts=[int(c) for c in acquisition.counts],
            observed_rate_hz=acquisition.observed_rate_hz,
            fano_factor=acquisition.fano_factor,
            wall_seconds=acquisition.wall_seconds,
            seed=seed,
            metadata=dict(acquisition.metadata),
        )

    def to_json(self) -> str:
        """Pretty JSON, keys sorted, NaN rendered as ``null``."""

        def default(obj: Any) -> Any:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, BackendMode):
                return obj.value
            raise TypeError(f"cannot serialise {type(obj).__name__}")

        payload = asdict(self)
        payload = json.loads(
            json.dumps(payload, default=default, allow_nan=True)
            .replace("NaN", "null")
            .replace("Infinity", "null")
        )
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_run_record(record: RunRecord, directory: str | Path = ".") -> str:
    """Write ``<label>.json`` into ``directory``. Returns the bare filename.

    The return value is a filename, not a path, so that a caller logging it
    cannot leak an absolute path into a committed artefact.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    name = f"{record.label}{_COMPLETE_SUFFIX}"
    (root / name).write_text(record.to_json(), encoding="utf-8")
    return name


def read_run_record(filename: str, directory: str | Path = ".") -> dict[str, Any]:
    """Read a record written by :func:`write_run_record`. Bare filename only."""
    if any(c in filename for c in "/\\"):
        raise ValueError(f"filename must be a bare name, got {filename!r}")
    return json.loads((Path(directory) / filename).read_text(encoding="utf-8"))


class RunJournal:
    """Crash-safe marker for a run in progress.

    Usage::

        journal = RunJournal("night_01", directory="runs")
        journal.begin({"window_s": 1e-3, "n_windows": 500})
        ...                       # acquire
        journal.commit(record)    # renames the in-progress file away

    If the process dies between ``begin`` and ``commit``, a file
    ``night_01.inprogress.json`` is left behind. That file is the signal
    :func:`backout_incomplete_runs` looks for. ``begin`` refuses to overwrite an
    existing in-progress journal, because doing so would destroy the evidence of
    the previous failure.
    """

    def __init__(self, label: str, directory: str | Path = ".") -> None:
        if not label or any(c in label for c in "/\\"):
            raise ValueError(f"label must be a bare name, got {label!r}")
        self.label = label
        self.directory = Path(directory)

    @property
    def in_progress_name(self) -> str:
        """Bare filename of the in-progress journal."""
        return f"{self.label}{IN_PROGRESS_SUFFIX}"

    @property
    def is_in_progress(self) -> bool:
        """Whether an in-progress journal for this label exists."""
        return (self.directory / self.in_progress_name).exists()

    def begin(self, request: dict[str, Any]) -> str:
        """Write the in-progress journal. Returns its bare filename."""
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / self.in_progress_name
        if target.exists():
            raise FileExistsError(
                f"{self.in_progress_name} already exists: a previous run of label "
                f"{self.label!r} did not finish. Run backout_incomplete_runs first; "
                "do not overwrite it."
            )
        payload = {
            "label": self.label,
            "request": request,
            "started_unix": time.time(),
            "environment": environment_summary(),
            "state": "in_progress",
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return self.in_progress_name

    def commit(self, record: RunRecord) -> str:
        """Write the completed record and remove the in-progress journal."""
        name = write_run_record(record, self.directory)
        (self.directory / self.in_progress_name).unlink(missing_ok=True)
        return name


def backout_incomplete_runs(directory: str | Path = ".") -> list[dict[str, str]]:
    """Back out every half-finished run in ``directory``.

    For each ``<label>.inprogress.json`` found, the file is renamed to
    ``<label>.aborted.json`` with ``state`` set to ``"aborted"`` and an
    ``aborted_unix`` timestamp added. Nothing is deleted and no completed record
    is touched. Returns one dict per backed-out run with keys ``label``,
    ``from``, ``to`` --- bare filenames only.

    Idempotent: a second call finds nothing and returns an empty list.
    """
    root = Path(directory)
    if not root.is_dir():
        return []
    out: list[dict[str, str]] = []
    for path in sorted(root.glob(f"*{IN_PROGRESS_SUFFIX}")):
        label = path.name[: -len(IN_PROGRESS_SUFFIX)]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {"label": label, "note": "journal unreadable"}
        payload["state"] = "aborted"
        payload["aborted_unix"] = time.time()
        dest = root / f"{label}{_ABORTED_SUFFIX}"
        dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        path.unlink()
        out.append({"label": label, "from": path.name, "to": dest.name})
    return out
