"""Path bootstrap, output tee and declared constants for every script here.

Six things in this module exist because of defects this mission has shipped.

**The working directory.** Every script calls :func:`bootstrap` as its first
statement after setting ``sys.dont_write_bytecode``. It puts ``<repo>/src`` and
``<repo>/tests`` on ``sys.path`` and then changes the process working directory
to the repository root, derived from ``__file__`` and never from the caller.
Three products in one batch of this mission passed every check in their own
build and then failed the release gate, which runs each validation script with
its working directory set to the script's own directory rather than to the
product root. ``probe_working_directories.py`` runs every script from four
directories and requires the same exit status from all four.

**Absolute paths.** No committed file here may contain an absolute path: it is
not reproducible for a reader with a cold clone and it publishes the build
machine's filesystem layout. Everything printed goes through :func:`relative`,
and the repository root prints as its own directory name and nothing above it.

**Subprocess environments.** :func:`subprocess_environment` points ``HOME`` and
``MPLCONFIGDIR`` at a temporary directory outside the repository and sets
``PYTHONDONTWRITEBYTECODE``, because a validation script in an earlier product
set ``HOME`` to the product root and matplotlib wrote ``.config/matplotlib``
into the repository, which the release gate fails twice over.

**Bytecode.** Every runnable script here sets ``sys.dont_write_bytecode = True``
as its first statement, before any other import. It cannot be delegated to this
module, because CPython writes a module's bytecode cache BEFORE executing that
module, so this module cannot set it on its own behalf. The subsequent
module-level imports then carry ``# noqa: E402``.

**Measured data is required, by name, and is never substituted.** The scripts
split into two kinds and the split is most of the point of this product.
:func:`require_measured_campaign` loads ``data/traces/`` and raises if it is
absent or incomplete, naming the cells it looked for. The scripts that call it
compute MEASURED figures and they fail loudly rather than falling back to the
synthetic generator. The scripts that call :func:`synthetic_campaign` compute
SYNTHETIC figures, label every number so, and are checks of the arithmetic
rather than measurements of anything.

**The completion marker is part of the requirement.** The campaign writes
``CAMPAIGN_COMPLETE`` last of all. A measured script run against a directory
without it would publish a figure from a partial campaign, so
:func:`require_measured_campaign` refuses unless the marker exists, and says so.

``POWERENVELOPE_TRACES`` changes WHICH directory a measured script requires. It
exists so that the refusal path can be tested hermetically, by pointing it at an
empty directory. It never makes a measurement optional, it never selects the
synthetic generator, and a directory whose ``manifest.json`` declares itself
synthetic is REFUSED by :func:`require_measured_campaign` rather than reported
as measured.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Where the campaign's MEASURED traces live. Relative, always.
MEASURED_TRACES = "data/traces"

#: Environment variable that moves the required directory. See the module
#: docstring: it changes WHICH directory is required, never WHETHER one is.
TRACES_ENVIRONMENT_VARIABLE = "POWERENVELOPE_TRACES"

#: The cells every measured check requires by name. A campaign missing any of
#: these cannot produce the headline, the between-run interval, the permissive
#: control or either held-out split, and the loader says which are absent.
REQUIRED_CELLS: tuple[str, ...] = (
    "floor",
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
    "w64_d050_permissive",
)

#: The cell the headline is measured on, and no other.
HEADLINE_CELL = "w64_d050"

#: The permissive-gate control cell.
PERMISSIVE_CELL = "w64_d050_permissive"

#: The token that stands where a measured number belongs and is not yet known.
#: It is never replaced by an estimate, an interpolation or an illustration.
MEASURED_PENDING = "MEASURED_PENDING"

#: Declared seeds. One place, so no two reported numbers rest on different ones.
SEED_ANALYSIS = 930093
SEED_FIXTURE = 930093
SEED_CONTROL = 5150

#: ``POWERENVELOPE_FAST`` lowers example counts and is used only by
#: ``probe_working_directories.py``, whose question is whether a script runs
#: from a given directory and not what the numbers are.
FAST = os.environ.get("POWERENVELOPE_FAST") == "1"

#: Hypothesis examples for the property hunt.
HUNT_EXAMPLES = 60 if FAST else 600

#: Confidence level of every reported interval.
CONFIDENCE = 0.95


def bootstrap() -> Path:
    """Put ``src`` and ``tests`` on ``sys.path``, chdir to the root, return it."""
    for directory in (ROOT / "src", ROOT / "tests"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    os.chdir(ROOT)
    return ROOT


def relative(path: str | Path) -> str:
    """Render ``path`` relative to the repository root, never absolutely."""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except ValueError:
        return candidate.name


def measurement_model() -> str:
    """The measurement model, from the one place it is written down.

    Imported lazily: this module is imported before :func:`bootstrap` has put
    ``src`` on ``sys.path``.
    """
    from powerenvelope.report import MEASUREMENT_MODEL

    return MEASUREMENT_MODEL


class Tee:
    """Collect report lines, print them, and write them to a file at the end.

    Every script writes its raw output beside itself as
    ``<script stem>_output.txt`` so that every number in ``README.md`` and
    ``validation/VALIDATION.md`` traces to a file produced by a script that was
    actually run.
    """

    def __init__(self, path: str | Path, directory: str | Path | None = None) -> None:
        # ``directory`` exists so that a TEST can exercise a script's reporting
        # code without overwriting a committed output file. Scripts never pass
        # it; they write beside themselves, which is the one committed layout.
        base = Path(directory) if directory is not None else ROOT / "validation"
        self.path = base / Path(path).name
        self.lines: list[str] = []
        self.failures: list[str] = []
        self.unmeasurable: list[str] = []
        self.checks = 0
        self.controls_expected = 0
        self.controls_fired = 0
        self.must_fire = 0
        self.must_fire_fired = 0
        self.must_not_fire = 0
        self.must_not_fire_silent = 0

    def __call__(self, text: str = "") -> None:
        for line in str(text).splitlines() or [""]:
            self.lines.append(line)
            print(line)

    def rule(self, title: str) -> None:
        self("")
        self("=" * 78)
        self(title)
        self("=" * 78)

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        """Record a named check. A failure is reported, never suppressed."""
        self.checks += 1
        verdict = "PASS" if passed else "FAIL"
        self(f"[{verdict}] {name}{(' -- ' + detail) if detail else ''}")
        if not passed:
            self.failures.append(name)
        return passed

    def pending(self, name: str, reason: str) -> None:
        """Record a check that cannot run until the measured campaign lands.

        It does not pass. It is counted separately, printed with the
        :data:`MEASURED_PENDING` token so a reader can grep for it, and
        summarised at the end of the file.
        """
        self.unmeasurable.append(name)
        self(f"[{MEASURED_PENDING}] {name}")
        self(f"           reason: {reason}")

    def unmeasurable_check(self, name: str, reason: str) -> None:
        """Record a check that CANNOT run in this campaign at all.

        Counted separately from both passes and failures. A check reported this
        way is neither evidence for nor evidence against anything.
        """
        self.unmeasurable.append(name)
        self(f"[STRUCTURALLY UNMEASURABLE] {name}")
        self(f"           reason: {reason}")

    def control(self, name: str, fired: bool, must_fire: bool, detail: str = "") -> bool:
        """Record a negative control and whether it FIRED as declared.

        A control that does not fire is a failure of the control, not a success
        of the product: it means the check it guards would not have caught the
        violation it exists to catch. Both directions are asserted, because a
        control that fires when it must not is equally wrong.
        """
        self.controls_expected += 1
        passed = fired == must_fire
        if fired:
            self.controls_fired += 1
        if must_fire:
            self.must_fire += 1
            if fired:
                self.must_fire_fired += 1
        else:
            self.must_not_fire += 1
            if not fired:
                self.must_not_fire_silent += 1
        expectation = "must fire" if must_fire else "must NOT fire"
        status = "FIRED" if fired else "did not fire"
        self.check(f"control: {name} ({expectation})", passed, f"{status}; {detail}")
        return passed

    def save(self) -> str:
        """Write the collected lines beside this script and return the relative path."""
        self("")
        self(f"CHECKS RUN: {self.checks}")
        if self.controls_expected:
            self(
                f"CONTROLS: {self.controls_expected} declared, "
                f"{self.controls_fired} fired; "
                f"must-fire {self.must_fire_fired}/{self.must_fire}, "
                f"must-NOT-fire silent {self.must_not_fire_silent}/{self.must_not_fire}"
            )
        if self.unmeasurable:
            self(
                f"NOT MEASURED ({len(self.unmeasurable)}): " + ", ".join(self.unmeasurable)
            )
        if self.failures:
            self(f"FAILED CHECKS ({len(self.failures)}): {', '.join(self.failures)}")
        else:
            self("FAILED CHECKS (0): none")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n")
        return relative(self.path)


def _distribution_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def environment_lines() -> list[str]:
    """Versions, core count and platform, recorded in the output of every script.

    The ARCHITECTURE line matters here. The MEASURED traces were produced on
    aarch64 on the board; every piece of ARITHMETIC in this repository runs on
    the machine named below. Those are different machines and the distinction is
    printed rather than left to be inferred.
    """
    import platform

    import numpy

    return [
        f"python           : {sys.version.split()[0]}",
        f"numpy            : {numpy.__version__}",
        f"scipy            : {_distribution_version('scipy')}",
        f"scikit-learn     : {_distribution_version('scikit-learn')}",
        f"pytest           : {_distribution_version('pytest')}",
        f"hypothesis       : {_distribution_version('hypothesis')}",
        f"matplotlib       : {_distribution_version('matplotlib')}",
        f"ruff             : {_distribution_version('ruff')}",
        f"analysis machine : {platform.machine()} {platform.system()}, "
        f"{len(os.sched_getaffinity(0))} cores  (the TRACES are measured on aarch64 ON "
        "THE BOARD; the arithmetic over them runs here)",
        f"repository root  : {ROOT.name} (printed relatively; see validation/_bootstrap.py)",
        f"hunt examples    : {HUNT_EXAMPLES}"
        + ("  [POWERENVELOPE_FAST=1: reduced, path probe only]" if FAST else ""),
    ]


def subprocess_environment() -> dict[str, str]:
    """Environment for a spawned subprocess that cannot pollute this repository."""
    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="powerenvelope-validation-"))
    environment = dict(os.environ)
    environment.update(
        {
            "HOME": str(scratch),
            "MPLCONFIGDIR": str(scratch / "mpl"),
            "MPLBACKEND": "Agg",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT / "tests")]),
            "HYPOTHESIS_STORAGE_DIRECTORY": str(scratch / "hypothesis"),
        }
    )
    return environment


def measured_trace_label() -> str:
    """The relative label of the directory the measured scripts require."""
    override = os.environ.get(TRACES_ENVIRONMENT_VARIABLE)
    if not override:
        return MEASURED_TRACES
    candidate = Path(override)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except ValueError:
        return candidate.name


def measured_trace_directory() -> Path:
    """Path to the measured trace directory. Existence is not implied."""
    override = os.environ.get(TRACES_ENVIRONMENT_VARIABLE)
    return Path(override) if override else ROOT / MEASURED_TRACES


def measured_data_present() -> bool:
    """Whether any power CSV is on disk. Used only to word a refusal."""
    directory = measured_trace_directory()
    return directory.is_dir() and any(directory.glob("power_*.csv"))


def require_measured_campaign(cells: tuple[str, ...] = REQUIRED_CELLS):
    """Load the measured campaign, or raise naming what is absent.

    There is no fallback. A script that cannot find a COMPLETE measured campaign
    fails, loudly, with the names of the files it looked for. Falling back to the
    synthetic generator here would produce a figure that looks like a
    measurement and is not one, which is the single failure mode this product
    exists to make impossible.

    Raises
    ------
    powerenvelope.trace.TraceDataAbsentError
        With the directory and the required cell labels in the message, with the
        missing completion marker named, or with a refusal to treat a
        self-declared synthetic directory as measured.
    """
    from powerenvelope.provenance import Origin
    from powerenvelope.trace import (
        TraceDataAbsentError,
        campaign_is_complete,
        load_campaign,
    )

    directory = measured_trace_directory()
    label = measured_trace_label()
    if not measured_data_present():
        raise TraceDataAbsentError(
            "MEASURED TRACES ABSENT. Expected a campaign directory at "
            f"{label}/ holding manifest.json, campaign.log, CAMPAIGN_COMPLETE and, "
            "per cell, power_<cell>.csv, monitor_<cell>.csv and (for every cell that "
            "ran the pipeline) decisions_<cell>.csv. Cells required by name: "
            + ", ".join(cells)
            + ". Nothing was substituted: this check computes MEASURED figures and "
            "there is no synthetic fallback. Re-run once the hardware campaign has "
            "written its traces into that directory."
        )
    if not campaign_is_complete(directory):
        present = sorted(path.name for path in directory.glob("power_*.csv"))
        raise TraceDataAbsentError(
            f"MEASURED CAMPAIGN INCOMPLETE. {label}/CAMPAIGN_COMPLETE is absent, and "
            "the campaign writes that marker last of all. Power files present: "
            f"{present}. No final number is written from a partial campaign; re-run "
            "this script once the marker exists."
        )
    campaign = load_campaign(
        directory, origin=Origin.MEASURED, require=cells, directory_label=label
    )
    if campaign.manifest.get("synthetic"):
        raise TraceDataAbsentError(
            f"{label}/manifest.json declares itself SYNTHETIC. This check reports "
            "MEASURED figures and will not relabel generated data as measurement. "
            "Point it at the campaign's own output directory."
        )
    if campaign.manifest.get("aborted"):
        raise TraceDataAbsentError(
            f"{label}/manifest.json has aborted=true, at cell "
            f"{campaign.manifest.get('abort_cell')!r}. A campaign that aborted is not "
            "a campaign that measured, and no figure is reported from one."
        )
    return campaign


def synthetic_campaign(directory: Path | None = None, **kwargs):
    """Write and load a SYNTHETIC campaign into a temporary directory.

    Every number derived from this is SYNTHETIC and every script that uses it
    says so in its header. The directory is outside the repository, because a
    fixture written into the tree would be a build artifact.
    """
    import tempfile

    from fixtures import synthetic_campaign as fixture

    target = directory or Path(tempfile.mkdtemp(prefix="powerenvelope-synthetic-"))
    kwargs.setdefault("seed", SEED_FIXTURE)
    return fixture.campaign(target, **kwargs)
