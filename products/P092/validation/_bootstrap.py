"""Path bootstrap, output tee and declared constants for every script here.

Five things in this module exist because of defects this mission has shipped.

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

**Measured data is required, by name, and is never substituted.** The scripts
split into two kinds and the split is the point of this product.
:func:`require_measured_campaign` loads ``data/traces/`` and raises if it is
absent or incomplete, naming the runs it looked for. The scripts that call it
compute MEASURED figures and they fail loudly rather than falling back to the
synthetic generator. The scripts that call :func:`synthetic_campaign` compute
SYNTHETIC figures, label every number so, and are checks of the code rather
than measurements of anything.

**The completion marker is part of the requirement.** The campaign writes
``CAMPAIGN_COMPLETE`` last of all. A measured script run against a directory
without it would publish a figure from a partial campaign, so
:func:`require_measured_campaign` refuses unless the marker exists, and says so.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Where the campaign's MEASURED traces live. Relative, always.
MEASURED_TRACES = "data/traces"

#: The runs every measured check requires by name. A campaign missing any of
#: these cannot produce the product's headline or its held-out-mix split, and
#: the loader says which are absent.
REQUIRED_RUNS: tuple[str, ...] = (
    "gt_long",
    "mix_c0",
    "mix_c1",
    "mix_c2",
    "mix_c3",
    "mix_c4",
    "mix_c5",
    "mix_c6",
    "mix_mem2",
    "mix_c2mem1",
)

#: The ground-truth run. The headline is measured on this one and no other.
GROUND_TRUTH_RUN = "gt_long"

#: The idle run the classical speed-scaling predictor takes its execution-time
#: distribution from: zero stressors, so the board is as idle as a shared board
#: gets. Foreign occupancy is still measured and reported.
IDLE_RUN = "mix_c0"

#: Declared seeds. One place, so no two reported numbers rest on different ones.
SEED_ANALYSIS = 920092
SEED_FIXTURE = 920092
SEED_CONTROL = 5150

#: Bootstrap resamples for the reported intervals. ``DEADLINERISK_FAST`` lowers
#: it and is used only by ``probe_working_directories.py``, whose question is
#: whether a script runs from a given directory and not what the numbers are.
FAST = os.environ.get("DEADLINERISK_FAST") == "1"
RESAMPLES = 200 if FAST else 4000

#: Confidence level of every reported interval.
CONFIDENCE = 0.95

#: The measurement model, stated once here so that every script's output file
#: carries it verbatim. It is the same text as in README.md, VALIDATION.md and
#: the loader's docstring.
MEASUREMENT_MODEL = (
    "MEASUREMENT MODEL. Period T, implicit deadline D = T. Job i is released at "
    "t0 + i*T and its response time is end - release. A job whose response "
    "exceeds D is a MISS. If a job finishes after later releases have already "
    "passed, those releases never execute: they are DROPPED, and each dropped "
    "release is also a miss. Every executed row carries dropped_before, so "
    "releases = sum(1 + dropped_before) and misses = sum(missed + "
    "dropped_before) are both reconstructible from the committed traces."
)


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


class Tee:
    """Collect report lines, print them, and write them to a file at the end.

    Every script writes its raw output beside itself as
    ``<script stem>_output.txt`` so that every number in ``README.md`` and
    ``validation/VALIDATION.md`` traces to a file produced by a script that was
    actually run.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = ROOT / "validation" / Path(path).name
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

    def unmeasurable_check(self, name: str, reason: str) -> None:
        """Record a check that CANNOT run in this campaign. It does not pass.

        Counted separately from both passes and failures, printed with its
        reason, and summarised at the end of the file. A check reported this way
        is neither evidence for nor evidence against anything.
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
                f"STRUCTURALLY UNMEASURABLE ({len(self.unmeasurable)}): "
                + ", ".join(self.unmeasurable)
            )
        if self.failures:
            self(f"FAILED CHECKS ({len(self.failures)}): {', '.join(self.failures)}")
        else:
            self("FAILED CHECKS (0): none")
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
        f"pandas           : {_distribution_version('pandas')}",
        f"joblib           : {_distribution_version('joblib')}",
        f"ruff             : {_distribution_version('ruff')}",
        f"analysis machine : {platform.machine()} {platform.system()}, "
        f"{len(os.sched_getaffinity(0))} cores  (the TRACES were measured on aarch64 ON "
        "THE BOARD; the arithmetic over them runs here)",
        f"repository root  : {ROOT.name} (printed relatively; see validation/_bootstrap.py)",
        f"resamples        : {RESAMPLES}"
        + ("  [DEADLINERISK_FAST=1: reduced, path probe only]" if FAST else ""),
    ]


def subprocess_environment() -> dict[str, str]:
    """Environment for a spawned subprocess that cannot pollute this repository."""
    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="deadlinerisk-validation-"))
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


def measured_trace_directory() -> Path:
    """Path to the measured trace directory. Existence is not implied."""
    return ROOT / MEASURED_TRACES


def measured_data_present() -> bool:
    """Whether any measured CSV is on disk. Used only to word a refusal."""
    directory = measured_trace_directory()
    return directory.is_dir() and any(directory.glob("jobs_*.csv"))


def require_measured_campaign(extra_runs: tuple[str, ...] = ()):
    """Load ``data/traces/`` as MEASURED, or raise naming what is absent.

    There is no fallback. A script that cannot find a COMPLETE measured campaign
    fails, loudly, with the names of the files it looked for. Falling back to
    the synthetic generator here would produce a figure that looks like a
    measurement and is not one, which is the single failure mode this product
    exists to make impossible.

    Raises
    ------
    deadlinerisk.trace.TraceDataAbsentError
        With the directory and the required run labels in the message, or with
        the missing completion marker named.
    """
    from deadlinerisk.provenance import Origin
    from deadlinerisk.trace import (
        TraceDataAbsentError,
        campaign_is_complete,
        load_campaign,
    )

    directory = measured_trace_directory()
    required = (*REQUIRED_RUNS, *extra_runs)
    if not measured_data_present():
        raise TraceDataAbsentError(
            "MEASURED TRACES ABSENT. Expected CSV traces and manifest.json under "
            f"{MEASURED_TRACES}/, by name: "
            + ", ".join(f"jobs_{label}.csv" for label in required)
            + ", manifest.json, CAMPAIGN_COMPLETE. Nothing was substituted: this "
            "check computes MEASURED figures and there is no synthetic fallback. "
            "Re-run once the hardware campaign has written its traces into that "
            "directory."
        )
    if not campaign_is_complete(directory):
        present = sorted(
            path.name for path in measured_trace_directory().glob("jobs_*.csv")
        )
        raise TraceDataAbsentError(
            f"MEASURED CAMPAIGN INCOMPLETE. {MEASURED_TRACES}/CAMPAIGN_COMPLETE is "
            "absent, and the campaign writes that zero-byte marker last of all. "
            f"Job files present: {present}. No final number is written from a partial "
            "campaign; re-run this script once the marker exists."
        )
    campaign = load_campaign(
        directory,
        origin=Origin.MEASURED,
        require=required,
        directory_label=MEASURED_TRACES,
    )
    if not campaign.complete:
        raise TraceDataAbsentError(
            f"{MEASURED_TRACES}/manifest.json has complete=false while "
            "CAMPAIGN_COMPLETE exists. The two disagree, so nothing is reported."
        )
    return campaign


def synthetic_campaign(directory: Path | None = None, **kwargs):
    """Write and load a SYNTHETIC campaign into a temporary directory.

    Every number derived from this is SYNTHETIC and every script that uses it
    says so in its header. The directory is outside the repository, because a
    fixture written into the tree would be a build artifact.
    """
    import tempfile

    from fixtures import synthetic_trace

    target = directory or Path(tempfile.mkdtemp(prefix="deadlinerisk-synthetic-"))
    return synthetic_trace.load_synthetic_campaign(
        target, seed=kwargs.pop("seed", SEED_FIXTURE), **kwargs
    )
