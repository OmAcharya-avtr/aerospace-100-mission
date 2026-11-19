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

**Bytecode.** Every runnable script here sets ``sys.dont_write_bytecode = True``
as its first statement, before any other import. It cannot be delegated to this
module, because CPython writes a module's bytecode cache BEFORE executing that
module, so this module cannot set it on its own behalf. The subsequent
module-level imports then carry ``# noqa: E402``.

**Measured data is required by name and is never substituted.** Every script
here that reports a hardware figure calls
:func:`hiltimingdiff.traces.load_campaign`, which raises with the missing files
named rather than falling back to anything. There is no synthetic generator in
this product to fall back to, by design: the hardware half is the measurement
and the simulation half is measured live.

**A script here must never silently replace the committed evidence.** Every
script writes its ``<stem>_output.txt`` beside itself, and
``probe_working_directories.py`` executes every one of them four times. Until
2026-11-16 that meant a probe run left the committed outputs holding the
probe's own reduced ``HILTIMINGDIFF_FAST=1`` figures. :data:`OUTPUT_DIRECTORY_VARIABLE`
redirects :class:`Tee` to a temporary directory for exactly that case, so the
committed outputs come only from a full run.

**No git command is run by this build agent or by any script it executes.** The
coordinator forbids it for this build. ``validate_tree_hygiene.py`` therefore
checks ignore patterns with an explicit matcher over the two relevant
``.gitignore`` files and says so in its own output; ``git check-ignore -v`` is
the authority and belongs to the gate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Where the measured traces live. Relative, always.
MEASURED_TRACES = "data/traces"

#: Simulation-mode passes per cell for the scripts that measure it. Twelve
#: passes over nine cells at the board's 1500-frame stack is 162000 timed
#: frames and about 35 s on two cores, inside the mission's three-minute
#: compute budget.
SIMULATION_PASSES = 12

#: Simulation-mode passes for the fast path used by the working-directory
#: probe, whose question is whether a script runs from a given directory and
#: not what the numbers are.
FAST_SIMULATION_PASSES = 1

#: Bootstrap resamples for every published interval.
RESAMPLES = 4000

#: Confidence level of every published interval.
CONFIDENCE = 0.95

#: ``HILTIMINGDIFF_FAST=1`` lowers the pass count and the resample count. Used
#: only by ``probe_working_directories.py``.
FAST = os.environ.get("HILTIMINGDIFF_FAST") == "1"

#: Where :class:`Tee` writes, when it must not write beside the script.
#:
#: ``probe_working_directories.py`` executes every script here four times, and
#: every script WRITES its ``<stem>_output.txt``. Without this override the
#: probe silently replaced the committed evidence with its own reduced
#: ``HILTIMINGDIFF_FAST=1`` runs -- one simulation pass and 200 bootstrap
#: resamples instead of twelve and 4000 -- so the committed outputs stopped
#: being the runs that VALIDATION.md quotes. The probe now points this at a
#: temporary directory outside the repository, and the committed outputs are
#: written only by a full run of each script.
OUTPUT_DIRECTORY_VARIABLE = "HILTIMINGDIFF_OUTPUT_DIR"

#: The pass count and resample count actually used, after FAST.
PASSES = FAST_SIMULATION_PASSES if FAST else SIMULATION_PASSES
RESAMPLES_USED = 200 if FAST else RESAMPLES


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

    def __init__(self, path: str | Path, directory: str | Path | None = None) -> None:
        # ``directory`` exists so that a test can exercise a script's reporting
        # code without overwriting a committed output file. Scripts never pass
        # it; they write beside themselves, which is the one committed layout.
        #
        # OUTPUT_DIRECTORY_VARIABLE does the same for the working-directory
        # probe, which runs every script here four times and would otherwise
        # replace the committed evidence with its own reduced FAST-mode runs.
        if directory is not None:
            base = Path(directory)
        else:
            override = os.environ.get(OUTPUT_DIRECTORY_VARIABLE)
            base = Path(override) if override else ROOT / "validation"
        self.path = base / Path(path).name
        self.lines: list[str] = []
        self.failures: list[str] = []
        self.findings: list[str] = []
        self.falsifications: list[str] = []
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
        """A section banner."""
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

    def falsified(
        self, name: str, detail: str, pinned_ok: bool, pinned_detail: str = ""
    ) -> bool:
        """Record a check whose premise the measurement FALSIFIES.

        The falsification is the finding, not a defect in the run, so the check
        is not edited, not skipped and its tolerance is not loosened. What is
        asserted instead is that the falsification is EXACTLY the pinned one:
        ``pinned_ok`` must be true, and it is a real check that fails if the
        measured value ever moves. A script with a falsified check exits 0 while
        the falsification matches its pinned literal and non-zero the moment it
        does not.
        """
        self.checks += 1
        self.falsifications.append(name)
        self(f"[FALSIFIED] {name} -- {detail}")
        return self.check(
            f"the falsification of {name!r} is exactly the pinned one",
            pinned_ok,
            pinned_detail,
        )

    def finding(self, text: str) -> None:
        """Record a measured finding that is neither a pass nor a failure.

        Used for weak, null and refuting results, which are published rather
        than retuned. Each one is repeated in the script's own summary so that
        none can be lost between the body and the footer.
        """
        self.findings.append(text)
        self(f"[FINDING] {text}")

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
        if self.falsifications:
            self(
                f"FALSIFIED CHECKS ({len(self.falsifications)}): "
                + ", ".join(self.falsifications)
                + " -- each published as falsified, with its measured value pinned as a "
                "literal so this script fails if the falsification ever changes"
            )
        if self.findings:
            self(f"FINDINGS ({len(self.findings)}):")
            for item in self.findings:
                self(f"  - {item}")
        if self.failures:
            self(f"FAILED CHECKS ({len(self.failures)}): {', '.join(self.failures)}")
        else:
            self("FAILED CHECKS (0): none")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n")
        if self.failures:
            # The release gate reports a non-zero script as
            # "<script> exit <code>: <last line of STDERR>", and prints
            # "no output" when stderr is empty. Every line above went to
            # stdout, which the gate discards, so a failing check used to
            # reach the gate's log as "exit 1: no output" -- a true statement
            # about stderr that named neither the check nor the measurement.
            # That is what "validate_measured_baselines.py exit 1: no output"
            # was on 2026-11-16. The failing checks are therefore repeated on
            # stderr, last, so the gate's one-line message names them.
            print(
                f"FAILED CHECKS ({len(self.failures)}) in "
                f"{self.path.name}: {'; '.join(self.failures)}",
                file=sys.stderr,
            )
        return relative(self.path)


def _distribution_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def environment_lines() -> list[str]:
    """Versions, core count and platform, recorded in the output of every script.

    The ARCHITECTURE line matters here. The hardware traces were produced on
    aarch64 on the board; every piece of ARITHMETIC in this repository, and the
    whole of simulation mode, runs on the machine named below. Those are
    different machines and the distinction is printed rather than inferred.
    """
    import platform

    import numpy

    return [
        f"python           : {sys.version.split()[0]}",
        f"numpy            : {numpy.__version__}",
        f"pandas           : {_distribution_version('pandas')}",
        f"scipy            : {_distribution_version('scipy')}  (cross-check only)",
        f"pytest           : {_distribution_version('pytest')}",
        f"hypothesis       : {_distribution_version('hypothesis')}",
        f"matplotlib       : {_distribution_version('matplotlib')}",
        f"ruff             : {_distribution_version('ruff')}",
        f"analysis machine : {platform.machine()} {platform.system()}, "
        f"{len(os.sched_getaffinity(0))} cores  (the HARDWARE traces are measured on "
        "aarch64 ON THE BOARD; simulation mode and all arithmetic run here)",
        f"repository root  : {ROOT.name} (printed relatively; see validation/_bootstrap.py)",
        f"simulation passes: {PASSES}" + ("  [HILTIMINGDIFF_FAST=1]" if FAST else ""),
        f"bootstrap resamples: {RESAMPLES_USED}"
        + ("  [HILTIMINGDIFF_FAST=1]" if FAST else ""),
    ]


def subprocess_environment() -> dict[str, str]:
    """Environment for a spawned subprocess that cannot pollute this repository."""
    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="hiltimingdiff-validation-"))
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


def measurement_model() -> str:
    """The measurement model, from the one place it is written down.

    Imported lazily: this module is imported before :func:`bootstrap` has put
    ``src`` on ``sys.path``.
    """
    from hiltimingdiff.report import MEASUREMENT_MODEL

    return MEASUREMENT_MODEL


def header(tee: Tee, title: str) -> None:
    """Write the standard header: title, environment, measurement model."""
    tee(title)
    tee("=" * len(title))
    tee("")
    for line in environment_lines():
        tee(f"  {line}")
    tee("")
    tee("MEASUREMENT MODEL")
    tee("-" * 17)
    for line in measurement_model().strip().splitlines():
        tee(f"  {line}")
