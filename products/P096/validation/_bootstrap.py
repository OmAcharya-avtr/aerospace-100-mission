"""Path bootstrap, output tee and declared constants for every script here.

Five things in this module exist because of defects this mission has shipped,
and the arrangement is copied from ``products/P092`` rather than invented.

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

**The corpus is required, by name, and is never substituted.**
:func:`require_corpus` loads ``data/corpus/handovers.csv``, checks its sha256
against the manifest, and raises naming both paths and the generator if either
is absent or the hash disagrees. There is no synthetic fallback at validation
time: a fallback would produce a figure that looks like the committed corpus and
is not it. Every number in ``README.md``, ``validation/VALIDATION.md`` and
``MODEL_CARD.md`` comes from a script that called this function.

**Every number here is SYNTHETIC and is labelled so.** This product is
validation level 2: the evidence is agreement between a closed form, an
independently derived bound and a simulated trace. Nothing in this repository
was measured on hardware or on a real optical link, and each script's header
says so in its own output file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Where the committed corpus lives. Relative, always.
CORPUS_CSV = "data/corpus/handovers.csv"

#: Its manifest, carrying the sha256 the loader checks.
CORPUS_MANIFEST = "data/corpus/manifest.json"

#: The generator that writes both, named in every refusal.
CORPUS_GENERATOR = "data/generate_corpus.py"

#: Declared seeds. One place, so no two reported numbers rest on different ones.
SEED_ANALYSIS = 96_096
SEED_CONTROL = 5_150
SEED_TRACE = 96_096_2026

#: Bootstrap resamples for the reported intervals. ``HANDOVERLOSS_FAST`` lowers
#: it and is used only by ``probe_working_directories.py``, whose question is
#: whether a script runs from a given directory and not what the numbers are.
FAST = os.environ.get("HANDOVERLOSS_FAST") == "1"
RESAMPLES = 200 if FAST else 4000

#: Events attributed by the scripts that run the trace simulator. Reduced under
#: FAST for the same reason.
TRACE_EVENTS = 96 if FAST else 512

#: Confidence level of every reported interval.
CONFIDENCE = 0.95

#: The accounting model, stated once here so that every script's output file
#: carries it verbatim. It is the same text as in README.md and VALIDATION.md.
ACCOUNTING_MODEL = (
    "ACCOUNTING MODEL. A handover is a four-phase timeline from the break of the "
    "outgoing optical link: ACQUISITION [0, T_acq) with zero egress; RESTART "
    "[T_acq, T_o) with zero egress, T_o = T_acq + T_rst; then DRAIN [T_o, H), in "
    "which the egress ramps linearly 0 -> R_d over T_ramp and holds at R_d. A "
    "source of mean rate R_s feeds a tail-drop buffer of capacity B holding q0 at "
    "the break. Every dropped bit is attributed to the phase of the slot in which "
    "it was dropped, so the three mechanisms partition the window exactly. The "
    "TOTAL is computed a second way, from mass conservation alone (produced - "
    "delivered - (q_end - q0)), and the difference between the two is the "
    "DECOMPOSITION RESIDUAL, which is a reported number and not an assumption."
)

#: Provenance line. Every output file carries it.
ORIGIN = (
    "INPUT ORIGIN: SYNTHETIC. No measured handover, no measured acquisition time "
    "and no measured link appears anywhere in this repository. Validation level 2: "
    "the evidence is agreement between a closed form, an independently derived "
    "bound and a simulated trace."
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
    ``<script stem>_output.txt`` so that every number in ``README.md``,
    ``validation/VALIDATION.md`` and ``MODEL_CARD.md`` traces to a file produced
    by a script that was actually run.
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
        """Record a check that CANNOT run here. It does not pass.

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
    """Versions, core count and platform, recorded in the output of every script."""
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
        f"machine          : {platform.machine()} {platform.system()}, "
        f"{len(os.sched_getaffinity(0))} cores available to this process",
        f"repository root  : {ROOT.name} (printed relatively; see validation/_bootstrap.py)",
        f"resamples        : {RESAMPLES}"
        + ("  [HANDOVERLOSS_FAST=1: reduced, path probe only]" if FAST else ""),
        f"trace events     : {TRACE_EVENTS}"
        + ("  [HANDOVERLOSS_FAST=1: reduced, path probe only]" if FAST else ""),
    ]


def subprocess_environment() -> dict[str, str]:
    """Environment for a spawned subprocess that cannot pollute this repository."""
    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="handoverloss-validation-"))
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


def require_corpus():
    """Load the committed corpus as the ONLY input, or raise naming what is absent.

    There is no fallback. A script that cannot find the committed corpus with a
    matching hash fails, loudly, with the names of the files it looked for and
    the generator that writes them. Falling back to a fresh generator run here
    would produce a figure that looks like the committed corpus and is not it.

    Raises
    ------
    handoverloss.corpus.CorpusAbsentError
        With both paths and the generator named.
    ValueError
        On a hash mismatch, with both digests printed.
    """
    from handoverloss.corpus import CorpusAbsentError, load_corpus

    if not (ROOT / CORPUS_CSV).is_file() or not (ROOT / CORPUS_MANIFEST).is_file():
        raise CorpusAbsentError(
            f"CORPUS ABSENT. Expected {CORPUS_CSV} and {CORPUS_MANIFEST}. "
            f"Regenerate both, deterministically, with `python3 {CORPUS_GENERATOR}` "
            "from the repository root. Nothing was substituted."
        )
    return load_corpus(CORPUS_CSV, CORPUS_MANIFEST, verify_hash=True)


def corpus_event_parameters(corpus, count: int):
    """``EventParameters`` for the first ``count`` corpus events, realised durations.

    Used by every script that re-runs the trace simulator over corpus events. The
    REALISED durations are used, because this is baseline 3 reproducing the
    ground truth rather than any model predicting it.
    """
    import numpy

    from handoverloss.trace import EventParameters

    index = numpy.arange(min(count, len(corpus)))
    table = corpus.table
    return EventParameters(
        source_rate_bps=table["source_rate_bps"][index],
        downlink_rate_bps=table["downlink_rate_bps"][index],
        buffer_capacity_bits=table["buffer_capacity_bits"][index],
        initial_occupancy_bits=table["initial_occupancy_bits"][index],
        acquisition_s=table["realised_acquisition_s"][index],
        restart_s=table["realised_restart_s"][index],
        ramp_s=table["declared_ramp_s"][index],
        horizon_s=float(table["horizon_s"][0]),
    )
