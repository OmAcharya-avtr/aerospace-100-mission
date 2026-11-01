"""Path bootstrap, output tee and declared constants, used by every script here.

Three things in this module exist because of defects this mission has shipped,
and all three are worth stating rather than leaving as idiom.

**The working directory.** Every script in this directory calls
:func:`bootstrap` as its first statement. It puts ``<repo>/src`` on
``sys.path`` and then changes the process working directory to the repository
root, derived from ``__file__`` and never from the caller. Three products in
one batch of this mission passed every check in their own build and then failed
the release gate, which runs each validation script with its working directory
set to the script's own directory rather than to the product root. One
``os.chdir`` fixes every script in the directory and is a no-op when the script
is already run from the product root. Each script here is executed from the
filesystem root, the system temporary directory, the product root and
``validation/`` -- which is the gate's working directory -- by
``probe_working_directories.py`` before hand-off, and all four must exit zero.

**Absolute paths.** No committed file in this repository may contain an
absolute path: it is not reproducible for a reader with a cold clone, and it
publishes the build machine's filesystem layout. Everything printed by these
scripts goes through :func:`relative`, and the repository root itself is
printed as its own directory name and nothing above it.

**Subprocess environments.** :func:`subprocess_environment` points ``HOME`` and
``MPLCONFIGDIR`` at a temporary directory outside the repository and sets
``PYTHONDONTWRITEBYTECODE``, because a validation script in an earlier product
set ``HOME`` to the product root and matplotlib wrote ``.config/matplotlib``
into the repository, which the release gate fails twice over: once as a tracked
build artifact and once as an absolute path inside one.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The declared confidence for every bound reported by these scripts.
CONFIDENCE = 0.95

#: The declared proportion for every tolerance and prediction limit here.
PROPORTION = 0.95

#: The field sample sizes this product is about: tens, as a flight-test
#: campaign actually delivers.
FIELD_SIZES: tuple[int, ...] = (15, 25, 40, 60)

#: Monte Carlo repetitions for every coverage measurement here. Sized so that
#: the slowest script finishes inside the three-minute budget on two cores; the
#: binomial standard error of a rate near 0.95 at this count is 0.0126.
REPETITIONS = 300

#: The held-out pool used to estimate a per-repetition coverage.
POOL_SIZE = 4000

#: Declared seeds. One place, so no two reported numbers rest on different ones.
SEED_TRUTH = 20260101
SEED_FIELD = 770000
SEED_POOL = 880000
SEED_MODEL = 990000


def bootstrap() -> Path:
    """Put ``src`` on ``sys.path``, change to the repository root, return it."""
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.chdir(ROOT)
    return ROOT


def relative(path: str | Path) -> str:
    """Render ``path`` relative to the repository root, never absolutely.

    A path outside the repository is reduced to its final component, which is
    the only part reproducible for a reader with a cold clone.
    """
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except ValueError:
        return candidate.name


class Tee:
    """Collect report lines, print them, and write them to a file at the end.

    Every script writes its raw output beside itself as
    ``<script stem>_output.txt`` so that every number in ``README.md``,
    ``MODEL_CARD.md`` and ``validation/VALIDATION.md`` traces to a file
    produced by a script that was actually run.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = ROOT / "validation" / Path(path).name
        self.lines: list[str] = []
        self.failures: list[str] = []
        self.checks = 0
        self.controls_fired = 0
        self.controls_expected = 0

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

    def control(self, name: str, fired: bool, must_fire: bool, detail: str = "") -> bool:
        """Record a negative control and whether it FIRED as declared.

        A control that does not fire is a failure of the control, not a success
        of the product: it means the check it guards would not have caught the
        violation it was built to catch. Both directions are asserted, because
        a control that fires when it must not is equally wrong.
        """
        self.controls_expected += 1
        passed = fired == must_fire
        if fired:
            self.controls_fired += 1
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
                f"{self.controls_fired} fired"
            )
        if self.failures:
            self(f"FAILED CHECKS ({len(self.failures)}): {', '.join(self.failures)}")
        else:
            self("FAILED CHECKS (0): none")
        self.path.write_text("\n".join(self.lines) + "\n")
        return relative(self.path)


class Timer:
    """Accumulating wall-clock timer whose result is only ever reported in a table."""

    def __init__(self) -> None:
        self.elapsed_s = 0.0
        self.calls = 0
        self._start = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exception: object) -> None:
        self.elapsed_s += time.perf_counter() - self._start
        self.calls += 1

    @property
    def mean_ms(self) -> float:
        """Mean duration per context entry, in milliseconds."""
        return 1000.0 * self.elapsed_s / self.calls if self.calls else 0.0


def _distribution_version(name: str) -> str:
    """Installed version of ``name``, or ``"not installed"``."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def environment_lines() -> list[str]:
    """Versions and core count, recorded in the output of every script.

    ``mapie``, ``crepes`` and ``torch`` are listed deliberately. The first two
    are named in this product's README as the mature alternatives for
    conformal prediction, the third is named as unavailable; none of them is
    installed in the measured environment, nothing was installed to get them,
    and no number in this repository is compared against any of them. Each
    output file states their absence in its own header rather than leaving the
    reader to infer it.
    """
    import numpy
    import scipy
    import sklearn

    return [
        f"python           : {sys.version.split()[0]}",
        f"numpy            : {numpy.__version__}",
        f"scipy            : {scipy.__version__}",
        f"scikit-learn     : {sklearn.__version__}",
        f"matplotlib       : {_distribution_version('matplotlib')}",
        f"pytest           : {_distribution_version('pytest')}",
        f"hypothesis       : {_distribution_version('hypothesis')}",
        f"ruff             : {_distribution_version('ruff')}",
        f"mapie            : {_distribution_version('mapie')}  (named in README; no "
        "number here is compared against it)",
        f"crepes           : {_distribution_version('crepes')}  (named in README; no "
        "number here is compared against it)",
        f"torch            : {_distribution_version('torch')}  (named in README as "
        "unavailable here)",
        f"cpu cores        : {len(os.sched_getaffinity(0))}",
        f"repository root  : {ROOT.name} (printed relatively; see validation/_bootstrap.py)",
    ]


def subprocess_environment() -> dict[str, str]:
    """Environment for a spawned subprocess that cannot pollute this repository."""
    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="transfergap-validation-"))
    environment = dict(os.environ)
    environment.update(
        {
            "HOME": str(scratch),
            "MPLCONFIGDIR": str(scratch / "mpl"),
            "MPLBACKEND": "Agg",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT / "src"),
            # Hypothesis writes its example database into `.hypothesis` in the
            # CURRENT WORKING DIRECTORY, which for a subprocess started here is
            # the repository root. Pointing it at the scratch directory keeps
            # this product's own scripts from creating a cache directory inside
            # the tree they then check for cache directories.
            "HYPOTHESIS_STORAGE_DIRECTORY": str(scratch / "hypothesis"),
        }
    )
    return environment
