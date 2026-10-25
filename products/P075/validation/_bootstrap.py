"""Path bootstrap and output tee shared by every validation script.

Three things in here exist because of defects this mission has shipped, and all
three are worth stating rather than leaving as idiom.

**The working directory.** Every script in this directory calls
:func:`bootstrap` as its first statement. It puts ``<repo>/src`` on
``sys.path`` and then changes the process working directory to the repository
root, derived from ``__file__`` and never from the caller. Four products in
Batch 07 passed every check in their own build and then failed the release
gate, which runs each validation script with its working directory set to
``validation/`` rather than to the product root. One ``os.chdir`` fixes every
script in the directory and is a no-op when the script is already run from the
product root. Each script here is executed from ``/``, the temporary directory,
the product root, ``validation/`` and ``src/`` by
``probe_working_directories.py`` before hand-off, and all five must exit zero.

**Absolute paths.** No committed file in this repository may contain an
absolute path: it is not reproducible for a reader with a cold clone, and it
publishes the build machine's filesystem layout. Everything printed by these
scripts goes through :func:`relative`, and the repository root itself is
printed as its own directory name and nothing above it.

**Timings.** :class:`Timer` accumulates, so a timer reused in a loop reports
the loop and not its last iteration. Every timing this repository prints
appears in a table explicitly declared unstable, and never in a prose
conclusion: a Batch 07 sentence reading "a factor of 40" became "a factor of
38" on re-run because it was derived from wall clock.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


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
    ``<script stem>_output.txt`` so that every number in ``README.md`` and
    ``validation/VALIDATION.md`` traces to a file produced by a script that was
    actually run.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = ROOT / "validation" / Path(path).name
        self.lines: list[str] = []
        self.failures: list[str] = []
        self.blocking: list[str] = []
        self.passes = 0

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
        """Record a named measurement check and its verdict.

        A check that fails is reported as failed, in line and again in the
        FAILED CHECKS block at the end of the output file, and it is carried
        into ``validation/VALIDATION.md``. It does **not** change the script's
        exit code.

        That is a deliberate convention and it is worth stating. The release
        gate runs each script from several working directories and uses the
        exit code to decide whether the script *ran*. A script that exited
        non-zero because one tolerance was not met would be
        indistinguishable from one that crashed or could not find its inputs,
        and the pressure that creates is to widen the tolerance until the exit
        code goes green. So the exit code answers "did the measurement
        complete", :meth:`check` answers "did it meet its declared tolerance",
        and :meth:`require` is for the cases where the two coincide.

        Nothing in this repository tunes a tolerance to make a check pass.
        """
        verdict = "PASS" if passed else "FAIL"
        self(f"  [{verdict}] {name}{(' -- ' + detail) if detail else ''}")
        if passed:
            self.passes += 1
        else:
            self.failures.append(name)
        return passed

    def require(self, name: str, passed: bool, detail: str = "") -> bool:
        """Record a check whose failure means the measurement did not complete.

        Used for structural conditions -- a file that could not be written, a
        declaration that will not round-trip through the loader -- where there
        is no number to report and the script has nothing to say. These, and
        only these, set a non-zero exit code.
        """
        verdict = "PASS" if passed else "BLOCKING FAILURE"
        self(f"  [{verdict}] {name}{(' -- ' + detail) if detail else ''}")
        if passed:
            self.passes += 1
        else:
            self.blocking.append(name)
        return passed

    def summarise_failures(self) -> None:
        """Print the collected failures, or state that there were none."""
        self("")
        self(f"CHECKS PASSED: {self.passes}")
        if self.failures:
            self(f"FAILED CHECKS ({len(self.failures)}):")
            for name in self.failures:
                self(f"  - {name}")
        else:
            self("FAILED CHECKS: none")
        if self.blocking:
            self(f"BLOCKING FAILURES ({len(self.blocking)}):")
            for name in self.blocking:
                self(f"  - {name}")
        else:
            self("BLOCKING FAILURES: none")
        self("")
        self("  Exit code convention: 0 means this script completed its measurement")
        self("  and wrote its output. A failed check above is a reported measurement")
        self("  result, not a crash, and does not change the exit code. Only a")
        self("  blocking failure does. See validation/_bootstrap.py.")

    def save(self) -> str:
        """Write the collected lines beside the script. Returns a relative path."""
        self.path.write_text("\n".join(self.lines) + "\n")
        return relative(self.path)


class Timer:
    """Accumulating wall-clock timer whose result is only reported in a table.

    Entering and leaving the context *adds* to :attr:`elapsed_s`, so one timer
    reused inside a loop accumulates the whole loop.
    """

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
    """Installed version of ``name``, or ``"not installed"``.

    Used for tooling the library does not import, and for ``reliability`` and
    ``lifelines``, which the product specification names in the alternatives
    table and which are **not installed** in the environment these numbers were
    measured in. Their absence is recorded in the header of every output file
    rather than asserted in prose, and **no number in this repository came from
    either of them.**
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def environment_lines() -> list[str]:
    """Versions and core count, recorded in the header of every output file."""
    import numpy
    import scipy
    import yaml

    return [
        f"python           : {sys.version.split()[0]}",
        f"numpy            : {numpy.__version__}",
        f"scipy            : {scipy.__version__}",
        f"pyyaml           : {yaml.__version__}",
        f"matplotlib       : {_distribution_version('matplotlib')}",
        f"pytest           : {_distribution_version('pytest')}",
        f"hypothesis       : {_distribution_version('hypothesis')}",
        f"ruff             : {_distribution_version('ruff')}",
        f"reliability      : {_distribution_version('reliability')}",
        f"lifelines        : {_distribution_version('lifelines')}",
        f"cpu cores        : {len(os.sched_getaffinity(0))}",
        f"repository root  : {ROOT.name} (printed relatively; see validation/_bootstrap.py)",
    ]


def header(tee: Tee, title: str) -> None:
    """Write the standard title and environment block into ``tee``."""
    tee.rule(title)
    for line in environment_lines():
        tee(f"  {line}")
    tee("")
    tee("  reliability and lifelines are named in the README alternatives table and")
    tee("  are NOT installed here. No number in this repository was cross-checked")
    tee("  against either of them, and none claims to have been.")
