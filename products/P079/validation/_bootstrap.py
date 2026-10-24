"""Path bootstrap, output tee and environment block shared by every validation script.

Three things in here exist because of defects this mission has shipped, and
each is worth stating rather than leaving as idiom.

**The working directory.** Every script in this directory calls
:func:`bootstrap` as its first statement. It puts ``<root>/src`` on
``sys.path`` and then changes the process working directory to the product
root, derived from ``__file__`` and never from the caller. Four products in
Batch 07 passed every check in their own build and then failed the release
gate, which runs each validation script with its working directory set to
``validation/`` rather than to the product root. One ``os.chdir`` fixes every
script in the directory and is a no-op when the script is already run from the
product root. Each script here is executed from ``/``, ``/tmp``, the product
root and ``validation/`` before hand-off and must exit zero from all four;
``validate_cwd_independence.py`` is the script that does that and records it.

**Absolute paths.** No committed file in this repository may contain an
absolute path: it is not reproducible for a reader with a cold clone, and it
publishes the build machine's filesystem layout. Everything printed by these
scripts goes through :func:`relative`, and the root itself prints as its own
directory name and nothing above it.

**Fast mode.** ``validate_cwd_independence.py`` re-runs every sibling script
from four working directories, and it only checks the exit status. Those runs
set ``TIMETRANSFER_FAST=1``, which makes :func:`fast_divisor` return a number
greater than one so the Monte Carlo loops shrink, and makes :meth:`Tee.save`
write **nothing** -- because a fast probe run must never overwrite a committed
output file with numbers taken at a reduced sample size.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FAST_ENV = "TIMETRANSFER_FAST"
"""Environment variable that puts the scripts into reduced-sample probe mode."""


def bootstrap() -> Path:
    """Put ``src`` on ``sys.path``, change to the product root, return it."""
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.chdir(ROOT)
    return ROOT


def is_fast() -> bool:
    """``True`` when the script is running as an exit-status probe."""
    return os.environ.get(FAST_ENV, "") not in ("", "0", "false", "False")


def fast_divisor() -> int:
    """Divide every Monte Carlo count by this. ``1`` normally, larger in probe mode."""
    return 40 if is_fast() else 1


def scaled(n: int, *, minimum: int = 8) -> int:
    """``n`` divided by :func:`fast_divisor`, never below ``minimum``."""
    return max(int(minimum), int(n) // fast_divisor())


def relative(path: str | Path) -> str:
    """Render ``path`` relative to the product root, never absolutely.

    A path outside the root is reduced to its final component, which is the
    only part reproducible for a reader with a cold clone.
    """
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except ValueError:
        return candidate.name


class Tee:
    """Collect report lines, print them, and write them beside the script.

    Every script writes its raw output to ``validation/<script stem>_output.txt``
    so that every number in README.md, validation/VALIDATION.md and
    MODEL_CARD.md traces to a file produced by a script that was actually run.
    In fast probe mode :meth:`save` writes nothing, so a probe cannot replace a
    committed output with reduced-sample numbers.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = ROOT / "validation" / Path(path).name
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        for line in str(text).splitlines() or [""]:
            self.lines.append(line)
            print(line)

    def rule(self, title: str) -> None:
        self("")
        self("=" * 78)
        self(title)
        self("=" * 78)

    def save(self) -> str:
        if is_fast():
            return "(fast probe: output not written)"
        self.path.write_text("\n".join(self.lines) + "\n")
        return relative(self.path)


class Timer:
    """Accumulating wall-clock timer whose result only ever appears in a table.

    Entering and leaving the context *adds* to :attr:`elapsed_s`, so one timer
    reused inside a loop accumulates the whole loop. Timings in this repository
    appear only in tables explicitly declared unstable and never in a prose
    conclusion: a Batch 07 sentence reading "a factor of 40" became "a factor
    of 38" on re-run for exactly this reason.
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

    Used for packages the library does not import, so the committed output
    records what every number was measured against without this module
    importing pytest, ruff, or a package that is deliberately absent.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def environment_lines() -> list[str]:
    """Versions and core count, recorded in every committed output."""
    import matplotlib
    import numpy
    import scipy
    import sklearn

    return [
        f"python           : {sys.version.split()[0]}",
        f"numpy            : {numpy.__version__}",
        f"scipy            : {scipy.__version__}",
        f"scikit-learn     : {sklearn.__version__}",
        f"matplotlib       : {matplotlib.__version__}",
        f"joblib           : {_distribution_version('joblib')}",
        f"pytest           : {_distribution_version('pytest')}",
        f"hypothesis       : {_distribution_version('hypothesis')}",
        f"ruff             : {_distribution_version('ruff')}",
        f"allantools       : {_distribution_version('allantools')}  "
        f"(NOT used; Allan deviation is implemented in timetransfer.allan)",
        f"ntplib           : {_distribution_version('ntplib')}  "
        f"(NOT used; no number here comes from it)",
        f"cpu cores        : {len(os.sched_getaffinity(0))}",
        f"product root     : {ROOT.name} (printed relatively; see validation/_bootstrap.py)",
        f"fast probe mode  : {is_fast()} (divisor {fast_divisor()})",
    ]
