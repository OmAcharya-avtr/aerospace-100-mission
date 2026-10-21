"""Path bootstrap and output tee shared by every validation script.

Two things in here exist because of defects this mission has shipped, and both
are worth stating rather than leaving as idiom.

**The working directory.** Every script in this directory calls
:func:`bootstrap` as its first statement. It puts ``<repo>/src`` on
``sys.path`` and then changes the process working directory to the repository
root, derived from ``__file__`` and never from the caller. Four products in
Batch 07 passed every check in their own build and then failed the release
gate, which runs each validation script with its working directory set to
``validation/`` rather than to the product root. One ``os.chdir`` fixes every
script in the directory and is a no-op when the script is already run from the
product root. Each script in this directory is executed from ``/``, ``/tmp``,
the product root and ``validation/`` before hand-off, and all four must exit
zero.

**Absolute paths.** No committed file in this repository may contain an
absolute path: it is not reproducible for a reader with a cold clone, and it
publishes the build machine's filesystem layout. Everything printed by these
scripts goes through :func:`relative`, which renders a path relative to the
repository root, and the repository root itself is printed as its own directory
name and nothing above it.
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
    ``<script stem>_output.txt`` so that every number in README.md,
    validation/VALIDATION.md and MODEL_CARD.md traces to a file produced by a
    script that was actually run.
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
        self.path.write_text("\n".join(self.lines) + "\n")
        return relative(self.path)


class Timer:
    """Accumulating wall-clock timer whose result is only reported in a table.

    Entering and leaving the context *adds* to :attr:`elapsed_s`, so one timer
    reused inside a loop accumulates the whole loop. A non-accumulating version
    of this class silently reported the duration of the last iteration only.

    Timings in this repository appear in tables explicitly declared unstable and
    never in a prose conclusion. A Batch 07 sentence reading "a factor of 40"
    became "a factor of 38" on re-run for exactly this reason.
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

    Used for tooling the library does not import, so that the committed output
    records the versions every number was measured against without this module
    importing pytest or ruff.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def environment_lines() -> list[str]:
    """Versions, solver availability and core count, recorded in every output."""
    import matplotlib
    import numpy
    import pulp
    import scipy
    import sklearn
    import yaml

    try:
        solvers = pulp.listSolvers(onlyAvailable=True)
    except Exception:  # pragma: no cover - PuLP probes the filesystem
        solvers = []

    return [
        f"python           : {sys.version.split()[0]}",
        f"numpy            : {numpy.__version__}",
        f"scipy            : {scipy.__version__}",
        f"scikit-learn     : {sklearn.__version__}",
        f"matplotlib       : {matplotlib.__version__}",
        f"pulp             : {pulp.__version__}",
        f"pulp solvers     : {solvers or 'none available'}",
        f"pyyaml           : {yaml.__version__}",
        f"pandas           : {_distribution_version('pandas')}",
        f"joblib           : {_distribution_version('joblib')}",
        f"pytest           : {_distribution_version('pytest')}",
        f"hypothesis       : {_distribution_version('hypothesis')}",
        f"ruff             : {_distribution_version('ruff')}",
        f"astropy          : {_distribution_version('astropy')}",
        f"cpu cores        : {len(os.sched_getaffinity(0))}",
        f"repository root  : {ROOT.name} (printed relatively; see validation/_bootstrap.py)",
    ]
