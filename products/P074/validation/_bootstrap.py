"""Path bootstrap, output tee and environment block shared by every validation script.

Four things in here exist because of defects this mission has actually
shipped, and each is worth stating rather than leaving as idiom.

**The working directory.** Every script in this directory calls
:func:`bootstrap` as its first statement. It puts ``<root>/src`` on
``sys.path`` and then changes the process working directory to the product
root, derived from ``__file__`` and never from the caller. Four products in
one batch of this mission passed every check in their own build and then
failed the release gate, which runs each validation script with its working
directory set to ``validation/`` rather than to the product root. One
``os.chdir`` fixes every script in the directory and is a no-op when the
script is already run from the product root. Every script here is executed
from ``/``, ``/tmp``, the product root and ``validation/`` before hand-off
and must exit zero from all four; ``validate_cwd_independence.py`` is the
script that does that and records it.

**Absolute paths.** No committed file in this repository may contain an
absolute path: it is not reproducible for a reader with a cold clone, and it
publishes the build machine's filesystem layout. Everything printed by these
scripts goes through :func:`relative`, and the root itself prints as its own
directory name and nothing above it.

**Fast mode.** ``validate_cwd_independence.py`` re-runs every sibling script
from four working directories and only checks the exit status. Those runs set
``ORBITDRIFT_FAST=1``, which makes :func:`is_fast` true so the expensive
studies shrink their grids, and makes :meth:`Tee.save` write **nothing** --
because a fast probe must never overwrite a committed output file with
numbers taken at a reduced sample size or a coarser integration step.

**The committed dataset.** The divergence dataset takes about three minutes of
two-core time to regenerate, which is more than the per-script budget allows
for every script that needs it. It is therefore generated once by
``regenerate_dataset.py`` into ``fixtures/divergence_dataset.npz``, committed,
and loaded by :func:`load_dataset` everywhere else. A script that silently
regenerated it instead would be reporting numbers from a different population
than the one in the repository.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FAST_ENV = "ORBITDRIFT_FAST"
"""Environment variable that puts the scripts into reduced-grid probe mode."""

DATASET_PATH = "fixtures/divergence_dataset.npz"
POPULATION_PATH = "fixtures/population.tle"


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


def scaled(n: int, *, minimum: int = 4, divisor: int = 8) -> int:
    """``n`` divided by ``divisor`` in probe mode, never below ``minimum``."""
    return max(int(minimum), int(n) // divisor) if is_fast() else int(n)


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


FAST_OBJECT_LIMIT = 24
"""Objects kept from the committed table in probe mode.

Every script that fits a predictor costs time proportional to the number of
rows, and ``validate_cwd_independence.py`` runs all of them four times. Keeping
a fifth of the objects brings the whole matrix inside the per-script budget. A
probe run writes no output file, so no committed number is ever taken from the
reduced table.
"""


def load_dataset():
    """Load the committed divergence table, with an actionable error if it is missing.

    In probe mode the table is restricted to the first
    :data:`FAST_OBJECT_LIMIT` objects. :meth:`Tee.save` writes nothing in that
    mode, so a reduced table can never reach a committed output.
    """
    import numpy as np

    from orbitdrift import DivergenceTable

    target = ROOT / DATASET_PATH
    if not target.exists():
        raise SystemExit(
            f"{DATASET_PATH} is missing from this clone. Regenerate it with\n"
            "    python validation/regenerate_dataset.py\n"
            "which takes about three minutes on two cores."
        )
    table = DivergenceTable.load(target)
    if not is_fast():
        return table
    keep = np.flatnonzero(table.column("object_index") < FAST_OBJECT_LIMIT)
    return DivergenceTable(
        values=table.values[keep],
        columns=table.columns,
        tle_text=table.tle_text,
        detail=table.detail + f" [PROBE SUBSET: first {FAST_OBJECT_LIMIT} objects]",
    )


class Tee:
    """Collect report lines, print them, and write them beside the script.

    Every script writes its raw output to ``validation/<script stem>_output.txt``
    so that every number in README.md, validation/VALIDATION.md,
    MODEL_CARD.md and DATASET_CARD.md traces to a file produced by a script
    that was actually run. In fast probe mode :meth:`save` writes nothing.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = ROOT / "validation" / Path(path).name
        self.lines: list[str] = []
        self.failures: list[str] = []

    def __call__(self, text: str = "") -> None:
        for line in str(text).splitlines() or [""]:
            self.lines.append(line)
            print(line)

    def rule(self, title: str) -> None:
        self("")
        self("=" * 78)
        self(title)
        self("=" * 78)

    def check(self, label: str, passed: bool, detail: str = "") -> bool:
        """Record a pass or fail line. A failure is collected, never hidden.

        Returns ``passed`` so a caller can branch on it. Failures accumulate in
        :attr:`failures` and :meth:`save` prints them again at the end of the
        file, so a reader of the committed output cannot miss one.
        """
        mark = "PASS" if passed else "FAIL"
        self(f"  [{mark}] {label}" + (f" -- {detail}" if detail else ""))
        if not passed:
            self.failures.append(f"{label}" + (f" -- {detail}" if detail else ""))
        return passed

    def save(self) -> str:
        if self.failures:
            self("")
            self("-" * 78)
            self(f"{len(self.failures)} CHECK(S) FAILED in this script:")
            for item in self.failures:
                self(f"  - {item}")
            self("-" * 78)
        if is_fast():
            return "(fast probe: output not written)"
        self.path.write_text("\n".join(self.lines) + "\n")
        return relative(self.path)


class Timer:
    """Accumulating wall-clock timer whose result only ever appears in a table.

    Timings in this repository appear only in tables explicitly declared
    unstable and never in a prose conclusion: elsewhere in this mission a
    sentence reading "a factor of 40" became "a factor of 38" on re-run for
    exactly this reason.
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
    def mean_s(self) -> float:
        return self.elapsed_s / self.calls if self.calls else 0.0


def _distribution_version(name: str) -> str:
    """Installed version of ``name``, or ``"not installed"``."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def environment_lines() -> list[str]:
    """Versions and core count, recorded in every committed output."""
    import numpy
    import scipy
    import sklearn

    return [
        f"python            : {sys.version.split()[0]}",
        f"numpy             : {numpy.__version__}",
        f"scipy             : {scipy.__version__}",
        f"scikit-learn      : {sklearn.__version__}",
        f"sgp4              : {_distribution_version('sgp4')}",
        f"matplotlib        : {_distribution_version('matplotlib')}",
        f"pytest            : {_distribution_version('pytest')}",
        f"hypothesis        : {_distribution_version('hypothesis')}",
        f"ruff              : {_distribution_version('ruff')}",
        f"astropy           : {_distribution_version('astropy')}  "
        "(NOT used; no number here is cross-checked against it)",
        f"torch             : {_distribution_version('torch')}  "
        "(NOT used; the learned model is scikit-learn)",
        f"cpu cores         : {len(os.sched_getaffinity(0))}",
        f"product root      : {ROOT.name} (printed relatively; see validation/_bootstrap.py)",
        f"fast probe mode   : {is_fast()}",
    ]
