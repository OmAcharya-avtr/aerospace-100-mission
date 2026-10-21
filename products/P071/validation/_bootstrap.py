"""Path bootstrap and output tee shared by every validation script.

Two things matter here and both exist because of defects this mission has
shipped.

**The working directory.** Every script in this directory calls
:func:`bootstrap` first. It puts ``<repo>/src`` on ``sys.path`` and then
changes the process working directory to the repository root, which is derived
from ``__file__`` and not from the caller. Four products in Batch 07 passed
every check in their own build and then failed the release gate, which runs each
validation script with its working directory set to ``validation/`` rather than
to the product root. One ``os.chdir`` fixes every script in the directory and is
a no-op when the script is already run from the product root. These scripts are
executed from ``/``, ``/tmp``, the product root and ``validation/`` before
hand-off, and all four must exit zero.

**Absolute paths.** No committed file in this repository may contain an
absolute path: it is not reproducible for a reader with a cold clone and it
publishes the build machine's directory layout. Everything printed by these
scripts goes through :func:`relative`, which renders a path relative to the
repository root, and the repository root itself is printed as its own directory
name and nothing above it.
"""

from __future__ import annotations

import os
import sys
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
    the only part that is reproducible for a reader. This is how the SGP4
    verification files inside the installed ``sgp4`` package are named in the
    committed output.
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
    validation/VALIDATION.md and MODEL_CARD.md can be traced to a file produced
    by a script that was actually run.
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


def _distribution_version(name: str) -> str:
    """Installed version of ``name``, or ``"not installed"``.

    Used for the tooling that is not imported by the library itself, so that the
    committed output records the versions every number was measured against
    without this module importing pytest or ruff.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def environment_lines() -> list[str]:
    """Versions and core count, so every output file records what produced it."""
    import matplotlib
    import numpy
    import scipy
    import sgp4
    import sklearn
    import yaml

    return [
        f"python           : {sys.version.split()[0]}",
        f"numpy            : {numpy.__version__}",
        f"scipy            : {scipy.__version__}",
        f"scikit-learn     : {sklearn.__version__}",
        f"matplotlib       : {matplotlib.__version__}",
        f"sgp4             : {sgp4.__version__}",
        f"pyyaml           : {yaml.__version__}",
        f"pytest           : {_distribution_version('pytest')}",
        f"hypothesis       : {_distribution_version('hypothesis')}",
        f"ruff             : {_distribution_version('ruff')}",
        f"astropy          : {_distribution_version('astropy')}",
        f"cpu cores        : {len(os.sched_getaffinity(0))}",
        f"repository root  : {ROOT.name} (printed relatively; see validation/_bootstrap.py)",
    ]
