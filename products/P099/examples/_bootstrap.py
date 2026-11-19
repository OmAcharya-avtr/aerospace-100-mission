"""Path bootstrap and figure style for the examples.

Anchored on ``__file__`` and never on the caller's working directory: three
products in one batch of this mission passed every check in their own build
and then failed the release gate, which runs a script with its working
directory set to the script's own directory rather than to the product root.

Every example sets ``sys.dont_write_bytecode = True`` as its first statement,
before any other import. It cannot be delegated to this module, because
CPython writes a module's bytecode cache BEFORE executing that module, so this
module cannot set it on its own behalf.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Where every example writes its figure, relative to the product root.
SCREENSHOTS = "screenshots"

#: The one declared seed. The same value as ``validation/_bootstrap.SEED``;
#: ``tests/test_determinism.py`` asserts they agree, so the two cannot drift.
SEED = 99099

#: Confidence level of every interval an example prints. Same value as
#: ``validation/_bootstrap.CONFIDENCE``.
CONFIDENCE = 0.95

#: Bootstrap resamples. Same value as
#: ``validation/_bootstrap.BOOTSTRAP_RESAMPLES``.
BOOTSTRAP_RESAMPLES = 2000

#: Nominal coverage of every predictive interval an example draws. Same value
#: as ``validation/_bootstrap.NOMINAL_COVERAGE``.
NOMINAL_COVERAGE = 0.80

#: ``PASSYIELD_FAST`` lowers run counts and is used only by
#: ``validation/probe_working_directories.py``, whose question is whether a
#: script runs from a given directory and not what its figures are. It is
#: never set when a committed screenshot is produced, and every example
#: prints whether it was set.
FAST = os.environ.get("PASSYIELD_FAST") == "1"


def record_limit() -> int | None:
    """Rows of the committed record an example should use, or ``None`` for all.

    On the fast path an example uses a small slice, because the probe that
    sets the fast path is asking whether the script runs from a given
    directory and not what the figure looks like. A committed screenshot is
    never produced on the fast path.
    """
    return 160 if FAST else None


def bootstrap() -> Path:
    """Put ``src`` on ``sys.path``, chdir to the product root, return it."""
    source = ROOT / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    os.chdir(ROOT)
    (ROOT / SCREENSHOTS).mkdir(exist_ok=True)
    return ROOT


def relative(path: str | Path) -> str:
    """Render ``path`` relative to the product root, never absolutely."""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except ValueError:
        return candidate.name


def style() -> None:
    """Apply the shared figure style. Agg only; never ``show()``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
