"""Path bootstrap and figure style for the examples.

Anchored on ``__file__`` and never on the caller's working directory: three
products in one batch of this mission passed every check in their own build and
then failed the release gate, which runs a script with its working directory
set to the script's own directory rather than to the product root.

Every example sets ``sys.dont_write_bytecode = True`` as its first statement,
before any other import. It cannot be delegated to this module, because CPython
writes a module's bytecode cache BEFORE executing that module, so this module
cannot set it on its own behalf.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Where every example writes its figure, relative to the product root.
SCREENSHOTS = "screenshots"

#: The one declared seed. The same value as ``validation/_bootstrap.SEED``; a
#: test asserts the two agree, so they cannot drift apart.
SEED = 980098

#: Bootstrap resamples for any interval an example draws.
RESAMPLES = 2000

#: Grouped folds for any split an example makes.
GROUP_SPLITS = 5

#: ``EVIDENCESTALE_FAST`` lowers resample and fold counts. It is set only by
#: ``validation/probe_working_directories.py``, whose question is whether a
#: script runs from a given directory and not what its figure contains. Every
#: example prints whether it was set.
FAST = os.environ.get("EVIDENCESTALE_FAST") == "1"

#: Counts actually used, after the fast path.
EFFECTIVE_RESAMPLES = 100 if FAST else RESAMPLES
EFFECTIVE_SPLITS = 2 if FAST else GROUP_SPLITS

#: Figure style, so the four figures look like one set.
FIGURE_SIZE = (9.0, 5.0)
DPI = 130
GRID_STYLE = {"alpha": 0.3, "linewidth": 0.6}

#: Colours, chosen so the two error directions are never the same colour.
COLOUR_TRUTH = "#1b3a5c"
COLOUR_CLOSURE = "#2e7d5b"
COLOUR_TIMESTAMP = "#a8481c"
COLOUR_LEARNED = "#6a4c93"
COLOUR_MISSED = "#a8481c"
COLOUR_ALARM = "#d9a23b"


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


def save(figure, name: str) -> str:
    """Save ``figure`` into ``screenshots/`` and return the relative path."""
    target = ROOT / SCREENSHOTS / name
    figure.savefig(target, dpi=DPI, bbox_inches="tight")
    return relative(target)
