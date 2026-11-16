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

#: The one declared seed. The same value as ``validation/_bootstrap.SEED``;
#: a test asserts they agree, so the two cannot drift apart.
SEED = 1000044


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
