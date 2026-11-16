"""Path bootstrap, figure style and output tee for the examples.

Same two rules as the validation scripts, for the same reasons. Every runnable
script here sets ``sys.dont_write_bytecode = True`` as its first statement after
``from __future__ import annotations`` and before any other import, because
CPython writes a module's bytecode cache before executing it and a shared helper
cannot set the flag on its own behalf. Every path is anchored on ``__file__``
and never on the caller's working directory.

**No figure in ``screenshots/`` carries a timestamp, a hostname or a run
counter.** A stamp like that is stale the moment the data is regenerated and
the image is not, and it makes a figure look newer than its numbers. Every
label in every figure is a value measured from the data the figure draws.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Where the examples write their PNGs. Relative, always.
SCREENSHOTS = "screenshots"

#: Figure size and resolution. One place, so three figures look like one set.
FIGURE_SIZE = (11.0, 6.2)
DPI = 130

#: Colours. Chosen for a grey-scale-safe ordering as well as a colour one.
INK = "#1b1b1b"
GRID = "#d6d6d6"
HARDWARE = "#1f4e79"
SIMULATION = "#8c5a00"
DECLARED = "#9b1c31"
NEUTRAL = "#6b6b6b"


def bootstrap() -> Path:
    """Put ``src`` on ``sys.path``, chdir to the product root, return it."""
    source = ROOT / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    os.chdir(ROOT)
    return ROOT


def relative(path: str | Path) -> str:
    """Render ``path`` relative to the product root, never absolutely."""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except ValueError:
        return candidate.name


def figure_path(name: str) -> Path:
    """Absolute path of a screenshot, with the directory created."""
    target = ROOT / SCREENSHOTS
    target.mkdir(parents=True, exist_ok=True)
    return target / name


def style():
    """Configure matplotlib for the Agg backend and return the pyplot module."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    pyplot.rcParams.update(
        {
            "figure.figsize": FIGURE_SIZE,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.size": 10.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.frameon": False,
            "figure.autolayout": False,
        }
    )
    return pyplot


class Tee:
    """Print and collect, then write beside the script as ``<stem>_output.txt``."""

    def __init__(self, stem: str) -> None:
        self.path = ROOT / "examples" / f"{stem}_output.txt"
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        for line in str(text).splitlines() or [""]:
            self.lines.append(line)
            print(line)

    def save(self) -> str:
        self.path.write_text("\n".join(self.lines) + "\n")
        return relative(self.path)


def scope_header(tee: Tee, title: str) -> None:
    """Title plus the one-line scope statement that travels with every figure."""
    from hiltimingdiff import RESEARCH_GRADE, SCOPE_LINE

    tee(title)
    tee("=" * len(title))
    tee(SCOPE_LINE)
    tee(RESEARCH_GRADE)
    tee("")
