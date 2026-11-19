"""Shared setup for every example: paths, the Agg backend and a figure saver.

Each example writes its PNG into ``screenshots/`` and prints what a reader
should notice. The backend is forced to Agg and ``show()`` is never called, so
an example runs headless.

Every example reads the committed corpus. There is no synthetic fallback at
example time: a fallback would draw a figure that looks like the committed
corpus and is not it. An example run without the corpus prints the refusal,
names ``data/generate_corpus.py``, and exits 2.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Where every example writes its PNG. Relative, always.
SCREENSHOTS = "screenshots"

#: Reduced work under this flag, which only the working-directory probe sets.
FAST = os.environ.get("HANDOVERLOSS_FAST") == "1"


def bootstrap() -> Path:
    """Put ``src`` on ``sys.path``, chdir to the repository root, return it."""
    source = ROOT / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(ROOT)
    return ROOT


def relative(path: str | Path) -> str:
    """Render ``path`` relative to the repository root, never absolutely."""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except ValueError:
        return candidate.name


def corpus():
    """Load the committed corpus, or raise naming the generator. No fallback."""
    from handoverloss.corpus import (
        CORPUS_CSV,
        CORPUS_MANIFEST,
        CorpusAbsentError,
        load_corpus,
    )

    if not (ROOT / CORPUS_CSV).is_file() or not (ROOT / CORPUS_MANIFEST).is_file():
        raise CorpusAbsentError(
            f"CORPUS ABSENT. Expected {CORPUS_CSV} and {CORPUS_MANIFEST}. Regenerate "
            "both, deterministically, with `python3 data/generate_corpus.py` from the "
            "repository root. This example draws no figure from substituted data."
        )
    return load_corpus(CORPUS_CSV, CORPUS_MANIFEST, verify_hash=True)


def figure(name: str, size: tuple[float, float] = (11.0, 6.5)):
    """Create a matplotlib figure and return ``(figure, save)``.

    ``save`` writes the PNG into ``screenshots/`` and returns its relative path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    handle = plt.figure(figsize=size, dpi=120)

    def save() -> str:
        target = ROOT / SCREENSHOTS / f"{name}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        handle.tight_layout()
        handle.savefig(target)
        plt.close(handle)
        return relative(target)

    return handle, save
