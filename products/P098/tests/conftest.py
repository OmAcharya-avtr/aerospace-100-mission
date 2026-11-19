"""Shared fixtures. The committed corpus is loaded once per session.

The product root is derived from ``__file__``, never from the caller's working
directory, so the suite passes when pytest is invoked from anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evidencestale.corpus import load_corpus  # noqa: E402
from evidencestale.graph import DependencyGraph  # noqa: E402
from evidencestale.model import ChangeEvent, EvidenceItem  # noqa: E402
from evidencestale.replay import ChangeHistory  # noqa: E402


@pytest.fixture(scope="session")
def product_root() -> Path:
    """The product root directory."""
    return ROOT


@pytest.fixture(scope="session")
def layered_corpus():
    """The committed layered corpus, loaded once."""
    return load_corpus(ROOT / "data" / "corpus_layered.json")


@pytest.fixture(scope="session")
def cyclic_corpus():
    """The committed cyclic corpus, loaded once."""
    return load_corpus(ROOT / "data" / "corpus_cyclic.json")


@pytest.fixture
def chain_graph() -> DependencyGraph:
    """``top -> middle -> bottom``: a three-artifact chain of depth 2."""
    return DependencyGraph.build(
        ["top", "middle", "bottom"], [("top", "middle"), ("middle", "bottom")]
    )


@pytest.fixture
def chain_history() -> ChangeHistory:
    """A hand-written history over the chain fixture.

    Revision 0 lays down ``*:v0`` for all three. Then:

    - revision 4: ``bottom`` written to ``bottom:v1``   (a real content change)
    - revision 6: ``top`` touched                        (no content change)
    - revision 8: ``bottom`` written back to ``bottom:v0`` (a revert)
    """
    return ChangeHistory(
        ["top", "middle", "bottom"],
        {"top": "top:v0", "middle": "middle:v0", "bottom": "bottom:v0"},
        [
            ChangeEvent(time=4, artifact="bottom", kind="write", content="bottom:v1"),
            ChangeEvent(time=6, artifact="top", kind="touch"),
            ChangeEvent(time=8, artifact="bottom", kind="write", content="bottom:v0"),
        ],
    )


@pytest.fixture
def chain_items() -> tuple[EvidenceItem, ...]:
    """Two evidence items about ``top``, produced at revisions 1 and 7."""
    return (
        EvidenceItem(
            identifier="e0", subject="top", produced_at=1, goal="G1", solution="Sn0"
        ),
        EvidenceItem(
            identifier="e1", subject="top", produced_at=7, goal="G1", solution="Sn1"
        ),
    )
