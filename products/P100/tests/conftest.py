"""Shared fixtures.

The fixtures build **synthetic** trees in ``tmp_path``. No test reads or writes
the repository this product lives in: a test suite for a tool that audits a
live repository must not be part of that repository's state, or the suite's own
result depends on whatever else is happening in the tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parent.parent / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

#: A script that declares a seed and writes a committed-style output file whose
#: single number is a deterministic function of that seed.
SEEDED_SCRIPT = '''"""A seeded script with a committed output."""

from pathlib import Path

SEED_LOCAL = 7

HERE = Path(__file__).resolve().parent


def main() -> None:
    value = (SEED_LOCAL * 37) % 1000 / 8.0
    text = "value %.6f\\nconstant 42\\n" % value
    (HERE / "seeded_script_output.txt").write_text(text, encoding="utf-8")
    print(text, end="")


main()
'''

#: A script that declares a seed and whose output does not depend on it.
SEED_INDEPENDENT_SCRIPT = '''"""A seeded script whose number does not depend on the seed."""

from pathlib import Path

SEED_LOCAL = 11

HERE = Path(__file__).resolve().parent


def main() -> None:
    text = "value 3.141593\\nconstant 42\\n"
    (HERE / "seed_independent_script_output.txt").write_text(text, encoding="utf-8")
    print(text, end="")


main()
'''

#: A script that declares a seed and never writes its committed output, so its
#: claim is unevaluable rather than clean.
SILENT_SCRIPT = '''"""A seeded script that writes nothing."""

SEED_LOCAL = 13


def main() -> None:
    print("value %.6f" % (SEED_LOCAL / 2.0))


main()
'''


@pytest.fixture
def seeded_tree(tmp_path: Path) -> Path:
    """A product-shaped tree with one seeded script and its committed output."""
    product = tmp_path / "products" / "PTEST"
    validation = product / "validation"
    validation.mkdir(parents=True)
    (validation / "seeded_script.py").write_text(SEEDED_SCRIPT, encoding="utf-8")
    value = (7 * 37) % 1000 / 8.0
    (validation / "seeded_script_output.txt").write_text(
        f"value {value:.6f}\nconstant 42\n", encoding="utf-8"
    )
    return product


@pytest.fixture
def fixture_repository(tmp_path: Path) -> Path:
    """A synthetic repository with three seeded candidates of known behaviour.

    One number moves under a seed change, one does not, and one script never
    writes its output so its claim is unevaluable. The expected headline for
    this tree is therefore exactly 1.
    """
    root = tmp_path / "fixture_repo"
    (root / "products" / "PA" / "validation").mkdir(parents=True)
    (root / "products" / "PB" / "validation").mkdir(parents=True)
    (root / "products" / "PC" / "validation").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "products.yaml").write_text("products: []\n", encoding="utf-8")

    moving = root / "products" / "PA" / "validation"
    (moving / "seeded_script.py").write_text(SEEDED_SCRIPT, encoding="utf-8")
    value = (7 * 37) % 1000 / 8.0
    (moving / "seeded_script_output.txt").write_text(
        f"value {value:.6f}\nconstant 42\n", encoding="utf-8"
    )

    stable = root / "products" / "PB" / "validation"
    (stable / "seed_independent_script.py").write_text(SEED_INDEPENDENT_SCRIPT, encoding="utf-8")
    (stable / "seed_independent_script_output.txt").write_text(
        "value 3.141593\nconstant 42\n", encoding="utf-8"
    )

    silent = root / "products" / "PC" / "validation"
    (silent / "silent_script.py").write_text(SILENT_SCRIPT, encoding="utf-8")
    (silent / "silent_script_output.txt").write_text("value 6.500000\n", encoding="utf-8")
    return root


@pytest.fixture
def polluting_tree(tmp_path: Path) -> Path:
    """A tree holding one deliberately self-polluting check."""
    from seedaudit.corpus import generate_corpus

    root = tmp_path / "corpus"
    generate_corpus(root)
    return root
