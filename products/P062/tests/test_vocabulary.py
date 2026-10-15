"""Vocabulary and documentation tests.

The mission forbids a set of words and requires a set of statements. A grep
written down as a test is the only way those stay true after an edit.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN = (
    "flight-safe",
    "flight safe",
    "mission-ready",
    "mission ready",
    "production-ready",
    "production ready",
    "revolutionary",
    "cutting-edge",
    "game-changing",
    "seamless",
    "turnkey",
    "state-of-the-art",
)

REQUIRED_SAFETY = (
    "not flight-qualified",
    "not certified",
    "not approved for operational aerospace use",
)

CREDITS_LINE = "This is under reserved rights obtained by OPTIMA Organisation."


def _text_files() -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.md", "*.toml", "*.cff", "*.yml"):
        paths.extend(ROOT.glob(pattern))
    for folder in ("src", "tests", "examples", "validation", "docs", ".github"):
        for pattern in ("*.py", "*.md", "*.yml"):
            paths.extend((ROOT / folder).rglob(pattern))
    return [p for p in paths if p.is_file()]


def test_no_forbidden_marketing_word_anywhere():
    offenders: list[str] = []
    for path in _text_files():
        # This file necessarily contains the forbidden list itself.
        if path.name == "test_vocabulary.py":
            continue
        lowered = path.read_text(encoding="utf-8").lower()
        for word in FORBIDDEN:
            if word in lowered:
                offenders.append(f"{path.relative_to(ROOT)}: {word}")
    assert offenders == []


def test_no_exclamation_marks_in_prose_or_documentation():
    offenders: list[str] = []
    for path in _text_files():
        if path.suffix == ".py":
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            # Markdown image syntax is "![alt](path)" and the repository
            # standard requires it, so an exclamation mark immediately followed
            # by "[" is not prose. Everything else is.
            stripped = re.sub(r"!\[", "", line)
            if "!" in stripped and not stripped.lstrip().startswith("#!"):
                offenders.append(f"{path.relative_to(ROOT)}:{number}")
    assert offenders == []


def test_no_emoji_in_any_shipped_file():
    # The pattern is built from code points rather than written as literal
    # characters, so that this test file does not itself contain the thing it
    # forbids. Covered: the whole supplementary planes (where every modern
    # emoji lives), the Miscellaneous Symbols and Dingbats blocks, the
    # Miscellaneous Symbols and Arrows block, and the variation selector that
    # turns a symbol into an emoji presentation. Typographic arrows in the
    # Arrows block (U+2190 to U+21FF) are NOT emoji and are not forbidden.
    ranges = [
        (0x1F000, 0x1FAFF),
        (0x1F300, 0x1F9FF),
        (0x2600, 0x27BF),
        (0x2B00, 0x2BFF),
        (0xFE0F, 0xFE0F),
    ]
    pattern = re.compile(
        "[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in ranges) + "]"
    )
    offenders: list[str] = []
    for path in _text_files():
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}")
    assert offenders == []


@pytest.mark.parametrize("name", ["README.md", "MODEL_CARD.md", "DATASET_CARD.md"])
def test_required_safety_statement_is_present(name):
    # Newlines are collapsed first, because the required phrases are long enough
    # to wrap and a line break must not be allowed to hide one.
    text = re.sub(r"\s+", " ", (ROOT / name).read_text(encoding="utf-8").lower())
    for phrase in REQUIRED_SAFETY:
        assert phrase in text, f"{name} is missing: {phrase}"


def test_readme_status_header_is_testing():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Status: TESTING" in text


def test_credits_line_appears_exactly_once_in_the_readme():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert text.count(CREDITS_LINE) == 1
    index = text.index(CREDITS_LINE)
    assert "Credits" in text[max(0, index - 1200) : index]


def test_model_card_carries_the_required_sentence():
    text = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "This model is not certified for operational flight use." in text


def test_model_card_has_every_required_section():
    text = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8").lower()
    for heading in (
        "problem",
        "baseline",
        "architecture",
        "dataset",
        "training procedure",
        "metrics",
        "test-split",
        "uncertainty",
        "failure",
        "reproducib",
        "compute",
        "ethic",
    ):
        assert heading in text, f"MODEL_CARD.md is missing a section about {heading}"


def test_alternatives_table_names_every_required_package():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in (
        "lmfit",
        "scipy.optimize.least_squares",
        "emcee",
        "SALib",
        "pysindy",
    ):
        assert name in text, f"the alternatives table must name {name}"


def test_readme_refers_only_to_screenshots_that_exist():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for match in re.finditer(r"!\[[^\]]*\]\((screenshots/[^)]+)\)", text):
        assert (ROOT / match.group(1)).is_file(), f"missing {match.group(1)}"


def test_readme_refers_only_to_validation_files_that_exist():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for match in re.finditer(r"`(validation/[A-Za-z0-9_./-]+)`", text):
        assert (ROOT / match.group(1)).is_file(), f"missing {match.group(1)}"


def test_validation_md_refers_only_to_outputs_that_exist():
    text = (ROOT / "validation" / "VALIDATION.md").read_text(encoding="utf-8")
    for match in re.finditer(r"`(validate_[A-Za-z0-9_]+(?:\.py|_output\.txt))`", text):
        assert (
            ROOT / "validation" / match.group(1)
        ).is_file(), f"missing validation/{match.group(1)}"


def test_every_validation_script_has_its_committed_output():
    scripts = sorted((ROOT / "validation").glob("*.py"))
    assert len(scripts) >= 7
    for script in scripts:
        if script.name.startswith("_"):
            continue
        expected = script.with_name(f"{script.stem}_output.txt")
        assert expected.is_file(), f"missing {expected.name}"


def test_library_code_contains_no_print_statement():
    offenders: list[str] = []
    for path in (ROOT / "src").rglob("*.py"):
        if path.name == "__main__.py":
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if re.search(r"(^|[^A-Za-z_.])print\s*\(", line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}")
    assert offenders == []


def test_examples_use_the_agg_backend_and_never_show():
    for path in sorted((ROOT / "examples").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert 'matplotlib.use("Agg")' in text, f"{path.name} must select Agg"
        assert "plt.show" not in text, f"{path.name} must not call show"
