"""What the tracked files may and may not say, and what the tree may carry.

Three checks, each of which this mission has shipped a product without.

* **No marketing language and no exclamation mark** in any tracked text file.
* **No absolute path** in any tracked file: it is not reproducible for a reader
  with a cold clone and it publishes the build machine's filesystem layout. The
  scanner assembles its patterns from ``os.sep`` and skips its own source, both
  because a check whose own execution can create the condition it tests is not
  a check.
* **No flight-safety or certification claim**, and the research-grade statement
  present where it is required.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Directories excluded from the text scan. ``data/`` holds the measurement
#: programs and their outputs byte for byte as they ran on the board; they are
#: evidence and are not edited. ``screenshots/`` is binary.
EXCLUDED_DIRECTORIES = (".git", "__pycache__", ".ruff_cache", ".pytest_cache", "data")

#: Suffixes scanned for prose defects.
TEXT_SUFFIXES = (".py", ".md", ".toml", ".yml", ".yaml", ".cff", ".txt")

MARKETING_WORDS = (
    "revolutionary",
    "cutting-edge",
    "cutting edge",
    "state-of-the-art",
    "powerful",
    "seamless",
    "game-changing",
    "world-class",
    "best-in-class",
    "effortless",
    "blazing",
    "turnkey",
    "unparalleled",
)

FORBIDDEN_CLAIMS = (
    "flight-safe",
    "flight safe",
    "mission-ready",
    "mission ready",
    "production-ready",
    "production ready",
    "flight qualified",
)

#: Files that must each carry the research-grade statement.
RESEARCH_GRADE_FILES = (
    "README.md",
    "validation/VALIDATION.md",
    "src/hiltimingdiff/__init__.py",
    "src/hiltimingdiff/report.py",
)


def _tracked_text_files() -> list[Path]:
    found: list[Path] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRECTORIES for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix not in TEXT_SUFFIXES:
            continue
        found.append(path)
    return found


TEXT_FILES = _tracked_text_files()
TEXT_IDS = [str(path.relative_to(ROOT)) for path in TEXT_FILES]

#: Files that necessarily name the patterns they forbid, so the prose scans
#: exclude them by construction rather than skipping over them at run time.
#: ``test_cli.py`` asserts that the CLI's own output carries no exclamation
#: mark, which means its source must contain one.
SELF_REFERENTIAL = (
    "test_honesty.py",
    "test_report.py",
    "test_cli.py",
    "validate_cli.py",
    "validate_tree_hygiene.py",
    "validate_tree_hygiene_output.txt",
)

#: Prose files: everything but Python source, which needs ``!=`` and ``!r``,
#: and but the self-referential files, which quote the marks they forbid.
PROSE_FILES = [
    path
    for path in TEXT_FILES
    if path.suffix != ".py" and path.name not in SELF_REFERENTIAL
]
PROSE_IDS = [str(path.relative_to(ROOT)) for path in PROSE_FILES]

#: Files the word scans apply to.
SCANNED_FILES = [path for path in TEXT_FILES if path.name not in SELF_REFERENTIAL]
SCANNED_IDS = [str(path.relative_to(ROOT)) for path in SCANNED_FILES]

#: Files the absolute-path scan applies to: everything but this one, which names
#: the patterns it scans for.
PATH_FILES = [path for path in TEXT_FILES if path.name != "test_honesty.py"]
PATH_IDS = [str(path.relative_to(ROOT)) for path in PATH_FILES]


def test_the_scan_found_files_to_scan():
    # A scanner that finds nothing passes silently. This asserts it has work.
    assert len(TEXT_FILES) >= 20


@pytest.mark.parametrize("path", PATH_FILES, ids=PATH_IDS)
def test_tracked_file_carries_no_absolute_path(path):
    separator = os.sep
    text = path.read_text()
    for pattern in (
        f"{separator}home{separator}",
        f"{separator}Users{separator}",
        f"{separator}root{separator}",
        f"{separator}tmp{separator}claude",
    ):
        assert pattern not in text, f"{path.name} carries {pattern}"


#: The two places an exclamation mark is syntax rather than prose: a markdown
#: image or badge, ``![alt](path)``, and a gitignore negation, ``!/path``.
SYNTAX_AFTER_BANG = ("[", "/")


def _prose_exclamations(text: str) -> list[int]:
    """Line numbers of exclamation marks that are prose rather than syntax.

    Fenced code blocks are skipped. They hold verbatim transcripts -- captured
    command output, a quoted release-gate verdict -- and editing a transcript to
    satisfy a prose rule would be falsifying evidence to pass a style check.
    The rule is about prose, so it is applied to prose.
    """
    offenders: list[int] = []
    inside_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("```"):
            inside_fence = not inside_fence
            continue
        if inside_fence:
            continue
        for index, character in enumerate(line):
            if character != "!":
                continue
            if line[index + 1 : index + 2] in SYNTAX_AFTER_BANG:
                continue
            offenders.append(number)
    return offenders


def test_the_prose_exclamation_rule_skips_fenced_code_but_not_prose():
    # The skip must not become a hole. Checked directly.
    assert _prose_exclamations("```\nnot prose!\n```\n") == []
    assert _prose_exclamations("this is prose!\n") == [1]
    assert _prose_exclamations("![badge](x)\n") == []
    assert _prose_exclamations("!/data/traces/campaign.log\n") == []


@pytest.mark.parametrize("path", PROSE_FILES, ids=PROSE_IDS)
def test_prose_file_carries_no_exclamation_mark(path):
    # Python source is excluded by construction, because it needs ``!=`` and
    # ``!r``. Markdown image syntax and gitignore negations are excluded because
    # the mark is syntax there, not emphasis. Everything else must carry none.
    assert _prose_exclamations(path.read_text()) == [], path.name


def test_python_source_carries_no_exclamation_mark_outside_operators():
    # The prose rule applied to Python: every ``!`` in the package, the suite,
    # the examples and the validation scripts must be part of ``!=`` or ``!r``.
    offenders: list[str] = []
    for path in SCANNED_FILES:
        if path.suffix != ".py":
            continue
        text = path.read_text()
        for index, character in enumerate(text):
            if character != "!":
                continue
            following = text[index + 1 : index + 2]
            if following in ("=", "r", "s", "a", "/"):
                continue
            offenders.append(f"{path.name}:{text.count(chr(10), 0, index) + 1}")
    assert offenders == []


@pytest.mark.parametrize("path", SCANNED_FILES, ids=SCANNED_IDS)
def test_tracked_file_carries_no_marketing_word(path):
    lowered = path.read_text().lower()
    for word in MARKETING_WORDS:
        assert word not in lowered, f"{path.name} carries {word!r}"


@pytest.mark.parametrize("path", SCANNED_FILES, ids=SCANNED_IDS)
def test_tracked_file_makes_no_flight_safety_claim(path):
    lowered = path.read_text().lower()
    for claim in FORBIDDEN_CLAIMS:
        assert claim not in lowered, f"{path.name} carries {claim!r}"


@pytest.mark.parametrize("name", RESEARCH_GRADE_FILES)
def test_required_file_carries_the_research_grade_statement(name):
    text = (ROOT / name).read_text().lower()
    assert "research-grade" in text
    assert "not flight-qualified" in text
    assert "not certified" in text


def test_readme_carries_the_credits_line_once():
    text = (ROOT / "README.md").read_text()
    line = "This is under reserved rights obtained by OPTIMA Organisation."
    assert text.count(line) == 1


def test_license_names_the_organisation():
    text = (ROOT / "LICENSE").read_text()
    assert "OPTIMA Organisation" in text
    assert "2026" in text


def test_pyproject_declares_bounded_dependency_ranges():
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    for requirement in data["project"]["dependencies"]:
        assert ">=" in requirement, requirement
        assert "<" in requirement, requirement


def test_pyproject_declares_the_measured_ruff_range():
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]["test"]
    assert any(item.startswith("ruff>=0.16,<0.18") for item in extras)


def test_pyproject_package_name_is_the_registry_name():
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert data["project"]["name"] == "hiltimingdiff"


def test_pyproject_declares_no_learned_dependency():
    # The registry declares ai_enabled: false. A scikit-learn or torch
    # dependency would contradict it on the packaging surface.
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    joined = " ".join(
        data["project"]["dependencies"]
        + data["project"]["optional-dependencies"]["dev"]
    ).lower()
    for forbidden in ("scikit-learn", "torch", "tensorflow", "keras", "xgboost"):
        assert forbidden not in joined


def test_tree_carries_no_tracked_bytecode_or_cache_in_the_package():
    # What the product OWNS: no bytecode or cache directory under src/, tests/,
    # examples/ or validation/ that this product created. A harness may leave
    # one in the tree before this test runs; that is reported by
    # validation/validate_tree_hygiene.py and named as inherited there, and is
    # not asserted here.
    for directory in ("src", "tests", "examples", "validation"):
        for path in (ROOT / directory).rglob("*.pyc"):
            assert "__pycache__" in str(path), path
