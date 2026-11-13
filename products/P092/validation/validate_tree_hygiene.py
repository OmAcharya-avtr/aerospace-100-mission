"""What this product owns: that it ADDS no artifact, and that git tracks none.

The mission's build guide narrows this check deliberately, because the broad
version cost an earlier product a night and could never have been satisfied. A
self-check asserts two things and not the third:

- that running the scripts here ADDS no cache directory and no bytecode file --
  the tree is snapshotted at entry and compared at exit, and anything new is a
  failure;
- that git TRACKS no cache directory, bytecode file or build artifact, which is
  what a cold clone actually carries and what release-gate check 9 enforces
  independently;
- **not** that no such artifact EXISTS, because release-gate check 2 runs
  ``ruff check`` with its working directory set to the product root and writes
  ``.ruff_cache`` into the tree one check before check 6 re-executes this script.
  Anything found that this script did not create is reported, named as
  inherited, and does not fail.

The absolute-path scan below deletes its own previous output before scanning and
assembles its patterns from ``os.sep``, because three consecutive sessions of
this mission introduced a self-reporting scanner in the act of writing about one.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

#: Artifact names release-gate check 9 fails on.
ARTIFACT_NAMES = (
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".hypothesis",
    ".config",
    "dist",
    "build",
)

#: Suffixes of build artifacts and model checkpoints.
ARTIFACT_SUFFIXES = (".pyc", ".pyo", ".ckpt", ".pt", ".pth", ".onnx", ".egg-info")

#: Files this product MUST ship. Each is checked against .gitignore, because a
#: pattern with no slash matches at any depth and silently excluded a required
#: fixture in an earlier product of this mission.
MUST_SHIP = (
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "MODEL_CARD.md",
    "DATASET_CARD.md",
    "CITATION.cff",
    ".gitignore",
    ".github/workflows/tests.yml",
    "docs/SPEC.md",
    "docs/REQUIREMENTS.md",
    "data/campaign.py",
    "tests/fixtures/synthetic_trace.py",
    "validation/VALIDATION.md",
    "validation/_bootstrap.py",
)

#: Paths that MUST be ignored, so a cache never reaches a clone.
MUST_IGNORE = (
    "__pycache__/x.pyc",
    "src/deadlinerisk/__pycache__/trace.cpython-313.pyc",
    ".pytest_cache/CACHEDIR.TAG",
    ".ruff_cache/content",
    ".hypothesis/examples/x",
    "build/lib/x.py",
    "dist/deadlinerisk-0.1.0.tar.gz",
    "junit.xml",
)


def snapshot() -> set[str]:
    """Relative paths of every artifact-shaped entry currently in the tree."""
    found: set[str] = set()
    for path in ROOT.rglob("*"):
        name = path.name
        if name in ARTIFACT_NAMES or name.endswith(ARTIFACT_SUFFIXES):
            found.add(_bootstrap.relative(path))
    return found


def _git(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _ignore_match(name: str) -> tuple[bool, str]:
    """``(is_ignored, matched pattern)`` for ``name``, handling NEGATION patterns.

    ``git check-ignore`` exits 0 whenever ANY pattern matched, **including a
    negation pattern**, so its exit status alone says "a pattern mentioned this
    path", not "this path is ignored". Measured in a scratch repository with
    this product's own ``.gitignore``: ``git check-ignore -v
    data/traces/jobs_probe.csv`` exits 0 and prints ``!/data/traces/*.csv``,
    while ``git status --ignored`` shows the file as stageable. A check keyed on
    the exit status alone would have reported every committed measured trace as
    wrongly excluded the moment the campaign wrote one.

    A path is ignored only when the LAST matching pattern is not a negation.
    """
    result = _git("check-ignore", "-v", "--no-index", name)
    if result.returncode != 0 or not result.stdout.strip():
        return False, ""
    last = result.stdout.strip().splitlines()[-1]
    fields = last.split("\t")
    pattern = fields[0].rsplit(":", 1)[-1] if fields else ""
    return not pattern.startswith("!"), last


def main() -> int:
    out = _bootstrap.Tee("validate_tree_hygiene_output.txt")
    entry = snapshot()
    # Delete this script's own previous output before the absolute-path scan, so
    # the scan cannot report itself.
    own_output = ROOT / "validation" / "validate_tree_hygiene_output.txt"
    if own_output.exists():
        own_output.unlink()

    out.rule("TREE HYGIENE -- what this product owns")
    out("INPUT ORIGIN: this repository's own tree and git index.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)

    out.rule("1  inherited artifacts, reported and NOT failed")
    if entry:
        for name in sorted(entry):
            out(f"  inherited: {name}")
        out(
            f"  {len(entry)} artifact-shaped entr(ies) were already in the tree when this "
            "script started. What a harness leaves in the tree before calling a script is "
            "not the script's doing, and release-gate check 2 writes .ruff_cache into the "
            "product root one check before check 6 runs this file."
        )
    else:
        out("  none present at entry")

    out.rule("2  git tracks no artifact")
    # `git ls-files` run with the working directory set to the product root
    # prints paths RELATIVE TO THAT DIRECTORY and lists only that subtree. An
    # earlier form of this check filtered those paths for the prefix
    # "products/P092/", which they never carry, so the filtered list was always
    # empty and the check could not fail -- in the monorepo before the
    # coordinator commits AND after. A check whose result cannot depend on the
    # thing it tests is not a check; this mission has shipped that class five
    # times and it is named in templates/PRODUCT_BUILD_GUIDE.md. The fix uses
    # --full-name, which prints repository-relative paths, and makes no
    # assumption about where in a repository the product sits: in the mission
    # monorepo the paths come back as products/P092/..., and in the published
    # single-product repository as src/..., and the artifact test below reads
    # the path components either way.
    tracked = _git("ls-files", "--full-name").stdout.split()
    out(f"  git tracks {len(tracked)} file(s) under this product tree")
    if not tracked:
        out(
            "  NOTE: nothing under this product tree is tracked yet. This build does not "
            "commit, by instruction; the coordinator commits. The check below is therefore "
            "vacuous at the moment it runs, and the .gitignore coverage check in section 3 "
            "is the one that has content. The NOTE is printed only when the list is "
            "genuinely empty, so it cannot mask a populated tree."
        )
    offending = [
        name
        for name in tracked
        if any(part in ARTIFACT_NAMES for part in Path(name).parts)
        or name.endswith(ARTIFACT_SUFFIXES)
    ]
    out.check(
        "no artifact-shaped file is tracked under this product tree",
        not offending,
        f"{len(offending)} offending path(s)" + (f": {offending[:3]}" if offending else ""),
    )

    out.rule("3  .gitignore coverage, in both directions")
    missing = [name for name in MUST_SHIP if not (ROOT / name).exists()]
    out.check(
        "every file this product must ship exists on disk",
        not missing,
        f"absent: {missing}" if missing else f"{len(MUST_SHIP)} checked",
    )
    wrongly_ignored = []
    for name in MUST_SHIP:
        if not (ROOT / name).exists():
            continue
        ignored, matched = _ignore_match(name)
        if ignored:
            wrongly_ignored.append(f"{name} <- {matched}")
    out.check(
        "no file this product must ship is excluded by .gitignore",
        not wrongly_ignored,
        f"{len(wrongly_ignored)} wrongly ignored" if wrongly_ignored
        else f"{len(MUST_SHIP)} checked, none ignored",
    )
    for name in wrongly_ignored:
        out(f"    WRONGLY IGNORED: {name}")
    not_ignored = []
    for name in MUST_IGNORE:
        ignored, _matched = _ignore_match(name)
        if not ignored:
            not_ignored.append(name)
    out.check(
        "every cache and build path is excluded by .gitignore",
        not not_ignored,
        f"{len(not_ignored)} not ignored: {not_ignored}" if not_ignored
        else f"{len(MUST_IGNORE)} checked, all ignored",
    )
    # The anchoring lesson, asserted on what this product owns: its OWN
    # .gitignore must anchor every scratch pattern.
    own = (ROOT / ".gitignore").read_text().splitlines()
    unanchored = [
        line
        for line in own
        if line.strip()
        and not line.startswith(("#", "!"))
        and "/" not in line.rstrip("/")
        and not line.startswith("*")
        and not line.endswith("/")
    ]
    out.check(
        "every scratch pattern in this product's own .gitignore is anchored or a suffix",
        not unanchored,
        f"unanchored: {unanchored}" if unanchored else f"{len(own)} lines checked",
    )
    out.check(
        "this product's .gitignore anchors junit.xml as /junit.xml",
        any(line.strip() == "/junit.xml" for line in own)
        and not any(line.strip() == "junit.xml" for line in own),
        "a bare junit.xml would match at any depth",
    )
    nested = "tests/fixtures/junit.xml"
    nested_ignored, nested_match = _ignore_match(nested)
    if nested_ignored:
        out(
            f"  INHERITED FINDING, reported and not failed: {nested} IS excluded, by "
            f"{nested_match.split(chr(9))[0]} at the REPOSITORY root, not by this "
            "product's .gitignore. That pattern would swallow a nested junit.xml fixture "
            "in any product that shipped one. P092 ships none, so nothing of this "
            "product's is lost; the finding belongs to whoever owns the repository-level "
            "file."
        )
    else:
        out(f"  {nested} is not excluded by any .gitignore in scope")

    out.rule("3b  the measured traces must reach a cold clone when they exist")
    traces = _bootstrap.measured_trace_directory()
    present = sorted(path.name for path in traces.glob("*.csv")) if traces.is_dir() else []
    marker_present = (traces / "CAMPAIGN_COMPLETE").exists()
    out(
        f"  {_bootstrap.MEASURED_TRACES}/ holds {len(present)} CSV file(s)"
        + (f": {present[:4]}{' ...' if len(present) > 4 else ''}" if present else "")
    )
    out(f"  {_bootstrap.MEASURED_TRACES}/CAMPAIGN_COMPLETE present: {marker_present}")
    if present:
        swallowed = [
            name
            for name in present
            if _ignore_match(f"{_bootstrap.MEASURED_TRACES}/{name}")[0]
        ]
        marker_swallowed = marker_present and _ignore_match(
            f"{_bootstrap.MEASURED_TRACES}/CAMPAIGN_COMPLETE"
        )[0]
        out.check(
            "the campaign completion marker is not excluded by .gitignore",
            not marker_swallowed,
            "CAMPAIGN_COMPLETE is a tracked zero-byte file and part of the evidence"
            if marker_present
            else "marker not present yet",
        )
        out.check(
            "no measured trace is excluded by .gitignore",
            not swallowed,
            f"{len(swallowed)} swallowed: {swallowed[:3]}" if swallowed
            else f"{len(present)} checked, none ignored",
        )
    else:
        out.unmeasurable_check(
            "no measured trace is excluded by .gitignore",
            f"{_bootstrap.MEASURED_TRACES}/ holds no CSV at the time this script ran, so "
            "there is nothing to check. This is not a pass: it means the campaign's "
            "traces were not on disk. Re-run this script once they are.",
        )

    out.rule("4  no absolute path in any file this product ships")
    patterns = (
        os.sep + "home" + os.sep + "claude" + os.sep,
        os.sep + "Users" + os.sep,
        os.sep + "root" + os.sep,
        os.sep + "tmp" + os.sep,
    )
    hits: list[str] = []
    scanned = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative_name = _bootstrap.relative(path)
        if any(part in ARTIFACT_NAMES for part in path.parts):
            continue
        if path.suffix in (".png", ".gz", ".whl"):
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        scanned += 1
        for pattern in patterns:
            if pattern in text:
                hits.append(f"{relative_name}: {pattern!r}")
    out.check(
        "no absolute path in any text file of this product",
        not hits,
        f"{scanned} file(s) scanned" if not hits else f"{len(hits)} hit(s)",
    )
    for hit in hits[:20]:
        out(f"    {hit}")

    out.rule("5  this script added nothing")
    added = snapshot() - entry
    out.check(
        "running this script added no cache directory and no bytecode file",
        not added,
        f"added: {sorted(added)}" if added else "nothing added",
    )
    out(
        "  Run this script twice in a row from a clean tree and the second run must agree "
        "with the first; a check whose own execution can create the condition it tests is "
        "not a check. probe_working_directories.py runs it four times over."
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
