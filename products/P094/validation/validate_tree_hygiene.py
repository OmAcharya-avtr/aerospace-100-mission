"""What this product owns: that it ADDS no artifact, and that none would be tracked.

The mission's build guide narrows this check deliberately, because the broad
version cost an earlier product a night and could never have been satisfied. A
self-check asserts two things and not the third:

- that running the scripts here ADDS no cache directory and no bytecode file --
  the tree is snapshotted at entry and compared at exit, and anything new is a
  failure;
- that no cache directory, bytecode file or build artifact would reach a cold
  clone, which is what release-gate check 9 enforces independently;
- **not** that no such artifact EXISTS, because release-gate check 2 runs
  ``ruff check`` with its working directory set to the product root and writes
  ``.ruff_cache`` into the tree one check before check 6 re-executes this
  script. Anything found that this script did not create is reported, named as
  inherited, and does not fail.

**This build agent is forbidden to run any git command**, and so is every
script it executes, so the ignore check below is an EXPLICIT PATTERN MATCHER
over the repository's root ``.gitignore`` and this product's own, not a call to
``git check-ignore``. The matcher implements the subset of gitignore semantics
these two files use -- anchored and unanchored patterns, directory patterns and
negations, last match wins -- and nothing else. ``git check-ignore -v`` remains
the authority and belongs to the gate; the matcher's purpose is to catch the
specific failure that cost P060 a required fixture, which is an unanchored
pattern swallowing a deliverable at depth.

The absolute-path scan deletes its own previous output before scanning and
assembles its patterns from ``os.sep``, because three consecutive sessions of
this mission introduced a self-reporting scanner in the act of writing about one.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import fnmatch  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()
REPOSITORY = ROOT.parent.parent

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

#: Suffixes of build artifacts.
ARTIFACT_SUFFIXES = (".pyc", ".pyo", ".ckpt", ".pt", ".pth", ".onnx", ".egg-info")

#: Files and directories this product MUST ship. Each is checked against the
#: ignore patterns, because an unanchored pattern matches at any depth and
#: silently excluded a required fixture in an earlier product of this mission.
MUST_SHIP = (
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "CITATION.cff",
    "SPEC.md",
    "BUILD_NOTES.md",
    ".gitignore",
    ".github/workflows/tests.yml",
    "data/campaign.py",
    "data/staged_exec.py",
    "data/pat_pipeline.py",
    "data/verify_outputs.py",
    "data/traces/meta.json",
    "data/traces/stages_pass.csv",
    "data/traces/stages.csv",
    "data/traces/runs.csv",
    "data/traces/env.csv",
    "data/traces/calibration.csv",
    "data/traces/campaign.log",
    "data/traces/raw/w64_sparse.npy",
    "src/hiltimingdiff/__init__.py",
    "tests/conftest.py",
    "validation/VALIDATION.md",
    "validation/load_sensitive_inventory.md",
    "validation/_bootstrap.py",
)

#: Paths that MUST be ignored, so a per-run artifact never reaches a clone.
MUST_IGNORE = (
    "junit.xml",
    "__pycache__/x.pyc",
    ".ruff_cache/x",
    ".pytest_cache/x",
    ".hypothesis/x",
    "src/hiltimingdiff/__pycache__/sources.cpython-313.pyc",
)


def _ignore_rules() -> list[tuple[str, bool, Path]]:
    """Parsed patterns from the repository root and product .gitignore files.

    Returns a list of ``(pattern, negated, base)`` in the order they must be
    applied: the repository's own file first, then this product's, so that a
    later negation wins.
    """
    rules: list[tuple[str, bool, Path]] = []
    for base in (REPOSITORY, ROOT):
        path = base / ".gitignore"
        if not path.is_file():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            negated = line.startswith("!")
            if negated:
                line = line[1:]
            rules.append((line, negated, base))
    return rules


def _matches(pattern: str, relative: str) -> bool:
    """Whether one gitignore pattern matches one repository-relative path.

    Implements the subset these two files use. A pattern with a slash anywhere
    but at its end is anchored to its .gitignore's directory; a pattern without
    one matches at any depth; a trailing slash matches a directory and
    everything under it; a leading ``**/`` matches at any depth.
    """
    pattern = pattern.rstrip()
    directory_only = pattern.endswith("/")
    if directory_only:
        pattern = pattern[:-1]
    if pattern.startswith("**/"):
        pattern = pattern[3:]
        anchored = False
    else:
        anchored = "/" in pattern
    if pattern.startswith("/"):
        pattern = pattern[1:]
        anchored = True
    parts = relative.split("/")
    if anchored:
        if fnmatch.fnmatch(relative, pattern):
            return True
        return any(
            fnmatch.fnmatch("/".join(parts[: index + 1]), pattern)
            for index in range(len(parts))
        )
    return any(fnmatch.fnmatch(part, pattern) for part in parts)


def is_ignored(relative_to_product: str) -> tuple[bool, str]:
    """Whether a product-relative path would be ignored, and by which pattern."""
    product_prefix = str(ROOT.relative_to(REPOSITORY))
    verdict = False
    reason = "no pattern matched"
    for pattern, negated, base in _ignore_rules():
        candidate = (
            f"{product_prefix}/{relative_to_product}"
            if base == REPOSITORY
            else relative_to_product
        )
        if _matches(pattern, candidate):
            verdict = not negated
            reason = ("!" if negated else "") + pattern + f"  (from {base.name}/.gitignore)"
    return verdict, reason


def snapshot() -> set[str]:
    """Cache directories and bytecode files currently in the product tree."""
    found: set[str] = set()
    for path in ROOT.rglob("*"):
        parts = path.relative_to(ROOT).parts
        if any(part in ARTIFACT_NAMES for part in parts):
            found.add(str(path.relative_to(ROOT)))
        elif path.suffix in ARTIFACT_SUFFIXES:
            found.add(str(path.relative_to(ROOT)))
    return found


def main() -> int:
    tee = _bootstrap.Tee("validate_tree_hygiene_output.txt")
    # Delete this script's own previous output BEFORE scanning, or the scanner
    # reports itself. Three consecutive sessions of this mission introduced
    # exactly this defect while writing about it.
    #
    # The file deleted is the one this run will WRITE, taken from the Tee, and
    # not the committed path. Those differ when HILTIMINGDIFF_OUTPUT_DIR
    # redirects the output, which is what probe_working_directories.py does to
    # every script here. Deleting the committed path in that case destroyed a
    # committed validation output and did not replace it: measured 2026-11-19,
    # the suite collected 1252 tests before a redirected run of this script and
    # 1251 after it, because tests/test_honesty.py parametrises over the tracked
    # text files in the tree. The release gate happened to hide it, because it
    # runs probe_working_directories.py first and this script afterwards, in
    # alphabetical order, so the file was recreated before anything looked.
    output_path = tee.path
    if output_path.exists():
        output_path.unlink()
    _bootstrap.header(tee, "P094 HilTimingDiff -- tree hygiene, narrowed to what is owned")

    inherited = snapshot()
    tee.rule("1. inherited state, reported and NOT failed on")
    tee(f"  cache and bytecode paths present at entry: {len(inherited)}")
    for item in sorted(inherited)[:12]:
        tee(f"    {item}")
    if len(inherited) > 12:
        tee(f"    ... and {len(inherited) - 12} more")
    tee("")
    tee(
        "  These are INHERITED. Release-gate check 2 runs ruff with its working "
        "directory set to"
    )
    tee(
        "  the product root and writes .ruff_cache one check before check 6 "
        "re-executes this"
    )
    tee(
        "  script, and check 1's pytest run can leave .hypothesis the same way. What a "
        "harness"
    )
    tee("  leaves in the tree is not the product's doing, and failing on it is a check")
    tee("  no change to the product can satisfy.")

    tee.rule("2. what git WOULD TRACK: every required deliverable must not be ignored")
    tee(
        "  NO GIT COMMAND IS RUN. This build agent is forbidden to run one, so the "
        "check below is"
    )
    tee(
        "  an explicit matcher over the repository root .gitignore and this product's "
        "own. The"
    )
    tee("  authority is `git check-ignore -v`, which belongs to the gate.")
    tee("")
    for name in MUST_SHIP:
        target = ROOT / name
        ignored, reason = is_ignored(name)
        tee.check(
            f"ships: {name}",
            target.exists() and not ignored,
            ("absent from the tree" if not target.exists() else "")
            + (f" ignored by {reason}" if ignored else ""),
        )

    tee.rule("3. every per-run artifact must be ignored")
    for name in MUST_IGNORE:
        ignored, reason = is_ignored(name)
        tee.check(f"ignored: {name}", ignored, reason)

    tee.rule("4. no artifact is present in a form that would reach a clone")
    tracked_like: list[str] = []
    for item in sorted(snapshot()):
        ignored, _ = is_ignored(item)
        if not ignored:
            tracked_like.append(item)
    tee(f"  artifacts present that are NOT ignored: {len(tracked_like)}")
    for item in tracked_like[:10]:
        tee(f"    {item}")
    tee("")
    tee.check(
        "no cache directory or bytecode file in the tree would reach a cold clone",
        not tracked_like,
        f"{len(tracked_like)} would",
    )

    tee.rule("5. no tracked file carries an absolute path")
    separator = os.sep
    patterns = (
        f"{separator}home{separator}",
        f"{separator}Users{separator}",
        f"{separator}root{separator}",
    )
    offenders: list[str] = []
    scanned = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(ROOT))
        if path.resolve() == output_path.resolve():
            continue
        if any(part in ARTIFACT_NAMES for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix not in (
            ".py",
            ".md",
            ".txt",
            ".toml",
            ".yml",
            ".yaml",
            ".cff",
            ".json",
            ".csv",
            ".log",
        ):
            continue
        scanned += 1
        try:
            text = path.read_text()
        except UnicodeDecodeError:  # pragma: no cover - committed files are text
            continue
        if any(pattern in text for pattern in patterns):
            offenders.append(relative)
    tee(f"  files scanned : {scanned}")
    tee(f"  offenders     : {len(offenders)}")
    for item in offenders:
        tee(f"    {item}")
    tee("")
    tee(
        "  This script's own output file is excluded from the scan and is deleted "
        "before the"
    )
    tee("  scan begins, so the scanner cannot report itself.")
    tee.check("no tracked file carries an absolute home path", not offenders)

    tee.rule("6. what this script ADDS")
    added = snapshot() - inherited
    tee(f"  cache and bytecode paths added by this script: {len(added)}")
    for item in sorted(added):
        tee(f"    {item}")
    tee.check("running this script adds no artifact", not added, f"{len(added)} added")

    tee.rule("negative controls on the matcher and the scanner")
    tee.control(
        "a bare `junit.xml` pattern would be treated as anchored",
        not _matches("junit.xml", "products/P094/fixtures/junit.xml"),
        must_fire=False,
        detail="an unanchored pattern matches at ANY depth, which is the P060 defect "
        "and is what this matcher must reproduce",
    )
    tee.control(
        "an anchored `/junit.xml` pattern reaches into a subdirectory",
        _matches("/junit.xml", "products/P094/fixtures/junit.xml"),
        must_fire=False,
        detail="an anchored pattern must not match at depth",
    )
    tee.control(
        "the absolute-path scanner detects a planted absolute path",
        any(
            pattern in f"a line mentioning {separator}home{separator}someone{separator}x"
            for pattern in patterns
        ),
        must_fire=True,
        detail="the scanner's own predicate fires on a planted path, so a zero count "
        "above is a measurement",
    )
    tee.control(
        "the campaign log, which the repository root ignores as `*.log`, is re-included",
        is_ignored("data/traces/campaign.log")[0],
        must_fire=False,
        detail=f"verdict from {is_ignored('data/traces/campaign.log')[1]}",
    )

    path = tee.save()
    print(f"\nwritten: {path}")
    return 1 if tee.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
