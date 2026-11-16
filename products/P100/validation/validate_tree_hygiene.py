"""What this product owns: that it ADDS no artifact, and that nothing ships it.

The mission's build guide narrows this check deliberately, because the broad
version cost an earlier product a night and could never have been satisfied. A
self-check asserts two things and not the third:

- that running the scripts here ADDS no cache directory and no bytecode file --
  the tree is snapshotted at entry and compared at exit, and anything new is a
  failure;
- that nothing a cold clone would carry is a cache directory, bytecode file or
  build artifact, which is what release-gate check 9 enforces independently;
- **not** that no such artifact EXISTS, because release-gate check 2 runs
  ``ruff check`` with its working directory set to the product root and writes
  ``.ruff_cache`` into the tree one check before check 6 re-executes this
  script. Anything found that this script did not create is reported, named as
  inherited, and does not fail.

The corpus-wide audit campaign is **not** in ``validation/``; it is
``data/repository_audit.py`` and the gate does not run it. This script checks
that both the campaign and the transcript of the run that produced the
committed evidence reach a cold clone.

**No ``git`` command is run here, and that is a deliberate restriction rather
than an oversight.** The build agent for this product is forbidden from running
``git`` at all: the coordinator commits, gates and publishes. So instead of
asking ``git check-ignore``, this script **re-implements the subset of
``.gitignore`` matching the check needs** -- anchoring, directory-only
patterns, basename-at-any-depth patterns, negation and last-match-wins -- and
applies it to both this product's ``.gitignore`` and the repository's, when one
is present. The re-implementation is a subset and its limits are stated in
:data:`IGNORE_SEMANTICS`; the authoritative check is release-gate check 9,
which runs against the real index.

This script is also this product's own worked example of the class it detects:
it scans the tree it writes its own output into. Two things make it a check
rather than a hazard, and both are the product's own prescription. The
absolute-path scan **deletes its own previous output before scanning and
assembles its patterns from ``os.sep``**, because three consecutive sessions of
this mission introduced a self-reporting scanner in the act of writing about
one. And the artifact comparison is a DIFFERENCE between two snapshots, so a
file that was already there cannot be reported as added.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import fnmatch  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

#: Artifact directory names release-gate check 9 fails on.
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

#: Files this product MUST ship. Each is checked against the ignore rules,
#: because a pattern with no slash matches at any depth and silently excluded a
#: required fixture in an earlier product of this mission.
MUST_SHIP = (
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "CITATION.cff",
    "SPEC.md",
    "BUILD_NOTES.md",
    "src/seedaudit/__init__.py",
    "src/seedaudit/power.py",
    "tests/test_power.py",
    "examples/power_table.py",
    "screenshots/detection_power.png",
    "validation/VALIDATION.md",
    "validation/_bootstrap.py",
    "validation/repository_findings.json",
    "validation/validate_power_model_output.txt",
    "data/repository_audit.py",
    "data/repository_audit_output.txt",
)

#: Paths that must be IGNORED, so a cold clone never carries them.
MUST_IGNORE = (
    "junit.xml",
    "__pycache__/x.pyc",
    "src/seedaudit/__pycache__/power.pyc",
    ".ruff_cache/CACHEDIR.TAG",
    ".pytest_cache/CACHEDIR.TAG",
    ".hypothesis/unicode_data/x",
)

#: What the re-implemented matcher covers, and what it does not.
IGNORE_SEMANTICS = (
    "covered: comments and blank lines; a leading '!' negation; a trailing '/' "
    "making a pattern directory-only; a leading '/' anchoring a pattern to the "
    "file's own directory; a pattern with no slash matching a BASENAME at any "
    "depth; a pattern with a slash matched against the relative path; "
    "'**' segments; last match wins. "
    "NOT covered: core.excludesFile, .git/info/exclude, per-directory "
    ".gitignore files below this product, and the index itself. The "
    "authoritative check is release-gate check 9."
)

#: The file this script writes.
OWN_OUTPUT = "validate_tree_hygiene_output.txt"


def read_ignore_rules() -> list[tuple[str, str, bool, bool]]:
    """Collect ignore rules as ``(base, pattern, negated, directory_only)``.

    ``base`` is the directory the pattern is relative to, as a posix string
    relative to this product root, or ``".."`` segments for a pattern that
    comes from a ``.gitignore`` above it.
    """
    rules: list[tuple[str, str, bool, bool]] = []
    sources: list[tuple[Path, str]] = []
    repository = _bootstrap.repository_root()
    if repository is not None and (repository / ".gitignore").is_file():
        sources.append((repository / ".gitignore", os.path.relpath(repository, ROOT)))
    if (ROOT / ".gitignore").is_file():
        sources.append((ROOT / ".gitignore", "."))
    for path, base in sources:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            negated = line.startswith("!")
            if negated:
                line = line[1:]
            directory_only = line.endswith("/")
            if directory_only:
                line = line[:-1]
            if not line:
                continue
            rules.append((base, line, negated, directory_only))
    return rules


def is_ignored(relative: str, rules: list[tuple[str, str, bool, bool]]) -> bool:
    """Whether ``relative`` (relative to this product root) would be ignored.

    Last matching rule wins, which is git's own precedence. A rule from a
    ``.gitignore`` above this product is applied to the path as that file sees
    it, so a repository-level ``**/junit.xml`` reaches this product's
    ``junit.xml``.
    """
    verdict = False
    product_name = ROOT.name
    parent_name = ROOT.parent.name
    for base, pattern, negated, directory_only in rules:
        if base == ".":
            candidate = relative
        else:
            candidate = f"{parent_name}/{product_name}/{relative}"
        parts = candidate.split("/")
        if directory_only and "/" not in relative and not pattern.endswith("*"):
            # A directory-only pattern can still match a parent directory of
            # the path, which is what excludes everything under it.
            pass
        matched = False
        if "/" in pattern.lstrip("/"):
            anchored = pattern.lstrip("/")
            matched = fnmatch.fnmatch(candidate, anchored) or fnmatch.fnmatch(
                candidate, anchored + "/*"
            )
            if anchored.startswith("**/"):
                matched = matched or fnmatch.fnmatch(candidate, anchored[3:]) or any(
                    fnmatch.fnmatch("/".join(parts[index:]), anchored[3:])
                    for index in range(len(parts))
                )
        elif pattern.startswith("/"):
            matched = fnmatch.fnmatch(parts[0], pattern[1:]) and (
                len(parts) == 1 or directory_only or True
            )
            matched = matched and (
                len(parts) == 1 or fnmatch.fnmatch(candidate, pattern[1:] + "/*")
            )
        else:
            matched = any(fnmatch.fnmatch(part, pattern) for part in parts)
            if directory_only:
                matched = any(fnmatch.fnmatch(part, pattern) for part in parts[:-1]) or (
                    fnmatch.fnmatch(parts[-1], pattern)
                )
        if matched:
            verdict = not negated
    return verdict


def snapshot() -> set[str]:
    """Every artifact path currently in the tree, relative to the root."""
    found: set[str] = set()
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT).as_posix()
        if path.is_dir() and path.name in ARTIFACT_NAMES:
            found.add(relative)
        elif path.is_file() and path.suffix in ARTIFACT_SUFFIXES:
            found.add(relative)
    return found


def main() -> int:
    """Check what this product owns."""
    out = _bootstrap.Tee(OWN_OUTPUT)
    out("Tree hygiene -- P100 SeedAudit")
    out("")
    out("this check asserts what the product OWNS: that running it ADDS no cache")
    out("directory or bytecode file, and that nothing a cold clone carries is one.")
    out("It does NOT assert that none EXISTS, because what the release-gate harness")
    out("leaves in the tree before calling this script is not this product's doing.")
    out("")
    out("NO git COMMAND IS RUN HERE. The build agent for this product is forbidden")
    out("from running git; the coordinator commits, gates and publishes. The")
    out(".gitignore semantics the check needs are re-implemented instead, and the")
    out("subset is declared below.")
    out("")
    out(f"fast path (SEEDAUDIT_FAST=1): {_bootstrap.FAST}")
    out(f"product root directory name: {ROOT.name}")

    before = snapshot()
    out.rule("1. INHERITED ARTIFACTS -- reported, named as inherited, not failed")
    out(f"artifacts present at entry: {len(before)}")
    for item in sorted(before):
        out(f"  inherited: {item}")
    if not before:
        out("  none")

    out.rule("2. WHAT A COLD CLONE WOULD CARRY, by the re-implemented ignore rules")
    rules = read_ignore_rules()
    out(f"ignore rules collected: {len(rules)}")
    out("")
    out(IGNORE_SEMANTICS)
    out("")
    shipped = 0
    artifacts_shipped: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if is_ignored(relative, rules):
            continue
        shipped += 1
        if any(part in ARTIFACT_NAMES for part in Path(relative).parts) or Path(
            relative
        ).suffix in ARTIFACT_SUFFIXES:
            artifacts_shipped.append(relative)
    out(f"files a cold clone would carry: {shipped}")
    out(f"of those, cache or build artifacts: {len(artifacts_shipped)}")
    for item in artifacts_shipped:
        out(f"  WOULD SHIP ARTIFACT: {item}")
    out.check(
        "nothing a cold clone would carry is a cache directory, bytecode file or "
        "build artifact",
        not artifacts_shipped,
        f"{len(artifacts_shipped)} offenders",
    )
    out("")
    for item in MUST_SHIP:
        present = (ROOT / item).exists()
        ignored = is_ignored(item, rules)
        out.check(
            f"{item} exists and is NOT ignored",
            present and not ignored,
            f"exists={present}, ignored={ignored}",
        )
    out("")
    for item in MUST_IGNORE:
        out.check(f"{item} IS ignored", is_ignored(item, rules))

    out.rule("3. ABSOLUTE PATHS IN TRACKED TEXT")
    out("this scan deletes its own previous output before scanning and assembles")
    out("its patterns from os.sep, because three consecutive sessions of this")
    out("mission introduced a self-reporting scanner in the act of writing about")
    out("one. Its own output file is additionally excluded by name.")
    out("")
    target = ROOT / "validation" / OWN_OUTPUT
    if target.exists():
        target.unlink()
    separator = os.sep
    patterns = (
        f"{separator}home{separator}",
        f"{separator}Users{separator}",
        f"{separator}root{separator}",
        f"{separator}tmp{separator}",
    )
    out(f"patterns, assembled from os.sep = {separator!r}: {patterns}")
    suffixes = (".py", ".md", ".txt", ".toml", ".cff", ".json", ".yml", ".yaml", ".cfg")
    offenders: list[tuple[str, int, str]] = []
    scanned = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if any(part in ARTIFACT_NAMES for part in path.parts):
            continue
        if path.name == OWN_OUTPUT:
            continue
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for pattern in patterns:
                if pattern in line:
                    offenders.append((path.relative_to(ROOT).as_posix(), number, pattern))
    out(f"files scanned: {scanned}")
    out(f"lines carrying an absolute path pattern: {len(offenders)}")
    for item in offenders[:40]:
        out(f"  {item[0]}:{item[1]} contains {item[2]!r}")
    out.check(
        "no tracked text file carries an absolute home or temp path",
        not offenders,
        f"{len(offenders)} occurrences",
    )

    out.rule("4. WHAT THIS RUN ADDED")
    after = snapshot()
    added = sorted(after - before)
    out(f"artifacts present at exit: {len(after)}")
    out(f"artifacts ADDED by this run: {len(added)}")
    for item in added:
        out(f"  ADDED: {item}")
    if not added:
        out("  none")
    out.check(
        "running this script adds no cache directory and no bytecode file",
        not added,
        f"{len(added)} added",
    )
    out("")
    out("run this script twice in a row from a clean tree and the second run must")
    out("agree with the first. That is the product's own criterion and this script")
    out("is not exempt from it: probe_working_directories.py runs it four times.")
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
