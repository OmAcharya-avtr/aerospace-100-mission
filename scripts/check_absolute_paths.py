#!/usr/bin/env python3
"""Fail if an absolute private path would be committed.

Why this exists
---------------
Release-gate check 8 fails any tracked file containing an absolute path from
the build machine: it is not reproducible for a reader with a cold clone and it
publishes the machine's filesystem layout. The gate is the right place to
ENFORCE that, and the wrong place to DISCOVER it. On 2026-11-01 the gate found
one such literal in ``tracking/SESSION_CHECKPOINT.json`` after a full product
run, and the re-run cost roughly twenty-two minutes on a product that had
otherwise passed. Five sessions had recommended this check before that one paid
for its absence.

This script is the same test, applied to what is about to be committed, in
under a second.

Two deliberate design points
----------------------------
**The patterns are assembled from** ``os.sep`` **and never written as
literals.** A scanner for absolute paths that contains the literal patterns is
itself a tracked file containing absolute paths, so it reports itself -- a
class of defect this mission has now shipped five times, most recently in
P084's own ``validate_cli.py``. Assembling the needles means this file can be
committed to the repository it guards.

**The regex matches the gate's exactly.** A pre-commit check that is stricter
than the gate blocks work the gate would have allowed; one that is looser gives
false confidence. Both require a FOLLOWING path component, so prose that merely
names the shape of a pattern does not trip either. If the gate's regex changes,
change it here in the same commit.

Usage
-----
    python3 scripts/check_absolute_paths.py              # staged changes (hook)
    python3 scripts/check_absolute_paths.py --tracked    # every tracked file
                                                         # (what the gate does)

Install as a hook, from the repository root:

    git config core.hooksPath hooks

Exit 0 = nothing found. Exit 1 = an offending line, printed with its location.
Exit 2 = the check could not run, which is NOT a pass.
"""

from __future__ import annotations

import os
import subprocess
import sys

SEPARATOR = os.sep

#: Assembled, never written out. Equivalent to the release gate's own regex.
PATH_RE = (
    f"({SEPARATOR}Users{SEPARATOR}[A-Za-z0-9._-]+{SEPARATOR}"
    f"|{SEPARATOR}home{SEPARATOR}claude{SEPARATOR}"
    f"|{SEPARATOR}root{SEPARATOR}[A-Za-z0-9._-])"
)

#: Files whose own job is to describe the pattern. The gate excludes its own
#: filename for the same reason; this list must stay in step with it.
SELF_EXCLUDED = ("release_gate.py", os.path.basename(__file__))


def repository_root() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=False,
    )
    return completed.stdout.strip() or None


def offenders(root: str, tracked: bool) -> tuple[list[str], int]:
    """Matching lines, and the git exit status, for staged or tracked content."""
    if tracked:
        argv = ["git", "grep", "-InE", PATH_RE, "HEAD", "--"]
    else:
        argv = ["git", "grep", "--cached", "-InE", PATH_RE, "--"]
    completed = subprocess.run(
        argv, cwd=root, capture_output=True, text=True, check=False,
    )
    # git grep exits 1 when it matches nothing, which is the success case here,
    # and 2 or above on a real error. Anything above 1 is not a pass.
    if completed.returncode > 1:
        return [], completed.returncode
    lines = [
        line for line in completed.stdout.splitlines()
        if line.strip() and not any(name in line for name in SELF_EXCLUDED)
    ]
    return lines, 0


def main(argv: list[str]) -> int:
    tracked = "--tracked" in argv[1:]
    root = repository_root()
    if root is None:
        print("check_absolute_paths: not a git repository; the check did not run.")
        return 2
    lines, git_status = offenders(root, tracked)
    if git_status:
        print(f"check_absolute_paths: git grep exited {git_status}; the check did "
              "not run. This is not a pass.")
        return 2
    scope = "tracked files" if tracked else "staged changes"
    if not lines:
        print(f"check_absolute_paths: no absolute private path in {scope}.")
        return 0
    print(f"check_absolute_paths: {len(lines)} offending line(s) in {scope}.")
    print("An absolute path from this machine is not reproducible for a reader")
    print("with a cold clone, and release-gate check 8 fails the repository for")
    print("it. Print the path relatively instead.")
    print()
    for line in lines[:20]:
        print(f"  {line[:160]}")
    if len(lines) > 20:
        print(f"  ... and {len(lines) - 20} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
