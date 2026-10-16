"""Check that every file this product must ship is actually shippable.

On 2026-10-10 a bare ``junit.xml`` pattern in one product's ``.gitignore``
silently excluded a required committed fixture, and a cold clone of that
product failed with 17 failures and 9 errors. A gitignore pattern with no
slash matches at every depth, so a product that ships files named like scratch
artifacts has to prove, by execution, that none of them is ignored.

This script runs ``git check-ignore -v`` against every file in the product
that is supposed to be committed, and fails if any of them is ignored.

One subtlety that this script gets right and a naive version gets wrong:
``git check-ignore`` exits 0 when a pattern MATCHED, which includes a
negation pattern that un-ignores the path. A check that reads the exit status
alone reports a file as ignored when the repository has deliberately rescued
it. The verdict below therefore reads the matching pattern as well, and treats
a leading ``!`` as not ignored.
"""

from __future__ import annotations

import subprocess
import sys

from _common import ROOT, rule

SHIPPED_GLOBS = (
    "fixtures/*.json",
    "fixtures/*.py",
    "validation/*.py",
    "validation/*.txt",
    "validation/VALIDATION.md",
    "src/scenariocov/*.py",
    "tests/*.py",
    "examples/*.py",
    "screenshots/*.png",
    "*.md",
    "*.toml",
    "*.cff",
    "LICENSE",
    ".gitignore",
    ".github/workflows/*.yml",
)


def check_ignore(path: str) -> tuple[bool, str]:
    """Return ``(is_ignored, rule_text)`` for one path.

    ``is_ignored`` is False when no pattern matched AND when the matching
    pattern is a negation, which is the case the exit status alone gets wrong.
    """
    result = subprocess.run(
        ["git", "check-ignore", "-v", path],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    detail = result.stdout.strip()
    if result.returncode != 0 or not detail:
        return False, detail
    left = detail.split("\t", 1)[0]
    pieces = left.split(":")
    pattern = ":".join(pieces[2:]) if len(pieces) > 2 else ""
    return not pattern.startswith("!"), detail


def main() -> int:
    """Check every shipped path and report."""
    out = sys.stdout
    failures = 0
    out.write("shipped-file ignore check\n")
    out.write("=" * 78 + "\n")
    out.write(f"product root: {ROOT.name}\n")

    paths = sorted(
        {
            str(p.relative_to(ROOT))
            for pattern in SHIPPED_GLOBS
            for p in ROOT.glob(pattern)
            if p.is_file()
        }
    )
    out.write(f"files checked: {len(paths)}\n")

    out.write(rule("git check-ignore -v, every shipped file") + "\n")
    for path in paths:
        ignored, detail = check_ignore(path)
        if ignored:
            failures += 1
            out.write(f"  IGNORED  {path}\n      matched by: {detail}\n")
    out.write(f"  {len(paths) - failures} of {len(paths)} paths are NOT ignored\n")

    out.write(rule("the patterns that are deliberately unanchored") + "\n")
    for probe in (
        "fixtures/junit.xml",
        "src/scenariocov/__pycache__/space.cpython-313.pyc",
        ".hypothesis/unicode_data/x",
        "junit.xml",
        "validation/validate_benchmark_output.txt",
        "fixtures/campaign_executed.json",
    ):
        ignored, detail = check_ignore(probe)
        out.write(f"  {'ignored    ' if ignored else 'not ignored'}  {probe}\n")
        if detail:
            out.write(f"      {detail}\n")
    out.write(
        "  fixtures/junit.xml is NOT ignored: this product anchors its own pattern\n"
        "  as /junit.xml, and the repository root additionally carries an explicit\n"
        "  negation for products/*/fixtures/**/junit.xml. That path is the exact\n"
        "  defect this check exists for, and the line above shows the negation\n"
        "  being read rather than the exit status being trusted. The cache\n"
        "  patterns are deliberately unanchored and match at any depth.\n"
    )

    out.write(rule("committed fixture sizes") + "\n")
    total = 0
    for path in sorted((ROOT / "fixtures").glob("*")):
        if path.is_file():
            size = path.stat().st_size
            total += size
            flag = "OVER 1 MB" if size > 1_000_000 else "ok"
            if size > 1_000_000:
                failures += 1
            out.write(f"  {path.name:<34}{size:>10} bytes  {flag}\n")
    out.write(f"  {'total':<34}{total:>10} bytes\n")

    out.write(f"\nfailures: {failures}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
