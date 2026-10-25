"""Run every validation script and example from five working directories.

Four products in Batch 07 of this mission passed every check in their own
build and then failed the release gate, which runs each validation script with
its working directory set to the script's own directory rather than to the
product root. The build agent, its self-report and the coordinator's
re-measurement all ran from the product root -- three independent verifications
that shared an assumption instead of a method.

This script removes the assumption. Every script in ``validation/`` and every
script in ``examples/`` is executed as a subprocess from the filesystem root,
the temporary directory, the product root, ``validation/`` and ``src/``, and
every run must exit zero.

``python -I`` is NOT used for the probe. ``-I`` strips the script's own
directory from ``sys.path``, which makes ``import _bootstrap`` and
``import _common`` fail and produces a false failure that looks exactly like
the defect being probed for.

``audit_approx_tolerances.py`` is excluded: it runs the whole test suite twice
and takes about thirty seconds, so including it would multiply to five minutes
of the probe's own budget for no additional coverage -- it is run once, from
``validation/``, by the CI workflow and by the release gate. The exclusion is
recorded in the output.

The output records working directories as labels, never as absolute paths, so
no committed file in this repository carries one.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from _bootstrap import ROOT, Tee, bootstrap, header, relative

bootstrap()

SKIP = {
    "_bootstrap.py",
    "_common.py",
    "probe_working_directories.py",
    "audit_approx_tolerances.py",
}

#: Scripts run from TWO working directories instead of five, with the reason
#: stated in the output. Both are subprocess harnesses that set ``cwd`` to the
#: repository root explicitly for every child they spawn, so the only
#: path-dependence they can have is their own bootstrap, and the product root
#: against ``validation/`` already distinguishes that -- the latter being the
#: directory the release gate uses and the one the Batch 07 defect was found
#: in. The measured cost of running them from all five is 110 s of this
#: script's budget for no additional coverage.
HEAVY = {"validate_cli.py"}


def directories() -> list[tuple[str, Path]]:
    """``(label, path)`` for the five working directories, labels not paths."""
    return [
        ("filesystem root", Path("/")),
        ("temporary directory", Path(tempfile.gettempdir())),
        ("product root", ROOT),
        ("validation directory", ROOT / "validation"),
        ("source directory", ROOT / "src"),
    ]


def scripts() -> list[Path]:
    """Every runnable script in ``validation/`` and ``examples/``, sorted."""
    found: list[Path] = []
    for folder in ("validation", "examples"):
        for path in sorted((ROOT / folder).glob("*.py")):
            if path.name not in SKIP:
                found.append(path)
    return found


def main() -> int:
    tee = Tee("probe_working_directories_output.txt")
    header(tee, "every validation script and example, run from five working directories")
    tee("")
    tee("  python -I is deliberately NOT used: it strips the script's own")
    tee("  directory from sys.path and makes 'import _bootstrap' fail, which is a")
    tee("  false failure indistinguishable from the defect being probed for.")
    tee("")
    tee("  audit_approx_tolerances.py is excluded: it runs the whole test suite")
    tee("  twice, taking 31 s measured, so five invocations would cost two and a")
    tee("  half minutes of this probe's budget for no extra coverage. It is run")
    tee("  once, from validation/, by the CI workflow and by the release gate.")
    tee("")
    tee("  validate_cli.py is run from TWO directories rather than five, for the")
    tee("  same reason: it spawns 22 subprocesses and takes 22 s measured, and it")
    tee("  sets cwd to the repository root explicitly for every child, so its")
    tee("  only path-dependence is its own bootstrap. The product root against")
    tee("  validation/ distinguishes that, and validation/ is the directory the")
    tee("  release gate uses.")

    found = scripts()
    places = directories()
    tee("")
    tee(f"  scripts                            : {len(found)}")
    tee(f"  working directories                : {len(places)}")
    heavy = [p for p in found if p.name in HEAVY]
    invocations = (len(found) - len(heavy)) * len(places) + len(heavy) * 2
    tee(f"  scripts run from all five          : {len(found) - len(heavy)}")
    tee(f"  scripts run from two               : {len(heavy)}")
    tee(f"  subprocess invocations             : {invocations}")
    tee("")
    tee("  working directories, by label:")
    for label, _ in places:
        tee(f"    - {label}")

    tee.rule("results")
    header_row = f"  {'script':<42s}" + "".join(
        f"{label.split()[0][:6]:>9s}" for label, _ in places
    )
    tee(header_row)
    tee(f"  {'-' * 42}" + "".join(f"{'-' * 9}" for _ in places))

    failures: list[tuple[str, str, int, str]] = []
    for script in found:
        row = f"  {relative(script):<42s}"
        chosen = (
            [(label, place) for label, place in places if "product root" in label
             or "validation" in label]
            if script.name in HEAVY
            else places
        )
        for label, place in places:
            if (label, place) not in chosen:
                row += f"{'-':>9s}"
                continue
            result = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                cwd=str(place),
                env={"MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin"},
                check=False,
                timeout=900,
            )
            row += f"{result.returncode:>9d}"
            if result.returncode != 0:
                tail = (result.stderr or "").strip().splitlines()
                failures.append(
                    (relative(script), label, result.returncode, tail[-1] if tail else "")
                )
        tee(row)

    tee("")
    if failures:
        tee(f"  NON-ZERO EXITS: {len(failures)}")
        for name, label, code, message in failures:
            tee(f"    {name} from the {label}: exit {code}")
            tee(f"      {message}")
    else:
        tee("  NON-ZERO EXITS: none")
    tee.check(
        "every script exits zero from every working directory it was run from",
        not failures,
        f"{len(failures)} of {invocations} invocations exited non-zero",
    )
    tee("  A dash in the table marks an invocation deliberately not made; see the")
    tee("  HEAVY note above.")

    tee.rule("absolute paths in committed files")
    tee("  Release-gate check 8 fails a whole batch on an absolute")
    tee("  home-directory path in any tracked file. The scan below covers every")
    tee("  text file in this product, not just the code.")
    tee("")
    # Assembled from fragments rather than written out, so that this file does
    # not match its own patterns. Two consecutive sessions of this mission
    # introduced the absolute-path defect in the act of writing about it.
    patterns = (
        "/" + "home" + "/",
        "/" + "Users" + "/",
        "/" + "tmp" + "/",
        "/" + "root" + "/",
        "C:" + chr(92),
    )
    offenders: list[tuple[str, str, int]] = []
    scanned = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(
            part in {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis"}
            for part in path.parts
        ):
            continue
        if path.suffix in {".png", ".pyc"}:
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        scanned += 1
        for number, line in enumerate(text.splitlines(), start=1):
            for pattern in patterns:
                if pattern in line:
                    offenders.append((relative(path), pattern, number))
    tee(f"  text files scanned                 : {scanned}")
    tee(f"  patterns                           : {len(patterns)} absolute-path prefixes")
    tee(f"  matches                            : {len(offenders)}")
    for name, pattern, number in offenders:
        tee(f"    {name}:{number} contains {pattern!r}")
    tee.check("no committed text file carries an absolute path", not offenders)

    tee.rule("every example wrote its screenshot")
    tee("")
    expected = sorted(
        p.stem for p in (ROOT / "examples").glob("*.py") if p.name != "_common.py"
    )
    present = sorted(p.stem for p in (ROOT / "screenshots").glob("*.png"))
    tee(f"  examples                           : {len(expected)}")
    tee(f"  screenshots present                : {len(present)}")
    missing = sorted(set(expected) - set(present))
    extra = sorted(set(present) - set(expected))
    for name in expected:
        path = ROOT / "screenshots" / f"{name}.png"
        size = path.stat().st_size if path.exists() else 0
        tee(f"    {name:<34s}{size:>10d} bytes")
    if missing:
        tee(f"  MISSING: {missing}")
    if extra:
        tee(f"  unexpected: {extra}")
    tee.check("every example produced a screenshot", not missing, f"missing {missing}")
    tee.check(
        "every screenshot is non-empty",
        all((ROOT / "screenshots" / f"{n}.png").stat().st_size > 1000 for n in present),
    )

    tee.rule("gitignore does not exclude a committed deliverable")
    tee("")
    tee("  A .gitignore pattern with no leading slash matches at ANY depth: a")
    tee("  bare junit.xml in a product .gitignore silently excluded a REQUIRED")
    tee("  committed fixture in P060 of this mission, and a cold clone then")
    tee("  failed with 17 failures and 9 errors while every local check passed.")
    tee("")
    tee("  git is not run here -- this build is forbidden to run it -- so the")
    tee("  patterns are matched directly against the deliverable paths, with the")
    tee("  anchoring rule applied explicitly. A pattern beginning with '!' is a")
    tee("  NEGATION and re-includes the file: 'git check-ignore' exits 0 when a")
    tee("  negation pattern matches, so reading only its exit status would")
    tee("  report a deliberately re-included file as ignored.")
    tee("")
    ignore_file = ROOT / ".gitignore"
    raw = ignore_file.read_text().splitlines() if ignore_file.exists() else []
    patterns_list = [line.strip() for line in raw if line.strip() and not line.startswith("#")]
    tee(f"  patterns in .gitignore             : {len(patterns_list)}")
    unanchored = [
        p
        for p in patterns_list
        if "/" not in p.rstrip("/") and not p.startswith("!") and not p.startswith("*")
    ]
    tee(f"  unanchored, non-glob patterns      : {len(unanchored)}")
    for pattern in unanchored:
        tee(f"    {pattern}  (matches at any depth)")
    negations = [p for p in patterns_list if p.startswith("!")]
    tee(f"  negation patterns                  : {len(negations)}")
    for pattern in negations:
        tee(f"    {pattern}  (RE-INCLUDES; a check reading only an exit status would")
        tee("       report the file as ignored)")

    deliverables = sorted(
        relative(p)
        for p in ROOT.rglob("*")
        if p.is_file()
        and not any(
            part in {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis"}
            for part in p.parts
        )
    )
    shadowed: list[tuple[str, str]] = []
    for deliverable in deliverables:
        name = Path(deliverable).name
        for pattern in patterns_list:
            if pattern.startswith("!"):
                continue
            bare = pattern.rstrip("/")
            if "/" in bare:
                continue
            if bare == name:
                shadowed.append((deliverable, pattern))
    tee("")
    tee(f"  deliverables scanned               : {len(deliverables)}")
    tee(f"  deliverables matched by an unanchored pattern : {len(shadowed)}")
    for deliverable, pattern in shadowed:
        tee(f"    {deliverable} matched by {pattern!r}")
    tee.check(
        "no committed deliverable is matched by an unanchored .gitignore pattern",
        not shadowed,
        f"{len(shadowed)} shadowed",
    )

    tee.summarise_failures()
    tee("")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
