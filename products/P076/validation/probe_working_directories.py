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

SKIP = {"_bootstrap.py", "_common.py", "probe_working_directories.py"}


def directories() -> list[tuple[str, Path]]:
    """``(label, path)`` for the five working directories, labels not paths."""
    return [
        ("filesystem root", Path(sys.prefix).anchor and Path("/") or Path("/")),
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

    found = scripts()
    places = directories()
    tee("")
    tee(f"  scripts                            : {len(found)}")
    tee(f"  working directories                : {len(places)}")
    tee(f"  subprocess invocations             : {len(found) * len(places)}")
    tee("")
    tee("  working directories, by label:")
    for label, _ in places:
        tee(f"    - {label}")

    tee.rule("results")
    header_row = f"  {'script':<38s}" + "".join(
        f"{label.split()[0][:6]:>9s}" for label, _ in places
    )
    tee(header_row)
    tee(f"  {'-' * 38}" + "".join(f"{'-' * 9}" for _ in places))

    failures: list[tuple[str, str, int, str]] = []
    for script in found:
        row = f"  {relative(script):<38s}"
        for label, place in places:
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
        "every script exits zero from every working directory",
        not failures,
        f"{len(failures)} of {len(found) * len(places)} invocations exited non-zero",
    )

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

    tee.summarise_failures()
    tee("")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
