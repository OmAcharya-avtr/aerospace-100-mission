"""Every script in this directory, run from five different working directories.

The release gate runs each validation script with its working directory set to
the script's own directory rather than to the product root. Products in earlier
batches of this mission passed every local check and failed there, because the
build agent, the agent's own self-report and the coordinator's re-measurement
all ran the scripts the same way. Three independent verifications agreeing does
not help when they share an assumption.

Each script is therefore spawned from the filesystem root, the system temporary
directory, the product root, ``validation/`` and ``src/``, and all five must
exit zero. Each child gets a temporary ``HOME`` and ``MPLCONFIGDIR`` outside the
repository and ``PYTHONDONTWRITEBYTECODE=1``.

This script does not run itself, to avoid unbounded recursion, and it does not
re-run ``validate_controls.py``'s subprocess audit from five directories either:
that script spawns a full pytest run of its own and running it five times would
dominate the whole validation budget. Its own working-directory independence is
established the same way as the others' -- it imports ``_bootstrap`` and calls
``bootstrap()`` as its first statement -- and it IS included here, once, from
``validation/``, which is the directory the gate uses.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from _bootstrap import ROOT, Tee, bootstrap, environment_lines, subprocess_environment

bootstrap()

#: Scripts run from all five directories.
FAST = (
    "validate_observability.py",
    "validate_recovery_grid.py",
    "worked_example.py",
)

#: Scripts whose cost makes five runs unreasonable on two shared cores. Each is
#: run once, from ``validation/``, which is the directory the release gate uses.
SLOW = (
    "validate_recursive_agreement.py",
    "validate_unobservable.py",
    "validate_learned.py",
    "validate_controls.py",
    "validate_cli.py",
)


def main() -> int:
    out = Tee("probe_working_directories_output.txt")
    out("WORKING-DIRECTORY PROBE")
    out("")
    for line in environment_lines():
        out(f"  {line}")

    temporary = Path(tempfile.gettempdir())
    directories = [
        ("filesystem root", Path("/")),
        ("system temporary", temporary),
        ("product root", ROOT),
        ("validation", ROOT / "validation"),
        ("src", ROOT / "src"),
    ]
    environment = subprocess_environment()

    out.rule("every script from five working directories")
    out("")
    out(
        "  Directory names are printed as labels rather than as paths, because no "
        "committed file in this repository may carry an absolute path."
    )
    out("")
    out(f"{'script':<36s} " + " ".join(f"{label[:16]:>17s}" for label, _ in directories))
    for script in FAST:
        statuses = []
        for _, directory in directories:
            completed = subprocess.run(
                [sys.executable, str(ROOT / "validation" / script)],
                cwd=str(directory),
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            statuses.append(completed.returncode)
        out(f"{script:<36s} " + " ".join(f"{status:>17d}" for status in statuses))
        out.check(
            f"{script} exits 0 from all five directories",
            all(status == 0 for status in statuses),
            f"statuses {statuses}",
        )

    out.rule("the expensive scripts, once each, from the directory the gate uses")
    out("")
    out(
        "  Each of these spawns work of its own -- a full pytest run, a learned-model "
        "fit, or eleven command-line subprocesses -- and running them from five "
        "directories would dominate the validation budget on two shared cores. Each "
        "calls bootstrap() as its first statement exactly as the scripts above do."
    )
    out("")
    out(f"{'script':<36s} {'exit from validation/':>22s}")
    for script in SLOW:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "validation" / script)],
            cwd=str(ROOT / "validation"),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        out(f"{script:<36s} {completed.returncode:>22d}")
        out.check(
            f"{script} exits 0 from validation/",
            completed.returncode == 0,
            f"status {completed.returncode}",
        )

    out.rule("the output files all landed in validation/, not beside the caller")
    expected = [Path(script).stem + "_output.txt" for script in FAST + SLOW]
    missing = [name for name in expected if not (ROOT / "validation" / name).exists()]
    out(f"  expected output files : {len(expected)}")
    out(f"  missing               : {missing if missing else 'none'}")
    stray = sorted(
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*_output.txt")
        if path.parent != ROOT / "validation"
    )
    out(f"  stray output files    : {stray if stray else 'none'}")
    out.check(
        "every script wrote its output into validation/",
        not missing and not stray,
        f"{len(expected) - len(missing)} of {len(expected)} present, {len(stray)} stray",
    )

    out.rule("no absolute path in any committed text file")
    out(
        "  The patterns are assembled from os.sep rather than written out, and this "
        "script's own previous output is deleted before the scan, because three "
        "consecutive sessions of this mission introduced exactly that defect in the "
        "act of writing about it."
    )
    import os

    separator = os.sep
    home = separator + "home" + separator
    root_users = separator + "root" + separator
    patterns = (home, root_users, separator + "Users" + separator, separator + "tmp" + separator)
    own_output = ROOT / "validation" / "probe_working_directories_output.txt"
    if own_output.exists():
        own_output.unlink()
    offenders: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(ROOT)
        if any(
            part in {"__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis", ".git"}
            for part in relative_path.parts
        ):
            continue
        if path.suffix in {".png", ".pyc"}:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:  # pragma: no cover - no binary text files expected
            continue
        for pattern in patterns:
            if pattern in text:
                offenders.append(f"{relative_path} contains {pattern!r}")
                break
    out("")
    out(f"  files scanned : "
        f"{sum(1 for p in ROOT.rglob('*') if p.is_file() and p.suffix not in {'.png', '.pyc'})}")
    out(f"  offenders     : {len(offenders)}")
    for entry in offenders:
        out(f"    {entry}")
    out.check(
        "no committed text file contains an absolute container or host path",
        not offenders,
        f"{len(offenders)} offenders",
    )

    path = out.save()
    print(f"\nwritten to {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
