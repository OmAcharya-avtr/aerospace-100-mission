"""Run every validation script from five working directories and record exit status.

Why this is a committed script rather than a shell loop
------------------------------------------------------

Three products in one batch of this mission passed every check in their own
build and then failed the release gate, because the gate runs each validation
script with its working directory set to the script's own directory rather
than to the product root. Measured on one of them: exit 0 from the product
root, exit 1 from anywhere else. The build agent, the agent's self-report and
the coordinator's re-measurement all ran the scripts the same way. Three
independent verifications agreeing does not help when they share an
assumption, so the probe is committed and its output with it.

The five directories are the filesystem root, the system temporary directory,
the repository root, ``validation/`` (which is what the gate uses) and
``src/``. ``python -I`` is deliberately **not** used: isolated mode strips the
script's own directory from ``sys.path``, so ``import _bootstrap`` fails and
the probe reports a failure that does not exist.

This script runs the whole validation suite five times, two subprocesses at a
time on the two cores this product was measured on. Elapsed times are recorded
for information and are explicitly unstable.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from _bootstrap import Tee, bootstrap, environment_lines, relative, subprocess_environment

ROOT = bootstrap()

from tmtcdiff.report import format_table  # noqa: E402

VALIDATION = ROOT / "validation"

#: Every validation script and the worked example, but not this one, which
#: would recurse.
SCRIPTS = sorted(
    path.name
    for path in VALIDATION.glob("*.py")
    if path.name not in {"_bootstrap.py", Path(__file__).name}
)

DIRECTORIES: list[tuple[str, Path]] = [
    ("filesystem root", Path("/")),
    ("temporary directory", Path(tempfile.gettempdir())),
    ("repository root", ROOT),
    ("validation (the gate's cwd)", VALIDATION),
    ("src", ROOT / "src"),
]


def run_one(script: str, label: str, directory: Path) -> tuple[str, str, int, float]:
    """Run ``script`` with ``cwd=directory`` and return its exit status and elapsed time."""
    started = time.perf_counter()
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(VALIDATION / script)],
        cwd=str(directory),
        capture_output=True,
        text=True,
        env=subprocess_environment(),
        check=False,
    )
    return (script, label, int(completed.returncode), time.perf_counter() - started)


def main() -> int:
    out = Tee("probe_working_directories_output.txt")
    out.rule("tmtcdiff -- every validation script, from five working directories")
    for line in environment_lines():
        out(line)
    out("")
    out(f"scripts probed     : {len(SCRIPTS)}")
    out(f"directories probed : {len(DIRECTORIES)}")
    for label, directory in DIRECTORIES:
        shown = relative(directory) if directory == ROOT or ROOT in directory.parents else label
        out(f"  {label:<28s} -> {shown}")
    out("")
    out("Note: python -I is NOT used. Isolated mode removes the script's own")
    out("directory from sys.path, so 'import _bootstrap' fails and the probe")
    out("reports a failure that does not exist.")

    jobs = [
        (script, label, directory)
        for script in SCRIPTS
        for label, directory in DIRECTORIES
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda job: run_one(*job), jobs))

    by_script: dict[str, dict[str, tuple[int, float]]] = {}
    for script, label, status, elapsed in results:
        by_script.setdefault(script, {})[label] = (status, elapsed)

    labels = [label for label, _ in DIRECTORIES]
    rows = []
    failures = 0
    for script in SCRIPTS:
        row: list[object] = [script]
        for label in labels:
            status, _ = by_script[script][label]
            row.append(str(status))
            if status != 0:
                failures += 1
        row.append(
            "ok" if all(by_script[script][label][0] == 0 for label in labels) else "FAILED"
        )
        rows.append(row)
    out.rule("exit status by script and working directory")
    out(format_table(["script", *labels, "verdict"], rows))
    out("")
    out(f"non-zero exit statuses : {failures} of {len(jobs)}")
    out.check("every script exits zero from every working directory", failures == 0)

    out.rule("elapsed time -- UNSTABLE, two subprocesses at a time on two cores")
    timing_rows = [
        [script, *[f"{by_script[script][label][1]:.1f}" for label in labels]]
        for script in SCRIPTS
    ]
    out(format_table(["script", *labels], timing_rows))
    out("")
    out("No conclusion in this repository rests on a timing or on a ratio of timings.")

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
