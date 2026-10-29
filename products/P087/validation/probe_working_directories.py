"""Run every validation script and example from five working directories.

Why this is a committed script rather than a one-off shell loop
--------------------------------------------------------------
Four products in one batch of this mission passed every check in their own
build and then failed the release gate, because the gate runs each validation
script with its working directory set to the script's own directory rather than
to the product root. Measured on one of them: exit 0 from the product root,
exit 1 from anywhere else. The build agent, the agent's self-report and the
coordinator's re-measurement all ran the scripts the same way. Three
independent verifications agreeing does not help when they share an assumption,
so the probe is committed and its output is committed with it.

The five directories are the filesystem root, the system temporary directory,
the repository root, ``validation/`` -- which is what the gate uses -- and
``src/``. ``python -I`` is deliberately **not** used: isolated mode strips the
script's own directory from ``sys.path``, so ``import _bootstrap`` fails and the
probe reports a failure that does not exist.

``validate_published_references.py`` exits 0 even though one of its six
external checks fails, because a failing external cross-check is the measured
result and must not stop the gate. That is stated here so a reader of this
file does not read an exit 0 as six passes.

This script is slow: it runs the whole validation suite and every example five
times, two subprocesses at a time on the two cores this product was measured
on. Elapsed times are recorded for information and are explicitly unstable.
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

from refractpath import format_table  # noqa: E402

VALIDATION = ROOT / "validation"
EXAMPLES = ROOT / "examples"

#: Scripts probed: every validation script and every example, but not this one,
#: which would recurse.
SCRIPTS: tuple[tuple[str, Path], ...] = tuple(
    [
        (f"validation/{path.name}", path)
        for path in sorted(VALIDATION.glob("*.py"))
        if path.name not in {"_bootstrap.py", Path(__file__).name}
    ]
    + [
        (f"examples/{path.name}", path)
        for path in sorted(EXAMPLES.glob("*.py"))
        if path.name != "_common.py"
    ]
)

DIRECTORIES: tuple[tuple[str, Path], ...] = (
    ("filesystem root", Path("/")),
    ("temporary directory", Path(tempfile.gettempdir())),
    ("repository root", ROOT),
    ("validation (the gate's cwd)", VALIDATION),
    ("src", ROOT / "src"),
)


def run_one(
    label: str, script: Path, directory_label: str, directory: Path, environment: dict[str, str]
) -> tuple[str, str, int, float, str]:
    """Run ``script`` with ``cwd=directory`` and return its status and elapsed time."""
    started = time.perf_counter()
    completed = subprocess.run(  # fixed argv, no shell
        [sys.executable, str(script)],
        cwd=str(directory),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    return (
        label,
        directory_label,
        int(completed.returncode),
        time.perf_counter() - started,
        completed.stderr.strip()[-300:],
    )


def main() -> int:
    out = Tee("probe_working_directories_output.txt")
    out.rule("refractpath -- every script, from five working directories")
    for line in environment_lines():
        out(line)
    out("")
    out(f"scripts probed     : {len(SCRIPTS)}")
    out(f"directories probed : {len(DIRECTORIES)}")
    for label, directory in DIRECTORIES:
        shown = (
            relative(directory)
            if directory == ROOT or ROOT in directory.parents or ROOT == directory.parent
            else label
        )
        out(f"  {label:<28s} -> {shown}")
    out("")
    out("Note: python -I is NOT used. Isolated mode removes the script's own")
    out("directory from sys.path, so 'import _bootstrap' fails and the probe")
    out("reports a failure that does not exist.")

    environment = subprocess_environment()
    jobs = [
        (label, script, directory_label, directory, environment)
        for label, script in SCRIPTS
        for directory_label, directory in DIRECTORIES
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda job: run_one(*job), jobs))

    by_script: dict[str, dict[str, tuple[int, float, str]]] = {}
    for label, directory_label, status, elapsed, stderr in results:
        by_script.setdefault(label, {})[directory_label] = (status, elapsed, stderr)

    labels = [label for label, _ in DIRECTORIES]
    rows = []
    failures = 0
    for label, _ in SCRIPTS:
        row = [label]
        for directory_label in labels:
            status, _, _ = by_script[label][directory_label]
            row.append(str(status))
            if status != 0:
                failures += 1
        row.append("ok" if all(by_script[label][name][0] == 0 for name in labels) else "FAILED")
        rows.append(row)
    out.rule("exit status by script and working directory")
    out(format_table(["script", *labels, "verdict"], rows))
    out("")
    out(f"non-zero exit statuses : {failures} of {len(jobs)}")
    for label, _ in SCRIPTS:
        for directory_label in labels:
            status, _, stderr = by_script[label][directory_label]
            if status != 0 and stderr:
                out(f"  {label} from {directory_label}: {stderr}")
    out.check("every script exits zero from every working directory", failures == 0)

    out.rule("elapsed time -- UNSTABLE, two subprocesses at a time on two cores")
    out(
        format_table(
            ["script", *labels],
            [
                [label, *[f"{by_script[label][name][1]:.1f}" for name in labels]]
                for label, _ in SCRIPTS
            ],
        )
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
