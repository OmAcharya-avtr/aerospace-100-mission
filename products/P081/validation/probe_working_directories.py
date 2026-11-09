"""Run every validation script from four working directories, including the gate's.

Why this is a committed script rather than a one-off shell loop
--------------------------------------------------------------
Three products in one batch of this mission passed every check in their own
build and then failed the release gate, because the gate runs each validation
script with its working directory set to the script's own directory rather than
to the product root. Measured on one of them: exit 0 from the product root,
exit 1 from anywhere else. The build agent, the agent's self-report and the
coordinator's re-measurement all ran the scripts the same way. Three independent
verifications agreeing does not help when they share an assumption, so the probe
is committed and its output is committed with it.

The four directories are the filesystem root, the system temporary directory,
the repository root, and ``validation/`` -- which is what the gate uses.
``python -I`` is deliberately **not** used: isolated mode strips the script's
own directory from ``sys.path``, so ``import _bootstrap`` fails and the probe
reports a failure that does not exist.

The examples are probed too, from the repository root and from ``examples/``,
which are the two directories they are run from in practice.

This script is slow: it runs the whole validation suite four times and the
examples twice. Jobs are dispatched longest-first over two workers, which is
what keeps it inside the release gate's per-script time limit on two cores.
Elapsed times are recorded for information and are explicitly unstable.
"""

from __future__ import annotations

# See validate_plant.py for why this statement is here and not in _bootstrap.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

from _bootstrap import (  # noqa: E402
    Tee,
    bootstrap,
    environment_lines,
    relative,
    subprocess_environment,
)

ROOT = bootstrap()

from twindiverge.report import format_table  # noqa: E402

VALIDATION = ROOT / "validation"
EXAMPLES = ROOT / "examples"

#: Rough cost of each validation script in seconds, measured during the build on
#: 2026-11-09 and used ONLY to dispatch the longest jobs first. A wrong value
#: here costs wall-clock time and cannot change a verdict.
COST = {
    "validate_approx_audit.py": 48.0,
    "validate_cli.py": 32.0,
    "validate_profile.py": 29.0,
    "validate_examples.py": 26.0,
    "validate_independent_bound.py": 24.0,
    "validate_negative_controls.py": 20.0,
    "validate_learned_model.py": 8.0,
    "validate_attribution.py": 4.0,
    "validate_identifiability.py": 2.0,
    "validate_plant.py": 2.0,
    "worked_example.py": 2.0,
}

#: Validation scripts probed: every one but this, which would recurse, and not
#: the underscore-prefixed support modules, which are imported rather than run.
SCRIPTS: tuple[tuple[str, Path], ...] = tuple(
    (f"validation/{path.name}", path)
    for path in sorted(VALIDATION.glob("*.py"))
    if not path.name.startswith("_") and path.name != Path(__file__).name
)

EXAMPLE_SCRIPTS: tuple[tuple[str, Path], ...] = tuple(
    (f"examples/{path.name}", path)
    for path in sorted(EXAMPLES.glob("*.py"))
    if path.name != "_common.py"
)

DIRECTORIES: tuple[tuple[str, Path], ...] = (
    ("filesystem root", Path("/")),
    ("temporary directory", Path(tempfile.gettempdir())),
    ("repository root", ROOT),
    ("validation (the gate's cwd)", VALIDATION),
)

EXAMPLE_DIRECTORIES: tuple[tuple[str, Path], ...] = (
    ("repository root", ROOT),
    ("examples", EXAMPLES),
)


def run_one(
    label: str,
    script: Path,
    directory_label: str,
    directory: Path,
    environment: dict[str, str],
) -> tuple[str, str, int, float, str]:
    """Run ``script`` with ``cwd=directory`` and return its status and elapsed time."""
    started = time.perf_counter()
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(script)],
        cwd=str(directory),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    detail = completed.stderr.strip()[-300:]
    if not detail:
        # These scripts report their own failed checks on STDOUT and exit 1
        # with an empty stderr. Reporting only stderr printed nothing at all
        # for the one real failure this probe has ever caught, which cost a
        # diagnosis. Fall back to stdout rather than to silence.
        detail = completed.stdout.strip()[-300:]
    return (
        label,
        directory_label,
        int(completed.returncode),
        time.perf_counter() - started,
        detail,
    )


def run_script_everywhere(
    label: str,
    script: Path,
    directories: tuple[tuple[str, Path], ...],
    environment: dict[str, str],
) -> list[tuple[str, str, int, float, str]]:
    """Run one script from every directory, SERIALLY. Different scripts still
    run two at a time.

    A script must never overlap with another copy of ITSELF.
    ``validate_examples.py`` unlinks each shared ``screenshots/*.png`` and
    regenerates it, so a concurrent second copy observes a deleted or
    half-written figure and reports a defect that does not exist. Measured
    2026-11-09 before this change: 1 non-zero exit in 44 runs,
    ``validate_examples.py`` from the repository root, not reproducible by
    running that script alone from that directory.

    This is the probe creating the condition it tests, which the build guide
    names as the class a check must not belong to. Grouping by script removes
    the self-overlap and keeps the wall-clock cost, because the same 44 runs
    are still spread over the same two workers.
    """
    return [
        run_one(label, script, directory_label, directory, environment)
        for directory_label, directory in directories
    ]


def probe(
    out: Tee,
    title: str,
    scripts: tuple[tuple[str, Path], ...],
    directories: tuple[tuple[str, Path], ...],
    environment: dict[str, str],
) -> int:
    """Run every script from every directory and tabulate the exit statuses."""
    out.rule(title)
    groups = sorted(scripts, key=lambda item: -COST.get(Path(item[1]).name, 5.0))
    with ThreadPoolExecutor(max_workers=2) as pool:
        grouped = list(pool.map(
            lambda item: run_script_everywhere(
                item[0], item[1], directories, environment
            ),
            groups,
        ))
    results = [entry for group in grouped for entry in group]

    by_script: dict[str, dict[str, tuple[int, float, str]]] = {}
    for label, directory_label, status, elapsed, stderr in results:
        by_script.setdefault(label, {})[directory_label] = (status, elapsed, stderr)

    labels = [label for label, _ in directories]
    rows = []
    failures = 0
    for label, _ in scripts:
        row = [label]
        for directory_label in labels:
            status, _, _ = by_script[label][directory_label]
            row.append(str(status))
            if status != 0:
                failures += 1
        row.append(
            "ok" if all(by_script[label][name][0] == 0 for name in labels) else "FAILED"
        )
        rows.append(row)
    out(format_table(["script", *labels, "verdict"], rows))
    out("")
    out(f"non-zero exit statuses : {failures} of {len(results)}")
    for label, _ in scripts:
        for directory_label in labels:
            status, _, stderr = by_script[label][directory_label]
            if status != 0 and stderr:
                out(f"  {label} from {directory_label}: {stderr}")
    out("")
    out("elapsed seconds -- UNSTABLE, two subprocesses at a time on two cores")
    out(format_table(
        ["script", *labels],
        [
            [label, *[f"{by_script[label][name][1]:.1f}" for name in labels]]
            for label, _ in scripts
        ],
    ))
    return failures


def main() -> int:
    out = Tee("probe_working_directories_output.txt")
    started = time.perf_counter()
    out.rule("twindiverge -- every script, from every working directory that matters")
    for line in environment_lines():
        out(line)
    out("")
    out(f"validation scripts probed : {len(SCRIPTS)}")
    out(f"example scripts probed    : {len(EXAMPLE_SCRIPTS)}")
    for label, directory in DIRECTORIES:
        shown = (
            relative(directory)
            if directory == ROOT or ROOT in directory.parents
            else label
        )
        out(f"  {label:<28s} -> {shown}")
    out("")
    out("Note: python -I is NOT used. Isolated mode removes the script's own")
    out("directory from sys.path, so 'import _bootstrap' fails and the probe")
    out("reports a failure that does not exist.")

    environment = subprocess_environment()
    failures = probe(
        out,
        "validation scripts, from four working directories",
        SCRIPTS,
        DIRECTORIES,
        environment,
    )
    failures += probe(
        out,
        "examples, from two working directories",
        EXAMPLE_SCRIPTS,
        EXAMPLE_DIRECTORIES,
        environment,
    )
    out("")
    out(f"total wall clock : {time.perf_counter() - started:.1f} s")
    out("")
    out.check(
        "every script exits zero from every working directory probed",
        failures == 0,
        f"{failures} non-zero exit statuses in total",
    )
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
