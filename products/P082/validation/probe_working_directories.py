"""Run every script from four working directories, including the release gate's.

Why this is a committed script rather than a one-off shell loop
---------------------------------------------------------------
Three products in one batch of this mission passed every check in their own
build and then failed the release gate, because the gate runs each validation
script with its working directory set to the script's own directory rather than
to the product root. Measured on one of them: exit 0 from the product root, exit
1 from anywhere else. The build agent, the agent's self-report and the
coordinator's re-measurement all ran the scripts the same way. Three independent
verifications agreeing does not help when they share an assumption, so the probe
is committed and its output is committed with it.

The four directories are the filesystem root, the system temporary directory,
the repository root, and ``validation/`` -- which is what the gate uses.
``python -I`` is deliberately NOT used: isolated mode strips the script's own
directory from ``sys.path``, so ``import _bootstrap`` fails and the probe
reports a failure that does not exist.

A check whose own execution can create the condition it tests is not a check
-------------------------------------------------------------------------
This probe dispatches scripts in parallel and therefore belongs to exactly the
class the build guide warns about. Two mechanisms are guarded against, and both
are guarded by the GROUPING rather than by hope.

**A script must never overlap with another copy of ITSELF.** Each script's four
runs are dispatched to one worker and run serially there. A predecessor in this
mission measured 1 non-zero exit in 44 runs from exactly this cause: its
example-validating script unlinked each shared figure and regenerated it, so a
concurrent second copy observed a deleted or half-written file and reported a
defect that did not exist.

**Two DIFFERENT scripts that write the same files must never overlap either.**
``validate_examples.py`` runs every example, and the examples themselves write
the same figures into ``screenshots/``. Grouping by script alone would not stop
``validate_examples.py`` from overlapping ``examples/decomposition.py``, which
is the same defect wearing a different hat. Everything that touches
``screenshots/`` is therefore put in ONE group and run serially: the examples'
own runs and ``validate_examples.py``'s four runs, in one sequence on one
worker.

``validate_approx_audit.py`` is run in its declared PATHS-ONLY mode
-------------------------------------------------------------------
That script runs the whole test suite twice. Four copies is eight suite runs and
about 440 seconds, which would put this probe outside the release gate's
600-second per-script limit. Its paths-only mode does everything that depends on
the caller's working directory -- the bootstrap, the repository root, the scan
of ``tests/``, the read of ``tests/conftest.py`` and the resolution of its
output path -- skips only the two pytest subprocesses, and writes to a
temporary file outside the repository so that a probe run cannot overwrite the
committed evidence. The mode is reported in the table below and the committed
output file is always from a full run.

Elapsed times are recorded for information and are explicitly unstable: two
subprocesses at a time on two shared cores.
"""

from __future__ import annotations

# See the module docstring of validation/_bootstrap.py for why this statement
# is here and not in _bootstrap.
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

from sitediverse.report import format_table  # noqa: E402

VALIDATION = ROOT / "validation"
EXAMPLES = ROOT / "examples"

#: Rough cost of each script in seconds, measured during the build on
#: 2026-11-10 and used ONLY to dispatch the longest groups first. A wrong value
#: here costs wall-clock time and cannot change a verdict.
COST = {
    "validate_examples.py": 38.0,
    "validate_cli.py": 33.0,
    "validate_learned_model.py": 27.0,
    "validate_correlation.py": 8.0,
    "validate_copula.py": 4.0,
    "validate_overstatement.py": 3.0,
    "validate_empirical.py": 3.0,
    "validate_climatology.py": 2.0,
    "validate_negative_controls.py": 2.0,
    "validate_field.py": 1.5,
    "validate_approx_audit.py": 1.5,
    "validate_bounds.py": 1.0,
    "validate_geometry.py": 1.0,
}

#: Scripts whose runs must not overlap anything else that writes the same files.
#: ``validate_examples.py`` runs every example, and the examples write into
#: ``screenshots/``, so all of them share one serial group.
SCREENSHOT_WRITERS = {"validate_examples.py"}

#: Environment overrides applied to particular scripts. See the module docstring.
SCRIPT_ENVIRONMENT: dict[str, dict[str, str]] = {
    "validate_approx_audit.py": {"SITEDIVERSE_APPROX_AUDIT_MODE": "paths"},
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
    local = dict(environment)
    local.update(SCRIPT_ENVIRONMENT.get(script.name, {}))
    started = time.perf_counter()
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(script)],
        cwd=str(directory),
        capture_output=True,
        text=True,
        check=False,
        env=local,
    )
    detail = completed.stderr.strip()[-300:]
    if not detail:
        # These scripts report their own failed checks on STDOUT and exit 1 with
        # an empty stderr. Reporting only stderr printed nothing at all for the
        # one real failure a predecessor's probe ever caught, which cost a
        # diagnosis. Fall back to stdout rather than to silence.
        detail = completed.stdout.strip()[-300:]
    return (
        label,
        directory_label,
        int(completed.returncode),
        time.perf_counter() - started,
        detail,
    )


def run_group(
    jobs: tuple[tuple[str, Path, str, Path], ...], environment: dict[str, str]
) -> list[tuple[str, str, int, float, str]]:
    """Run one group of jobs SERIALLY. Different groups still run two at a time."""
    return [
        run_one(label, script, directory_label, directory, environment)
        for label, script, directory_label, directory in jobs
    ]


def build_groups(
    scripts: tuple[tuple[str, Path], ...],
    directories: tuple[tuple[str, Path], ...],
    example_scripts: tuple[tuple[str, Path], ...],
    example_directories: tuple[tuple[str, Path], ...],
) -> list[tuple[float, tuple[tuple[str, Path, str, Path], ...]]]:
    """Group the jobs so that nothing can overlap anything it shares files with."""
    groups: list[tuple[float, tuple[tuple[str, Path, str, Path], ...]]] = []
    shared: list[tuple[str, Path, str, Path]] = []
    shared_cost = 0.0
    for label, script in scripts:
        jobs = tuple(
            (label, script, directory_label, directory)
            for directory_label, directory in directories
        )
        cost = COST.get(script.name, 5.0) * len(directories)
        if script.name in SCREENSHOT_WRITERS:
            shared.extend(jobs)
            shared_cost += cost
        else:
            groups.append((cost, jobs))
    for label, script in example_scripts:
        shared.extend(
            (label, script, directory_label, directory)
            for directory_label, directory in example_directories
        )
        shared_cost += COST.get(script.name, 6.0) * len(example_directories)
    if shared:
        groups.append((shared_cost, tuple(shared)))
    return sorted(groups, key=lambda item: -item[0])


def tabulate(
    out: Tee,
    title: str,
    scripts: tuple[tuple[str, Path], ...],
    directories: tuple[tuple[str, Path], ...],
    by_script: dict[str, dict[str, tuple[int, float, str]]],
) -> int:
    """Tabulate one family of scripts and return its non-zero exit count."""
    out.rule(title)
    labels = [label for label, _ in directories]
    rows = []
    failures = 0
    for label, script in scripts:
        row = [label]
        for directory_label in labels:
            status, _, _ = by_script[label][directory_label]
            row.append(str(status))
            if status != 0:
                failures += 1
        row.append(
            "ok" if all(by_script[label][name][0] == 0 for name in labels) else "FAILED"
        )
        if script.name in SCRIPT_ENVIRONMENT:
            row[0] = f"{label} (paths only)"
        rows.append(row)
    out(format_table(["script", *labels, "verdict"], rows))
    out("")
    out(f"non-zero exit statuses : {failures} of "
        f"{len(scripts) * len(directories)}")
    for label, _ in scripts:
        for directory_label in labels:
            status, _, detail = by_script[label][directory_label]
            if status != 0 and detail:
                out(f"  {label} from {directory_label}: {detail}")
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
    out.rule("sitediverse -- every script, from every working directory that matters")
    for line in environment_lines():
        out(line)
    out("")
    out(f"validation scripts probed : {len(SCRIPTS)}")
    out(f"example scripts probed    : {len(EXAMPLE_SCRIPTS)}")
    out(f"validation runs           : {len(SCRIPTS) * len(DIRECTORIES)}")
    out(f"example runs              : "
        f"{len(EXAMPLE_SCRIPTS) * len(EXAMPLE_DIRECTORIES)}")
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
    out("")
    out("Grouping: every run of one script is serial on one worker, and everything")
    out("that writes into screenshots/ -- validate_examples.py and all seven")
    out("examples -- shares a single serial group, because two DIFFERENT scripts")
    out("writing the same file is the same defect as one script overlapping itself.")
    out("")
    out(f"scripts run in a reduced mode : "
        f"{', '.join(sorted(SCRIPT_ENVIRONMENT)) or 'none'}")

    environment = subprocess_environment()
    groups = build_groups(SCRIPTS, DIRECTORIES, EXAMPLE_SCRIPTS, EXAMPLE_DIRECTORIES)
    out(f"groups dispatched             : {len(groups)} over 2 workers")
    with ThreadPoolExecutor(max_workers=2) as pool:
        collected = list(
            pool.map(lambda item: run_group(item[1], environment), groups)
        )
    results = [entry for group in collected for entry in group]
    by_script: dict[str, dict[str, tuple[int, float, str]]] = {}
    for label, directory_label, status, elapsed, detail in results:
        by_script.setdefault(label, {})[directory_label] = (status, elapsed, detail)

    failures = tabulate(
        out,
        "validation scripts, from four working directories",
        SCRIPTS,
        DIRECTORIES,
        by_script,
    )
    failures += tabulate(
        out,
        "examples, from two working directories",
        EXAMPLE_SCRIPTS,
        EXAMPLE_DIRECTORIES,
        by_script,
    )
    out("")
    out(f"total runs       : {len(results)}")
    out(f"total wall clock : {time.perf_counter() - started:.1f} s")
    out("")
    out.check(
        "every script exits zero from every working directory probed",
        failures == 0,
        f"{failures} non-zero exit statuses in {len(results)} runs",
    )
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
