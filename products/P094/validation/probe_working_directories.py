"""Every validation script, run from four working directories, same exit status.

Three products in one batch of this mission passed every check in their own
build and then failed the release gate, which runs each validation script with
its working directory set to the script's own directory rather than to the
product root. Nothing in those builds caught it, because the build agent, its
self-report and the coordinator's re-measurement all ran the scripts the same
way.

So every script here is executed from the filesystem root, the system temporary
directory, the product root and ``validation/``, and the four exit statuses must
agree. ``HILTIMINGDIFF_FAST=1`` lowers the pass and resample counts: the
question this probe asks is whether a script RUNS from a given directory, not
what its numbers are, and the committed outputs are the ones from the full runs.

This probe is excluded from its own sweep, because a script that runs itself
from four directories recurses.

**This probe AMPLIFIES any load-sensitive check by four.** It executes every
script here four times and requires the same exit status from all four, so a
check whose outcome depends on the machine's load fails this probe rather than
the script that owns it, and the failure is reported against the wrong file.
That is one reason the load-sensitive assertion class was removed at its design
rather than patched instance by instance; see
validation/load_sensitive_inventory.md.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

#: This script, excluded from its own sweep.
SELF = Path(__file__).name

#: The four directories every script must run from with the same exit status.
DIRECTORIES = (
    Path(os.sep),
    Path(tempfile.gettempdir()),
    ROOT,
    ROOT / "validation",
)

#: The live-network probe is excluded: a transient network failure would make
#: this probe report a working-directory defect that does not exist. It is run
#: on its own by the sweep in .github/workflows/tests.yml.
EXCLUDED = (SELF, "validate_pypi_probe.py")


def label(directory: Path) -> str:
    """A readable, RELATIVE label for a working directory.

    ``_bootstrap.relative`` renders the filesystem root and the product root as
    an empty string and ``.`` respectively, which reads as a missing value in a
    committed output. No absolute path is ever printed.
    """
    if directory == Path(os.sep):
        return "the filesystem root"
    if directory == ROOT:
        return f"the product root ({ROOT.name})"
    if directory == Path(tempfile.gettempdir()):
        return "the system temporary directory"
    return _bootstrap.relative(directory)


def scripts() -> list[Path]:
    """Every runnable validation script, excluding helpers and the exclusions."""
    return [
        path
        for path in sorted((ROOT / "validation").glob("*.py"))
        if not path.name.startswith("_") and path.name not in EXCLUDED
    ]


def main() -> int:
    tee = _bootstrap.Tee("probe_working_directories_output.txt")
    _bootstrap.header(tee, "P094 HilTimingDiff -- working-directory independence")

    tee.rule("what is probed")
    selected = scripts()
    for path in selected:
        tee(f"  {_bootstrap.relative(path)}")
    tee("")
    tee(f"  excluded : {', '.join(EXCLUDED)}")
    tee("  reasons  : this probe would recurse into itself; the PyPI probe needs the")
    tee("             network and a transient failure would look like a path defect.")
    tee("")
    for directory in DIRECTORIES:
        tee(f"  working directory : {label(directory)}")
    tee("")
    tee.check("the sweep has scripts to run", bool(selected))

    environment = _bootstrap.subprocess_environment()
    environment["HILTIMINGDIFF_FAST"] = "1"
    # Every script this probe runs WRITES its own committed output file. Four
    # runs each would leave the committed evidence holding this probe's reduced
    # FAST-mode figures rather than the full runs VALIDATION.md quotes, so the
    # subject scripts are redirected to a temporary directory outside the
    # repository. Checked at the end of this script, not assumed.
    sink = Path(tempfile.mkdtemp(prefix="hiltimingdiff-probe-outputs-"))
    environment[_bootstrap.OUTPUT_DIRECTORY_VARIABLE] = str(sink)

    tee.rule("the sweep")
    disagreements: list[str] = []
    for path in selected:
        statuses: list[int] = []
        for directory in DIRECTORIES:
            result = subprocess.run(
                [sys.executable, str(path)],
                cwd=str(directory),
                env=environment,
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )
            statuses.append(result.returncode)
            if result.returncode != 0:
                tail = (result.stderr.strip() or result.stdout.strip()).splitlines()
                for line in tail[-4:]:
                    tee(f"      {line[:150]}")
        agree = len(set(statuses)) == 1
        if not agree:
            disagreements.append(path.name)
        tee(
            f"  {path.name:<42s} exits {statuses} "
            f"{'agree' if agree else 'DISAGREE'}"
        )
        tee.check(
            f"{path.name} returns the same exit status from all four directories",
            agree,
            f"statuses {statuses}",
        )
        tee.check(
            f"{path.name} exits 0 from every directory",
            set(statuses) == {0},
            f"statuses {statuses}",
        )

    tee.rule("summary")
    tee(f"  scripts probed  : {len(selected)}")
    tee(f"  directories     : {len(DIRECTORIES)}")
    tee(f"  executions      : {len(selected) * len(DIRECTORIES)}")
    tee(f"  disagreements   : {len(disagreements)}")
    tee.check("no script depends on the caller's working directory", not disagreements)

    tee.rule("this probe must not have replaced the committed evidence")
    tee(
        "  Every script above writes its own <stem>_output.txt. Four runs each would "
        "leave the"
    )
    tee(
        "  committed outputs holding this probe's reduced HILTIMINGDIFF_FAST=1 figures "
        "instead of"
    )
    tee(
        "  the full runs VALIDATION.md quotes, so the subject scripts were redirected "
        "to a"
    )
    tee("  temporary directory. That redirection is verified here, not assumed.")
    tee("")
    redirected = sorted(path.name for path in sink.glob("*_output.txt"))
    tee(f"  outputs written into the temporary sink : {len(redirected)}")
    for name in redirected:
        tee(f"    {name}")
    tee("")
    tee.check(
        "every probed script wrote into the temporary sink, not into validation/",
        len(redirected) == len(selected),
        f"{len(redirected)} sink files against {len(selected)} probed scripts",
    )
    # Two precautions, both because a check whose own execution can create the
    # condition it tests is not a check. This script's own output file NAMES the
    # marker in its prose, so it is excluded by name; and the match is on the
    # environment-header FORM that only a genuine FAST run emits, not on the
    # bare string, so prose anywhere else cannot trip it either.
    marker = "simulation passes: "
    fast_suffix = "[HILTIMINGDIFF_FAST=1]"
    own_output = Path(tee.path).name
    stale = sorted(
        path.name
        for path in (ROOT / "validation").glob("*_output.txt")
        if path.name != own_output
        and any(
            line.lstrip().startswith(marker) and fast_suffix in line
            for line in path.read_text().splitlines()
        )
    )
    tee.check(
        "no committed validation output carries the FAST marker",
        not stale,
        f"{len(stale)} committed outputs are FAST-mode runs: {', '.join(stale)}"
        if stale
        else f"every committed output is a full run ({own_output} excluded: it names "
        "the marker in its own prose)",
    )

    tee.rule("negative control: a script that DOES depend on the working directory")
    scratch = Path(tempfile.mkdtemp(prefix="hiltimingdiff-probe-"))
    bad = scratch / "relative_path_script.py"
    bad.write_text(
        "import sys\n\n"
        "sys.dont_write_bytecode = True\n\n"
        "# Deliberately passes a BARE RELATIVE PATH to a loader, which is exactly\n"
        "# what cost three products of this mission a release gate.\n"
        "with open('data/traces/meta.json') as handle:\n"
        "    handle.read()\n"
    )
    statuses = []
    for directory in DIRECTORIES:
        result = subprocess.run(
            [sys.executable, str(bad)],
            cwd=str(directory),
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        statuses.append(result.returncode)
    tee(f"  control script exits {statuses}")
    tee.control(
        "a bare-relative-path script disagrees across the four directories",
        len(set(statuses)) > 1,
        must_fire=True,
        detail=f"statuses {statuses}: the probe can see the defect it exists to catch",
    )
    import shutil

    shutil.rmtree(scratch, ignore_errors=True)

    path = tee.save()
    print(f"\nwritten: {path}")
    return 1 if tee.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
