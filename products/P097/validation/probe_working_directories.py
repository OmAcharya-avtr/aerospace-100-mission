"""Every script here must run from any working directory, and this proves it.

Three products in one batch of this mission failed the release gate after
passing every check that preceded it, because a validation script passed a bare
relative path to a loader. Nothing in those builds caught it, and the reason is
worth restating: the build agent, the agent's self-report and the coordinator's
re-measurement all ran the script the same way, from the product root. The gate
runs it with ``cwd=validation/``. Three independent verifications agreeing does
not help when they share an assumption.

So each script is run from four directories -- the filesystem root, the system
temporary directory, the product root and the script's own directory -- and the
same exit status is required from all four.

**Every probe runs in a PRISTINE COPY of this product, never in the tree
itself.** A probe that ran the scripts in place would overwrite every committed
``*_output.txt`` with fast-path numbers, which is the self-polluting-check class
exactly: the probe's own execution would create the condition the rest of the
validation set is judged on. The copy costs a second and removes the hazard,
and the bytes of every committed output file are recorded before the probe and
compared after it rather than the absence of pollution being asserted.

``FLAKERANK_FAST=1`` is set for every probe, because the question here is
whether a script RUNS from a given directory and not what its numbers are. No
number in ``VALIDATION.md`` comes from this script.

NO CLOCK IS READ. Subprocess exit statuses are compared; no duration is
measured, and the per-script timeout is a guard against a hang, not a
measurement -- a timeout is reported as ``TIMEOUT`` and fails the check rather
than being compared with anything.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

#: Scripts excluded, with the reason.
#:
#: ``_bootstrap.py`` and ``_approx_audit_plugin.py`` are support modules, not
#: runnable scripts; the release gate executes them and they exit 0 by doing
#: nothing, which is not a working-directory question.
#:
#: ``validate_approx_audit.py`` runs the WHOLE pytest suite in a subprocess,
#: four times over in this probe, which would dominate the probe's cost for a
#: question it cannot answer differently from the other scripts: it resolves
#: every path through ``_bootstrap`` exactly as they do. It is covered by its
#: own run from ``validation/``, which is the gate's own arrangement.
EXCLUDED = {
    "_bootstrap.py",
    "_approx_audit_plugin.py",
    "probe_working_directories.py",
    "validate_approx_audit.py",
}

#: Per-script timeout, seconds. A guard against a hang, never a measurement.
TIMEOUT = 600.0

#: Directory and file names not copied into the pristine copy.
SKIP = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis", ".venv"}


def copy_tree(destination: Path) -> None:
    """Copy this product into ``destination``, skipping caches."""
    shutil.copytree(
        ROOT,
        destination,
        ignore=shutil.ignore_patterns(*SKIP),
        dirs_exist_ok=True,
    )


def main() -> int:
    """Run every script here and every example from four directories."""
    out = _bootstrap.Tee("probe_working_directories_output.txt")
    out("Working-directory probe -- P097 flakerank")
    out("")
    out("Each script is run from four directories and the same exit status is")
    out("required from all four. Every probe runs in a PRISTINE COPY of this")
    out("product, so no committed output file is touched.")
    out("")
    out(f"excluded, with the reason in the module source: {sorted(EXCLUDED)}")
    out(f"per-script timeout, seconds: {TIMEOUT!r} -- a guard against a hang,")
    out("  never a measurement; a timeout fails the check rather than being")
    out("  compared with anything")
    out("FLAKERANK_FAST=1 is set for every probe: the question is whether a script")
    out("RUNS from a directory, not what its numbers are.")

    scripts = sorted(
        path.name
        for path in (ROOT / "validation").glob("*.py")
        if path.name not in EXCLUDED
    )
    examples = sorted(
        path.name for path in (ROOT / "examples").glob("*.py") if not path.name.startswith("_")
    )
    out("")
    out(f"validation scripts probed: {len(scripts)}")
    for name in scripts:
        out(f"  validation/{name}")
    out(f"examples probed: {len(examples)}")
    for name in examples:
        out(f"  examples/{name}")

    before = {
        path.name: path.read_bytes()
        for path in sorted((ROOT / "validation").glob("*_output.txt"))
    }
    screenshots_before = {
        path.name: path.read_bytes() for path in sorted((ROOT / "screenshots").glob("*.png"))
    }

    system_temporary = tempfile.gettempdir()
    results: dict[str, list[tuple[str, int | None]]] = {}
    with tempfile.TemporaryDirectory(prefix="flakerank-probe-") as temporary:
        base = Path(temporary)
        copy = base / "product"
        home = base / "home"
        home.mkdir()
        copy_tree(copy)
        directories = [
            ("filesystem root", os.sep),
            ("system temporary directory", system_temporary),
            ("product root", str(copy)),
            ("the script's own directory", None),
        ]
        environment = _bootstrap.subprocess_environment(home)
        environment["FLAKERANK_FAST"] = "1"
        for folder, names in (("validation", scripts), ("examples", examples)):
            for name in names:
                target = copy / folder / name
                statuses: list[tuple[str, int | None]] = []
                for label, directory in directories:
                    working = str(target.parent) if directory is None else directory
                    try:
                        completed = subprocess.run(  # noqa: S603 - fixed argument vector
                            [sys.executable, str(target)],
                            cwd=working,
                            capture_output=True,
                            env=environment,
                            timeout=TIMEOUT,
                            check=False,
                        )
                        statuses.append((label, completed.returncode))
                    except subprocess.TimeoutExpired:
                        statuses.append((label, None))
                results[f"{folder}/{name}"] = statuses

    out.rule("EXIT STATUS FROM EACH DIRECTORY")
    out("")
    out(f"{'script':>44}  {'root':>7}  {'tmp':>7}  {'product':>8}  {'own dir':>8}  {'same':>5}")
    disagreements: list[str] = []
    nonzero: list[str] = []
    for script in sorted(results):
        statuses = [status for _label, status in results[script]]
        same = len(set(statuses)) == 1
        if not same:
            disagreements.append(script)
        if any(status != 0 for status in statuses):
            nonzero.append(script)
        rendered = ["TIMEOUT" if status is None else str(status) for status in statuses]
        out(
            f"{script:>44}  {rendered[0]:>7}  {rendered[1]:>7}  {rendered[2]:>8}"
            f"  {rendered[3]:>8}  {str(same):>5}"
        )
    out("")
    out.check(
        "every script returns the same exit status from all four directories",
        not disagreements,
        f"{len(disagreements)} disagree: {disagreements}",
    )
    out.check(
        "every script exits 0 from all four directories",
        not nonzero,
        f"{len(nonzero)} did not: {nonzero}",
    )

    out.rule("THE COPY WAS NOT THE TREE -- checked, not asserted")
    out("")
    out("A probe that ran these scripts in place would overwrite every committed")
    out("*_output.txt with fast-path numbers and every committed screenshot,")
    out("which is the self-polluting-check class exactly. So the bytes of every")
    out("committed output file and every committed PNG are recorded before the")
    out("probe and compared after it. This script's own output is excluded,")
    out("because the Tee writes it at exit and it is the one file this script is")
    out("supposed to change.")
    out("")
    after = {
        path.name: path.read_bytes()
        for path in sorted((ROOT / "validation").glob("*_output.txt"))
    }
    screenshots_after = {
        path.name: path.read_bytes() for path in sorted((ROOT / "screenshots").glob("*.png"))
    }
    own = "probe_working_directories_output.txt"
    changed = sorted(
        name
        for name in set(before) | set(after)
        if name != own and before.get(name) != after.get(name)
    )
    changed_png = sorted(
        name
        for name in set(screenshots_before) | set(screenshots_after)
        if screenshots_before.get(name) != screenshots_after.get(name)
    )
    out(f"committed output files before the probe: {len(before)}")
    out(f"committed output files after the probe : {len(after)}")
    out(f"committed screenshots before / after   : {len(screenshots_before)} / "
        f"{len(screenshots_after)}")
    out(f"output files changed by the probe, excluding its own: {len(changed)}")
    for name in changed:
        out(f"  CHANGED: {name}")
    out(f"screenshots changed by the probe: {len(changed_png)}")
    for name in changed_png:
        out(f"  CHANGED: {name}")
    out.check(
        "the probe changed no committed output file other than its own",
        not changed,
        f"{len(changed)} changed",
    )
    out.check(
        "the probe changed no committed screenshot",
        not changed_png,
        f"{len(changed_png)} changed",
    )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
