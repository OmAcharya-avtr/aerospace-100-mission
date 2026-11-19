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
``*_output.txt`` with fast-path numbers and redraw every committed figure, which
is the self-polluting-check class exactly: the probe's own execution would
create the condition the rest of the validation set is judged on. The copy
costs a second and removes the hazard, and the bytes of every committed output
file are compared before and after to confirm the copy really was the copy.

``EVIDENCESTALE_FAST=1`` is set for every probe, because the question here is
whether a script RUNS from a given directory and not what its numbers are. No
number in ``VALIDATION.md`` comes from this script.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

#: Scripts excluded, with the reason. This script would recurse into itself,
#: and the two leading-underscore modules are support modules rather than
#: scripts -- the release gate does execute them, and they exit 0 when run
#: directly because they only define things, which is checked in section 3.
EXCLUDED = {
    "_bootstrap.py",
    "_approx_audit_plugin.py",
    "probe_working_directories.py",
}

#: Per-script timeout, seconds. The release gate allows 600 s per script; this
#: is deliberately below that so a hang is reported here rather than at the
#: gate.
TIMEOUT = 420.0


def main() -> int:
    """Run every script here and in examples/ from four directories."""
    out = _bootstrap.Tee("probe_working_directories_output.txt")
    out("Working-directory probe -- P098 EvidenceStale")
    out("")
    out("each script is run from four directories and the same exit status is")
    out("required from all four. Every probe runs in a PRISTINE COPY of this")
    out("product, so no committed output file and no committed figure is touched.")
    out("")
    out(f"excluded, with the reason in the module docstring: {sorted(EXCLUDED)}")
    out(f"per-script timeout, seconds: {TIMEOUT!r}")
    out("EVIDENCESTALE_FAST=1 is set for every probe: the question is whether a")
    out("script RUNS from a directory, not what its numbers are.")

    scripts = sorted(
        path.name
        for path in (ROOT / "validation").glob("*.py")
        if path.name not in EXCLUDED
    )
    support = sorted(
        path.name
        for path in (ROOT / "validation").glob("*.py")
        if path.name in EXCLUDED and path.name != "probe_working_directories.py"
    )
    examples = sorted(
        path.name
        for path in (ROOT / "examples").glob("*.py")
        if not path.name.startswith("_")
    )
    out("")
    out(f"validation scripts probed: {len(scripts)} -- {scripts}")
    out(f"validation support modules probed: {len(support)} -- {support}")
    out(f"examples probed: {len(examples)} -- {examples}")

    before_text = {
        path.name: path.read_bytes()
        for path in sorted((ROOT / "validation").glob("*_output.txt"))
    }
    before_figures = {
        path.name: path.read_bytes()
        for path in sorted((ROOT / "screenshots").glob("*.png"))
    }

    system_temporary = tempfile.gettempdir()
    results: dict[str, list[tuple[str, int | None]]] = {}
    with _bootstrap.pristine_copy(ROOT) as (copy, home):
        directories = [
            ("filesystem root", os.sep),
            ("system temporary directory", system_temporary),
            ("product root", str(copy)),
            ("the script's own directory", None),
        ]
        environment = _bootstrap.subprocess_environment(home)
        environment["EVIDENCESTALE_FAST"] = "1"
        for folder, names in (
            ("validation", [*scripts, *support]),
            ("examples", examples),
        ):
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

    out.rule("1. EXIT STATUS FROM EACH DIRECTORY")
    out(f"{'script':>46}  {'root':>7}  {'tmp':>7}  {'product':>8}  {'own dir':>8}  {'same':>5}")
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
            f"{script:>46}  {rendered[0]:>7}  {rendered[1]:>7}  {rendered[2]:>8}"
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

    out.rule("2. THE COPY WAS NOT THE TREE -- checked, not asserted")
    out("a probe that ran these scripts in place would overwrite every committed")
    out("*_output.txt with fast-path numbers and redraw every committed figure.")
    out("So the bytes of every committed output file and every committed figure are")
    out("recorded before the probe and compared after it. This script's own output")
    out("is excluded, because Tee writes it at exit and it is the one file this")
    out("script is supposed to change.")
    out("")
    after_text = {
        path.name: path.read_bytes()
        for path in sorted((ROOT / "validation").glob("*_output.txt"))
    }
    after_figures = {
        path.name: path.read_bytes()
        for path in sorted((ROOT / "screenshots").glob("*.png"))
    }
    own = "probe_working_directories_output.txt"
    changed_text = sorted(
        name
        for name in set(before_text) | set(after_text)
        if name != own and before_text.get(name) != after_text.get(name)
    )
    changed_figures = sorted(
        name
        for name in set(before_figures) | set(after_figures)
        if before_figures.get(name) != after_figures.get(name)
    )
    out(f"committed output files before the probe : {len(before_text)}")
    out(f"committed output files after the probe  : {len(after_text)}")
    out(f"committed figures before the probe      : {len(before_figures)}")
    out(f"committed figures after the probe       : {len(after_figures)}")
    out(f"output files changed, excluding its own  : {len(changed_text)}")
    for name in changed_text:
        out(f"  CHANGED: {name}")
    out(f"figures changed                          : {len(changed_figures)}")
    for name in changed_figures:
        out(f"  CHANGED: {name}")
    out.check(
        "the probe changed no committed output file other than its own",
        not changed_text,
        f"{len(changed_text)} changed",
    )
    out.check(
        "the probe changed no committed figure",
        not changed_figures,
        f"{len(changed_figures)} changed",
    )

    out.rule("3. THE SUPPORT MODULES EXIT 0 WHEN EXECUTED DIRECTLY")
    out("the release gate executes EVERY validation/*.py, including the two")
    out("leading-underscore support modules, which are not scripts. They only")
    out("define things, so they exit 0, and the table in section 1 is the evidence")
    out("rather than an assumption about them.")
    for name in support:
        statuses = [status for _label, status in results[f"validation/{name}"]]
        out(f"  validation/{name}: {statuses}")
        out.check(
            f"validation/{name} exits 0 from all four directories",
            all(status == 0 for status in statuses),
            f"{statuses}",
        )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
