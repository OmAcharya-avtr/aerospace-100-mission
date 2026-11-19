"""Every script here must run from any working directory, and this proves it.

Three products in one batch of this mission failed the release gate after
passing every check that preceded it, because a validation script passed a
bare relative path to a loader. Nothing in those builds caught it, and the
reason is worth restating: the build agent, the agent's self-report and the
coordinator's re-measurement all ran the script the same way, from the product
root. The gate runs it with ``cwd=validation/``. Three independent
verifications agreeing does not help when they share an assumption.

So each script is run from four directories -- the filesystem root, the system
temporary directory, the product root and the script's own directory -- and
the same exit status is required from all four.

**Every probe runs in a PRISTINE COPY of this product, never in the tree
itself.** A probe that ran the scripts in place would overwrite every
committed ``*_output.txt`` with fast-path numbers, which is the
self-polluting-check class exactly: the probe's own execution would create the
condition the rest of the validation set is judged on. The copy costs a second
and removes the hazard, and the bytes of every committed output file are
recorded before the probe and compared after it to show the copy was not the
tree.

``PASSYIELD_FAST=1`` is set for every probe, because the question here is
whether a script RUNS from a given directory and not what its numbers are. No
number in ``validation/VALIDATION.md`` comes from this script.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

#: Modules excluded, with the reason. Both are imported, never executed as a
#: script: ``_bootstrap`` is the shared support module and
#: ``_approx_audit_plugin`` is a pytest plugin. This script excludes itself,
#: because a probe that ran itself inside its own pristine copy would recurse.
EXCLUDED = {
    "_bootstrap.py",
    "_approx_audit_plugin.py",
    "probe_working_directories.py",
}

#: Per-script timeout, seconds. The release gate allows 600 s per script; this
#: is set higher than any script needs on the fast path so that a TIMEOUT in
#: the table below means a hang and not a slow core.
TIMEOUT = 900.0


def main() -> int:
    """Run every script here from four directories inside a pristine copy."""
    out = _bootstrap.Tee("probe_working_directories_output.txt")
    out("Working-directory probe -- P099 PassYield")
    out("")
    out("each script is run from four directories and the same exit status is")
    out("required from all four. Every probe runs in a PRISTINE COPY of this")
    out("product, so no committed output file is touched.")
    out("")
    out(f"excluded, with the reason in the module docstring: {sorted(EXCLUDED)}")
    out(f"per-script timeout, seconds: {TIMEOUT!r}")
    out("PASSYIELD_FAST=1 is set for every probe: the question is whether a script")
    out("RUNS from a directory, not what its numbers are.")

    scripts = sorted(
        path.name
        for path in (ROOT / "validation").glob("*.py")
        if path.name not in EXCLUDED
    )
    examples = sorted(
        path.name for path in (ROOT / "examples").glob("*.py")
        if not path.name.startswith("_")
    )
    out("")
    out(f"validation scripts probed: {len(scripts)}")
    out(f"examples probed          : {len(examples)}")

    before = {
        path.name: path.read_bytes()
        for path in sorted((ROOT / "validation").glob("*_output.txt"))
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
        environment["PASSYIELD_FAST"] = "1"
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
    out(f"{'script':>44}  {'root':>6}  {'tmp':>6}  {'product':>8}  {'own dir':>8}  {'same':>5}")
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
            f"{script:>44}  {rendered[0]:>6}  {rendered[1]:>6}  {rendered[2]:>8}"
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
    out("a probe that ran these scripts in place would overwrite every committed")
    out("*_output.txt with fast-path numbers, which is the self-polluting-check")
    out("class exactly. So the bytes of every committed output file are recorded")
    out("before the probe and compared after it. This script's own output is")
    out("excluded, because Tee writes it at exit and it is the one file this")
    out("script is supposed to change.")
    out("")
    after = {
        path.name: path.read_bytes()
        for path in sorted((ROOT / "validation").glob("*_output.txt"))
    }
    own = "probe_working_directories_output.txt"
    changed = sorted(
        name
        for name in set(before) | set(after)
        if name != own and before.get(name) != after.get(name)
    )
    out(f"committed output files before the probe: {len(before)}")
    out(f"committed output files after the probe : {len(after)}")
    out(f"files changed by the probe, excluding its own: {len(changed)}")
    for name in changed:
        out(f"  CHANGED: {name}")
    out.check(
        "the probe changed no committed output file other than its own",
        not changed,
        f"{len(changed)} changed",
    )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
