"""What this product's own execution adds to the tree, and nothing else.

A SELF-CHECK MAY ONLY ASSERT WHAT THE PRODUCT OWNS
--------------------------------------------------
This script asserts two things and deliberately not a third.

- **That running a script ADDS no cache directory and no bytecode file.** The
  tree is snapshotted at entry and compared at exit, and the scripts under
  ``validation/`` and ``examples/`` are additionally run in a subprocess with
  the snapshot taken around each one.
- **That no cache directory, bytecode file or build artifact would reach a
  cold clone**, which is what matters, and which is checked here by
  re-implementing the ``.gitignore`` semantics rather than by running git,
  because build agents in this batch are forbidden from running git.
- **NOT that no such artifact exists.** Release-gate check 2 runs
  ``ruff check src/ tests/`` with its working directory set to the product
  root, which writes ``.ruff_cache`` into the tree; check 6 then re-executes
  the validation scripts. A script asserting that no cache directory EXISTS
  fails on the directory the gate itself created one check earlier, and no
  change to this product can fix it. Inherited artifacts are **reported, named
  as inherited, and not failed on.**

That narrowing cost another product of this mission a whole night, and the
general rule is worth stating once: a check whose own execution can create the
condition it tests is not a check. This script therefore runs its own scanner
twice and requires the two runs to agree.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import subprocess  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

#: Scripts run here under the snapshot, inside a pristine copy. This script's
#: question is whether a script LEAVES ANYTHING BEHIND and not what its
#: numbers are, so ``PASSYIELD_FAST`` is always set for these subprocesses.
#: One validation script and one example, because the two directories have
#: different support modules and the example additionally imports matplotlib,
#: which is what wrote a cache into another product's repository.
PROBED_SCRIPTS = (
    "validation/validate_record.py",
    "examples/pass_geometry.py",
)


def main() -> int:
    """Measure what this product's own execution adds to its tree."""
    out = _bootstrap.Tee("validate_tree_hygiene_output.txt")
    out("Tree hygiene -- P099 PassYield")
    out("")
    out(f"fast path (PASSYIELD_FAST=1): {_bootstrap.FAST}")

    out.rule("0. WHAT WAS ALREADY HERE -- inherited, reported, not failed on")
    entry = _bootstrap.tree_snapshot(ROOT)
    out(f"cache directories and bytecode files present at entry: {len(entry)}")
    for name in sorted(entry)[:20]:
        out(f"  INHERITED: {name}")
    if len(entry) > 20:
        out(f"  ... and {len(entry) - 20} more")
    out("")
    out("these are NOT a defect of this product. The release gate's own ruff run")
    out("writes .ruff_cache into the tree one check before the validation scripts")
    out("are executed, and its pytest run can leave .hypothesis the same way. What")
    out("a harness leaves behind before calling a script is not the script's doing.")

    out.rule("1. DOES RUNNING A SCRIPT ADD ANYTHING?")
    out("each script below is run in a subprocess with HOME and MPLCONFIGDIR")
    out("pointed outside the product and PYTHONDONTWRITEBYTECODE set, and the tree")
    out("is snapshotted around each run. A previous product of this mission set")
    out("HOME to the product root and matplotlib wrote .config/matplotlib INTO the")
    out("repository, which the release gate fails twice over.")
    out("")
    out("THE RUNS HAPPEN IN A PRISTINE COPY, not in this tree. Running")
    out("validate_record.py here would overwrite validation/validate_record_output.txt")
    out("with fast-path numbers, and the release gate runs the validation scripts in")
    out("alphabetical order, so a script that regenerated another one's committed")
    out("evidence would collide with it on the gate's own ordering. The copy is a")
    out("faithful product tree, so the measurement is the same and the hazard is")
    out("gone. The bytes of every committed output file here are compared before and")
    out("after to show that the copy was not the tree.")
    out("")
    added_total: set[str] = set()
    probed = () if _bootstrap.FAST else PROBED_SCRIPTS
    if _bootstrap.FAST:
        out.unavailable(
            "the subprocess probes",
            "the fast path exists for the working-directory probe, which already "
            "runs every script four times; spawning them again from inside it "
            "would multiply the cost of a shared two-core runner for nothing",
        )
    committed_before = {
        path.name: path.read_bytes()
        for path in sorted((ROOT / "validation").glob("*_output.txt"))
    }
    if probed:
        with _bootstrap.pristine_copy(ROOT) as (copy, home):
            environment = _bootstrap.subprocess_environment(home)
            environment["PASSYIELD_FAST"] = "1"
            for name in probed:
                before = _bootstrap.tree_snapshot(copy)
                completed = subprocess.run(  # noqa: S603 - fixed argument vector
                    [sys.executable, str(copy / name)],
                    cwd=str(copy / "validation"),
                    capture_output=True,
                    env=environment,
                    check=False,
                    timeout=600.0,
                )
                after = _bootstrap.tree_snapshot(copy)
                added = after - before
                added_total |= added
                out(f"{name}: exit {completed.returncode}, added {len(added)}")
                for item in sorted(added):
                    out(f"  ADDED: {item}")
                out.check(
                    f"{name} exits 0 from validation/ in a pristine copy",
                    completed.returncode == 0,
                    f"exit {completed.returncode}",
                )
        out("")
        out.check(
            "running these scripts added no cache directory and no bytecode file",
            not added_total,
            f"{len(added_total)} added: {sorted(added_total)}",
        )
    committed_after = {
        path.name: path.read_bytes()
        for path in sorted((ROOT / "validation").glob("*_output.txt"))
    }
    own = "validate_tree_hygiene_output.txt"
    changed = sorted(
        name
        for name in set(committed_before) | set(committed_after)
        if name != own and committed_before.get(name) != committed_after.get(name)
    )
    out("")
    out(f"committed output files changed by this script, excluding its own: {len(changed)}")
    for name in changed:
        out(f"  CHANGED: {name}")
    out.check(
        "this script regenerated no other script's committed evidence",
        not changed,
        f"{len(changed)} changed: {changed}",
    )

    out.rule("2. THIS SCRIPT'S OWN INTERPRETER")
    out("a validation script in another product of this mission imported its own")
    out("package to call it, which wrote 11 .pyc files into the tree, and then")
    out("scanned the tree for exactly those files and failed itself. Setting")
    out("PYTHONDONTWRITEBYTECODE for SPAWNED subprocesses does nothing about that:")
    out("the polluting interpreter is the script's own. Hence")
    out("sys.dont_write_bytecode = True as the first statement after the __future__")
    out("import in every runnable script here, which cannot be delegated to")
    out("_bootstrap because CPython writes a module's bytecode cache BEFORE")
    out("executing that module.")
    out("")
    out(f"sys.dont_write_bytecode in this process: {sys.dont_write_bytecode!r}")
    out.check("this script disabled bytecode writing", sys.dont_write_bytecode is True)

    scripts = sorted((ROOT / "validation").glob("*.py")) + sorted(
        (ROOT / "examples").glob("*.py")
    ) + sorted((ROOT / "data").glob("*.py"))
    missing = []
    for path in scripts:
        if path.name.startswith("_"):
            continue
        if "sys.dont_write_bytecode = True" not in path.read_text():
            missing.append(_bootstrap.relative(path))
    out(f"runnable scripts checked: {len([p for p in scripts if not p.name.startswith('_')])}")
    out.check(
        "every runnable script sets the flag",
        not missing,
        f"missing in {missing}",
    )

    out.rule("3. WOULD ANY ARTIFACT REACH A COLD CLONE?")
    out("the .gitignore semantics are re-implemented here rather than delegated to a")
    out("git command, because build agents in this batch are forbidden from running")
    out("git. The question is not whether an artifact exists but whether a cold")
    out("clone would carry it.")
    out("")
    patterns, negations = _bootstrap.gitignore_rules()
    out(f"ignore patterns  : {patterns}")
    out(f"negation patterns: {negations}")
    out("")

    uncovered: list[str] = []
    for name in sorted(_bootstrap.tree_snapshot(ROOT)):
        flag, pattern = _bootstrap.gitignore_match(name, patterns, negations)
        if not flag:
            uncovered.append(f"{name} ({pattern or 'no pattern covers it'})")
    out(f"artifacts present in the tree          : {len(_bootstrap.tree_snapshot(ROOT))}")
    out(f"artifacts a cold clone would carry     : {len(uncovered)}")
    for name in uncovered:
        out(f"  WOULD BE TRACKED: {name}")
    out.check(
        "no cache directory or bytecode file would reach a cold clone",
        not uncovered,
        f"{len(uncovered)} would",
    )

    out.rule("4. THE SCANNER RUN TWICE, FROM THE SAME TREE")
    out("a check whose own execution can create the condition it tests is not a")
    out("check. This is the fifth occurrence of that class in this mission, so the")
    out("scanner is run twice and the two runs must agree.")
    out("")
    first = _bootstrap.tree_snapshot(ROOT)
    second = _bootstrap.tree_snapshot(ROOT)
    out(f"first run  : {len(first)} artifacts")
    out(f"second run : {len(second)} artifacts")
    out(f"difference : {sorted(first ^ second)}")
    out.check("the scanner agrees with itself on a second run", first == second)

    exit_snapshot = _bootstrap.tree_snapshot(ROOT)
    out("")
    out(f"artifacts at entry : {len(entry)}")
    out(f"artifacts at exit  : {len(exit_snapshot)}")
    out(f"added by this whole script: {sorted(exit_snapshot - entry)}")
    out.check(
        "this script added nothing to the tree",
        not (exit_snapshot - entry),
        f"added {sorted(exit_snapshot - entry)}",
    )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
