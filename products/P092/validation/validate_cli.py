"""The command line, exercised in subprocesses that cannot pollute this tree.

Every subprocess gets ``HOME`` and ``MPLCONFIGDIR`` pointed at a temporary
directory outside the repository and ``PYTHONDONTWRITEBYTECODE=1``, because a
validation script in an earlier product of this mission set ``HOME`` to the
product root and matplotlib wrote ``.config/matplotlib`` into the repository,
which the release gate fails twice over.

This script's OWN interpreter is handled by ``sys.dont_write_bytecode = True``
as the first statement after ``from __future__ import annotations``. That cannot
be delegated to ``_bootstrap``, because CPython writes a module's bytecode cache
BEFORE executing that module.

It also snapshots the artifact-shaped entries in the tree at entry and compares
at exit, and fails only on what IT added. What a harness left in the tree before
calling it is reported as inherited and does not fail: release-gate check 2 runs
``ruff check`` with its working directory set to the product root and writes
``.ruff_cache`` there one check before check 6 re-executes this file.

INPUT ORIGIN: the MEASURED campaign when present, the synthetic generator in a
temporary directory otherwise. Every invocation below says which it used, and
the exit statuses are reported either way: the CLI's behaviour with absent
traces (exit 3) is part of its contract and is checked explicitly.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

ARTIFACT_NAMES = (
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".hypothesis",
    ".config",
    "dist",
    "build",
)
ARTIFACT_SUFFIXES = (".pyc", ".pyo")


def snapshot() -> set[str]:
    found: set[str] = set()
    for path in ROOT.rglob("*"):
        if path.name in ARTIFACT_NAMES or path.name.endswith(ARTIFACT_SUFFIXES):
            found.add(_bootstrap.relative(path))
    return found


def run(out, label: str, arguments: list[str], expected: int, environment) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "deadlinerisk", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
        timeout=1800,
        check=False,
    )
    out.check(
        f"{label}: exit {expected}",
        completed.returncode == expected,
        f"got {completed.returncode}",
    )
    stream = completed.stdout if completed.returncode == 0 else completed.stderr
    head = [line for line in stream.splitlines() if line.strip()][:6]
    for line in head:
        out(f"      {line}")
    if completed.returncode == 0:
        out(f"      ... {len(stream.splitlines())} line(s) of output in total")


def main() -> int:
    entry = snapshot()
    out = _bootstrap.Tee("validate_cli_output.txt")
    out.rule("COMMAND LINE -- every subcommand, in a subprocess that cannot pollute")
    out("INPUT ORIGIN: the MEASURED campaign when present; a SYNTHETIC campaign in a")
    out("temporary directory for the paths that do not need measured data.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)

    environment = _bootstrap.subprocess_environment()

    out.rule("1  absent traces -- exit 3, which is part of the contract")
    missing = Path(tempfile.mkdtemp(prefix="deadlinerisk-absent-")) / "nothing"
    run(out, "manifest on an absent directory", ["--traces", str(missing), "manifest"], 3,
        environment)

    out.rule("2  a usage error -- exit 2")
    completed = subprocess.run(
        [sys.executable, "-m", "deadlinerisk", "--traces", "data/traces"],
        cwd=ROOT, capture_output=True, text=True, env=environment, timeout=600, check=False,
    )
    out.check("no subcommand: exit 2", completed.returncode == 2, f"got {completed.returncode}")

    out.rule("3  every subcommand on a SYNTHETIC campaign")
    synthetic = Path(tempfile.mkdtemp(prefix="deadlinerisk-cli-synthetic-"))
    _bootstrap.synthetic_campaign(synthetic)
    base = ["--traces", str(synthetic), "--origin", "synthetic"]
    for label, arguments in (
        ("manifest", ["manifest"]),
        ("account, all runs", ["account"]),
        ("account, one run", ["account", "--run", "gt_long"]),
        ("headline", ["headline", "--run", "gt_long", "--resamples", "200"]),
        ("evt", ["evt", "--run", "gt_long"]),
        ("learned", ["learned"]),
    ):
        out(f"  $ python -m deadlinerisk --traces <synthetic> --origin synthetic "
            f"{' '.join(arguments)}")
        run(out, f"SYNTHETIC {label}", [*base, *arguments], 0, environment)
    out("  $ python -m deadlinerisk --traces <synthetic> --origin synthetic headline "
        "--run nope")
    run(out, "SYNTHETIC unknown run", [*base, "headline", "--run", "nope"], 3, environment)

    out.rule("4  every subcommand on the MEASURED campaign")
    if _bootstrap.measured_data_present():
        measured = ["--traces", _bootstrap.MEASURED_TRACES]
        for label, arguments in (
            ("manifest", ["manifest"]),
            ("account, all runs", ["account"]),
            ("headline", ["headline", "--run", _bootstrap.GROUND_TRUTH_RUN]),
            ("evt", ["evt", "--run", _bootstrap.GROUND_TRUTH_RUN]),
            ("learned", ["learned"]),
        ):
            out(f"  $ python -m deadlinerisk --traces {_bootstrap.MEASURED_TRACES} "
                f"{' '.join(arguments)}")
            run(out, f"MEASURED {label}", [*measured, *arguments], 0, environment)
    else:
        out.unmeasurable_check(
            "the CLI against the MEASURED campaign",
            f"{_bootstrap.MEASURED_TRACES}/ held no jobs_*.csv when this script ran. "
            "The CLI's absent-traces contract is checked in section 1 instead; this is "
            "not a pass.",
        )

    out.rule("5  this script added nothing of its own")
    added = snapshot() - entry
    out.check(
        "running this script added no cache directory and no bytecode file",
        not added,
        f"added: {sorted(added)}" if added else "nothing added",
    )
    if entry:
        out(f"  inherited at entry, reported and NOT failed: {sorted(entry)}")
    else:
        out("  no artifact-shaped entry was present at entry")

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
