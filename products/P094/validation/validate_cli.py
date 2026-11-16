"""Every CLI command, run as a subprocess, with its output captured and checked.

Each subprocess gets ``HOME`` and ``MPLCONFIGDIR`` in a temporary directory
outside the repository and ``PYTHONDONTWRITEBYTECODE=1``, because a build that
pollutes its own tree through a subprocess environment it sets is a defect class
this mission has shipped. The script also checks what the product OWNS about
tree hygiene: that running it ADDS no cache directory and no bytecode file.
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

#: Commands, and the pass count each is run with. The fast two do not measure
#: simulation mode at all.
COMMANDS = {
    "gate": [],
    "clock": [],
    "headline": ["--passes", "1", "--resamples", "300", "--quiet"],
    "baselines": ["--passes", "1", "--resamples", "300", "--quiet"],
    "drift": ["--passes", "1", "--resamples", "300", "--quiet"],
}

#: Fragments each command's output must contain.
EXPECTED = {
    "gate": ("every branch fired at least once : True", "TOTAL", "810000"),
    "clock": ("governor            : schedutil", "MAXN_SUPER", "paired points       : 54"),
    "headline": (
        "Divergence spread max(R)/min(R)",
        "SPEC §7 primary (divergence spread): CONFIRMED",
        "SPEC §7 secondary (baseline 2 held-out transfer error): REFUTED",
        "argmax(R) = centroid",
        "argmin(R) = ingest",
        "(3, 0, 1, 5, 4, 2)",
    ),
    "baselines": ("BASELINE 1", "BASELINE 2", "BASELINE 3", "pass-unit 95% interval"),
    "drift": (
        "RAW MEASURED (headline)",
        "1.0752408447278534",
        "CONFOUNDING CHECK",
        "corr(order_index, W)",
    ),
}

#: Artifact names the tree snapshot looks for.
ARTIFACTS = ("__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis", ".config")


def _environment() -> dict[str, str]:
    return _bootstrap.subprocess_environment()


def _snapshot() -> set[str]:
    found: set[str] = set()
    for path in ROOT.rglob("*"):
        relative = str(path.relative_to(ROOT))
        if any(part in ARTIFACTS for part in path.relative_to(ROOT).parts):
            found.add(relative)
        elif path.suffix in (".pyc", ".pyo"):
            found.add(relative)
    return found


def main() -> int:
    tee = _bootstrap.Tee("validate_cli_output.txt")
    _bootstrap.header(tee, "P094 HilTimingDiff -- command-line interface")

    inherited = _snapshot()
    tee.rule("tree state at entry")
    tee(f"  cache and bytecode paths already present: {len(inherited)}")
    if inherited:
        tee(
            "  INHERITED, named and not failed on: a harness that ran ruff or pytest "
            "with its"
        )
        tee(
            "  working directory set to the product root leaves these before this "
            "script runs,"
        )
        tee("  and what a harness leaves in the tree is not the product's doing.")
        for item in sorted(inherited)[:10]:
            tee(f"    {item}")
        if len(inherited) > 10:
            tee(f"    ... and {len(inherited) - 10} more")

    results: dict[str, subprocess.CompletedProcess] = {}
    for name, arguments in COMMANDS.items():
        tee.rule(f"python -m hiltimingdiff {name} {' '.join(arguments)}".rstrip())
        result = subprocess.run(
            [sys.executable, "-m", "hiltimingdiff", name, *arguments],
            cwd=str(ROOT),
            env=_environment(),
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        results[name] = result
        lines = result.stdout.splitlines()
        for line in lines[: 40 if name in ("gate", "clock") else 28]:
            tee(f"  {line}")
        if len(lines) > 40:
            tee(f"  ... {len(lines) - 40} further lines")
        tee("")
        tee.check(f"{name} exits 0", result.returncode == 0, result.stderr.strip()[:200])
        for fragment in EXPECTED[name]:
            tee.check(
                f"{name} output contains {fragment!r}",
                fragment in result.stdout,
            )
        tee.check(
            f"{name} output carries no exclamation mark",
            "!" not in result.stdout,
        )
        separator = os.sep
        tee.check(
            f"{name} output carries no absolute home path",
            f"{separator}home{separator}" not in result.stdout
            and f"{separator}Users{separator}" not in result.stdout,
        )
        tee.check(
            f"{name} output states the research-grade limit",
            "Research-grade" in result.stdout,
        )
        tee.check(
            f"{name} output states the unpinned clock with the figures",
            "unpinned schedutil-governed clock" in result.stdout,
        )

    tee.rule("the refusal path")
    nowhere = Path(tempfile.mkdtemp(prefix="hiltimingdiff-cli-absent-")) / "nothing"
    refused = subprocess.run(
        [sys.executable, "-m", "hiltimingdiff", "gate", "--traces", str(nowhere)],
        cwd=str(ROOT),
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    tee(f"  exit status : {refused.returncode}")
    tee(f"  stderr      : {refused.stderr.strip()[:240]}...")
    tee("")
    tee.check("an absent trace directory exits 2", refused.returncode == 2)
    tee.check("the refusal is printed to stderr", "REFUSED" in refused.stderr)
    tee.check("stdout is empty on a refusal", refused.stdout.strip() == "")

    tee.rule("running from a different working directory")
    for where in (Path(os.sep), Path(tempfile.gettempdir()), ROOT, ROOT / "validation"):
        result = subprocess.run(
            [sys.executable, "-m", "hiltimingdiff", "gate"],
            cwd=str(where),
            env=_environment(),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        tee.check(
            f"gate exits 0 from {_bootstrap.relative(where)}",
            result.returncode == 0,
            result.stderr.strip()[:120],
        )

    tee.rule("what this script ADDS to the tree")
    added = _snapshot() - inherited
    tee(f"  cache and bytecode paths added by this script: {len(added)}")
    for item in sorted(added):
        tee(f"    {item}")
    tee("")
    tee.check(
        "running this script adds no cache directory and no bytecode file",
        not added,
        f"{len(added)} added",
    )

    tee.rule("negative controls")
    bad = subprocess.run(
        [sys.executable, "-m", "hiltimingdiff", "not-a-command"],
        cwd=str(ROOT),
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    tee.control(
        "an unknown command is rejected",
        bad.returncode != 0,
        must_fire=True,
        detail=f"exit {bad.returncode}",
    )
    helped = subprocess.run(
        [sys.executable, "-m", "hiltimingdiff", "--help"],
        cwd=str(ROOT),
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    tee.control(
        "--help is rejected",
        helped.returncode != 0,
        must_fire=False,
        detail=f"exit {helped.returncode}",
    )
    tee.check(
        "--help names every command",
        all(name in helped.stdout for name in COMMANDS),
    )

    path = tee.save()
    print(f"\nwritten: {path}")
    return 1 if tee.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
