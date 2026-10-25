"""Every CLI invocation the README quotes, executed, with its exit code recorded.

Each command is run as a subprocess with ``PYTHONPATH=src`` and nothing else on
``sys.path``, which is how the release gate runs them. The command strings
appear here exactly as the README quotes them, so a README command that stopped
working would fail this script rather than a reader's first attempt.
"""

from __future__ import annotations

import subprocess
import sys

from _bootstrap import ROOT, Tee, Timer, bootstrap, header

bootstrap()

#: Every invocation the README quotes, plus the ones the gate runs.
COMMANDS: tuple[tuple[str, ...], ...] = (
    ("--version",),
    ("--help",),
    ("info",),
    ("show", "smallsat_bus"),
    ("show", "voting_triple", "--enumerate"),
    ("quantify", "smallsat_bus"),
    ("quantify", "smallsat_bus", "--mission", "leo_phased"),
    ("quantify", "redundant_pair", "--hours", "8766"),
    ("compare", "smallsat_bus"),
    ("compare", "voting_triple"),
    ("compare", "warm_standby"),
    ("importance", "voting_triple"),
    ("importance", "load_sharing_pair"),
    ("mission", "smallsat_bus", "--mission", "leo_phased"),
    ("simulate", "redundant_pair", "--trials", "200000", "--seed", "1"),
    ("bridge",),
    ("compare", "--diagram", "examples/diagrams/smallsat_bus.yaml"),
    (
        "quantify",
        "--diagram",
        "examples/diagrams/voting_triple.yaml",
        "--mission-file",
        "examples/diagrams/one_year.yaml",
    ),
    ("show", "--diagram", "examples/diagrams/load_sharing_pair.yaml"),
)

#: Invocations that must FAIL, with the exit code they must not return.
MUST_FAIL: tuple[tuple[str, ...], ...] = (
    (),
    ("show", "not_a_case"),
    ("quantify", "--diagram", "examples/diagrams/does_not_exist.yaml"),
)


def _run(arguments: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "reliablock", *arguments],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src", "MPLBACKEND": "Agg"},
        timeout=600,
    )


def main() -> int:
    tee = Tee("validate_cli_output.txt")
    header(tee, "every CLI invocation the README quotes")
    tee("")
    tee("  Run as a subprocess with PYTHONPATH=src and nothing else, which is")
    tee("  how the release gate runs them. Commands are quoted relatively; no")
    tee("  absolute path appears in this file.")

    timer = Timer()

    tee.rule("invocations that must succeed")
    tee("")
    tee(f"  {'command':<66s}{'exit':>6s}{'stdout lines':>14s}{'stderr':>8s}")
    tee(f"  {'-' * 94}")
    failures: list[str] = []
    with timer:
        for arguments in COMMANDS:
            rendered = "python -m reliablock " + " ".join(arguments)
            result = _run(arguments)
            lines = len(result.stdout.splitlines())
            tee(
                f"  {rendered:<66s}{result.returncode:>6d}{lines:>14d}"
                f"{len(result.stderr):>8d}"
            )
            if result.returncode != 0:
                failures.append(rendered)
                for line in (result.stderr or "").strip().splitlines()[-3:]:
                    tee(f"      {line}")
    tee("")
    tee(f"  invocations                        : {len(COMMANDS)}")
    tee(f"  non-zero exits                     : {len(failures)}")
    tee.check(
        "every documented CLI invocation exits zero",
        not failures,
        f"{len(failures)} of {len(COMMANDS)} failed",
    )

    tee.rule("invocations that must fail")
    tee("")
    tee("  A CLI that exits zero on a bad argument is worse than one that")
    tee("  crashes, so the refusals are checked too.")
    tee("")
    tee(f"  {'command':<66s}{'exit':>6s}{'refused':>10s}")
    tee(f"  {'-' * 84}")
    wrongly_succeeded: list[str] = []
    with timer:
        for arguments in MUST_FAIL:
            rendered = "python -m reliablock " + " ".join(arguments)
            result = _run(arguments)
            refused = result.returncode != 0
            tee(f"  {rendered:<66s}{result.returncode:>6d}{str(refused):>10s}")
            if not refused:
                wrongly_succeeded.append(rendered)
    tee("")
    tee.check(
        "every malformed invocation exits non-zero",
        not wrongly_succeeded,
        f"{len(wrongly_succeeded)} wrongly succeeded",
    )

    tee.rule("the import check the gate runs")
    tee("")
    with timer:
        imported = subprocess.run(
            [sys.executable, "-c", "import reliablock; print(reliablock.__version__)"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src"},
            timeout=300,
        )
    tee(f"  python -c 'import reliablock'      : exit {imported.returncode}")
    tee(f"  version printed                    : {imported.stdout.strip()}")
    tee.check(
        "the package imports in a clean interpreter with only PYTHONPATH=src",
        imported.returncode == 0,
        (imported.stderr or "").strip().splitlines()[-1] if imported.returncode else "",
    )

    tee.rule("AI_ENABLED is false and stays false")
    tee("")
    tee("  This product has no learned component, deliberately. The flag is")
    tee("  asserted here so that adding one would fail a validation script and")
    tee("  not only a review.")
    with timer:
        flag = subprocess.run(
            [sys.executable, "-c", "import reliablock; print(reliablock.AI_ENABLED)"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src"},
            timeout=300,
        )
    tee(f"  reliablock.AI_ENABLED              : {flag.stdout.strip()}")
    tee.check("AI_ENABLED is False", flag.stdout.strip() == "False", flag.stdout.strip())
    tee("")
    tee("  and no machine-learning library is imported anywhere in src/:")
    offenders: list[str] = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        text = path.read_text()
        for library in ("sklearn", "torch", "tensorflow", "keras", "xgboost", "lightgbm"):
            if f"import {library}" in text or f"from {library}" in text:
                offenders.append(f"{path.name}: {library}")
    tee(f"  machine-learning imports found     : {len(offenders)}")
    for offender in offenders:
        tee(f"    {offender}")
    tee.check("no machine-learning library is imported in src/", not offenders)

    tee.rule("timing, declared unstable")
    tee("")
    tee("  | quantity | value | stability |")
    tee("  |---|---|---|")
    tee(f"  | all subprocess invocations | {timer.elapsed_s:.2f} s | UNSTABLE, shared hardware |")

    tee.summarise_failures()
    tee("")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
