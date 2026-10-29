"""Section 7: the command line, run as a real subprocess from a clean environment.

Each subcommand is spawned with a temporary ``HOME`` and ``MPLCONFIGDIR`` outside
the repository and with ``PYTHONDONTWRITEBYTECODE=1`` set, because a validation
script in an earlier product of this mission set ``HOME`` to the product root
and matplotlib wrote ``.config/matplotlib`` into the repository. After the runs,
the tree is checked for exactly that kind of pollution.
"""

from __future__ import annotations

import subprocess
import sys

from _bootstrap import ROOT, Tee, bootstrap, environment_lines, subprocess_environment

bootstrap()

COMMANDS: tuple[tuple[str, list[str], tuple[str, ...]], ...] = (
    ("version", ["--version"], ("aligncal 0.1.0",)),
    (
        "info",
        ["info"],
        (
            "observability first",
            "(8)       the observability Gramian",
            "(11)      the noise floor",
            "not flight-qualified",
        ),
    ),
    (
        "observability",
        ["observability", "--profile", "single_axis", "--samples", "800"],
        (
            "OBSERVABILITY VERDICT",
            "RANK DEFICIENT",
            "does NOT support",
            "NO ESTIMATE WAS COMPUTED",
        ),
    ),
    (
        "align, rich record",
        ["align", "--profile", "three_axis", "--samples", "800"],
        ("BATCH ESTIMATE", "injected truth", "error on the observable subspace"),
    ),
    (
        "align, single-axis record",
        ["align", "--profile", "single_axis", "--samples", "800"],
        ("WITHHELD", "the record does not support it", "returned exactly 0"),
    ),
    (
        "align, three secondaries",
        ["align", "--samples", "500", "--secondaries", "3"],
        ("secondary_1", "secondary_2", "secondary_3"),
    ),
    (
        "compare",
        ["compare", "--profile", "single_axis", "--samples", "600"],
        ("BATCH AGAINST RECURSIVE", "mekf sigma", "withheld by the batch solution"),
    ),
    (
        "unobservable",
        [
            "unobservable",
            "--profile",
            "single_axis",
            "--samples",
            "150",
            "--train",
            "400",
            "--configuration",
            "bagged_ridge",
        ],
        (
            "injected truth",
            "inf (refused)",
            "the analytic ratio is infinite",
            "times larger along the unobservable direction",
        ),
    ),
    (
        "benchmark",
        ["benchmark", "--train", "300", "--test", "100", "--samples", "150"],
        ("SHORT-RECORD BENCHMARK", "trivial (mean)", "lowest total RMSE"),
    ),
    (
        "a rejected argument",
        ["observability", "--amplitude", "-1"],
        ("amplitude must be finite",),
    ),
    (
        "a rejected floor factor",
        ["observability", "--floor-factor", "0.25"],
        ("at least 1",),
    ),
)

#: Commands expected to exit non-zero, with the status they must exit with.
EXPECTED_STATUS = {"a rejected argument": 2, "a rejected floor factor": 2}

POLLUTION = (".config", ".pytest_cache", ".ruff_cache", ".hypothesis", "__pycache__")


def main() -> int:
    out = Tee("validate_cli_output.txt")
    out("VALIDATION 7 -- THE COMMAND LINE")
    out("")
    for line in environment_lines():
        out(f"  {line}")
    out("")
    out(
        "  every command below is spawned as `python3 -m aligncal ...` with a "
        "temporary HOME and MPLCONFIGDIR outside the repository and "
        "PYTHONDONTWRITEBYTECODE=1 set"
    )

    out.rule("7.1 every subcommand")
    out("")
    out(f"{'command':<26s} {'exit':>5s} {'expected':>9s} {'stdout lines':>13s} "
        f"{'fragments found':>16s}")
    environment = subprocess_environment()
    for label, arguments, fragments in COMMANDS:
        completed = subprocess.run(
            [sys.executable, "-m", "aligncal", *arguments],
            cwd=str(ROOT),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        expected = EXPECTED_STATUS.get(label, 0)
        text = completed.stdout + completed.stderr
        found = sum(1 for fragment in fragments if fragment in text)
        out(
            f"{label:<26s} {completed.returncode:>5d} {expected:>9d} "
            f"{len(completed.stdout.splitlines()):>13d} "
            f"{f'{found}/{len(fragments)}':>16s}"
        )
        out.check(
            f"{label}: exit status {expected}",
            completed.returncode == expected,
            f"got {completed.returncode}",
        )
        missing = [fragment for fragment in fragments if fragment not in text]
        out.check(
            f"{label}: every expected fragment present",
            not missing,
            f"missing {missing}" if missing else f"{found} of {len(fragments)}",
        )

    out.rule("7.2 the tree was not polluted by any of those subprocesses")
    found_pollution = []
    for pattern in POLLUTION:
        for path in ROOT.rglob(pattern):
            found_pollution.append(str(path.relative_to(ROOT)))
    out(f"  patterns checked : {', '.join(POLLUTION)}")
    out(f"  entries found    : {len(found_pollution)}")
    for entry in sorted(found_pollution):
        out(f"    {entry}")
    out("")
    out(
        "  __pycache__ and the tool caches are produced by running the test suite and "
        "the scripts, and every one of them is covered by an anchored pattern in "
        ".gitignore. What must NOT appear is .config, which is what a subprocess with a "
        "mis-set HOME writes."
    )
    out.check(
        "no .config directory anywhere in the tree",
        not any(entry.split("/")[-1] == ".config" for entry in found_pollution),
        f"{sum(1 for entry in found_pollution if entry.split('/')[-1] == '.config')} found",
    )

    out.rule("7.3 the CLI's own refusal language")
    completed = subprocess.run(
        [sys.executable, "-m", "aligncal", "info"],
        cwd=str(ROOT),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    forbidden = (
        "flight-safe",
        "flight safe",
        "certified",
        "mission-ready",
        "production-ready",
        "calibrated alignment",
    )
    present = [word for word in forbidden if word in completed.stdout.lower()]
    out(f"  forbidden phrases searched : {', '.join(forbidden)}")
    out(f"  present                    : {present if present else 'none'}")
    out.check(
        "the CLI makes no flight-safety or certification claim",
        not [
            word
            for word in present
            if word != "certified"
            or "not certified" not in completed.stdout.lower()
        ],
        "the only occurrence of 'certified' is in the phrase 'not certified'",
    )

    path = out.save()
    print(f"\nwritten to {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
