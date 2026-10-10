"""Re-run every command quoted in README.md and the other documents, verbatim.

This script exists because an agent in an earlier batch of this portfolio wrote
a README output block for a command it had not run. Every fenced output block in
this repository's documentation is a paste from this file, so the claim "this is
what the command prints" is checkable by running one script.

The commands are listed in COMMANDS below in the order they appear in the
documents. If a document quotes a command that is not in this list, that is a
defect in the document.
"""

from __future__ import annotations

import os
import subprocess
import sys

from _harness import SRC, rel, run

#: Every command quoted in a document, with the document and section that quotes
#: it. Commands that calibrate at the full budget take about a minute each, so
#: the documents quote the reduced budget where the point being made does not
#: need the published precision, and say so in the surrounding prose.
COMMANDS: list[tuple[list[str], str]] = [
    (["--help"], "README.md - Command-line interface"),
    (["--version"], "README.md - Install and first run"),
    (["detectors"], "README.md - Command-line interface"),
    (["defaults"], "README.md - The problem / Validation evidence"),
    (["calibrate", "--budget", "quick"], "README.md - Command-line interface"),
    (["arl", "--change", "mean_step", "--magnitude", "1.0", "--replicates", "60",
      "--budget", "quick"], "README.md - Command-line interface"),
    (["arl", "--change", "transient", "--magnitude", "4.0", "--replicates", "60",
      "--budget", "quick"], "README.md - Change types"),
    (["transient", "--replicates", "60", "--budget", "quick"],
     "README.md - Change types"),
    (["tradeoff", "--detector", "ks", "--points", "4", "--span", "1.6",
      "--replicates", "40", "--budget", "quick"],
     "README.md - Limitations (the KS non-monotonicity)"),
    (["trace", "--change", "mean_step", "--magnitude", "1.0", "--post", "400",
      "--budget", "quick"], "README.md - Command-line interface"),
    (["detectors", "--json"], "README.md - Command-line interface"),
]

#: Non-CLI commands the documents quote, run here as well so the whole set is
#: covered by one script. Each is a module invocation, not a shell pipeline.
SCRIPTS: list[tuple[list[str], str]] = [
    (["examples/change_types.py"], "README.md - Screenshots"),
    (["validation/worked_example.py"], "README.md - A worked example"),
]


def body(report) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    # argparse wraps --help to the inferred terminal width and drops its
    # description when that width is very small. Pinning COLUMNS makes the
    # quoted help block reproducible rather than terminal-dependent.
    env["COLUMNS"] = "100"
    repo = SRC.parent

    report.section("1. Every CLI command quoted in a document")
    print(f"  {len(COMMANDS)} commands. COLUMNS=100, MPLBACKEND=Agg, "
          f"PYTHONPATH=<repository root>/{rel(SRC)}, cwd=<repository root>.")
    print()
    for args, where in COMMANDS:
        printable = " ".join(args)
        print("=" * 74)
        print(f"$ python -m telemdrift {printable}")
        print(f"  quoted in: {where}")
        print("=" * 74)
        res = subprocess.run(
            [sys.executable, "-m", "telemdrift", *args],
            capture_output=True, text=True, env=env, cwd=repo, timeout=900,
        )
        for line in res.stdout.rstrip("\n").splitlines():
            print(f"  {line}")
        if res.stderr.strip():
            for line in res.stderr.rstrip("\n").splitlines():
                print(f"  [stderr] {line}")
        print(f"  -> exit status {res.returncode}")
        print()
        report.check(f"`python -m telemdrift {printable}` exits 0 or 2 (a finding)",
                     res.returncode in (0, 2), f"exit {res.returncode}")

    report.section("2. Documented non-zero exit behaviour")
    print("  The documents state that exit status 2 means a finding and that an")
    print("  invalid invocation exits non-zero. Both are asserted on the return")
    print("  code here rather than described.")
    print()
    cases = [
        (["calibrate", "--target-arl0", "100000000", "--budget", "quick"], 2,
         "an unreachable ARL0 target cannot be bracketed"),
        (["arl", "--target-arl0", "100000000", "--replicates", "12",
          "--budget", "quick"], 2,
         "a censored ARL1 is only a lower bound"),
        (["kalman"], 2, "an unknown subcommand"),
        ([], 2, "no subcommand at all"),
        (["defaults", "--budget", "enormous"], 2, "an invalid budget"),
    ]
    for args, expected, why in cases:
        res = subprocess.run(
            [sys.executable, "-m", "telemdrift", *args],
            capture_output=True, text=True, env=env, cwd=repo, timeout=900,
        )
        printable = " ".join(args) if args else "(no arguments)"
        print(f"  $ python -m telemdrift {printable}")
        print(f"    {why}: exit status {res.returncode}, expected {expected}")
        tail = (res.stdout + res.stderr).rstrip().splitlines()
        for line in tail[-3:]:
            print(f"      {line}")
        report.check(f"`{printable}` exits {expected} ({why})",
                     res.returncode == expected, f"exit {res.returncode}")

    report.section("3. Every script quoted in a document")
    for args, where in SCRIPTS:
        print("=" * 74)
        print(f"$ python {' '.join(args)}")
        print(f"  quoted in: {where}")
        print("=" * 74)
        res = subprocess.run(
            [sys.executable, *args],
            capture_output=True, text=True, env=env, cwd=repo, timeout=900,
        )
        for line in res.stdout.rstrip("\n").splitlines():
            print(f"  {line}")
        if res.stderr.strip():
            for line in res.stderr.rstrip("\n").splitlines()[-5:]:
                print(f"  [stderr] {line}")
        print(f"  -> exit status {res.returncode}")
        print()
        report.check(f"`python {' '.join(args)}` exits 0", res.returncode == 0,
                     f"exit {res.returncode}")

    report.section("4. The cold-clone quickstart, as the README writes it")
    print("  The README's install block is `pip install -e '.[test]'` followed by")
    print("  pytest and an example. The pip step is not re-run here: this container")
    print("  already has the dependencies and a fresh install would add nothing to")
    print("  the evidence while costing minutes. What IS re-run is the pair of")
    print("  commands whose output the README quotes, from the repository root with")
    print("  only PYTHONPATH set -- which is what an editable install provides.")
    print()
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header",
         "-p", "no:cacheprovider"],
        capture_output=True, text=True, env=env, cwd=repo, timeout=1800,
    )
    print("  $ python -m pytest tests/ -q")
    for line in res.stdout.rstrip("\n").splitlines()[-6:]:
        print(f"    {line}")
    print(f"    -> exit status {res.returncode}")
    report.check("the test suite passes from the repository root",
                 res.returncode == 0, f"exit {res.returncode}")
    print()
    print("  The authoritative test count is read from junit XML, not from this")
    print("  line: a pytest configuration that collects nothing prints success.")
    print("  See validation/outputs/validate_tests.txt and its junit_report.xml.")


if __name__ == "__main__":
    raise SystemExit(run("validate_cli",
                         "telemdrift 0.1.0 - every quoted command, re-run",
                         body))
