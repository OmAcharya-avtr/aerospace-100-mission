"""Run the CLI as a real subprocess and record its exit codes verbatim.

The nonzero exit on an incomplete case is a contract: a continuous-integration
job keys on it. Checking it through ``main()`` in-process would not prove the
process exit status, so every row below is an actual ``python -m assuregraph``
invocation and the recorded number is ``returncode``.

Exit-code contract: 0 complete, 1 incomplete (``check`` only), 2 malformed input
or usage error.

Run: ``python validation/validate_cli.py``
Output: ``validation/validate_cli_output.txt``
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import _bootstrap

ENV = dict(os.environ, PYTHONPATH=_bootstrap.SRC)
COMPLETE = os.path.join(_bootstrap.EXAMPLES, "complete_case.yaml")
INCOMPLETE = os.path.join(_bootstrap.EXAMPLES, "incomplete_case.yaml")
CYCLIC = os.path.join(_bootstrap.EXAMPLES, "cyclic_case.yaml")


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "assuregraph", *args],
        cwd=_bootstrap.REPO_ROOT,
        env=ENV,
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> int:
    lines = [
        "CLI exit codes, measured from real subprocesses -- assuregraph 0.1.0",
        "",
        "Contract: 0 complete, 1 incomplete (check only), 2 malformed input or usage.",
        "",
    ]
    failures = 0

    with tempfile.TemporaryDirectory() as scratch:
        malformed = os.path.join(scratch, "malformed.yaml")
        with open(malformed, "w", encoding="utf-8") as handle:
            handle.write("name: empty case\nnodes: []\n")
        bad_yaml = os.path.join(scratch, "bad.yaml")
        with open(bad_yaml, "w", encoding="utf-8") as handle:
            handle.write("name: [unclosed\n")
        mmd = os.path.join(scratch, "out.mmd")

        probes: list[tuple[str, list[str], int]] = [
            ("--help", ["--help"], 0),
            ("--version", ["--version"], 0),
            ("check --help", ["check", "--help"], 0),
            ("mermaid --help", ["mermaid", "--help"], 0),
            ("summary --help", ["summary", "--help"], 0),
            ("no subcommand", [], 2),
            ("check complete_case.yaml", ["check", COMPLETE], 0),
            ("check complete_case.yaml --strict", ["check", COMPLETE, "--strict"], 0),
            ("check incomplete_case.yaml", ["check", INCOMPLETE], 1),
            ("check incomplete_case.yaml --json", ["check", INCOMPLETE, "--json"], 1),
            ("check incomplete_case.yaml --markdown", ["check", INCOMPLETE, "--markdown"], 1),
            ("check cyclic_case.yaml", ["check", CYCLIC], 1),
            ("check a missing file", ["check", os.path.join(scratch, "nope.yaml")], 2),
            ("check an empty-nodes case", ["check", malformed], 2),
            ("check unparsable YAML", ["check", bad_yaml], 2),
            ("summary complete_case.yaml", ["summary", COMPLETE], 0),
            ("summary incomplete_case.yaml", ["summary", INCOMPLETE], 0),
            ("mermaid incomplete_case.yaml", ["mermaid", INCOMPLETE], 0),
            ("mermaid --highlight", ["mermaid", INCOMPLETE, "--highlight"], 0),
            ("mermaid -o <file>", ["mermaid", COMPLETE, "-o", mmd], 0),
            ("mermaid a bad direction", ["mermaid", COMPLETE, "--direction", "ZZ"], 2),
            (
                "mermaid -o into a missing directory",
                ["mermaid", COMPLETE, "-o", os.path.join(scratch, "no", "x.mmd")],
                2,
            ),
        ]

        lines.append(f"{'invocation':<42} {'expected':>8} {'actual':>7}  verdict")
        lines.append("-" * 76)
        for label, args, expected in probes:
            result = run(args)
            ok = result.returncode == expected
            failures += not ok
            lines.append(
                f"{label:<42} {expected:>8} {result.returncode:>7}  "
                f"{'ok' if ok else 'FAIL'}"
            )

        lines.append("")
        lines.append(f"mermaid -o wrote {os.path.getsize(mmd)} bytes to a file")
        with open(mmd, encoding="utf-8") as handle:
            first = handle.readline().strip()
        lines.append(f"first line of that file: {first!r}")
        if first != "flowchart TD":
            failures += 1

        payload = json.loads(run(["check", INCOMPLETE, "--json"]).stdout)
        lines.extend(
            [
                "",
                "JSON output of `check incomplete_case.yaml --json`:",
                f"  exit_code in payload             : {payload['exit_code']}",
                f"  is_complete                      : {payload['is_complete']}",
                f"  counts                           : {payload['counts']}",
                f"  findings in payload              : {len(payload['findings'])}",
                f"  evidence entries in payload      : {len(payload['evidence'])}",
                "  fractions with their denominators:",
            ]
        )
        for key, value in payload["coverage"]["fractions"].items():
            denominator = payload["coverage"]["denominators"][key]
            shown = "undefined" if value is None else f"{value:.6f}"
            lines.append(f"    {key:<24} {shown:>10}  denominator {denominator}")
        if payload["exit_code"] != 1:
            failures += 1

        stderr_probe = run(["check", os.path.join(scratch, "nope.yaml")])
        lines.extend(
            [
                "",
                "A malformed input writes to stderr and nothing to stdout:",
                f"  stdout length                    : {len(stderr_probe.stdout)}",
                "  stderr (scratch directory redacted so the committed output is stable):",
                f"    {stderr_probe.stderr.strip().replace(scratch, '<scratch>')}",
            ]
        )
        if stderr_probe.stdout:
            failures += 1

        # argparse wraps the epilog, so compare against whitespace-normalised text.
        help_text = " ".join(run(["--help"]).stdout.split())
        for phrase in (
            "not a safe system",
            "not a DO-178C or ARP4754A compliance tool",
            "not flight-qualified",
        ):
            present = phrase in help_text
            lines.append(
                f"  --help (whitespace-normalised) contains {phrase!r}: "
                f"{'yes' if present else 'NO -- FAIL'}"
            )
            failures += not present

    lines.extend(["", f"FAILURES: {failures}"])
    target = __file__.replace(".py", "_output.txt")
    with open(target, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
