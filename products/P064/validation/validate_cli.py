"""Exercise every CLI subcommand and record its exit status and output.

    python validation/validate_cli.py > validation/validate_cli_output.txt
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

from _common import ROOT

from regressselect.__main__ import main as cli_main

SAMPLE = "fixtures/sample_project"
HISTORY = "fixtures/history/history.json"

CASES: list[tuple[str, list[str], int]] = [
    ("graph, no change set", ["graph", SAMPLE], 0),
    ("graph, one changed module", ["graph", SAMPLE, "--changed", "sampleproj.stats"], 0),
    ("graph, package init changed", ["graph", SAMPLE, "--changed", "sampleproj"], 0),
    (
        "select, dependency",
        ["select", SAMPLE, "--history", HISTORY, "--changed", "sampleproj.orbit",
         "--strategy", "dependency"],
        0,
    ),
    (
        "select, history",
        ["select", SAMPLE, "--history", HISTORY, "--changed", "sampleproj.stats",
         "--churn", "sampleproj.stats=12", "--budget", "0.25", "--strategy", "history"],
        0,
    ),
    (
        "select, learned",
        ["select", SAMPLE, "--history", HISTORY, "--changed", "sampleproj.quat",
         "--churn", "sampleproj.quat=31", "--budget", "0.25", "--strategy", "learned"],
        0,
    ),
    (
        "benchmark, table",
        ["benchmark", SAMPLE, "--history", HISTORY, "--budgets", "0.25", "1.0"],
        0,
    ),
    (
        "benchmark, json",
        ["benchmark", SAMPLE, "--history", HISTORY, "--budgets", "1.0", "--json"],
        0,
    ),
    (
        "select, malformed churn is rejected",
        ["select", SAMPLE, "--history", HISTORY, "--changed", "sampleproj.quat",
         "--churn", "sampleproj.quat"],
        2,
    ),
    (
        "benchmark, missing history is rejected",
        ["benchmark", SAMPLE, "--history", "fixtures/history/absent.json"],
        2,
    ),
    (
        "graph, missing root is rejected",
        ["graph", "fixtures/not_a_project"],
        2,
    ),
]


def _rel(text: str) -> str:
    """Strip the absolute repository-root prefix out of captured CLI output.

    The CLI resolves a project root before reporting it, so its own error text
    carries this machine's absolute path. Committed validation output must not:
    it is not reproducible for a reader, and an absolute home-directory path in
    a tracked file is exactly what release-gate check 8 exists to catch. The
    product's behaviour is unchanged -- only what is written to the committed
    transcript is normalised, and the normalisation is stated here.
    """
    repo = str(ROOT.parent.parent)
    return text.replace(repo + "/", "").replace(repo, ".")


def main() -> int:
    out = sys.stdout
    os.chdir(ROOT)
    failures = 0
    out.write("CLI subcommand check: python -m regressselect\n")
    out.write("=" * 88 + "\n\n")

    with tempfile.TemporaryDirectory() as tmp:
        order_path = Path(tmp) / "order.json"
        det_path = Path(tmp) / "detecting.json"
        order_path.write_text(json.dumps(["t1", "t2", "t3", "t4", "t5"]), encoding="utf-8")
        det_path.write_text(json.dumps({"f1": ["t1"], "f2": ["t3"]}), encoding="utf-8")
        cases = [
            *CASES,
            (
                "apfd, hand-worked n=5 m=2 TF=(1,3) -> 0.7",
                ["apfd", "--order", str(order_path), "--detecting", str(det_path)],
                0,
            ),
        ]
        for label, argv, expected_code in cases:
            buffer = io.StringIO()
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                code = cli_main(argv, out=buffer)
            ok = code == expected_code
            failures += 0 if ok else 1
            text = _rel(buffer.getvalue())
            out.write(f"--- {label}\n")
            out.write(f"    argv            {argv}\n")
            out.write(f"    exit status     {code} (expected {expected_code}) ")
            out.write(f"{'PASS' if ok else 'FAIL'}\n")
            out.write(f"    output lines    {len(text.splitlines())}\n")
            for line in text.splitlines()[:6]:
                out.write(f"    | {line}\n")
            if len(text.splitlines()) > 6:
                out.write(f"    | ... {len(text.splitlines()) - 6} more lines\n")
            for line in _rel(errors.getvalue()).splitlines():
                out.write(f"    ! {line}\n")
            out.write("\n")

    out.write(f"checks run {len(CASES) + 1}, failures {failures}\n")
    out.write("RESULT: " + ("PASS\n" if failures == 0 else f"FAIL ({failures})\n"))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
