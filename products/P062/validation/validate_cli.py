"""Validate the command-line interface: every invocation and its exit status.

Run from the repository root: ``python validation/validate_cli.py``
Writes ``validation/validate_cli_output.txt``.

The interface's one rule is checked here as well as in the test suite: no
subcommand prints a parameter value without the identifiability verdict for the
design it came from.

Exit statuses: 0 means nothing the command checked came out negative, 1 means
something did and is a RESULT rather than an error, 2 means argparse rejected
the arguments.
"""

from __future__ import annotations

import io
import time
from contextlib import redirect_stdout

from _bootstrap import Tee, add_src_to_path

ROOT = add_src_to_path()

from twincalib.__main__ import main as cli_main  # noqa: E402

INVOCATIONS = (
    (["designs"], 0),
    (["identify", "--design", "baseline-eclipse", "--exact"], 0),
    (["identify", "--design", "coarse-sampled", "--exact"], 0),
    (["identify", "--design", "lumped-nodes", "--exact"], 0),
    (["identify", "--design", "weak-excitation", "--exact"], 0),
    (["identify", "--design", "full-sun", "--exact"], 1),
    (["fit", "--design", "baseline-eclipse", "--exact", "--show-truth"], 0),
    (["fit", "--design", "full-sun", "--exact"], 1),
    (["fit", "--design", "lumped-nodes"], 0),
    (["profile", "--design", "coarse-sampled", "--exact", "--points", "3"], 0),
    (["profile", "--design", "full-sun", "--exact", "--points", "2"], 1),
    (["residual", "--design", "baseline-eclipse", "--n-train", "2", "--n-test", "1"], 0),
    (["residual", "--design", "lumped-nodes", "--n-train", "2", "--n-test", "1"], 0),
)


def main() -> None:
    started = time.perf_counter()
    out = Tee(ROOT / "validation" / "validate_cli_output.txt")
    out("validate_cli.py -- every CLI invocation, its exit status and its output")

    failures = 0
    out.rule("1. Exit statuses")
    out(f"  {'status':>7}{'expected':>10}  command")
    for argv, expected in INVOCATIONS:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            status = cli_main(argv)
        ok = status == expected
        failures += int(not ok)
        mark = "" if ok else "   <- MISMATCH"
        out(f"  {status:>7d}{expected:>10d}  python -m twincalib {' '.join(argv)}{mark}")
    out("")
    out(f"verdict: {'PASS' if failures == 0 else 'FAIL'}  {len(INVOCATIONS)} invocations")

    out.rule("2. --help and --version")
    for argv in (["--help"], ["--version"]):
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                cli_main(argv)
            code = 0
        except SystemExit as exit_code:
            code = int(exit_code.code or 0)
        out(f"  {' '.join(argv):<12} exit {code}")
        out(f"  first line: {buffer.getvalue().splitlines()[0]}")
    out("")
    out("  argparse exits 2 on a bad argument:")
    for argv in ([], ["identify", "--design", "nope"]):
        try:
            with redirect_stdout(io.StringIO()):
                cli_main(argv)
            out(f"    {argv} did NOT exit  <- defect")
            failures += 1
        except SystemExit as exit_code:
            out(f"    {argv} exited {int(exit_code.code or 0)}")

    out.rule("3. No parameter value is printed without the verdict above it")
    for argv in (
        ["fit", "--design", "baseline-eclipse", "--exact"],
        ["fit", "--design", "full-sun", "--exact"],
    ):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            cli_main(argv)
        text = buffer.getvalue()
        verdict_at = text.find("VERDICT")
        estimate_at = text.find("weighted least-squares estimate")
        ok = 0 <= verdict_at < estimate_at
        failures += int(not ok)
        out(
            f"  {' '.join(argv):<44} verdict at {verdict_at}, estimate at "
            f"{estimate_at}, ordered {ok}"
        )
    out("")
    out(f"verdict: {'PASS' if failures == 0 else 'FAIL'}")

    out.rule("4. Full output of the two invocations the README quotes")
    for argv in (
        ["identify", "--design", "full-sun", "--exact"],
        ["profile", "--design", "full-sun", "--exact", "--points", "2"],
    ):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            status = cli_main(argv)
        out("")
        out(f"$ python -m twincalib {' '.join(argv)}")
        out(buffer.getvalue().rstrip("\n"))
        out(f"[exit status {status}]")

    out("")
    out(f"total mismatches: {failures}")
    out(f"elapsed {time.perf_counter() - started:.1f} s")
    out.save()


if __name__ == "__main__":
    main()
