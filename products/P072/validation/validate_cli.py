"""The command-line interface, exercised and its output recorded.

Every subcommand is run in process through
:func:`downlinkpick.__main__.main` with its output captured, so the transcript
below is what a reader gets from the same command. Exit statuses are checked,
including the failure paths: a missing case file, a malformed case file and a
case with an undeclared value must each exit 2 with an actionable message on
standard error rather than a traceback.

The LP file written by the ``lp`` subcommand goes to a temporary directory and
is not committed. Its first lines are reproduced here with the temporary path
normalised out, because a committed transcript may not contain an absolute path.
"""

from __future__ import annotations

import io
import tempfile
from contextlib import redirect_stderr
from pathlib import Path

from _bootstrap import ROOT, Tee, bootstrap, relative

bootstrap()

from downlinkpick.__main__ import main as cli_main  # noqa: E402

CASE = "examples/cases/three_pass_declared.yaml"
MISSING_CASE = "examples/cases/missing_declarations.yaml"
SMALL_CASE = "examples/cases/single_short_pass.yaml"

COMMANDS = (
    (["--version"], 0, "version"),
    (["info"], 0, "versions and solver availability"),
    (["compare", CASE], 0, "greedy against exact on the reference case"),
    (["compare", SMALL_CASE], 0, "greedy against exact on the small case"),
    (["select", SMALL_CASE, "--method", "exact", "--delta-s", "30"], 0, "full report"),
    (["marginal", SMALL_CASE, "--delta-s", "60"], 0, "both selectors priced"),
    (
        ["stability", SMALL_CASE, "--trials", "8", "--sigmas", "0.0", "0.05", "0.2"],
        0,
        "stability curve",
    ),
)

FAILURES = (
    (["select", "examples/cases/does_not_exist.yaml"], 2, "missing case file"),
    (["select", MISSING_CASE], 2, "a case with undeclared values"),
    (["compare", "README.md"], 2, "a file that is not a case"),
)


def run(argv: list[str]) -> tuple[int, str, str]:
    """Run the CLI in process, returning ``(status, stdout, stderr)``."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            status = cli_main(argv, stream=stdout)
    except SystemExit as exit_code:  # --version exits through argparse
        status = int(exit_code.code or 0)
    return status, stdout.getvalue(), stderr.getvalue()


def main() -> int:
    out = Tee("validate_cli_output.txt")
    out.rule("downlinkpick validation: the command-line interface")
    out("every command below was run in process and its output captured verbatim")
    failures = 0

    for argv, expected, description in COMMANDS:
        rendered = "python -m downlinkpick " + " ".join(argv)
        out.rule(f"{rendered}   [{description}]")
        status, stdout, stderr = run(list(argv))
        out(stdout.rstrip("\n"))
        if stderr.strip():
            out("stderr:")
            out(stderr.rstrip("\n"))
        out(f"exit status: {status} (expected {expected})")
        if status != expected:
            failures += 1
            out(f"  FAIL expected exit {expected}, got {status}")

    out.rule("failure paths")
    out("each of these must exit 2 with a message on stderr and no traceback")
    out("")
    for argv, expected, description in FAILURES:
        rendered = "python -m downlinkpick " + " ".join(argv)
        status, stdout, stderr = run(list(argv))
        message = stderr.strip().splitlines()[0] if stderr.strip() else "(no stderr)"
        out(f"{rendered}")
        out(f"  description : {description}")
        out(f"  exit status : {status} (expected {expected})")
        out(f"  stderr      : {message}")
        if status != expected:
            failures += 1
            out(f"  FAIL expected exit {expected}, got {status}")
        if not stderr.strip():
            failures += 1
            out("  FAIL nothing was written to stderr")
        if "Traceback" in stderr:
            failures += 1
            out("  FAIL a traceback reached the user")

    out.rule("python -m downlinkpick lp   [the portable LP file]")
    out("the LP file is the statement of the MILP for a reader who has a solver this")
    out("environment does not. It is written to a temporary directory and not committed;")
    out("the temporary path is normalised out of the transcript below.")
    out("")
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "small.lp"
        status, stdout, stderr = run(["lp", SMALL_CASE, str(target)])
        out(f"exit status: {status}")
        if status != 0:
            failures += 1
            out(f"  FAIL lp subcommand exited {status}: {stderr.strip()}")
        else:
            text = target.read_text()
            out(f"lp file lines : {len(text.splitlines())}")
            out(f"lp file bytes : {len(text)}")
            out("first 14 lines:")
            for line in text.splitlines()[:14]:
                out(f"  {line}")
            if directory in text:
                failures += 1
                out("  FAIL the LP file contains the temporary directory path")
    out("")
    out("stdout of the lp subcommand, with the temporary path normalised: "
        "wrote <tmp>/small.lp")

    out.rule("absolute-path check on this transcript")
    out("no committed file in this repository may contain an absolute path")
    joined = "\n".join(out.lines)
    offenders = [
        token
        for token in joined.replace("(", " ").replace(")", " ").split()
        if token.startswith("/") and len(token) > 1
    ]
    out(f"tokens beginning with a path separator: {len(offenders)}")
    for token in offenders[:10]:
        out(f"  {token}")
    if offenders:
        failures += 1
        out("  FAIL the transcript contains an absolute path")
    out(f"repository root rendered relatively as: {relative(ROOT)!r}")

    out.rule("summary")
    out(f"failures: {failures}")
    out(f"output  : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
