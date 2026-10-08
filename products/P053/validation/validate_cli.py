"""Validate that every CLI entry point runs and returns the right exit status."""

from __future__ import annotations

import contextlib
import io

from _bootstrap import add_src_to_path

add_src_to_path()

from twininvalidate.__main__ import main  # noqa: E402

HELP_CASES = [
    ["--help"],
    ["twin", "--help"],
    ["scenarios", "--help"],
    ["calibrate", "--help"],
    ["curve", "--help"],
    ["benchmark", "--help"],
    ["ambiguity", "--help"],
]

RUN_CASES = [
    ["twin"],
    ["scenarios"],
    ["calibrate", "--runs", "40", "--samples", "800", "--target", "300"],
    ["curve", "parameter_step", "--runs", "40", "--samples", "800", "--points", "4"],
    ["ambiguity", "--runs", "10", "--samples", "600"],
]

ERROR_CASES = [
    ([], "no subcommand"),
    (["curve", "nonsense"], "unknown scenario"),
    (["calibrate", "--runs", "2", "--samples", "20", "--target", "1.0"], "target below 1"),
]


def run(argv: list[str]) -> tuple[int, int]:
    """Run the CLI, returning ``(exit status, characters of output)``."""
    buf = io.StringIO()
    err = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            status = main(argv)
    except SystemExit as exc:
        status = int(exc.code or 0)
    return status, len(buf.getvalue()) + len(err.getvalue())


def main_script() -> int:
    results: list[bool] = []
    print("=== --help must exit 0 and print something ===")
    for argv in HELP_CASES:
        status, chars = run(argv)
        ok = status == 0 and chars > 0
        results.append(ok)
        print(f"  {' '.join(argv):<28}exit {status}  {chars:>6} chars  "
              f"{'PASS' if ok else 'FAIL'}")

    print()
    print("=== subcommands must run and exit 0 ===")
    for argv in RUN_CASES:
        status, chars = run(argv)
        ok = status == 0 and chars > 0
        results.append(ok)
        print(f"  {' '.join(argv):<60}exit {status}  {chars:>6} chars  "
              f"{'PASS' if ok else 'FAIL'}")

    print()
    print("=== invalid input must exit non-zero with a message, not a traceback ===")
    for argv, why in ERROR_CASES:
        status, chars = run(argv)
        ok = status != 0 and chars > 0
        results.append(ok)
        print(f"  {why:<28}exit {status}  {chars:>6} chars  {'PASS' if ok else 'FAIL'}")

    print()
    print(f"SUMMARY  {sum(results)}/{len(results)} checks PASS")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main_script())
