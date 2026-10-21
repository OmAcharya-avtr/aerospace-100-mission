"""The command-line interface, exercised exactly as a reader would.

Every invocation is run in-process through :func:`conjprob.__main__.main` with a
captured stream, so the exit status and the output are both checked. The case
paths are relative, which works from any directory because
``validation/_bootstrap.py`` changes to the repository root first; that is the
whole point of the bootstrap and this script is where it is exercised.
"""

from __future__ import annotations

import io

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

from conjprob.__main__ import main as cli_main  # noqa: E402

INVOCATIONS = (
    (
        ["assess", "examples/cases/leo_crossing.yaml"],
        0,
        ("foster", "chan", "alfano", "DilutionWarning"),
    ),
    (
        ["assess", "examples/cases/leo_crossing.yaml", "--monte-carlo", "200000"],
        0,
        ("monte_carlo", "CI ["),
    ),
    (
        ["assess", "examples/cases/large_body_tight_covariance.yaml"],
        0,
        ("disagreement (the primary output)", "aspect ratio exceeds 2"),
    ),
    (
        ["assess", "examples/cases/far_miss_tight_covariance.yaml"],
        0,
        ("Mahalanobis distance exceeds 5",),
    ),
    (
        ["dilution", "examples/cases/leo_crossing.yaml", "--points", "9"],
        0,
        ("maximum Pc", "pc_small_body"),
    ),
    (
        ["regimes", "--points", "20", "--aspects", "1.0", "10.0", "--mahalanobis", "1.0"],
        0,
        ("chan", "small_body", "nan means"),
    ),
    (
        ["screen", "--n-train", "6000", "--n-test", "3000"],
        0,
        ("mahalanobis", "learned", "recall"),
    ),
    (["assess", "examples/cases/does_not_exist.yaml"], 2, ()),
    (["screen", "--regime", "not-a-regime", "--n-train", "600", "--n-test", "600"], 2, ()),
)


def main() -> int:
    out = Tee("validate_cli_output.txt")
    out.rule("conjprob validation: command-line interface")
    for line in environment_lines():
        out(line)
    failures = 0

    out.rule("1. invocations, exit statuses and required output fragments")
    for argv, expected_status, fragments in INVOCATIONS:
        stream = io.StringIO()
        status = cli_main(argv, stream=stream)
        text = stream.getvalue()
        missing = [fragment for fragment in fragments if fragment not in text]
        ok = status == expected_status and not missing
        out("")
        out(f"python -m conjprob {' '.join(argv)}")
        out(f"  exit status {status} (expected {expected_status})")
        out(f"  output lines {len(text.splitlines())}")
        if fragments:
            out(
                f"  required fragments present: {len(fragments) - len(missing)} "
                f"of {len(fragments)}"
            )
        if missing:
            out(f"  MISSING: {missing}")
        out(f"  result: {'PASS' if ok else 'FAIL'}")
        failures += 0 if ok else 1

    out.rule("2. the assess report for the shipped leo-crossing case, verbatim")
    stream = io.StringIO()
    status = cli_main(["assess", "examples/cases/leo_crossing.yaml"], stream=stream)
    out(f"exit status {status}")
    out("")
    for line in stream.getvalue().splitlines():
        out(line)

    out.rule("3. no absolute path appears in this output")
    joined = "\n".join(out.lines)
    offenders = [
        token
        for token in joined.split()
        if token.startswith("/") and len(token) > 1 and not token.startswith("/ ")
    ]
    out(f"tokens beginning with a path separator: {len(offenders)}")
    if offenders:
        out(f"  {offenders[:10]}")
    ok = not offenders
    out(f"result: {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    out.rule("summary")
    out(f"checks failed: {failures}")
    out(f"written to {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
