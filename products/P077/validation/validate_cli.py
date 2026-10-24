"""The command line, exercised in process so no absolute path can appear.

``relfiledelivery.__main__.main`` is called with explicit argument lists and
its stdout captured, rather than launched as a subprocess. That is deliberate:
a subprocess would be invoked through an interpreter path, and an interpreter
path on this machine is an absolute path, which no committed file in this
repository may contain.
"""

from __future__ import annotations

import contextlib
import io

from _bootstrap import Tee, bootstrap, environment_lines, relative

bootstrap()

from relfiledelivery.__main__ import main  # noqa: E402

out = Tee("validate_cli_output.txt")

INVOCATIONS: list[tuple[str, list[str], list[str]]] = [
    (
        "info",
        ["info"],
        ["cfdp-py", "spacepackets", "NOT a CFDP implementation", "Not flight-qualified"],
    ),
    (
        "analytic, deferred-NAK",
        ["analytic", "--file-bytes", "262144", "--loss", "0.05", "--mode", "deferred_nak"],
        ["E[rounds]", "series tail bound", "E[delivery time]"],
    ),
    (
        "analytic, unacknowledged",
        ["analytic", "--file-bytes", "262144", "--loss", "0.05", "--mode", "unacknowledged"],
        ["completion probability"],
    ),
    (
        "simulate one run",
        ["simulate", "--file-bytes", "262144", "--loss", "0.05", "--seed", "20261024"],
        ["source digest", "forward transmissions", "retransmissions"],
    ),
    (
        "simulate over contact windows",
        [
            "simulate",
            "--file-bytes", "262144",
            "--loss", "0.05",
            "--windows", "6",
            "--window-duration-s", "4",
            "--window-period-s", "10",
            "--mean-up-s", "1.5",
            "--mean-down-s", "1.0",
        ],
        ["6 windows", "outage process"],
    ),
    (
        "distribution with the closed-form check",
        ["distribution", "--file-bytes", "262144", "--loss", "0.05", "--runs", "300"],
        ["tail, stated separately", "difference / std error"],
    ),
    (
        "compare every mode",
        ["compare", "--file-bytes", "262144", "--loss", "0.05", "--runs", "200"],
        ["unacknowledged", "deferred_nak", "immediate_nak"],
    ),
]

out.rule("environment")
for line in environment_lines():
    out(line)

out.rule("invocations")
failures: list[str] = []
transcripts: list[tuple[str, list[str], str]] = []
for label, argv, expected in INVOCATIONS:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(argv)
    text = buffer.getvalue()
    missing = [token for token in expected if token not in text]
    ok = code == 0 and not missing
    if not ok:
        failures.append(f"{label}: exit {code}, missing {missing}")
    out(f"{'PASS' if ok else 'FAIL'}  {label:<42} exit {code}  {len(text.splitlines())} lines")
    transcripts.append((label, argv, text))

out("")
out(f"invocations run                : {len(INVOCATIONS)}")
out(f"failures                       : {len(failures)}")
for failure in failures:
    out(f"  FAILED {failure}")

out.rule("rejected input")
out("A measurement harness has to refuse a scenario it cannot measure rather")
out("than measure a different one.")
out("")
rejections: list[tuple[str, list[str], str]] = [
    (
        "closed form with outages declared",
        ["analytic", "--mean-up-s", "10", "--mean-down-s", "5"],
        "assumes no outages",
    ),
    (
        "closed form with contact windows declared",
        ["analytic", "--windows", "3"],
        "continuous contact",
    ),
    ("unknown transfer mode", ["simulate", "--mode", "selective_repeat"], "unknown transfer mode"),
    ("half-declared outage process", ["simulate", "--mean-up-s", "10"], "must be given together"),
    ("loss probability of one", ["simulate", "--loss", "1.0"], r"[0, 1)"),
    ("negative light time", ["simulate", "--light-time-s", "-1"], "one_way_light_time_s"),
    ("zero-byte file", ["simulate", "--file-bytes", "0"], "file_bytes"),
]
rejection_failures: list[str] = []
for label, argv, fragment in rejections:
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            main(argv)
    except (ValueError, TypeError) as error:
        message = str(error)
        ok = fragment in message
        shown = message.splitlines()[0][:58]
    else:  # pragma: no cover - a silent acceptance is the failure being guarded
        ok = False
        shown = "accepted"
    if not ok:
        rejection_failures.append(f"{label}: {shown}")
    out(f"{'PASS' if ok else 'FAIL'}  {label:<42} {shown}")
out("")
out(f"rejections checked             : {len(rejections)}")
out(f"failures                       : {len(rejection_failures)}")

out.rule("full transcript: python -m relfiledelivery info")
for line in transcripts[0][2].splitlines():
    out(line)

out.rule("full transcript: python -m relfiledelivery compare --file-bytes 262144 "
        "--loss 0.05 --runs 200")
for line in transcripts[-1][2].splitlines():
    out(line)

out.rule("summary")
checks = {
    "every invocation exits zero and prints what it claims": not failures,
    "every invalid scenario is refused with an actionable message": not rejection_failures,
}
for name, passed in checks.items():
    out(f"{'PASS' if passed else 'FAIL'}  {name}")
out("")
out(f"output written to {relative(out.path)}")
out.save()
if not all(checks.values()):
    raise SystemExit("validate_cli.py: at least one check FAILED")
