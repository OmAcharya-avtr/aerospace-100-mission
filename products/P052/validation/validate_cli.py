"""Validate the CLI: exit statuses, and that nothing it prints reads as a pass.

Falsification is one-sided: finding no violation is not evidence of correctness.
A command-line tool that prints "PASS" when its search came up empty invites
exactly the misreading this product exists to prevent, so every subcommand's
output is scanned for a verdict vocabulary and the result is committed.

Exit statuses checked:

* ``--help`` exits 0, which the build guide requires.
* every subcommand exits 0 when it ran, whatever it found.
* ``--exit-on-violation`` exits 1 on a found violation and **0** when nothing
  was found, because the absence of a violation is never encoded in the exit
  status in either mode.
* a usage or validation error exits 2.
"""

from __future__ import annotations

import io
import re
import subprocess
import sys
from pathlib import Path

from _bootstrap import Tee, add_src_to_path

ROOT = add_src_to_path()

from falsifyloop.__main__ import main  # noqa: E402

OUT = Path(__file__).with_name("validate_cli_output.txt")
say = Tee(OUT)

FORBIDDEN = (
    "pass", "passes", "passed", "passing", "verified", "verification", "compliant",
    "compliance", "clean", "ok", "okay", "success", "successful", "proven",
    "guaranteed", "correct",
)
NEGATED_ONLY = (
    "safe", "certified", "flight-qualified", "flight-safe", "mission-ready",
    "production-ready", "approved",
)
_PATTERN = re.compile(r"\b(" + "|".join(FORBIDDEN) + r")\b", re.IGNORECASE)
_NEGATED = re.compile(r"(?<!not )\b(" + "|".join(NEGATED_ONLY) + r")\b", re.IGNORECASE)


def offending(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text)
    hits = {m.group(0).lower() for m in _PATTERN.finditer(flat)}
    hits |= {m.group(0).lower() for m in _NEGATED.finditer(flat)}
    return sorted(hits)


COMMANDS = [
    ("instances", ["instances"], 0),
    (
        "falsify easy (violation expected)",
        ["falsify", "--instance", "overshoot-loose", "--budget", "100", "--seed", "0"],
        0,
    ),
    (
        "falsify hard, tiny budget (nothing expected)",
        ["falsify", "--instance", "rate-envelope", "--budget", "5", "--seed", "0"],
        0,
    ),
    (
        "falsify hard, full budget (violation expected)",
        ["falsify", "--instance", "rate-envelope", "--budget", "100", "--seed", "0"],
        0,
    ),
    (
        "falsify with repeats and a curve",
        [
            "falsify", "--instance", "overshoot-tight", "--strategy", "surrogate-guided",
            "--budget", "40", "--repeats", "5", "--seed", "11",
        ],
        0,
    ),
    (
        "falsify --exit-on-violation, violation found",
        [
            "falsify", "--instance", "overshoot-loose", "--budget", "100", "--seed", "0",
            "--exit-on-violation",
        ],
        1,
    ),
    (
        "falsify --exit-on-violation, nothing found",
        [
            "falsify", "--instance", "rate-envelope", "--budget", "5", "--seed", "0",
            "--exit-on-violation",
        ],
        0,
    ),
    (
        "benchmark",
        [
            "benchmark", "--instances", "overshoot-loose", "rate-envelope",
            "--strategies", "uniform-random", "surrogate-guided",
            "--budget", "40", "--repeats", "5",
        ],
        0,
    ),
    (
        "evaluate, violating point",
        ["evaluate", "--instance", "overshoot-loose", "--input", "5,1.9,0.3,2.5,10,0.8"],
        0,
    ),
    (
        "evaluate, satisfying point",
        ["evaluate", "--instance", "rate-envelope", "--input", "1,0.6,1.2,0.6,0,0.1"],
        0,
    ),
    ("difficulty", ["difficulty", "--instances", "overshoot-loose", "--draws", "200"], 0),
    (
        "evaluate, wrong argument count",
        ["evaluate", "--instance", "overshoot-loose", "--input", "1,2,3"],
        2,
    ),
    (
        "evaluate, outside the declared box",
        ["evaluate", "--instance", "overshoot-loose", "--input", "99,1,1,1,1,1"],
        2,
    ),
    (
        "falsify, zero budget",
        ["falsify", "--instance", "overshoot-loose", "--budget", "0"],
        2,
    ),
    ("difficulty, zero draws", ["difficulty", "--draws", "0"], 2),
]


def main_script() -> int:
    say("falsifyloop - CLI validation")
    say(
        "Every command below was run in this session. Exit status 0 means the command "
        "ran; it never means anything was or was not found. Status 2 is a usage or "
        "validation error."
    )

    say.rule("Subprocess check: python -m falsifyloop --help")
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-m", "falsifyloop", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=str(ROOT),
    )
    say(f"exit status : {completed.returncode} (required: 0)")
    say(f"stdout bytes: {len(completed.stdout)}")
    say(f"forbidden vocabulary in --help: {offending(completed.stdout) or 'none'}")
    failures = 0 if completed.returncode == 0 else 1
    if offending(completed.stdout):
        failures += 1

    say.rule("Exit statuses and vocabulary, every subcommand")
    header = f"{'command':<46s} {'expected':>9s} {'actual':>7s} {'forbidden words':>28s}"
    say(header)
    say("-" * len(header))
    outputs = {}
    for label, argv, expected in COMMANDS:
        stream = io.StringIO()
        status = main(argv, out=stream)
        text = stream.getvalue()
        outputs[label] = text
        words = offending(text)
        if status != expected or words:
            failures += 1
        say(
            f"{label:<46s} {expected:>9d} {status:>7d} "
            f"{(', '.join(words) if words else 'none'):>28s}"
        )
    say("")
    say(f"CLI failures: {failures}")

    say.rule("Verbatim output: a search that found nothing")
    say(outputs["falsify hard, tiny budget (nothing expected)"].rstrip())

    say.rule("Verbatim output: a search that found a violation")
    say(outputs["falsify easy (violation expected)"].rstrip())

    say.rule("Verbatim output: the same hard instance at the full budget")
    say(
        "The same instance and seed as the no-violation block above, with the budget "
        "raised from 5 to 100. Committed because the README quotes the budget-5 form "
        "and a reader should be able to see what the other outcome looks like."
    )
    say("")
    say(outputs["falsify hard, full budget (violation expected)"].rstrip())

    say.rule("Verbatim output: evaluate at a satisfying point")
    say(outputs["evaluate, satisfying point"].rstrip())

    say.rule("SUMMARY")
    say(f"total CLI failures: {failures}")
    say("Falsification is one-sided: finding no violation is not evidence of correctness.")
    say.save()
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main_script())
