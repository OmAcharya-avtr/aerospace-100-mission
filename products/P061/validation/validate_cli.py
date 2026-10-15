"""Every command-line invocation this repository documents, run and recorded.

The committed output of this script is where the README's quoted CLI output
comes from, so a README fragment and a real run cannot drift apart.

Also checked: no invocation prints a word this product is not allowed to print.
"""

from __future__ import annotations

import contextlib
import io
import time
from pathlib import Path

from _bootstrap import Tee, add_src_to_path, environment_lines

add_src_to_path()

from activityplan.__main__ import main as cli_main  # noqa: E402

OUTPUT = Path(__file__).with_name("validate_cli_output.txt")

FORBIDDEN = (
    "flight-safe",
    "flight safe",
    "mission-ready",
    "mission ready",
    "production-ready",
    "production ready",
    "revolutionary",
    "cutting-edge",
    "game-changing",
    "seamless",
    "!",
)

INVOCATIONS: tuple[tuple[str, ...], ...] = (
    ("instances",),
    ("declare", "--instance", "leo-imaging"),
    ("declare", "--instance", "tiny-enumerable"),
    ("plan", "--instance", "leo-imaging", "--planner", "greedy"),
    ("plan", "--instance", "leo-imaging", "--planner", "exact", "--time-limit", "150"),
    ("plan", "--instance", "energy-bound", "--planner", "greedy"),
    ("plan", "--instance", "recorder-bound", "--planner", "exact", "--time-limit", "150"),
    ("plan", "--instance", "window-bound", "--planner", "exact", "--time-limit", "150"),
    ("compare", "--instance", "leo-imaging", "--time-limit", "150"),
    ("compare", "--instance", "tiny-enumerable", "--time-limit", "150"),
    ("predict", "--n-train", "900", "--n-test", "300", "--mode", "normalised"),
    ("predict", "--n-train", "900", "--n-test", "300", "--mode", "absolute"),
    ("robust", "--n-train", "600", "--n-instances", "4", "--instance-seed", "9000"),
    (
        "warmstart",
        "--n-train",
        "600",
        "--n-instances",
        "4",
        "--instance-seed",
        "9000",
        "--time-limit",
        "150",
    ),
)


def main() -> int:
    say = Tee(OUTPUT)
    started = time.perf_counter()
    say("activityplan -- validate_cli.py")
    say("every documented command-line invocation, run and recorded")
    for line in environment_lines():
        say(line)

    failures = 0
    vocabulary_failures = 0
    for index, argv in enumerate(INVOCATIONS, start=1):
        say.rule(f"invocation {index}: python -m activityplan {' '.join(argv)}")
        buffer = io.StringIO()
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(buffer):
                status = cli_main(list(argv))
        except SystemExit as exit_info:  # pragma: no cover - argparse paths
            status = int(exit_info.code or 0)
        elapsed = time.perf_counter() - t0
        text = buffer.getvalue()
        say(text.rstrip("\n"))
        say("")
        say(f"exit status : {status}")
        say(f"wall clock  : {elapsed:.2f} s")
        if status != 0:
            failures += 1
            say("FAIL: non-zero exit status")
        lowered = text.lower()
        offending = [word for word in FORBIDDEN if word in lowered]
        if offending:
            vocabulary_failures += 1
            say(f"FAIL: forbidden vocabulary {offending}")

    say.rule("--help")
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            cli_main(["--help"])
    except SystemExit as exit_info:
        help_status = int(exit_info.code or 0)
    else:  # pragma: no cover - argparse always exits
        help_status = 0
    say(buffer.getvalue().rstrip("\n"))
    say("")
    say(f"exit status : {help_status}")
    if help_status != 0:
        failures += 1

    say.rule("summary")
    say(f"invocations run          : {len(INVOCATIONS)} plus --help")
    say(f"non-zero exit statuses   : {failures}")
    say(f"forbidden-word failures  : {vocabulary_failures}")
    say(f"wall clock               : {time.perf_counter() - started:.1f} s")
    overall = failures + vocabulary_failures
    say(f"overall                  : {'FAIL' if overall else 'PASS'}")
    say(f"raw output written to    : {OUTPUT.name}")
    say.save()
    return 1 if overall else 0


if __name__ == "__main__":
    raise SystemExit(main())
