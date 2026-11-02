"""Every example run as a subprocess, with its printed numbers committed.

The README quotes numbers that the example scripts print -- the certified
overrun in the installation section, the recall figures under the
recall-against-cost screenshot -- and every number quoted anywhere in this
repository has to trace to a committed output file. The examples write PNGs,
not text, so this script runs each one, checks that it exits zero and that its
figure was written, and records its stdout verbatim here.

Each example is run with ``HOME`` and ``MPLCONFIGDIR`` in a temporary
directory outside the repository and with ``PYTHONDONTWRITEBYTECODE=1``,
because an earlier product of this mission let matplotlib write
``.config/matplotlib`` into its own tree through a subprocess environment it
set itself.
"""

from __future__ import annotations

# Bytecode must not be written anywhere inside this tree. See
# validation/validate_cli.py section 3; CPython writes a module's bytecode
# cache BEFORE executing it, so _bootstrap cannot set this on its own behalf.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import subprocess  # noqa: E402
import time  # noqa: E402

from _bootstrap import (  # noqa: E402
    Tee,
    bootstrap,
    environment_lines,
    relative,
    subprocess_environment,
)

ROOT = bootstrap()

from hilverdict.report import format_table  # noqa: E402

EXAMPLES = ROOT / "examples"
SCREENSHOTS = ROOT / "screenshots"


def main() -> int:
    out = Tee("validate_examples_output.txt")
    out.rule("hilverdict -- every example, run, with its printed numbers recorded")
    for line in environment_lines():
        out(line)
    out("")
    out("Each example is run from the repository root as a subprocess with HOME and")
    out("MPLCONFIGDIR in a temporary directory outside the repository.")
    environment = subprocess_environment()
    scripts = sorted(
        path for path in EXAMPLES.glob("*.py") if path.name != "_common.py"
    )
    rows = []
    transcripts: list[tuple[str, str]] = []
    for script in scripts:
        figure = SCREENSHOTS / f"{script.stem}.png"
        before = figure.stat().st_mtime if figure.exists() else 0.0
        started = time.perf_counter()
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, str(script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        elapsed = time.perf_counter() - started
        written = figure.exists() and figure.stat().st_mtime > before
        rows.append(
            [
                relative(script),
                str(completed.returncode),
                "yes" if written else "NO",
                f"{figure.stat().st_size // 1024 if figure.exists() else 0}",
                f"{elapsed:.1f}",
            ]
        )
        transcripts.append((relative(script), completed.stdout.strip()))
        out.check(
            f"{relative(script)} exits zero and writes its figure",
            completed.returncode == 0 and written,
            f"exit {completed.returncode}, figure "
            f"{'rewritten' if written else 'NOT rewritten'}",
        )
    out("")
    out(
        format_table(
            ["example", "exit", "figure rewritten", "size [KiB]", "elapsed [s]"],
            rows,
        )
    )
    out("")
    out("Elapsed seconds are UNSTABLE on shared hardware and are reported for")
    out("information only.")

    out.rule("what each example printed")
    for name, text in transcripts:
        out("")
        out(f"--- {name} ---")
        for line in text.splitlines():
            out(f"  {line}")
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
