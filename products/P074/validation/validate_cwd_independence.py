"""Every script in this directory and in examples/, run from four working directories.

Four products in one batch of this mission passed every check in their own
build and then failed the release gate, which runs each validation script with
its working directory set to the script's own directory rather than to the
product root. The build agent, its self-report and the coordinator's
re-measurement had all run the scripts from the product root: three
independent verifications that shared an assumption rather than a method.

This script is the method. It runs every sibling script and every example from
``/``, ``/tmp``, the product root and ``validation/`` (or ``examples/``), and
requires exit zero from all four. The runs set ``ORBITDRIFT_FAST=1``, which
shrinks the expensive studies and makes ``Tee.save`` write nothing, so a probe
can never replace a committed output with reduced-sample numbers.

It is the last validation script to run, and it runs nothing of its own: its
only output is the matrix of exit statuses.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile

from _bootstrap import Tee, Timer, bootstrap, environment_lines

ROOT = bootstrap()

SKIP = {"validate_cwd_independence.py", "_bootstrap.py", "_common.py"}


def main() -> int:
    out = Tee("validate_cwd_independence_output.txt")
    out.rule("orbitdrift -- every script runs from any working directory")
    for line in environment_lines():
        out(line)

    scripts: list[tuple[str, str]] = []
    for directory in ("validation", "examples"):
        for path in sorted((ROOT / directory).glob("*.py")):
            if path.name in SKIP:
                continue
            scripts.append((directory, path.name))

    with tempfile.TemporaryDirectory() as scratch:
        directories = [
            ("/", "/"),
            ("scratch", scratch),
            ("product root", str(ROOT)),
            ("own directory", None),
        ]
        out("")
        out("  working directories exercised:")
        for label, value in directories:
            if value is None:
                shown = "the script's own directory"
            elif value == "/":
                shown = "/"
            else:
                shown = "a temporary directory outside the repository"
            out(f"    {label:14s} {shown}")
        out("")
        out(f"  scripts: {len(scripts)}")
        out("")
        header = "  script".ljust(44) + "  ".join(label[:12].rjust(12) for label, _ in directories)
        out(header)

        timer = Timer()
        failures = 0
        for directory, name in scripts:
            statuses = []
            for _label, value in directories:
                cwd = str(ROOT / directory) if value is None else value
                with timer:
                    completed = subprocess.run(
                        [sys.executable, str(ROOT / directory / name)],
                        cwd=cwd,
                        env={
                            "PATH": "/usr/bin:/bin",
                            "PYTHONPATH": str(ROOT / "src"),
                            "MPLBACKEND": "Agg",
                            "ORBITDRIFT_FAST": "1",
                            "HOME": scratch,
                            "MPLCONFIGDIR": scratch,
                            "PYTHONDONTWRITEBYTECODE": "1",
                        },
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                statuses.append(completed.returncode)
                if completed.returncode != 0:
                    failures += 1
                    out(
                        f"  FAILED {directory}/{name} from "
                        f"{'own directory' if value is None else value}"
                    )
                    for line in completed.stderr.strip().splitlines()[-6:]:
                        out(f"    {line}")
            out(
                f"  {directory + '/' + name:42s}"
                + "  ".join(f"{status:12d}" for status in statuses)
            )
            out.check(
                f"{directory}/{name} exits zero from all four directories",
                all(status == 0 for status in statuses),
                ", ".join(str(status) for status in statuses),
            )

    out.rule("summary")
    out(f"  scripts                       : {len(scripts)}")
    out(f"  invocations                   : {timer.calls}")
    out(f"  non-zero exits                : {failures}")
    out.check("every invocation exited zero", failures == 0, f"{failures} failures")

    out.rule("what this probe does and does not leave behind")
    out("  ORBITDRIFT_FAST=1 makes Tee.save write nothing, so none of the runs")
    out("  above replaced a committed validation output with reduced-sample")
    out("  numbers.")
    out("")
    out("  The screenshots in screenshots/ ARE rewritten by the example runs, and")
    out("  in probe mode they are drawn from a 24-object subset rather than the")
    out("  full 192. That is stated here rather than quietly tolerated: after this")
    out("  script runs, the committed screenshots must be regenerated by running")
    out("  the examples without ORBITDRIFT_FAST set. The reproduction commands in")
    out("  README.md put the example runs after this one for that reason.")

    out.rule("timings (UNSTABLE: wall clock on a shared two-core container)")
    out(f"  {timer.calls} invocations          : {timer.elapsed_s:.1f} s total")
    out(f"  mean per invocation           : {timer.mean_s:.2f} s")

    out("")
    out(f"wrote {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    sys.exit(main())
