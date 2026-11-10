"""Every example runs, writes its figure, and prints the fragments README quotes.

Each example is executed as a SUBPROCESS, because that is how the release gate
runs them and because in-process execution shares a matplotlib state that an
unattended run does not have. Every subprocess gets ``HOME`` and ``MPLCONFIGDIR``
in a temporary directory outside the repository and ``PYTHONDONTWRITEBYTECODE``,
so that no example can pollute the tree it is being measured in.

The figure files are checked for existence and size AFTER the run, and the
script reports the size rather than asserting a magic number: a figure that
collapses to a few kilobytes is usually an empty axis, and the size is the
cheapest signal of that.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from _bootstrap import (  # noqa: E402
    Tee,
    bootstrap,
    environment_lines,
    relative,
    subprocess_environment,
)

ROOT = bootstrap()

from sitediverse.report import format_table, number  # noqa: E402

#: Each example, the figure it must write, and fragments of its own output that
#: README.md quotes. A missing fragment means README.md has drifted.
EXAMPLES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "overstatement_by_separation.py",
        "overstatement_by_separation.png",
        ("pairs                          : 190", "mean ratio under 300 km",
         "largest pair ratio", "pairs with a ratio below 1"),
    ),
    (
        "network_availability.py",
        "network_availability.png",
        ("within one group (<300 km)", "one per group (>1500 km)",
         "nines claimed"),
    ),
    (
        "correlation_models.py",
        "correlation_models.png",
        ("pooled fit rho_0, L", "held-out SITES MAE vs truth, learned"),
    ),
    (
        "holdout_scores.py",
        "holdout_scores.png",
        ("held-out SITES", "held-out TIME", "reference half-width"),
    ),
    (
        "decomposition.py",
        "decomposition.png",
        ("co-modulation", "identity residual"),
    ),
    (
        "learned_interval.py",
        "learned_interval.png",
        ("nominal coverage", "quantile crossings, all rows"),
    ),
    (
        "record_structure.py",
        "record_structure.png",
        ("declared marginal range", "worst climatology error"),
    ),
)


def main() -> int:
    out = Tee("validate_examples_output.txt")
    out.rule("sitediverse -- every example, as a subprocess, with its figure")
    for line in environment_lines():
        out(line)
    out("")
    out("Every subprocess runs with HOME and MPLCONFIGDIR in a temporary directory")
    out("outside the repository and PYTHONDONTWRITEBYTECODE=1, so no example can")
    out("pollute the tree it is being measured in.")
    environment = subprocess_environment()
    rows = []
    for name, figure, fragments in EXAMPLES:
        target = ROOT / "screenshots" / figure
        before = target.stat().st_mtime if target.exists() else 0.0
        started = time.perf_counter()
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, str(ROOT / "examples" / name)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        elapsed = time.perf_counter() - started
        exists = target.exists()
        rewritten = exists and target.stat().st_mtime > before
        size = target.stat().st_size if exists else 0
        missing = [
            fragment
            for fragment in fragments
            if fragment not in completed.stdout
        ]
        passed = (
            completed.returncode == 0 and exists and rewritten and not missing
        )
        rows.append(
            [
                name,
                str(completed.returncode),
                f"{elapsed:.1f}",
                relative(target) if exists else "MISSING",
                f"{size / 1024.0:.1f}",
                "ok" if passed else "FAILED",
            ]
        )
        detail = f"exit {completed.returncode}"
        if not exists:
            detail += "; figure MISSING"
        elif not rewritten:
            detail += "; figure not rewritten"
        if missing:
            detail += f"; missing fragments {missing}"
        if completed.returncode != 0 and completed.stderr.strip():
            detail += f"; {completed.stderr.strip().splitlines()[-1][:100]}"
        out.check(f"examples/{name}", passed, detail)
    out("")
    out(format_table(
        ["example", "exit", "seconds", "figure", "size KiB", "verdict"], rows
    ))
    out("")
    total = sum(float(row[2]) for row in rows)
    out(f"  examples                 : {len(EXAMPLES)}")
    out(f"  total wall clock         : {number(total, 1)} s")
    out.check(
        "every example finishes inside the release gate's per-script limit",
        all(float(row[2]) < 600.0 for row in rows),
        f"slowest {max(float(row[2]) for row in rows):.1f} s against a limit of 600 s",
    )
    out.check(
        "every figure is larger than 20 KiB, so no axis came out empty",
        all(float(row[4]) > 20.0 for row in rows),
        f"smallest {min(float(row[4]) for row in rows):.1f} KiB",
    )
    out("")
    out("Every figure embedded in README.md is produced by the example of the same")
    out("name, so a figure can never drift from the code that made it.")
    _ = Path
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
