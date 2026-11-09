"""Every example runs, writes its PNG, and prints the numbers README.md quotes.

An example that no longer runs is a broken README, because the screenshots it
produces are the ones embedded there. This script runs each one as a subprocess
with a temporary ``HOME`` and ``MPLCONFIGDIR`` outside the repository, checks the
exit status, checks that the PNG it claims to write exists and is not empty, and
searches its stdout for the fragments README.md quotes from it.

The PNG is removed before each run and its size is recorded after, so an example
that silently stopped writing its figure cannot pass on a stale file left by an
earlier run.
"""

from __future__ import annotations

# See validate_plant.py for why this statement is here and not in _bootstrap.
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

from twindiverge.report import format_table  # noqa: E402

EXAMPLES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "sensitivity_spectrum.py",
        "sensitivity_spectrum.png",
        ("rank 4 of 5", "rank 5 of 5", "rank 3 of 5", "refused 5", "refused 0"),
    ),
    (
        "nullspace_equivalence.py",
        "nullspace_equivalence.png",
        ("worst rate difference", "as a fraction of the noise",
         "units in the last place"),
    ),
    (
        "record_length_sweep.py",
        "record_length_sweep.png",
        ("rank 4", "worst bound falls by a factor of"),
    ),
    (
        "profile_likelihood.py",
        "profile_likelihood.png",
        ("overrides True", "overrides False", "max delta chi2"),
    ),
    (
        "crb_vs_empirical.py",
        "crb_vs_empirical.png",
        ("linear, eq (12)", "nonlinear fit", "least ratio over all points"),
    ),
    (
        "attribution_map.py",
        "attribution_map.png",
        ("verdicts R R R R R", "verdicts I I W W W",
         "rate_full identifiable combinations"),
    ),
    (
        "learned_confidence.py",
        "learned_confidence.png",
        ("mean confidence", "population equivalent",
         "the single and equivalent confidence distributions are identical: True"),
    ),
)


def main() -> int:
    out = Tee("validate_examples_output.txt")
    out.rule("twindiverge -- every example, run as a subprocess")
    for line in environment_lines():
        out(line)
    out("")
    out(f"examples declared : {len(EXAMPLES)}")
    out("Each PNG is deleted before its example runs, so a stale figure cannot make")
    out("a broken example pass.")
    environment = subprocess_environment()

    rows = []
    for script, figure, fragments in EXAMPLES:
        target = ROOT / "screenshots" / figure
        if target.exists():
            target.unlink()
        started = time.perf_counter()
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, str(ROOT / "examples" / script)],
            cwd=str(ROOT / "examples"),
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        elapsed = time.perf_counter() - started
        exists = target.exists()
        size = target.stat().st_size if exists else 0
        missing = [
            fragment for fragment in fragments if fragment not in completed.stdout
        ]
        verdict = (
            completed.returncode == 0 and exists and size > 10_000 and not missing
        )
        rows.append([
            script,
            str(completed.returncode),
            f"{elapsed:.1f}",
            str(len(completed.stdout.splitlines())),
            f"{size}" if exists else "MISSING",
            "ok" if verdict else "FAILED",
        ])
        detail = f"exit {completed.returncode}, figure {size} bytes"
        if missing:
            detail += f", missing fragments {missing}"
        if completed.returncode != 0 and completed.stderr.strip():
            detail += f", stderr {completed.stderr.strip().splitlines()[-1][:120]}"
        out.check(f"examples/{script}", verdict, detail)

    out("")
    out(format_table(
        ["example", "exit", "seconds", "stdout lines", "figure bytes", "verdict"],
        rows,
    ))
    out("")
    out("Elapsed times are UNSTABLE: two cores, and the scripts are run one at a")
    out("time here but concurrently by probe_working_directories.py.")
    out("")
    out("figures in screenshots/, as a cold clone carries them:")
    out("")
    out(format_table(
        ["figure", "bytes"],
        [
            [relative(path), str(path.stat().st_size)]
            for path in sorted((ROOT / "screenshots").glob("*.png"))
        ],
    ))

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
