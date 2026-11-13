"""A longer Hypothesis hunt than the committed suite runs, and what it found.

``.hypothesis/`` is gitignored, so a falsifying example found by one run is lost
on a fresh clone. The committed property tests in ``tests/test_properties.py``
carry per-test ``@settings(max_examples=...)`` decorators, and a decorator
overrides any profile, so a deeper hunt cannot be arranged by loading a profile.
This script therefore copies the suite into a temporary directory OUTSIDE the
repository, raises the example counts there with a textual substitution, and
runs it at several seeds. The repository's own tests are not modified.

What the hunt has found, recorded because it is the point of running it:

  2026-11-13, seed 1, 4000 examples per test:
  ``test_mean_plus_margin_k_is_invariant_under_a_common_rescaling`` failed at
  responses = [0.09479453252194603, 0.09496984880350386], deadline = 0.03125,
  scale = 0.001, with a discrepancy of 1.62e-11 against a tolerance of 1.45e-11.
  The TOLERANCE was wrong, not the implementation: it followed the conditioning
  of ``D - mean`` and ignored that of the sample variance, whose relative
  condition number for two values differing in the fifth significant figure is
  about 1.2e6. The tolerance now carries both terms and the case is pinned as
  ``test_regression_k_invariance_two_nearly_equal_responses``.

``DEADLINERISK_FAST=1`` reduces the hunt to one short pass; it is set by
``probe_working_directories.py``, whose question is whether this script runs
from a given directory and not what it finds.

INPUT ORIGIN: this repository's own property tests. No measured data is read.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

#: Example counts and seeds. The full hunt costs a few minutes on two cores.
HUNT_EXAMPLES = 150 if _bootstrap.FAST else 1200
HUNT_SEEDS = (1,) if _bootstrap.FAST else (1, 777)


def main() -> int:
    out = _bootstrap.Tee("validate_property_hunt_output.txt")
    out.rule("PROPERTY HUNT -- deeper than the committed suite")
    out("INPUT ORIGIN: this repository's own property tests. No measured data is read.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)
    out("")
    out(f"examples per property : {HUNT_EXAMPLES}")
    out(f"seeds                 : {list(HUNT_SEEDS)}")
    out(
        "  The committed suite runs 200 to 400 examples per property. This hunt raises "
        "the count in a COPY outside the repository, because a @settings decorator "
        "overrides any Hypothesis profile and the repository's tests are not edited."
    )

    scratch = Path(tempfile.mkdtemp(prefix="deadlinerisk-hunt-"))
    shutil.copytree(ROOT / "src", scratch / "src")
    shutil.copytree(ROOT / "tests", scratch / "tests")
    target = scratch / "tests" / "test_properties.py"
    text = target.read_text()
    raised, substitutions = re.subn(
        r"max_examples=\d+", f"max_examples={HUNT_EXAMPLES}", text
    )
    target.write_text(raised)
    out("")
    out(f"properties whose example count was raised: {substitutions}")
    out.check(
        "every property's example count was raised in the copy",
        substitutions >= 12,
        f"{substitutions} substitution(s)",
    )

    environment = _bootstrap.subprocess_environment()
    environment["PYTHONPATH"] = f"{scratch / 'src'}:{scratch / 'tests'}"
    for seed in HUNT_SEEDS:
        out.rule(f"seed {seed}")
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_properties.py",
                "-q",
                "-p",
                "no:cacheprovider",
                f"--hypothesis-seed={seed}",
            ],
            cwd=scratch,
            capture_output=True,
            text=True,
            env=environment,
            timeout=3600,
            check=False,
        )
        passed = completed.returncode == 0
        out.check(
            f"no property falsified at {HUNT_EXAMPLES} examples, seed {seed}",
            passed,
            f"exit {completed.returncode}",
        )
        tail = [line for line in completed.stdout.splitlines() if line.strip()][-6:]
        for line in tail:
            out(f"    {line}")
        if not passed:
            out("")
            out("    FALSIFYING OUTPUT, verbatim, to be pinned in tests/test_regressions.py:")
            for line in completed.stdout.splitlines():
                if "Falsifying example" in line or line.strip().startswith("E "):
                    out(f"      {line.strip()}")

    out.rule("pinned counterexamples in the committed suite")
    regressions = (ROOT / "tests" / "test_regressions.py").read_text()
    names = re.findall(r"^def (test_regression_\w+)", regressions, flags=re.MULTILINE)
    for name in names:
        out(f"  {name}")
    out.check(
        "the counterexample this hunt found is pinned as a named literal test",
        "test_regression_k_invariance_two_nearly_equal_responses" in names,
        f"{len(names)} pinned regression test(s)",
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
