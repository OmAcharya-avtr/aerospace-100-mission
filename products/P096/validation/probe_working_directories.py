"""Every script here, run from four working directories, must agree with itself.

Three products in one batch of this mission passed every check in their own
build and then failed the release gate, because a validation script passed a
bare relative path to a loader. The build agent, the agent's own self-report and
the coordinator's re-measurement all ran the scripts from the product root; the
gate runs them with the working directory set to ``validation/``. Three
independent verifications agreeing does not help when they share an assumption.

So each script is executed from the filesystem root, the system temporary
directory, the product root and ``validation/``, and the EXIT STATUS must be the
same from all four. The property under test is path-independence, which is not
the same as success: a script that refuses because the measured traces are
absent must refuse identically from all four directories, and that is reported
as the agreement it is rather than being counted as a pass.

``HANDOVERLOSS_FAST=1`` is set for the probed runs. The question here is whether
a script runs from a given directory, not what its numbers are; the numbers come
from the full-resample runs recorded in each script's own output file. Under that
flag the bootstrap resample count drops from 4000 to 200, the trace event count
from 512 to 96, and ``validate_approx_audit.py`` runs two test files rather than
the whole suite -- without the last of those this probe would run the suite
sixteen times.

This script does not probe itself.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

#: Directory labels, so no absolute path reaches the committed output.
DIRECTORIES = (
    ("filesystem root", Path("/")),
    ("system temporary directory", Path(tempfile.gettempdir())),
    ("product root", ROOT),
    ("validation/", ROOT / "validation"),
)

#: Scripts excluded from the probe. This one, because it would recurse.
EXCLUDED = ("probe_working_directories.py",)


def main() -> int:
    out = _bootstrap.Tee("probe_working_directories_output.txt")
    out.rule("WORKING-DIRECTORY PROBE -- same exit status from four directories")
    out("INPUT ORIGIN: this repository's own scripts.")
    out(_bootstrap.ORIGIN)
    out("")
    for line in _bootstrap.environment_lines():
        out(line)
    out("")
    out("directories probed: " + ", ".join(label for label, _ in DIRECTORIES))
    out("HANDOVERLOSS_FAST=1 is set, so resample counts are reduced; the question is")
    out("whether each script RUNS from each directory.")

    scripts = sorted(
        path.name
        for path in (ROOT / "validation").glob("*.py")
        if not path.name.startswith("_") and path.name not in EXCLUDED
    )
    also = sorted(path.name for path in (ROOT / "examples").glob("*.py")
                  if not path.name.startswith("_"))
    out("")
    out(f"validation scripts probed: {len(scripts)}")
    out(f"example scripts probed   : {len(also)}")

    environment = _bootstrap.subprocess_environment()
    environment["HANDOVERLOSS_FAST"] = "1"

    disagreements: list[str] = []
    for subdirectory, names in (("validation", scripts), ("examples", also)):
        out.rule(f"{subdirectory}/")
        for name in names:
            target = ROOT / subdirectory / name
            statuses: list[int] = []
            for label, directory in DIRECTORIES:
                try:
                    completed = subprocess.run(
                        [sys.executable, str(target)],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                        env=environment,
                        timeout=420,
                        check=False,
                    )
                    statuses.append(completed.returncode)
                except subprocess.TimeoutExpired:
                    statuses.append(-1)
                    out(f"  {name}: TIMEOUT from the {label}")
            agree = len(set(statuses)) == 1
            rendered = ", ".join(
                f"{label}={status}"
                for (label, _), status in zip(DIRECTORIES, statuses, strict=True)
            )
            passed = out.check(
                f"{subdirectory}/{name} gives the same exit status from all four",
                agree,
                rendered,
            )
            if not passed:
                disagreements.append(f"{subdirectory}/{name}: {rendered}")
            if agree and statuses[0] != 0:
                out(
                    f"      NOTE: the agreed status is {statuses[0]}, not 0. That is "
                    "path-independent REFUSAL, not success. The script's own output file "
                    "says why."
                )

    out.rule("summary")
    if disagreements:
        for line in disagreements:
            out(f"  DISAGREEMENT: {line}")
    else:
        out("  every probed script agreed with itself across all four directories")
    out(
        "  A disagreement here means a bare relative path reached a loader, which is the "
        "defect class that blocked three products of this mission in one batch."
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
