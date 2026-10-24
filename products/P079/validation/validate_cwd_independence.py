"""Check 12 -- every validation script exits zero from any working directory.

Why this script exists
----------------------
Three products in one Batch 07 batch -- P065, P066 and P069 -- failed the
release gate after passing every check that preceded it, because a validation
script passed a bare relative path to a loader. Measured on P066: exit 0 from
the product root, exit 1 from anywhere else. Nothing in those builds caught it,
and the reason is worth stating: the build agent, its self-report and the
coordinator's re-measurement all ran the scripts the same way, from the product
root. The gate runs each one with ``cwd`` set to ``validation/``. Three
independent verifications agreeing does not help when they share an
assumption.

So every sibling script in this directory is executed from four working
directories -- the filesystem root, the system temporary directory, the product
root and ``validation/`` -- and must exit zero from all four.

The probe runs set ``TIMETRANSFER_FAST=1``, which makes the Monte Carlo loops
smaller and, importantly, makes :meth:`_bootstrap.Tee.save` write **nothing**,
so a probe cannot replace a committed output file with reduced-sample numbers.
The probe checks the exit status; the committed numbers come from the ordinary
runs.

Run: ``python validation/validate_cwd_independence.py``
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from _bootstrap import FAST_ENV, Tee, bootstrap, environment_lines, is_fast, relative

ROOT = bootstrap()

SELF = Path(__file__).name


def main() -> int:
    out = Tee("validate_cwd_independence_output.txt")
    out("timetransfer validation 12 -- every validation script runs from any directory")
    out("=" * 78)
    for line in environment_lines():
        out(line)

    if is_fast():
        out("")
        out("Running in fast probe mode, which means this script was invoked BY itself.")
        out("It does not recurse: the probe returns immediately.")
        return 0

    scripts = sorted(
        path.name
        for path in (ROOT / "validation").glob("*.py")
        if path.name not in (SELF, "_bootstrap.py")
    )
    directories = [
        ("filesystem root", Path(os.sep)),
        ("temporary directory", Path(tempfile.gettempdir())),
        ("product root", ROOT),
        ("validation directory", ROOT / "validation"),
    ]
    out("")
    out(f"scripts probed      : {len(scripts)}")
    out(f"directories probed  : {len(directories)}")
    out("interpreter         : (basename only, so no absolute path is committed) "
        f"{Path(sys.executable).name}")
    out(f"{FAST_ENV}=1 set on every probe run")
    out("")
    out("directories:")
    for label, path in directories:
        shown = "the filesystem root" if path == Path(os.sep) else (
            "the system temporary directory"
            if path == Path(tempfile.gettempdir())
            else relative(path) or "."
        )
        out(f"  {label:<22} {shown}")

    env = dict(os.environ)
    env[FAST_ENV] = "1"
    env["MPLBACKEND"] = "Agg"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    out.rule("results")
    header = f"{'script':>36}" + "".join(f"{label[:10]:>12}" for label, _ in directories)
    out(header)
    failures = 0
    detail: list[str] = []
    for script in scripts:
        row = f"{script:>36}"
        for _, directory in directories:
            completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [sys.executable, str(ROOT / "validation" / script)],
                cwd=str(directory),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            status = completed.returncode
            row += f"{status:>12d}"
            if status != 0:
                failures += 1
                tail = (completed.stderr or completed.stdout).strip().splitlines()
                detail.append(f"{script} exited {status}")
                detail.extend(f"    {line}" for line in tail[-6:])
        out(row)

    if detail:
        out.rule("failing runs")
        for line in detail:
            out(line)

    out.rule("summary")
    out(f"runs              : {len(scripts) * len(directories)}")
    out(f"non-zero statuses : {failures}")
    out("Every number in the table is an exit status; zero is a pass. The probe runs")
    out("write no output files, so the committed outputs in this directory are the")
    out("ones produced by the ordinary runs.")
    out(f"hard failures     : {failures}")
    out(f"saved to          : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
