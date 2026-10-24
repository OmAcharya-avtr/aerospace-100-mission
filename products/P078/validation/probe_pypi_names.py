"""Verify, against the live index, that every package this README names exists.

A README that names a package which does not exist on PyPI is worse than one
that names none: it sends a reader looking for something that is not there.
This script probes each name with ``pip index versions`` -- a read-only query,
no installation -- and records what the index said.

The probe is validated in both directions before any of its answers are
trusted. ``numpy`` is a known-taken control and must come back with versions;
two nonsense names are known-free controls and must come back not found. A
probe that cannot distinguish the two is not evidence about anything, so if
either control fails, every result in the run is marked unverified.

Network access is not guaranteed. When the probe cannot reach the index this
script says so and exits zero **without overwriting the committed transcript**,
because a committed output file that records "no network" is less useful than
the one recording the last successful probe. The committed transcript carries
the date it was taken.
"""

from __future__ import annotations

import re
import subprocess
import sys

import _bootstrap
from _bootstrap import Tee, header

ROOT = _bootstrap.bootstrap()

#: Packages the README names, plus the controls.
TAKEN_CONTROL = "numpy"
FREE_CONTROLS = ("cmdguard-aero-nonexistent-probe", "optima-p078-free-control-name")
NAMED_IN_README = (
    "cmdguard",
    "spacepackets",
    "transitions",
    "python-statemachine",
    "yamcs-client",
    "pytest",
    "scikit-learn",
    "pyyaml",
)
TIMEOUT_S = 90


def probe(name: str) -> tuple[str, str]:
    """``(status, detail)`` for ``name``: ``taken``, ``free`` or ``error``."""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "index", "versions", name],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        return "error", type(error).__name__
    text = completed.stdout + completed.stderr
    if completed.returncode == 0:
        match = re.search(rf"^{re.escape(name)} \(([^)]+)\)", completed.stdout, re.MULTILINE)
        return "taken", match.group(1) if match else "version not parsed"
    if "No matching distribution found" in text or "no matching distribution" in text.lower():
        return "free", "no matching distribution"
    first = next(
        (line.strip() for line in text.splitlines() if line.strip()), "no output"
    )
    return "error", first[:120]


def main() -> int:
    tee = Tee("probe_pypi_names_output.txt")
    header(tee, "cmdguard -- PyPI name probe, read-only, with controls in both directions")
    tee("")
    tee("  Nothing is installed by this script. 'pip index versions' is a query.")
    tee(f"  timeout per probe: {TIMEOUT_S} s")

    tee.rule("1. the probe is validated before it is trusted")
    taken_status, taken_detail = probe(TAKEN_CONTROL)
    tee(f"  known-taken control  {TAKEN_CONTROL:<40s} -> {taken_status} ({taken_detail})")
    if taken_status == "error":
        tee("")
        tee("  The index could not be reached. No name is reported as free or taken on")
        tee("  this run, and the committed transcript is left as it was, carrying the")
        tee("  last successful probe. Re-run this script with network access.")
        tee("")
        tee("  FAILED CHECKS: none (the probe did not run)")
        return 0
    free_results = {name: probe(name) for name in FREE_CONTROLS}
    for name, (status, detail) in free_results.items():
        tee(f"  known-free control   {name:<40s} -> {status} ({detail})")
    discriminates = taken_status == "taken" and all(
        status == "free" for status, _ in free_results.values()
    )
    tee.check("the probe distinguishes taken from free", discriminates,
              "a probe that cannot do this is not evidence")
    if not discriminates:
        tee("")
        tee("  The controls disagree with reality, so every result below is UNVERIFIED")
        tee("  and no claim in the README rests on this run.")

    tee.rule("2. every name the README mentions")
    tee(f"  {'name':<34s} {'status':>7s}  detail")
    results: dict[str, tuple[str, str]] = {}
    for name in NAMED_IN_README:
        status, detail = probe(name)
        results[name] = (status, detail)
        tee(f"  {name:<34s} {status:>7s}  {detail}")

    tee.rule("3. what this means for the README")
    free = sorted(name for name, (status, _) in results.items() if status == "free")
    taken = sorted(name for name, (status, _) in results.items() if status == "taken")
    errors = sorted(name for name, (status, _) in results.items() if status == "error")
    tee(f"  taken on the index : {len(taken)} -- {', '.join(taken) or 'none'}")
    tee(f"  free on the index  : {len(free)} -- {', '.join(free) or 'none'}")
    tee(f"  not determined     : {len(errors)} -- {', '.join(errors) or 'none'}")
    tee("")
    tee("  'cmdguard' is expected to be FREE: this package has not been published, and")
    tee("  a name this repository claims and does not hold is recorded as free rather")
    tee("  than implied to be taken.")
    tee.check("the package name this product claims is free on the index",
              results.get("cmdguard", ("error", ""))[0] == "free",
              f"probed as {results.get('cmdguard', ('error', ''))[0]}")
    tee.check("every alternative the README names as existing does exist",
              all(results[name][0] == "taken" for name in NAMED_IN_README
                  if name != "cmdguard"),
              "a README must not send a reader after a package that is not there")

    code = tee.summarise()
    tee(f"transcript written to {tee.save()}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
