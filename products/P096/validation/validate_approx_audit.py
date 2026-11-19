"""Audit: how many assertions in the suite depend on ``pytest.approx``'s hidden abs.

``pytest.approx(expected, rel=r)`` passes if EITHER the relative or the absolute
bound is met, and the absolute default is ``1e-12``. For any quantity of order
1e-5 that silently replaces every relative tolerance tighter than 2e-8 and the
assertion no longer tests what it says. This product's quantities span
fifteen orders of magnitude -- decomposition residuals of order 1e-3 bits beside
losses of order 1e10 bits, and conditioning-scaled tolerances of order 1e-8 --
so it is exactly the case where the default bites.

The audit runs the whole suite in a subprocess with a plugin that records every
bare ``rel=`` call site and forces ``abs=0.0``, and reports the COUNT. It is a
number, not an assumption.

INPUT ORIGIN: this repository's own test suite. No measured data is read.

Under ``HANDOVERLOSS_FAST=1`` the DYNAMIC half of the audit runs only
``tests/test_accounting.py`` and ``tests/test_baselines.py`` rather than the
whole suite. That setting is used only by ``probe_working_directories.py``,
which runs every script four times over and would otherwise multiply the suite
by four; the committed figures in this script's own output file come from the
full-suite run. The reduction is printed in the output so a reader can see which
run produced the numbers they are looking at.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()


def main() -> int:
    out = _bootstrap.Tee("validate_approx_audit_output.txt")
    out.rule("pytest.approx AUDIT -- assertions depending on the default abs=1e-12")
    out("INPUT ORIGIN: this repository's own test suite. No measured data is read.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)

    sources = sorted((ROOT / "tests").glob("test_*.py"))
    bare = 0
    with_abs = 0
    for path in sources:
        text = path.read_text()
        for index, line in enumerate(text.splitlines(), start=1):
            if "rel=" not in line:
                continue
            # The call may be split across lines; look at the following line too.
            window = line
            lines = text.splitlines()
            if index < len(lines):
                window = line + " " + lines[index]
            if "abs=" in window:
                with_abs += 1
            else:
                bare += 1
    out.rule("1  static scan of the test sources")
    out(f"  files scanned                         : {len(sources)}")
    out(f"  rel= sites with an explicit abs=      : {with_abs}")
    out(f"  rel= sites WITHOUT an explicit abs=   : {bare}")
    out.check(
        "every rel= call site in tests/ carries an explicit abs=",
        bare == 0,
        f"{bare} bare site(s)",
    )

    out.rule("2  dynamic audit -- the suite re-run with the default forced off")
    scratch = Path(tempfile.mkdtemp(prefix="handoverloss-approx-"))
    record = scratch / "approx_audit.json"
    environment = _bootstrap.subprocess_environment()
    environment["HANDOVERLOSS_APPROX_AUDIT"] = str(record)
    environment["PYTHONPATH"] = (
        str(ROOT / "src") + ":" + str(ROOT / "tests") + ":" + str(ROOT / "validation")
    )
    targets = (
        ["tests/test_accounting.py", "tests/test_baselines.py"]
        if _bootstrap.FAST
        else ["tests/"]
    )
    out(f"  pytest targets                          : {targets}")
    if _bootstrap.FAST:
        out(
            "  HANDOVERLOSS_FAST=1 is set, so the dynamic audit runs a subset. The\n"
            "  committed figures come from the full-suite run; this one answers only\n"
            "  whether the script runs from the current working directory."
        )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *targets,
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "_approx_audit_plugin",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
        timeout=1800,
        check=False,
    )
    out(f"  suite exit status with the plugin loaded: {completed.returncode}")
    out.check(
        "the whole suite still passes with pytest.approx's absolute default forced to 0",
        completed.returncode == 0,
        "a failure here is an assertion that was only passing because of the default",
    )
    if record.exists():
        entries = json.loads(record.read_text())
        out(f"  call sites recorded as bare rel=      : {len(entries)}")
        for entry in entries[:20]:
            out(f"    {entry['site']}  rel={entry['rel']}")
        out.check(
            "no assertion in this suite depends on the pytest.approx absolute default",
            len(entries) == 0,
            f"{len(entries)} dependent assertion(s)",
        )
    else:
        out.check(
            "the audit plugin wrote its record",
            False,
            "no record file; the plugin did not run",
        )
    if completed.returncode != 0:
        out("")
        out("  last 30 lines of the audited run:")
        for line in completed.stdout.splitlines()[-30:]:
            out(f"    {line}")

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
