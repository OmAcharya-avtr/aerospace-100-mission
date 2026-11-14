"""How many assertions in this suite depend on ``pytest.approx``'s hidden default.

INPUT ORIGIN: this repository's own test suite. No measured data is involved.

``pytest.approx(expected, rel=r)`` passes if EITHER the relative or the absolute
bound is met, and the absolute default is ``1e-12``. For any quantity of order
1e-5 that silently replaces every relative tolerance tighter than 2e-8, and the
assertion no longer tests what it says it tests. An earlier product of this
mission found exactly 3 of 271 assertions depending on the default, and one of
those was hiding a real 2.086218e-10 relative discrepancy.

The rule in this product is an explicit ``abs=`` on every ``rel=`` call, with
``abs=0.0`` where the relative bound is the one meant. The rule is AUDITED here
and not assumed: the suite is re-run in a subprocess with
``validation/_approx_audit_plugin.py`` loaded, which replaces
``pytest.approx`` with a wrapper that records every bare ``rel=`` call site and
forces ``abs=0.0`` for it. The audit then reports the count and the suite's
verdict under that substitution.

Two numbers come out of this and both are reported:

* how many call sites passed ``rel=`` without ``abs=`` -- which should be zero;
* whether the suite still passes with the default forced off -- which it must,
  because if it does not then an assertion was relying on the default to pass.
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
    out.rule("APPROX AUDIT -- assertions depending on the hidden absolute default")
    out("INPUT ORIGIN: this repository's own test suite.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)

    scratch = Path(tempfile.mkdtemp(prefix="powerenvelope-approx-"))
    record = scratch / "record.json"
    environment = _bootstrap.subprocess_environment()
    environment["POWERENVELOPE_APPROX_AUDIT"] = str(record)
    environment["PYTHONPATH"] = (
        str(ROOT / "src") + ":" + str(ROOT / "tests") + ":" + str(ROOT / "validation")
    )

    out.rule("1  the suite, re-run with the absolute default forced off")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "_approx_audit_plugin",
            f"--junit-xml={scratch / 'junit_audit.xml'}",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
        timeout=1800,
        check=False,
    )
    tail = [line for line in completed.stdout.splitlines() if line.strip()][-3:]
    for line in tail:
        out(f"  {line}")
    out.check(
        "the suite passes with pytest.approx's absolute default forced to 0.0",
        completed.returncode == 0,
        f"pytest exit status {completed.returncode}",
    )
    if completed.returncode != 0:
        for line in completed.stdout.splitlines()[-40:]:
            out(f"    {line}")

    out.rule("2  the count of call sites that passed rel= without abs=")
    if record.exists():
        sites = json.loads(record.read_text())
    else:
        sites = []
        out("  the plugin wrote no record file")
    out(f"  call sites recorded: {len(sites)}")
    for site in sites[:40]:
        out(f"    {site['site']}  rel={site['rel']}")
    out.check(
        "no assertion in this suite depends on the hidden absolute default",
        not sites,
        f"{len(sites)} bare rel= call site(s)",
    )

    out.rule("3  what the audit cannot tell you")
    out(
        "  It counts call sites that omitted abs=, and it re-runs the suite with the "
        "default forced off. It cannot tell you whether an explicit abs= is the RIGHT "
        "value: an abs= chosen large enough to pass is the same defect with the "
        "paperwork filed. Where a relative bound is the one meant, this product passes "
        "abs=0.0, which is checkable by reading the call."
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
