"""Audit every ``pytest.approx`` call for dependence on the hidden 1e-12 default.

``pytest.approx(expected, rel=r)`` passes if EITHER bound is met and the
absolute default is ``1e-12``, so for a quantity of order 1e-5 every relative
tolerance tighter than 2e-8 is silently replaced. The build guide requires
this to be **audited, not assumed**: three products of this mission did so on
one day and one of them found five real cases first, one of which was hiding
a 2.086e-10 relative discrepancy.

Two independent measurements, because a source scan and a run can disagree:

1. **Static.** Every ``approx`` call under ``tests/`` is parsed from the
   syntax tree and counted by whether it passes ``rel``, ``abs``, both or
   neither.
2. **Dynamic.** The whole suite is re-run in a subprocess with a plugin that
   makes a bare ``rel=`` raise. A suite that passes under it has no assertion
   resting on the default at run time, which the static scan cannot establish
   on its own because a call can be built indirectly.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import ast  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

#: The hidden absolute default, as pytest documents it.
HIDDEN_ABSOLUTE_DEFAULT = 1e-12


def approx_calls(source: str) -> list[tuple[int, bool, bool]]:
    """Every ``approx`` call as ``(line, has_rel, has_abs)``."""
    found: list[tuple[int, bool, bool]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = ""
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        if name != "approx":
            continue
        keywords = {keyword.arg for keyword in node.keywords}
        found.append((node.lineno, "rel" in keywords, "abs" in keywords))
    return found


def main() -> int:
    """Audit the suite statically and dynamically."""
    out = _bootstrap.Tee("validate_approx_audit_output.txt")
    out("pytest.approx hidden-default audit -- P099 PassYield")
    out("")
    out(f"the hidden absolute default is {HIDDEN_ABSOLUTE_DEFAULT!r}. approx passes if")
    out("EITHER bound is met, so a relative bound tighter than about 2e-8 on a")
    out("quantity of order 1e-5 is silently replaced by it. This product has")
    out("quantities exactly there: the decomposition residual is of order 1e-3 Mbit")
    out("on a shortfall of order 1e4 Mbit.")
    out("")
    out(f"fast path (PASSYIELD_FAST=1): {_bootstrap.FAST}")

    out.rule("1. STATIC -- every approx call under tests/, from the syntax tree")
    total = rel_only = both = abs_only = neither = files = 0
    offenders: list[str] = []
    for path in sorted((ROOT / "tests").rglob("*.py")):
        files += 1
        for line, has_rel, has_abs in approx_calls(path.read_text(encoding="utf-8")):
            total += 1
            if has_rel and has_abs:
                both += 1
            elif has_rel:
                rel_only += 1
                offenders.append(f"{_bootstrap.relative(path)}:{line}")
            elif has_abs:
                abs_only += 1
            else:
                neither += 1
    out(f"test files scanned          : {files}")
    out(f"approx calls found          : {total}")
    out(f"  rel= and abs= both given  : {both}")
    out(f"  abs= only                 : {abs_only}")
    out(f"  rel= only (the defect)    : {rel_only}")
    out(f"  neither (approx defaults) : {neither}")
    for offender in offenders:
        out(f"  OFFENDER: {offender}")
    out("")
    out.check("no approx call passes rel= without abs=", not offenders, f"{rel_only} do")

    out.rule("2. DYNAMIC -- the whole suite re-run with the default forbidden")
    out("the plugin in validation/_approx_audit_plugin.py replaces pytest.approx")
    out("with a wrapper that RAISES when rel is given and abs is not. A suite that")
    out("passes under it has no assertion resting on the default at run time.")
    out("")
    if _bootstrap.FAST:
        out.unavailable(
            "the dynamic re-run of the suite",
            "the fast path exists for the working-directory probe, and re-running "
            "the whole suite four times inside it is minutes of a shared two-core "
            "runner for a question the static scan already answers",
        )
        out.check(
            "no approx call passes rel= without abs= (static only on the fast path)",
            rel_only == 0,
            f"{rel_only} do",
        )
        return out.finish()
    with tempfile.TemporaryDirectory(prefix="passyield-approx-") as directory:
        environment = _bootstrap.subprocess_environment(Path(directory))
        environment["PYTHONPATH"] = str(ROOT / "validation")
        completed = subprocess.run(  # noqa: S603 - fixed argument vector
            [
                sys.executable, "-m", "pytest", "tests/", "-q",
                "-p", "no:cacheprovider", "-p", "_approx_audit_plugin",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env=environment,
            check=False,
            timeout=1800.0,
        )
    tail = [line for line in completed.stdout.strip().splitlines() if line.strip()][-3:]
    for line in tail:
        out(f"  pytest: {line}")
    out(f"  exit status: {completed.returncode}")
    out("")
    out("the test count in that transcript is pytest stdout and is NOT the number")
    out("this product reports. Test counts come from the junit XML only: a suite")
    out("that collects zero tests prints success and exits 0, which is how a")
    out("product of this mission shipped with a recorded 251 tests and a config")
    out("that collected none.")
    out("")
    out.check(
        "the suite passes with the hidden absolute default forbidden",
        completed.returncode == 0,
        f"exit status {completed.returncode}",
    )
    out.check(
        "assertions depending on the hidden default: 0",
        rel_only == 0 and completed.returncode == 0,
        f"static count {rel_only}, dynamic exit {completed.returncode}",
    )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
