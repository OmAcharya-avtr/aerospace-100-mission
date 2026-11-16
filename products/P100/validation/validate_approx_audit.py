"""Audit every ``pytest.approx`` call for dependence on the hidden 1e-12 default.

``pytest.approx(expected, rel=r)`` passes if EITHER bound is met and the
absolute default is ``1e-12``, so for a quantity of order 1e-5 every relative
tolerance tighter than 2e-8 is silently replaced. The build guide requires this
to be **audited, not assumed**, because three products did so on one day and
one of them found five real cases first.

Two independent measurements, because a source scan and a run can disagree:

1. **Static.** Every ``approx`` call in ``tests/`` is parsed from the syntax
   tree and counted by whether it passes ``rel``, ``abs``, both or neither.
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
    out: list[tuple[int, bool, bool]] = []
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
        out.append((node.lineno, "rel" in keywords, "abs" in keywords))
    return out


def main() -> int:
    """Audit the suite statically and dynamically."""
    out = _bootstrap.Tee("validate_approx_audit_output.txt")
    out("pytest.approx hidden-default audit -- P100 SeedAudit")
    out("")
    out(f"the hidden absolute default is {HIDDEN_ABSOLUTE_DEFAULT!r}. approx passes if")
    out("EITHER bound is met, so a relative bound tighter than about 2e-8 on a")
    out("quantity of order 1e-5 is silently replaced by it.")
    out("")
    out(f"fast path (SEEDAUDIT_FAST=1): {_bootstrap.FAST}")

    out.rule("1. STATIC -- every approx call in tests/, from the syntax tree")
    total = 0
    rel_only = 0
    both = 0
    abs_only = 0
    neither = 0
    offenders: list[tuple[str, int]] = []
    files = 0
    for path in sorted((ROOT / "tests").rglob("*.py")):
        files += 1
        for line, has_rel, has_abs in approx_calls(path.read_text(encoding="utf-8")):
            total += 1
            if has_rel and has_abs:
                both += 1
            elif has_rel:
                rel_only += 1
                offenders.append((path.relative_to(ROOT).as_posix(), line))
            elif has_abs:
                abs_only += 1
            else:
                neither += 1
                offenders.append((path.relative_to(ROOT).as_posix(), line))
    out(f"test files parsed: {files}")
    out(f"approx calls found: {total}")
    out(f"  rel= and abs= both given: {both}")
    out(f"  abs= only: {abs_only}")
    out(f"  rel= only (DEPENDS ON THE DEFAULT): {rel_only}")
    out(f"  neither given (DEPENDS ON THE DEFAULT): {neither}")
    for path, line in offenders:
        out(f"  OFFENDER {path}:{line}")
    out.check(
        "no approx call in tests/ relies on the hidden absolute default",
        not offenders,
        f"{len(offenders)} offenders of {total} calls",
    )

    out.rule("2. DYNAMIC -- the suite re-run with a bare rel= made an error")
    out("a static scan cannot see a call built indirectly, so the suite is re-run")
    out("in a subprocess with pytest.approx replaced by a wrapper that raises when")
    out("rel is given and abs is not.")
    out("")
    with tempfile.TemporaryDirectory(prefix="seedaudit-approx-") as temporary:
        home = Path(temporary)
        environment = _bootstrap.subprocess_environment(home)
        environment["PYTHONPATH"] = str(ROOT / "validation")
        command = [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "_approx_audit_plugin",
        ]
        out("command: python -m pytest tests/ -q -p no:cacheprovider -p _approx_audit_plugin")
        completed = subprocess.run(  # noqa: S603 - fixed argument vector
            command,
            cwd=str(ROOT),
            capture_output=True,
            env=environment,
            timeout=1800.0,
            check=False,
        )
    stdout = completed.stdout.decode("utf-8", errors="replace")
    tail = [line for line in stdout.splitlines() if line.strip()][-6:]
    out(f"exit status: {completed.returncode}")
    for line in tail:
        out(f"  {line}")
    out.check(
        "the whole suite passes with the hidden absolute default forbidden",
        completed.returncode == 0,
        f"exit {completed.returncode}",
    )

    out.rule("3. THE CONTROL -- it must pass WITHOUT the plugin and fail WITH it")
    out("a check that cannot fail is not a check, and a control that fails for the")
    out("ordinary reason proves nothing about the plugin. So the offending")
    out("assertion is chosen to PASS on its own: 1e-15 against 2e-15 differ by")
    out("1e-15, which is inside the hidden absolute default of 1e-12, so the")
    out("relative bound of 1e-30 is silently replaced and the assertion passes.")
    out("It is then run again WITH the plugin and must fail, carrying the plugin's")
    out("own refusal message so an ordinary failure cannot be mistaken for it.")
    out("")
    with tempfile.TemporaryDirectory(prefix="seedaudit-approx-control-") as temporary:
        base = Path(temporary)
        home = base / "home"
        home.mkdir()
        offending = base / "test_offender.py"
        offending.write_text(
            "import pytest\n\n\n"
            "def test_depends_on_the_hidden_absolute_default() -> None:\n"
            "    assert 1e-15 == pytest.approx(2e-15, rel=1e-30)\n",
            encoding="utf-8",
        )
        environment = _bootstrap.subprocess_environment(home)
        environment["PYTHONPATH"] = str(ROOT / "validation")

        def run_control(with_plugin: bool) -> tuple[int, str]:
            command = [sys.executable, "-m", "pytest", str(offending), "-q",
                       "-p", "no:cacheprovider"]
            if with_plugin:
                command += ["-p", "_approx_audit_plugin"]
            completed = subprocess.run(  # noqa: S603 - fixed argument vector
                command,
                cwd=str(base),
                capture_output=True,
                env=environment,
                timeout=300.0,
                check=False,
            )
            return completed.returncode, completed.stdout.decode("utf-8", errors="replace")

        without_status, _without_output = run_control(False)
        with_status, with_output = run_control(True)
    out(f"without the plugin, exit status: {without_status}")
    out(f"with the plugin, exit status   : {with_status}")
    out(f"with the plugin, output carries the refusal message: "
        f"{'rel= and no abs=' in with_output}")
    out.control(
        "the offending assertion passes without the plugin, which is the defect",
        without_status == 0,
        must_fire=True,
        detail=f"exit {without_status}",
    )
    out.control(
        "the hidden-default plugin refuses the same assertion",
        with_status != 0 and "rel= and no abs=" in with_output,
        must_fire=True,
        detail=f"exit {with_status}",
    )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
