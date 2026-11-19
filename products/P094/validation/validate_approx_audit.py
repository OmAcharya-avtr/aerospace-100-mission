"""Audit: does any assertion in the suite depend on pytest.approx's hidden 1e-12.

``pytest.approx(expected, rel=r)`` passes if EITHER bound is met and the
absolute default is ``1e-12``, so a bare ``rel=`` on a small quantity is not
testing what it says. This script runs the WHOLE suite twice: once normally and
once with ``_approx_audit_plugin``, which makes a bare ``rel=`` raise. The
measurement is the count of assertions that depend on the default, and it is a
count from an execution rather than from a grep.

**This script AMPLIFIES any load-sensitive test by two.** It runs the whole
suite twice and asserts that the plain run passes, so a test whose outcome
depends on the machine's load fails this script rather than the test file that
owns it, and twice as often. That is one reason the load-sensitive assertion
class was removed at its design rather than patched instance by instance; see
validation/load_sensitive_inventory.md.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import re  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

#: What the two pytest runs cover. The audit's question is whether any
#: assertion depends on ``pytest.approx``'s hidden 1e-12 absolute default, and
#: answering it honestly means running the WHOLE suite twice -- which is what a
#: committed run does.
#:
#: ``HILTIMINGDIFF_FAST=1`` narrows the two runs to one representative file.
#: That mode is used only by ``probe_working_directories.py``, whose question is
#: whether this script RUNS from a given working directory and not what its
#: numbers are, and whose output is redirected away from the committed file. It
#: exists because the two full-suite runs are 67 s, and the probe executes this
#: script four times, which was 268 s of the probe's 493 s -- more than half of
#: it -- against a 600 s release-gate timeout. The cost is stated plainly: in
#: FAST mode the dynamic count is about one file rather than 1209 tests. The
#: committed output and the gate's own run are full-suite and unaffected.
FAST_TARGET = "tests/test_properties.py"

SUMMARY = re.compile(r"(\d+) passed")
FAILED = re.compile(r"(\d+) failed")
ERRORS = re.compile(r"(\d+) error")


def _environment() -> dict[str, str]:
    """Subprocess environment with ``validation/`` importable for the plugin."""
    environment = _bootstrap.subprocess_environment()
    environment["PYTHONPATH"] = (
        str(ROOT / "validation") + ":" + environment.get("PYTHONPATH", "")
    )
    return environment


def _run(arguments: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", *arguments],
        cwd=str(ROOT),
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )


def _counts(output: str) -> tuple[int, int, int]:
    passed = int(SUMMARY.search(output).group(1)) if SUMMARY.search(output) else 0
    failed = int(FAILED.search(output).group(1)) if FAILED.search(output) else 0
    errors = int(ERRORS.search(output).group(1)) if ERRORS.search(output) else 0
    return passed, failed, errors


def main() -> int:
    tee = _bootstrap.Tee("validate_approx_audit_output.txt")
    _bootstrap.header(tee, "P094 HilTimingDiff -- pytest.approx hidden-tolerance audit")

    tee.rule("a static count first, so the dynamic count has something to agree with")
    total_approx = 0
    total_rel = 0
    bare_rel: list[str] = []
    for path in sorted((ROOT / "tests").glob("*.py")):
        text = path.read_text()
        total_approx += text.count("pytest.approx")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if "rel=" in line:
                total_rel += line.count("rel=")
                if "abs=" not in line:
                    bare_rel.append(f"{path.name}:{line_number}")
    tee(f"  pytest.approx call sites : {total_approx}")
    tee(f"  rel= occurrences         : {total_rel}")
    tee(f"  rel= without abs= on the same line : {len(bare_rel)}")
    for item in bare_rel:
        tee(f"    {item}")
    tee("")
    tee.check(
        "no rel= appears without an explicit abs= on the same line",
        not bare_rel,
        f"{len(bare_rel)} occurrences",
    )

    tee.rule("the dynamic audit: run the whole suite with a bare rel= made to raise")
    target = FAST_TARGET if _bootstrap.FAST else "tests/"
    if _bootstrap.FAST:
        tee(
            f"  HILTIMINGDIFF_FAST=1: the two runs below cover {target} only, not the "
            "whole suite."
        )
        tee(
            "  This mode is used only by the working-directory probe. The committed "
            "output is a"
        )
        tee("  full-suite run.")
        tee("")
    plain = _run([target, "-q", "--no-header"])
    plain_counts = _counts(plain.stdout)
    tee(f"  plain run      : passed={plain_counts[0]} failed={plain_counts[1]} "
        f"errors={plain_counts[2]} exit={plain.returncode}")
    audited = _run(
        [
            target,
            "-q",
            "--no-header",
            "-p",
            "_approx_audit_plugin",
        ]
    )
    audited_counts = _counts(audited.stdout)
    tee(f"  audited run    : passed={audited_counts[0]} failed={audited_counts[1]} "
        f"errors={audited_counts[2]} exit={audited.returncode}")
    tee("")
    dependent = audited_counts[1] + audited_counts[2]
    tee(f"  assertions depending on the hidden 1e-12 default: {dependent}")
    tee("")
    tee.check(
        "the plain run passes",
        plain.returncode == 0 and plain_counts[1] == 0 and plain_counts[2] == 0,
        plain.stdout.strip().splitlines()[-1] if plain.stdout.strip() else "no output",
    )
    tee.check(
        "no assertion in the suite depends on the hidden absolute default",
        dependent == 0,
        f"{dependent} of {plain_counts[0]} tests fail or error when a bare rel= raises"
        + ("  [FAST: one file, not the whole suite]" if _bootstrap.FAST else ""),
    )
    tee.check(
        "both runs collect the same number of tests",
        plain_counts[0] == audited_counts[0] + dependent,
        f"plain {plain_counts[0]}, audited {audited_counts[0]} passed plus "
        f"{dependent} affected",
    )

    tee.rule("the plugin is proved to fire, so a zero count is a measurement")
    probe_directory = Path(_bootstrap.subprocess_environment()["HOME"]) / "probe"
    probe_directory.mkdir(parents=True, exist_ok=True)
    (probe_directory / "test_bare_rel.py").write_text(
        "import pytest\n\n\n"
        "def test_bare_relative_tolerance():\n"
        "    # 1e-20 differs from 0.0 by 1e-20, far inside the 1e-12 absolute\n"
        "    # default, so this assertion passes on the default and NOT on its\n"
        "    # stated relative bound.\n"
        "    assert 1e-20 == pytest.approx(0.0, rel=1e-30)\n"
    )
    without = subprocess.run(
        [sys.executable, "-m", "pytest", str(probe_directory), "-q", "--no-header"],
        cwd=str(ROOT),
        env=_bootstrap.subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    environment = _environment()
    with_plugin = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(probe_directory),
            "-q",
            "--no-header",
            "-p",
            "_approx_audit_plugin",
        ],
        cwd=str(ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    tee(f"  probe without the plugin : exit {without.returncode}")
    tee(f"  probe with the plugin    : exit {with_plugin.returncode}")
    tee("")
    tee.control(
        "a deliberately bare rel= assertion passes without the plugin",
        without.returncode != 0,
        must_fire=False,
        detail="it passes on the hidden absolute default, which is the defect",
    )
    tee.control(
        "the plugin makes that same assertion fail",
        with_plugin.returncode != 0,
        must_fire=True,
        detail="so a zero count above is a measurement and not a plugin that never ran",
    )

    path = tee.save()
    print(f"\nwritten: {path}")
    return 1 if tee.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
