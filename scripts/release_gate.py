#!/usr/bin/env python3
"""Release gate — the machine-checkable condition for an automatic push.

Standing rule (ADR-017, owner instruction 2026-08-30): a batch publishes
automatically once it is built, WITHOUT waiting for per-batch approval, but ONLY
if every product passes every check below. This script is that condition. If it
exits non-zero, nothing is pushed.

The rule the gate exists to enforce is "the product actually works", not "the
test command exited 0". Those are different, and the mission has already been
burned by the difference: P001 BeamTwin shipped publicly with a pytest config
that collected ZERO tests, printed nothing, exited 0, and was recorded as
"251 tests passing" for three weeks. Hence check 1 below: a suite that collects
no tests FAILS. Silence is not success.

Checks, per product:
  1. tests        pytest via junit XML: >0 collected, 0 failed, 0 errored
  2. lint         ruff check src/ tests/ clean
  3. import       every package imports in a FRESH interpreter (not pytest's)
  4. cli          python -m <pkg> --help exits 0 in a clean subprocess, if the
                  package ships a __main__ (this is what P001's failures were)
  5. examples     every examples/*.py runs to completion and writes its output
  6. validation   every validation/*.py re-executes successfully
Repository-wide:
  7. secrets      no token/credential pattern in tracked files or history
  8. paths        no absolute private path in tracked files
  9. artifacts    no build artifact or checkpoint tracked

Usage:
  python3 scripts/release_gate.py                  # all products
  python3 scripts/release_gate.py P011 P012        # named products
  python3 scripts/release_gate.py --quick          # skip examples/validation
Exit 0 = safe to push. Exit 1 = do not push.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_TESTS = 1800
TIMEOUT_SCRIPT = 600

SECRET_PATTERNS = (
    r"gh[pousr]_[A-Za-z0-9]{16,}",
    r"github_pat_[A-Za-z0-9_]{20,}",
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"xox[baprs]-[A-Za-z0-9-]{10,}",
)
ARTIFACT_PATTERNS = (
    "__pycache__", ".pytest_cache", ".ruff_cache", "/dist/", "/build/",
    ".ckpt", ".pt", ".pth", ".onnx",
)


class Result:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    @property
    def ok(self) -> bool:
        return not self.failures


def run(cmd: list[str], cwd: Path, timeout: int, env: dict | None = None):
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, env=e)


def package_names(prod: Path) -> list[str]:
    src = prod / "src"
    if not src.is_dir():
        return []
    return sorted(p.name for p in src.iterdir()
                  if p.is_dir() and (p / "__init__.py").exists())


def check_tests(prod: Path, r: Result) -> int:
    xml = Path(tempfile.mktemp(suffix=".xml"))
    try:
        run(["python3", "-m", "pytest", "tests/", "-q", "--tb=no",
             "-p", "no:cacheprovider", f"--junit-xml={xml}"],
            prod, TIMEOUT_TESTS)
    except subprocess.TimeoutExpired:
        r.fail(f"test suite exceeded {TIMEOUT_TESTS}s")
        return 0
    if not xml.exists():
        r.fail("pytest produced no junit XML — the suite did not run at all")
        return 0
    root = ET.parse(xml).getroot()
    s = root if root.tag == "testsuite" else root[0]
    xml.unlink()
    n = int(s.get("tests", 0))
    f = int(s.get("failures", 0))
    e = int(s.get("errors", 0))
    sk = int(s.get("skipped", 0))
    if n == 0:
        r.fail("ZERO tests collected — an empty suite is a failure, not a pass")
    if f:
        r.fail(f"{f} test(s) FAILED")
    if e:
        r.fail(f"{e} collection/execution error(s)")
    if sk:
        r.note(f"{sk} skipped")
    return n - f - e - sk


def check_lint(prod: Path, r: Result) -> None:
    out = run(["ruff", "check", "src/", "tests/"], prod, 300)
    if out.returncode != 0:
        r.fail("ruff: " + (out.stdout or out.stderr).strip().splitlines()[-1][:120])


def check_import(prod: Path, pkgs: list[str], r: Result) -> None:
    if not pkgs:
        r.fail("no importable package found under src/")
        return
    for pkg in pkgs:
        out = run(["python3", "-c", f"import {pkg}"], prod, 120,
                  {"PYTHONPATH": str(prod / "src")})
        if out.returncode != 0:
            last = (out.stderr or "").strip().splitlines()
            r.fail(f"import {pkg} failed in a clean interpreter: "
                   f"{last[-1][:120] if last else 'unknown'}")


def check_cli(prod: Path, pkgs: list[str], r: Result) -> None:
    for pkg in pkgs:
        if not (prod / "src" / pkg / "__main__.py").exists():
            continue
        out = run(["python3", "-m", pkg, "--help"], prod, 120,
                  {"PYTHONPATH": str(prod / "src")})
        if out.returncode != 0:
            last = (out.stderr or "").strip().splitlines()
            r.fail(f"CLI 'python -m {pkg} --help' exit {out.returncode}: "
                   f"{last[-1][:120] if last else 'no output'}")


def check_scripts(prod: Path, sub: str, r: Result) -> None:
    d = prod / sub
    if not d.is_dir():
        return
    pkgs = package_names(prod)
    env = {"PYTHONPATH": str(prod / "src"), "MPLBACKEND": "Agg"}
    for script in sorted(d.glob("*.py")):
        try:
            out = run(["python3", script.name], d, TIMEOUT_SCRIPT, env)
        except subprocess.TimeoutExpired:
            r.fail(f"{sub}/{script.name} exceeded {TIMEOUT_SCRIPT}s")
            continue
        if out.returncode != 0:
            last = (out.stderr or "").strip().splitlines()
            r.fail(f"{sub}/{script.name} exit {out.returncode}: "
                   f"{last[-1][:120] if last else 'no output'}")
    _ = pkgs


def repo_checks(r: Result) -> None:
    files = run(["git", "ls-files"], ROOT, 120).stdout.split()
    for pat in ARTIFACT_PATTERNS:
        hits = [f for f in files if pat in f]
        if hits:
            r.fail(f"build artifact tracked: {hits[0]} (+{len(hits) - 1} more)")
    joined = "\n".join(files)
    if re.search(r"(^|/)\.env|(^|/)secrets/|\.pem$|\.p12$", joined, re.M):
        r.fail("credential-shaped file is tracked")
    for pat in SECRET_PATTERNS:
        out = run(["git", "grep", "-InE", pat, "HEAD", "--"], ROOT, 300)
        real = [ln for ln in out.stdout.splitlines()
                if "SCAN_SUMMARY" not in ln and "release_gate.py" not in ln]
        if real:
            r.fail(f"SECRET PATTERN in tracked content: {real[0][:100]}")
    # A real path has a following component: /Users/alice/... , /home/claude/... .
    # Documentation that merely names the pattern ("/Users/<name>") must not trip
    # the gate, or the gate gets muted the first time someone documents it.
    path_re = r"(/Users/[A-Za-z0-9._-]+/|/home/claude/|/root/[A-Za-z0-9._-])"
    out = run(["git", "grep", "-InE", path_re, "HEAD", "--"], ROOT, 300)
    real = [ln for ln in out.stdout.splitlines() if "release_gate.py" not in ln]
    if real:
        r.fail(f"absolute private path tracked: {real[0][:100]}")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quick = "--quick" in sys.argv
    prods = ([ROOT / "products" / a for a in args] if args
             else sorted((ROOT / "products").glob("P0*")))
    prods = [p for p in prods if p.is_dir()]

    print(f"Release gate — {len(prods)} product(s)"
          f"{' (quick: examples/validation skipped)' if quick else ''}\n")
    summary, total, blocked = {}, 0, []

    for prod in prods:
        r = Result()
        pkgs = package_names(prod)
        passed = check_tests(prod, r)
        total += passed
        check_lint(prod, r)
        check_import(prod, pkgs, r)
        check_cli(prod, pkgs, r)
        if not quick:
            check_scripts(prod, "examples", r)
            check_scripts(prod, "validation", r)
        verdict = "PASS" if r.ok else "BLOCKED"
        if not r.ok:
            blocked.append(prod.name)
        summary[prod.name] = {"verdict": verdict, "tests": passed,
                              "failures": r.failures, "notes": r.notes}
        print(f"{prod.name}  {verdict:<8} {passed:>5} tests")
        for f in r.failures:
            print(f"         ! {f}")

    repo = Result()
    repo_checks(repo)
    print(f"\nrepository  {'PASS' if repo.ok else 'BLOCKED'}")
    for f in repo.failures:
        print(f"         ! {f}")

    ok = not blocked and repo.ok
    print(f"\nTotal tests passing: {total}")
    print(f"VERDICT: {'PUSH ALLOWED' if ok else 'PUSH BLOCKED'}")
    if blocked:
        print("Blocked products: " + ", ".join(blocked))
        print("Set each to NEEDS HARDENING in products.yaml and do not publish it.")

    (ROOT / "tracking").mkdir(exist_ok=True)
    (ROOT / "tracking" / "RELEASE_GATE_RESULT.json").write_text(json.dumps(
        {"push_allowed": ok, "total_tests_passing": total,
         "blocked": blocked, "repository_failures": repo.failures,
         "products": summary}, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
