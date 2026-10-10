"""Record the exact environment every other number in this directory came from.

Exits 0 always; it reports, it does not judge.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import traceaudit  # noqa: E402


def version_of(name: str) -> str:
    try:
        module = __import__(name)
    except ImportError:
        return "not installed"
    return getattr(module, "__version__", "unknown")


def main() -> int:
    print("environment")
    print(f"  python              : {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"  platform            : {platform.system()} {platform.release()} {platform.machine()}")
    print(f"  os.cpu_count()      : {os.cpu_count()}")
    try:
        print(f"  sched_getaffinity   : {len(os.sched_getaffinity(0))}")
    except AttributeError:
        print("  sched_getaffinity   : unavailable on this platform")
    print(f"  traceaudit          : {traceaudit.__version__}")
    for name in ("pytest", "hypothesis", "matplotlib", "numpy"):
        print(f"  {name:<20s}: {version_of(name)}")
    print()
    print("declared runtime dependencies of traceaudit: none")
    print("  the audit uses re, ast, xml.etree.ElementTree, json, argparse, pathlib,")
    print("  dataclasses and enum, all from the standard library.")
    print("  matplotlib is an optional extra, needed only for the figures;")
    print("  numpy is pulled in by matplotlib and is used only inside plotting.py.")
    print()
    print("bundled fixtures")
    for rel in (
        "fixtures/sample_project/docs/REQUIREMENTS.md",
        "fixtures/sample_project/tests/test_monitor.py",
        "fixtures/sample_project/junit.xml",
        "fixtures/sample_project/collect_only.txt",
        "fixtures/heuristic/labels.json",
        "fixtures/heuristic/test_truly_empty.py",
        "fixtures/heuristic/test_truly_asserts.py",
        "fixtures/heuristic/test_assertion_elsewhere.py",
        "docs/REQUIREMENTS.md",
    ):
        path = ROOT / rel
        print(f"  {rel:<52s} {path.stat().st_size:>7d} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
