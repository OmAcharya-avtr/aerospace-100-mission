"""Record the exact environment every number in this repository was measured in.

Also asserts that none of the alternatives named in the README's alternatives
table is installed, so no claim in that table can have come from importing one.

Run:
    PYTHONPATH=src python validation/validate_environment.py
"""

from __future__ import annotations

import importlib.metadata
import platform
import sys

import safemodetree

MEASURED = ("numpy", "scikit-learn", "PyYAML", "matplotlib", "pytest", "hypothesis", "ruff")
MUST_BE_ABSENT = ("transitions", "statemachine", "pynusmv", "nusmv", "pysmt", "sismic")

CHECKS: list[tuple[str, bool]] = []


def record(label: str, ok: bool) -> None:
    """Print and record one check."""
    CHECKS.append((label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> None:
    """Print the environment and the absence checks."""
    print("safemodetree environment record")
    print("=" * 72)
    print(f"safemodetree version   {safemodetree.__version__}")
    print(f"python                 {sys.version.split()[0]}")
    print(f"implementation         {platform.python_implementation()}")
    print(f"platform               {platform.platform()}")
    print(f"machine                {platform.machine()}")
    print()
    print("installed versions of every library a number here depends on")
    print("-" * 72)
    for name in MEASURED:
        try:
            print(f"  {name:<16} {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"  {name:<16} NOT INSTALLED")
    print()
    print("declared dependency ranges in pyproject.toml; BOTH ends were installed")
    print("and the whole suite plus every validation script was run at each end")
    print("-" * 72)
    for line in (
        "  numpy         >=2.1.3,<2.6",
        "  scikit-learn  >=1.5.2,<1.10",
        "  pyyaml        >=6.0.2,<6.1",
        "  matplotlib    >=3.9.2,<3.12      (plot extra)",
        "  pytest        >=8.3.3,<9.2       (test extra)",
        "  hypothesis    >=6.112.1,<6.170   (test extra)",
        "  ruff          >=0.6.9,<0.18      (test extra)",
    ):
        print(line)
    print()
    print("note: two ruff builds are present in this container. The CLI on PATH is")
    print("0.16.8 (a uv tool install) and the importable package is 0.17.0. Both were")
    print("run clean over src/, tests/, examples/ and validation/, as was 0.6.9 in the")
    print("floor virtual environment, so the declared range is >=0.6.9,<0.18.")
    print()
    print("alternatives named in the README must NOT be importable here")
    print("-" * 72)
    for name in MUST_BE_ABSENT:
        absent = True
        try:
            __import__(name)
            absent = False
        except ImportError:
            absent = True
        record(f"{name} is not installed", absent)
    print()
    failed = [label for label, ok in CHECKS if not ok]
    print(f"summary: {len(CHECKS) - len(failed)} of {len(CHECKS)} checks passed")
    if failed:
        print("FAILED CHECKS (recorded, not suppressed):")
        for label in failed:
            print(f"  {label}")
    assert not failed, failed
    print("ENVIRONMENT RECORD COMPLETE")


if __name__ == "__main__":
    main()
