"""Record the interpreter and the library versions every number was measured at.

A declared dependency range is only honest if the versions it was executed at
are written down. This script writes that record for the version currently
installed; ``validate_dependency_ends.py`` runs the suite at both ends of every
range and records the comparison.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import sys

from _common import rule

PACKAGES = ("numpy", "scikit-learn", "scipy", "matplotlib", "pytest", "hypothesis", "ruff")

DECLARED = {
    "numpy": ">=2.3.4,<2.6",
    "scikit-learn": ">=1.8.0,<1.10",
    "matplotlib": ">=3.10.7,<3.12  [plot, test]",
    "pytest": ">=9.0.1,<9.2  [test]",
    "hypothesis": ">=6.169.3,<6.171  [test]",
    "ruff": ">=0.17.0,<0.18  [test]",
    "scipy": "not declared; pulled in by scikit-learn, never imported by scenariocov",
}


def main() -> int:
    """Write the environment record."""
    out = sys.stdout
    out.write("scenariocov environment record\n")
    out.write("=" * 78 + "\n")
    out.write(f"python              {sys.version.split()[0]}\n")
    out.write(f"implementation      {platform.python_implementation()}\n")
    out.write(f"platform            {platform.platform()}\n")
    out.write(f"machine             {platform.machine()}\n")
    out.write(f"cpu count           {os.cpu_count()}\n")
    out.write(rule("installed versus declared") + "\n")
    out.write(f"{'package':<16}{'installed':<14}declared range\n")
    for name in PACKAGES:
        try:
            version = md.version(name)
        except md.PackageNotFoundError:
            version = "ABSENT"
        out.write(f"{name:<16}{version:<14}{DECLARED.get(name, '-')}\n")
    out.write(rule("serialised artifacts") + "\n")
    out.write(
        "None. This package trains a kernel density estimate at call time and\n"
        "never writes a model to disk, so no dependency bound here is a\n"
        "serialisation-format bound. The scikit-learn ceiling is a numeric\n"
        "behaviour bound: KernelDensity bandwidth selection and the value of\n"
        "score_samples are not contractually stable across minor releases, and\n"
        "the benchmark numbers would move with them.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
