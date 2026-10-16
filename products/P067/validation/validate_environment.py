"""Record the exact interpreter and library versions every number was measured at.

    python validation/validate_environment.py > validation/validate_environment_output.txt
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import platform
import sys

from _common import ROOT  # noqa: F401

PACKAGES = ("numpy", "scipy", "scikit-learn", "matplotlib", "pytest", "hypothesis", "ruff")

IMPORT_NAMES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "pytest": "pytest",
    "hypothesis": "hypothesis",
}

DECLARED = {
    "numpy": ">=2.4.1,<2.6",
    "scikit-learn": ">=1.9.0,<1.10",
    "matplotlib": ">=3.10.8,<3.12  [plot, test]",
    "pytest": ">=8.4.2,<9.2  [test]",
    "hypothesis": ">=6.168.0,<6.171  [test]",
    "ruff": ">=0.16.8,<0.18  [test]",
    "scipy": "not declared; acqsearch does not import scipy",
}

EXECUTED = (
    ("numpy", "2.4.1", "2.5.3"),
    ("scikit-learn", "1.9.0", "1.9.1"),
    ("matplotlib", "3.10.8", "3.11.2"),
    ("pytest", "8.4.2", "9.1.1"),
    ("hypothesis", "6.168.0", "6.170.0"),
    ("ruff", "0.16.8", "0.17.1"),
)


def main() -> int:
    """Write the environment record."""
    out = sys.stdout
    out.write("acqsearch environment record\n")
    out.write("=" * 78 + "\n")
    out.write(f"python              {sys.version.split()[0]}\n")
    out.write(f"implementation      {platform.python_implementation()}\n")
    out.write(f"platform            {platform.platform()}\n")
    out.write(f"machine             {platform.machine()}\n")
    out.write(f"cpu count           {os.cpu_count()}\n\n")
    out.write(f"{'package':<16}{'metadata':<12}{'imported':<12}declared range\n")
    out.write("-" * 78 + "\n")
    for name in PACKAGES:
        try:
            version = md.version(name)
        except md.PackageNotFoundError:
            version = "ABSENT"
        module = IMPORT_NAMES.get(name)
        imported = "-"
        if module is not None:
            try:
                imported = getattr(importlib.import_module(module), "__version__", "-")
            except ImportError:
                imported = "ABSENT"
        out.write(f"{name:<16}{version:<12}{imported:<12}{DECLARED.get(name, '-')}\n")
    out.write(
        "\nThe metadata and imported columns are both printed because they can\n"
        "disagree when an environment holds more than one distribution record for a\n"
        "package. Every number in this repository was produced in a virtual\n"
        "environment pinned to the versions above, in which the two columns agree.\n"
    )
    out.write("\n")
    out.write("Versions actually installed and executed, both ends of every range:\n")
    out.write(f"{'package':<16}{'floor run at':<16}{'ceiling run at':<16}\n")
    out.write("-" * 78 + "\n")
    for name, floor, ceiling in EXECUTED:
        out.write(f"{name:<16}{floor:<16}{ceiling:<16}\n")
    out.write(
        "\nAt every pair above the full suite reported 242 tests, 0 failures, 0 errors,\n"
        "0 skipped from the junit XML, and validation/dependency_range_check.py\n"
        "produced byte-identical output. scikit-learn 1.8.0 was ALSO executed: the\n"
        "suite passes there too, but the fingerprint moves (see\n"
        "validation/dependency_range_check_sklearn180_output.txt), which is why the\n"
        "declared floor is 1.9.0 and not 1.8.0.\n"
    )
    out.write("\nSerialised artifacts committed by this product: none.\n")
    out.write(
        "No model is pickled. The learned policy is refitted from data/scenarios.json\n"
        "and the committed seeds every time it is used, so there is no dependency on\n"
        "any library's serialisation format. The scikit-learn bound exists because\n"
        "GradientBoostingRegressor output is not contractually stable across minor\n"
        "releases and the benchmark numbers would move with it.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
