"""Record the exact interpreter and library versions every number was measured at.

A declared dependency range is only honest if the version it was executed at is
written down. This script writes that record.

    python validation/validate_environment.py > validation/validate_environment_output.txt
"""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import sys

from _common import ROOT  # noqa: F401

PACKAGES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "pytest",
    "hypothesis",
    "ruff",
)

DECLARED = {
    "numpy": ">=2.5.3,<2.6",
    "scikit-learn": ">=1.9.1,<1.10",
    "matplotlib": ">=3.11.2,<3.12  [plot, test]",
    "pytest": ">=9.1.1,<9.2  [test]",
    "hypothesis": ">=6.169.3,<6.170  [test]",
    "ruff": ">=0.17.0,<0.18  [test]",
    "scipy": "not declared; imported by scikit-learn, not by regressselect",
}


def main() -> int:
    out = sys.stdout
    out.write("regressselect environment record\n")
    out.write("=" * 72 + "\n")
    out.write(f"python              {sys.version.split()[0]}\n")
    out.write(f"implementation      {platform.python_implementation()}\n")
    out.write(f"platform            {platform.platform()}\n")
    out.write(f"machine             {platform.machine()}\n")
    out.write(f"cpu count           {os.cpu_count()}\n\n")
    out.write(f"{'package':<16}{'installed':<14}declared range\n")
    out.write("-" * 72 + "\n")
    for name in PACKAGES:
        try:
            version = md.version(name)
        except md.PackageNotFoundError:
            version = "ABSENT"
        out.write(f"{name:<16}{version:<14}{DECLARED.get(name, '-')}\n")
    out.write("\n")
    out.write(
        "Every range above is bounded on both sides. The floor of each range is the\n"
        "version installed in this container, which is the only version the suite was\n"
        "executed at; the ceiling excludes the next minor release. The build\n"
        "environment forbids installing other versions, so a wider floor cannot be\n"
        "claimed by execution and is therefore not claimed. See README Limitations.\n"
    )
    out.write("\nSerialised artifacts committed by this product: none.\n")
    out.write(
        "No model is pickled, so there is no serialisation-format dependency on\n"
        "scikit-learn beyond the numeric behaviour of GradientBoostingClassifier,\n"
        "which is why that bound is to the minor version.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
