"""Record the environment every other number in this directory was produced in."""

from __future__ import annotations

import os
import platform
import sys

import hypothesis
import matplotlib
import numpy as np
import pytest
import scipy

import invariantset


def main() -> None:
    print("environment")
    print(f"  python           : {sys.version.split()[0]} ({platform.platform()})")
    print(f"  numpy            : {np.__version__}")
    print(f"  scipy            : {scipy.__version__}")
    print(f"  matplotlib       : {matplotlib.__version__}")
    print(f"  pytest           : {pytest.__version__}")
    print(f"  hypothesis       : {hypothesis.__version__}")
    print(f"  invariantset     : {invariantset.__version__}")
    print(f"  os.cpu_count()   : {os.cpu_count()}")
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - platform dependent
        affinity = -1
    print(f"  sched_getaffinity: {affinity}")
    print("  LP solver        : scipy.optimize.linprog, method='highs'")
    print("  hull             : scipy.spatial.ConvexHull (Qhull)")
    print("  runtime deps     : numpy, scipy (matplotlib for plotting only)")
    print("  no ML, no learned component, no PyTorch, no scikit-learn")


if __name__ == "__main__":
    main()
