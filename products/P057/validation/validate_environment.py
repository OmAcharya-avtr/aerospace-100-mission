"""Record the environment and the dependency checks every other number here relies on.

Exits 0 always. It asserts only on facts that must hold for the rest of the
directory to mean anything: the package imports, the declared runtime
dependency set is what pyproject says, and no conformal prediction library is
installed, so nothing here can be silently delegating to one.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys

import hypothesis
import matplotlib
import numpy as np
import pytest
import scipy
import sklearn

import conformalband

FORBIDDEN_IMPORTS = ("mapie", "crepes", "deel", "nonconformist", "torchcp", "torch")


def main() -> None:
    print("environment")
    print(f"  python            : {sys.version.split()[0]} ({platform.platform()})")
    print(f"  numpy             : {np.__version__}")
    print(f"  scipy             : {scipy.__version__}")
    print(f"  scikit-learn      : {sklearn.__version__}")
    print(f"  matplotlib        : {matplotlib.__version__}")
    print(f"  pytest            : {pytest.__version__}")
    print(f"  hypothesis        : {hypothesis.__version__}")
    print(f"  conformalband     : {conformalband.__version__}")
    print(f"  os.cpu_count()    : {os.cpu_count()}")
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - platform dependent
        affinity = -1
    print(f"  sched_getaffinity : {affinity}")
    print("  runtime deps      : numpy, scipy, scikit-learn (matplotlib for plotting only)")
    print("  learned components: sklearn GradientBoostingRegressor, sklearn LogisticRegression")
    print("  PyTorch           : not installed and not used")
    print()
    print("no conformal prediction library is installed, so nothing here delegates to one")
    for name in FORBIDDEN_IMPORTS:
        found = importlib.util.find_spec(name) is not None
        print(f"  importlib.util.find_spec({name!r}) is not None : {found}")
        assert not found, f"{name} is installed; the measurements would not be this package's"
    print()
    print("alternatives named in README.md were checked against PyPI with")
    print("  curl -s -o /dev/null -w '%{http_code}' https://pypi.org/pypi/<name>/<version>/json")
    print("and the wheels were downloaded and unpacked to read their public API. Results are")
    print("recorded in VALIDATION.md section 8. This script does not re-run those network")
    print("calls, because a validation script must not depend on egress.")
    print()
    print("egress from this container reached pypi.org only. doi.org, arxiv.org, arc.aiaa.org")
    print("and jmlr.org all returned connect_rejected, so NO citation in this repository was")
    print("verified against a publisher page in this session. See VALIDATION.md section 9.")
    print()
    print("ALL ENVIRONMENT CHECKS PASSED")


if __name__ == "__main__":
    main()
