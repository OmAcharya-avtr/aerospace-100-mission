"""Record the exact environment every other number in this repository came from."""

from __future__ import annotations

import os
import platform
import sys

from _harness import Recorder


def main() -> int:
    rec = Recorder("validate_environment")
    rec.header("Environment of record - calibaudit 0.1.0")

    import matplotlib
    import numpy
    import scipy
    import sklearn

    import calibaudit

    rec.say(f"python           : {sys.version.splitlines()[0]}")
    rec.say(f"platform         : {platform.platform()}")
    rec.say(f"os.cpu_count     : {os.cpu_count()}")
    rec.say(f"sched_getaffinity: {len(os.sched_getaffinity(0))}")
    rec.say(f"numpy            : {numpy.__version__}")
    rec.say(f"scipy            : {scipy.__version__}")
    rec.say(f"scikit-learn     : {sklearn.__version__}")
    rec.say(f"matplotlib       : {matplotlib.__version__}")
    try:
        import hypothesis
        import pytest

        rec.say(f"pytest           : {pytest.__version__}")
        rec.say(f"hypothesis       : {hypothesis.__version__}")
    except ImportError:  # pragma: no cover - test extras absent
        rec.say("pytest/hypothesis: not installed in this interpreter")
    rec.say(f"calibaudit       : {calibaudit.__version__}")
    rec.say()
    rec.say("Declared runtime dependencies: numpy, scipy, scikit-learn.")
    rec.say("matplotlib is needed only for calibaudit.plotting and examples/.")
    rec.say()

    rec.check(
        "package version matches the release under validation",
        reference="pyproject.toml version 0.1.0",
        measured=calibaudit.__version__,
        expectation="0.1.0",
        passed=calibaudit.__version__ == "0.1.0",
    )
    cores = len(os.sched_getaffinity(0))
    rec.check(
        "core count recorded, not assumed",
        reference="len(os.sched_getaffinity(0))",
        measured=str(cores),
        expectation="any value, recorded so timings are interpretable",
        passed=cores >= 1,
    )
    major, minor = (int(p) for p in sklearn.__version__.split(".")[:2])
    rec.check(
        "scikit-learn is at least 1.9, where CalibratedClassifierCV lost cv='prefit'",
        reference="sklearn.__version__",
        measured=sklearn.__version__,
        expectation=">= 1.9",
        passed=(major, minor) >= (1, 9),
    )
    return rec.finish()


if __name__ == "__main__":
    raise SystemExit(main())
