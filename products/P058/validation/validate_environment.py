"""Record the exact environment every other number in this repository was measured in."""

from __future__ import annotations

import os
import platform
import sys

from _harness import run


def body(report) -> None:
    import matplotlib
    import numpy
    import scipy
    import sklearn

    import telemdrift

    report.section("1. Versions")
    rows = [
        ("python", sys.version.split()[0]),
        ("platform", platform.platform()),
        ("numpy", numpy.__version__),
        ("scipy", scipy.__version__),
        ("scikit-learn", sklearn.__version__),
        ("matplotlib", matplotlib.__version__),
        ("telemdrift", telemdrift.__version__),
    ]
    try:
        import pytest

        rows.append(("pytest", pytest.__version__))
    except ImportError:  # pragma: no cover
        rows.append(("pytest", "not importable"))
    try:
        import hypothesis

        rows.append(("hypothesis", hypothesis.__version__))
    except ImportError:  # pragma: no cover
        rows.append(("hypothesis", "not importable"))
    try:
        import joblib

        rows.append(("joblib", joblib.__version__))
    except ImportError:  # pragma: no cover
        rows.append(("joblib", "not importable"))
    for k, v in rows:
        print(f"  {k:14s} {v}")
    report.check("telemdrift version is 0.1.0", telemdrift.__version__ == "0.1.0")

    report.section("2. Compute budget")
    cpu = os.cpu_count()
    aff = len(os.sched_getaffinity(0))
    print(f"  os.cpu_count()               {cpu}")
    print(f"  len(os.sched_getaffinity(0)) {aff}")
    report.check("two cores, as the batch specification assumes", aff == 2,
                 f"affinity={aff}")

    report.section("3. The SciPy defect this package routes around")
    print("  scipy.stats.kstest(x, 'norm', args=(loc, scale)) raises TypeError on")
    print("  the installed SciPy. This package never calls it; the windowed KS")
    print("  detector uses its own two-sample statistic, checked against")
    print("  scipy.stats.ks_2samp in validate_known_answers.py.")
    import numpy as np
    from scipy.stats import kstest

    x = np.random.default_rng(0).standard_normal(50)
    raised = None
    try:
        kstest(x, "norm", args=(0.0, 1.0))
    except TypeError as exc:
        raised = str(exc)
    if raised is None:
        print("  NOTE: the call did NOT raise on this SciPy build.")
        report.finding(
            "scipy.stats.kstest(x,'norm',args=(loc,scale)) did not raise here; "
            "the defect recorded by an earlier session may be build-specific. "
            "This package does not depend on either behaviour."
        )
        report.check("the documented route-around remains valid either way", True)
    else:
        print(f"  TypeError: {raised}")
        report.check("the documented SciPy defect reproduces", True, raised[:60])

    report.section("4. Benchmark configuration")
    cfg = telemdrift.STANDARD
    print(f"  target ARL0            {cfg.target_arl0:.0f} samples")
    print(f"  calibration seeds      {list(cfg.cal_seeds)} x {cfg.cal_length} samples"
          f" = {len(cfg.cal_seeds) * cfg.cal_length}")
    print(f"  evaluation seeds       {list(cfg.eval_seeds)} x {cfg.eval_length} samples"
          f" = {len(cfg.eval_seeds) * cfg.eval_length}")
    print(f"  sweep seeds            {list(cfg.sweep_seeds)} x {cfg.sweep_length} samples")
    print(f"  ARL1 replicates        {cfg.arl1_replicates}")
    print(f"  pre-change length      {cfg.pre_length} samples")
    print(f"  ARL1 censoring budget  {cfg.arl1_budget} samples")
    print(f"  learned window         {cfg.window} samples")
    report.check("calibration and evaluation seeds are disjoint", cfg.seeds_disjoint)
    report.check("pre-change length exceeds the longest detector warm-up (300)",
                 cfg.pre_length > 300, f"pre_length={cfg.pre_length}")

    report.section("5. Declared runtime dependencies")
    print("  numpy, scipy, scikit-learn, joblib. Nothing else, per the batch")
    print("  specification, so the batch pip-audit stays diffable. matplotlib is")
    print("  an optional extra needed only for the figures.")
    report.check("no alternatives-table package is imported at runtime",
                 "river" not in sys.modules and "ruptures" not in sys.modules)


if __name__ == "__main__":
    raise SystemExit(run("validate_environment",
                         "telemdrift 0.1.0 - environment and configuration record",
                         body))
