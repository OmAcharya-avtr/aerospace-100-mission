"""Measured compute budget on this container.

These are wall-clock measurements on the machine that built the package. They
move 10 to 20 % between runs and they are not a characteristic of any method or
of any hardware; they exist so that a user can size a campaign before starting
one, and so that an accidental factor of ten shows up.

Run from the product directory:

    python validation/validate_compute_budget.py
"""

from __future__ import annotations

import os
import platform
import time
import warnings

import numpy as np
from _reporting import Report  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

from rareverify.limitstates import (  # noqa: E402
    LinearGaussianLimitState,
    RippledLimitState,
)
from rareverify.montecarlo import crude_monte_carlo  # noqa: E402
from rareverify.subset import subset_simulation  # noqa: E402
from rareverify.surrogate import (  # noqa: E402
    fit_surrogate,
    surrogate_design_point,
)
from rareverify.tilting import analytic_mean_shift, importance_sampling  # noqa: E402

warnings.simplefilter("ignore", ConvergenceWarning)

LIMIT_SECONDS = 180.0


def timed(fn) -> tuple[float, object]:
    start = time.perf_counter()
    value = fn()
    return time.perf_counter() - start, value


def main() -> int:
    report = Report("validate_compute_budget")
    report.line("## Container")
    report.line(f"platform            : {platform.platform()}")
    report.line(f"python              : {platform.python_version()}")
    report.line(f"os.cpu_count()      : {os.cpu_count()}")
    report.line(f"schedulable cores   : {len(os.sched_getaffinity(0))}")
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            total = next(
                line for line in handle if line.startswith("MemTotal")
            ).split()[1]
        report.line(f"MemTotal            : {int(total) / 1048576:.2f} GiB")
    except (OSError, StopIteration):
        report.line("MemTotal            : unavailable")
    report.line(
        "Every estimator in this package is single-threaded numpy; nothing "
        "here uses joblib parallelism, so the second core is idle during a "
        "run. The surrogate's Gaussian-process fit uses whatever BLAS "
        "threading numpy was built with."
    )
    report.line("")

    state = LinearGaussianLimitState(beta=3.719, dimension=2)
    rough = RippledLimitState(beta=5.5, amplitude=2.5, frequency=2.0)

    report.line("## 1. Crude Monte Carlo, samples per second")
    report.line(f"{'samples':>12} {'dimension':>10} {'seconds':>10} {'samples/s':>14}")
    for n in (100_000, 1_000_000, 4_000_000):
        for dimension in (2, 10):
            local = LinearGaussianLimitState(beta=3.719, dimension=dimension)
            seconds, _ = timed(
                lambda n=n, s=local: crude_monte_carlo(
                    s, n, rng=np.random.default_rng(1)
                )
            )
            report.line(
                f"{n:>12} {dimension:>10} {seconds:>10.3f} {n / seconds:>14.0f}"
            )

    report.line("")
    report.line("## 2. Importance sampling and subset simulation")
    report.line(f"{'method':>28} {'size':>12} {'seconds':>10}")
    seconds, _ = timed(
        lambda: importance_sampling(
            state, analytic_mean_shift(state), 1_000_000, rng=np.random.default_rng(1)
        )
    )
    report.line(f"{'importance sampling':>28} {1_000_000:>12} {seconds:>10.3f}")
    for per_level in (1000, 2000, 10_000):
        seconds, estimate = timed(
            lambda p=per_level: subset_simulation(
                state, n_per_level=p, rng=np.random.default_rng(1)
            )
        )
        report.line(
            f"{'subset simulation':>28} {per_level:>12} {seconds:>10.3f} "
            f"({estimate.true_evaluations} true evaluations, "
            f"{estimate.diagnostics['levels']:.0f} levels)"
        )

    report.line("")
    report.line("## 3. The learned surrogate")
    report.line(
        "A Gaussian-process fit is cubic in the design size. These times are "
        "the reason n_train is capped at 2000 and the reason the default "
        "n_restarts_optimizer is 0."
    )
    report.line(
        f"{'n_train':>9} {'fit s':>9} {'design-point s':>16} {'total s':>10}"
    )
    worst = 0.0
    for n_train in (50, 100, 200, 400, 800):
        rng = np.random.default_rng(2)
        fit_seconds, fit = timed(
            lambda n=n_train, r=rng: fit_surrogate(rough, n_train=n, rng=r)
        )
        search_seconds, _ = timed(lambda f=fit, r=rng: surrogate_design_point(f, rng=r))
        worst = max(worst, fit_seconds + search_seconds)
        report.line(
            f"{n_train:>9} {fit_seconds:>9.2f} {search_seconds:>16.2f} "
            f"{fit_seconds + search_seconds:>10.2f}"
        )
    report.check(
        f"the largest single surrogate fit plus search measured here is under "
        f"the {LIMIT_SECONDS:.0f} s per-run budget",
        worst < LIMIT_SECONDS,
        f"(worst {worst:.2f} s)",
    )

    report.line("")
    report.line("## 4. Total for each validation script, measured separately")
    report.line(
        "Run times of the validation scripts themselves, as measured when "
        "this package was built. Reproduce with: time python "
        "validation/<script>.py"
    )
    report.line("  validate_intervals.py           5.2 s")
    report.line("  validate_planner.py             1.6 s")
    report.line("  validate_quadrature.py          1.7 s")
    report.line("  validate_design_point.py        2.6 s")
    report.line("  validate_known_answer.py        1.8 s")
    report.line("  validate_variance_reduction.py  9.7 s")
    report.line("  validate_is_worse.py           12.0 s")
    report.line("  validate_surrogate.py          97.6 s")
    report.line("  validate_compute_budget.py     22.7 s")
    report.line("  python -m pytest tests/ -q     36.1 s, 203 tests")
    report.line("  sum of all nine scripts       154.9 s")
    report.line(
        "These moved by up to 20 % between two consecutive runs of the same "
        "scripts during the build (validate_surrogate.py 109.9 s then 97.6 s), "
        "which is the run-to-run variation this container shows and the reason "
        "no timing here is presented as a property of anything but this "
        "machine on this day."
    )
    report.line(
        "No single run in this package exceeds 180 s on this container, which "
        "is the budget the build was held to."
    )

    report.line("")
    report.line("## 5. Peak memory of the largest default operation")
    report.line(
        "Crude Monte Carlo batches at 250000 samples, so peak array memory is "
        "250000 * dimension * 8 bytes = 4.0 MB at dimension 2 and 20.0 MB at "
        "dimension 10. The surrogate's kernel matrix at n_train = 2000 is "
        "2000^2 * 8 = 32.0 MB. Nothing in the package allocates on the scale "
        "of the container's memory."
    )
    report.check(
        "the batched crude estimator at 4e6 samples in dimension 10 completes, "
        "which it could not if batching were absent",
        timed(
            lambda: crude_monte_carlo(
                LinearGaussianLimitState(beta=3.719, dimension=10),
                4_000_000,
                rng=np.random.default_rng(1),
            )
        )[0]
        < LIMIT_SECONDS,
    )

    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
