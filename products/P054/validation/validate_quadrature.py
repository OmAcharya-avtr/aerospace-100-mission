"""Validate the Gauss-Hermite reference for the rippled limit state.

The rippled limit state has no closed-form failure probability; its reference
value is a one-dimensional Gaussian expectation evaluated by Gauss-Hermite
quadrature. A known-answer tolerance can only be as tight as that reference is
converged, so the convergence is measured here rather than assumed.

Run from the product directory:

    python validation/validate_quadrature.py
"""

from __future__ import annotations

import math

import numpy as np
from _reporting import Report  # noqa: E402

from rareverify.limitstates import (  # noqa: E402
    LinearGaussianLimitState,
    RippledLimitState,
)


def main() -> int:
    report = Report("validate_quadrature")

    report.line("## 1. Zero amplitude must reproduce the closed form exactly")
    report.line(
        "At amplitude 0 the rippled limit state is the linear one, so the "
        "quadrature must return Phi(-beta) to machine precision."
    )
    report.line(f"{'beta':>8} {'quadrature':>22} {'Phi(-beta)':>22} {'rel diff':>12}")
    exact_ok = True
    for beta in (1.0, 2.5, 3.719, 4.753, 5.5):
        quad = RippledLimitState(beta=beta, amplitude=0.0).analytic_probability()
        closed = LinearGaussianLimitState(beta=beta).analytic_probability()
        exact_ok &= abs(quad / closed - 1.0) < 1e-13
        report.line(f"{beta:>8.3f} {quad:>22.16e} {closed:>22.16e} {quad / closed - 1:>12.3e}")
    report.check("zero-amplitude quadrature matches Phi(-beta) to 1e-13", exact_ok)

    report.line("")
    report.line("## 2. Node-count convergence at three ripple frequencies")
    report.line(
        "Reference taken as the 3200-node value. The package default is 400 "
        "nodes. Frequency is in reciprocal standard-normal units."
    )
    nodes = (20, 50, 100, 200, 400, 800, 1600)
    worst_default = 0.0
    for amplitude, frequency in ((0.8, 1.5), (2.5, 2.0), (0.8, 6.0), (1.0, 12.0)):
        state = RippledLimitState(beta=3.719, amplitude=amplitude, frequency=frequency)
        reference = state.analytic_probability(nodes=3200)
        report.line(f"amplitude={amplitude} frequency={frequency} ref(3200)={reference:.16e}")
        report.line(f"{'nodes':>8} {'value':>22} {'rel error':>14}")
        for n in nodes:
            value = state.analytic_probability(nodes=n)
            error = abs(value / reference - 1.0)
            if n == 400:
                worst_default = max(worst_default, error)
            report.line(f"{n:>8} {value:>22.16e} {error:>14.3e}")
    report.check(
        "the default 400 nodes is converged to better than 1e-5 relative at "
        "every frequency tested up to 12",
        worst_default < 1e-5,
        f"(worst relative error at 400 nodes: {worst_default:.3e})",
    )
    report.line(
        "The first version of this package defaulted to 200 nodes, which is "
        "converged to 5.7e-5 relative at frequency 12 and would have made a "
        "tolerance stated in counting-noise floors meaningless for a "
        "high-frequency instance. The default was raised to 400 after this "
        "measurement, not before it."
    )
    report.line(
        "The node requirement grows with frequency, as it must: the integrand "
        "oscillates faster. A user who raises the frequency beyond 12 must "
        "re-run this script before trusting a tolerance stated in noise floors."
    )

    report.line("")
    report.line("## 3. Quadrature against a direct Monte-Carlo count")
    report.line(
        "An independent check that the one-dimensional reduction is right, "
        "not just converged. Tolerance 4 counting-noise floors."
    )
    report.line(
        f"{'amplitude':>10} {'frequency':>10} {'quadrature':>16} {'count':>16} "
        f"{'floors':>8}"
    )
    mc_ok = True
    rng = np.random.default_rng(4242)
    for amplitude, frequency in ((0.0, 1.5), (0.8, 1.5), (2.0, 2.0), (0.8, 6.0)):
        state = RippledLimitState(beta=2.0, amplitude=amplitude, frequency=frequency)
        reference = state.analytic_probability()
        x = rng.standard_normal((2_000_000, 2))
        count = float(np.mean(state.failure(x)))
        floor = math.sqrt(reference * (1 - reference) / 2_000_000)
        floors = abs(count - reference) / floor
        mc_ok &= floors <= 4.0
        report.line(
            f"{amplitude:>10.2f} {frequency:>10.2f} {reference:>16.8e} "
            f"{count:>16.8e} {floors:>8.3f}"
        )
    report.check(
        "the quadrature reference agrees with a 2e6-sample count within 4 "
        "counting-noise floors at every setting",
        mc_ok,
    )

    report.line("")
    report.line("## 4. Tool defect recorded: numpy.polynomial.hermite.hermgauss")
    report.line(
        "numpy 2.5.3: hermgauss(n) overflows and returns NaN weights for "
        "n >= 400, so the first implementation of the rippled reference "
        "silently produced NaN at 400 nodes. scipy.special.roots_hermite is "
        "stable to at least 3200 nodes and is what the package uses."
    )
    broken = []
    for n in (200, 300, 400, 800):
        with np.errstate(all="ignore"):
            _, weights = np.polynomial.hermite.hermgauss(n)
        if not np.all(np.isfinite(weights)):
            broken.append(n)
        report.line(f"  numpy hermgauss({n}): finite weights = {np.all(np.isfinite(weights))}")
    from scipy import special

    for n in (200, 400, 800, 1600, 3200):
        _, weights = special.roots_hermite(n)
        report.line(
            f"  scipy roots_hermite({n}): finite weights = "
            f"{bool(np.all(np.isfinite(weights)))}"
        )
    report.check(
        "numpy hermgauss is confirmed broken at 400 nodes and above on this "
        "numpy version",
        400 in broken,
        f"(NaN weights at n in {broken})",
    )
    report.check(
        "scipy roots_hermite is finite at 3200 nodes",
        bool(np.all(np.isfinite(special.roots_hermite(3200)[1]))),
    )

    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
