"""The worked example printed in the README, so the two can never drift."""

from __future__ import annotations

import numpy as np

from invariantset import (
    Polytope,
    maximal_robust_invariant_set,
    minkowski_sum,
    pontryagin_difference,
    verify_robust_invariance,
)


def main() -> None:
    # Single-axis attitude loop, x = [theta, theta_dot] in [rad, rad/s].
    dt = 0.05
    A_plant = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5 * dt * dt], [dt]])
    K = np.array([[8.0, 3.8]])          # poles at 0.9 +/- 0.1j
    A_cl = A_plant - B @ K

    # |theta| <= 0.30 rad, |theta_dot| <= 0.50 rad/s, |u| = |K x| <= 3 rad/s^2
    X = Polytope(
        np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [8.0, 3.8], [-8.0, -3.8]], float),
        np.array([0.30, 0.30, 0.50, 0.50, 3.0, 3.0]),
    )
    # 0.12 rad/s^2 of unmodelled acceleration held for one sample
    W = Polytope.from_box([0.0, 0.0], [0.5 * dt * dt * 0.12, dt * 0.12])

    result = maximal_robust_invariant_set(A_cl, X, W, max_iter=50)
    print(result.report())

    invariant, margin = verify_robust_invariance(A_cl, result.polytope, W)
    print(f"independently invariant (tol 1e-9): "
          f"{verify_robust_invariance(A_cl, result.polytope, W, tol=1e-9)[0]}")
    print(f"worst facet margin [rad or rad/s]: {margin:.3e}")
    print(f"area of S_inf [rad.rad/s]        : {result.polytope.volume():.9f}")
    print(f"area of X     [rad.rad/s]        : {X.volume():.9f}")
    print(f"S_inf covers                     : "
          f"{100.0 * result.polytope.volume() / X.volume():.4f} % of X")

    # The set algebra, and the identity that does not hold.
    eroded = pontryagin_difference(result.polytope, W)
    reopened = minkowski_sum(eroded, W)
    print(f"area of S_inf (-) W              : {eroded.volume():.9f}")
    print(f"area of (S_inf (-) W) (+) W      : {reopened.volume():.9f}")
    print(f"the erode-dilate deficit         : "
          f"{result.polytope.volume() - reopened.volume():.3e} "
          f"({100.0 * (1 - reopened.volume() / result.polytope.volume()):.4f} % of S_inf)")


if __name__ == "__main__":
    main()
