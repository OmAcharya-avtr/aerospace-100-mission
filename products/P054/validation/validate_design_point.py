"""Validate the two design-point searches against each other and the closed form.

The design point is the whole basis of the analytic importance-sampling tilt
and of the surrogate's contribution, so it is found by two independent paths
and both are compared with the closed-form answer where one exists.

The ray search (find_design_point_radial) samples directions, so its accuracy
degrades with input dimension. That degradation is measured here rather than
left for a user to discover, and it is the reason the ray search finishes with
one constrained-optimiser step.

Run from the product directory:

    python validation/validate_design_point.py
"""

from __future__ import annotations

import numpy as np
from _reporting import Report  # noqa: E402

from rareverify.limitstates import (  # noqa: E402
    LinearGaussianLimitState,
    LognormalRatioLimitState,
    RippledLimitState,
)
from rareverify.tilting import find_design_point, find_design_point_radial  # noqa: E402


def main() -> int:
    report = Report("validate_design_point")

    report.line("## 1. Closed-form instances: both searches against the exact answer")
    report.line(
        "For a linear limit state the design point is beta * a exactly; for "
        "the lognormal ratio it is the projection of the origin onto the "
        "linear failure boundary in u-space, with norm beta."
    )
    report.line(
        f"{'instance':>28} {'exact beta':>12} {'ray search':>12} {'SLSQP':>12} "
        f"{'ray rel err':>13} {'SLSQP rel err':>14}"
    )
    ok = True
    instances = [
        ("linear beta=2.5 d=2", LinearGaussianLimitState(beta=2.5, dimension=2), 2.5),
        ("linear beta=3.719 d=2", LinearGaussianLimitState(beta=3.719, dimension=2), 3.719),
        ("linear beta=4.753 d=5", LinearGaussianLimitState(beta=4.753, dimension=5), 4.753),
        (
            "linear beta=3.719 d=8 skew",
            LinearGaussianLimitState(
                beta=3.719, dimension=8, direction=[1, 2, 3, 0, -1, 0.5, 2, -3]
            ),
            3.719,
        ),
        ("lognormal default", LognormalRatioLimitState(), 3.719),
    ]
    for label, state, exact in instances:
        ray, ray_ok = find_design_point_radial(
            state.g, state.dimension, rng=np.random.default_rng(11)
        )
        slsqp, slsqp_ok = find_design_point(
            state.g, state.dimension, n_starts=24, rng=np.random.default_rng(11)
        )
        ray_norm = float(np.linalg.norm(ray))
        slsqp_norm = float(np.linalg.norm(slsqp))
        ok &= ray_ok and slsqp_ok
        ok &= abs(ray_norm / exact - 1.0) < 1e-5
        ok &= abs(slsqp_norm / exact - 1.0) < 1e-5
        report.line(
            f"{label:>28} {exact:>12.6f} {ray_norm:>12.6f} {slsqp_norm:>12.6f} "
            f"{ray_norm / exact - 1:>13.3e} {slsqp_norm / exact - 1:>14.3e}"
        )
    report.check(
        "both searches recover the closed-form reliability index to 1e-5 "
        "relative on every closed-form instance",
        ok,
    )

    report.line("")
    report.line("## 2. The rippled instances, where no closed form exists")
    report.line(
        "The two searches are each other's only reference here. The smooth "
        "design point is shown for contrast: it is what an analyst gets "
        "without a surrogate and it is not the most probable failure point."
    )
    report.line(
        f"{'instance':>28} {'smooth beta':>12} {'ray search':>12} {'SLSQP':>12} "
        f"{'agreement':>12}"
    )
    agree = True
    for label, state in (
        ("rippled A=0.8 w=1.5", RippledLimitState()),
        ("rippled A=2.0 w=2.0 b=5.0", RippledLimitState(beta=5.0, amplitude=2.0, frequency=2.0)),
        ("rippled A=2.5 w=2.0 b=5.5", RippledLimitState(beta=5.5, amplitude=2.5, frequency=2.0)),
        ("rippled A=0.8 w=6.0", RippledLimitState(amplitude=0.8, frequency=6.0)),
    ):
        ray, _ = find_design_point_radial(
            state.g, state.dimension, n_directions=512, rng=np.random.default_rng(13)
        )
        slsqp, _ = find_design_point(
            state.g, state.dimension, n_starts=40, rng=np.random.default_rng(13)
        )
        ray_norm = float(np.linalg.norm(ray))
        slsqp_norm = float(np.linalg.norm(slsqp))
        relative = abs(ray_norm / slsqp_norm - 1.0)
        agree &= relative < 1e-4
        report.line(
            f"{label:>28} {float(np.linalg.norm(state.design_point())):>12.6f} "
            f"{ray_norm:>12.6f} {slsqp_norm:>12.6f} {relative:>12.3e}"
        )
    report.check(
        "the two searches agree to 1e-4 relative on every rippled instance",
        agree,
    )

    report.line("")
    report.line("## 3. Measured limitation: the ray search degrades with dimension")
    report.line(
        "Rippled limit state, beta=3.719, A=0.8, w=1.5. The design point lies "
        "in the plane of the two active directions, so the true answer is "
        "3.072735 in every dimension. With polish=False the search is the raw "
        "ray search; with polish=True it finishes with one SLSQP step from "
        "the ray-search point."
    )
    report.line(
        f"{'d':>4} {'n_dir':>7} {'raw ray':>12} {'raw rel err':>13} "
        f"{'polished':>12} {'polished rel err':>18}"
    )
    truth = 3.072735
    polished_ok = True
    for dimension in (2, 4, 6, 10):
        for n_dir in (192, 512, 2048):
            raw, _ = find_design_point_radial(
                RippledLimitState(dimension=dimension).g,
                dimension,
                n_directions=n_dir,
                polish=False,
                rng=np.random.default_rng(17),
            )
            polished, _ = find_design_point_radial(
                RippledLimitState(dimension=dimension).g,
                dimension,
                n_directions=n_dir,
                polish=True,
                rng=np.random.default_rng(17),
            )
            raw_norm = float(np.linalg.norm(raw))
            polished_norm = float(np.linalg.norm(polished))
            polished_ok &= abs(polished_norm / truth - 1.0) < 1e-4
            report.line(
                f"{dimension:>4} {n_dir:>7} {raw_norm:>12.6f} "
                f"{raw_norm / truth - 1:>13.3e} {polished_norm:>12.6f} "
                f"{polished_norm / truth - 1:>18.3e}"
            )
    report.check(
        "the polished ray search is within 1e-4 relative of the true "
        "reliability index in every dimension and direction count tested",
        polished_ok,
    )
    raw_errors = {}
    for dimension in (2, 4, 6, 10):
        raw, _ = find_design_point_radial(
            RippledLimitState(dimension=dimension).g,
            dimension,
            n_directions=192,
            polish=False,
            rng=np.random.default_rng(17),
        )
        raw_errors[dimension] = abs(float(np.linalg.norm(raw)) / truth - 1.0)
    report.line(f"raw ray-search relative error at 192 directions: {raw_errors}")
    report.check(
        "the raw ray-search error grows with dimension, which is the "
        "documented limitation",
        raw_errors[10] > raw_errors[2],
        f"(d=2: {raw_errors[2]:.3e}, d=10: {raw_errors[10]:.3e})",
    )

    report.line("")
    report.line("## 4. Non-convergence is reported, not papered over")
    state = LinearGaussianLimitState(beta=12.0, dimension=2)
    point, converged = find_design_point_radial(
        state.g, 2, max_radius=4.0, rng=np.random.default_rng(1)
    )
    report.line(
        f"beta=12 with max_radius=4: converged={converged}, "
        f"point={np.round(point, 6).tolist()}"
    )
    report.check(
        "a design point outside max_radius returns converged=False",
        converged is False and float(np.linalg.norm(point)) == 0.0,
    )

    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
