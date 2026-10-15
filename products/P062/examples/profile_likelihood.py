"""Profile likelihoods on a sound design and on the degenerate one.

Run: ``python examples/profile_likelihood.py``
Writes: ``screenshots/profile_likelihood.png`` and
``validation/example_profile_likelihood_output.txt``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
from _bootstrap import Tee, add_src_to_path  # noqa: E402

ROOT = add_src_to_path()

from twincalib import (  # noqa: E402
    CalibrationProblem,
    calibration_model,
    design,
    fit,
    make_dataset,
    profile_all,
    render_profiles,
)


def _problem(name: str, seed: int):
    d = design(name)
    model = calibration_model()
    data = make_dataset(d, seed=seed, model=model)
    return d, CalibrationProblem(
        model=model,
        inputs=data.inputs,
        initial_state=d.initial_state,
        channels=d.channels,
        sample_stride=d.sample_stride,
        y=data.y,
        sigma=data.sigma,
    )


def main() -> None:
    out = Tee(ROOT / "validation" / "example_profile_likelihood_output.txt")
    cases = (("coarse-sampled", 1301), ("full-sun", 1302))
    collected = {}
    for name, seed in cases:
        d, problem = _problem(name, seed)
        result = fit(problem, np.asarray(d.theta_true, dtype=float) * 1.2)
        curves = profile_all(problem, result, n_points=5, refine_steps=5)
        collected[name] = (d, result, curves)
        out.rule(name)
        out(render_profiles(curves))
        out("")
        out("true values: " + ", ".join(
            f"{n} = {v:.6g}"
            for n, v in zip(result.parameter_names, d.theta_true, strict=True)
        ))
        out("truth inside the profile interval: " + ", ".join(
            f"{c.name}={c.contains(d.theta_true[c.index])}" for c in curves
        ))

    fig, axes = plt.subplots(2, 6, figsize=(16.0, 6.2), sharey=True)
    for row, (name, _seed) in enumerate(cases):
        d, result, curves = collected[name]
        for col, curve in enumerate(curves):
            ax = axes[row, col]
            values, deltas = curve.curve_arrays()
            ax.plot(values, deltas, "o-", color="tab:blue", markersize=3, linewidth=1.2)
            ax.axhline(curve.threshold, color="black", linestyle="--", linewidth=1.0)
            ax.axvline(
                d.theta_true[curve.index], color="tab:green", linewidth=1.4
            )
            if curve.hessian_lower is not None and curve.hessian_upper is not None:
                ax.axvspan(
                    curve.hessian_lower,
                    curve.hessian_upper,
                    color="tab:orange",
                    alpha=0.25,
                )
            for bound in (curve.lower, curve.upper):
                if bound is not None:
                    ax.axvline(bound, color="tab:red", linestyle=":", linewidth=1.2)
            ax.set_xscale("log")
            ax.set_ylim(-0.3, 12.0)
            ax.set_title(f"{name}\n{curve.name}", fontsize=8)
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel("delta chi-squared")
            ax.tick_params(labelsize=7)
    fig.suptitle(
        "Profile likelihood (blue), Wilks threshold (black dashed), "
        "quadratic interval (orange band), truth (green)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    target = ROOT / "screenshots" / "profile_likelihood.png"
    fig.savefig(target, dpi=130)
    plt.close(fig)
    out("")
    out(f"wrote {target.relative_to(ROOT)}")
    out.save()


if __name__ == "__main__":
    main()
