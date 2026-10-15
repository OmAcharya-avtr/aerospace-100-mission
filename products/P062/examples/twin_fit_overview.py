"""The twin, the data, the fit and what is left over.

Run: ``python examples/twin_fit_overview.py``
Writes: ``screenshots/twin_fit_overview.png`` and
``validation/example_twin_fit_overview_output.txt``.
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
    analyse,
    calibration_model,
    design,
    fit,
    make_dataset,
    render_correlation,
    render_design,
    render_fit,
    render_identifiability,
    truth_model,
)


def main() -> None:
    out = Tee(ROOT / "validation" / "example_twin_fit_overview_output.txt")
    d = design("baseline-eclipse")
    model = calibration_model()
    data = make_dataset(d, seed=3301, model=truth_model())
    problem = CalibrationProblem(
        model=model,
        inputs=data.inputs,
        initial_state=d.initial_state,
        channels=d.channels,
        sample_stride=d.sample_stride,
        y=data.y,
        sigma=data.sigma,
    )
    result = fit(problem, np.asarray(d.theta_true, dtype=float) * 1.3, max_nfev=2000)
    report = analyse(result.sensitivity)
    out(render_design(d))
    out("")
    out(render_identifiability(report))
    out("")
    out(render_fit(result, units=model.parameter_units, theta_true=d.theta_true))
    out("")
    out(render_correlation(result.parameter_names, result.correlation))

    fast, slow = model.time_constants(result.theta, 280.0)
    out("")
    out(
        f"linearised time constants at the fit: {fast:.1f} s and {slow:.1f} s; "
        f"the record is {d.duration:g} s at {d.dt:g} s sampling"
    )
    t1_ss, t2_ss = model.steady_state(result.theta, d.dissipation_high, 1.0)
    out(
        f"algebraic equilibrium at full sun and {d.dissipation_high:g} W: "
        f"T1 = {t1_ss:.3f} K, T2 = {t2_ss:.3f} K"
    )

    prediction = problem.predict(result.theta)
    residual = data.y - prediction
    n = d.n_samples
    times = data.sample_times / 3600.0

    fig, axes = plt.subplots(3, 2, figsize=(13.0, 9.0), height_ratios=[2, 1, 1])
    for col, label in enumerate(d.channel_labels):
        sl = slice(col * n, (col + 1) * n)
        ax = axes[0, col]
        ax.plot(times, data.y[sl], ".", color="0.65", markersize=2.5, label="measured")
        ax.plot(times, prediction[sl], "-", color="tab:blue", linewidth=1.4, label="fitted twin")
        ax.set_ylabel(f"{label} [K]")
        ax.set_title(f"{label}: measurement and fitted twin")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax = axes[1, col]
        ax.plot(times, residual[sl], ".", color="tab:red", markersize=2.5)
        ax.axhline(0.0, color="black", linewidth=0.8)
        for sign in (-1.0, 1.0):
            ax.axhline(
                sign * d.sigma_k, color="grey", linestyle="--", linewidth=0.9
            )
        ax.set_ylabel("residual [K]")
        ax.set_title(
            f"residual, RMS {float(np.sqrt(np.mean(residual[sl] ** 2))):.4f} K "
            f"against {d.sigma_k:g} K of noise (dashed)"
        )
        ax.grid(alpha=0.3)
    ax = axes[2, 0]
    ax.plot(times, data.inputs.dissipation[:: d.sample_stride], drawstyle="steps-post")
    ax.set_xlabel("time [h]")
    ax.set_ylabel("dissipation [W]")
    ax.set_title("unit dissipation, a known input")
    ax.grid(alpha=0.3)
    ax = axes[2, 1]
    ax.plot(
        times,
        data.inputs.sun[:: d.sample_stride],
        drawstyle="steps-post",
        color="tab:orange",
    )
    ax.set_xlabel("time [h]")
    ax.set_ylabel("illumination fraction")
    ax.set_title("illumination, the input that separates the two heat loads")
    ax.grid(alpha=0.3)
    fig.suptitle(
        "baseline-eclipse fitted with the calibration model against data from the "
        "truth model: the residual is structured, not noise",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    target = ROOT / "screenshots" / "twin_fit_overview.png"
    fig.savefig(target, dpi=130)
    plt.close(fig)
    out("")
    out(f"wrote {target.relative_to(ROOT)}")
    out.save()


if __name__ == "__main__":
    main()
