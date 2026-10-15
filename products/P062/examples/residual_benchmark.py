"""Physics alone against physics plus a learned residual, on held-out records.

Run: ``python examples/residual_benchmark.py``
Writes: ``screenshots/residual_benchmark.png`` and
``validation/example_residual_benchmark_output.txt``.
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
    build_features,
    calibration_model,
    design,
    fit,
    make_dataset,
    run_benchmark,
    truth_model,
)
from twincalib.residual import ResidualModel  # noqa: E402

DESIGNS = ("baseline-eclipse", "full-sun", "lumped-nodes", "weak-excitation")


def main() -> None:
    out = Tee(ROOT / "validation" / "example_residual_benchmark_output.txt")
    results = {}
    for name in DESIGNS:
        result = run_benchmark(design(name), n_train=4, n_test=3)
        results[name] = result
        out.rule(name)
        out(result.summary())
        out("")
        out(result.splits[1e-2].summary())
        out("")
        out(f"fitted kernel: {result.residual_kernel}")

    out.rule("verdict ledger")
    out(f"  {'design':<18}{'discrep K':>11}{'phys RMSE':>11}{'comb RMSE':>11}"
        f"{'improve':>9}{'weak frac':>11}{'winner':>18}")
    for name, r in results.items():
        out(
            f"  {name:<18}{r.discrepancy_rms:>11.4f}{r.rmse_physics_pooled:>11.4f}"
            f"{r.rmse_combined_pooled:>11.4f}{r.improvement_pooled:>+9.4f}"
            f"{r.splits[1e-2].weak_fraction:>11.5f}{r.winner:>18}"
        )

    # A detail panel for one held-out trajectory on the baseline design.
    d = design("baseline-eclipse")
    model = calibration_model()
    train = make_dataset(d, seed=4200, phase=0.0, model=truth_model())
    test = make_dataset(d, seed=5200, phase=0.5, model=truth_model())
    problem = CalibrationProblem(
        model=model,
        inputs=train.inputs,
        initial_state=d.initial_state,
        channels=d.channels,
        sample_stride=d.sample_stride,
        y=train.y,
        sigma=train.sigma,
    )
    physics = fit(problem, np.asarray(d.theta_true, dtype=float), max_nfev=2000)
    f_train, p_train = build_features(
        model, physics.theta, train.inputs, d.initial_state, d.channels, d.sample_stride
    )
    learner = ResidualModel(noise_level=d.sigma_k**2).fit(f_train, train.y - p_train)
    f_test, p_test = build_features(
        model, physics.theta, test.inputs, d.initial_state, d.channels, d.sample_stride
    )
    pred = learner.predict(f_test)
    n = d.n_samples
    times = test.sample_times / 3600.0

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.0))
    for col, label in enumerate(d.channel_labels):
        sl = slice(col * n, (col + 1) * n)
        ax = axes[0, col]
        ax.plot(times, test.y[sl], ".", color="0.6", markersize=2.5, label="measured")
        ax.plot(times, p_test[sl], "-", color="tab:blue", linewidth=1.3, label="physics only")
        ax.plot(
            times,
            p_test[sl] + pred.mean[sl],
            "-",
            color="tab:red",
            linewidth=1.3,
            label="physics + residual",
        )
        ax.set_ylabel(f"{label} [K]")
        ax.set_title(f"held-out trajectory, {label}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        ax = axes[1, col]
        resid = test.y[sl] - p_test[sl]
        ax.plot(times, resid, ".", color="0.6", markersize=2.5, label="physics residual")
        ax.plot(times, pred.mean[sl], "-", color="tab:red", linewidth=1.3, label="learned mean")
        band = np.sqrt(pred.std[sl] ** 2 + d.sigma_k**2)
        ax.fill_between(
            times,
            pred.mean[sl] - 1.96 * band,
            pred.mean[sl] + 1.96 * band,
            color="tab:red",
            alpha=0.18,
            label="95 per cent predictive band",
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("time [h]")
        ax.set_ylabel(f"{label} residual [K]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(
        "baseline-eclipse, held-out phase 0.5: what the residual model adds and "
        "what it is unsure about",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    target = ROOT / "screenshots" / "residual_benchmark.png"
    fig.savefig(target, dpi=130)
    plt.close(fig)
    out("")
    out(f"wrote {target.relative_to(ROOT)}")
    out.save()


if __name__ == "__main__":
    main()
