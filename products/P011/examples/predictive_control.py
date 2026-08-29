"""Learned predictive control against the classical integrator and pure delay.

Produces ``screenshots/predictive_control.png``.  Every number in the figure is
measured on phase screens the model was not trained on, and the classical
integrator's gain is tuned on a separate tuning screen so that it is presented
at its best configuration.

Run from products/P011:
    PYTHONPATH=src python examples/predictive_control.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from waveforge.datasets import make_slope_dataset  # noqa: E402
from waveforge.loop import AOConfig, AOSystem  # noqa: E402
from waveforge.predictor import (  # noqa: E402
    LinearSlopePredictor,
    PureDelayPredictor,
    build_lagged_dataset,
)

OUTPUT = Path(__file__).resolve().parent.parent / "screenshots" / "predictive_control.png"
BASE = AOConfig(seed=0)
N_HISTORY = 4


def main() -> None:
    data = make_slope_dataset(BASE, n_frames=350, train_seeds=(101, 102), test_seeds=(901,))
    models = {
        horizon: LinearSlopePredictor(
            n_history=N_HISTORY, horizon=horizon, n_members=8, random_state=0
        ).fit(data.train)
        for horizon in (1, 2, 3, 4)
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))

    horizons = np.array([1, 2, 3, 4])
    ml_rmse, delay_rmse = [], []
    for horizon in horizons:
        x, y = build_lagged_dataset(data.test, N_HISTORY, int(horizon))
        mean, _ = models[int(horizon)].predict_batch(x)
        ml_rmse.append(float(np.sqrt(np.mean((mean - y) ** 2))))
        delay_rmse.append(float(np.sqrt(np.mean((x[:, -data.n_slopes :] - y) ** 2))))
    axes[0, 0].plot(horizons, delay_rmse, "k-o", ms=5, label="pure delay (no prediction)")
    axes[0, 0].plot(horizons, ml_rmse, "-s", ms=5, color="tab:blue", label="ridge ensemble")
    axes[0, 0].set_xlabel("forecast horizon [frames]")
    axes[0, 0].set_ylabel("slope RMSE [rad/m]")
    axes[0, 0].set_title("Open-loop forecast error on held-out screens")
    axes[0, 0].set_xticks(horizons)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    for h, a, b in zip(horizons, ml_rmse, delay_rmse, strict=True):
        axes[0, 0].annotate(f"x{b / a:.2f}", (h, a), textcoords="offset points",
                            xytext=(6, -12), fontsize=7, ha="left", color="tab:blue")

    x, y = build_lagged_dataset(data.test, N_HISTORY, 2)
    mean, sigma = models[2].predict_batch(x)
    z = np.abs(mean - y) / sigma
    ks = np.linspace(0.1, 3.5, 40)
    measured = [float(np.mean(z < k)) for k in ks]
    from math import erf

    nominal = [erf(k / np.sqrt(2.0)) for k in ks]
    axes[0, 1].plot(nominal, measured, "-", color="tab:purple", lw=1.8)
    axes[0, 1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0, 1].set_xlabel("nominal Gaussian coverage")
    axes[0, 1].set_ylabel("measured coverage")
    axes[0, 1].set_title(
        "Uncertainty calibration, horizon 2\n"
        "(diagonal = calibrated; below = intervals too narrow)"
    )
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)

    tuning = AOSystem(replace(BASE, seed=555))
    evaluation = AOSystem(replace(BASE, seed=901))
    gains = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    width = 0.26
    positions = np.arange(len(horizons))
    bars = {"integrator": [], "pure delay": [], "learned": []}
    for delay in horizons:
        for name, controller in (
            ("integrator", None),
            ("pure delay", PureDelayPredictor()),
            ("learned", models[int(delay)]),
        ):
            best_gain, best_value = None, np.inf
            for gain in gains:
                value = tuning.run(
                    250, warmup_frames=80, rng=4242, gain=gain,
                    delay_frames=int(delay), predictor=controller,
                ).mean_residual_variance
                if value < best_value:
                    best_gain, best_value = gain, value
            bars[name].append(
                evaluation.run(
                    400, warmup_frames=120, rng=31337, gain=best_gain,
                    delay_frames=int(delay), predictor=controller,
                ).mean_residual_variance
            )
    for index, (name, colour) in enumerate(
        (("integrator", "0.45"), ("pure delay", "tab:orange"), ("learned", "tab:blue"))
    ):
        axes[1, 0].bar(positions + (index - 1) * width, bars[name], width,
                       label=name, color=colour)
    axes[1, 0].set_xticks(positions)
    axes[1, 0].set_xticklabels([str(int(h)) for h in horizons])
    axes[1, 0].set_xlabel("total loop latency [frames]")
    axes[1, 0].set_ylabel(r"mean residual variance [rad$^2$]")
    axes[1, 0].set_title("Closed loop, each controller at its own tuned gain")
    axes[1, 0].grid(True, axis="y", alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    winds = (5.0, 10.0, 15.0, 20.0)
    classical_v, learned_v = [], []
    for wind in winds:
        system = AOSystem(replace(BASE, seed=901, wind_speed_m_s=wind))
        frames = min(300, int(system.atmosphere.max_frames) - 1)
        classical_v.append(
            system.run(frames, warmup_frames=100, rng=31337, gain=0.4,
                       delay_frames=2).mean_residual_variance
        )
        learned_v.append(
            system.run(frames, warmup_frames=100, rng=31337, gain=0.4, delay_frames=2,
                       predictor=models[2]).mean_residual_variance
        )
    axes[1, 1].plot(winds, classical_v, "k-o", ms=5, label="classical integrator")
    axes[1, 1].plot(winds, learned_v, "-s", ms=5, color="tab:red",
                    label="learned (trained at 10 m/s)")
    axes[1, 1].axvline(10.0, color="tab:green", ls=":", lw=1.5)
    axes[1, 1].annotate("training wind", (10.2, max(classical_v) * 0.95), fontsize=7)
    axes[1, 1].set_xlabel("wind speed [m/s]")
    axes[1, 1].set_ylabel(r"mean residual variance [rad$^2$]")
    axes[1, 1].set_title("Failure mode: wind speed outside the training set")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(
        "WaveForge — learned predictive control (research-grade, not flight-qualified; "
        "not certified for operational flight use)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
