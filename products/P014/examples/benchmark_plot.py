#!/usr/bin/env python3
"""Example: plot the ML-vs-baseline reconstruction error curves.

A reduced-size (faster) rerun of the flux and dropout sweeps in
``validation/validate_dropout.py``, plotted rather than tabulated. The full,
larger validation run and its saved raw output are what `MODEL_CARD.md` and
`README.md` cite for numbers; this script is for the illustrative figure.
Saves ``../screenshots/benchmark_flux_dropout.png``.

Run: ``python examples/benchmark_plot.py`` from ``products/P014`` (~20-30 s).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wavelab.dataset import build_modal_geometry, generate_batch
from wavelab.ml import ZernikeSlopeEnsemble
from wavelab.modal import ModalReconstructor

NOLL = list(range(2, 14))
FLUX_LEVELS = (100.0, 300.0, 1000.0, 3000.0, 10000.0)
DROPOUT_LEVELS = (0.0, 0.15, 0.3, 0.45, 0.6)


def rms(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2)))


def evaluate(baseline, model, test):
    base_pred = np.array(
        [baseline.reconstruct(test.slopes[i], active=test.active[i]) for i in range(len(test))]
    )
    ml_pred = model.predict(test.slopes, test.active)
    return rms(base_pred - test.coeffs), rms(ml_pred - test.coeffs)


def main() -> None:
    geometry = build_modal_geometry(NOLL, n_side=8)
    baseline = ModalReconstructor(NOLL, geometry.sub_x, geometry.sub_y, method="tikhonov", reg=3e-3)
    model = ZernikeSlopeEnsemble(
        geometry.n_sub, geometry.n_modes, n_estimators=4, max_iter=250, random_state=0
    )
    train = generate_batch(geometry, 900, photon_flux=800.0, dropout_rate=0.25, seed=100)
    model.fit(train.slopes, train.active, train.coeffs)

    base_flux, ml_flux = [], []
    for flux in FLUX_LEVELS:
        test = generate_batch(geometry, 150, photon_flux=flux, dropout_rate=0.0, seed=9000)
        b, m = evaluate(baseline, model, test)
        base_flux.append(b)
        ml_flux.append(m)

    base_drop, ml_drop = [], []
    for dropout in DROPOUT_LEVELS:
        test = generate_batch(geometry, 150, photon_flux=800.0, dropout_rate=dropout, seed=9500)
        b, m = evaluate(baseline, model, test)
        base_drop.append(b)
        ml_drop.append(m)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].loglog(FLUX_LEVELS, base_flux, "o-", label="regularized modal LSQ (baseline)")
    axes[0].loglog(FLUX_LEVELS, ml_flux, "s-", label="learned ensemble (ML)")
    axes[0].set_xlabel("photon flux N [subaperture$^{-1}$]")
    axes[0].set_ylabel("Zernike coefficient RMS error [rad]")
    axes[0].set_title("vs photon flux (dropout = 0)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogy(DROPOUT_LEVELS, base_drop, "o-", label="regularized modal LSQ (baseline)")
    axes[1].semilogy(DROPOUT_LEVELS, ml_drop, "s-", label="learned ensemble (ML)")
    axes[1].set_xlabel("subaperture dropout rate")
    axes[1].set_ylabel("Zernike coefficient RMS error [rad]")
    axes[1].set_title("vs dropout rate (flux = 800)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)

    fig.suptitle("WaveLab: learned ensemble vs regularized least-squares baseline (measured, not tuned to a preferred outcome)")
    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent.parent / "screenshots"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "benchmark_flux_dropout.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")
    for flux, b, m in zip(FLUX_LEVELS, base_flux, ml_flux):
        print(f"flux={flux:8.0f}  baseline={b:.5f}  ML={m:.5f}  winner={'baseline' if b < m else 'ML'}")
    for dropout, b, m in zip(DROPOUT_LEVELS, base_drop, ml_drop):
        print(f"dropout={dropout:5.2f}  baseline={b:.5f}  ML={m:.5f}  winner={'baseline' if b < m else 'ML'}")


if __name__ == "__main__":
    main()
