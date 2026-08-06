"""Example: centroid error vs SNR, classical baselines vs ML ensemble.

Trains a small MLCentroider ensemble across a range of signal levels, then
compares RMS centroid error against the plain and thresholded centre-of-
gravity and the calibrated quad-cell on held-out data.  Saves
../screenshots/error_vs_snr.png.  Runtime ~1 min on 2 CPU cores.

Run from products/P008/:  PYTHONPATH=src python examples/error_vs_snr.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from centroidnet import (  # noqa: E402
    MLCentroider,
    cog_centroid,
    generate_spots,
    quadcell_centroid,
    snr_estimate,
)

GRID, SIGMA, B, R = 16, 1.5, 2.0, 3.0
SIGNALS = [100.0, 200.0, 500.0, 1000.0, 3000.0, 10000.0]
SCALE = SIGMA * np.sqrt(np.pi / 2.0)  # quad-cell small-offset calibration [px]


def rms(pred, truth):
    return float(np.sqrt(np.mean(np.sum((pred - truth) ** 2, axis=1))))


def main() -> None:
    train_x, train_y = [], []
    for i, s in enumerate(SIGNALS):
        im, tr = generate_spots(500, GRID, SIGMA, s, B, R, seed=300 + i)
        train_x.append(im)
        train_y.append(tr)
    model = MLCentroider(n_estimators=5, random_state=0)
    model.fit(np.concatenate(train_x), np.concatenate(train_y))

    snrs, e_cog, e_cogt, e_quad, e_ml, e_std = [], [], [], [], [], []
    for i, s in enumerate(SIGNALS):
        im, tr = generate_spots(300, GRID, SIGMA, s, B, R, seed=8800 + i)
        snrs.append(snr_estimate(s, B, R, GRID))
        e_cog.append(rms(np.array([cog_centroid(f) for f in im]), tr))
        cogt = []
        for f in im:
            try:
                cogt.append(cog_centroid(f, threshold=B + R))
            except ValueError:  # threshold removed all flux at very low SNR
                cogt.append(cog_centroid(f))
        e_cogt.append(rms(np.array(cogt), tr))
        e_quad.append(rms(np.array([quadcell_centroid(f, scale=SCALE) for f in im]), tr))
        pred, std = model.predict(im, return_std=True)
        e_ml.append(rms(pred, tr))
        e_std.append(float(std.mean()))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(snrs, e_cog, "o-", label="CoG (plain)")
    ax.loglog(snrs, e_cogt, "s-", label="CoG (threshold B+R)")
    ax.loglog(snrs, e_quad, "^-", label="quad-cell (calibrated)")
    ax.loglog(snrs, e_ml, "d-", color="crimson", label="ML ensemble (5 x MLP)")
    ax.fill_between(
        snrs,
        np.array(e_ml) - np.array(e_std),
        np.array(e_ml) + np.array(e_std),
        color="crimson",
        alpha=0.15,
        label="ML +/- mean ensemble spread",
    )
    ax.set_xlabel("detection SNR")
    ax.set_ylabel("RMS radial centroid error [px]")
    ax.set_title(
        f"Centroid error vs SNR ({GRID}x{GRID}, $\\sigma$={SIGMA} px, "
        f"B={B} e-/px, R={R} e-)"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out = pathlib.Path(__file__).resolve().parents[1] / "screenshots" / "error_vs_snr.png"
    fig.savefig(out, dpi=130)
    sys.stdout.write(f"saved {out}\n")


if __name__ == "__main__":
    main()
