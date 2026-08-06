"""Example: gallery of synthetic spots with true and estimated centroids.

Generates a small set of noisy Gaussian spot frames at several signal
levels, overlays the true centroid and the estimates from the classical
centre-of-gravity, the calibrated quad-cell and the trained ML ensemble
(with its ensemble-spread uncertainty), and saves
../screenshots/spot_gallery.png.  Runtime ~40 s on 2 CPU cores.

Run from products/P008/:  PYTHONPATH=src python examples/spot_gallery.py
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
TRAIN_SIGNALS = [100.0, 300.0, 1000.0, 3000.0]  # e-
GALLERY_SIGNALS = [100.0, 300.0, 1000.0, 3000.0]  # one column per signal
N_ROWS = 2
SCALE = SIGMA * np.sqrt(np.pi / 2.0)  # quad-cell small-offset calibration [px]


def main() -> None:
    # --- train the ML ensemble on seeded synthetic data -------------------
    train_x, train_y = [], []
    for i, s in enumerate(TRAIN_SIGNALS):
        im, tr = generate_spots(500, GRID, SIGMA, s, B, R, seed=500 + i)
        train_x.append(im)
        train_y.append(tr)
    model = MLCentroider(n_estimators=5, random_state=0)
    model.fit(np.concatenate(train_x), np.concatenate(train_y))

    # --- build the gallery on unseen seeds --------------------------------
    fig, axes = plt.subplots(
        N_ROWS, len(GALLERY_SIGNALS), figsize=(3.1 * len(GALLERY_SIGNALS), 3.4 * N_ROWS)
    )
    axes = np.atleast_2d(axes)
    centre = (GRID - 1) / 2.0
    for col, s in enumerate(GALLERY_SIGNALS):
        im, tr = generate_spots(N_ROWS, GRID, SIGMA, s, B, R, seed=7700 + col)
        snr = snr_estimate(s, B, R, GRID)
        pred, std = model.predict(im, return_std=True)
        for row in range(N_ROWS):
            ax = axes[row, col]
            frame = im[row]
            ax.imshow(frame, cmap="viridis", origin="upper", interpolation="nearest")
            x_t, y_t = tr[row]
            x_c, y_c = cog_centroid(frame)
            x_q, y_q = quadcell_centroid(frame, scale=SCALE)
            x_m, y_m = pred[row]
            # image coordinates = centre + offset [px]
            ax.plot(centre + x_t, centre + y_t, "+", color="magenta", ms=16, mew=2.2,
                    label="true")
            ax.plot(centre + x_c, centre + y_c, "o", mfc="none", mec="orange", ms=10,
                    mew=1.8, label="CoG")
            ax.plot(centre + x_q, centre + y_q, "^", mfc="none", mec="deepskyblue", ms=9,
                    mew=1.6, label="quad-cell")
            ax.errorbar(
                centre + x_m, centre + y_m,
                xerr=std[row, 0], yerr=std[row, 1],
                fmt="d", mfc="none", mec="red", ecolor="red", ms=9, mew=1.8,
                capsize=3, label="ML $\\pm$ ensemble std",
            )
            err_c = float(np.hypot(x_c - x_t, y_c - y_t))
            err_m = float(np.hypot(x_m - x_t, y_m - y_t))
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"S={s:.0f} e-, SNR={snr:.1f}", fontsize=10)
            ax.set_xlabel(
                f"|CoG err|={err_c:.2f} px   |ML err|={err_m:.2f} px", fontsize=8
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        f"Synthetic spot gallery ({GRID}x{GRID}, $\\sigma$={SIGMA} px, "
        f"B={B} e-/px, R={R} e- RMS) with centroid estimates",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out = pathlib.Path(__file__).resolve().parents[1] / "screenshots" / "spot_gallery.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    sys.stdout.write(f"saved {out}\n")


if __name__ == "__main__":
    main()
