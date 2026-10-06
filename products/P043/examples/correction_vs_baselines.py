"""The learned rate correction against the closed-form inversions, by regime.

Writes ``../screenshots/correction_vs_baselines.png``.

What to notice. Left: median relative error against the true loading n tau. Below
n tau = 0.5 everything works and the curves sit on the counting-noise floor (grey
band). Past n tau = 1 **every** estimator collapses: the paralyzable forward map
has turned over, the lower-branch inverse returns the wrong root, and the learned
model does not recover it either. That is a real limit of the measurement, not of
the method. Right: the same data split by afterpulse probability. Where
afterpulsing is negligible the closed forms win; where it is present the learned
model wins, because neither textbook inversion can accept an afterpulse
probability at all.

This example uses ``validation/rate_corrector.joblib`` if it exists, and otherwise
trains a smaller corrector in-line so that it runs standalone.

Runtime: about 45 s on two shared cores.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from photoncount.correction import (  # noqa: E402
    RateCorrector,
    composed_baseline,
    interval_coverage,
    matched_baseline,
    rate_error_metrics,
)
from photoncount.dataset import generate_dataset  # noqa: E402

MODEL = Path(__file__).resolve().parents[1] / "validation" / "rate_corrector.joblib"
OUT = Path(__file__).resolve().parents[1] / "screenshots" / "correction_vs_baselines.png"
TEST_ROWS = 1500
TEST_SEED = 515151


def _median_by_band(values, errors, edges):
    centres, medians = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (values >= lo) & (values < hi) & np.isfinite(errors)
        if mask.sum() >= 8:
            centres.append(float(np.sqrt(lo * hi)))
            medians.append(float(np.median(errors[mask])))
    return np.asarray(centres), np.asarray(medians)


def main() -> int:
    if MODEL.exists():
        corrector = RateCorrector.load(MODEL)
        provenance = f"loaded validation/{MODEL.name}"
    else:
        corrector = RateCorrector(n_estimators=120).fit(generate_dataset(2500, 20261006))
        provenance = "trained in-line (2500 rows, 120 trees)"

    test = generate_dataset(TEST_ROWS, TEST_SEED)
    learned = corrector.predict_rate(test.features, test.dead_time_s)
    estimates = {
        "matched closed form (D2/D5)": (
            matched_baseline(test.observed_rate_hz, test.dead_time_s, test.is_paralyzable),
            "#1b4965",
        ),
        "composed closed form (A2+D2/D5)": (
            composed_baseline(
                test.observed_rate_hz,
                test.dead_time_s,
                test.is_paralyzable,
                test.afterpulse_probability,
            ),
            "#5fa8d3",
        ),
        "learned correction": (learned["median"], "#c1121f"),
    }

    floor = float(np.median(1.0 / np.sqrt(test.count_mean * 25)))
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13.0, 4.9))

    edges = np.logspace(np.log10(test.true_x.min()), np.log10(test.true_x.max()), 13)
    for name, (est, colour) in estimates.items():
        rel = np.abs((est - test.true_rate_hz) / test.true_rate_hz)
        centres, medians = _median_by_band(test.true_x, rel, edges)
        ax_left.loglog(centres, medians, "o-", color=colour, lw=1.8, ms=4, label=name)
    ax_left.axhspan(0.0, 0.6745 * floor * 1.6, color="grey", alpha=0.18,
                    label="counting-noise floor")
    ax_left.axvline(1.0, color="black", ls="--", lw=1.1)
    ax_left.annotate("paralyzable maximum\n$n\\tau=1$", (1.0, 0.45),
                     textcoords="offset points", xytext=(6, 0), fontsize=8)
    ax_left.set_xlabel(r"true loading $n\tau$")
    ax_left.set_ylabel("median relative error in recovered rate")
    ax_left.set_title("Everything fails past the paralyzable maximum")
    ax_left.legend(fontsize=8, loc="upper left")
    ax_left.grid(alpha=0.3, which="both")

    bands = [("p <= 0.02", test.afterpulse_probability <= 0.02),
             ("0.02 < p <= 0.08", (test.afterpulse_probability > 0.02)
              & (test.afterpulse_probability <= 0.08)),
             ("p > 0.08", test.afterpulse_probability > 0.08)]
    width = 0.26
    positions = np.arange(len(bands))
    for offset, (name, (est, colour)) in enumerate(estimates.items()):
        values = []
        for _, mask in bands:
            sub = mask & (test.true_x < 1.0)
            values.append(
                rate_error_metrics(est[sub], test.true_rate_hz[sub])["median_abs_rel_error"]
            )
        ax_right.bar(positions + (offset - 1) * width, values, width, color=colour, label=name)
        for pos, value in zip(positions + (offset - 1) * width, values, strict=True):
            ax_right.annotate(f"{value:.4f}", (pos, value), ha="center",
                              textcoords="offset points", xytext=(0, 3), fontsize=7)
    ax_right.axhline(0.6745 * floor, color="grey", ls="--", lw=1.2,
                     label="counting-noise floor")
    ax_right.set_xticks(positions)
    ax_right.set_xticklabels([name for name, _ in bands])
    ax_right.set_ylabel("median relative error")
    ax_right.set_title(r"Below $n\tau=1$, split by afterpulse probability")
    ax_right.legend(fontsize=8)
    ax_right.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)

    cov = interval_coverage(learned["lower"], learned["upper"], test.true_rate_hz)
    print("correction_vs_baselines.py")
    print(f"   wrote screenshots/{OUT.name}")
    print(f"   corrector: {provenance}")
    print(f"   test rows: {len(test)} (seed {TEST_SEED}); counting-noise floor "
          f"{0.6745 * floor:.5f} median relative")
    print(f"\n   {'band':20s} {'estimator':32s} {'median rel err':>15}")
    for name, mask in bands:
        for est_name, (est, _) in estimates.items():
            sub = mask & (test.true_x < 1.0)
            value = rate_error_metrics(est[sub], test.true_rate_hz[sub])[
                "median_abs_rel_error"
            ]
            print(f"   {name:20s} {est_name:32s} {value:15.5f}")
    print(f"\n   learned 5-95 interval coverage {cov['coverage']:.4f} "
          f"(nominal {cov['nominal']:.2f}), median relative width "
          f"{cov['median_relative_width']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
