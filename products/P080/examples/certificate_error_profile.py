"""Per-sample error against the declared bound, with the argmax sample marked.

The figure a certificate is about: not a mean, but every sample's error, the
running maximum over the whole array, and the one sample that came closest to
breaking the promise.

Left: a 400-sample window around the worst sample, so the individual errors
are legible. Middle: the running maximum of the absolute error over all 4096
samples against the bound, which is the summary a mean would hide -- it can
only rise, and the question is whether it ever touches the red line. Right:
the error distribution, which fills the admissible band rather than hugging
zero, because a uniform quantiser at the bound-implied step is supposed to.
"""

from __future__ import annotations

from _common import BOUND, LINEAR, PREVIOUS, SHADE, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from boundcompress import dataset  # noqa: E402
from boundcompress.codec import decode, encode  # noqa: E402

style()

WINDOW = 400


def main() -> None:
    signal = dataset.generate("battery_voltage", samples=4096, seed=0)
    bound = 5e-4
    result = encode(signal.values, "abs", bound, predictor="linear")
    decoded = decode(result.blob)
    error = signal.values - decoded
    certificate = result.certificate
    worst = certificate.argmax_index

    start = max(0, min(worst - WINDOW // 2, signal.samples - WINDOW))
    stop = start + WINDOW
    window = np.arange(start, stop)

    figure = plt.figure(figsize=(12.0, 3.6))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.4, 1.4, 1.0], wspace=0.46)
    left = figure.add_subplot(grid[0, 0])
    middle = figure.add_subplot(grid[0, 1])
    right = figure.add_subplot(grid[0, 2])

    left.axhspan(-bound, bound, color=SHADE, zorder=0)
    left.vlines(window, 0.0, error[start:stop], linewidth=0.6, color=LINEAR, zorder=2)
    left.axhline(bound, color=BOUND, linewidth=1.2, zorder=3)
    left.axhline(-bound, color=BOUND, linewidth=1.2, zorder=3)
    left.plot([worst], [error[worst]], "o", color=BOUND, markersize=5, zorder=4)
    left.set_xlim(start, stop)
    left.set_xlabel("sample index")
    left.set_ylabel(f"reconstruction error [{signal.unit}]")
    left.set_title(f"{WINDOW} samples around the worst one", fontsize=9)

    running = np.maximum.accumulate(np.abs(error))
    middle.plot(running, linewidth=1.0, color=PREVIOUS)
    middle.axhline(bound, color=BOUND, linewidth=1.2)
    middle.plot([worst], [running[worst]], "o", color=BOUND, markersize=5)
    middle.annotate(
        f"worst sample {worst}\n"
        f"max |error| {certificate.max_absolute_error:.4e}\n"
        f"margin {certificate.margin:.4e}\n"
        f"{certificate.violations} of {certificate.samples} outside",
        xy=(worst, running[worst]),
        xytext=(0.06, 0.18),
        textcoords="axes fraction",
        fontsize=7.5,
        arrowprops={"arrowstyle": "->", "color": BOUND, "linewidth": 0.8},
    )
    middle.set_xlim(0, signal.samples)
    # Zoomed to the top 2 per cent of the admissible band: a uniform quantiser
    # at the bound-implied step fills the band within the first few samples, so
    # on a full-height axis the running maximum and the bound are one line and
    # the margin is invisible. The margin is the interesting quantity.
    middle.set_ylim(bound * 0.98, bound * 1.004)
    middle.set_xlabel("sample index")
    middle.set_ylabel(f"running max |error| [{signal.unit}]")
    middle.set_title(
        "running maximum against the bound\n(y zoomed to the top 2% of the band)",
        fontsize=9,
    )

    right.hist(error, bins=51, color=LINEAR, edgecolor="none", orientation="horizontal")
    right.axhline(bound, color=BOUND, linewidth=1.2)
    right.axhline(-bound, color=BOUND, linewidth=1.2)
    right.set_ylabel(f"error [{signal.unit}]")
    right.set_xlabel("samples")
    right.set_title("error distribution", fontsize=9)

    figure.suptitle(
        f"battery_voltage, declared absolute bound {bound:.0e} {signal.unit}: "
        f"{certificate.violations} of {certificate.samples} samples outside",
        fontsize=10.5,
    )
    figure.subplots_adjust(top=0.82)
    print(save(figure, "certificate_error_profile"))
    print(
        f"max |error| {certificate.max_absolute_error:.6e} <= bound {bound:.6e}; "
        f"violations {certificate.violations}; "
        f"independent check {float(np.max(np.abs(error))):.6e}; "
        f"worst sample {worst}"
    )


if __name__ == "__main__":
    main()
