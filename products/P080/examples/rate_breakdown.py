"""Where the bytes went. Nothing is omitted from the rate, including the header.

The fixed header is 38 bytes and the frequency table is shipped in full. On a
4096-sample array that is a few per cent; on a 64-sample array it is most of
the container. Both regimes are shown, because "the header is negligible" is a
claim and not an observation.
"""

from __future__ import annotations

from _common import BASELINE, LINEAR, LOSSLESS, PREVIOUS, QUADRATIC, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from boundcompress import dataset  # noqa: E402
from boundcompress.codec import encode  # noqa: E402
from boundcompress.container import block_sizes, unpack  # noqa: E402

style()

BLOCKS = ("fixed_header", "warmup", "frequency_table", "escapes", "payload")
COLOURS = (LOSSLESS, PREVIOUS, BASELINE, QUADRATIC, LINEAR)
LENGTHS = (64, 256, 1024, 4096)


def main() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    signals = dataset.generate_all(samples=4096)
    names = list(signals)
    bottoms = np.zeros(len(names))
    for block, colour in zip(BLOCKS, COLOURS, strict=True):
        heights = []
        for name in names:
            signal = signals[name]
            span = float(signal.values.max() - signal.values.min())
            result = encode(signal.values, "abs", 1e-4 * span, predictor="previous")
            heights.append(block_sizes(unpack(result.blob))[block])
        axes[0].bar(names, heights, bottom=bottoms, color=colour, label=block)
        bottoms = bottoms + np.asarray(heights, dtype=np.float64)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    axes[0].set_ylabel("bytes")
    axes[0].set_title("Container composition, bound = 1e-4 of dynamic range")
    axes[0].legend(fontsize=7)

    overhead_pct = []
    for samples in LENGTHS:
        signal = dataset.generate("battery_voltage", samples=samples, seed=0)
        result = encode(signal.values, "abs", 1e-3, predictor="linear")
        sizes = block_sizes(unpack(result.blob))
        overhead = sizes["fixed_header"] + sizes["warmup"] + sizes["frequency_table"]
        overhead_pct.append(100.0 * overhead / result.certificate.encoded_bytes)
    bars = axes[1].bar([str(length) for length in LENGTHS], overhead_pct,
                       color=BASELINE, width=0.55)
    for bar, value in zip(bars, overhead_pct, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, value + 1.2,
                     f"{value:.1f}%", ha="center", fontsize=8)
    axes[1].set_xlabel("array length [samples]")
    axes[1].set_ylabel("header + table, % of container")
    axes[1].set_ylim(0, max(overhead_pct) * 1.25)
    axes[1].set_title("Fixed cost against array length, battery_voltage at 1e-3 V")

    figure.subplots_adjust(wspace=0.3)
    print(save(figure, "rate_breakdown"))
    for length, value in zip(LENGTHS, overhead_pct, strict=True):
        print(f"  {length:5d} samples: header+table = {value:.2f}% of the container")


if __name__ == "__main__":
    main()
