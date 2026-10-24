"""What the guarantee cost: raw, lossless, and bound-certified, on one axis.

The question this product exists for is not "how small can it get" but "what
did I pay to be able to prove the bound". This figure answers it in bytes: the
raw float64 array, the best lossless coding of the same array at zero
distortion, and the bound-certified container at three declared bounds.
"""

from __future__ import annotations

from _common import BASELINE, LINEAR, LOSSLESS, PREVIOUS, QUADRATIC, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from boundcompress import baselines, dataset  # noqa: E402
from boundcompress.codec import encode  # noqa: E402
from boundcompress.predictors import CLASSICAL_NAMES, get_predictor  # noqa: E402
from boundcompress.quantise import make_spec, quantise  # noqa: E402

style()

FRACTIONS = (1e-3, 1e-4, 1e-5)
BAR_COLOURS = (LOSSLESS, BASELINE, PREVIOUS, QUADRATIC, LINEAR)


def main() -> None:
    signals = dataset.generate_all(samples=4096)
    names = list(signals)
    positions = np.arange(len(names))
    width = 0.16

    figure, axis = plt.subplots(figsize=(10.0, 4.4))
    series: list[tuple[str, list[float]]] = []

    raw = [8.0 * signals[name].samples for name in names]
    series.append(("raw float64 (no coding)", raw))

    lossless = []
    for name in names:
        signal = signals[name]
        spec = make_spec(signal.values, "abs", 1e-3 * float(
            signal.values.max() - signal.values.min()))
        symbols, _, _ = quantise(signal.values, spec)
        residuals = get_predictor("linear").residuals(symbols)
        lossless.append(
            float(min(
                baselines.measure(
                    method, values=signal.values, symbols=symbols, residuals=residuals
                ).total_bytes
                for method in baselines.LOSSLESS_METHODS
            ))
        )
    series.append(("best lossless, zero distortion", lossless))

    for fraction in FRACTIONS:
        heights = []
        for name in names:
            signal = signals[name]
            span = float(signal.values.max() - signal.values.min())
            best = min(
                encode(signal.values, "abs", fraction * span, predictor=predictor)
                .certificate.encoded_bytes
                for predictor in CLASSICAL_NAMES
            )
            heights.append(float(best))
        series.append((f"certified at {fraction:.0e} of range", heights))

    for index, ((label, heights), colour) in enumerate(
        zip(series, BAR_COLOURS, strict=True)
    ):
        axis.bar(positions + (index - 2.0) * width, heights, width=width,
                 color=colour, label=label)

    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
    axis.set_yscale("log")
    axis.set_ylabel("bytes (actual, log scale)")
    axis.set_title(
        "The cost of the guarantee: raw, lossless at zero distortion, and "
        "bound-certified at three declared bounds"
    )
    axis.legend(fontsize=7.5, ncol=2)

    print(save(figure, "guarantee_cost"))
    for name, raw_bytes, lossless_bytes in zip(names, raw, lossless, strict=True):
        print(f"  {name:<18s} raw {raw_bytes:>8.0f}  lossless {lossless_bytes:>8.0f}")


if __name__ == "__main__":
    main()
