"""The entropy reference against the realised rate, and why it is not a floor.

Left: realised rate against the two entropy figures. The symbol-alphabet
entropy, which the certificate carries, collapses on an escape-heavy stream
because every large residual is one escape symbol. The residual-value entropy
has no escape and is the figure comparable with a byte count.

Right: the payload of a coder with memory against the residual-value entropy.
Points below the diagonal are a real coder going below the "floor", which is
the thing a reader has to see once to stop treating the entropy as a bound.
"""

from __future__ import annotations

from _common import BASELINE, LINEAR, LOSSLESS, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from boundcompress import baselines, dataset  # noqa: E402
from boundcompress.codec import encode  # noqa: E402
from boundcompress.entropy import value_entropy_bits  # noqa: E402
from boundcompress.predictors import get_predictor  # noqa: E402
from boundcompress.quantise import make_spec, quantise  # noqa: E402
from boundcompress.ratedistortion import geometric_bounds  # noqa: E402

style()


def main() -> None:
    signals = dataset.generate_all(samples=4096)
    symbol_bits: list[float] = []
    value_bits: list[float] = []
    realised: list[float] = []
    memory_bits: list[float] = []
    labels: list[str] = []

    for name, signal in signals.items():
        for bound in geometric_bounds(signal.values, count=5):
            result = encode(signal.values, "abs", bound, predictor="previous")
            spec = make_spec(signal.values, "abs", bound)
            symbols, _, _ = quantise(signal.values, spec)
            residuals = get_predictor("previous").residuals(symbols)
            best_payload = min(
                baselines.measure(
                    method, values=signal.values, symbols=symbols, residuals=residuals
                ).payload_bytes
                for method in ("lzma_residual", "bz2_residual")
            )
            symbol_bits.append(result.certificate.residual_entropy_bits)
            value_bits.append(value_entropy_bits(residuals))
            realised.append(result.certificate.bits_per_sample)
            memory_bits.append(8.0 * best_payload / residuals.size)
            labels.append(name)

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    limit = max(max(realised), max(value_bits), max(memory_bits)) * 1.08

    axes[0].plot([0, limit], [0, limit], color=LOSSLESS, linewidth=1.0, linestyle=":")
    axes[0].plot(symbol_bits, realised, "o", markersize=4, color=BASELINE,
                 label="symbol-alphabet entropy (certificate field)")
    axes[0].plot(value_bits, realised, "s", markersize=4, color=LINEAR,
                 label="residual-value entropy")
    axes[0].set_xlabel("entropy reference [bits/symbol]")
    axes[0].set_ylabel("realised rate [bits/sample, actual bytes]")
    axes[0].set_xlim(0, limit)
    axes[0].set_ylim(0, limit)
    axes[0].set_title("Two entropy figures, one realised rate")
    axes[0].legend(fontsize=7)

    axes[1].plot([0, limit], [0, limit], color=LOSSLESS, linewidth=1.0, linestyle=":")
    axes[1].plot(value_bits, memory_bits, "^", markersize=4, color=BASELINE)
    below = int(np.count_nonzero(np.asarray(memory_bits) < np.asarray(value_bits)))
    axes[1].set_xlabel("residual-value entropy [bits/symbol]")
    axes[1].set_ylabel("best lzma/bz2 payload [bits/symbol]")
    axes[1].set_xlim(0, limit)
    axes[1].set_ylim(0, limit)
    axes[1].set_title(
        f"A coder with memory goes below the reference\n"
        f"in {below} of {len(value_bits)} cells (points under the diagonal)"
    )

    figure.subplots_adjust(wspace=0.28)
    print(save(figure, "entropy_versus_realised_rate"))
    print(f"cells: {len(value_bits)}; memory coder below the value entropy: {below}")


if __name__ == "__main__":
    main()
