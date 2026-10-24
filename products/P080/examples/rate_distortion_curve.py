"""Rate against declared bound, every predictor and the best available baseline.

Rate is actual encoded bytes. The distortion is identical across every curve
at a given bound, by construction, so the vertical distance between two curves
is the whole comparison.
"""

from __future__ import annotations

from _common import (
    BASELINE,
    LOSSLESS,
    PREDICTOR_COLOURS,
    bootstrap,
    fit_models,
    save,
    style,
)

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from boundcompress import baselines, dataset  # noqa: E402
from boundcompress.codec import encode  # noqa: E402
from boundcompress.predictors import CLASSICAL_NAMES, get_predictor  # noqa: E402
from boundcompress.quantise import make_spec, quantise  # noqa: E402
from boundcompress.ratedistortion import geometric_bounds  # noqa: E402

style()

SIGNALS = ("battery_voltage", "wheel_rate", "gyro_rate", "packet_counter")


def main() -> None:
    models = fit_models()
    signals = dataset.generate_all(samples=4096)
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))

    for axis, name in zip(axes.ravel(), SIGNALS, strict=True):
        signal = signals[name]
        bounds = geometric_bounds(signal.values, count=7)
        for predictor in CLASSICAL_NAMES:
            rates = [
                encode(signal.values, "abs", bound, predictor=predictor)
                .certificate.bits_per_sample
                for bound in bounds
            ]
            axis.plot(bounds, rates, marker="o", markersize=3, linewidth=1.1,
                      color=PREDICTOR_COLOURS[predictor], label=predictor,
                      linestyle="--" if predictor == "previous" else "-")
        for label, model in models.items():
            rates = [
                encode(signal.values, "abs", bound, predictor="learned", model=model)
                .certificate.bits_per_sample
                for bound in bounds
            ]
            axis.plot(bounds, rates, "-s", markersize=3, linewidth=1.1,
                      color=PREDICTOR_COLOURS[f"learned_{label}"],
                      label=f"learned_{label}")

        best_baseline = []
        for bound in bounds:
            spec = make_spec(signal.values, "abs", bound)
            symbols, _, _ = quantise(signal.values, spec)
            residuals = get_predictor("linear").residuals(symbols)
            measurements = baselines.measure_all(
                values=signal.values, symbols=symbols, residuals=residuals
            )
            best_baseline.append(
                min(
                    measurements[method].bits_per_sample
                    for method in baselines.LOSSY_METHODS
                )
            )
        axis.plot(bounds, best_baseline, "--^", markersize=3, linewidth=1.1,
                  color=BASELINE, label="best of zlib/lzma/bz2")

        spec = make_spec(signal.values, "abs", bounds[0])
        symbols, _, _ = quantise(signal.values, spec)
        residuals = get_predictor("linear").residuals(symbols)
        lossless = min(
            baselines.measure(
                method, values=signal.values, symbols=symbols, residuals=residuals
            ).bits_per_sample
            for method in baselines.LOSSLESS_METHODS
        )
        axis.axhline(lossless, color=LOSSLESS, linewidth=1.0, linestyle=":",
                     label="best lossless (zero distortion)")

        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_title(f"{name}  [{signal.unit}]")
        axis.set_xlabel(f"declared absolute bound [{signal.unit}]  (tighter to the right)")
        axis.set_ylabel("rate [bits/sample, actual bytes]")
        axis.tick_params(labelsize=7.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, fontsize=8, ncol=4, loc="lower center",
                  bbox_to_anchor=(0.5, -0.02))
    figure.suptitle(
        "Rate against declared bound. Distortion is identical across curves at "
        "each bound.",
        fontsize=11,
    )
    figure.subplots_adjust(hspace=0.38, wspace=0.26, top=0.92, bottom=0.14)
    print(save(figure, "rate_distortion_curve"))


if __name__ == "__main__":
    main()
