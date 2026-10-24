"""The learned predictor's uncertainty output, measured against the realised error.

An ensemble dispersion is a dispersion measure and not a confidence interval
(Wager, Hastie & Efron 2014, JMLR 15). Whether it is *informative* is a
measurement, and the measurement here is a Spearman rank correlation against
the realised absolute residual. On three of the four channels shown it is
close to zero, which is reported rather than smoothed.
"""

from __future__ import annotations

from _common import LEARNED_MLP, bootstrap, fit_models, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from boundcompress import dataset  # noqa: E402
from boundcompress.learned import WINDOW  # noqa: E402
from boundcompress.quantise import make_spec, quantise  # noqa: E402

style()

SIGNALS = ("battery_voltage", "wheel_rate", "gyro_rate", "white_noise")


def main() -> None:
    model = fit_models()["mlp"]
    signals = dataset.generate_all(samples=4096)
    figure, axes = plt.subplots(1, 4, figsize=(12.0, 3.2))

    for axis, name in zip(axes, SIGNALS, strict=True):
        signal = signals[name]
        span = float(signal.values.max() - signal.values.min())
        spec = make_spec(signal.values, "abs", 1e-4 * span)
        symbols, _, _ = quantise(signal.values, spec)
        predictions, dispersion = model.predict_batch(symbols)
        residuals = np.abs(symbols[WINDOW:] - predictions).astype(np.float64)
        correlation = float(spearmanr(dispersion, residuals).statistic)
        axis.plot(dispersion, residuals, ".", markersize=1.6, color=LEARNED_MLP,
                  alpha=0.5)
        axis.set_xlabel("ensemble dispersion")
        axis.set_title(f"{name}\nSpearman {correlation:+.4f}", fontsize=9)
        axis.tick_params(labelsize=7)
        print(f"  {name:<18s} spearman {correlation:+.4f}")
    axes[0].set_ylabel("|realised residual| [symbols]")

    figure.suptitle(
        "learned_mlp ensemble dispersion against realised residual. A dispersion "
        "measure, not a confidence interval.",
        fontsize=10,
    )
    figure.subplots_adjust(wspace=0.3, top=0.78)
    print(save(figure, "learned_uncertainty"))


if __name__ == "__main__":
    main()
