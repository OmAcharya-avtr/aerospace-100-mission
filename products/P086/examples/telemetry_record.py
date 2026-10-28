"""The generated record, with every injected anomaly class marked on its channel."""

from __future__ import annotations

import numpy as np

from _common import CHANNEL_COLOURS, VERDICT_COLOURS, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from telemgap import dataset  # noqa: E402


def main() -> int:
    style()
    generated = dataset.generate(samples=1800, seed=515, n_clock_resets=0)
    record = generated.record
    figure, axes = plt.subplots(4, 1, figsize=(9.5, 8.0), sharex=True)
    for column, (channel, axis) in enumerate(zip(record.schema.channels, axes, strict=True)):
        values = record.values[:, column]
        axis.plot(
            record.timestamps,
            values,
            color=CHANNEL_COLOURS[channel.name],
            linewidth=0.7,
            label="received",
        )
        axis.axhline(channel.low, color="#999999", linewidth=0.6, linestyle="--")
        axis.axhline(channel.high, color="#999999", linewidth=0.6, linestyle="--")
        for code, name in enumerate(dataset.VALUE_CLASSES):
            if code == 0:
                continue
            marked = generated.value_truth[:, column] == code
            if not marked.any():
                continue
            axis.scatter(
                record.timestamps[marked],
                values[marked],
                s=14,
                color=VERDICT_COLOURS[name],
                zorder=5,
                label=f"true {name} ({int(marked.sum())})",
            )
        span = channel.high - channel.low
        axis.set_ylim(channel.low - 0.25 * span, channel.high + 0.25 * span)
        axis.set_ylabel(f"{channel.name}\n[{channel.unit}]")
        axis.legend(loc="upper left", ncol=4, fontsize=7)
    axes[-1].set_xlabel("timestamp [s]")
    counts = generated.truth_counts()
    axes[0].set_title(
        "telemgap.dataset: one generated record with its ground truth\n"
        f"{counts['frames']} frames received, {counts['missing_frames']} deleted, "
        f"{counts['corrupt']} corrupt words ({counts['corrupt_in_range']} of them inside "
        f"their declared range), {counts['misparsed']} misparsed words\n"
        "dashed lines are the declared valid ranges"
    )
    figure.subplots_adjust(hspace=0.22)
    path = save(figure, "telemetry_record")
    print(f"written: {path}")
    print(f"ground truth: {counts}")
    assert np.isfinite(record.values).all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
