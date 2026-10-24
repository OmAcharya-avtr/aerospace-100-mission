"""The eight synthetic telemetry channels every number in this repository uses."""

from __future__ import annotations

from _common import bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from boundcompress import dataset  # noqa: E402

style()


def main() -> None:
    signals = dataset.generate_all(samples=4096)
    figure, axes = plt.subplots(4, 2, figsize=(9.5, 8.0))
    for axis, (name, signal) in zip(axes.ravel(), signals.items(), strict=True):
        axis.plot(signal.values, linewidth=0.7, color="#1f6fb4")
        axis.set_title(f"{name}  [{signal.unit}]")
        axis.set_xlim(0, signal.samples)
        axis.tick_params(labelsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("sample index")
    figure.suptitle(
        "Synthetic telemetry channels (seeded; no flight data is used anywhere)",
        fontsize=11,
    )
    figure.subplots_adjust(hspace=0.55, wspace=0.22, top=0.93)
    print(save(figure, "telemetry_shapes"))


if __name__ == "__main__":
    main()
