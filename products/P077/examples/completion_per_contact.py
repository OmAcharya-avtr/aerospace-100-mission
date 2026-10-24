"""The probability the file is down by the end of pass k, which is a vector.

Produces ``screenshots/completion_per_contact.png``: cumulative completion
probability against contact index for the three modes, and for three file
sizes, over the same declared contact plan. A single expected delivery time
cannot express this, which is why the package reports it.
"""

from __future__ import annotations

from _common import MODE_COLOURS, bootstrap, save, style

bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from relfiledelivery import (  # noqa: E402
    ALL_MODES,
    FileTransferSpec,
    LinkConditions,
    OutageModel,
    run_ensemble,
    uniform_windows,
)

RUNS = 1200
WINDOWS = 8

SPECS = {
    "60 PDUs": FileTransferSpec(60 * 512, 512, 12),
    "180 PDUs": FileTransferSpec(180 * 512, 512, 12),
    "420 PDUs": FileTransferSpec(420 * 512, 512, 12),
}

LINK = LinkConditions(
    5.0e5,
    1.0e4,
    0.8,
    0.06,
    outage=OutageModel(mean_up_s=4.0, mean_down_s=2.0),
    windows=uniform_windows(WINDOWS, 2.5, 6.0),
)


def main() -> str:
    figure, axes = plt.subplots(1, len(SPECS), figsize=(10.2, 3.9), sharey=True)
    contacts = np.arange(WINDOWS)
    for axis, (name, spec) in zip(axes, SPECS.items(), strict=True):
        for mode in ALL_MODES:
            distribution = run_ensemble(spec, LINK, mode, RUNS, 8080)
            cells = distribution.completion_by_contact
            values = [cells.get(index, 0.0) for index in contacts]
            axis.step(
                contacts,
                values,
                where="post",
                color=MODE_COLOURS[mode.value],
                linewidth=1.6,
                label=f"{mode.value} ({values[-1]:.3f})",
            )
            axis.plot(contacts, values, ".", color=MODE_COLOURS[mode.value], markersize=4)
        axis.set_xlabel("contact index")
        axis.set_title(f"{name}, {RUNS} runs", fontsize=9)
        axis.set_ylim(-0.03, 1.03)
        axis.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("fraction of all runs complete")
    figure.suptitle(
        f"Cumulative completion by end of pass k: {WINDOWS} contacts of 2.5 s every 6 s, "
        "6% loss, outage duty 0.667",
        fontsize=10,
    )
    figure.subplots_adjust(wspace=0.12, top=0.84)
    return save(figure, "completion_per_contact")


if __name__ == "__main__":
    print(main())
