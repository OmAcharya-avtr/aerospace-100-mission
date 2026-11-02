"""Clock drift and reset recovery against synthetic truth, and what they cost.

What to notice: the drift is recovered to a fraction of a part per million at
every jitter level that matters, and the recovery degrades only when the
timestamp jitter approaches a hundredth of a period. The right panel is the
reason the clock check is part of the timing verdict: at a two per cent rate
error the set of iterations that APPEAR to have missed their deadline differs
from the true set by several misses per run, and the correction removes it.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from _common import (  # noqa: E402
    CERTIFIED,
    CONSEQUENTIAL,
    NEUTRAL,
    REFERENCE,
    TOLERANCE,
    bootstrap,
    save,
    style,
)

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from hilverdict.clock import detect_resets, drift_corrected_log, estimate_clock  # noqa: E402
from hilverdict.synthetic import CAMPAIGNS, ClockSpec, generate_clock, generate_run  # noqa: E402
from hilverdict.timeline import miss_events  # noqa: E402

STEPS = 200
PERIOD = 0.010
DRIFTS = (-20000.0, -5000.0, -500.0, -50.0, 50.0, 500.0, 5000.0, 20000.0)
JITTERS = (2.0e-7, 1.0e-6, 1.0e-5, 1.0e-4)
SEEDS = tuple(range(8400, 8409))


def main() -> int:
    style()
    host = np.arange(STEPS + 1, dtype=float) * PERIOD
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    colours = (CERTIFIED, REFERENCE, NEUTRAL, CONSEQUENTIAL)
    for jitter, colour in zip(JITTERS, colours, strict=True):
        errors = []
        for drift in DRIFTS:
            spec = ClockSpec(
                drift_ppm=drift, jitter_s=jitter, tick_s=1.0e-7,
                reset_fractions=(), name="sweep",
            )
            device, _resets = generate_clock(
                spec, host, np.random.default_rng(int(abs(drift)) + 7)
            )
            estimate = estimate_clock(host, device, 100.0)
            errors.append(abs(estimate.drift_ppm - drift))
        axes[0].plot(
            DRIFTS, np.maximum(errors, 1e-4), "o-", color=colour,
            label=f"jitter {jitter * 1e6:g} us",
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("injected drift [ppm]")
    axes[0].set_ylabel("absolute recovery error [ppm]")
    axes[0].set_title("drift recovery against known truth")
    axes[0].legend(loc="upper center", fontsize=7.5, ncol=2)

    spurious = []
    true_counts = []
    apparent = []
    corrected = []
    for drift in DRIFTS:
        truth_total = 0
        apparent_total = 0
        corrected_total = 0
        wrong = 0
        for position, seed in enumerate(SEEDS):
            profile, amplitude = CAMPAIGNS[position % len(CAMPAIGNS)]
            spec = ClockSpec(
                drift_ppm=drift, jitter_s=2.0e-7, tick_s=1.0e-7,
                reset_fractions=(), name="consequence",
            )
            drifted = generate_run(
                seed=seed, profile=profile, amplitude_deg=amplitude, steps=STEPS,
                clock=spec,
            )
            truth = generate_run(
                seed=seed, profile=profile, amplitude_deg=amplitude, steps=STEPS
            )
            estimate = estimate_clock(
                drifted.host_time_s, drifted.device_time_s, 100.0
            )
            fixed = drift_corrected_log(drifted, estimate.drift_ppm)
            truth_indices = {event.index for event in miss_events(truth)}
            apparent_indices = {event.index for event in miss_events(drifted)}
            corrected_indices = {event.index for event in miss_events(fixed)}
            truth_total += len(truth_indices)
            apparent_total += len(apparent_indices)
            corrected_total += len(corrected_indices)
            wrong += len(apparent_indices ^ truth_indices)
        true_counts.append(truth_total)
        apparent.append(apparent_total)
        corrected.append(corrected_total)
        spurious.append(wrong)
    axes[1].bar(
        np.arange(len(DRIFTS)) - 0.2, spurious, width=0.4, color=CONSEQUENTIAL,
        label="misses wrongly gained or lost, uncorrected",
    )
    axes[1].bar(
        np.arange(len(DRIFTS)) + 0.2,
        [abs(c - t) for c, t in zip(corrected, true_counts, strict=True)],
        width=0.4, color=CERTIFIED, label="residual after correction",
    )
    axes[1].set_xticks(np.arange(len(DRIFTS)))
    axes[1].set_xticklabels([f"{d:g}" for d in DRIFTS], fontsize=7.5)
    axes[1].set_xlabel("injected drift [ppm]")
    axes[1].set_ylabel(f"misses, summed over {len(SEEDS)} runs")
    axes[1].set_title("an uncorrected clock changes which iterations appear to miss")
    axes[1].legend(loc="upper center", fontsize=7.5)

    reset_device = host.copy()
    reset_device[int(0.45 * host.shape[0]):] -= 0.9
    detected = detect_resets(host, reset_device)
    axes[1].text(
        0.02, 0.02,
        f"reset at sample {int(0.45 * host.shape[0])} detected as {list(detected)}",
        transform=axes[1].transAxes, fontsize=7.5, color=TOLERANCE,
    )

    figure.suptitle(
        "hilverdict: host-to-device clock drift and its consequence", fontsize=10
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path = save(figure, "clock_drift")
    print(f"written: {path}")
    print(f"true misses per drift level: {true_counts}")
    print(f"apparent misses            : {apparent}")
    print(f"after correction           : {corrected}")
    print(f"wrongly gained or lost     : {spurious}")
    print(f"reset detected at {list(detected)}")
    print(f"np version {np.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
