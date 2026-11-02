"""Three misses from one run, traced, with the certificate's bound over each.

What to notice: the divergence peaks within about ten sample instants of the
overrun and then decays with the closed loop, which is why the horizon factor
of equation (12) saturates so early. The dismissed miss never approaches the
tolerance line and its bound does not either -- that is the proof. The
consequential miss crosses the tolerance within a few samples.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from _common import (  # noqa: E402
    CERTIFIED,
    CONSEQUENTIAL,
    TOLERANCE,
    TRUTH,
    UNDECIDED,
    bootstrap,
    save,
    style,
)

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from hilverdict.certificate import certify_run  # noqa: E402
from hilverdict.resimulate import resimulate_run, simulate  # noqa: E402
from hilverdict.synthetic import generate_run  # noqa: E402

STEPS = 200
SEED = 902


def main() -> int:
    style()
    log = generate_run(seed=SEED, profile="loaded", amplitude_deg=8.0, steps=STEPS)
    model = log.model
    certificate = certify_run(log, 1.0)
    results = {r.event.index: r for r in resimulate_run(log, 1.0)}
    items = {item.event.index: item for item in certificate.certificates}

    chosen: list[tuple[str, int, str]] = []
    for index, item in items.items():
        if item.dismissed and item.bound > 0.0 and not chosen:
            chosen.append(("dismissed by eq (13)", index, CERTIFIED))
    for index, item in items.items():
        if not item.dismissed and not results[index].consequential:
            chosen.append(("undecided, then inconsequential", index, UNDECIDED))
            break
    for index in items:
        if results[index].consequential:
            chosen.append(("consequential", index, CONSEQUENTIAL))
            break

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for label, index, colour in chosen:
        event = items[index].event
        perturbed = simulate(
            model,
            log.command_rad,
            {index: event.overrun_s},
            start_index=index,
            start_state=log.states[index],
            prior_command_rad=event.stale_command_rad,
        )
        difference = perturbed.states - np.asarray(log.states)[index:]
        norms = model.metric.norm(difference)
        offsets = np.arange(norms.shape[0])
        axes[0].plot(
            offsets, np.maximum(norms, 1e-12), color=colour,
            label=f"{label} (iter {index}, tau "
                  f"{event.overrun_s * 1e3:.2f} ms)",
        )
        axes[0].axhline(items[index].bound, color=colour, lw=0.8, ls=":")
    axes[0].axhline(1.0, color=TOLERANCE, lw=1.2, ls="--")
    axes[0].text(2.0, 1.25, "declared tolerance", color=TOLERANCE, fontsize=8)
    axes[0].set_yscale("log")
    axes[0].set_xlim(0, 60)
    axes[0].set_ylim(1e-8, 1e3)
    axes[0].set_xlabel("sample instants after the overrun")
    axes[0].set_ylabel("divergence in the declared metric")
    axes[0].set_title("solid: re-simulated. dotted: the bound for that miss")
    axes[0].legend(loc="lower left", fontsize=7.5)

    states = np.asarray(log.states)
    worst = max(
        (result for result in results.values() if result.consequential),
        key=lambda result: result.peak_divergence,
        default=None,
    )
    if worst is not None:
        index = worst.event.index
        perturbed = simulate(
            model,
            log.command_rad,
            {index: worst.event.overrun_s},
            start_index=index,
            start_state=states[index],
            prior_command_rad=worst.event.stale_command_rad,
        )
        window = slice(max(index - 10, 0), min(index + 60, STEPS + 1))
        axes[1].plot(
            np.arange(window.start, window.stop),
            np.rad2deg(states[window, 0]),
            color=TRUTH,
            label="miss-free reference",
        )
        offsets = np.arange(index, index + perturbed.states.shape[0])
        keep = offsets < window.stop
        axes[1].plot(
            offsets[keep],
            np.rad2deg(perturbed.states[keep, 0]),
            color=CONSEQUENTIAL,
            ls="--",
            label=f"with the miss at iter {index}",
        )
        axes[1].axvline(index, color=TOLERANCE, lw=0.8, ls=":")
        axes[1].set_xlabel("iteration")
        axes[1].set_ylabel("incidence [deg]")
        axes[1].set_title(
            f"the consequential miss, peak divergence {worst.peak_divergence:.2f}"
        )
        axes[1].legend(loc="best", fontsize=8)

    figure.suptitle(
        f"hilverdict: divergence traces from run {log.label}", fontsize=10
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path = save(figure, "divergence_traces")
    print(f"written: {path}")
    for label, index, _colour in chosen:
        print(f"{label}: iteration {index}, bound {items[index].bound:.6f}, "
              f"re-simulated {results[index].peak_divergence:.6f}")
    print(f"np version {np.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
