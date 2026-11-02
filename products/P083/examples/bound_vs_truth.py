"""The certificate against the re-simulated truth, and what makes it loose.

What to notice: every point lies below the diagonal, which is the soundness
statement; the points nearest the diagonal are the short overruns, where the
bound is within about a factor of 1.4 of the truth; and the points furthest
below it are the multi-period holds, where the exponential of equation (9)
overstates a divergence that in fact grows roughly linearly in the overrun.
The right panel shows that the looseness is driven by the declared metric's
anisotropy and is therefore a consequence of declaring a tight incidence
tolerance, not an accident of the derivation.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

from _common import (  # noqa: E402
    CERTIFIED,
    CONSEQUENTIAL,
    NEUTRAL,
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
from hilverdict.plant import DivergenceMetric, LoopModel  # noqa: E402
from hilverdict.resimulate import resimulate_run  # noqa: E402
from hilverdict.synthetic import CAMPAIGNS, generate_run  # noqa: E402

STEPS = 200
SEEDS = tuple(range(8100, 8127))


def gather(model: LoopModel, seeds: tuple[int, ...]):
    bounds = []
    divergences = []
    periods = []
    consequential = []
    for position, seed in enumerate(seeds):
        profile, amplitude = CAMPAIGNS[position % len(CAMPAIGNS)]
        log = generate_run(
            seed=seed, profile=profile, amplitude_deg=amplitude, steps=STEPS,
            model=model,
        )
        certificate = certify_run(log, 1.0)
        if certificate.refusals:
            continue
        results = resimulate_run(log, 1.0)
        for item, result in zip(certificate.certificates, results, strict=True):
            bounds.append(item.bound)
            divergences.append(result.peak_divergence)
            periods.append(item.event.overrun_s / model.sampling.period_s)
            consequential.append(result.consequential)
    return (
        np.array(bounds),
        np.array(divergences),
        np.array(periods),
        np.array(consequential),
    )


def main() -> int:
    style()
    model = LoopModel()
    bounds, divergences, periods, consequential = gather(model, SEEDS)
    usable = (bounds > 0.0) & (divergences > 0.0)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    bands = (
        (periods < 1.0, CERTIFIED, "o", "overrun below one period"),
        ((periods >= 1.0) & (periods < 2.0), UNDECIDED, "s", "one to two periods"),
        (periods >= 2.0, CONSEQUENTIAL, "^", "two periods or more"),
    )
    for mask, colour, marker, label in bands:
        selected = mask & usable
        axes[0].scatter(
            bounds[selected], divergences[selected], s=14, c=colour, marker=marker,
            alpha=0.75, label=label,
        )
    limits = [
        float(np.min(bounds[usable])) * 0.5,
        float(np.max(bounds[usable])) * 2.0,
    ]
    axes[0].plot(limits, limits, color=TRUTH, lw=1.1,
                 label="bound = truth (soundness edge)")
    axes[0].axhline(1.0, color=TOLERANCE, lw=1.0, ls="--")
    axes[0].axvline(1.0, color=TOLERANCE, lw=1.0, ls="--")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlim(*limits)
    axes[0].set_ylim(*limits)
    axes[0].set_xlabel("certificate bound, eq (13)")
    axes[0].set_ylabel("re-simulated peak divergence")
    axes[0].set_title(
        f"{int(np.count_nonzero(usable))} misses, "
        f"{int(np.count_nonzero(divergences[usable] > bounds[usable]))} above the line"
    )
    axes[0].legend(loc="upper left", fontsize=7.5)

    anisotropies = []
    medians = []
    worst = []
    for rate_degrees in (0.05, 0.2, 0.5, 1.0, 2.0):
        metric = DivergenceMetric(
            alpha_tolerance_rad=math.radians(0.01),
            rate_tolerance_rad_s=math.radians(rate_degrees),
        )
        variant = LoopModel(metric=metric)
        variant_bounds, variant_divergences, _periods, _flags = gather(
            variant, SEEDS[:8]
        )
        mask = variant_bounds > 0.0
        ratio = variant_divergences[mask] / variant_bounds[mask]
        anisotropies.append(metric.anisotropy)
        medians.append(float(np.median(ratio)))
        worst.append(float(np.max(ratio)))
    axes[1].plot(anisotropies, medians, "o-", color=CERTIFIED,
                 label="median truth / bound")
    axes[1].plot(anisotropies, worst, "s--", color=NEUTRAL,
                 label="worst truth / bound")
    axes[1].axhline(1.0, color=TOLERANCE, lw=1.1, ls="--")
    axes[1].text(anisotropies[0], 1.03, "soundness edge", color=TOLERANCE, fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_ylim(0.0, 1.15)
    axes[1].set_xlabel("declared metric anisotropy")
    axes[1].set_ylabel("ratio of truth to bound")
    axes[1].set_title("the looseness is a cost of the declared metric")
    axes[1].legend(loc="lower left", fontsize=8)

    figure.suptitle(
        "hilverdict: the bound against the re-simulated reference", fontsize=10
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path = save(figure, "bound_vs_truth")
    print(f"written: {path}")
    ratio = divergences[usable] / bounds[usable]
    print(f"misses compared {int(np.count_nonzero(usable))}")
    print(f"violations {int(np.count_nonzero(ratio > 1.0))}")
    print(f"worst ratio {float(np.max(ratio)):.6f}, median {float(np.median(ratio)):.6f}")
    print("anisotropy sweep medians " + ", ".join(f"{m:.6f}" for m in medians))
    print(f"np version {np.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
