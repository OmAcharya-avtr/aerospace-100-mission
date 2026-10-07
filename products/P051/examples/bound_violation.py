"""What happens when the declared disturbance bound is wrong.

Writes ``../screenshots/bound_violation.png``.

Left panel: the rate of constraint violations and of invariant-set exits per
recorded state, against the factor by which the realised disturbance exceeds the
declared bound. The guard is not rebuilt: it keeps using the declared set. At a
factor of 1 both rates are exactly zero, which is the guarantee; above 1 nothing
is guaranteed and the curve is a measurement. The two samplers differ because
the worst-case vertex sampler realises the extreme of every coordinate at every
step, and the uniform sampler almost never does.

Right panel: the fraction of the unguarded run's violations that the guard
eliminates, against the same factor. It is 1.0 exactly at the declared bound and
falls steadily above it. A reader who takes one number from this package should
take this curve: the guard's worth is a function of how right the declared bound
is, and nothing in the architecture protects against a bound that is too small.

The dashed vertical line is the declared bound. The first failure appears at a
factor of 1.1 under the vertex sampler, a 10 % underestimate, because the
invariant set is MAXIMAL and therefore holds its invariance with equality on
some facet -- there is no designed margin to spend.

Runtime: about 30 s on one contended core.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
from _bootstrap import add_src_to_path  # noqa: E402

ROOT = add_src_to_path()

from simplexguard import (  # noqa: E402
    SimplexGuard,
    bound_violation_sweep,
    disturbance_sequence,
    reference_controllers,
    reference_plant,
    robust_invariant_set,
    simulate_unguarded,
    square_wave_reference,
)

OUT = ROOT / "screenshots" / "bound_violation.png"
SCALES = (1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
SEED = 51055
EPISODES = 5
STEPS = 1200


def main() -> None:
    plant = reference_plant()
    baseline, performance = reference_controllers(plant)
    invariant = robust_invariant_set(plant, baseline).polytope
    guard = SimplexGuard(plant, baseline, invariant)

    sweeps = {
        mode: bound_violation_sweep(
            plant,
            guard,
            performance,
            invariant,
            scales=SCALES,
            n_episodes=EPISODES,
            n_steps=STEPS,
            seed=SEED,
            disturbance_mode=mode,
        )
        for mode in ("uniform", "vertex")
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))

    ax = axes[0]
    styles = {"uniform": ("tab:blue", "o"), "vertex": ("tab:red", "s")}
    for mode, sweep in sweeps.items():
        colour, marker = styles[mode]
        scales = [p.scale for p in sweep.points]
        ax.plot(
            scales,
            [max(p.violation_rate, 1e-7) for p in sweep.points],
            marker=marker,
            color=colour,
            lw=1.4,
            label=f"{mode}: X violations",
        )
        ax.plot(
            scales,
            [max(p.exit_rate, 1e-7) for p in sweep.points],
            marker=marker,
            color=colour,
            lw=1.0,
            ls="--",
            mfc="none",
            label=f"{mode}: S exits (certificate lost)",
        )
    ax.axvline(1.0, color="k", lw=1.0, ls=":")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("realised disturbance / declared bound")
    ax.set_ylabel("failures per recorded state (log; 1e-7 means zero)")
    ax.set_title(
        f"{EPISODES} episodes x {STEPS} steps per point; zero at the declared bound"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    guarded_by_scale = {p.scale: p.constraint_violations for p in sweeps["vertex"].points}
    fractions, shown = [], []
    for scale in SCALES:
        total = 0
        for episode in range(EPISODES):
            rng = np.random.default_rng(SEED + 7919 * episode)
            amp = float(rng.uniform(0.75, 1.25)) * 0.18
            w = disturbance_sequence(plant, STEPS, rng, "vertex", scale=scale)
            ep = simulate_unguarded(
                plant, performance, STEPS, square_wave_reference(amp, 80, 2), w
            )
            total += int(ep.constraint_violations().size)
        if total:
            fractions.append((total - guarded_by_scale[scale]) / total)
            shown.append(scale)
    ax.plot(shown, fractions, marker="s", color="tab:red", lw=1.6)
    ax.axvline(1.0, color="k", lw=1.0, ls=":")
    ax.axhline(1.0, color="0.6", lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("realised disturbance / declared bound")
    ax.set_ylabel("fraction of unguarded violations eliminated")
    ax.set_title("worst-case vertex sampler: what the guard is still worth")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        "simplexguard: the guarantee holds at the declared bound and degrades above it "
        "(research-grade, not flight-qualified)",
        fontsize=10,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}")
    for mode, sweep in sweeps.items():
        print(
            f"  {mode:<8s} first X violation at "
            f"{sweep.threshold_text(sweep.first_scale_with_constraint_violation)}, "
            f"first S exit at "
            f"{sweep.threshold_text(sweep.first_scale_with_invariant_exit)}"
        )
    for scale, frac in zip(shown, fractions, strict=True):
        print(f"  rho={scale:6.2f}  violations eliminated {frac:.6f}")


if __name__ == "__main__":
    main()
