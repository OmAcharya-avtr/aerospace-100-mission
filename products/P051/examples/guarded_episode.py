"""One guarded episode, step by step, with the switching margin underneath.

Writes ``../screenshots/guarded_episode.png``.

Top panel: the tracked angle against the square-wave reference, for the guarded
run, the same performance controller run unguarded on the same disturbance
sequence, and the baseline alone. The shaded bands mark the steps at which the
guard held authority. The unguarded trace is the one that goes outside the
declared limit, and it is the one with the lower tracking cost -- those are the
same fact seen twice.

Middle panel: the body rate, with the declared limit drawn. This is the
constraint the unguarded controller actually breaks; the angle limit is not the
binding one on this scenario.

Bottom panel: the switching-condition margin
``min_j (d_j - c_j^T (A x + B u) - h_W(c_j))``. It is a signed distance in state
units, computed exactly, and it crosses zero precisely where the authority
changes. There is no tuned threshold anywhere in the figure.

Runtime: about 10 s on one contended core.
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
    Mode,
    SimplexGuard,
    account,
    disturbance_sequence,
    reference_controllers,
    reference_plant,
    robust_invariant_set,
    simulate_baseline,
    simulate_guarded,
    simulate_unguarded,
    square_wave_reference,
)

OUT = ROOT / "screenshots" / "guarded_episode.png"
STEPS = 500


def main() -> None:
    plant = reference_plant()
    baseline, performance = reference_controllers(plant)
    result = robust_invariant_set(plant, baseline)
    guard = SimplexGuard(plant, baseline, result.polytope)
    rng = np.random.default_rng(51)
    w = disturbance_sequence(plant, STEPS, rng, "uniform")
    reference = square_wave_reference(0.18, 80, plant.n_states)
    guarded = simulate_guarded(plant, guard, performance, STEPS, reference, w)
    unguarded = simulate_unguarded(plant, performance, STEPS, reference, w)
    base_only = simulate_baseline(plant, baseline, STEPS, reference, w)
    report = account(guarded, unguarded, base_only, result.polytope)

    t = np.arange(STEPS + 1) * plant.dt
    fired = guarded.modes == Mode.BASELINE.value
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 9.0), sharex=True)

    def shade(ax):
        for k in np.flatnonzero(fired):
            ax.axvspan(
                k * plant.dt, (k + 1) * plant.dt, color="tab:red", alpha=0.18, lw=0
            )

    ax = axes[0]
    shade(ax)
    ax.plot(t[:-1], guarded.references[:, 0], color="0.4", lw=1.0, ls=":", label="reference")
    ax.plot(t, guarded.states[:, 0], color="tab:blue", lw=1.2, label="guarded")
    ax.plot(t, unguarded.states[:, 0], color="tab:orange", lw=1.0, label="unguarded")
    ax.plot(t, base_only.states[:, 0], color="tab:green", lw=1.0, label="baseline only")
    limit = plant.state_constraints.b[0]
    ax.axhline(limit, color="k", lw=1.0, ls="--")
    ax.axhline(-limit, color="k", lw=1.0, ls="--", label="declared |theta| limit")
    ax.set_ylabel("theta [rad]")
    ax.legend(loc="upper right", ncol=5, fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(
        f"guarded cost {report.guarded_cost:.3f}, unguarded {report.unguarded_cost:.3f} "
        f"({report.conservatism_cost_ratio:.4f}x), baseline only "
        f"{report.baseline_cost:.3f}; X violations "
        f"{report.guarded_violations} guarded / {report.unguarded_violations} unguarded"
    )

    ax = axes[1]
    shade(ax)
    ax.plot(t, guarded.states[:, 1], color="tab:blue", lw=1.2, label="guarded")
    ax.plot(t, unguarded.states[:, 1], color="tab:orange", lw=1.0, label="unguarded")
    ax.plot(t, base_only.states[:, 1], color="tab:green", lw=1.0, label="baseline only")
    rate_limit = plant.state_constraints.b[1]
    ax.axhline(rate_limit, color="k", lw=1.0, ls="--")
    ax.axhline(-rate_limit, color="k", lw=1.0, ls="--", label="declared |theta_dot| limit")
    ax.set_ylabel("theta_dot [rad/s]")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    shade(ax)
    ax.plot(t[:-1], guarded.margins, color="tab:purple", lw=1.0,
            label="switching-condition margin")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_ylabel("margin [state units]")
    ax.set_xlabel("time [s]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(
        f"the margin is negative at exactly the {int(fired.sum())} shaded steps; "
        f"median {report.margin_median:.4f} overall, "
        f"{report.margin_at_switch_median:.4f} at the firing steps",
        fontsize=9,
    )

    fig.suptitle(
        "simplexguard: one guarded episode against its unguarded and baseline-only "
        "pairs on the same disturbance (research-grade, not flight-qualified)",
        fontsize=10,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  steps                    {STEPS}")
    print(f"  baseline authority steps {int(fired.sum())}")
    print(f"  authority changes        {report.n_switches}")
    print(f"  guarded cost             {report.guarded_cost:.6f}")
    print(f"  unguarded cost           {report.unguarded_cost:.6f}")
    print(f"  baseline-only cost       {report.baseline_cost:.6f}")
    print(f"  conservatism cost ratio  {report.conservatism_cost_ratio:.6f}")
    print(f"  guarded X violations     {report.guarded_violations}")
    print(f"  unguarded X violations   {report.unguarded_violations}")
    print(f"  unguarded worst residual {report.unguarded_worst_residual:.9e}")


if __name__ == "__main__":
    main()
