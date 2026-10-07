"""The certified envelope, and where the guard fires on it.

Writes ``../screenshots/invariant_set.png``.

Left panel: the declared state-constraint set ``X``, the computed robust
invariant set ``S`` of the baseline controller, and the eroded set ``S (-) W``
that the one-step-ahead nominal state must lie in for the performance input to be
admitted. The gap between ``S`` and ``S (-) W`` is the whole of the disturbance
allowance, and at this disturbance bound it is thin -- that thinness is a
measured property of the declaration, not a drawing choice.

Right panel: a guarded trajectory over 600 steps on the shipped reference
scenario, with the steps at which the guard handed control to the baseline
marked. Notice that they cluster on the lower-right and upper-left boundary of
``S``, where the aggressive performance controller is accelerating toward the
rate limit; and notice that ``S`` touches the angle facets of ``X`` exactly, so
in the angle direction there is no slack between the certificate and the
constraint at all.

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
    disturbance_sequence,
    reference_controllers,
    reference_plant,
    robust_invariant_set,
    simulate_guarded,
    square_wave_reference,
)

OUT = ROOT / "screenshots" / "invariant_set.png"


def polygon(ax, polytope, **kwargs):
    verts = polytope.vertices_2d()
    closed = np.vstack([verts, verts[0]])
    ax.plot(closed[:, 0], closed[:, 1], **kwargs)


def main() -> None:
    plant = reference_plant()
    baseline, performance = reference_controllers(plant)
    result = robust_invariant_set(plant, baseline)
    guard = SimplexGuard(plant, baseline, result.polytope)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))

    ax = axes[0]
    polygon(ax, plant.state_constraints, color="0.25", lw=1.8, label="declared X")
    polygon(ax, result.polytope, color="tab:blue", lw=1.8, label="robust invariant S")
    polygon(
        ax,
        guard.eroded_set,
        color="tab:orange",
        lw=1.4,
        ls="--",
        label="eroded S - W (switching test)",
    )
    ax.set_xlabel("theta [rad]")
    ax.set_ylabel("theta_dot [rad/s]")
    ax.set_title(
        f"area(S)/area(X) = {result.polytope.area_2d() / plant.state_constraints.area_2d():.6f}"
        f",  {result.n_halfspaces} facets,  {result.iterations} iterations"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    # The erosion is 1.5e-4 rad in angle and 6.0e-3 rad/s in rate, which is
    # invisible at the scale of X. The inset is where it becomes a picture.
    inset = ax.inset_axes((0.08, 0.20, 0.38, 0.30))
    polygon(inset, plant.state_constraints, color="0.25", lw=1.4)
    polygon(inset, result.polytope, color="tab:blue", lw=1.4)
    polygon(inset, guard.eroded_set, color="tab:orange", lw=1.2, ls="--")
    inset.set_xlim(0.055, 0.105)
    inset.set_ylim(-0.503, -0.487)
    inset.set_title("zoom: the erosion", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.grid(alpha=0.3)
    ax.indicate_inset_zoom(inset, edgecolor="0.4")

    ax = axes[1]
    rng = np.random.default_rng(51)
    w = disturbance_sequence(plant, 600, rng, "uniform")
    reference = square_wave_reference(0.18, 80, plant.n_states)
    episode = simulate_guarded(plant, guard, performance, 600, reference, w)
    polygon(ax, plant.state_constraints, color="0.25", lw=1.8, label="declared X")
    polygon(ax, result.polytope, color="tab:blue", lw=1.8, label="robust invariant S")
    polygon(ax, guard.eroded_set, color="tab:orange", lw=1.0, ls="--", label="S - W")
    ax.plot(
        episode.states[:, 0],
        episode.states[:, 1],
        color="0.45",
        lw=0.8,
        label="guarded trajectory",
    )
    fired = episode.modes == Mode.BASELINE.value
    ax.plot(
        episode.states[:-1][fired, 0],
        episode.states[:-1][fired, 1],
        "o",
        ms=3.5,
        color="tab:red",
        label=f"baseline authority ({int(fired.sum())} of {episode.n_steps} steps)",
    )
    ax.set_xlabel("theta [rad]")
    ax.set_ylabel("theta_dot [rad/s]")
    ax.set_title(
        f"600 guarded steps, {int(np.sum(episode.modes[1:] != episode.modes[:-1]))} "
        f"authority changes, {episode.constraint_violations().size} violations"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "simplexguard: the certified envelope and the exact switching boundary "
        "(research-grade, not flight-qualified)",
        fontsize=10,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  area(X)      {plant.state_constraints.area_2d():.9f}")
    print(f"  area(S)      {result.polytope.area_2d():.9f}")
    print(f"  area(S-W)    {guard.eroded_set.area_2d():.9f}")
    print(f"  facets of S  {result.n_halfspaces}")
    print(f"  baseline authority steps {int(fired.sum())} of {episode.n_steps}")
    print(f"  violations of X          {episode.constraint_violations().size}")


if __name__ == "__main__":
    main()
