"""The assurance accounting: dwell times, and what hysteresis buys and costs.

Writes ``../screenshots/accounting_tradeoff.png``.

Top left: the dwell-time distribution of each authority. The mean is not the
story -- the median baseline dwell is 2 steps and the median performance dwell
is 1 step, so most of the switching is the guard chattering on the boundary of
the eroded set rather than taking over for a sustained interval. A reader who
only saw the switch rate would not know that.

Top right: the minimum-baseline-dwell trade. Holding the baseline longer than
the condition demands is always safe, because the invariant set is invariant
under the baseline for every admissible disturbance, so the violation count
stays at zero all along the curve. It reduces the switch rate and it costs
tracking performance, and both are measured. Note the plateau at a dwell of 2,
where the authority fraction rises for no reduction in the switch rate at all:
the worst setting on the curve.

Bottom left: the conservatism cost as the reference amplitude rises, that is, as
the performance controller is asked to push harder against the certified
envelope. The guarded cost sits between the unguarded cost and the baseline-only
cost at every amplitude, and the gap it recovers is plotted on the right axis.

Bottom right: the unguarded run's violations at the same amplitudes, which is
what the conservatism is being paid for. Below an amplitude of about 0.12 rad
the performance controller never leaves the envelope and the guard is pure
overhead with nothing to show for it; the figure includes that region on
purpose.

Runtime: about 25 s on one contended core.
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

OUT = ROOT / "screenshots" / "accounting_tradeoff.png"
STEPS = 2000
SEED = 51


def main() -> None:
    plant = reference_plant()
    baseline, performance = reference_controllers(plant)
    invariant = robust_invariant_set(plant, baseline).polytope

    def run(dwell: int = 1, amplitude: float = 0.18):
        guard = SimplexGuard(plant, baseline, invariant, min_baseline_dwell=dwell)
        rng = np.random.default_rng(SEED)
        w = disturbance_sequence(plant, STEPS, rng, "uniform")
        reference = square_wave_reference(amplitude, 80, plant.n_states)
        g = simulate_guarded(plant, guard, performance, STEPS, reference, w)
        u = simulate_unguarded(plant, performance, STEPS, reference, w)
        b = simulate_baseline(plant, baseline, STEPS, reference, w)
        return account(g, u, b, invariant)

    main_report = run()
    dwells = [1, 2, 4, 8, 16]
    dwell_reports = [run(dwell=d) for d in dwells]
    amplitudes = [0.06, 0.10, 0.14, 0.18, 0.22, 0.26]
    amp_reports = [run(amplitude=a) for a in amplitudes]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))

    ax = axes[0, 0]
    base_lengths = np.array(main_report.baseline_dwell_lengths)
    perf_lengths = np.array(main_report.performance_dwell_lengths)
    bins = np.arange(0.5, 21.5, 1.0)
    ax.hist(
        np.clip(base_lengths, 1, 20), bins=bins, color="tab:red", alpha=0.75,
        label=f"baseline (n={base_lengths.size}, median "
              f"{main_report.baseline_dwell.median:.1f})",
    )
    ax.hist(
        np.clip(perf_lengths, 1, 20), bins=bins, color="tab:blue", alpha=0.55,
        label=f"performance (n={perf_lengths.size}, median "
              f"{main_report.performance_dwell.median:.1f}, max "
              f"{main_report.performance_dwell.maximum} clipped at 20)",
    )
    ax.set_yscale("log")
    ax.set_xlabel("dwell length [steps]")
    ax.set_ylabel("intervals (log)")
    ax.set_title(
        f"dwell distribution, {STEPS} steps: "
        f"{main_report.performance_dwell.fraction_of_length_one:.3f} of performance "
        f"intervals last one step",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 1]
    ax.plot(
        dwells,
        [r.switches_per_1000_steps for r in dwell_reports],
        marker="o",
        color="tab:red",
        label="authority changes per 1000 steps",
    )
    ax.set_xlabel("minimum baseline dwell [steps]")
    ax.set_ylabel("authority changes per 1000 steps", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")
    ax.grid(alpha=0.3)
    twin = ax.twinx()
    twin.plot(
        dwells,
        [r.conservatism_cost_ratio for r in dwell_reports],
        marker="s",
        color="tab:blue",
        label="guarded / unguarded cost",
    )
    twin.plot(
        dwells,
        [r.baseline_fraction for r in dwell_reports],
        marker="^",
        color="tab:green",
        label="fraction under baseline",
    )
    twin.set_ylabel("cost ratio and authority fraction")
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=8)
    ax.set_title(
        "hysteresis: safe at every setting (0 violations throughout), "
        "cheaper in switches, dearer in tracking",
        fontsize=10,
    )

    ax = axes[1, 0]
    ax.plot(amplitudes, [r.unguarded_cost for r in amp_reports], marker="o",
            color="tab:orange", label="unguarded")
    ax.plot(amplitudes, [r.guarded_cost for r in amp_reports], marker="s",
            color="tab:blue", label="guarded")
    ax.plot(amplitudes, [r.baseline_cost for r in amp_reports], marker="^",
            color="tab:green", label="baseline only")
    ax.set_xlabel("reference amplitude [rad]")
    ax.set_ylabel(f"tracking cost over {STEPS} steps")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    twin = ax.twinx()
    twin.plot(
        amplitudes,
        [r.performance_gap_recovered for r in amp_reports],
        marker="d",
        color="0.35",
        ls="--",
        label="fraction of the baseline-to-unguarded gap recovered",
    )
    twin.set_ylim(0.0, 1.05)
    twin.set_ylabel("gap recovered")
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="center left", fontsize=8)
    ax.set_title("conservatism cost against how hard the controller pushes", fontsize=10)

    ax = axes[1, 1]
    ax.bar(
        np.array(amplitudes) - 0.004,
        [r.unguarded_violations for r in amp_reports],
        width=0.008,
        color="tab:orange",
        label="unguarded X violations",
    )
    ax.bar(
        np.array(amplitudes) + 0.004,
        [r.guarded_violations for r in amp_reports],
        width=0.008,
        color="tab:blue",
        label="guarded X violations (all zero)",
    )
    for a, r in zip(amplitudes, amp_reports, strict=True):
        ax.text(a, r.unguarded_violations + 4, str(r.unguarded_violations),
                ha="center", fontsize=7)
    ax.set_xlabel("reference amplitude [rad]")
    ax.set_ylabel(f"constraint violations over {STEPS} steps")
    ax.set_title(
        "what the conservatism buys; below 0.12 rad it buys nothing because the "
        "unguarded run is already safe",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "simplexguard: the assurance accounting (research-grade, not flight-qualified)",
        fontsize=11,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(
        f"  dwell medians: baseline {main_report.baseline_dwell.median:.1f}, "
        f"performance {main_report.performance_dwell.median:.1f}"
    )
    for d, r in zip(dwells, dwell_reports, strict=True):
        print(
            f"  dwell={d:<3d} switches/1k={r.switches_per_1000_steps:7.2f} "
            f"cost ratio={r.conservatism_cost_ratio:.6f} "
            f"base frac={r.baseline_fraction:.6f} X-viol={r.guarded_violations}"
        )
    for a, r in zip(amplitudes, amp_reports, strict=True):
        print(
            f"  amp={a:.2f} guarded cost={r.guarded_cost:10.4f} "
            f"unguarded={r.unguarded_cost:10.4f} baseline={r.baseline_cost:10.4f} "
            f"gap recovered={r.performance_gap_recovered:.6f} "
            f"unguarded X-viol={r.unguarded_violations}"
        )


if __name__ == "__main__":
    main()
