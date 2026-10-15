"""Where a learned residual's energy goes, and what that says about it.

Run: ``python examples/absorption_subspaces.py``
Writes: ``screenshots/absorption_subspaces.png`` and
``validation/example_absorption_subspaces_output.txt``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
from _bootstrap import Tee, add_src_to_path  # noqa: E402

ROOT = add_src_to_path()

from twincalib import DEFAULT_STRONG_CUT, design, run_benchmark  # noqa: E402

DESIGNS = ("baseline-eclipse", "full-sun", "lumped-nodes", "weak-excitation")
CUTS = (1e-1, DEFAULT_STRONG_CUT, 1e-3)


def main() -> None:
    out = Tee(ROOT / "validation" / "example_absorption_subspaces_output.txt")
    results = {name: run_benchmark(design(name), n_train=4, n_test=3) for name in DESIGNS}

    out("subspace decomposition of the learned residual field on held-out data")
    out("strong: the parameters could have done this and the data would have seen it")
    out("weak  : the parameters could have done this and the data cannot tell  <- absorption")
    out("orth  : no parameter value can reproduce this  <- genuine model-form error")
    for cut in CUTS:
        out("")
        out(f"cut s/s_max = {cut:g}")
        out(f"  {'design':<18}{'strong':>10}{'weak':>10}{'orth':>10}{'n_strong':>10}{'n_weak':>8}")
        for name in DESIGNS:
            s = results[name].splits[cut]
            out(
                f"  {name:<18}{s.strong_fraction:>10.5f}{s.weak_fraction:>10.5f}"
                f"{s.orthogonal_fraction:>10.5f}{s.n_strong:>10d}{s.n_weak:>8d}"
            )

    out("")
    out("absorption ratio against the sound design, at the default cut")
    reference = results["baseline-eclipse"].splits[DEFAULT_STRONG_CUT].weak_fraction
    out(f"  baseline-eclipse weak fraction = {reference:.6f}")
    for name in DESIGNS:
        value = results[name].splits[DEFAULT_STRONG_CUT].weak_fraction
        out(f"  {name:<18} weak fraction {value:.6f}   ratio {value / reference:>8.2f}")

    out("")
    out("parameter drift under the learned discrepancy term")
    out(f"  {'design':<18}{'RMS log err before':>20}{'after':>12}{'verdict':>16}")
    for name in DESIGNS:
        drift = results[name].drift
        verdict = "WORSE" if drift.parameters_got_worse else "better"
        out(
            f"  {name:<18}{drift.rms_relative_error_before:>20.6f}"
            f"{drift.rms_relative_error_after:>12.6f}{verdict:>16}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    ax = axes[0]
    x = np.arange(len(DESIGNS))
    bottom = np.zeros(len(DESIGNS))
    for key, colour, label in (
        ("strong_fraction", "tab:blue", "strong: parameters could, data sees it"),
        ("weak_fraction", "tab:red", "weak: parameters could, data cannot tell"),
        (
            "orthogonal_fraction",
            "tab:green",
            "orthogonal: no parameter value can",
        ),
    ):
        values = np.array(
            [getattr(results[n].splits[DEFAULT_STRONG_CUT], key) for n in DESIGNS]
        )
        ax.bar(x, values, 0.6, bottom=bottom, color=colour, label=label)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(DESIGNS, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("fraction of learned residual energy")
    ax.set_title(f"Where the learned field lives (cut {DEFAULT_STRONG_CUT:g})")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    for name, marker in zip(DESIGNS, ["o", "s", "^", "D"], strict=True):
        ax.semilogy(
            CUTS,
            [max(results[name].splits[c].weak_fraction, 1e-7) for c in CUTS],
            marker + "-",
            label=name,
        )
    ax.set_xscale("log")
    ax.set_xlabel("strong/weak cut, s/s_max")
    ax.set_ylabel("weak (absorption) fraction")
    ax.set_title("The absorption number depends on the cut, so all three are shown")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

    ax = axes[2]
    before = [results[n].drift.rms_relative_error_before for n in DESIGNS]
    after = [results[n].drift.rms_relative_error_after for n in DESIGNS]
    ax.bar(x - 0.2, before, 0.4, label="physics only", color="tab:blue")
    ax.bar(x + 0.2, after, 0.4, label="with learned residual", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(DESIGNS, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("RMS of ln(estimate/truth)")
    ax.set_title("Red taller than blue means the physics got worse")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    target = ROOT / "screenshots" / "absorption_subspaces.png"
    fig.savefig(target, dpi=130)
    plt.close(fig)
    out("")
    out(f"wrote {target.relative_to(ROOT)}")
    out.save()


if __name__ == "__main__":
    main()
