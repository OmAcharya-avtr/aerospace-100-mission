"""Singular spectra of the five shipped designs, and the combinations they miss.

Run: ``python examples/identifiability_spectrum.py``
Writes: ``screenshots/identifiability_spectrum.png`` and
``validation/example_identifiability_spectrum_output.txt``.
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

from twincalib import (  # noqa: E402
    DESIGN_NAMES,
    analyse,
    build_inputs,
    calibration_model,
    design,
    log_sensitivity,
    worst_subsets,
)


def main() -> None:
    out = Tee(ROOT / "validation" / "example_identifiability_spectrum_output.txt")
    model = calibration_model()
    spectra: dict[str, np.ndarray] = {}
    for name in DESIGN_NAMES:
        d = design(name)
        sens = log_sensitivity(
            model,
            d.theta_true,
            build_inputs(d),
            d.initial_state,
            d.channels,
            d.sigma_k,
            d.sample_stride,
        )
        report = analyse(sens)
        spectra[name] = report.singular_values / report.singular_values[0]
        out.rule(name)
        out(report.verdict())
        out(f"condition number   : {report.condition_number:.6g}")
        out(f"collinearity index : {report.collinearity:.6g}")
        out(
            "worst parameter pair: "
            + ", ".join(
                f"{'+'.join(names)} = {value:.4g}"
                for names, value in worst_subsets(sens, 2, 1)
            )
        )
        for combo in report.weak_combinations:
            out(f"unconstrained combination (s/s_max = {combo.ratio:.4e}):")
            out(f"  {combo.as_text()}")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax = axes[0]
    markers = ["o", "s", "^", "D", "v"]
    for (name, ratios), marker in zip(spectra.items(), markers, strict=True):
        ax.semilogy(
            np.arange(ratios.size),
            np.maximum(ratios, 1e-14),
            marker=marker,
            label=name,
            linewidth=1.4,
            markersize=6,
        )
    ax.axhline(1e-8, color="black", linestyle="--", linewidth=1.0)
    ax.text(0.05, 1.4e-8, "rank cut, 1e-8", fontsize=8)
    ax.axhline(1e-3, color="grey", linestyle=":", linewidth=1.0)
    ax.text(0.05, 1.4e-3, "near-null reporting cut, 1e-3", fontsize=8)
    ax.set_xlabel("singular direction, best determined first")
    ax.set_ylabel("singular value / largest singular value")
    ax.set_title("What each design can see")
    ax.set_ylim(1e-13, 3.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    names = list(DESIGN_NAMES)
    width = 0.15
    model = calibration_model()
    for j, pname in enumerate(model.parameter_names):
        values = []
        for name in names:
            d = design(name)
            sens = log_sensitivity(
                model,
                d.theta_true,
                build_inputs(d),
                d.initial_state,
                d.channels,
                d.sigma_k,
                d.sample_stride,
            )
            sd = float(analyse(sens).relative_sd[j])
            values.append(min(sd, 100.0) if np.isfinite(sd) else 100.0)
        ax.bar(np.arange(len(names)) + j * width, values, width, label=pname)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(names)) + 2.5 * width)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.text(-0.4, 1.2, "100 per cent relative uncertainty", fontsize=8)
    ax.set_ylabel("Cramer-Rao relative standard deviation")
    ax.set_title("Bars at the top are clipped at 100 (infinite or worse)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7, ncols=3)

    fig.tight_layout()
    target = ROOT / "screenshots" / "identifiability_spectrum.png"
    target.parent.mkdir(exist_ok=True)
    fig.savefig(target, dpi=130)
    plt.close(fig)
    out("")
    out(f"wrote {target.relative_to(ROOT)}")
    out.save()


if __name__ == "__main__":
    main()
