#!/usr/bin/env python3
"""Where the analytic sum-of-stages model is right, and where it is not.

Draws an injected three-stage pipeline, plots each stage's latency
distribution and the end-to-end total, and overlays the analytic prediction.
The point of the figure: equations (1) and (2) of latencynet.analytic land the
mean and the standard deviation exactly, and the Fenton-Wilkinson map from
those two moments to a tail quantile does not. The gap at p99.9 is the
systematic error the analytic predictor carries.

Writes ../screenshots/stage_decomposition.png. Runtime about 4 s.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from latencynet.analytic import SumOfStagesModel
from latencynet.pipeline import make_lognormal_pipeline, sample_stage_latencies
from latencynet.tails import quantile
from latencynet.units import s_to_us

N = 400_000
SEED = 20260407
OUT = Path(__file__).resolve().parents[1] / "screenshots" / "stage_decomposition.png"


def us(values: np.ndarray) -> np.ndarray:
    """Array seconds to microseconds. latencynet.units is scalar-only by design."""
    return np.asarray(values, dtype=float) * 1.0e6



def main() -> None:
    spec = make_lognormal_pipeline(
        (60e-6, 180e-6, 30e-6),
        (12e-6, 40e-6, 6e-6),
        names=("preprocess", "inference", "postprocess"),
    )
    trace = sample_stage_latencies(spec, N, SEED)
    total = trace.sum(axis=1)
    pred = SumOfStagesModel().predict(
        spec.stage_mean_s, spec.stage_std_s, (0.5, 0.99, 0.999)
    )

    fig, (ax_stages, ax_total) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for i, stage in enumerate(spec.stages):
        ax_stages.hist(
            us(trace[:, i]),
            bins=160,
            histtype="step",
            density=True,
            label=f"{stage.name}: mean {s_to_us(stage.mean_s):.0f} us, cv {stage.cv:.2f}",
        )
    ax_stages.set_xlabel("stage latency (us)")
    ax_stages.set_ylabel("density (1/us)")
    ax_stages.set_title(f"injected stage latencies, n = {N}, seed {SEED}")
    ax_stages.legend(fontsize=8)
    ax_stages.grid(alpha=0.3)

    ax_total.hist(us(total), bins=200, histtype="step", density=True, color="black",
                  label="end-to-end, injected")
    for p, colour in ((0.5, "tab:green"), (0.99, "tab:orange"), (0.999, "tab:red")):
        emp = quantile(total, p, "linear")
        ana = pred.quantile_s[p]
        ax_total.axvline(s_to_us(emp), color=colour, lw=1.4,
                         label=f"p{p * 100:g} injected {s_to_us(emp):.1f} us")
        ax_total.axvline(s_to_us(ana), color=colour, lw=1.4, ls="--",
                         label=f"p{p * 100:g} analytic {s_to_us(ana):.1f} us "
                               f"({(ana - emp) / emp * 100:+.2f} %)")
    ax_total.axvline(s_to_us(pred.mean_s), color="tab:blue", lw=1.0, ls=":",
                     label=f"analytic mean {s_to_us(pred.mean_s):.1f} us (exact)")
    ax_total.set_xlabel("end-to-end latency (us)")
    ax_total.set_ylabel("density (1/us)")
    ax_total.set_title("end-to-end total: moments exact, tail map approximate")
    ax_total.legend(fontsize=7.5, loc="upper right")
    ax_total.grid(alpha=0.3)
    ax_total.set_xlim(s_to_us(float(total.min())), s_to_us(quantile(total, 0.9999, "linear")))

    fig.suptitle(
        "latencynet: injected stage latencies and the analytic sum-of-stages prediction",
        fontsize=11,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)

    print(f"injected mean {s_to_us(spec.injected_mean_s()):.4f} us   "
          f"sampled mean {s_to_us(float(total.mean())):.4f} us")
    print(f"injected sd   {s_to_us(spec.injected_std_s()):.4f} us   "
          f"sampled sd   {s_to_us(float(total.std(ddof=1))):.4f} us")
    for p in (0.5, 0.99, 0.999):
        emp = quantile(total, p, "linear")
        ana = pred.quantile_s[p]
        print(
            f"p{p * 100:<6g} injected {s_to_us(emp):9.4f} us   analytic {s_to_us(ana):9.4f} us   "
            f"relative error {(ana - emp) / emp:+.5f}"
        )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
