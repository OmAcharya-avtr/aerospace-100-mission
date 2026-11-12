"""The learned predictor against the RC baseline, on held-out duty cycles.

What to notice: each point is one held-out row -- one run, one warm-up window
length. The horizontal axis is the measured steady-state die temperature and the
vertical axis is what each method predicted from the first N seconds alone. The
diagonal is a perfect prediction. The shaded band is the run-to-run bootstrap
interval of the measured steady state, which is the yardstick: a method whose
error is inside that band is not distinguishable from one whose error is zero,
and a learned margin inside it is not an improvement.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import _common  # noqa: E402

ROOT = _common.bootstrap()

import numpy as np  # noqa: E402

from thermalbudget.baselines import sustained_truth  # noqa: E402
from thermalbudget.predictor import (  # noqa: E402
    build_dataset,
    compare_against_rc,
    leave_one_duty_out,
)


def main() -> int:
    plt = _common.pyplot()
    campaign, is_measured = _common.load_for_example("no_throttle")
    colour = _common.MEASURED_COLOUR if is_measured else _common.SYNTHETIC_COLOUR

    dataset = build_dataset(
        campaign.soaks(),
        equilibrium_window_s=_common.WINDOW_S,
        threshold_c_per_s=_common.THRESHOLD_C_PER_S,
    )
    truth = sustained_truth(
        campaign.soaks(1.0),
        window_s=_common.WINDOW_S,
        threshold_c_per_s=_common.THRESHOLD_C_PER_S,
        seed=_common.SEED,
        n_resamples=_common.RESAMPLES,
    )
    result = leave_one_duty_out(dataset, campaign.soaks(), kind="ridge", seed=_common.SEED)
    verdict = compare_against_rc(result, truth.die_c)

    figure, (left, right) = plt.subplots(1, 2, figsize=(9.8, 4.8), width_ratios=[1.1, 1.0])

    truths, learned, sigmas, rc_values = [], [], [], []
    for split in result.splits:
        for _label, _window, truth_c, prediction, sigma, rc in split.rows:
            truths.append(truth_c)
            learned.append(prediction)
            sigmas.append(sigma)
            rc_values.append(rc)
    truths = np.asarray(truths)
    learned = np.asarray(learned)
    sigmas = np.asarray(sigmas)
    rc_values = np.asarray(rc_values)

    span = [float(min(truths.min(), np.nanmin(rc_values), learned.min())) - 1.0,
            float(max(truths.max(), np.nanmax(rc_values), learned.max())) + 1.0]
    left.plot(span, span, color="#999999", lw=1.0, ls="--", label="perfect prediction")
    left.fill_between(
        span,
        [value - truth.die_c.half_width for value in span],
        [value + truth.die_c.half_width for value in span],
        color="#8fb8d8",
        alpha=0.3,
        label=f"run-to-run half-width {truth.die_c.half_width:.3f} degC",
    )
    left.errorbar(
        truths, learned, yerr=2.0 * sigmas, fmt="o", color=colour, markersize=5,
        ecolor="#9a9a9a", elinewidth=0.9, capsize=3,
        label=f"learned ridge, MAE {result.learned_mae_c:.3f} degC (2 sigma bars)",
    )
    left.scatter(
        truths, rc_values, marker="^", s=34, color=_common.ACCENT,
        label=f"RC on the same window, MAE {result.rc_mae_c:.3f} degC",
    )
    left.set_xlim(span)
    left.set_ylim(span)
    left.set_xlabel("measured steady-state die temperature, degC")
    left.set_ylabel("predicted from the warm-up window alone, degC")
    left.set_title("Held-out duty cycle: predicted against measured", fontsize=9.5)
    left.legend(loc="upper left", fontsize=7.2)

    held = [split.held_out for split in result.splits]
    learned_mae = [split.learned_mae_c for split in result.splits]
    rc_mae = [split.rc_mae_c for split in result.splits]
    positions = np.arange(len(held), dtype=float)
    right.bar(positions - 0.18, learned_mae, width=0.34, color=colour, label="learned ridge")
    right.bar(positions + 0.18, rc_mae, width=0.34, color=_common.ACCENT, label="RC baseline")
    right.axhline(
        truth.die_c.half_width,
        color="#2a6ea6",
        lw=1.3,
        ls=":",
        label=f"run-to-run half-width {truth.die_c.half_width:.3f} degC",
    )
    right.set_xticks(positions)
    right.set_xticklabels(held, fontsize=8)
    right.set_ylabel("mean absolute error, degC")
    right.set_title(
        f"VERDICT {verdict.verdict}\n"
        f"margin {verdict.margin_c:+.4f} degC = {verdict.ratio:.2f} x the run-to-run "
        "half-width",
        fontsize=9.5,
    )
    right.legend(fontsize=7.4, loc="upper left")

    _common.stamp(
        figure,
        [left, right],
        is_measured,
        detail=(
            f"{dataset.n_rows} rows from {len(dataset.runs)} runs; rows within a run are "
            f"correlated, so splits are grouped by run and scored by duty. Two-sigma "
            f"coverage {result.coverage:.3f}."
        ),
    )
    figure.subplots_adjust(bottom=0.22, wspace=0.27)
    path = _common.save(figure, "learned_vs_rc.png")
    plt.close(figure)
    print(f"wrote {path}")
    print(f"  learned MAE {result.learned_mae_c:.4f} degC, RC MAE {result.rc_mae_c:.4f} degC")
    print(f"  margin {result.margin_c:+.4f} degC, ratio {verdict.ratio:.4f}")
    print(f"  VERDICT {verdict.verdict}")
    print("  held-out AMBIENT is STRUCTURALLY UNMEASURABLE in this campaign and has no "
          "score; see MODEL_CARD.md")
    print(f"  origin: {'MEASURED' if is_measured else 'SYNTHETIC'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
