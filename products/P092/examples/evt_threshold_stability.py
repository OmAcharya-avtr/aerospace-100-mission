"""The threshold-stability diagnostic as a committed figure.

Draws, from the MEASURED campaign's ground-truth run:

- top: the GPD shape ``xi`` against the candidate threshold;
- middle: the modified scale ``beta* = beta - xi*u``, the quantity that is
  constant in ``u`` above a threshold at which the GPD approximation holds
  (Coles 2001 eq. 4.10);
- bottom: the resulting ``P(resp > D)`` against the threshold, with the counted
  rate and its moving-block interval drawn as a band so a reader can see which
  thresholds would and would not reproduce the measurement.

The chosen threshold is marked. It is chosen by the declared rule in
``deadlinerisk.baselines.THRESHOLD_RULE``, applied by code and not by eye.

Writes ``screenshots/evt_threshold_stability.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _common  # noqa: E402

ROOT = _common.bootstrap()

from deadlinerisk.baselines import choose_threshold, threshold_stability  # noqa: E402
from deadlinerisk.bootstrap import counted_rate_interval  # noqa: E402
from deadlinerisk.trace import TraceDataAbsentError  # noqa: E402

RESAMPLES = 2000
SEED = 920092


def main() -> int:
    try:
        campaign = _common.measured_campaign()
    except TraceDataAbsentError as error:
        print(f"REFUSED: {error}")
        return 1

    jobs = campaign.jobs["gt_long"]
    fits = threshold_stability(jobs.response_s, jobs.deadline_s)
    if not fits:
        print(
            "REFUSED: no candidate threshold on the declared grid left enough "
            "exceedances to fit, so there is no stability diagnostic to draw."
        )
        return 1
    choice = choose_threshold(fits)
    counted = counted_rate_interval(jobs, RESAMPLES, SEED)

    quantiles = np.asarray([fit.quantile for fit in fits])
    shapes = np.asarray([fit.shape for fit in fits])
    modified = np.asarray([fit.modified_scale for fit in fits]) * 1e3
    probabilities = np.asarray([fit.probability for fit in fits])
    counts = np.asarray([fit.n_exceedances for fit in fits])
    chosen_quantile = choice.chosen.quantile

    handle, save = _common.figure("evt_threshold_stability")
    top = handle.add_subplot(3, 1, 1)
    top.plot(quantiles, shapes, "o-", ms=4, lw=1.3)
    top.axvline(chosen_quantile, color="crimson", lw=1.2, ls="--")
    top.set_ylabel("GPD shape xi")
    top.set_title(
        "threshold stability on gt_long; the chosen threshold is marked in red"
    )
    top.grid(alpha=0.3)

    middle = handle.add_subplot(3, 1, 2, sharex=top)
    middle.plot(quantiles, modified, "s-", ms=4, lw=1.3, color="darkorange")
    middle.axvline(chosen_quantile, color="crimson", lw=1.2, ls="--")
    middle.set_ylabel("modified scale\nbeta - xi*u  (ms)")
    middle.grid(alpha=0.3)

    bottom = handle.add_subplot(3, 1, 3, sharex=top)
    positive = probabilities > 0.0
    if np.any(positive):
        bottom.semilogy(
            quantiles[positive], probabilities[positive], "^-", ms=4, lw=1.3,
            color="seagreen", label="GPD P(resp > D) at this threshold",
        )
    if np.any(~positive):
        bottom.plot(
            quantiles[~positive],
            np.full(int((~positive).sum()), max(counted.low, 1e-12) * 0.3),
            "x", color="grey", ms=7,
            label="GPD endpoint below D: model says P = 0 exactly",
        )
    bottom.axhspan(
        max(counted.low, 1e-300), max(counted.high, 1e-300),
        color="crimson", alpha=0.18,
        label=f"counted rate 95% interval (block {counted.block.block_length})",
    )
    bottom.axhline(counted.rate, color="crimson", lw=1.1)
    bottom.axvline(chosen_quantile, color="crimson", lw=1.2, ls="--")
    bottom.set_xlabel("candidate threshold, as a quantile of the measured response times")
    bottom.set_ylabel("P(resp > D)")
    bottom.grid(alpha=0.3, which="both")
    bottom.legend(fontsize=7, loc="best")

    path = save()
    print(f"admissible candidates    {len(fits)} of 20")
    print(f"exceedance counts        {counts.min()} .. {counts.max()}")
    print(f"shape xi range           {shapes.min():.6f} .. {shapes.max():.6f}")
    print(f"chosen threshold u       {choice.chosen.threshold_s * 1e3:.6f} ms")
    print(f"chosen quantile          {chosen_quantile:.4f}")
    print(f"chosen exceedances       {choice.chosen.n_exceedances}")
    print(f"chosen xi                {choice.chosen.shape:.6f}")
    print(f"chosen beta              {choice.chosen.scale * 1e3:.6f} ms")
    print(f"chosen modified scale    {choice.chosen.modified_scale * 1e3:.6f} ms")
    print(f"chosen GPD P(resp > D)   {choice.chosen.probability:.6e}")
    print(f"counted miss rate        {counted.rate:.6e}")
    print(f"counted 95% interval     [{counted.low:.6e}, {counted.high:.6e}]")
    print(f"figure written to        {path}")
    print(
        "What to notice: the range of thresholds over which xi and the modified scale "
        "are flat. The chosen threshold sits at the start of that range, which is what "
        "the declared rule selects, and the bottom panel shows whether the resulting "
        "probability reaches the counted rate's interval."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
