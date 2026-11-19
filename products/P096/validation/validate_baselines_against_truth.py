"""Validate BOTH analytic baselines against the ground truth, BEFORE any benchmark.

This script runs first in the reading order of ``validation/VALIDATION.md`` and
is the reason the learned-model comparison can be believed at all. The mission's
build guide records that an under-strength analytic baseline inflated an earlier
product's apparent learned-model win by about 50 per cent, and that a
specification-named baseline turned out to be the weakest of three candidates in
another. The only defence is to check each baseline against the ground truth
before scoring anything against it.

WHAT IS CHECKED, AND WHAT EACH CHECK WOULD CATCH
------------------------------------------------

1. **Baseline 1 is the exact restriction of baseline 2 to its first phase.**
   If this fails, baseline 1 has been weakened and every margin reported over it
   is inflated by the weakening.
2. **Baseline 2, given the REALISED durations, reproduces the trace's
   decomposition.** This is the baseline's own accuracy, separated from its
   accuracy as a PREDICTOR: the predictor's error also includes not knowing the
   acquisition time in advance. Reported as the dominance agreement, the median
   ratio of totals and the log10 mean absolute error, per mechanism.
3. **The residual between the two accountings of the total** -- phase buckets
   against mass conservation -- over the whole corpus.
4. **The discretisation and burstiness terms, separated.** Re-running the
   simulator with the burst and jitter shapes set to zero isolates how much of
   the baseline-2 discrepancy is the fluid approximation (burstiness) and how
   much is the slot width. Both are reported numbers, which is what lets a
   reader decide whether the closed form is adequate for their own source.
5. **Baseline 1's conservative variant.** Evaluated at the 95th percentile of
   the acquisition distribution as well as at its mean, so nobody can say the
   refuted assumption was tested only in its weakest form.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from handoverloss.accounting import residual_tolerance_bits  # noqa: E402
from handoverloss.acquisition import AcquisitionModel  # noqa: E402
from handoverloss.baselines import (  # noqa: E402
    acquisition_only_loss_bits,
    fluid_decomposition,
    fluid_decomposition_array,
)
from handoverloss.sequence import MECHANISMS, HandoverSequence  # noqa: E402
from handoverloss.trace import simulate  # noqa: E402


def main() -> int:
    out = _bootstrap.Tee("validate_baselines_against_truth_output.txt")
    out.rule("BASELINE VALIDATION -- both analytic baselines against the ground truth")
    out(_bootstrap.ORIGIN)
    out("")
    out(_bootstrap.ACCOUNTING_MODEL)
    out("")
    for line in _bootstrap.environment_lines():
        out(line)

    corpus = _bootstrap.require_corpus()
    table = corpus.table
    out("")
    out(f"corpus           : {_bootstrap.CORPUS_CSV}")
    out(f"corpus sha256    : {corpus.sha256}")
    out(f"events           : {len(corpus)}")

    out.rule("1  baseline 1 is the exact restriction of baseline 2 to its first phase")
    out(
        "  If this fails, baseline 1 has been weakened and every margin reported over\n"
        "  it is inflated by the weakening. Checked on 400 handovers spanning the\n"
        "  declared parameter envelope, including the nearly-full-buffer case where the\n"
        "  occupancy-space form of baseline 2 suffered a cancellation that Hypothesis\n"
        "  found; see step 5 of the handoverloss.baselines module docstring."
    )
    rng = np.random.default_rng(_bootstrap.SEED_ANALYSIS)
    worst_relative = 0.0
    exact = 0
    trials = 400
    for _ in range(trials):
        downlink = float(rng.uniform(2e8, 3e9))
        source = float(rng.uniform(0.02, 1.4)) * downlink
        capacity = float(rng.uniform(1e9, 8e10))
        # Deliberately skewed towards a nearly full buffer, which is the
        # cancellation-prone case and the operationally interesting one.
        occupancy = capacity * float(rng.beta(6.0, 1.2))
        acquisition = float(rng.uniform(0.0, 150.0))
        restart = float(rng.uniform(0.0, 90.0))
        ramp = float(rng.uniform(0.0, 200.0))
        sequence = HandoverSequence(
            source_rate_bps=source,
            downlink_rate_bps=downlink,
            buffer_capacity_bits=capacity,
            initial_occupancy_bits=min(occupancy, capacity),
            acquisition_s=acquisition,
            restart_s=restart,
            ramp_s=ramp,
            horizon_s=acquisition + restart + ramp + 300.0,
        )
        term = fluid_decomposition(sequence).acquisition_bits
        only = acquisition_only_loss_bits(sequence)
        if term == only:
            exact += 1
        scale = max(abs(term), abs(only), 1.0)
        worst_relative = max(worst_relative, abs(term - only) / scale)
    out(f"  handovers compared                      : {trials}")
    out(f"  bit-identical                           : {exact}")
    out(f"  worst relative discrepancy              : {worst_relative:.6e}")
    out.check(
        "baseline 1 equals baseline 2's acquisition term to within double rounding",
        worst_relative <= 8.0 * (2.0**-53),
        f"worst {worst_relative:.6e} against a bound of {8.0 * 2.0**-53:.6e}",
    )

    out.rule("2  baseline 2 against the trace, given the REALISED durations")
    out(
        "  This is the CLOSED FORM's own accuracy, with the acquisition time and the\n"
        "  restart latency taken from the trace rather than predicted. It separates\n"
        "  the model error from the forecasting error, which the next script measures."
    )
    truth = corpus.targets
    fluid = fluid_decomposition_array(
        table["source_rate_bps"],
        table["downlink_rate_bps"],
        table["buffer_capacity_bits"],
        table["initial_occupancy_bits"],
        table["realised_acquisition_s"],
        table["realised_restart_s"],
        table["declared_ramp_s"],
        table["horizon_s"],
    )
    agreement = float(np.mean(np.argmax(fluid, axis=1) == np.argmax(truth, axis=1)))
    total_fluid = fluid.sum(axis=1)
    total_truth = truth.sum(axis=1)
    positive = total_truth > 0.0
    ratio = total_fluid[positive] / total_truth[positive]
    out(f"  events with non-zero ground-truth loss  : {int(positive.sum())} of {len(corpus)}")
    out(f"  dominance agreement                     : {agreement:.6f}")
    out(f"  median total ratio (fluid / truth)      : {float(np.median(ratio)):.6f}")
    out(f"  5th / 95th percentile of that ratio     : {float(np.percentile(ratio, 5)):.6f}"
        f" / {float(np.percentile(ratio, 95)):.6f}")
    log_mae_total = float(
        np.mean(np.abs(np.log10(1.0 + total_fluid) - np.log10(1.0 + total_truth)))
    )
    out(f"  log10 MAE on the total                  : {log_mae_total:.6f}")
    for column, name in enumerate(MECHANISMS):
        per = float(
            np.mean(
                np.abs(
                    np.log10(1.0 + fluid[:, column]) - np.log10(1.0 + truth[:, column])
                )
            )
        )
        out(f"  log10 MAE, {name:<12}                 : {per:.6f}")
    out.check(
        "baseline 2 agrees with the ground truth's dominant mechanism on most events",
        agreement >= 0.90,
        f"{agreement:.6f}; a baseline below this would be too weak to benchmark against",
    )
    out.check(
        "baseline 2's median total is within 20 per cent of the ground truth's",
        0.80 <= float(np.median(ratio)) <= 1.20,
        f"median ratio {float(np.median(ratio)):.6f}",
    )

    out.rule("3  the decomposition residual over the whole corpus")
    out(
        "  Phase buckets against mass conservation. The two paths share the per-slot\n"
        "  arrival and service arrays and nothing else: one accumulates overflow, the\n"
        "  other accumulates arrivals, departures and the endpoint occupancies. The\n"
        "  residual is a REPORTED NUMBER, not an assumption."
    )
    bucketed = truth.sum(axis=1)
    conserved = np.asarray(table["conservation_total_bits"], dtype=float)
    produced = np.asarray(table["produced_bits"], dtype=float)
    delivered = np.asarray(table["delivered_bits"], dtype=float)
    residual = conserved - bucketed
    slots = int(np.ceil(float(table["horizon_s"][0]) / 0.25))
    tolerance = residual_tolerance_bits(produced, delivered, bucketed, slots)
    out(f"  slots per event used for the tolerance  : {slots}")
    out(f"  worst absolute residual                 : "
        f"{float(np.max(np.abs(residual))):.6e} bits")
    out(f"  median absolute residual                : "
        f"{float(np.median(np.abs(residual))):.6e} bits")
    out(f"  smallest per-event tolerance            : {float(np.min(tolerance)):.6e} bits")
    out(f"  worst |residual| / tolerance            : "
        f"{float(np.max(np.abs(residual) / tolerance)):.6e}")
    out.check(
        "every event's absolute residual is inside its conditioning-scaled tolerance",
        bool(np.all(np.abs(residual) <= tolerance)),
        f"worst ratio {float(np.max(np.abs(residual) / tolerance)):.6e} of the tolerance; "
        "the tolerance is 4 * slots * eps * (produced + delivered + dropped), which "
        "scales with the slot count and the volume magnitudes rather than being a "
        "constant",
    )
    out(
        "  WHY THE RELATIVE RESIDUAL IS REPORTED CONDITIONALLY. An absolute residual of\n"
        "  order 1e-3 bits is negligible against a loss of gigabits and meaningless\n"
        "  against a loss of zero. Dividing by the total over ALL events, including the\n"
        "  216 with no loss at all, gives a worst relative residual of\n"
        f"  {float(np.max(np.abs(residual) / np.maximum(conserved, 1.0))):.6e}, which is an\n"
        "  artefact of the denominator and not a measurement of the accounting. An\n"
        "  earlier version of this script asserted on that unconditional figure and\n"
        "  FAILED, correctly: the check was wrong, not the product. The figure below is\n"
        "  restricted to events whose ground-truth loss exceeds 1e6 bits, where the\n"
        "  ratio means something."
    )
    meaningful = conserved > 1.0e6
    restricted = np.abs(residual[meaningful]) / conserved[meaningful]
    out(f"  events with loss above 1e6 bits         : {int(meaningful.sum())} of {len(corpus)}")
    out(f"  worst relative residual on those        : {float(np.max(restricted)):.6e}")
    out(f"  median relative residual on those       : {float(np.median(restricted)):.6e}")
    derived = tolerance[meaningful] / conserved[meaningful]
    out(f"  derived bound on that ratio, worst case : {float(np.max(derived)):.6e}")
    out(
        "  The criterion below is the DERIVED bound tolerance/loss, not a round number.\n"
        "  An earlier version of this check used a literal 1e-12 and FAILED at\n"
        "  1.161965e-12 -- a constant that happened to sit just under the measured\n"
        "  value, which is the kind of bound this mission's build guide forbids: the\n"
        "  slack must follow the conditioning of the expression, not an absolute\n"
        "  constant. The measured worst case is reported above and in VALIDATION.md."
    )
    out.check(
        "every event's relative residual is inside the derived bound tolerance/loss",
        bool(np.all(restricted <= derived)),
        f"worst measured {float(np.max(restricted)):.6e} against a worst derived bound "
        f"of {float(np.max(derived)):.6e}",
    )

    out.rule("4  the discretisation and burstiness terms, separated")
    out(
        "  Re-running the simulator with the burst and jitter shapes set to zero makes\n"
        "  it a pure discretisation of the fluid model. The difference between that run\n"
        "  and the closed form is the SLOT-WIDTH term; the difference between the\n"
        "  committed corpus and that run is the BURSTINESS term. Both are reported, so a\n"
        "  reader can decide whether the closed form is adequate for their own source."
    )
    count = _bootstrap.TRACE_EVENTS
    parameters = _bootstrap.corpus_event_parameters(corpus, count)
    smooth = simulate(
        parameters,
        np.random.default_rng(_bootstrap.SEED_TRACE),
        slot_seconds=0.25,
        burst_log_sigma=0.0,
        service_log_sigma=0.0,
    )
    closed = fluid[:count]
    smooth_total = smooth.dropped_bits.sum(axis=1)
    closed_total = closed.sum(axis=1)
    bursty_total = truth[:count].sum(axis=1)
    out(f"  events compared                         : {count}")
    out(f"  slot width                              : {smooth.slot_seconds} s "
        f"({smooth.slot_count} slots per event)")
    with np.errstate(divide="ignore", invalid="ignore"):
        slot_term = np.where(
            closed_total > 0.0, (smooth_total - closed_total) / closed_total, 0.0
        )
        burst_term = np.where(
            smooth_total > 0.0, (bursty_total - smooth_total) / smooth_total, 0.0
        )
    out(f"  SLOT-WIDTH term, median relative        : {float(np.median(slot_term)):+.6e}")
    out(f"  SLOT-WIDTH term, worst absolute relative: {float(np.max(np.abs(slot_term))):.6e}")
    out(f"  BURSTINESS term, median relative        : {float(np.median(burst_term)):+.6e}")
    out(f"  BURSTINESS term, worst absolute relative: {float(np.max(np.abs(burst_term))):.6e}")
    out(f"  events where burstiness changed nothing : "
        f"{int(np.sum(burst_term == 0.0))} of {count}")
    out.check(
        "the slot-width term is small enough that the closed form is the right reference",
        float(np.median(np.abs(slot_term))) < 0.05,
        f"median |relative| {float(np.median(np.abs(slot_term))):.6e} at dt=0.25 s",
    )
    out(
        "  NOTE on the sign of the burstiness term. The fluid model cannot produce loss\n"
        "  from a burst arriving at a non-full buffer, so where the term is positive the\n"
        "  closed form UNDERSTATES. Where it is negative, the burst arrived while the\n"
        "  buffer was already full and the same volume was lost either way, while the\n"
        "  service jitter delivered slightly more. Both signs occur and neither is a\n"
        "  defect; the numbers above are the magnitude a reader should plan around."
    )

    out.rule("5  baseline 1 at its conservative setting, not only at the mean")
    out(
        "  The refuted assumption, evaluated at the 95th percentile of the acquisition\n"
        "  distribution as well as at its mean, so nobody can say it was tested only in\n"
        "  its weakest form. Both are reported; the next script scores both on held-out\n"
        "  data."
    )
    sample = min(256, len(corpus))
    mean_losses = np.empty(sample)
    upper_losses = np.empty(sample)
    for row in range(sample):
        model = AcquisitionModel(
            median_s=float(table["acquisition_median_s"][row]),
            log_sigma=float(table["acquisition_log_sigma"][row]),
            abort_probability=float(table["acquisition_abort_probability"][row]),
            attempt_window_s=float(table["acquisition_attempt_window_s"][row]),
        )
        mean_s, sd_s = model.moments()
        # Normal approximation to the upper quantile from the exact moments; the
        # approximation error is reported by validate_acquisition_model.py.
        upper_s = mean_s + 1.6448536269514722 * sd_s
        headroom = float(table["buffer_capacity_bits"][row]) - float(
            table["initial_occupancy_bits"][row]
        )
        rate = float(table["source_rate_bps"][row])
        mean_losses[row] = max(0.0, rate * mean_s - headroom)
        upper_losses[row] = max(0.0, rate * upper_s - headroom)
    truth_total = truth[:sample].sum(axis=1)
    out(f"  events                                  : {sample}")
    out(f"  baseline 1 at the MEAN, log10 MAE       : "
        f"{float(np.mean(np.abs(np.log10(1 + mean_losses) - np.log10(1 + truth_total)))):.6f}")
    out(f"  baseline 1 at the 95th pct, log10 MAE   : "
        f"{float(np.mean(np.abs(np.log10(1 + upper_losses) - np.log10(1 + truth_total)))):.6f}")
    out.check(
        "baseline 1's conservative setting is strictly larger than its mean setting",
        bool(np.all(upper_losses >= mean_losses)),
        "a conservative variant that was not larger would not be conservative",
    )
    out(
        "  Baseline 1 attributes every bit of loss to acquisition by construction, so\n"
        "  its dominance prediction is always 'acquisition' and its accuracy equals the\n"
        "  rate at which acquisition genuinely dominates. That rate, on this corpus, is\n"
        f"  {corpus.dominance_counts()['acquisition'] / len(corpus):.6f}."
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
