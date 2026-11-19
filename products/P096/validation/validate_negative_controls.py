"""Negative controls: anchored on the TRUE value, and each asserted to FIRE.

A check that cannot fail is not a check. Each control below injects a violation
anchored on the GROUND TRUTH rather than on an output, and the script reports how
many of each kind fired as its own line.

The anchoring is the whole point. Offsetting an output that already sits inside a
band does not guarantee leaving the band: only an offset strictly greater than
twice the bound does. An earlier product of this mission shipped two of eleven
controls silently too weak for exactly that reason, and the certificate read as
evidence while the control was wrong.

Two controls here are declared must-NOT-fire, and both directions are asserted,
because a control that fires when it must not is equally wrong.

No control's criterion is a seed. Where a control involves randomness its
criterion is derived -- a sampling bound or a factor on the true value -- and the
derivation is printed beside it.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from handoverloss.accounting import (  # noqa: E402
    AttributionReport,
    MechanismShare,
    attribute,
    residual_tolerance_bits,
)
from handoverloss.bootstrap import event_bootstrap  # noqa: E402
from handoverloss.corpus import load_corpus, station_pair_split  # noqa: E402
from handoverloss.predictor import HandoverLossPredictor  # noqa: E402
from handoverloss.sequence import MECHANISMS  # noqa: E402
from handoverloss.trace import TraceResult, simulate  # noqa: E402


def _perturbed(trace: TraceResult, column: int, factor: float) -> TraceResult:
    """Multiply one mechanism bucket, leaving the conservation totals alone."""
    dropped = trace.dropped_bits.copy()
    dropped[:, column] *= factor
    return TraceResult(
        dropped_bits=dropped,
        produced_bits=trace.produced_bits,
        delivered_bits=trace.delivered_bits,
        final_occupancy_bits=trace.final_occupancy_bits,
        initial_occupancy_bits=trace.initial_occupancy_bits,
        peak_occupancy_bits=trace.peak_occupancy_bits,
        slots_full=trace.slots_full,
        slot_seconds=trace.slot_seconds,
        slot_count=trace.slot_count,
    )


def main() -> int:
    out = _bootstrap.Tee("validate_negative_controls_output.txt")
    out.rule("NEGATIVE CONTROLS -- anchored on the true value, each asserted to fire")
    out(_bootstrap.ORIGIN)
    out("")
    for line in _bootstrap.environment_lines():
        out(line)

    corpus = _bootstrap.require_corpus()
    count = _bootstrap.TRACE_EVENTS
    parameters = _bootstrap.corpus_event_parameters(corpus, count)
    trace = simulate(parameters, np.random.default_rng(_bootstrap.SEED_TRACE))
    clean = attribute(trace, resamples=_bootstrap.RESAMPLES)
    out("")
    out(f"corpus sha256    : {corpus.sha256}")
    out(f"trace events     : {len(trace)}")
    out(f"clean residual   : {clean.residual_bits:.6e} bits, tolerance "
        f"{clean.residual_tolerance_bits:.6e} bits")

    out.rule("1  the decomposition residual guard")
    out(
        "  The residual compares the phase buckets against mass conservation. Breaking\n"
        "  one accounting and leaving the other must move it. Each injected error is a\n"
        "  FRACTION OF THE TRUE BUCKET VALUE, so the violation's size is known before\n"
        "  the check runs and the printed ratio shows how far past the tolerance it is."
    )
    for column, name in enumerate(MECHANISMS):
        for factor, description in ((1.01, "inflated by 1 per cent"), (0.0, "zeroed")):
            true_value = float(trace.dropped_bits[:, column].mean())
            dirty = attribute(_perturbed(trace, column, factor), resamples=200)
            injected = abs(true_value * (1.0 - factor))
            fired = not dirty.residual_within_tolerance
            out.control(
                f"{name} bucket {description}",
                fired,
                must_fire=True,
                detail=(
                    f"injected {injected:.6e} bits = "
                    f"{injected / max(dirty.residual_tolerance_bits, 1e-300):.3e} x the "
                    "tolerance"
                ),
            )
    out.control(
        "the unperturbed trace",
        not clean.residual_within_tolerance,
        must_fire=False,
        detail=f"residual {clean.residual_bits:.6e} bits inside "
        f"{clean.residual_tolerance_bits:.6e} bits; a guard that fired here would "
        "reject every correct accounting",
    )

    out.rule("2  the conditioning-scaled tolerance cannot be made vacuous")
    out(
        "  A tolerance floored at an absolute constant would admit the very error the\n"
        "  check exists to catch. This one collapses with the magnitudes, so a violation\n"
        "  against a small-magnitude window is still detected."
    )
    at_zero = float(residual_tolerance_bits(0.0, 0.0, 0.0, 1))
    out(f"  tolerance at zero magnitude             : {at_zero:.6e} bits")
    out.control(
        "a one-bit error against a zero-magnitude window",
        abs(1.0) > at_zero,
        must_fire=True,
        detail="an absolute floor would have swallowed this",
    )
    small = float(residual_tolerance_bits(1.0e6, 1.0e6, 1.0e3, 100))
    out(f"  tolerance at 1e6-bit magnitude, 100 slots: {small:.6e} bits")
    out.control(
        "a 1 per cent error against a 1e3-bit loss",
        10.0 > small,
        must_fire=True,
        detail=f"10 bits against a tolerance of {small:.6e} bits",
    )

    out.rule("3  the resampling unit")
    out(
        "  The reason the unit is declared AND asserted. Resampling below the unit of\n"
        "  independence narrows the interval, so a report carrying any other unit is\n"
        "  refused at construction rather than trusted."
    )
    per_event = event_bootstrap(
        trace.dropped_bits,
        lambda rows: float(rows[:, 0].mean()),
        resamples=_bootstrap.RESAMPLES,
        rng=np.random.default_rng(_bootstrap.SEED_CONTROL),
    )
    pieces = np.repeat(trace.dropped_bits[:, 0] / 10.0, 10).reshape(-1, 1)
    pseudo = event_bootstrap(
        pieces,
        lambda rows: float(rows[:, 0].mean() * 10.0),
        resamples=_bootstrap.RESAMPLES,
        rng=np.random.default_rng(_bootstrap.SEED_CONTROL),
    )
    out(f"  per-event interval width                : {per_event.width:.6e} bits "
        f"over {per_event.n_events} units")
    out(f"  pseudo-independent interval width       : {pseudo.width:.6e} bits "
        f"over {pseudo.n_events} units")
    out(f"  ratio (pseudo / per-event)              : "
        f"{pseudo.width / per_event.width:.6f}")
    out(
        "  Both widths are statistics of the same committed data, not wall-clock\n"
        "  readings, so a ratio between them is a legitimate assertion. Splitting each\n"
        "  event into ten equal pieces multiplies the apparent sample size by ten, so\n"
        "  the standard error falls by about sqrt(10) = 3.162 and the interval narrows\n"
        "  by roughly that factor. The criterion below is the derived 1/sqrt(10) with a\n"
        "  generous margin, not a tuned number."
    )
    out.control(
        "a bootstrap resampled below the unit of independence",
        pseudo.width < 0.6 * per_event.width,
        must_fire=True,
        detail=f"ratio {pseudo.width / per_event.width:.6f}; derived expectation "
        f"{1.0 / np.sqrt(10.0):.6f}",
    )

    shares = tuple(
        MechanismShare(name=name, loss_bits=1.0, low_bits=0.0, high_bits=2.0, share=1 / 3)
        for name in MECHANISMS
    )
    refused = False
    try:
        AttributionReport(
            mechanisms=shares,  # type: ignore[arg-type]
            dominant="drain",
            dominance_margin=0.0,
            dominance_is_decided=False,
            total_loss_bits=3.0,
            conservation_total_bits=3.0,
            residual_bits=0.0,
            residual_relative=0.0,
            residual_tolerance_bits=1.0,
            resampling_unit="accounting slot",
            n_events=10,
            n_resamples=10,
            confidence=0.95,
        )
    except ValueError:
        refused = True
    out.control(
        "a report claiming the accounting slot as its resampling unit",
        refused,
        must_fire=True,
        detail="refused at construction, not merely documented",
    )

    out.rule("4  the learned component's own guards")
    out(
        "  A predictor that could predict outside its training target range, or that\n"
        "  silently accepted a post-handover feature, would be a measurement dressed as\n"
        "  a prediction. Both are controlled."
    )
    loaded = load_corpus()
    split = station_pair_split(loaded)
    model = HandoverLossPredictor(n_estimators=80).fit(
        loaded.features[split.train], loaded.targets[split.train]
    )
    prediction = model.predict(loaded.features[split.test])
    ceiling = float(loaded.targets[split.train].max())
    out(f"  training target ceiling                 : {ceiling:.6e} bits")
    out(f"  largest prediction                      : "
        f"{float(prediction.loss_bits.max()):.6e} bits")
    out.control(
        "a prediction above the training target range",
        float(prediction.loss_bits.max()) > ceiling * (1.0 + 1e-9),
        must_fire=False,
        detail="a forest average lies inside the training range by construction; a "
        "prediction outside it would mean the estimator is not what this product "
        "claims",
    )
    bad = np.zeros((30, 4))
    refused = False
    try:
        model.predict(bad)
    except ValueError:
        refused = True
    out.control(
        "a prediction call with the wrong feature width",
        refused,
        must_fire=True,
        detail="18 features are declared and the width is checked",
    )

    out.rule("5  an uncertainty output that is not a point estimate")
    out(
        "  A band of zero width would be a point estimate wearing an interval's clothes.\n"
        "  The control asserts the band has width on a non-trivial fraction of events."
    )
    width = prediction.high_bits - prediction.low_bits
    non_degenerate = float(np.mean(width > 0.0))
    out(f"  fraction of (event, mechanism) cells with a non-zero band: "
        f"{non_degenerate:.6f}")
    out.control(
        "a degenerate zero-width uncertainty band",
        non_degenerate < 0.5,
        must_fire=False,
        detail=f"{non_degenerate:.6f} of cells carry a band; a value below 0.5 would "
        "mean the across-tree spread collapsed and the output is a point estimate",
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
