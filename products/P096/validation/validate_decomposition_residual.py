"""The headline: the three-way attribution with an interval on each, and its residual.

This script produces the number this product exists for. It runs baseline 3 --
direct accounting over the simulated handover trace -- on the committed corpus,
and reports:

* the per-mechanism loss with an event-level bootstrap interval on each;
* which mechanism dominates, and whether the data DECIDE that ordering (the
  leading interval disjoint from the runner-up's) as opposed to merely ranking
  it;
* the decomposition residual between the phase-bucket accounting and the
  mass-conservation accounting, as a reported number with its
  conditioning-scaled tolerance beside it;
* the same, split by the operating regime, because the whole point of the
  product is that WHICH mechanism dominates changes with the regime and a single
  corpus-wide answer would hide that.

THE RESAMPLING UNIT IS THE HANDOVER EVENT.
Slots within one handover share an occupancy state: once the buffer is full it
tends to stay full, so consecutive slots drop together. Resampling slots would
report an interval narrower than the data support. The unit is declared on
:class:`~handoverloss.bootstrap.BootstrapInterval` and on
:class:`~handoverloss.accounting.AttributionReport`, and both objects REFUSE to
be constructed with any other value. ``validate_negative_controls.py`` measures
how much narrower a slot-level interval would be.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from handoverloss.accounting import attribute  # noqa: E402
from handoverloss.bootstrap import RESAMPLING_UNIT  # noqa: E402
from handoverloss.sequence import MECHANISMS  # noqa: E402
from handoverloss.trace import EventParameters, simulate  # noqa: E402


def _regime_report(out, corpus, mask, label: str) -> None:
    """Attribute the subset selected by ``mask`` and print its verdict."""
    index = np.flatnonzero(mask)
    if index.size < 5:
        out.unmeasurable_check(
            f"regime {label}",
            f"only {index.size} event(s) fall in this regime, and the package declines "
            "to present a percentile interval over fewer than five handover events",
        )
        return
    table = corpus.table
    parameters = EventParameters(
        source_rate_bps=table["source_rate_bps"][index],
        downlink_rate_bps=table["downlink_rate_bps"][index],
        buffer_capacity_bits=table["buffer_capacity_bits"][index],
        initial_occupancy_bits=table["initial_occupancy_bits"][index],
        acquisition_s=table["realised_acquisition_s"][index],
        restart_s=table["realised_restart_s"][index],
        ramp_s=table["declared_ramp_s"][index],
        horizon_s=float(table["horizon_s"][0]),
    )
    trace = simulate(parameters, np.random.default_rng(_bootstrap.SEED_TRACE))
    report = attribute(
        trace, resamples=_bootstrap.RESAMPLES, confidence=_bootstrap.CONFIDENCE
    )
    out("")
    out(f"  REGIME: {label}  (n = {index.size})")
    for line in report.lines():
        out(f"    {line}")
    # The MEAN-weighted attribution above and the PER-EVENT dominance mix below
    # are different statistics and they do not agree. The mean is dominated by
    # the largest-loss events, which are acquisition-heavy in every regime; the
    # per-event mix is what an operator planning a typical handover needs. Both
    # are printed because reporting only one would be misleading, and the
    # disagreement is itself a finding -- it is recorded in README.md.
    truth = corpus.targets[index]
    per_event = np.argmax(truth, axis=1)
    mix = {
        name: int(np.sum(per_event == column)) for column, name in enumerate(MECHANISMS)
    }
    out(f"    per-event dominance mix: {mix}  "
        f"(fractions {{{', '.join(f'{k}: {v / index.size:.4f}' for k, v in mix.items())}}})")
    out(f"    mean-weighted dominant : {report.dominant};  "
        f"per-event modal dominant: {max(mix, key=lambda k: mix[k])}")
    out.check(
        f"[{label}] the decomposition residual is inside its conditioning-scaled tolerance",
        report.residual_within_tolerance,
        f"residual {report.residual_bits:.6e} bits against a tolerance of "
        f"{report.residual_tolerance_bits:.6e} bits",
    )


def main() -> int:
    out = _bootstrap.Tee("validate_decomposition_residual_output.txt")
    out.rule("ATTRIBUTION -- the three mechanisms, an interval on each, and the residual")
    out(_bootstrap.ORIGIN)
    out("")
    out(_bootstrap.ACCOUNTING_MODEL)
    out("")
    out(f"RESAMPLING UNIT: {RESAMPLING_UNIT}. Declared on every returned object and")
    out("asserted at construction; see validation/VALIDATION.md under 'Resampling unit'.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)

    corpus = _bootstrap.require_corpus()
    table = corpus.table
    out("")
    out(f"corpus sha256    : {corpus.sha256}")
    out(f"events           : {len(corpus)}")

    out.rule("1  the corpus-wide attribution")
    out(
        "  Read the per-mechanism intervals, not the total. The total says a handover is\n"
        "  expensive; the decomposition says whether to buy faster acquisition, a\n"
        "  different protocol or more buffer."
    )
    count = _bootstrap.TRACE_EVENTS
    parameters = _bootstrap.corpus_event_parameters(corpus, count)
    trace = simulate(parameters, np.random.default_rng(_bootstrap.SEED_TRACE))
    report = attribute(
        trace, resamples=_bootstrap.RESAMPLES, confidence=_bootstrap.CONFIDENCE
    )
    out("")
    for line in report.lines():
        out(f"  {line}")
    out("")
    out.check(
        "the attribution report carries the handover event as its resampling unit",
        report.resampling_unit == RESAMPLING_UNIT,
        report.resampling_unit,
    )
    out.check(
        "the decomposition residual is inside its conditioning-scaled tolerance",
        report.residual_within_tolerance,
        f"residual {report.residual_bits:.6e} bits, relative "
        f"{report.residual_relative:.6e}, tolerance "
        f"{report.residual_tolerance_bits:.6e} bits",
    )
    out.check(
        "every mechanism's interval contains its own point estimate",
        all(
            share.low_bits <= share.loss_bits <= share.high_bits
            for share in report.mechanisms
        ),
        "a percentile interval that excluded its own point estimate would mean the "
        "statistic and the resampling disagree",
    )
    out(
        "  NOTE on 'dominance decided'. The report distinguishes the mechanism with the\n"
        "  largest point estimate from an ordering the data actually decide. A dominance\n"
        "  that is not decided is a reason to measure more handovers, not a reason to buy\n"
        "  hardware, and the package never collapses the two."
    )

    out.rule("2  the residual over every event individually, not only in aggregate")
    out(
        "  An aggregate residual can hide a per-event cancellation. Each event's own\n"
        "  residual is checked against its own tolerance."
    )
    bucketed = trace.dropped_bits.sum(axis=1)
    conserved = trace.conservation_total_bits
    from handoverloss.accounting import residual_tolerance_bits  # noqa: PLC0415

    tolerance = residual_tolerance_bits(
        trace.produced_bits, trace.delivered_bits, bucketed, trace.slot_count
    )
    residual = conserved - bucketed
    ratio = np.abs(residual) / tolerance
    out(f"  events                                  : {len(trace)}")
    out(f"  slot count per event                    : {trace.slot_count}")
    out(f"  worst absolute residual                 : "
        f"{float(np.max(np.abs(residual))):.6e} bits")
    out(f"  worst |residual| / its own tolerance    : {float(np.max(ratio)):.6e}")
    out(f"  median |residual| / its own tolerance   : {float(np.median(ratio)):.6e}")
    out.check(
        "every single event's residual is inside its own tolerance",
        bool(np.all(ratio <= 1.0)),
        f"worst ratio {float(np.max(ratio)):.6e}; {int(np.sum(ratio > 1.0))} event(s) "
        "outside",
    )

    out.rule("3  which mechanism dominates, by operating regime")
    out(
        "  This is the product's substantive claim: the answer changes with the regime,\n"
        "  so a single corpus-wide figure would be misleading. The regimes below are cut\n"
        "  on quantities an operator knows before the handover -- the utilisation\n"
        "  rho = R_s / R_d and the buffer headroom in seconds -- and not on any outcome."
    )
    utilisation = np.asarray(table["utilisation"], dtype=float)
    headroom = np.asarray(table["headroom_s"], dtype=float)
    restart = np.asarray(table["declared_restart_s"], dtype=float)
    out("")
    out(f"  utilisation range in the corpus          : "
        f"[{float(utilisation.min()):.4f}, {float(utilisation.max()):.4f}]")
    out(f"  headroom_s range                         : "
        f"[{float(headroom.min()):.2f}, {float(headroom.max()):.2f}] s")
    out(f"  declared restart range                   : "
        f"[{float(restart.min()):.2f}, {float(restart.max()):.2f}] s")

    out(
        "\n  A STRUCTURAL CORRELATION IN THE CORPUS, STATED BEFORE THE REGIMES. The\n"
        "  generator draws the buffer capacity independently of the source rate, and\n"
        "  headroom_s = (B - q0) / R_s, so a high utilisation implies a high source rate\n"
        "  and therefore a SHORT headroom in seconds. The cell 'high utilisation AND\n"
        "  ample buffer' is consequently empty in this corpus and is reported below as\n"
        "  structurally unmeasurable rather than quietly dropped. A reader wanting that\n"
        "  cell must regenerate the corpus with the buffer capacity scaled to the source\n"
        "  rate; nothing in this repository measures it."
    )
    for label, mask in (
        ("low utilisation (rho < 0.25)", utilisation < 0.25),
        ("high utilisation (rho > 0.75)", utilisation > 0.75),
        ("tight buffer (headroom < 10 s)", headroom < 10.0),
        ("ample buffer (headroom > 120 s)", headroom > 120.0),
        ("slow protocol restart (declared > 30 s)", restart > 30.0),
        ("fast protocol restart (declared < 8 s)", restart < 8.0),
        ("high utilisation AND ample buffer -- the empty cell named above",
         (utilisation > 0.75) & (headroom > 120.0)),
    ):
        _regime_report(out, corpus, mask, label)

    out.rule("4  the dominance mix the ground truth actually has")
    counts = corpus.dominance_counts()
    for name in MECHANISMS:
        out(f"  {name:<14} dominates {counts[name]:5d} of {len(corpus)} events "
            f"({counts[name] / len(corpus):.6f})")
    out.check(
        "every mechanism dominates at least a tenth of the corpus",
        all(value >= 0.10 * len(corpus) for value in counts.values()),
        "a corpus in which one answer is nearly always right cannot distinguish a "
        "predictor from a constant; the generator's stratification grids exist for "
        "this reason and are declared in data/generate_corpus.py",
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
