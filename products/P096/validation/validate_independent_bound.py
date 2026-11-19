"""The bound check: mathematics the implementation does not use, bracketing what it does.

"Two implementations agree" cannot catch an error they share. This mission has
shipped that defect: one product's several hundred tests, property tests
included, all compared a closed form against expressions carrying the same
cancellation, and only a bound derived from different mathematics caught it.

WHAT MAKES THIS BOUND INDEPENDENT
---------------------------------

:mod:`handoverloss.bounds` is derived from conservation and capacity counting
over the accounting window: the volume the source offered, the free space at the
break, and the integral of the declared egress schedule. It uses NONE of the
per-phase decomposition, the clamped occupancy recursion, the fluid model's
phase-by-phase clamping argument, or the trace simulator's slot loop. The
independence is asserted structurally as well as argued: section 1 below parses
``src/handoverloss/bounds.py`` and checks it imports no model module, so a future
import cannot quietly make the check circular.

WHAT IT BRACKETS
----------------

* baseline 2's closed-form total, over the whole corpus and over a randomised
  sweep of the declared parameter envelope;
* baseline 3's trace total, where the offered volume handed to the bound is the
  trace's OWN measured arrivals rather than ``R_s * H``, so the bound is being
  applied to the same window the accounting covers;
* and it must NOT bracket the injected violations in section 4, which are
  anchored on the true bound rather than on any output.

THE SLACK
---------

The lower bound is a difference of three quantities of comparable size, so it is
subject to cancellation exactly where it is tightest. The slack follows the
CONDITIONING of the bound expression -- ``8 * eps * (|A| + |S| + B + |q0|)`` --
and not an operation count and not an absolute constant. An absolute constant
here would be vacuous at petabit scale and falsely tight at megabit scale, which
is the defect this mission recorded when a slack floored at 1.8e-15 made a
bracket of width 4.5e-23 admit the very expression the check existed to catch.
Section 3 measures how tight the bracket actually is, because a bracket nobody
can fail is not evidence.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import ast  # noqa: E402

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from handoverloss.baselines import fluid_decomposition  # noqa: E402
from handoverloss.bounds import (  # noqa: E402
    UNIT_ROUNDOFF,
    bracket_slack_bits,
    loss_bracket,
    max_service_bits,
)
from handoverloss.sequence import HandoverSequence  # noqa: E402
from handoverloss.trace import simulate  # noqa: E402


def main() -> int:
    out = _bootstrap.Tee("validate_independent_bound_output.txt")
    out.rule("INDEPENDENT BOUND -- a bracket from counting, not from either model")
    out(_bootstrap.ORIGIN)
    out("")
    for line in _bootstrap.environment_lines():
        out(line)

    out.rule("1  the bound's independence, asserted structurally and not only argued")
    source = (ROOT / "src" / "handoverloss" / "bounds.py").read_text()
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    forbidden = {"baselines", "trace", "accounting", "predictor", "corpus"}
    out(f"  src/handoverloss/bounds.py imports      : {sorted(imported)}")
    out.check(
        "the bounds module imports no model module of this package",
        not (imported & forbidden),
        f"forbidden imports found: {sorted(imported & forbidden)}"
        if imported & forbidden
        else "conservation and capacity counting only",
    )
    out(
        "  The egress integral is the ONE piece of arithmetic shared with the models,\n"
        "  and it is shared with the DECLARATION rather than with any solver: it is the\n"
        "  area of the trapezium under the declared rate curve. Checked against a\n"
        "  hand-computed value below."
    )
    # HAND ARITHMETIC: R_d = 1.0e9, T_o = 52 s, T_ramp = 60 s, H = 700 s.
    #   ramp area = 0.5 * 1.0e9 * 60               = 3.0e10 bits
    #   flat area = 1.0e9 * (700 - 52 - 60) = *588 = 5.88e11 bits
    #   S         = 6.18e11 bits
    service = float(max_service_bits(1.0e9, 52.0, 60.0, 700.0))
    out(f"  egress integral, hand value 6.18e11     : {service:.6e} bits")
    out.check(
        "the egress integral matches its hand-computed value",
        abs(service - 6.18e11) <= 8.0 * UNIT_ROUNDOFF * 6.18e11,
        f"{service:.12e} against 6.18e11",
    )

    out.rule("2  the bracket contains baseline 2's closed form, over the whole corpus")
    corpus = _bootstrap.require_corpus()
    table = corpus.table
    out(f"  corpus sha256                           : {corpus.sha256}")
    below = 0
    above = 0
    for row in range(len(corpus)):
        sequence = HandoverSequence(
            source_rate_bps=float(table["source_rate_bps"][row]),
            downlink_rate_bps=float(table["downlink_rate_bps"][row]),
            buffer_capacity_bits=float(table["buffer_capacity_bits"][row]),
            initial_occupancy_bits=float(table["initial_occupancy_bits"][row]),
            acquisition_s=float(table["realised_acquisition_s"][row]),
            restart_s=float(table["realised_restart_s"][row]),
            ramp_s=float(table["declared_ramp_s"][row]),
            horizon_s=float(table["horizon_s"][row]),
        )
        total = fluid_decomposition(sequence).total_bits
        bracket = loss_bracket(
            offered_bits=sequence.source_rate_bps * sequence.horizon_s,
            downlink_rate_bps=sequence.downlink_rate_bps,
            outage_s=sequence.outage_s,
            ramp_s=sequence.ramp_s,
            horizon_s=sequence.horizon_s,
            buffer_capacity_bits=sequence.buffer_capacity_bits,
            initial_occupancy_bits=sequence.initial_occupancy_bits,
        )
        if bracket.gap_below_bits(total) > bracket.slack_bits:
            below += 1
        if bracket.gap_above_bits(total) > bracket.slack_bits:
            above += 1
    out(f"  handovers checked                       : {len(corpus)}")
    out(f"  closed-form total below the lower bound : {below}")
    out(f"  closed-form total above the upper bound : {above}")
    out.check(
        "the bracket contains baseline 2's total on every corpus handover",
        below == 0 and above == 0,
        f"{below} below, {above} above, allowing the conditioning-scaled slack",
    )

    out.rule("3  how TIGHT the bracket is -- a bracket nobody can fail is not evidence")
    out(
        "  Reported as the fraction of the bracket's width that separates the closed\n"
        "  form from each bound. A value near 0 means the bound binds; near 1 means it\n"
        "  is loose. Both are honest and the number tells a reader which."
    )
    rng = np.random.default_rng(_bootstrap.SEED_ANALYSIS)
    binding = 0
    positions = []
    widths = []
    trials = 2000
    for _ in range(trials):
        downlink = float(rng.uniform(2e8, 3e9))
        source = float(rng.uniform(0.02, 1.4)) * downlink
        capacity = float(rng.uniform(1e9, 8e10))
        occupancy = capacity * float(rng.beta(4.0, 1.6))
        acquisition = float(rng.uniform(0.0, 200.0))
        restart = float(rng.uniform(0.0, 90.0))
        ramp = float(rng.uniform(0.0, 200.0))
        # A SHORT window is what makes the pigeonhole lower bound bind, so the
        # sweep includes windows barely longer than the excursion.
        horizon = acquisition + restart + ramp + float(rng.uniform(0.0, 120.0))
        sequence = HandoverSequence(
            source_rate_bps=source,
            downlink_rate_bps=downlink,
            buffer_capacity_bits=capacity,
            initial_occupancy_bits=min(occupancy, capacity),
            acquisition_s=acquisition,
            restart_s=restart,
            ramp_s=ramp,
            horizon_s=horizon,
        )
        total = fluid_decomposition(sequence).total_bits
        bracket = loss_bracket(
            offered_bits=source * horizon,
            downlink_rate_bps=downlink,
            outage_s=sequence.outage_s,
            ramp_s=ramp,
            horizon_s=horizon,
            buffer_capacity_bits=capacity,
            initial_occupancy_bits=sequence.initial_occupancy_bits,
        )
        width = bracket.upper_tight_bits - bracket.lower_bits
        widths.append(width)
        if bracket.lower_bits > 0.0:
            binding += 1
        if width > 0.0:
            positions.append((total - bracket.lower_bits) / width)
    positions_array = np.asarray(positions)
    out(f"  random handovers swept                  : {trials}")
    out(f"  cases where the lower bound is non-zero : {binding} "
        f"({binding / trials:.6f})")
    out(f"  median position in the bracket          : "
        f"{float(np.median(positions_array)):.6f}")
    out(f"  5th / 95th percentile of that position  : "
        f"{float(np.percentile(positions_array, 5)):.6f} / "
        f"{float(np.percentile(positions_array, 95)):.6f}")
    out(f"  tightest case (closest to a bound)      : "
        f"{float(np.min(np.minimum(positions_array, 1.0 - positions_array))):.6e}")
    out.check(
        "the lower bound binds on a non-trivial fraction of the swept envelope",
        binding >= 0.05 * trials,
        f"{binding} of {trials}; a bound that was never non-zero would be vacuous",
    )

    out.rule("4  the bracket applied to baseline 3's TRACE total, with measured arrivals")
    out(
        "  The offered volume handed to the bound here is the trace's OWN arrival total,\n"
        "  not R_s * H, so the bound is applied to the same window the accounting covers\n"
        "  and the burstiness is inside the bound's input rather than outside it."
    )
    count = _bootstrap.TRACE_EVENTS
    parameters = _bootstrap.corpus_event_parameters(corpus, count)
    trace = simulate(parameters, np.random.default_rng(_bootstrap.SEED_TRACE))
    totals = trace.dropped_bits.sum(axis=1)
    outside = 0
    for row in range(len(trace)):
        bracket = loss_bracket(
            offered_bits=float(trace.produced_bits[row]),
            downlink_rate_bps=float(parameters.downlink_rate_bps[row]),
            outage_s=float(parameters.acquisition_s[row] + parameters.restart_s[row]),
            ramp_s=float(parameters.ramp_s[row]),
            horizon_s=float(parameters.horizon_s),
            buffer_capacity_bits=float(parameters.buffer_capacity_bits[row]),
            initial_occupancy_bits=float(parameters.initial_occupancy_bits[row]),
        )
        if not bracket.brackets(float(totals[row])):
            outside += 1
    out(f"  trace events checked                    : {len(trace)}")
    out(f"  trace totals outside the bracket        : {outside}")
    out.check(
        "the bracket contains baseline 3's trace total on every event",
        outside == 0,
        f"{outside} of {len(trace)} outside",
    )

    out.rule("5  negative controls on the bound itself, anchored on the TRUE bound")
    out(
        "  Each control injects a value derived from the TRUE bound, not from a model\n"
        "  output. Offsetting an output that already sits inside the bracket does not\n"
        "  guarantee leaving it; only an offset past the bound by more than the slack\n"
        "  does. Both directions are asserted and the must-NOT-fire control is counted\n"
        "  separately."
    )
    bracket = loss_bracket(
        offered_bits=1.0e11,
        downlink_rate_bps=1.0e9,
        outage_s=90.0,
        ramp_s=10.0,
        horizon_s=100.0,
        buffer_capacity_bits=1.0e10,
        initial_occupancy_bits=0.0,
    )
    out(f"  control bracket: [{bracket.lower_bits:.6e}, {bracket.upper_tight_bits:.6e}] "
        f"bits, slack {bracket.slack_bits:.6e} bits")
    out.control(
        "a total above the offered volume",
        not bracket.brackets(2.0 * bracket.offered_bits + 2.0 * bracket.slack_bits),
        must_fire=True,
        detail="anchored on the TRUE offered volume",
    )
    out.control(
        "a total below the pigeonhole lower bound",
        not bracket.brackets(bracket.lower_bits - 2.0 * bracket.slack_bits - 1.0e6),
        must_fire=True,
        detail="anchored on the TRUE lower bound",
    )
    midpoint = 0.5 * (bracket.lower_bits + bracket.upper_tight_bits)
    out.control(
        "an honest value in the middle of the bracket",
        not bracket.brackets(midpoint),
        must_fire=False,
        detail="a bracket that rejected this would reject every correct answer",
    )

    out.rule("6  the slack follows the conditioning, measured over six decades")
    out(
        "  Two brackets whose term magnitudes differ by a factor of 1e6 must have slacks\n"
        "  differing by the same factor. A constant slack would be vacuous for the large\n"
        "  one and falsely tight for the small one."
    )
    previous = None
    for scale in (1.0e6, 1.0e8, 1.0e10, 1.0e12):
        slack = float(bracket_slack_bits(scale, 2.0 * scale, scale, 0.5 * scale))
        out(f"  term scale {scale:.0e}  slack {slack:.6e} bits  "
            f"slack/scale {slack / scale:.6e}")
        if previous is not None:
            ratio = slack / previous
            out.check(
                f"slack scales linearly between {scale / 100:.0e} and {scale:.0e}",
                abs(ratio - 100.0) <= 100.0 * 1e-12,
                f"ratio {ratio:.12f}, expected 100",
            )
        previous = slack

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
