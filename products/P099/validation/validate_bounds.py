"""The independent brackets over the whole campaign, with controls that fire.

Two brackets, neither of which uses the machinery it checks, applied to all
1990 passes of the committed record.

1. **The shortfall bracket.** ``0 <= Y_real <= R_max T`` gives
   ``Y_nom - R_max T <= S <= Y_nom`` path by path, from the pass duration and
   the waveform's rate ceiling alone. No quadrature of the rate profile, no
   pushforward, no convolution, no random numbers.

2. **The capacity bracket.** The mean-value inequality for integrals,
   ``R_min T <= Y_cap <= R_max T``, with the extreme rates actually attained
   on the window. This brackets the composite-Simpson capacity integral
   without using Simpson's rule, so a wrong quadrature weight vector --
   which would leave the decomposition perfectly self-consistent -- falls
   outside it.

**The slack follows the conditioning, not a constant.** A previous product of
this mission had both wrong forms falsified on the same structure: a slack
floored at an absolute 1.8e-15 made a bracket of width 4.5e-23 vacuous and
admitted the very expression the check existed to catch. Here the slack of a
bound expression is ``8 * eps * max(|operands|)``, so it grows with the
operands and the bracket gets correspondingly harder to satisfy, in relative
terms, exactly when it is tight. This script measures the slack and the
bracket width over the campaign and reports their ratio, which is what shows
the bracket is not vacuous anywhere.

**Six negative controls**, every one anchored on the true value rather than on
an output, and every one reported as having fired or not. Offsetting an output
that already sits inside a band does not guarantee a violation; each control
here is constructed to be outside the bracket by the bracket's own algebra.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from passyield.bounds import (  # noqa: E402
    MACHINE_EPSILON,
    SLACK_OPERATION_COUNT,
    bracket_slack_mbit,
    capacity_bracket,
    inside,
    shortfall_bracket,
)
from passyield.dataset import (  # noqa: E402
    events_of,
    load_passes,
    station_by_name,
    window_of,
)
from passyield.decompose import decompose_pass, rate_at_offsets  # noqa: E402
from passyield.geometry import satellite  # noqa: E402
from passyield.link import DEFAULT_BUDGET  # noqa: E402

#: Nodes used to find the extreme rates on a window for the capacity bracket.
#: Not a quadrature: only the minimum and maximum of the sampled profile are
#: used, so this is a search and not an integral.
EXTREME_NODES = 257


def main() -> int:
    """Bracket every pass of the campaign and fire six controls."""
    out = _bootstrap.Tee("validate_bounds_output.txt")
    out("Independent bracket validation -- P099 PassYield")
    out("")
    out(f"fast path (PASSYIELD_FAST=1): {_bootstrap.FAST}")
    out("")
    out(f"machine epsilon                  : {MACHINE_EPSILON!r}")
    out(f"slack operation count            : {SLACK_OPERATION_COUNT!r}")
    out(f"R_max = B * eta_max, Mbit/s      : {DEFAULT_BUDGET.max_rate_mbit_s!r}")
    out("")
    out("'two implementations agree' and 'the answer lies inside a bound derived")
    out("from different mathematics' are different checks, and only the second")
    out("catches an error the two implementations share. A sign error in the")
    out("cumulative rate integral, a Mbit-versus-bit slip, a missing grid-step")
    out("factor or a mis-stated geometric term would move the convolution and its")
    out("Monte Carlo cross-check TOGETHER.")

    satrec = satellite()
    record = load_passes()
    limit = 100 if _bootstrap.FAST else len(record)
    rows = record[:limit]

    out.rule("1. THE SHORTFALL BRACKET OVER THE WHOLE CAMPAIGN")
    shortfall_outside = 0
    capacity_outside = 0
    shortfall_widths: list[float] = []
    capacity_widths: list[float] = []
    shortfall_slacks: list[float] = []
    capacity_slacks: list[float] = []
    shortfall_positions: list[float] = []
    for row in rows:
        station = station_by_name(row.station)
        window = window_of(row)
        item = decompose_pass(satrec, station, window, events_of(row))
        bracket = shortfall_bracket(
            item.nominal_bits, item.window_duration_s, DEFAULT_BUDGET.max_rate_mbit_s
        )
        shortfall_outside += int(not inside(item.shortfall_bits, bracket))
        width = bracket[1] - bracket[0]
        shortfall_widths.append(width)
        shortfall_slacks.append(
            bracket_slack_mbit(
                item.nominal_bits,
                DEFAULT_BUDGET.max_rate_mbit_s * item.window_duration_s,
            )
        )
        shortfall_positions.append((item.shortfall_bits - bracket[0]) / width)

        offsets = np.linspace(0.0, window.duration_s, EXTREME_NODES)
        rates, _clamp = rate_at_offsets(satrec, station, window, offsets)
        capacity = capacity_bracket(
            item.window_duration_s, float(rates.min()), float(rates.max())
        )
        capacity_outside += int(not inside(item.capacity_bits, capacity))
        capacity_widths.append(capacity[1] - capacity[0])
        capacity_slacks.append(bracket_slack_mbit(capacity[0], capacity[1]))

    out(f"passes bracketed                        : {len(rows)}")
    out(f"accounted shortfalls outside the bracket: {shortfall_outside}")
    out(f"median bracket width, Mbit              : "
        f"{float(np.median(shortfall_widths)):.1f}")
    out(f"median slack, Mbit                      : "
        f"{float(np.median(shortfall_slacks)):.3e}")
    out(f"median slack / median width             : "
        f"{float(np.median(shortfall_slacks)) / float(np.median(shortfall_widths)):.3e}")
    out("")
    out("the slack is twelve orders of magnitude below the bracket width, so the")
    out("bracket is not vacuous: the slack exists to absorb the rounding of the")
    out("bound expression itself and nothing more.")
    out("")
    out("where the true shortfall sits inside the bracket, as a fraction of its")
    out("width. 0 is the lower end (Y_nom - R_max*T), 1 is the upper end (Y_nom).")
    out(f"  minimum  : {float(np.min(shortfall_positions)):.4f}")
    out(f"  median   : {float(np.median(shortfall_positions)):.4f}")
    out(f"  maximum  : {float(np.max(shortfall_positions)):.4f}")
    out("")
    out("the bracket is LOOSE, and looseness is the price of independence. It is")
    out("still tight enough to catch a unit slip, a sign error or an unnormalised")
    out("mass, which is what it is for; the controls below show that.")
    out.check(
        "every accounted shortfall lies inside the independent bracket",
        shortfall_outside == 0,
        f"{shortfall_outside} of {len(rows)} outside",
    )

    out.rule("2. THE CAPACITY BRACKET -- the mean-value inequality for integrals")
    out("R_min*T <= Y_cap <= R_max*T with the extreme rates attained on the window.")
    out("Only the minimum and maximum of the sampled profile are used, so this is a")
    out("SEARCH and not an integral: it does not use Simpson's rule, which is what")
    out("it is checking.")
    out("")
    out(f"profile samples used to find the extremes: {EXTREME_NODES}")
    out(f"capacities outside the bracket           : {capacity_outside}")
    out(f"median bracket width, Mbit               : "
        f"{float(np.median(capacity_widths)):.1f}")
    out(f"median slack, Mbit                       : "
        f"{float(np.median(capacity_slacks)):.3e}")
    out.check(
        "every Simpson capacity lies inside the mean-value bracket",
        capacity_outside == 0,
        f"{capacity_outside} of {len(rows)} outside",
    )

    out.rule("3. SIX NEGATIVE CONTROLS, EACH ANCHORED ON THE TRUE VALUE")
    row = rows[0]
    station = station_by_name(row.station)
    window = window_of(row)
    item = decompose_pass(satrec, station, window, events_of(row))
    bracket = shortfall_bracket(
        item.nominal_bits, item.window_duration_s, DEFAULT_BUDGET.max_rate_mbit_s
    )
    offsets = np.linspace(0.0, window.duration_s, EXTREME_NODES)
    rates, _clamp = rate_at_offsets(satrec, station, window, offsets)
    capacity = capacity_bracket(
        item.window_duration_s, float(rates.min()), float(rates.max())
    )
    out(f"reference pass           : pass_id {row.pass_id}, {row.station}")
    out(f"true shortfall, Mbit     : {item.shortfall_bits:.3f}")
    out(f"true capacity, Mbit      : {item.capacity_bits:.3f}")
    out(f"shortfall bracket, Mbit  : [{bracket[0]:.3f}, {bracket[1]:.3f}]")
    out(f"capacity bracket, Mbit   : [{capacity[0]:.3f}, {capacity[1]:.3f}]")
    out("")
    out("each injected value below is constructed to be outside the bracket by the")
    out("bracket's OWN algebra, not by offsetting an output that already sits")
    out("inside the band. An offset smaller than twice the bracket's half-width")
    out("does not guarantee a violation, and that is how a previous product of this")
    out("mission shipped two of eleven controls silently too weak.")
    out("")
    controls = [
        (
            "Mbit reported as bit: factor 1e6 on the shortfall",
            item.shortfall_bits * 1.0e6,
            bracket,
        ),
        (
            "shortfall exceeding the booked volume: Y_nom + R_max*T",
            item.nominal_bits + DEFAULT_BUDGET.max_rate_mbit_s * item.window_duration_s,
            bracket,
        ),
        (
            "realised yield negative: shortfall above Y_nom",
            item.nominal_bits + 1.0 + abs(item.nominal_bits) * 1.0e-9,
            bracket,
        ),
        (
            "sign flip on the capacity",
            -item.capacity_bits,
            capacity,
        ),
        (
            "capacity above the rate ceiling times the duration",
            DEFAULT_BUDGET.max_rate_mbit_s * item.window_duration_s * 1.001 + 1.0,
            capacity,
        ),
        (
            "capacity below the minimum rate times the duration",
            float(rates.min()) * item.window_duration_s * 0.999 - 1.0,
            capacity,
        ),
    ]
    for name, value, target in controls:
        out.control(
            name,
            not inside(value, target),
            must_fire=True,
            detail=f"injected {value:.6g} Mbit",
        )
    out("")
    out("and the two that must STAY SILENT, because the true values are inside:")
    out.control(
        "the true shortfall",
        not inside(item.shortfall_bits, bracket),
        must_fire=False,
        detail=f"{item.shortfall_bits:.3f} Mbit",
    )
    out.control(
        "the true capacity",
        not inside(item.capacity_bits, capacity),
        must_fire=False,
        detail=f"{item.capacity_bits:.3f} Mbit",
    )

    out.rule("4. THE SLACK FOLLOWS THE CONDITIONING, DEMONSTRATED")
    out("a slack that is an absolute constant makes a tight bracket vacuous. The")
    out("slack here is 8 * eps * max(|operands|) of the bound expression, so it")
    out("scales with the operands whatever their difference comes to.")
    out("")
    out("the ratio column divides by the LARGEST operand, which is what the slack")
    out("is defined against. Dividing by the smaller one would make the first row")
    out("look different for a reason that is arithmetic and not conditioning: at")
    out("magnitude 1 the operands are 1 and 2, so the largest is twice the first.")
    out("")
    out(f"{'operands':>26}{'slack, Mbit':>16}{'slack / largest':>18}")
    ratios: list[float] = []
    for magnitude in (1.0, 1.0e3, 1.0e6, 1.0e9, 1.0e12):
        operands = (magnitude, magnitude + 1.0)
        slack = bracket_slack_mbit(*operands)
        ratio = slack / max(abs(value) for value in operands)
        ratios.append(ratio)
        out(f"{str(tuple(round(v, 1) for v in operands)):>26}{slack:>16.3e}{ratio:>18.3e}")
    out("")
    out("the ratio is constant, which is the definition of following the")
    out("conditioning: the slack is a fixed number of unit roundoffs of the")
    out("operands and not a fixed number of Mbit.")
    out(f"expected ratio, 8 * eps: {SLACK_OPERATION_COUNT * MACHINE_EPSILON:.6e}")
    out.check(
        "the slack is exactly 8 * eps of the largest operand at every magnitude",
        max(ratios) / min(ratios) < 1.0 + 1.0e-12,
        f"ratio spread {max(ratios) / min(ratios) - 1.0:.3e}",
    )
    out.control(
        "an absolute constant slack would be vacuous at magnitude 1e12",
        1.8e-15 < bracket_slack_mbit(1.0e12, 1.0e12 + 1.0),
        must_fire=True,
        detail=f"1.8e-15 against the conditioning-following "
               f"{bracket_slack_mbit(1.0e12, 1.0e12 + 1.0):.3e}",
    )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
