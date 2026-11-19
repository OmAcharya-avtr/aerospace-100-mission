"""The README's worked example, executed, so the README cannot drift from the code.

Every line of output this script prints appears verbatim in ``README.md`` under
"A worked example". Re-running this script after any code change and pasting its
output back is the last step of every build of this product, because two products
of this mission caught their own wrong README numbers exactly that way.

The example is also the one case in this repository whose arithmetic is checkable
by hand, which is why those numbers and not others were chosen. The hand
derivation is in ``tests/test_baselines.py::test_fluid_decomposition_known_answer``
and is reproduced in the comments below.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from handoverloss import (  # noqa: E402
    HandoverSequence,
    acquisition_only_loss_bits,
    attribute,
    fluid_decomposition,
    loss_bracket,
)
from handoverloss.trace import EventParameters, simulate  # noqa: E402

#: The declared handover. Chosen so every term is hand-checkable:
#:   headroom      = 3.2e10 - 2.9e10            = 3.0e9  bits
#:   L_acq         = 2.0e8 * 40 - 3.0e9         = 5.0e9  bits
#:   L_rst         = 2.0e8 * 12                 = 2.4e9  bits
#:   ramp drive    = (2.0e8)^2 * 60 / (2 * 1e9) = 1.2e9  bits
#:   L_drain       = 1.2e9                      = 1.2e9  bits
#:   total                                      = 8.6e9  bits
SEQUENCE = dict(
    source_rate_bps=2.0e8,
    downlink_rate_bps=1.0e9,
    buffer_capacity_bits=3.2e10,
    initial_occupancy_bits=2.9e10,
    acquisition_s=40.0,
    restart_s=12.0,
    ramp_s=60.0,
    horizon_s=700.0,
)

#: Realisations of the same declared handover used for baseline 3.
REALISATIONS = 96 if _bootstrap.FAST else 512


def main() -> int:
    out = _bootstrap.Tee("worked_example_output.txt")
    out.rule("WORKED EXAMPLE -- the README's example, executed")
    out(_bootstrap.ORIGIN)
    out("")
    for line in _bootstrap.environment_lines():
        out(line)

    sequence = HandoverSequence(**SEQUENCE)
    out.rule("the declared handover")
    out(f"  source rate            {sequence.source_rate_bps:.6e} bit/s")
    out(f"  downlink rate          {sequence.downlink_rate_bps:.6e} bit/s")
    out(f"  buffer capacity        {sequence.buffer_capacity_bits:.6e} bits")
    out(f"  occupancy at the break {sequence.initial_occupancy_bits:.6e} bits")
    out(f"  acquisition            {sequence.acquisition_s:.1f} s")
    out(f"  protocol restart       {sequence.restart_s:.1f} s")
    out(f"  egress ramp            {sequence.ramp_s:.1f} s")
    out(f"  accounting window      {sequence.horizon_s:.1f} s")
    out("")
    out(f"  derived: outage        {sequence.outage_s:.1f} s")
    out(f"  derived: headroom      {sequence.headroom_bits:.6e} bits "
        f"= {sequence.headroom_s:.1f} s of source output")
    out(f"  derived: utilisation   {sequence.utilisation:.4f}")

    out.rule("baseline 1 -- the assumption that acquisition time is the whole story")
    only = acquisition_only_loss_bits(sequence)
    out(f"  loss                   {only:.6e} bits = {only / 1e9:.6f} Gbit")
    out("  hand check: max(0, 2.0e8 * 40 - 3.0e9) = 5.0e9 bits")

    out.rule("baseline 2 -- the analytic fluid decomposition, closed form")
    decomposition = fluid_decomposition(sequence)
    for name, value in decomposition.mechanisms.items():
        out(f"  {name:<14}         {value:.6e} bits = {value / 1e9:.6f} Gbit")
    out(f"  {'total':<14}         {decomposition.total_bits:.6e} bits "
        f"= {decomposition.total_bits / 1e9:.6f} Gbit")
    out(f"  dominant               {decomposition.dominant}")
    out(f"  ramp drive             {decomposition.ramp_drive_bits:.6e} bits")
    out(f"  backlog drains         {decomposition.drains}")
    out(f"  sum residual           {decomposition.sum_residual_bits:.6e} bits")
    out("")
    out("  BASELINE 1 UNDERSTATES BASELINE 2 BY A FACTOR OF "
        f"{decomposition.total_bits / only:.6f} on this handover, and the entire")
    out("  difference is the two mechanisms a link budget does not contain: the protocol")
    out(f"  restart ({decomposition.restart_bits / 1e9:.6f} Gbit) and the buffer drain "
        f"({decomposition.drain_bits / 1e9:.6f} Gbit).")

    out.rule("the independently derived bracket on the total")
    bracket = loss_bracket(
        offered_bits=sequence.source_rate_bps * sequence.horizon_s,
        downlink_rate_bps=sequence.downlink_rate_bps,
        outage_s=sequence.outage_s,
        ramp_s=sequence.ramp_s,
        horizon_s=sequence.horizon_s,
        buffer_capacity_bits=sequence.buffer_capacity_bits,
        initial_occupancy_bits=sequence.initial_occupancy_bits,
    )
    out("  From conservation and capacity counting alone. Neither baseline is used.")
    out(f"  offered volume A       {bracket.offered_bits:.6e} bits")
    out(f"  egress integral S      {bracket.max_service_bits:.6e} bits")
    out(f"  lower bound            {bracket.lower_bits:.6e} bits")
    out(f"  upper bound (tail drop) {bracket.upper_tight_bits:.6e} bits")
    out(f"  slack                  {bracket.slack_bits:.6e} bits")
    out(f"  brackets the closed-form total: "
        f"{bracket.brackets(decomposition.total_bits)}")

    out.rule("baseline 3 -- ground truth over simulated realisations, with intervals")
    out(f"  {REALISATIONS} realisations of the SAME declared handover. The spread comes")
    out("  from the per-slot source burstiness and service jitter, which the closed form")
    out("  does not contain.")
    parameters = EventParameters.from_sequences([sequence] * REALISATIONS)
    trace = simulate(parameters, np.random.default_rng(_bootstrap.SEED_TRACE))
    report = attribute(
        trace, resamples=_bootstrap.RESAMPLES, confidence=_bootstrap.CONFIDENCE
    )
    out("")
    for line in report.lines():
        out(f"  {line}")
    out("")
    ratio = report.total_loss_bits / decomposition.total_bits
    out(f"  ground truth / closed form: {ratio:.6f}")
    out.check(
        "the closed form is within 10 per cent of the ground truth on this handover",
        0.90 <= ratio <= 1.10,
        f"ratio {ratio:.6f}",
    )
    out.check(
        "the independently derived bracket contains the ground-truth total",
        bracket.brackets(report.total_loss_bits),
        f"{report.total_loss_bits:.6e} bits in "
        f"[{bracket.lower_bits:.6e}, {bracket.upper_tight_bits:.6e}]",
    )
    out.check(
        "the decomposition residual is inside its conditioning-scaled tolerance",
        report.residual_within_tolerance,
        f"{report.residual_bits:.6e} bits against {report.residual_tolerance_bits:.6e}",
    )

    out.rule("what an operator does with this")
    ranked = report.ranked()
    out(f"  The dominant mechanism is {ranked[0].name}, at "
        f"{ranked[0].share:.4f} of the total,")
    out(f"  ahead of {ranked[1].name} at {ranked[1].share:.4f}. The ordering is "
        f"{'DECIDED' if report.dominance_is_decided else 'NOT decided'} by these events:")
    out(f"  the leading interval is [{ranked[0].low_gbit:.6f}, {ranked[0].high_gbit:.6f}] "
        f"Gbit and the")
    out(f"  runner-up's is [{ranked[1].low_gbit:.6f}, {ranked[1].high_gbit:.6f}] Gbit.")
    out("")
    out("  On this handover the money goes on faster acquisition, not more buffer. The")
    out(f"  drain mechanism accounts for {report.share('drain').share:.4f} of the loss, so")
    out("  doubling the buffer would recover a small fraction of what halving the")
    out("  acquisition time would. A total-loss figure alone would not have said which.")

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
