"""The worked example quoted in README.md, run so that the quote is the output.

One operator question, answered end to end: a 2 MiB product has to come down
over a link with 8 per cent residual loss, a 2.5 second one-way light time, a
100:1 asymmetry, outages averaging 20 seconds at an 80 per cent duty cycle, and
six 30-second contacts every 180 seconds. How long will it take, how much
overhead will it cost, and by which pass is it likely to be down?
"""

from __future__ import annotations

from _bootstrap import Tee, bootstrap, environment_lines, relative

bootstrap()

from relfiledelivery import (  # noqa: E402
    FileTransferSpec,
    LinkConditions,
    OutageModel,
    TransferMode,
    analytic_delivery,
    extended_expected_delivery_time,
    run_ensemble,
    simulate_transfer,
    uniform_windows,
)

out = Tee("worked_example_output.txt")

out.rule("environment")
for line in environment_lines():
    out(line)

# --- the worked example itself, 24 lines of real API -------------------------
spec = FileTransferSpec(file_bytes=2 * 1024 * 1024, pdu_payload_bytes=1024, pdu_header_bytes=12)

link = LinkConditions(
    forward_rate_bps=2.0e6,
    return_rate_bps=2.0e4,
    one_way_light_time_s=2.5,
    pdu_loss_probability=0.08,
    outage=OutageModel(mean_up_s=80.0, mean_down_s=20.0),
    windows=uniform_windows(count=6, duration_s=30.0, period_s=180.0),
)

ideal = analytic_delivery(spec, link.without_outages(), TransferMode.DEFERRED_NAK)
extended = extended_expected_delivery_time(spec, link, TransferMode.DEFERRED_NAK)
measured = run_ensemble(spec, link, TransferMode.DEFERRED_NAK, runs=600, seed=20261024)

out.rule("the question")
for line in spec.describe():
    out(line)
for line in link.describe():
    out(line)

out.rule("step 1: the closed form, on the link with its outages removed")
out("Exact under A1-A4. This is the floor: what the transfer would cost if the")
out("link never dropped and contact never broke.")
out("")
for line in ideal.describe():
    out(line)

out.rule("step 2: the analytic extension, past the closed form's own assumptions")
out("Two steps: divide by the outage duty cycle to get the contact time the")
out("transfer needs, then lay that contact time over the declared windows in")
out("order to get the wall-clock time at which it runs out.")
out("")
out(f"outage duty cycle                         : {link.outage.duty_cycle:.6f}")
out(f"contact duty cycle                        : {link.contact_duty_cycle:.6f}")
out(f"closed-form E[T] with no outages          : {ideal.expected_delivery_time_s:.4f} s")
out(
    f"contact time needed: E[T] / outage duty   : "
    f"{ideal.expected_delivery_time_s / link.outage.duty_cycle:.4f} s"
)
out(f"declared window duration                  : {link.windows[0].duration_s:.4f} s")
out(f"extension, laid over the contact plan     : {extended:.4f} s")
out("")
out("The extension is not a theorem. It ignores that an outage destroys units")
out("already in flight, that a round boundary straddling an outage pays its")
out("turnaround twice, and that a unit cannot straddle a contact boundary.")
out("")
out("An earlier version of this function divided by the product of the two duty")
out("cycles instead. On this scenario with 300-second contacts that gave 164.57 s")
out("against a measured median of 26.80 s -- a factor of 6.14 -- because the")
out("transfer fitted inside the first contact and the contact duty cycle never")
out("bit at all. The learned model is benchmarked against this extension, and a")
out("baseline that wrong would have made the comparison meaningless.")

out.rule("step 3: the measurement")
for line in measured.describe():
    out(line)

out.rule("step 4: what the extension got wrong")
median = measured.quantiles[0.50]
q95 = measured.quantiles[0.95]
out(f"analytic extension                        : {extended:.4f} s")
out(f"measured median                           : {median:.4f} s")
out(f"measured q0.95                            : {q95:.4f} s")
out(f"extension / measured median               : {extended / median:.4f}")
out(f"measured q0.95 / measured median          : {q95 / median:.4f}")
out(f"measured maximum / measured median        : {measured.tail['max_over_median']:.4f}")
out("")
needed = ideal.expected_delivery_time_s / link.outage.duty_cycle
window = link.windows[0].duration_s
out(f"contact time the extension needs          : {needed:.4f} s")
out(f"declared contact window                   : {window:.4f} s")
out(f"needed / window                           : {needed / window:.4f}")
out(f"runs that finished in the first contact   : {measured.completion_by_contact[0]:.4f}")
out("")
out("The extension is a step function of the contact plan, and this scenario sits")
out("just on the wrong side of a step. The contact time it needs exceeds the")
out("declared window by six per cent, so it deposits the whole transfer into the")
out("second contact and reports a figure one window period larger. The simulation")
out("says 82.5 per cent of runs finish inside the first contact. The extension is")
out("not wrong about the arithmetic; it is wrong because it has one number where")
out("the answer is a distribution, and a distribution straddling a window edge is")
out("exactly where that fails hardest.")
out("")
out("The operational reading is the spread, not the centre. The median is 26.80 s")
out("and the ninety-fifth percentile is 205.18 s, a factor of 7.66, because a run")
out("that misses the first contact waits for the next one. A plan written against")
out("any single number -- the median, the mean, or the closed-form extension --")
out("cannot carry that, and the per-contact completion vector in step 3 is the")
out("shape the answer actually has.")

out.rule("step 5: one run in full, including the hash check")
single = simulate_transfer(spec, link, TransferMode.DEFERRED_NAK, seed=20261024)
for line in single.describe():
    out(line)
out("")
out(f"source digest  : {single.source_digest}")
out(f"delivered digest: {single.delivered_digest}")
out(f"digests equal  : {single.source_digest == single.delivered_digest}")
out("")
out("The digests are compared inside the simulator on every completed run. A")
out("mismatch raises CompletionInvariantError and no delivery time is reported.")

out.rule("step 6: all three modes, so the choice has a number against it")
out("")
from relfiledelivery.report import mode_comparison  # noqa: E402

distributions = {
    mode: run_ensemble(spec, link, mode, runs=300, seed=777) for mode in TransferMode
}
for line in mode_comparison(distributions):
    out(line)
out("")
out("Unacknowledged mode never completes this transfer: 2048 units at 8 per cent")
out("loss gives a completion probability of (1 - 0.08) ** 2048, which is")
out(f"{(1 - 0.08) ** 2048:.3e}. That is the row worth reading.")

out.rule("summary")
checks = {
    "the ensemble completed at least one run": measured.completed > 0,
    "the single run delivered and matched its source digest": (
        single.delivered and single.delivered_digest == single.source_digest
    ),
    "the measured median exceeds the no-outage closed form": (
        median > ideal.expected_delivery_time_s
    ),
}
for name, passed in checks.items():
    out(f"{'PASS' if passed else 'FAIL'}  {name}")
out("")
out(f"output written to {relative(out.path)}")
out.save()
if not all(checks.values()):
    raise SystemExit("worked_example.py: at least one check FAILED")
