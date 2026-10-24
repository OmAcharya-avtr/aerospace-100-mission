"""The headline check: the simulation against the closed form in the no-outage limit.

The closed form of :mod:`relfiledelivery.analytic` is exact under A1--A4, so in
that regime it is the reference and the simulation is the thing being checked,
not the other way round. The check is run over a declared grid of files, rates,
asymmetries, light times and loss probabilities, and for each cell it reports

* the exact expected delivery time,
* the simulated mean over a declared number of runs,
* the sampling standard error of that mean,
* the discrepancy expressed in units of that standard error.

**The tolerance is the sampling error, not a number chosen to pass.** A
simulated mean cannot agree with an exact expectation more closely than its own
standard error allows, so the only honest tolerance is a multiple of that
error. The threshold is declared before the loop at four standard errors, which
for a two-sided normal gives a false-alarm probability of 6.3e-05 per cell; over
the 24 cells below the expected number of spurious failures is 1.5e-03.

The transmission count is checked in the same pass. Its expectation
``n / (1 - p)`` is exact for every retransmitting mode and does not depend on
the timing model at all, so a cell that agrees on transmissions and disagrees
on time localises the defect to the time accounting.
"""

from __future__ import annotations

from _bootstrap import Tee, bootstrap, environment_lines, relative

bootstrap()

import numpy as np  # noqa: E402

from relfiledelivery.analytic import analytic_delivery, clean_pass_seconds  # noqa: E402
from relfiledelivery.link import LinkConditions  # noqa: E402
from relfiledelivery.modes import TransferMode  # noqa: E402
from relfiledelivery.montecarlo import run_ensemble, standard_error_of_mean  # noqa: E402
from relfiledelivery.payload import FileTransferSpec  # noqa: E402
from relfiledelivery.simulate import simulate_transfer  # noqa: E402

out = Tee("validate_simulation_vs_closed_form_output.txt")

RUNS = 1200
SIGMA_THRESHOLD = 4.0

GRID = [
    # (file bytes, pdu payload, forward bit/s, asymmetry, light time s, loss)
    (40 * 256, 256, 2.0e5, 10.0, 0.0, 0.00),
    (40 * 256, 256, 2.0e5, 10.0, 0.0, 0.05),
    (40 * 256, 256, 2.0e5, 10.0, 0.0, 0.20),
    (40 * 256, 256, 2.0e5, 10.0, 1.2, 0.05),
    (40 * 256, 256, 2.0e5, 100.0, 1.2, 0.05),
    (40 * 256, 256, 2.0e5, 100.0, 1.2, 0.30),
    (120 * 512 - 7, 512, 1.0e6, 10.0, 0.25, 0.02),
    (120 * 512 - 7, 512, 1.0e6, 10.0, 0.25, 0.10),
    (120 * 512 - 7, 512, 1.0e6, 50.0, 2.5, 0.10),
    (120 * 512 - 7, 512, 1.0e6, 50.0, 2.5, 0.25),
    (300 * 1024 + 11, 1024, 5.0e5, 20.0, 0.05, 0.05),
    (300 * 1024 + 11, 1024, 5.0e5, 20.0, 0.05, 0.15),
]

out.rule("environment")
for line in environment_lines():
    out(line)

out.rule("declared check parameters")
out(f"runs per cell                  : {RUNS}")
out(f"tolerance                      : {SIGMA_THRESHOLD:.1f} sampling standard errors")
out("regime                         : no outages, continuous contact, fixed-size NAK")
out("mode                           : deferred_nak (the mode with a closed-form time)")
out(f"grid cells                     : {len(GRID)}")

out.rule("deferred-NAK delivery time: exact expectation against the simulated mean")
header = (
    f"{'n':>5}{'p':>6}{'asym':>7}{'L s':>6}{'exact s':>12}{'sim mean s':>12}"
    f"{'std err':>11}{'diff/se':>9}{'rel diff':>12}{'ok':>5}"
)
out(header)
out("-" * len(header))

worst_sigma = 0.0
worst_relative = 0.0
time_failures: list[str] = []
for file_bytes, payload, forward, asymmetry, light, loss in GRID:
    spec = FileTransferSpec(file_bytes, payload, 12)
    link = LinkConditions(
        forward_rate_bps=forward,
        return_rate_bps=forward / asymmetry,
        one_way_light_time_s=light,
        pdu_loss_probability=loss,
    )
    exact = analytic_delivery(
        spec, link, TransferMode.DEFERRED_NAK
    ).expected_delivery_time_s
    distribution = run_ensemble(spec, link, TransferMode.DEFERRED_NAK, RUNS, 101)
    assert distribution.completed == RUNS
    if loss == 0.0:
        # The lossless case is degenerate: every run gives the same time, so the
        # sampling error is zero and the comparison is an exact-arithmetic one.
        error = 0.0
        sigmas = 0.0
        ok = abs(distribution.mean_s - exact) <= 1e-9 * exact
    else:
        error = standard_error_of_mean(distribution.delivery_times_s)
        sigmas = abs(distribution.mean_s - exact) / error
        ok = sigmas <= SIGMA_THRESHOLD
    relative_diff = (distribution.mean_s - exact) / exact
    worst_sigma = max(worst_sigma, sigmas)
    worst_relative = max(worst_relative, abs(relative_diff))
    if not ok:
        time_failures.append(
            f"n={spec.pdu_count}, p={loss}, asymmetry={asymmetry}, L={light}: "
            f"{sigmas:.2f} standard errors"
        )
    out(
        f"{spec.pdu_count:>5}{loss:>6.2f}{asymmetry:>7.0f}{light:>6.2f}"
        f"{exact:>12.6f}{distribution.mean_s:>12.6f}{error:>11.6f}"
        f"{sigmas:>9.2f}{relative_diff:>+12.3e}{str(ok):>5}"
    )

out("")
out(f"worst discrepancy              : {worst_sigma:.4f} standard errors")
out(f"worst relative discrepancy     : {worst_relative:.6e}")
out(f"cells outside tolerance        : {len(time_failures)} of {len(GRID)}")
for failure in time_failures:
    out(f"  FAILED {failure}")

out.rule("forward transmission count: exact expectation against the simulated mean")
out("Mode-independent in expectation, and independent of the timing model.")
out("")
header = (
    f"{'n':>5}{'p':>6}{'mode':>16}{'exact':>12}{'sim mean':>12}"
    f"{'std err':>11}{'diff/se':>9}{'ok':>5}"
)
out(header)
out("-" * len(header))
worst_tx_sigma = 0.0
tx_failures: list[str] = []
for file_bytes, payload, forward, asymmetry, light, loss in GRID:
    if loss == 0.0:
        continue
    spec = FileTransferSpec(file_bytes, payload, 12)
    link = LinkConditions(
        forward_rate_bps=forward,
        return_rate_bps=forward / asymmetry,
        one_way_light_time_s=light,
        pdu_loss_probability=loss,
    )
    exact = analytic_delivery(
        spec, link, TransferMode.DEFERRED_NAK
    ).expected_transmissions
    for mode in (TransferMode.DEFERRED_NAK, TransferMode.IMMEDIATE_NAK):
        distribution = run_ensemble(spec, link, mode, RUNS, 2024)
        counts = distribution.transmissions
        error = standard_error_of_mean(counts)
        sigmas = abs(float(np.mean(counts)) - exact) / error
        ok = sigmas <= SIGMA_THRESHOLD
        worst_tx_sigma = max(worst_tx_sigma, sigmas)
        if not ok:
            tx_failures.append(f"n={spec.pdu_count}, p={loss}, {mode.value}: {sigmas:.2f} se")
        out(
            f"{spec.pdu_count:>5}{loss:>6.2f}{mode.value:>16}{exact:>12.4f}"
            f"{float(np.mean(counts)):>12.4f}{error:>11.4f}{sigmas:>9.2f}{str(ok):>5}"
        )

out("")
out(f"worst discrepancy              : {worst_tx_sigma:.4f} standard errors")
out(f"cells outside tolerance        : {len(tx_failures)} of 20")
for failure in tx_failures:
    out(f"  FAILED {failure}")

out.rule("immediate-NAK against its analytic lower bound")
out("The closed form gives no delivery-time expression for immediate-NAK, only")
out("the busy-time lower bound 8 * W / ((1 - p) * R_fwd) + L. That is a bound on")
out("the EXPECTATION, because the busy time it contains is itself an expectation.")
out("The first run of this check compared it against the minimum over runs and")
out("reported seven of eleven cells as failures; a lucky run has fewer")
out("transmissions than the expectation, so a realisation legitimately falls")
out("below a bound built from the mean. Two separate statements are checked here")
out("instead.")
out("")
out("  (a) the simulated MEAN delivery time is at least the bound;")
out("  (b) on EVERY run, the idle time is at least one one-way light time,")
out("      which is the per-realisation statement and holds because the")
out("      completing unit still has to propagate.")
out("")
header = (
    f"{'n':>5}{'p':>6}{'L s':>6}{'bound s':>11}{'imm mean s':>12}"
    f"{'(a) ok':>8}{'min idle-L':>12}{'(b) ok':>8}"
)
out(header)
out("-" * len(header))
mean_bound_respected = True
idle_bound_respected = True
for file_bytes, payload, forward, asymmetry, light, loss in GRID:
    if loss == 0.0:
        continue
    spec = FileTransferSpec(file_bytes, payload, 12)
    link = LinkConditions(
        forward_rate_bps=forward,
        return_rate_bps=forward / asymmetry,
        one_way_light_time_s=light,
        pdu_loss_probability=loss,
    )
    bound = analytic_delivery(
        spec, link, TransferMode.IMMEDIATE_NAK
    ).delivery_time_lower_bound_s
    results = [
        simulate_transfer(spec, link, TransferMode.IMMEDIATE_NAK, 55 + index)
        for index in range(400)
    ]
    times = np.array([r.delivery_time_s for r in results], dtype=float)
    idles = np.array([r.idle_s for r in results], dtype=float)
    mean_ok = float(np.mean(times)) >= bound - 1e-9
    idle_slack = float(np.min(idles)) - light
    idle_ok = idle_slack >= -1e-9
    mean_bound_respected = mean_bound_respected and mean_ok
    idle_bound_respected = idle_bound_respected and idle_ok
    out(
        f"{spec.pdu_count:>5}{loss:>6.2f}{light:>6.2f}{bound:>11.6f}"
        f"{float(np.mean(times)):>12.6f}{str(mean_ok):>8}{idle_slack:>+12.3e}{str(idle_ok):>8}"
    )
out("")
out(f"(a) mean at least the bound    : {'PASS' if mean_bound_respected else 'FAIL'}")
out(f"(b) idle at least one light time: {'PASS' if idle_bound_respected else 'FAIL'}")

out.rule("immediate-NAK is not always faster than deferred-NAK")
out("Pipelining the retransmission requests shortens the transfer when the")
out("forward stream is long relative to the round trip, and lengthens it when it")
out("is not: a gap revealed late costs its own two light times, and with many")
out("gaps the sender ends up idling once per gap instead of once per round.")
out("The ratio below is the median immediate-NAK delivery time over the median")
out("deferred-NAK delivery time on the same scenario and the same seeds; above")
out("1.0 means immediate-NAK is slower.")
out("")
header = (
    f"{'n':>5}{'p':>6}{'L s':>6}{'clean pass s':>14}{'L/clean pass':>14}"
    f"{'imm q0.50':>12}{'def q0.50':>12}{'imm/def':>9}"
)
out(header)
out("-" * len(header))
ratios = []
for file_bytes, payload, forward, asymmetry, light, loss in GRID:
    if loss == 0.0:
        continue
    spec = FileTransferSpec(file_bytes, payload, 12)
    link = LinkConditions(
        forward_rate_bps=forward,
        return_rate_bps=forward / asymmetry,
        one_way_light_time_s=light,
        pdu_loss_probability=loss,
    )
    clean = clean_pass_seconds(spec, link)
    immediate = run_ensemble(spec, link, TransferMode.IMMEDIATE_NAK, 400, 55)
    deferred = run_ensemble(spec, link, TransferMode.DEFERRED_NAK, 400, 55)
    ratio = immediate.quantiles[0.50] / deferred.quantiles[0.50]
    ratios.append((ratio, spec.pdu_count, loss, light, light / clean))
    out(
        f"{spec.pdu_count:>5}{loss:>6.2f}{light:>6.2f}{clean:>14.6f}"
        f"{light / clean:>14.4f}{immediate.quantiles[0.50]:>12.6f}"
        f"{deferred.quantiles[0.50]:>12.6f}{ratio:>9.4f}"
    )
out("")
best = min(ratios)
worst = max(ratios)
out(
    f"best for immediate-NAK         : ratio {best[0]:.4f} at n={best[1]}, p={best[2]}, "
    f"L={best[3]} s, L/clean pass {best[4]:.4f}"
)
out(
    f"worst for immediate-NAK        : ratio {worst[0]:.4f} at n={worst[1]}, p={worst[2]}, "
    f"L={worst[3]} s, L/clean pass {worst[4]:.4f}"
)
out(f"cells where immediate-NAK is slower: {sum(1 for r in ratios if r[0] > 1.0)} of {len(ratios)}")
out("")
out("This is a measured property of the declared model under M5, not a property")
out("of any protocol. It is reported because the opposite is the natural")
out("assumption and the measurement contradicts it in part of the grid.")
immediate_slower = sum(1 for r in ratios if r[0] > 1.0)

out.rule("summary")
checks = {
    "deferred-NAK delivery time vs closed form": not time_failures,
    "transmission count vs n / (1 - p)": not tx_failures,
    "immediate-NAK mean respects the analytic lower bound": mean_bound_respected,
    "immediate-NAK idle is at least one light time on every run": idle_bound_respected,
}
for name, passed in checks.items():
    out(f"{'PASS' if passed else 'FAIL'}  {name}")
out("")
out("")
out(f"immediate-NAK slower than deferred-NAK in {immediate_slower} of {len(ratios)} cells")
out(f"output written to {relative(out.path)}")
out.save()
if not all(checks.values()):
    raise SystemExit("validate_simulation_vs_closed_form.py: at least one check FAILED")
