"""Retransmission accounting against two independent counts and the closed form.

Three routes to the same number, and they must agree:

1. the simulator's aggregate forward-transmission counter;
2. the sum of its per-unit attempt array, which is separate bookkeeping;
3. the closed-form expectation ``n / (1 - p)``, which is exact under A1 and
   does not depend on the retransmission schedule at all.

Routes 1 and 2 must agree exactly on every run -- they are integer counts of
the same events. Route 3 is an expectation and is compared against the sample
mean within its sampling error.

The wire-byte accounting is checked the same way: the forward data bytes
recomputed from the attempt array plus the charged control bytes must equal the
aggregate byte counter exactly, and the overhead ratio must never fall below
the lossless floor.
"""

from __future__ import annotations

from _bootstrap import Tee, bootstrap, environment_lines, relative

bootstrap()

import numpy as np  # noqa: E402

from relfiledelivery.analytic import expected_total_transmissions  # noqa: E402
from relfiledelivery.link import LinkConditions, OutageModel, uniform_windows  # noqa: E402
from relfiledelivery.modes import ALL_MODES, TransferMode  # noqa: E402
from relfiledelivery.montecarlo import standard_error_of_mean  # noqa: E402
from relfiledelivery.payload import FileTransferSpec  # noqa: E402
from relfiledelivery.simulate import (  # noqa: E402
    accounting_discrepancy,
    expected_wire_bytes_from_attempts,
    independent_transmission_count,
    simulate_transfer,
)

out = Tee("validate_retransmission_accounting_output.txt")

RUNS = 600
SIGMA_THRESHOLD = 4.0

SPEC = FileTransferSpec(120 * 512 - 7, 512, 12)

CONTINUOUS = {
    0.02: LinkConditions(1.0e6, 1.0e4, 0.25, 0.02),
    0.10: LinkConditions(1.0e6, 1.0e4, 0.25, 0.10),
    0.25: LinkConditions(1.0e6, 1.0e4, 0.25, 0.25),
    0.40: LinkConditions(1.0e6, 1.0e4, 0.25, 0.40),
}

HARD_LINKS = {
    "bursty continuous": LinkConditions(
        1.0e6, 1.0e4, 0.25, 0.05, outage=OutageModel(mean_up_s=1.2, mean_down_s=0.4)
    ),
    "windowed bursty": LinkConditions(
        1.0e6,
        1.0e4,
        0.25,
        0.05,
        outage=OutageModel(mean_up_s=2.0, mean_down_s=0.8),
        windows=uniform_windows(12, 3.0, 6.0),
    ),
    "growing NAK": LinkConditions(1.0e6, 1.0e4, 0.25, 0.15, nak_bytes_per_segment=4.0),
}

out.rule("environment")
for line in environment_lines():
    out(line)

out.rule("check 1 and 2: aggregate counter against the per-unit attempt array")
out("Integer counts of the same events, so they must agree exactly. Checked on")
out("every run across every mode, every loss probability and every hard link,")
out("including runs that did not complete.")
out("")
header = f"{'link':>19}{'mode':>16}{'runs':>7}{'exact matches':>15}{'worst |diff|':>14}"
out(header)
out("-" * len(header))

all_links = {f"continuous p={p:.2f}": link for p, link in CONTINUOUS.items()}
all_links.update(HARD_LINKS)

discrepancies: list[str] = []
checked_runs = 0
for link_name, link in all_links.items():
    for mode in ALL_MODES:
        worst = 0
        matches = 0
        for seed in range(80):
            result = simulate_transfer(SPEC, link, mode, seed)
            checked_runs += 1
            difference = accounting_discrepancy(result)
            if difference == 0:
                matches += 1
            else:  # pragma: no cover - the defect this check exists to catch
                discrepancies.append(f"{link_name}/{mode.value}/seed {seed}: {difference}")
            worst = max(worst, abs(difference))
            assert independent_transmission_count(result) == sum(result.attempts)
        out(f"{link_name:>19}{mode.value:>16}{80:>7}{matches:>15}{worst:>14}")

out("")
out(f"runs checked                   : {checked_runs}")
out(f"discrepancies                  : {len(discrepancies)}")
for item in discrepancies:
    out(f"  FAILED {item}")

out.rule("check 3: sample mean transmission count against n / (1 - p)")
out("Exact under A1, mode-independent, and independent of the timing model.")
out("Only the outage-free continuous links are in scope: with outages the sender")
out("transmits into a dead link (M3) and the expectation no longer applies, which")
out("is measured separately below.")
out("")
header = (
    f"{'p':>6}{'mode':>16}{'exact':>12}{'sample mean':>13}{'std err':>10}"
    f"{'diff/se':>9}{'ok':>5}"
)
out(header)
out("-" * len(header))
worst_sigma = 0.0
count_failures: list[str] = []
for p, link in CONTINUOUS.items():
    exact = expected_total_transmissions(SPEC.pdu_count, p)
    for mode in (TransferMode.DEFERRED_NAK, TransferMode.IMMEDIATE_NAK):
        counts = np.array(
            [
                simulate_transfer(SPEC, link, mode, 9000 + index).forward_transmissions
                for index in range(RUNS)
            ],
            dtype=float,
        )
        error = standard_error_of_mean(counts)
        sigmas = abs(float(np.mean(counts)) - exact) / error
        ok = sigmas <= SIGMA_THRESHOLD
        worst_sigma = max(worst_sigma, sigmas)
        if not ok:
            count_failures.append(f"p={p}, {mode.value}: {sigmas:.2f} se")
        out(
            f"{p:>6.2f}{mode.value:>16}{exact:>12.4f}{float(np.mean(counts)):>13.4f}"
            f"{error:>10.4f}{sigmas:>9.2f}{str(ok):>5}"
        )
out("")
out(f"worst discrepancy              : {worst_sigma:.4f} standard errors")
out(f"tolerance                      : {SIGMA_THRESHOLD:.1f} standard errors")

out.rule("how far the closed-form overhead is from the measured overhead with outages")
out("Under M3 the sender keeps transmitting during an outage, so the obvious")
out("guess is that the transmission count inflates by one over the outage duty")
out("cycle. Measured below against the closed-form expectation, as a median and")
out("as a mean, with the naive factor beside them. The fraction of runs that saw")
out("no outage at all is the column that explains the answer.")
out("")
header = (
    f"{'link':>19}{'duty':>8}{'exact':>10}{'median':>10}{'mean':>10}"
    f"{'med/exact':>11}{'mean/exact':>12}{'1/duty':>9}{'no outage':>11}"
)
out(header)
out("-" * len(header))
inflation_rows = []
for link_name, link in HARD_LINKS.items():
    exact = expected_total_transmissions(SPEC.pdu_count, link.pdu_loss_probability)
    results = [
        simulate_transfer(SPEC, link, TransferMode.DEFERRED_NAK, 7000 + index)
        for index in range(400)
    ]
    counts = np.array([r.forward_transmissions for r in results], dtype=float)
    clean_fraction = float(np.mean([r.outage_seconds == 0.0 for r in results]))
    median = float(np.median(counts))
    mean = float(np.mean(counts))
    duty = link.outage.duty_cycle
    inflation_rows.append((link_name, mean / exact, 1.0 / duty, clean_fraction))
    out(
        f"{link_name:>19}{duty:>8.4f}{exact:>10.2f}{median:>10.2f}{mean:>10.2f}"
        f"{median / exact:>11.4f}{mean / exact:>12.4f}{1.0 / duty:>9.4f}"
        f"{clean_fraction:>11.4f}"
    )
out("")
out("The naive one-over-duty-cycle factor over-predicts on every row here, and")
out("the reason is in the last column: the clean-pass time of this file on this")
out("link is about half a second while the mean outage-free interval is one to")
out("two seconds, so a large fraction of runs never meets an outage at all and")
out("the ones that do pay far more than their share. An inflation factor derived")
out("from a duty cycle assumes the outage process is fine-grained relative to the")
out("transfer; when it is not, the duty cycle describes the long run and the")
out("transfer does not live in the long run.")
out("")
for name, measured_factor, naive, _clean in inflation_rows:
    if naive > 1.0:
        out(
            f"  {name:<19} measured {measured_factor:.4f}, naive {naive:.4f}, "
            f"naive over-predicts by {100.0 * (naive / measured_factor - 1.0):.1f} per cent"
        )
out("")
out("This is the same failure the analytic delivery-time extension has, measured")
out("on the transmission count instead of on the time, and it is the single")
out("clearest reason the learned predictor earns anything at all in the")
out("correlated-outage regime.")

out.rule("wire-byte accounting")
out("Forward data bytes recomputed from the attempt array, plus the charged")
out("control bytes, must equal the aggregate byte counter exactly.")
out("")
header = (
    f"{'link':>19}{'mode':>16}{'runs':>7}{'byte matches':>14}"
    f"{'min overhead':>14}{'floor':>10}{'ok':>5}"
)
out(header)
out("-" * len(header))
byte_failures: list[str] = []
floor = SPEC.minimum_overhead_ratio
for link_name, link in all_links.items():
    for mode in ALL_MODES:
        matches = 0
        min_overhead = float("inf")
        for seed in range(40):
            result = simulate_transfer(SPEC, link, mode, seed)
            control = (result.rounds - 1) * link.eof_bytes if mode.uses_return_link else 0
            recomputed = expected_wire_bytes_from_attempts(SPEC, result) + control
            if recomputed == result.forward_wire_bytes:
                matches += 1
            else:  # pragma: no cover - the defect this check exists to catch
                byte_failures.append(f"{link_name}/{mode.value}/seed {seed}")
            if result.delivered:
                min_overhead = min(min_overhead, result.overhead_ratio)
        ok = matches == 40 and (min_overhead == float("inf") or min_overhead >= floor - 1e-12)
        shown = "n/a" if min_overhead == float("inf") else f"{min_overhead:.6f}"
        out(
            f"{link_name:>19}{mode.value:>16}{40:>7}{matches:>14}"
            f"{shown:>14}{floor:>10.6f}{str(ok):>5}"
        )
out("")
out(f"byte-accounting mismatches     : {len(byte_failures)}")
out(f"lossless overhead floor        : {floor:.9f}")

out.rule("summary")
checks = {
    "aggregate counter equals the per-unit attempt sum on every run": not discrepancies,
    "sample mean transmission count matches n / (1 - p)": not count_failures,
    "wire-byte accounting closes exactly": not byte_failures,
}
for name, passed in checks.items():
    out(f"{'PASS' if passed else 'FAIL'}  {name}")
out("")
out(f"output written to {relative(out.path)}")
out.save()
if not all(checks.values()):
    raise SystemExit("validate_retransmission_accounting.py: at least one check FAILED")
