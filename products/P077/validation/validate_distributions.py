"""Delivery-time distributions as quantiles, with the tail stated separately.

This script exists to substantiate one claim in the README: **a mean delivery
time over an intermittent link is close to meaningless.** That is an assertion
about a distribution, so it is backed with the distribution.

Three pieces of evidence:

1. the quantile table and the tail, for one scenario per regime, with the ratio
   of mean to median and of maximum to median printed beside them;
2. the measured density of runs landing inside a window around the mean, which
   is the direct statement that the mean sits where nothing lands;
3. the per-contact completion probability, which is the shape of the answer an
   operator actually needs and which a single number cannot carry at all.
"""

from __future__ import annotations

from _bootstrap import Tee, bootstrap, environment_lines, relative

bootstrap()

import numpy as np  # noqa: E402

from relfiledelivery.analytic import analytic_delivery  # noqa: E402
from relfiledelivery.link import LinkConditions, OutageModel, uniform_windows  # noqa: E402
from relfiledelivery.modes import ALL_MODES, TransferMode  # noqa: E402
from relfiledelivery.montecarlo import run_ensemble  # noqa: E402
from relfiledelivery.payload import FileTransferSpec  # noqa: E402

out = Tee("validate_distributions_output.txt")

RUNS = 3000
SPEC = FileTransferSpec(120 * 512 - 7, 512, 12)

SCENARIOS = {
    "no outages, continuous": LinkConditions(5.0e5, 1.0e4, 0.8, 0.08),
    "correlated outages, continuous": LinkConditions(
        5.0e5, 1.0e4, 0.8, 0.05, outage=OutageModel(mean_up_s=1.5, mean_down_s=1.0)
    ),
    "correlated outages, 6 contacts": LinkConditions(
        5.0e5,
        1.0e4,
        0.8,
        0.05,
        outage=OutageModel(mean_up_s=1.5, mean_down_s=1.0),
        windows=uniform_windows(6, 4.0, 10.0),
    ),
}

out.rule("environment")
for line in environment_lines():
    out(line)

out.rule("the file and the declared scenarios")
for line in SPEC.describe():
    out(line)
out("")
for name, link in SCENARIOS.items():
    out(f"[{name}]")
    for line in link.describe():
        out(f"  {line}")
out("")
out(f"runs per scenario and mode     : {RUNS}")

distributions = {}
out.rule("quantiles and the tail, deferred-NAK")
for name, link in SCENARIOS.items():
    distribution = run_ensemble(SPEC, link, TransferMode.DEFERRED_NAK, RUNS, 606)
    distributions[name] = distribution
    out("")
    out(f"[{name}]")
    for line in distribution.describe():
        out(f"  {line}")

out.rule("is the mean a description of the distribution")
out("For each scenario: the fraction of completed runs landing within plus or")
out("minus five per cent of the mean, and within plus or minus one per cent. If")
out("the mean described the distribution those fractions would be substantial.")
out("The density of a unimodal distribution near its own mean is high; the")
out("density of a round-quantised or contact-quantised one can be zero.")
out("")
header = (
    f"{'scenario':>32}{'mean s':>11}{'median s':>11}{'mean/med':>10}"
    f"{'within 1%':>11}{'within 5%':>11}{'max/med':>9}"
)
out(header)
out("-" * len(header))
density_rows = []
for name, distribution in distributions.items():
    times = distribution.delivery_times_s
    mean = distribution.mean_s
    within_1 = float(np.mean(np.abs(times - mean) <= 0.01 * mean))
    within_5 = float(np.mean(np.abs(times - mean) <= 0.05 * mean))
    density_rows.append((name, within_1, within_5))
    out(
        f"{name:>32}{mean:>11.4f}{distribution.median_s:>11.4f}"
        f"{distribution.mean_over_median:>10.4f}{within_1:>11.4f}{within_5:>11.4f}"
        f"{distribution.tail['max_over_median']:>9.4f}"
    )
out("")
out("A uniform distribution over the same support would put about 2 per cent of")
out("its mass in a plus-or-minus-one-per-cent band around its mean, so a figure")
out("far below that is the quantitative version of the claim.")

out.rule("per-contact completion probability")
out("The answer to the operator's actual question. A single expected delivery")
out("time cannot express it: what is wanted is the probability the file is down")
out("by the end of pass k, for each k, and that is a vector.")
out("")
windowed = SCENARIOS["correlated outages, 6 contacts"]
header = f"{'mode':>16}" + "".join(f"{f'by pass {i}':>12}" for i in range(6))
out(header)
out("-" * len(header))
for mode in ALL_MODES:
    distribution = run_ensemble(SPEC, windowed, mode, RUNS, 909)
    cells = distribution.completion_by_contact
    row = f"{mode.value:>16}"
    for index in range(6):
        row += f"{cells.get(index, 0.0):>12.4f}"
    out(row)
out("")
out("Unacknowledged mode is in the table because its row is the point: with no")
out("retransmission the completion probability does not improve with more passes.")

out.rule("overhead distribution, stated as quantiles rather than as a mean")
out("")
header = (
    f"{'scenario':>32}{'mode':>16}{'q0.05':>9}{'q0.50':>9}{'q0.95':>9}"
    f"{'max':>9}{'floor':>9}"
)
out(header)
out("-" * len(header))
floor = SPEC.minimum_overhead_ratio
for name, link in SCENARIOS.items():
    for mode in (TransferMode.DEFERRED_NAK, TransferMode.IMMEDIATE_NAK):
        distribution = run_ensemble(SPEC, link, mode, 1000, 303)
        ratios = distribution.overhead_ratios
        out(
            f"{name:>32}{mode.value:>16}"
            f"{float(np.quantile(ratios, 0.05)):>9.4f}"
            f"{float(np.quantile(ratios, 0.50)):>9.4f}"
            f"{float(np.quantile(ratios, 0.95)):>9.4f}"
            f"{float(np.max(ratios)):>9.4f}{floor:>9.4f}"
        )
out("")
out("The floor is the lossless overhead of the declared segmentation, which no")
out("run can beat. On a windowed link a truncated run can fall below the floor of")
out("a COMPLETED transfer, because it stopped early; those runs are counted in the")
out("completion fraction and not in the delivery-time distribution.")

out.rule("the closed form against the measured quantiles, no-outage scenario")
out("The closed form predicts an expectation exactly and a quantile only")
out("approximately, since it combines two dependent marginals at the same level.")
out("The size of that approximation is measured here.")
out("")
from relfiledelivery.analytic import analytic_delivery_time_quantile  # noqa: E402

clean = SCENARIOS["no outages, continuous"]
clean_distribution = distributions["no outages, continuous"]
exact = analytic_delivery(SPEC, clean, TransferMode.DEFERRED_NAK)
out(f"closed-form E[T]               : {exact.expected_delivery_time_s:.6f} s")
out(f"measured mean                  : {clean_distribution.mean_s:.6f} s")
out("")
header = f"{'level':>8}{'semi-analytic s':>18}{'measured s':>13}{'rel diff':>12}"
out(header)
out("-" * len(header))
worst_quantile_error = 0.0
for level in (0.05, 0.25, 0.50, 0.75, 0.95):
    predicted = analytic_delivery_time_quantile(SPEC, clean, level)
    measured = clean_distribution.quantiles[level] if level in clean_distribution.quantiles \
        else float(np.quantile(clean_distribution.delivery_times_s, level))
    rel = (predicted - measured) / measured
    worst_quantile_error = max(worst_quantile_error, abs(rel))
    out(f"{level:>8.2f}{predicted:>18.6f}{measured:>13.6f}{rel:>+12.3e}")
out("")
out(f"worst relative quantile error  : {worst_quantile_error:.6f}")
out("This is reported, not asserted against a tolerance: the semi-analytic")
out("quantile is declared an approximation in analytic.py and its error is the")
out("quantity of interest. It is the reason the analytic interval under-covers in")
out("validate_predictors.py.")

out.rule("summary")
checks = {
    "every scenario completed at least one run": all(
        d.completed > 0 for d in distributions.values()
    ),
    "the mean lies where few runs land (under 20 per cent within 5 per cent)": any(
        row[2] < 0.20 for row in density_rows
    ),
    "quantiles are monotone in every scenario": all(
        list(d.quantiles.values()) == sorted(d.quantiles.values())
        for d in distributions.values()
    ),
}
for name, passed in checks.items():
    out(f"{'PASS' if passed else 'FAIL'}  {name}")
out("")
out(f"output written to {relative(out.path)}")
out.save()
if not all(checks.values()):
    raise SystemExit("validate_distributions.py: at least one check FAILED")
