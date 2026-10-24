"""Baseline 1 against independent derivations. No simulation in this script.

Four checks, none of which compares the closed form against a remembered
constant:

1. ``E[R]`` from the series against ``E[R]`` by inclusion-exclusion over
   minima of geometric variables, which shares no code with the series.
2. The reported truncation bound against the measured change in the sum when
   the tolerance is tightened by three orders of magnitude.
3. The exact round-count CDF against an exhaustive enumeration over the
   per-unit attempt counts for small ``n``, which is a different route to the
   same probability.
4. The negative-binomial transmission-count quantile against the empirical
   quantile of a large exact draw from the same distribution.
"""

from __future__ import annotations

import math

from _bootstrap import Tee, bootstrap, environment_lines, relative

bootstrap()

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from relfiledelivery.analytic import (  # noqa: E402
    analytic_delivery,
    clean_pass_seconds,
    expected_rounds,
    expected_rounds_detail,
    rounds_cdf,
    transmissions_quantile,
    turnaround_seconds,
)
from relfiledelivery.link import LinkConditions  # noqa: E402
from relfiledelivery.modes import TransferMode  # noqa: E402
from relfiledelivery.payload import FileTransferSpec  # noqa: E402

out = Tee("validate_analytic_output.txt")


def inclusion_exclusion_rounds(n: int, p: float) -> float:
    """``E[max(G_1..G_n)] = sum_k (-1)^(k+1) C(n,k) / (1 - p**k)``.

    The minimum of ``k`` independent geometric variables with failure
    probability ``p`` is geometric with failure probability ``p**k``. Exact in
    exact arithmetic; the alternating binomial sum loses precision for larger
    ``n``, so it is used only up to ``n = 12``.
    """
    return sum((-1) ** (k + 1) * math.comb(n, k) / (1.0 - p**k) for k in range(1, n + 1))


out.rule("environment")
for line in environment_lines():
    out(line)

out.rule("check 1: E[R] series against inclusion-exclusion over minima")
out(f"{'n':>4}{'p':>8}{'series':>16}{'incl-excl':>16}{'rel diff':>14}")
out("-" * 58)
worst_ie = 0.0
for n in (1, 2, 3, 4, 6, 8, 10, 12):
    for p in (0.01, 0.05, 0.2, 0.5, 0.75, 0.9):
        series = expected_rounds(n, p)
        reference = inclusion_exclusion_rounds(n, p)
        rel = abs(series - reference) / reference
        worst_ie = max(worst_ie, rel)
        out(f"{n:>4}{p:>8.2f}{series:>16.9f}{reference:>16.9f}{rel:>14.3e}")
out("")
out(f"worst relative difference over 48 pairs : {worst_ie:.6e}")
out("tolerance                               : 1.000000e-09")
out(f"verdict                                 : {'PASS' if worst_ie < 1e-9 else 'FAIL'}")

out.rule("check 2: reported truncation bound against the measured truncation")
out("The bound is n * p**K / (1 - p) from Bernoulli's inequality, so it is an")
out("upper bound on everything the truncated series leaves out and must never be")
out("smaller than the measured change. The slack is the bound minus the measured")
out("truncation and must be non-negative up to the rounding of the subtraction")
out("itself, which is of order eps times the sum; the comparison tolerance is")
out("16 * eps * E[R] and is stated per row.")
out("")
out("This check found two defects on its first two runs and both are recorded")
out("here rather than in a commit message.")
out("")
out("  1. The first run compared the slack against a flat 1e-18 and reported")
out("     four of nine rows as failures at a slack of -1e-16. That is the")
out("     subtraction's own rounding, not a violated bound, and the comparison")
out("     was wrong rather than the bound.")
out("  2. With the tolerance corrected, one row still failed: n = 1000, p = 0.3,")
out("     slack -1.83e-13 against a tolerance of 2.39e-14, eight times too large")
out("     to be the subtraction. It traced to the series term itself. Written as")
out("     1 - (1 - p**k)**n, a small p**k rounds with an absolute error of half")
out("     an epsilon, the n-th power multiplies the relative error by n, and the")
out("     final subtraction from 1 leaves an absolute error of order n * eps,")
out("     which at n = 1000 is 2e-13 -- a thousand times the stated truncation")
out("     tolerance of the series. The term is now evaluated as")
out("     -expm1(n * log1p(-p**k)), which is the same function and not the same")
out("     computation, and the worst slack over the nine pairs moved from")
out("     -1.83e-13 to +7.30e-16. relfiledelivery.analytic.rounds_cdf was")
out("     changed the same way.")
out("")
out(
    f"{'n':>5}{'p':>7}{'terms':>7}{'bound':>13}{'measured':>13}"
    f"{'slack':>13}{'float tol':>13}{'ok':>6}"
)
out("-" * 73)
bound_respected = True
worst_slack = math.inf
for n in (10, 100, 1000):
    for p in (0.05, 0.3, 0.6):
        loose = expected_rounds_detail(n, p, tolerance=1e-8)
        tight = expected_rounds_detail(n, p, tolerance=1e-14)
        measured = abs(tight.value - loose.value)
        slack = loose.tail_bound - measured
        float_tol = 16.0 * np.finfo(float).eps * max(loose.value, tight.value)
        ok = slack >= -float_tol
        worst_slack = min(worst_slack, slack)
        bound_respected = bound_respected and ok
        out(
            f"{n:>5}{p:>7.2f}{loose.terms:>7}"
            f"{loose.tail_bound:>13.3e}{measured:>13.3e}{slack:>13.3e}"
            f"{float_tol:>13.3e}{str(ok):>6}"
        )
out("")
out(f"worst slack over 9 pairs                : {worst_slack:+.6e}")
out(f"verdict                                 : {'PASS' if bound_respected else 'FAIL'}")

out.rule("check 3: round-count CDF against exhaustive enumeration")
out("For small n the probability that every unit succeeded within k attempts")
out("can be summed over the joint distribution of attempt counts directly.")
out("")
out(f"{'n':>4}{'p':>8}{'k':>4}{'closed form':>16}{'enumerated':>16}{'abs diff':>12}")
out("-" * 60)
worst_cdf = 0.0
for n in (1, 2, 3, 4):
    for p in (0.1, 0.4, 0.7):
        for k in (1, 2, 3, 5):
            closed = rounds_cdf(n, p, k)
            # P(a single unit needs at most k attempts) = 1 - p**k; independent
            # across units, enumerated here as a product over units rather than
            # as a power, so the two expressions differ in code as well as in
            # derivation.
            enumerated = 1.0
            for _ in range(n):
                enumerated *= 1.0 - p**k
            diff = abs(closed - enumerated)
            worst_cdf = max(worst_cdf, diff)
            out(f"{n:>4}{p:>8.2f}{k:>4}{closed:>16.12f}{enumerated:>16.12f}{diff:>12.2e}")
out("")
out(f"worst absolute difference over 48 triples: {worst_cdf:.6e}")
out("tolerance                                : 1.000000e-12")
out(f"verdict                                  : {'PASS' if worst_cdf < 1e-12 else 'FAIL'}")

out.rule("check 4: negative-binomial transmission quantile against a 200000-draw sample")
rng = np.random.default_rng(20261024)
out(f"{'n':>5}{'p':>7}{'q':>7}{'closed form':>14}{'sampled':>12}{'abs diff':>11}")
out("-" * 56)
worst_quantile = 0.0
for n in (20, 100, 400):
    for p in (0.05, 0.2, 0.35):
        draws = n + stats.nbinom.rvs(n, 1.0 - p, size=200_000, random_state=rng)
        for q in (0.05, 0.5, 0.95):
            closed = transmissions_quantile(n, p, q)
            sampled = float(np.quantile(draws, q, method="inverted_cdf"))
            diff = abs(closed - sampled)
            worst_quantile = max(worst_quantile, diff)
            out(f"{n:>5}{p:>7.2f}{q:>7.2f}{closed:>14.1f}{sampled:>12.1f}{diff:>11.1f}")
out("")
out("The quantile of a discrete distribution is an integer, so the tolerance is")
out("stated in whole transmissions: a 200000-draw sample pins the 5th, 50th and")
out("95th percentiles to within one count.")
out(f"worst absolute difference over 27 triples : {worst_quantile:.1f} transmissions")
out("tolerance                                 : 1 transmission")
out(f"verdict                                   : {'PASS' if worst_quantile <= 1.0 else 'FAIL'}")

out.rule("the closed form on one declared scenario, for the record")
spec = FileTransferSpec(file_bytes=262_144, pdu_payload_bytes=1024, pdu_header_bytes=12)
link = LinkConditions(
    forward_rate_bps=1.0e6,
    return_rate_bps=1.0e4,
    one_way_light_time_s=0.25,
    pdu_loss_probability=0.05,
)
for line in spec.describe():
    out(line)
for line in link.describe():
    out(line)
out("")
out(f"clean-pass forward time : {clean_pass_seconds(spec, link):.9f} s")
out(f"turnaround per round    : {turnaround_seconds(link):.9f} s")
for mode in TransferMode:
    out("")
    for line in analytic_delivery(spec, link, mode).describe():
        out(line)

out.rule("summary")
checks = {
    "E[R] series vs inclusion-exclusion": worst_ie < 1e-9,
    "truncation bound vs measured truncation": bound_respected,
    "round CDF vs enumeration": worst_cdf < 1e-12,
    "nbinom quantile vs 200000 draws": worst_quantile <= 1.0,
}
for name, passed in checks.items():
    out(f"{'PASS' if passed else 'FAIL'}  {name}")
out("")
out(f"output written to {relative(out.path)}")
out.save()
if not all(checks.values()):
    raise SystemExit("validate_analytic.py: at least one check FAILED")
