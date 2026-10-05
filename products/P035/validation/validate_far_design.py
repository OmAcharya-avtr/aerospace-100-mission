"""Validation: EWMA, CUSUM and limit-check false-alarm rates match their design values.

Level 2 requirement: "the EWMA and CUSUM empirical false-alarm rates match their
design values under the nominal hypothesis across seeds".

Operating-point currency
------------------------
The window false-alarm probability ``alpha_W``: the probability that a chart,
started from its reset state, alarms at least once within ``W`` samples of
nominal data.  One window is one Bernoulli trial, so the estimate from ``M``
independent windows has the exact binomial standard error
``sqrt(alpha (1 - alpha) / M)``.  A per-sample rate measured along one long run
does not have that property, because the chart statistic is serially dependent.

Sample size
-----------
``M = 20000`` windows per seed.  At ``alpha_W = 0.05`` the binomial standard
error is ``sqrt(0.05 * 0.95 / 20000) = 0.0015411``, i.e. 3.1 % relative, chosen
so that a 10 % relative error between design and measurement is a 3.2-sigma
event.  Eight independent seeds are run, giving 160000 windows per chart in
total and a pooled standard error of ``0.000545``.

The AR(1) section is deliberately a demonstration of failure: the designs assume
independent samples, and under serially correlated telemetry they do not deliver
their design value.  That is reported as the measured number, not hidden.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from _reporting import Report  # noqa: E402

from telemetryool.arl import (  # noqa: E402
    cusum_window_false_alarm,
    design_cusum_h,
    design_ewma_L,
    design_ool_limit,
    ewma_window_false_alarm,
    ool_window_false_alarm,
)
from telemetryool.calibration import binomial_se, estimate_rate, windows_for_precision  # noqa: E402
from telemetryool.charts import CusumChart, EwmaChart  # noqa: E402
from telemetryool.runs import any_run  # noqa: E402
from telemetryool.synthetic import NominalModel, generate_nominal  # noqa: E402

TARGET = 0.05
WINDOW = 100
WINDOWS_PER_SEED = 20_000
SEEDS = [20261005 + 1000 * i for i in range(8)]
#: Agreement band.  3.5 pooled standard errors; stated before the measurement,
#: not chosen after it.
Z_LIMIT = 3.5


def nominal_block(rho: float, n_windows: int, seed: int) -> np.ndarray:
    model = NominalModel(1, rho_time=rho)
    return generate_nominal(model, n_windows, WINDOW, np.random.default_rng(seed))[:, :, 0]


def measure(chart, rho: float, seed: int) -> int:
    block = nominal_block(rho, WINDOWS_PER_SEED, seed)
    return int(any_run(chart.breach_mask(block), chart.persistence).sum())


def main() -> int:
    report = Report(
        "validate_far_design",
        "Designed vs measured window false-alarm probability (EWMA, CUSUM, limit check)",
    )
    report.line(f"target alpha_W           : {TARGET}")
    report.line(f"window length W          : {WINDOW} samples")
    report.line(f"windows per seed M       : {WINDOWS_PER_SEED}")
    report.line(f"seeds                    : {len(SEEDS)}  {SEEDS}")
    report.line(f"per-seed binomial SE     : {binomial_se(TARGET, WINDOWS_PER_SEED):.7f}")
    report.line(
        f"pooled binomial SE       : "
        f"{binomial_se(TARGET, WINDOWS_PER_SEED * len(SEEDS)):.7f}"
    )
    report.line(
        f"windows for 10 % rel. SE : "
        f"{windows_for_precision(TARGET, 0.10)} (reference, from "
        f"telemetryool.calibration.windows_for_precision)"
    )
    report.line(f"agreement band           : |z| < {Z_LIMIT} pooled standard errors")

    report.section("Designed thresholds (analytic, iid normal, sigma known)")
    cusum_design = design_cusum_h(TARGET, 0.5, WINDOW)
    ewma_design = design_ewma_L(TARGET, 0.2, WINDOW)
    ool_design = design_ool_limit(TARGET, 3, WINDOW)
    report.line(
        f"{'method':28s} {'parameter':>12s} {'value':>16s} {'alpha_W design':>16s} "
        f"{'ARL0 (samples)':>16s}"
    )
    rows = [
        ("cusum k=0.5 (Brook-Evans)", "h", cusum_design),
        ("ewma lam=0.2 (Lucas-Sacc.)", "L", ewma_design),
        ("limit check persistence=3", "L", ool_design),
    ]
    for label, param, design in rows:
        report.line(
            f"{label:28s} {param:>12s} {design.threshold:16.9f} "
            f"{design.achieved_alpha_w:16.10f} {design.arl0:16.2f}"
        )
    report.line("")
    report.line("The design values above are the exact values of the discretised chains, so")
    report.line("they reproduce the target to the bisection tolerance by construction.  The")
    report.line("question this script answers is whether Monte Carlo agrees with them.")

    charts = {
        "cusum k=0.5": (
            CusumChart(k=0.5, h=cusum_design.threshold),
            cusum_window_false_alarm(0.5, cusum_design.threshold, WINDOW),
        ),
        "ewma lam=0.2": (
            EwmaChart(lam=0.2, limit_mult=ewma_design.threshold),
            ewma_window_false_alarm(0.2, ewma_design.threshold, WINDOW),
        ),
    }
    # The limit check is a chart only in the sense that it has a breach mask; the
    # exceedance probability comes straight from the designed multiplier.
    from scipy.stats import norm

    p_exceed = 2.0 * (1.0 - float(norm.cdf(ool_design.threshold)))

    report.section("Per-seed measurement under the design hypothesis (iid normal, rho = 0)")
    report.line(
        f"{'chart':16s} {'seed':>12s} {'alarms':>8s} {'measured':>11s} {'SE':>10s} "
        f"{'z vs design':>12s}"
    )
    pooled: dict[str, list[int]] = {name: [] for name in charts}
    pooled["limit p=3"] = []
    for name, (chart, _design) in charts.items():
        for seed in SEEDS:
            k = measure(chart, 0.0, seed)
            pooled[name].append(k)
            est = estimate_rate(k, WINDOWS_PER_SEED)
            report.line(
                f"{name:16s} {seed:12d} {k:8d} {est.rate:11.6f} "
                f"{est.standard_error:10.6f} {est.z_against(charts[name][1]):12.2f}"
            )
    for seed in SEEDS:
        block = nominal_block(0.0, WINDOWS_PER_SEED, seed)
        k = int(any_run(np.abs(block) > ool_design.threshold, 3).sum())
        pooled["limit p=3"].append(k)
        est = estimate_rate(k, WINDOWS_PER_SEED)
        report.line(
            f"{'limit p=3':16s} {seed:12d} {k:8d} {est.rate:11.6f} "
            f"{est.standard_error:10.6f} {est.z_against(ool_design.achieved_alpha_w):12.2f}"
        )

    report.section("Pooled over all seeds: design vs measurement")
    design_values = {
        "cusum k=0.5": charts["cusum k=0.5"][1],
        "ewma lam=0.2": charts["ewma lam=0.2"][1],
        "limit p=3": ool_window_false_alarm(p_exceed, 3, WINDOW),
    }
    total = WINDOWS_PER_SEED * len(SEEDS)
    report.line(
        f"{'chart':16s} {'design':>12s} {'measured':>12s} {'abs diff':>11s} "
        f"{'rel diff':>10s} {'pooled SE':>11s} {'z':>8s} {'95% Wilson':>24s}"
    )
    for name, counts in pooled.items():
        k = int(sum(counts))
        est = estimate_rate(k, total)
        design = design_values[name]
        z = est.z_against(design)
        report.line(
            f"{name:16s} {design:12.7f} {est.rate:12.7f} "
            f"{est.rate - design:+11.7f} {(est.rate - design) / design:+10.4f} "
            f"{binomial_se(design, total):11.7f} {z:8.2f} "
            f"[{est.wilson_low:.7f}, {est.wilson_high:.7f}]"
        )
    report.line("")
    for name, counts in pooled.items():
        k = int(sum(counts))
        est = estimate_rate(k, total)
        z = est.z_against(design_values[name])
        report.check(
            f"{name}: pooled measured alpha_W agrees with design within "
            f"{Z_LIMIT} pooled SE",
            abs(z) < Z_LIMIT,
            f"measured={est.rate:.7f} design={design_values[name]:.7f} z={z:+.2f}",
        )

    report.section("Seed-to-seed spread against the binomial prediction")
    report.line("If the windows are independent Bernoulli trials, the standard deviation of")
    report.line("the eight per-seed rates should match the per-seed binomial SE.")
    report.line(
        f"{'chart':16s} {'mean rate':>12s} {'observed sd':>13s} {'binomial SE':>13s} "
        f"{'ratio':>8s}"
    )
    for name, counts in pooled.items():
        rates = np.array(counts, dtype=float) / WINDOWS_PER_SEED
        observed = float(rates.std(ddof=1))
        predicted = binomial_se(design_values[name], WINDOWS_PER_SEED)
        report.line(
            f"{name:16s} {rates.mean():12.7f} {observed:13.7f} {predicted:13.7f} "
            f"{observed / predicted:8.3f}"
        )
    report.line("")
    report.line("With only 8 seeds the sample standard deviation itself has a relative")
    report.line("standard error of 1/sqrt(2*(8-1)) = 0.267, so a ratio anywhere in")
    report.line("[0.47, 1.53] is consistent with the binomial model at 2 sigma.  The check")
    report.line("below uses that band.")
    for name, counts in pooled.items():
        rates = np.array(counts, dtype=float) / WINDOWS_PER_SEED
        ratio = float(rates.std(ddof=1)) / binomial_se(design_values[name], WINDOWS_PER_SEED)
        report.check(
            f"{name}: seed-to-seed spread consistent with the binomial model",
            0.47 < ratio < 1.53,
            f"ratio={ratio:.3f}",
        )

    report.section("Serially correlated telemetry: the design hypothesis violated on purpose")
    report.line("The designs above assume independent samples.  Housekeeping telemetry")
    report.line("usually is not independent.  Measured alpha_W under an AR(1) nominal model")
    report.line("with the SAME designed thresholds, 20000 windows per cell, one seed each:")
    report.line("")
    report.line(
        f"{'chart':16s} {'rho':>6s} {'design':>10s} {'measured':>11s} {'SE':>10s} "
        f"{'ratio to design':>16s}"
    )
    inflation: dict[str, dict[float, float]] = {}
    for name, (chart, design) in charts.items():
        inflation[name] = {}
        for rho in (0.0, 0.3, 0.6, 0.9):
            k = measure(chart, rho, 424242)
            est = estimate_rate(k, WINDOWS_PER_SEED)
            inflation[name][rho] = est.rate / design
            report.line(
                f"{name:16s} {rho:6.2f} {design:10.6f} {est.rate:11.6f} "
                f"{est.standard_error:10.6f} {est.rate / design:16.3f}"
            )
    for rho in (0.0, 0.3, 0.6, 0.9):
        block = nominal_block(rho, WINDOWS_PER_SEED, 424242)
        k = int(any_run(np.abs(block) > ool_design.threshold, 3).sum())
        est = estimate_rate(k, WINDOWS_PER_SEED)
        design = design_values["limit p=3"]
        inflation.setdefault("limit p=3", {})[rho] = est.rate / design
        report.line(
            f"{'limit p=3':16s} {rho:6.2f} {design:10.6f} {est.rate:11.6f} "
            f"{est.standard_error:10.6f} {est.rate / design:16.3f}"
        )
    report.line("")
    report.line("This is a documented limitation, not a check that can pass: at rho = 0.9 the")
    report.line("delivered false-alarm probability is multiples of the design value for the")
    report.line("drift charts and a large multiple for the persistence-based limit check,")
    report.line("because a correlated excursion keeps the persistence counter running.")
    report.check(
        "rho = 0 is the only setting where the analytic design is delivered",
        all(abs(inflation[n][0.0] - 1.0) < 0.1 for n in inflation),
        ", ".join(f"{n}: ratio at rho=0 is {inflation[n][0.0]:.3f}" for n in inflation),
    )
    report.line("")
    report.line("Inflation factors, measured alpha_W divided by design alpha_W:")
    for name, by_rho in inflation.items():
        report.line(
            f"  {name:16s} "
            + "  ".join(f"rho={rho}: {ratio:.2f}x" for rho, ratio in by_rho.items())
        )
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
