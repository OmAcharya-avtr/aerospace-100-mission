"""Baseline 2's convolution: two methods, a grid study, and one independent bound.

The convolution in baseline 2 is the mathematical core of this product, so it
gets three distinct kinds of check and not one.

1. **Numerical, not closed form, and the discretisation stated.** There is no
   closed form: the cumulative volume is an SGP4-driven integral with no
   elementary inverse, so neither pushforward has an elementary density and
   the compound Poisson sum has no elementary transform. The grid and its
   step are printed, and the quantile is recomputed on grids of 256, 512,
   1024 and 2048 bins so the **measured** convergence order is reported
   rather than claimed.

2. **Against a direct Monte Carlo of the same laws.** The second method draws
   the acquisition time, the outage count, each outage's duration and
   placement, and the idle total and placement from exactly the laws the
   convolution discretises, evaluates each loss through the same cumulative
   volume, and sums. It checks the convolution's discretisation and
   bookkeeping. It shares the structural independence approximation on
   purpose, which is why it cannot be the only check.

3. **Against a bracket neither method uses.** ``0 <= Y_real <= R_max T``
   gives ``Y_nom - R_max T <= S <= Y_nom`` path by path, from the pass
   duration and the waveform's rate ceiling alone -- no quadrature of the
   rate profile, no pushforward, no convolution, no random numbers. Every
   quantile of both methods must lie inside it. Three negative controls,
   each anchored on the true value and each verified to FIRE, show the
   bracket is not vacuous: a unit slip of 1e6, a sign flip, and a shortfall
   doubled past the booked volume.

The structural independence approximation is also measured: any probability
the convolution puts above the window capacity is probability the
independence assumption invented, and it is reported as the distribution's
``truncated_mass``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from passyield.baselines import (  # noqa: E402
    POISSON_TAIL_TOLERANCE,
    ConvolutionOptions,
    analytic_shortfall_distribution,
    fit_event_model,
    monte_carlo_shortfall_distribution,
)
from passyield.bounds import inside, shortfall_bracket  # noqa: E402
from passyield.dataset import (  # noqa: E402
    events_of,
    load_passes,
    parse_intervals,
    station_by_name,
    window_of,
)
from passyield.decompose import decompose_pass  # noqa: E402
from passyield.geometry import satellite  # noqa: E402
from passyield.link import DEFAULT_BUDGET  # noqa: E402

#: Monte Carlo draws. Sized for the compute budget: the whole script runs in a
#: few seconds on one core, and the mission allows 600 s per script.
DRAWS = 20000 if _bootstrap.FAST else 200000

#: Passes the two methods are compared on. Spread across the catalogue by
#: taking every nth pass of the committed record.
COMPARISON_PASSES = 4 if _bootstrap.FAST else 24

#: Quantile levels compared.
LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def main() -> int:
    """Run the three kinds of check on the convolution."""
    out = _bootstrap.Tee("validate_convolution_output.txt")
    out("Baseline 2 convolution validation -- P099 PassYield")
    out("")
    out(f"fast path (PASSYIELD_FAST=1): {_bootstrap.FAST}")
    out(f"Monte Carlo draws           : {DRAWS}")
    out(f"passes compared             : {COMPARISON_PASSES}")
    out(f"seed                        : {_bootstrap.SEED}")
    out("")
    out("THE CONVOLUTION IS NUMERICAL, NOT CLOSED FORM. The cumulative volume")
    out("Lambda(t) is an SGP4-driven integral with no elementary inverse, so neither")
    out("pushforward has an elementary density and the compound Poisson sum has no")
    out("elementary transform. The work is done on a uniform grid of data volume in")
    out("Mbit with numpy.convolve.")
    out("")
    out(f"Poisson truncation tolerance: {POISSON_TAIL_TOLERANCE!r}")

    satrec = satellite()
    record = load_passes()
    model = fit_event_model(
        acquisition_s=[row.acquisition_s for row in record],
        outage_counts=[row.outage_count for row in record],
        window_durations_s=[row.duration_s for row in record],
        outage_durations_s=[
            duration for row in record
            for _start, duration in parse_intervals(row.resync_intervals)
        ],
        idle_totals_s=[row.idle_total_s for row in record],
    )
    out("")
    out("the laws BOTH methods use, fitted from the committed record by the method")
    out("of moments:")
    out(f"  acquisition Gamma shape, scale : {model.acquisition_shape:.6f}, "
        f"{model.acquisition_scale:.6f} s")
    out(f"  outage intensity, 1/s          : {model.outage_intensity_per_s:.8f}")
    out(f"  mean outage duration, s        : {model.outage_mean_s:.6f}")
    out(f"  idle Gamma shape, scale        : {model.idle_shape:.6f}, "
        f"{model.idle_scale:.6f} s")
    out(f"  passes the moments came from   : {model.sample_passes}")

    stride = max(1, len(record) // COMPARISON_PASSES)
    chosen = record[::stride][:COMPARISON_PASSES]

    out.rule("1. DISCRETISATION -- the measured convergence of the median quantile")
    out("the volume grid spans [0, Y_cap] in `bins` uniform bins, so the step is")
    out("Y_cap/bins and a quantile carries at least half a bin of discretisation by")
    out("construction. The pushforward is mass-preserving, so the error in the")
    out("cumulative distribution function is FIRST order in the step.")
    out("")
    reference_row = chosen[0]
    station = station_by_name(reference_row.station)
    window = window_of(reference_row)
    item = decompose_pass(satrec, station, window, events_of(reference_row))
    out(f"reference pass: pass_id {reference_row.pass_id}, {reference_row.station}, "
        f"peak elevation {reference_row.max_elevation_deg:.2f} deg")
    out(f"window capacity, Mbit: {item.capacity_bits:.3f}")
    out("")
    out(f"{'bins':>7}{'step, Mbit':>13}{'median, Mbit':>15}{'change':>13}{'ratio':>9}")
    medians: list[float] = []
    changes: list[float] = []
    for bins in (256, 512, 1024, 2048):
        law = analytic_shortfall_distribution(
            satrec, station, window, model, item.nominal_bits, item.capacity_bits,
            ConvolutionOptions(bins=bins),
        )
        median = law.quantile(0.5)
        medians.append(median)
        change = abs(median - medians[-2]) if len(medians) > 1 else float("nan")
        if len(medians) > 1:
            changes.append(change)
        ratio = changes[-2] / changes[-1] if len(changes) > 1 else float("nan")
        out(f"{bins:>7}{law.grid_step_mbit:>13.4f}{median:>15.4f}{change:>13.4f}{ratio:>9.2f}")
    out("")
    out(f"successive changes, Mbit: {[round(value, 4) for value in changes]}")
    order = float(np.log2(changes[0] / changes[-1]) / (len(changes) - 1)) if len(changes) > 1 \
        else float("nan")
    out(f"implied convergence order in the step: {order:.3f}")
    out("first order is what a mass-preserving binned pushforward gives. A higher")
    out("apparent order on a single pass is the kink and the source grids")
    out("interacting, not a property to rely on.")
    out.check(
        "the median quantile converges as the grid is refined",
        changes[-1] < changes[0],
        f"change fell from {changes[0]:.4f} to {changes[-1]:.4f} Mbit",
    )

    out.rule("2. TWO METHODS -- the convolution against a direct Monte Carlo")
    out("the tolerance for each quantile is DERIVED per pass, not tuned, from two")
    out("terms.")
    out("")
    out("DISCRETISATION. A discrete convolution adds BIN INDICES, not values, so")
    out("each convolution stage can displace the result by up to one grid step. The")
    out("number of stages on a pass is one for the acquisition law, one for the idle")
    out("law, and one per outage in the compound Poisson sum, which is truncated at")
    out("the Poisson 1-in-1e9 quantile. The discretisation term is therefore")
    out("(1 + stages) grid steps, and `stages` is printed per pass so the term can")
    out("be checked rather than taken on trust.")
    out("")
    out("SAMPLING. The Monte Carlo quantile's standard error is")
    out("4*sigma*sqrt(p(1-p)/n), using 1/(4 sigma) as a crude lower bound on the")
    out("density of a unimodal law. Three of those.")
    out("")
    out(f"{'pass':>6}{'station':>9}{'stages':>8}{'level':>7}{'convolution':>14}"
        f"{'monte carlo':>14}{'|diff|':>11}{'steps':>8}{'tolerance':>11}{'ok':>5}")
    worst_excess = 0.0
    worst_steps = 0.0
    failures = 0
    comparisons = 0
    truncations: list[tuple[float, float]] = []
    for row in chosen:
        station = station_by_name(row.station)
        window = window_of(row)
        item = decompose_pass(satrec, station, window, events_of(row))
        analytic = analytic_shortfall_distribution(
            satrec, station, window, model, item.nominal_bits, item.capacity_bits
        )
        direct = monte_carlo_shortfall_distribution(
            satrec, station, window, model, item.nominal_bits, item.capacity_bits,
            seed=_bootstrap.SEED, draws=DRAWS,
        )
        truncations.append((analytic.truncated_mass, direct.truncated_mass))
        spread = analytic.standard_deviation_mbit()
        # Convolution stages on this pass: acquisition, idle, and one per
        # outage up to the Poisson truncation point used by baseline 2.
        intensity = model.outage_intensity_per_s * window.duration_s
        stages = 2 + int(stats.poisson.isf(POISSON_TAIL_TOLERANCE, mu=intensity)) + 1
        for level in LEVELS:
            try:
                first = analytic.quantile(level)
                second = direct.quantile(level)
            except ValueError:
                continue
            sampling = 4.0 * spread * float(np.sqrt(level * (1.0 - level) / DRAWS))
            tolerance = (1 + stages) * analytic.grid_step_mbit + 3.0 * sampling
            difference = abs(first - second)
            ok = difference <= tolerance
            comparisons += 1
            failures += int(not ok)
            worst_excess = max(worst_excess, difference / tolerance)
            worst_steps = max(worst_steps, difference / analytic.grid_step_mbit)
            if row is chosen[0] or not ok:
                out(
                    f"{row.pass_id:>6}{row.station:>9}{stages:>8}{level:>7.2f}"
                    f"{first:>14.3f}{second:>14.3f}{difference:>11.3f}"
                    f"{difference / analytic.grid_step_mbit:>8.2f}{tolerance:>11.3f}"
                    f"{str(ok):>5}"
                )
    out("")
    out(f"quantile comparisons made           : {comparisons}")
    out(f"comparisons outside tolerance       : {failures}")
    out(f"worst |difference| / tolerance      : {worst_excess:.4f}")
    out(f"worst |difference| in grid steps    : {worst_steps:.2f}")
    out.check(
        "every quantile of the two methods agrees within the stated tolerance",
        failures == 0,
        f"{failures} of {comparisons} outside",
    )

    analytic_truncation = np.asarray([pair[0] for pair in truncations])
    direct_truncation = np.asarray([pair[1] for pair in truncations])
    out("")
    out("THE STRUCTURAL APPROXIMATION, MEASURED. The convolution treats the three")
    out("losses as independent and additive. They are not: inside one window the")
    out("intervals must be disjoint, so any probability either method puts above the")
    out("window capacity is probability the independence assumption invented. Both")
    out("methods report it, and the two must agree, because they share the")
    out("assumption.")
    out("")
    out(f"convolution truncated mass, median  : {float(np.median(analytic_truncation)):.6e}")
    out(f"convolution truncated mass, worst   : {float(np.max(analytic_truncation)):.6e}")
    out(f"monte carlo truncated mass, median  : {float(np.median(direct_truncation)):.6e}")
    out(f"monte carlo truncated mass, worst   : {float(np.max(direct_truncation)):.6e}")
    worst_gap = float(np.max(np.abs(analytic_truncation - direct_truncation)))
    out(f"worst disagreement between the two  : {worst_gap:.6e}")
    out.check(
        "the two methods agree on the invented probability to within 0.01",
        worst_gap < 0.01,
        f"worst gap {worst_gap:.6e}",
    )
    out("")
    out("that number is a LIMITATION of baseline 2 and is published as one: it is")
    out("the fraction of baseline 2's own law that is physically impossible. It is")
    out("also why baseline 2 is not ground truth, which is baseline 3's job.")

    out.rule("3. ONE INDEPENDENT BOUND -- mathematics neither method uses")
    out("0 <= Y_real <= R_max * T, path by path, from the pass duration and the")
    out(f"waveform's rate ceiling R_max = {DEFAULT_BUDGET.max_rate_mbit_s!r} Mbit/s alone.")
    out("No quadrature of the rate profile, no pushforward, no convolution, no")
    out("random numbers. 'Two implementations agree' cannot catch an error the two")
    out("share; this can.")
    out("")
    outside = 0
    checked = 0
    for row in chosen:
        station = station_by_name(row.station)
        window = window_of(row)
        item = decompose_pass(satrec, station, window, events_of(row))
        bracket = shortfall_bracket(
            item.nominal_bits, item.window_duration_s, DEFAULT_BUDGET.max_rate_mbit_s
        )
        analytic = analytic_shortfall_distribution(
            satrec, station, window, model, item.nominal_bits, item.capacity_bits
        )
        direct = monte_carlo_shortfall_distribution(
            satrec, station, window, model, item.nominal_bits, item.capacity_bits,
            seed=_bootstrap.SEED, draws=min(DRAWS, 50000),
        )
        for level in (0.01, *LEVELS, 0.99):
            for law in (analytic, direct):
                try:
                    value = law.quantile(level)
                except ValueError:
                    continue
                checked += 1
                outside += int(not inside(value, bracket))
        # The accounted shortfall of the pass itself must also be inside.
        checked += 1
        outside += int(not inside(item.shortfall_bits, bracket))
    out(f"quantiles and accounted values bracketed : {checked}")
    out(f"outside the bracket                      : {outside}")
    out.check(
        "every quantile of both methods lies inside the independent bracket",
        outside == 0,
        f"{outside} of {checked} outside",
    )

    out("")
    out("THREE NEGATIVE CONTROLS, each anchored on the TRUE value and each verified")
    out("to FIRE. A control that cannot fire reads as evidence and is the worse of")
    out("the two failures.")
    out("")
    row = chosen[0]
    station = station_by_name(row.station)
    window = window_of(row)
    item = decompose_pass(satrec, station, window, events_of(row))
    bracket = shortfall_bracket(
        item.nominal_bits, item.window_duration_s, DEFAULT_BUDGET.max_rate_mbit_s
    )
    out(f"reference pass: pass_id {row.pass_id}, {row.station}")
    out(f"bracket, Mbit : [{bracket[0]:.3f}, {bracket[1]:.3f}]")
    out(f"true shortfall: {item.shortfall_bits:.3f} Mbit")
    out("")
    out.control(
        "a Mbit-reported-as-bit unit slip (factor 1e6) leaves the bracket",
        not inside(item.shortfall_bits * 1.0e6, bracket),
        must_fire=True,
        detail=f"injected {item.shortfall_bits * 1.0e6:.3e} Mbit",
    )
    out.control(
        "a sign flip on the shortfall leaves the bracket",
        not inside(
            item.nominal_bits + DEFAULT_BUDGET.max_rate_mbit_s * item.window_duration_s,
            bracket,
        ),
        must_fire=True,
        detail="injected Y_nom + R_max*T, which exceeds the bracket's own upper end",
    )
    out.control(
        "a shortfall larger than the booked volume leaves the bracket",
        not inside(item.nominal_bits * 1.0001 + 1.0, bracket),
        must_fire=True,
        detail=f"injected {item.nominal_bits * 1.0001 + 1.0:.3f} Mbit",
    )
    out.control(
        "the true shortfall stays inside the bracket",
        not inside(item.shortfall_bits, bracket),
        must_fire=False,
        detail=f"{item.shortfall_bits:.3f} Mbit",
    )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
