"""The three baselines, measured, across the whole workload grid.

Reported here, in the order the specification requires them implemented:

1. **Baseline 1, energy per inference**, both variants, on every grid cell.
2. **Baseline 2, the analytic duty-cycle model**, with every component
   measured: the background floor from the no-pipeline cell, the steady active
   draw, and the wake transient's excess energy -- the last of which is a
   BANDWIDTH-LIMITED estimate, because the INA3221 averages 512 conversions in
   hardware, and which comes out NEGATIVE on this board for exactly that
   reason. See section 5.
3. **Baseline 3, ground truth**, with the between-run interval from the four
   repeats of the reference cell.

Plus three things the measurement itself forced into this file:

* the **floor-occupancy regression and the floor-sensitivity table**, because
  the board was shared and the duty-cycle half of the headline rests on the
  floor;
* the **run-order table**, because the run order was not randomised and frame
  dimension is confounded with elapsed time and with a falling foreign load;
* the **reject-reason census**, because two of the gate's five branches never
  fired on this campaign.

Baseline 2 is compared with Baseline 3 on every grid cell BEFORE any learned
component is benchmarked anywhere, because an under-strength analytic baseline
inflates a learned win, and because a baseline that has not been checked
against the ground truth is not a baseline.

INPUT ORIGIN: the MEASURED hardware campaign under ``data/traces/`` and nothing
else. No synthetic fallback.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import dataclasses  # noqa: E402

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from powerenvelope.baselines import (  # noqa: E402
    RAMP_SEPARABLE_MIN_S,
    RAMP_WINDOW_S,
    between_run_spread,
    duty_cycle_model,
    energy_per_inference,
    floor_occupancy_regression,
    floor_power,
    floor_variants,
    ground_truth,
    rail_accounting_curve,
    run_order_rows,
    sleep_power_mw,
)
from powerenvelope.report import (  # noqa: E402
    baseline2_lines,
    campaign_lines,
    census_lines,
    floor_lines,
    floor_regression_lines,
    rail_lines,
    run_order_lines,
    spread_lines,
)
from powerenvelope.trace import TraceDataAbsentError  # noqa: E402


def report(out: _bootstrap.Tee, campaign) -> None:
    """Write the whole measured report into ``out`` from an already-loaded campaign.

    Split out from :func:`main` so that ``tests/test_measured_reports.py`` can
    exercise every formatting and arithmetic path in this file against a
    SYNTHETIC campaign, without the refusal above being weakened and without a
    committed output file being overwritten. The split changes nothing about
    the measured path: ``main`` refuses, loads, and calls this.
    """
    out.rule("1  campaign identity")
    for line in campaign_lines(campaign):
        out(f"  {line}")

    out.rule("2  the measured background floor")
    floor = floor_power(campaign)
    for line in floor_lines(floor):
        out(f"  {line}")
    out.check(
        "the floor cell ran no pipeline",
        not campaign.require(floor.cell).has_pipeline,
        "the sampler and monitor ran; the pipeline did not",
    )
    out.check(
        "the floor's own quantisation floor is far below its mean power",
        floor.quantisation_mw < 0.05 * floor.mean_mw,
        f"{floor.quantisation_mw:.3f} mW quantum on a {floor.mean_mw:.3f} mW floor",
    )

    out.rule("3  how much of the floor is other people's load")
    try:
        regression = floor_occupancy_regression(campaign)
    except ValueError as error:
        regression = None
        out.unmeasurable_check("the floor-occupancy regression", str(error))
    if regression is not None:
        for line in floor_regression_lines(regression):
            out(f"  {line}")
        out.check(
            "the regression intercept lies below the measured floor mean",
            regression.intercept_mw < regression.power_mean_mw,
            f"{regression.intercept_mw:.3f} < {regression.power_mean_mw:.3f} mW",
        )
        out.check(
            "the correlation is reported whatever it is, and it is weak",
            abs(regression.pearson_r) < 0.8,
            f"Pearson r {regression.pearson_r:.6f}; a strong correlation would mean the "
            "floor is dominated by measurable foreign load, and it is not",
        )

    out.rule("4  floor sensitivity: what the headline and Baseline 2 do and do not inherit")
    out(
        "  THE HEADLINE FACTOR F DOES NOT DEPEND ON THE FLOOR AT ALL. It is "
        "(E_total / N_accepted) divided by (mean active power * mean latency): four "
        "measured integrals and two counts, none of which is the floor cell. The floor "
        "enters BASELINE 2, which models the sleep energy, and the table below is "
        "therefore a sensitivity of Baseline 2 and of the analytic model's error against "
        "ground truth. That distinction is the reason the headline is robust to the "
        "contaminated floor and the analytic model is not."
    )
    out("")
    variants = floor_variants(campaign, include_regression=regression is not None)
    reference = ground_truth(campaign.require(_bootstrap.HEADLINE_CELL))
    out(
        f"  {'floor variant':<24}{'floor mW':>11}{'B2(w64_d050)':>14}"
        f"{'vs truth':>10}{'grid max':>10}{'grid mean':>11}"
    )
    out("  " + "-" * 80)
    for name, value in variants.items():
        alternative = dataclasses.replace(floor, mean_mw=value)
        ratios = []
        for cell in campaign.grid_cells():
            model = duty_cycle_model(campaign, cell.label, floor=alternative)
            truth = ground_truth(cell)
            ratios.append(
                abs(
                    model.energy_per_useful_decision_mj
                    / truth.energy_per_useful_decision_mj
                    - 1.0
                )
            )
        headline_model = duty_cycle_model(
            campaign, _bootstrap.HEADLINE_CELL, floor=alternative
        )
        out(
            f"  {name:<24}{value:>11.3f}"
            f"{headline_model.energy_per_useful_decision_mj:>14.4f}"
            f"{headline_model.energy_per_useful_decision_mj / reference.energy_per_useful_decision_mj:>10.4f}"  # noqa: E501
            f"{max(ratios):>10.4f}{float(np.mean(ratios)):>11.4f}"
        )
    out("")
    out(
        "  The published headline uses the MEASURED floor mean. The other rows are the "
        "stated sensitivity and are not used anywhere else."
    )

    out.rule("5  the run order was NOT randomised, and here is the drift")
    for line in run_order_lines(run_order_rows(campaign)):
        out(line)
    rows = run_order_rows(campaign)
    first, last = rows[0], rows[-1]
    environment = campaign.manifest.get("env", {})
    start_load = environment.get("loadavg_at_start")
    end_load = environment.get("loadavg_at_end")
    drift = (
        f" and the 1-minute load average at the campaign's start and end were "
        f"{start_load[0]:.3f} and {end_load[0]:.3f}"
        if isinstance(start_load, list) and isinstance(end_load, list)
        else " and the manifest records no start and end load average"
    )
    out("")
    out(
        f"  The plan ran the floor first and then the three frame dimensions in turn, so "
        f"FRAME DIMENSION IS CONFOUNDED with elapsed time and with the board's foreign "
        f"load. Over the {last.elapsed_s / 60.0:.1f} minutes of the campaign the measured "
        f"occupancy of the floor run was {first.busy_mean:.4f}{drift}."
    )
    out(
        "  The floor was therefore measured at the HIGHEST foreign load of the campaign "
        "and the w96 cells ran at the lowest. This is a defect in the campaign's design, "
        "not in the analysis. Nothing was re-run to hide it and nothing is corrected out "
        "of the headline; a drift-adjusted variant is reported in section 7, clearly "
        "labelled as an adjustment."
    )
    lowest = min(rows, key=lambda row: row.mean_power_mw)
    out.check(
        "at least one pipeline cell drew LESS on average than the background floor",
        lowest.mean_power_mw < floor.mean_mw,
        f"{lowest.cell} at {lowest.mean_power_mw:.2f} mW against a floor of "
        f"{floor.mean_mw:.2f} mW. A 'floor' a running cell goes under is not a floor; "
        "this check is written to FIRE on that condition so the drift cannot be "
        "overlooked, and it is reported as a finding about the campaign",
    )

    out.rule("6  baseline 1 across the grid, both variants")
    out(
        f"  {'cell':<14}{'inf':>8}{'acc':>8}{'a':>8}"
        f"{'P_act mW':>11}{'L ms':>9}{'measured mJ':>13}{'nominal mJ':>12}"
    )
    out("  " + "-" * 83)
    for cell in campaign.grid_cells():
        measured = energy_per_inference(cell, "measured-active")
        nominal = energy_per_inference(cell, "nominal-rail")
        truth = ground_truth(cell)
        out(
            f"  {cell.label:<14}{truth.inferences:>8}{truth.accepted:>8}"
            f"{truth.acceptance_rate:>8.4f}{measured.mean_active_power_mw:>11.3f}"
            f"{measured.mean_latency_s * 1e3:>9.4f}{measured.energy_mj:>13.6f}"
            f"{nominal.energy_mj:>12.6f}"
        )
    sample = energy_per_inference(campaign.require(_bootstrap.HEADLINE_CELL))
    bus_voltage = float(campaign.require(_bootstrap.HEADLINE_CELL).power.vdd_in_mv.mean())
    out("")
    out(
        f"  The measured bus voltage on this board averages {bus_voltage:.1f} mV, BELOW "
        "the nominal 5000 mV. So the nominal-rail variant OVERSTATES the active power "
        "and therefore overstates energy per inference, which makes the understatement "
        "factor F SMALLER for that variant than for the measured one. A power budget "
        "quoting 5.0 V on this board is wrong in the direction that flatters it."
    )
    out(
        f"  interstitial samples on {sample.variant} at {_bootstrap.HEADLINE_CELL}: "
        f"{sample.interstitial.n_samples} in {sample.interstitial.n_gaps} gap(s), widest "
        f"inter-inference gap {sample.interstitial.max_gap_s * 1e3:.4f} ms."
    )

    out.rule("7  baseline 2, with every component measured")
    out(
        f"  settling window: {RAMP_WINDOW_S:.3f} s, declared before the campaign ran; "
        f"separable only when the active phase is at least {RAMP_SEPARABLE_MIN_S:.3f} s"
    )
    out("")
    out(
        f"  {'cell':<14}{'P_floor mW':>12}{'P_act mW':>11}{'E_ramp mJ':>12}"
        f"{'sep':>6}{'wakes':>7}{'per cyc mJ':>12}{'E_cycle mJ':>13}{'ident':>11}"
    )
    out("  " + "-" * 98)
    negative_ramps = 0
    for cell in campaign.grid_cells():
        model = duty_cycle_model(campaign, cell.label, floor=floor)
        if model.e_ramp_mj < 0.0:
            negative_ramps += 1
        out(
            f"  {cell.label:<14}{model.p_floor_mw:>12.3f}{model.p_active_mw:>11.3f}"
            f"{model.e_ramp_mj:>12.4f}{str(model.ramp_separable):>6}{model.n_wakes:>7}"
            f"{model.e_ramp_per_cycle_mj:>12.4f}{model.e_cycle_mj:>13.4f}"
            f"{model.identity_residual:>11.2e}"
        )
    out("")
    out(
        f"  E_RAMP IS NEGATIVE on {negative_ramps} of the {len(campaign.grid_cells())} "
        "grid cells, and that is a measurement of the instrument, not of the board. The "
        f"INA3221 averages 512 conversions in hardware, so a reading taken within "
        f"{RAMP_WINDOW_S:.2f} s of a wake still carries the sleep window's power. The "
        "observable settling interval therefore shows a DEFICIT against the steady level "
        "that follows, not the excess a wake transient physically is. The figure is "
        "reported as measured: it is a bandwidth-limited estimate and the bandwidth "
        "limit is the instrument's. Nothing here is a rise time, and the settling window "
        "was not re-chosen after seeing this."
    )
    out(
        "  The ramp is charged per WAKE and scaled by wakes/cycles. At duty 1.00 the "
        "active phases are contiguous, so the run has one wake and not 38; charging the "
        "per-wake figure to every cycle over-counted it 38-fold in the first "
        "implementation and the two-term identity in the last column found it."
    )
    out("")
    for line in baseline2_lines(duty_cycle_model(campaign, _bootstrap.HEADLINE_CELL, floor=floor)):
        out(f"  {line}")
    worst_identity = max(
        abs(duty_cycle_model(campaign, cell.label, floor=floor).identity_residual)
        for cell in campaign.pipeline_cells()
    )
    out.check(
        "the three-term and two-term cycle energies agree on every pipeline cell",
        worst_identity < 1e-4,
        f"largest relative residual {worst_identity:.3e}; the residual is the pipeline's "
        "loop overshoot past the declared phase boundary times the ramp's share of the "
        "cycle energy",
    )

    out.rule("8  baseline 2 against baseline 3, before anything is benchmarked")
    out(f"  {'cell':<14}{'analytic mJ/dec':>17}{'truth mJ/dec':>14}{'ratio':>9}")
    out("  " + "-" * 54)
    ratios = []
    for cell in campaign.grid_cells():
        model = duty_cycle_model(campaign, cell.label, floor=floor)
        truth = ground_truth(cell)
        ratio = model.energy_per_useful_decision_mj / truth.energy_per_useful_decision_mj
        ratios.append((cell.label, ratio))
        out(
            f"  {cell.label:<14}{model.energy_per_useful_decision_mj:>17.4f}"
            f"{truth.energy_per_useful_decision_mj:>14.4f}{ratio:>9.4f}"
        )
    worst_label, worst_ratio = max(ratios, key=lambda item: abs(item[1] - 1.0))
    worst = abs(worst_ratio - 1.0)
    out("")
    out(f"  largest relative gap: {worst:.4f}, on {worst_label}")
    out.check(
        "the analytic duty-cycle model is within 20 per cent of ground truth on every "
        "grid cell",
        worst <= 0.20,
        f"largest gap {worst:.4f} on {worst_label}; the threshold is declared here and "
        "the gap is reported whatever it is",
    )

    out.rule("9  a DRIFT-ADJUSTED variant of Baseline 2, labelled as an adjustment")
    out(
        "  The campaign floor was measured at the highest foreign load of the run order. "
        "Substituting, for each cell, the mean rail power measured during THAT CELL'S OWN "
        "sleep windows removes the drift confound without touching anything else. This is "
        "an ADJUSTMENT. It is not the headline, it is not used by any other figure here, "
        "and it is reported because it is informative about where the analytic model's "
        "error comes from."
    )
    out("")
    out(
        f"  {'cell':<22}{'campaign floor':>15}{'in-cell sleep':>15}"
        f"{'ratio (campaign)':>18}{'ratio (adjusted)':>18}"
    )
    out("  " + "-" * 90)
    adjusted = []
    for cell in campaign.pipeline_cells():
        if cell.duty >= 1.0:
            continue
        sleep_mean, _sleep_min = sleep_power_mw(cell)
        alternative = dataclasses.replace(floor, mean_mw=sleep_mean)
        base = duty_cycle_model(campaign, cell.label, floor=floor)
        tuned = duty_cycle_model(campaign, cell.label, floor=alternative)
        truth = ground_truth(cell)
        ratio_base = base.energy_per_useful_decision_mj / truth.energy_per_useful_decision_mj
        ratio_tuned = tuned.energy_per_useful_decision_mj / truth.energy_per_useful_decision_mj
        adjusted.append(ratio_tuned)
        out(
            f"  {cell.label:<22}{floor.mean_mw:>15.2f}{sleep_mean:>15.2f}"
            f"{ratio_base:>18.4f}{ratio_tuned:>18.4f}"
        )
    spread_of_adjusted = float(np.max(adjusted) - np.min(adjusted))
    out("")
    out(
        f"  adjusted ratios span {np.min(adjusted):.4f} to {np.max(adjusted):.4f}, a "
        f"width of {spread_of_adjusted:.4f}, against {min(abs(r - 1) for _l, r in ratios):.4f} "
        f"to {worst:.4f} of absolute error at the campaign floor."
    )
    out.check(
        "the in-cell adjustment collapses the scatter in the analytic model's error",
        spread_of_adjusted < 0.02,
        f"the adjusted ratios are uniform to {spread_of_adjusted:.4f}, which says the "
        "scatter in the unadjusted error is the floor drift and not the analytic form",
    )

    out.rule("10  the measured rail-accounting ratio against load")
    rail = rail_accounting_curve(campaign)
    for line in rail_lines(rail):
        out(f"  {line}")
    out("")
    out(f"  {'cell':<22}{'load mW':>12}{'ratio':>10}{'fitted':>10}{'residual':>11}")
    out("  " + "-" * 65)
    for label, load, ratio in zip(rail.cells, rail.load_mw, rail.ratio, strict=True):
        fitted = rail.slope_per_mw * load + rail.intercept
        out(f"  {label:<22}{load:>12.3f}{ratio:>10.6f}{fitted:>10.6f}{ratio - fitted:>11.6f}")
    out.check(
        "the ratio is between 0 and 1 on every cell",
        all(0.0 < value < 1.0 for value in rail.ratio),
        f"{min(rail.ratio):.6f} to {max(rail.ratio):.6f}",
    )
    out.check(
        "the linear fit of the ratio against load is reported with its R-squared, "
        "however weak",
        np.isfinite(rail.r_squared),
        f"R^2 = {rail.r_squared:.6f}; a low value means the ratio is not well described "
        "as a line in load, which is published rather than fitted around",
    )

    out.rule("11  baseline 3, ground truth, and the between-run interval")
    spread = between_run_spread(campaign)
    for line in spread_lines(spread):
        out(f"  {line}")
    out.check(
        "four repeats contributed",
        spread.n == 4,
        f"{', '.join(spread.cells)}",
    )
    out.check(
        "the interval has positive width",
        spread.half_width_mj > 0.0,
        f"half-width {spread.half_width_mj:.6f} mJ = "
        f"{100.0 * spread.half_width_mj / spread.mean_mj:.4f} per cent of the mean",
    )
    out(
        f"  The first repeat, {spread.cells[0]}, is the grid cell itself and ran "
        f"{run_order_rows(campaign)[0].elapsed_s:.0f} s into the campaign; the other three "
        "ran at the end. The spread therefore contains the drift as well as the "
        "run-to-run variation, which widens it and is the honest direction to err in."
    )

    out.rule("12  the reject-reason census, and the two branches that never fired")
    for line in census_lines(campaign):
        out(line)
    from powerenvelope.report import reject_census
    from powerenvelope.schema import REJECT_REASONS

    census = reject_census(campaign)
    never = [reason for reason in REJECT_REASONS if census.get(reason, 0) == 0]
    out.check(
        "every reject reason in the committed traces is one the campaign script can write",
        set(census) <= set(REJECT_REASONS),
        f"{sorted(census)}",
    )
    if never:
        out.unmeasurable_check(
            f"the gate branches {', '.join(never)}",
            "zero occurrences in the entire campaign. Those paths of the gate are "
            "exercised only by the synthetic generator, so the measured acceptance rate "
            "is a property of a gate two of whose five branches were never reached by "
            "measurement. No occurrence was synthesised to fill the table.",
        )

    out.rule("13  measurement back-action and the campaign's own scope")
    out(f"  {'cell':<22}{'active_cpu_s':>14}{'wall_s':>10}{'cpu share':>11}"
        f"{'sampler_rc':>12}{'monitor_rc':>12}")
    out("  " + "-" * 81)
    for label, cell in sorted(campaign.cells.items()):
        share = cell.active_cpu_s / cell.wall_s if cell.wall_s else float("nan")
        out(
            f"  {label:<22}{cell.active_cpu_s:>14.3f}{cell.wall_s:>10.1f}{share:>11.4f}"
            f"{str(cell.record.get('sampler_rc')):>12}{str(cell.record.get('monitor_rc')):>12}"
        )
    out("")
    out(
        "  The sampler draws power it is measuring. Its own CPU time per run is recorded "
        "by the campaign and is reported above as measurement back-action, not corrected "
        "for: a correction would need a model of the board's power against occupancy, "
        "which is the thing being measured."
    )
    out(
        "  Every figure in this file is a MAXN_SUPER figure on one board at one "
        "uninstrumented ambient with one CPython CPU-only pipeline at one 2.0 s cycle. "
        "Nothing here supports a general statement about energy per decision on Orin "
        "Nano."
    )


def main() -> int:
    out = _bootstrap.Tee("validate_measured_baselines_output.txt")
    out.rule("MEASURED BASELINES -- all three, across the workload grid")
    out("INPUT ORIGIN: the MEASURED hardware campaign only. No synthetic fallback.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)
    out("")
    out(_bootstrap.measurement_model())

    try:
        campaign = _bootstrap.require_measured_campaign()
    except TraceDataAbsentError as error:
        out.rule("REFUSED -- no measured campaign")
        for line in str(error).splitlines():
            out(f"  {line}")
        out("")
        out(
            f"  Every measured figure this script would report stands at "
            f"{_bootstrap.MEASURED_PENDING}. Nothing was estimated, interpolated or "
            "illustrated, and the synthetic generator was not substituted."
        )
        for name in (
            "measured background floor power",
            "the floor-occupancy regression and the floor-sensitivity table",
            "the run-order drift table",
            "baseline 1 across the grid, both variants",
            "measured steady active power per cell",
            "measured wake-transient excess energy E_ramp",
            "baseline 2 against baseline 3 across the grid",
            "the measured rail-accounting ratio against load",
            "the between-run interval from the four reference repeats",
            "the reject-reason census",
            "measurement back-action: the sampler's own CPU time",
        ):
            out.pending(name, "the measured campaign is not on disk")
        out.check(
            "the MEASURED campaign is on disk and complete",
            False,
            "nothing was reported and nothing was substituted",
        )
        out("")
        out(f"raw output written to {out.save()}")
        return 1

    report(out, campaign)

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
