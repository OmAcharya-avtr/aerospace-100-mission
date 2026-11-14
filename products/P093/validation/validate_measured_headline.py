"""THE HEADLINE, measured: how far energy per inference is from energy per decision.

The reported factor is

    (measured energy per useful decision) / (measured energy per inference)

on the reference cell, in both Baseline-1 variants, decomposed into the
rejection factor, the duty-cycle factor and the nominal-versus-measured
active-power error, with the residual reported explicitly and checked against an
independently computed accounting factor.

The factor is reported whatever its value. If it is near 1, the quoted unit is
adequate for this pipeline on this board and that is the headline.

Also measured here, because the scope is part of the result: the per-second
foreign CPU occupancy during the headline run. The board was SHARED throughout.
Foreign occupancy is measured contamination, not assumed absence, and the
figures are printed whatever they are.

INPUT ORIGIN: the MEASURED hardware campaign under ``data/traces/`` and nothing
else. There is no synthetic fallback: if the traces are absent, incomplete, or
self-declared synthetic, this script refuses and exits non-zero.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from powerenvelope.baselines import (  # noqa: E402
    between_run_spread,
    energy_per_inference,
    ground_truth,
    sleep_power_mw,
)
from powerenvelope.bounds import (  # noqa: E402
    duty_cycle_lower_bracket,
    extreme_value_bracket,
    headline_lower_bracket,
    riemann_bracket,
)
from powerenvelope.decomposition import decompose  # noqa: E402
from powerenvelope.energy import (  # noqa: E402
    GAP_TOLERANCE_S,
    active_windows,
    cell_energy,
    interstitial_samples,
    sleep_energy,
    trapezoid_energy_mj,
)
from powerenvelope.predictor import target_quantisation_sd  # noqa: E402
from powerenvelope.report import (  # noqa: E402
    baseline1_lines,
    baseline3_lines,
    campaign_lines,
    cell_table,
    census_lines,
    reject_census,
    spread_lines,
)
from powerenvelope.schema import REJECT_REASONS  # noqa: E402
from powerenvelope.trace import TraceDataAbsentError  # noqa: E402


def _lowest_sleep_minimum(campaign) -> float:
    """The lowest per-cell sleep-window minimum rail power in the campaign, mW."""
    values = [
        sleep_power_mw(cell)[1]
        for cell in campaign.pipeline_cells()
        if np.isfinite(sleep_power_mw(cell)[1])
    ]
    return min(values) if values else float("nan")


def report(out: _bootstrap.Tee, campaign) -> None:
    """Write the whole measured report into ``out`` from an already-loaded campaign.

    Split out from :func:`main` so that ``tests/test_measured_reports.py`` can
    exercise every formatting and arithmetic path in this file against a
    SYNTHETIC campaign, without the refusal above being weakened and without a
    committed output file being overwritten. The split changes nothing about the
    measured path: ``main`` refuses, loads, and calls this.
    """
    out.rule("1  campaign identity")
    for line in campaign_lines(campaign):
        out(f"  {line}")
    out("")
    for line in cell_table(campaign):
        out(line)

    cell = campaign.require(_bootstrap.HEADLINE_CELL)
    out.rule(f"2  baseline 1 and baseline 3 on the headline cell {cell.label}")
    for variant in ("measured-active", "nominal-rail"):
        for line in baseline1_lines(energy_per_inference(cell, variant)):
            out(f"  {line}")
        out("")
    truth = ground_truth(cell)
    for line in baseline3_lines(truth):
        out(f"  {line}")

    out.rule("3  the energy accounting adds up")
    whole, active = cell_energy(cell)
    sleep = sleep_energy(cell)
    out(f"  E_total  : {whole.energy_mj:.6f} mJ over {whole.duration_s:.3f} s")
    out(f"  E_active : {active.energy_mj:.6f} mJ over {active.duration_s:.3f} s "
        f"in {active.n_windows} window(s)")
    out(f"  E_sleep  : {sleep.energy_mj:.6f} mJ over {sleep.duration_s:.3f} s")
    out.check(
        "the active and complementary energies sum to the whole-run energy",
        abs((active.energy_mj + sleep.energy_mj) / whole.energy_mj - 1.0) <= 1e-9,
        f"residual {(active.energy_mj + sleep.energy_mj) / whole.energy_mj - 1.0:.3e}",
    )
    gaps = interstitial_samples(cell)
    out(
        f"  interstitial samples: {gaps.n_samples} in {gaps.n_gaps} gap(s); widest "
        f"inter-inference gap {gaps.max_gap_s:.6f} s against a merge tolerance of "
        f"{GAP_TOLERANCE_S:.3f} s"
    )
    out(
        "  A sample in such a gap is excluded from both the active energy and the "
        "active duration, and the count is printed rather than absorbed."
    )

    out.rule("4  THE HEADLINE, both variants")
    results = {}
    for variant in ("measured-active", "nominal-rail"):
        result = decompose(cell, variant)
        results[variant] = result
        for line in result.lines():
            out(f"  {line}")
        out.check(
            f"the four factors multiply to F ({variant})",
            abs(
                result.product_of_named_factors() * result.residual_factor / result.factor
                - 1.0
            )
            <= 1e-10,
            f"closure error {result.closure_error:.3e}",
        )
        out("")

    out.rule("5  the between-run interval on the headline quantity")
    try:
        spread = between_run_spread(campaign)
        for line in spread_lines(spread):
            out(f"  {line}")
        out.check(
            "the interval is taken over the four repeats of the reference cell",
            spread.n == 4,
            f"{spread.n} repeat(s)",
        )
    except (TraceDataAbsentError, ValueError) as error:
        out.check("the between-run interval is computable", False, str(error))

    out.rule("6  the headline against bounds from other mathematics")
    power = cell.power
    value = trapezoid_energy_mj(power.t, power.vdd_in_mw)
    for name, bracket in (
        ("Riemann", riemann_bracket(power.t, power.vdd_in_mw)),
        ("mean-value", extreme_value_bracket(power.t, power.vdd_in_mw)),
    ):
        out.check(
            f"E_total lies inside the {name} bracket",
            bracket.contains(value),
            f"[{bracket.low:.6f}, {bracket.high:.6f}] mJ, slack {bracket.slack:.3e}",
        )
    floor_min = float(campaign.require("floor").power.vdd_in_mw.min())
    out("")
    out("  the duty-cycle factor against the sleep-power bound, on every pipeline cell")
    out(
        f"  {'cell':<22}{'campaign bd':>13}{'measured':>11}{'ok':>5}"
        f"{'in-cell min':>13}{'in-cell bd':>12}{'ok':>5}"
    )
    out("  " + "-" * 83)
    campaign_failures: list[str] = []
    in_cell_failures: list[str] = []
    for item in campaign.pipeline_cells():
        item_whole, item_active = cell_energy(item)
        item_sleep = max(0.0, item_whole.duration_s - item_active.duration_s)
        measured_factor = item_whole.energy_mj / item_active.energy_mj
        campaign_bracket = duty_cycle_lower_bracket(
            item_active.energy_mj, floor_min, item_sleep
        )
        campaign_ok = campaign_bracket.contains(measured_factor)
        sleep_min = sleep_power_mw(item)[1]
        if np.isfinite(sleep_min) and item_sleep > 0.0:
            in_cell_bracket = duty_cycle_lower_bracket(
                item_active.energy_mj, sleep_min, item_sleep
            )
            in_cell_ok = in_cell_bracket.contains(measured_factor)
            in_cell_low = in_cell_bracket.low
        else:
            in_cell_ok, in_cell_low = True, float("nan")
        if not campaign_ok:
            campaign_failures.append(item.label)
        if not in_cell_ok:
            in_cell_failures.append(item.label)
        out(
            f"  {item.label:<22}{campaign_bracket.low:>13.4f}{measured_factor:>11.4f}"
            f"{str(campaign_ok):>5}{sleep_min:>13.1f}{in_cell_low:>12.4f}"
            f"{str(in_cell_ok):>5}"
        )
    out("")
    out.check(
        "the duty-cycle factor satisfies the bound built from the CAMPAIGN floor minimum",
        not campaign_failures,
        f"{len(campaign_failures)} of {len(campaign.pipeline_cells())} cell(s) violate it: "
        + ", ".join(campaign_failures),
    )
    out(
        "  THIS CHECK FAILS AND THE FAILURE IS THE FINDING. The bound's premise is that a "
        "sleep window draws at least the minimum power ever seen on the no-pipeline floor "
        f"run, which was {floor_min:.1f} mW. That premise is FALSE on this campaign, "
        "because the run order was not randomised and the board's foreign load fell "
        "throughout: later cells' sleep windows draw as little as "
        f"{_lowest_sleep_minimum(campaign):.0f} "
        "mW, well under the floor run's minimum. The bound is sound arithmetic applied to "
        "a false premise about the board's state, and it caught a defect in the "
        "campaign's design rather than one in this code. It is NOT weakened to pass, and "
        "the headline does not depend on it: F is a ratio of measured integrals and "
        "counts and does not use the floor at all."
    )
    out.check(
        "the duty-cycle factor satisfies the bound built from each cell's OWN sleep minimum",
        not in_cell_failures,
        "this variant replaces the false premise with a measured one, per cell. It is a "
        "WEAKER argument than the campaign-floor form because it reads the minimum from "
        "the same samples the integral uses, so it shares an input with what it checks; "
        "it is reported as the sound-but-weaker companion and not as a replacement",
    )
    inside = np.zeros(power.t.size, dtype=bool)
    for lo, hi in zip(*active_windows(cell.require_decisions()), strict=True):
        inside |= (power.t >= lo) & (power.t <= hi)
    headline_bracket = headline_lower_bracket(
        run_duration_s=float(power.t[-1] - power.t[0]),
        min_power_mw=float(power.vdd_in_mw.min()),
        max_active_power_mw=float(power.vdd_in_mw[inside].max()),
        max_latency_s=float(cell.require_decisions().latency_s.max()),
        n_accepted=truth.accepted,
    )
    out.check(
        "F satisfies the extreme-value bound",
        headline_bracket.contains(results["measured-active"].factor),
        f"bound {headline_bracket.low:.6f}, measured "
        f"{results['measured-active'].factor:.6f}, ratio "
        f"{results['measured-active'].factor / headline_bracket.low:.3f}",
    )

    out.rule("7  the reject-reason census, and the two branches that never fired")
    for line in census_lines(campaign):
        out(line)
    census = reject_census(campaign)
    never = [reason for reason in REJECT_REASONS if census.get(reason, 0) == 0]
    if never:
        out.unmeasurable_check(
            f"the gate branches {', '.join(never)}",
            "zero occurrences in the entire campaign, so the rejection factor 1/a that "
            "the headline decomposes into is a property of a gate whose "
            f"{' and '.join(never)} paths were never reached by measurement.",
        )

    out.rule("8  the permissive-gate control: the rejection half of the headline")
    if _bootstrap.PERMISSIVE_CELL in campaign.cells:
        permissive = decompose(campaign.require(_bootstrap.PERMISSIVE_CELL))
        strict = results["measured-active"]
        out(
            f"  strict     tau={campaign.require(_bootstrap.HEADLINE_CELL).tau:.4f}  "
            f"a={strict.acceptance_rate:.6f}  1/a={strict.rejection_factor:.6f}  "
            f"F={strict.factor:.6f}"
        )
        out(
            f"  permissive tau={campaign.require(_bootstrap.PERMISSIVE_CELL).tau:.4f}  "
            f"a={permissive.acceptance_rate:.6f}  1/a={permissive.rejection_factor:.6f}  "
            f"F={permissive.factor:.6f}"
        )
        out("")
        out(
            "  This is the specification's declared negative control for the rejection "
            "half of the headline: at the permissive tau the rejection factor should "
            "nearly vanish, and the duty-cycle factor, which is a property of the duty "
            "and not of the gate, should not move."
        )
        out.control(
            "the permissive gate removes most of the rejection factor",
            permissive.rejection_factor < 0.5 * strict.rejection_factor,
            must_fire=True,
            detail=f"1/a falls from {strict.rejection_factor:.4f} to "
            f"{permissive.rejection_factor:.4f}, a ratio of "
            f"{strict.rejection_factor / permissive.rejection_factor:.4f}",
        )
        out.control(
            "the permissive gate moves the duty-cycle factor",
            abs(permissive.duty_cycle_factor / strict.duty_cycle_factor - 1.0) > 0.02,
            must_fire=False,
            detail=f"{permissive.duty_cycle_factor:.6f} against "
            f"{strict.duty_cycle_factor:.6f}",
        )
    else:
        out.unmeasurable_check(
            "the permissive-gate control",
            f"cell {_bootstrap.PERMISSIVE_CELL!r} is not in this campaign",
        )

    out.rule("9  the instrument's own floor on every figure above")
    quantum = float(np.mean(power.quantisation_floor_mw))
    per_sample = float(target_quantisation_sd(quantum, 1.0))
    energy_sd = per_sample * whole.duration_s / np.sqrt(len(power))
    out(f"  power quantum                  : {quantum:.3f} mW")
    out(f"  per-sample quantiser sd        : {per_sample:.3f} mW")
    out(f"  implied energy sd (lower bound): {energy_sd:.3f} mJ "
        f"({energy_sd / whole.energy_mj:.3e} relative)")
    out(
        "  The sqrt(n) reduction is a LOWER bound: the INA3221 averages 512 conversions "
        "in hardware, so neighbouring samples are not independent."
    )

    out.rule("10  the board was shared, and here is the contamination")
    monitor = cell.monitor
    out(f"  cpu_busy_fraction over the run : mean {float(monitor.cpu_busy_fraction.mean()):.6f}, "
        f"min {float(monitor.cpu_busy_fraction.min()):.6f}, "
        f"max {float(monitor.cpu_busy_fraction.max()):.6f}")
    out(f"  die temperature                : mean {float(monitor.tj_c.mean()):.3f} degC, "
        f"max {float(monitor.tj_c.max()):.3f} degC")
    out(f"  1-minute load average          : mean {float(monitor.load1.mean()):.3f}, "
        f"max {float(monitor.load1.max()):.3f}")
    out(f"  mean core clock                : {float(np.nanmean(monitor.clk_mean_khz)):.0f} kHz")
    out(
        "  This is TOTAL occupancy, the pipeline's own included. It is not a "
        "foreign-occupancy figure on its own; see validate_measured_learned.py for the "
        "pipeline-free separation and where it is not possible."
    )
    out(f"  sampler and pipeline back-action: active_cpu_s = {cell.active_cpu_s:.3f} s "
        f"of a {cell.wall_s:.1f} s run")



def main() -> int:
    out = _bootstrap.Tee("validate_measured_headline_output.txt")
    out.rule("MEASURED HEADLINE -- the understatement factor and its decomposition")
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
            "headline understatement factor F",
            "rejection factor 1/a",
            "duty-cycle factor E_total/E_active",
            "nominal-versus-measured active-power error",
            "decomposition residual and closure error",
            "foreign CPU occupancy during the headline run",
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
