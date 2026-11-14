"""The three baselines against a SYNTHETIC truth known in closed form.

INPUT ORIGIN: SYNTHETIC. The generator's model is declared in
``src/powerenvelope/synthetic.py`` and the closed forms are derived from those
declared constants in ``tests/fixtures/synthetic_campaign.py``, written out on
paper there rather than by calling this product's integrator. Nothing below is a
measurement of any board.

This script runs BEFORE the learned component is benchmarked anywhere, because
an under-strength baseline inflates a learned win, and because a baseline that
has not been validated against a known truth is not a baseline.

Agreement is reported to the declared ``CLOSED_FORM_REL`` of 1e-9 and not to
bit-exactness. The reason is in the fixture's docstring: the campaign writes its
power column with ``%.3f``, so the value the loader reads is the three-decimal
rounding of ``V * I / 1000``, and the exact closed form cannot agree with it
better than about 1e-11 relative.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True


import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from powerenvelope.baselines import (  # noqa: E402
    RAMP_WINDOW_S,
    between_run_spread,
    duty_cycle_model,
    energy_per_inference,
    floor_power,
    ground_truth,
    rail_accounting_curve,
    ramp_energy,
)
from powerenvelope.decomposition import decompose  # noqa: E402
from powerenvelope.energy import cell_energy, whole_run_energy  # noqa: E402
from powerenvelope.report import (  # noqa: E402
    baseline1_lines,
    baseline2_lines,
    baseline3_lines,
    floor_lines,
    rail_lines,
    spread_lines,
)

from fixtures import synthetic_campaign as closed  # noqa: E402

#: The duties whose active-window closed form applies. Duty 1.00 is excluded on
#: purpose: there the active windows of consecutive cycles are contiguous and
#: merge into one, so the per-cycle formula does not hold and the identity
#: ``E_active == E_total`` is checked instead.
CLOSED_FORM_DUTIES = (0.10, 0.25, 0.50, 0.75)


def main() -> int:
    out = _bootstrap.Tee("validate_baselines_synthetic_output.txt")
    out.rule("BASELINES vs a SYNTHETIC closed form")
    out("INPUT ORIGIN: SYNTHETIC, from src/powerenvelope/synthetic.py. Not a measurement.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)
    out("")
    out(_bootstrap.measurement_model())
    out("")
    out(f"closed-form relative tolerance: {closed.CLOSED_FORM_REL:g}")
    out(
        "The power column is written with %.3f, exactly as data/campaign.py writes it, so "
        "the exact closed form and the file cannot agree better than about 1e-11 relative."
    )

    campaign = _bootstrap.synthetic_campaign()
    cycles = closed.CYCLES

    out.rule("1  the declared synthetic model, for reference")
    out(f"  floor power            : {closed.floor_mw():.6f} mW")
    out(f"  active excess          : {closed.active_excess_mw():.6f} mW")
    out(f"  ramp excess            : {closed.ramp_excess_mw():.6f} mW")
    out(f"  ramp duration          : {RAMP_WINDOW_S:.3f} s (the product's settling window)")
    out(f"  cycles per run         : {cycles}")
    out("  These are generator constants. They are not measurements of anything.")

    out.rule("2  RUN energy against the closed form, and the sampler's wider span")
    reference_cell = campaign.require(_bootstrap.HEADLINE_CELL)
    run_window = cell_energy(reference_cell)[0]
    span = whole_run_energy(reference_cell.power)
    out(f"  run    : {run_window.duration_s:.6f} s, {run_window.energy_mj:.6f} mJ")
    out(f"  sampler: {span.duration_s:.6f} s, {span.energy_mj:.6f} mJ")
    out.check(
        "the sampler's span is wider than the run, as the campaign intends",
        span.duration_s > run_window.duration_s,
        f"{span.duration_s - run_window.duration_s:.3f} s wider; integrating the span "
        "and calling it the run cost 1.1 per cent on the measured campaign before it "
        "was found, and the generator now reproduces the bracketing so a test can "
        "catch it",
    )
    out("")
    for dim in (48, 64, 96):
        for duty in (*CLOSED_FORM_DUTIES, 1.00):
            label = f"w{dim}_d{int(round(duty * 100)):03d}"
            cell = campaign.require(label)
            expected = closed.total_energy_mj(duty, cycles)
            measured = cell_energy(cell)[0].energy_mj
            error = abs(measured / expected - 1.0)
            out.check(
                f"E_total({label}) matches the closed form",
                error <= closed.CLOSED_FORM_REL,
                f"{measured:.6f} vs {expected:.6f} mJ, relative error {error:.3e}",
            )

    out.rule("3  active-window energy against the closed form")
    for duty in CLOSED_FORM_DUTIES:
        label = f"w64_d{int(round(duty * 100)):03d}"
        cell = campaign.require(label)
        expected = closed.active_energy_mj(duty, cycles)
        measured = cell_energy(cell)[1].energy_mj
        error = abs(measured / expected - 1.0)
        out.check(
            f"E_active({label}) matches the per-cycle closed form",
            error <= closed.CLOSED_FORM_REL,
            f"{measured:.6f} vs {expected:.6f} mJ, relative error {error:.3e}",
        )
    full = campaign.require("w64_d100")
    whole, active = cell_energy(full)
    out.check(
        "at duty 1.00 the active energy is the whole-run energy",
        abs(active.energy_mj / whole.energy_mj - 1.0) <= 1e-12,
        f"{active.energy_mj:.6f} vs {whole.energy_mj:.6f} mJ",
    )

    out.rule("4  baseline 1, energy per inference")
    cell = campaign.require(_bootstrap.HEADLINE_CELL)
    for variant in ("measured-active", "nominal-rail"):
        baseline = energy_per_inference(cell, variant)
        for line in baseline1_lines(baseline):
            out(f"  {line}")
        out("")
    measured_variant = energy_per_inference(cell, "measured-active")
    nominal_variant = energy_per_inference(cell, "nominal-rail")
    expected_power = closed.active_energy_mj(0.50, cycles) / (cycles * 1.0)
    out.check(
        "the mean active power is the closed-form active energy over the active duration",
        abs(measured_variant.mean_active_power_mw / expected_power - 1.0)
        <= closed.CLOSED_FORM_REL,
        f"{measured_variant.mean_active_power_mw:.6f} vs {expected_power:.6f} mW",
    )
    out.check(
        "the nominal-rail variant is below the measured variant by exactly the voltage ratio",
        abs(
            measured_variant.mean_active_power_mw
            / nominal_variant.mean_active_power_mw
            / closed.nominal_factor()
            - 1.0
        )
        <= 1e-9,
        f"ratio {measured_variant.mean_active_power_mw / nominal_variant.mean_active_power_mw:.9f} "
        f"vs {closed.nominal_factor():.9f}",
    )
    out.check(
        "the inference count is the closed-form count",
        measured_variant.n_inferences == closed.inferences(0.50, 64, cycles),
        f"{measured_variant.n_inferences}",
    )

    out.rule("5  baseline 2, the analytic duty-cycle model")
    floor = floor_power(campaign)
    for line in floor_lines(floor):
        out(f"  {line}")
    out("")
    out.check(
        "the measured floor equals the generator's declared floor",
        abs(floor.mean_mw / closed.floor_mw() - 1.0) <= closed.CLOSED_FORM_REL,
        f"{floor.mean_mw:.6f} vs {closed.floor_mw():.6f} mW",
    )
    model = duty_cycle_model(campaign, _bootstrap.HEADLINE_CELL, floor=floor)
    for line in baseline2_lines(model):
        out(f"  {line}")
    out("")
    out.check(
        "the three components sum to the cycle energy",
        abs(
            (model.e_sleep_mj + model.e_ramp_mj + model.e_active_mj) / model.e_cycle_mj
            - 1.0
        )
        <= 1e-12,
        f"{model.e_sleep_mj:.4f} + {model.e_ramp_mj:.4f} + {model.e_active_mj:.4f} "
        f"= {model.e_cycle_mj:.4f} mJ",
    )
    ramp = ramp_energy(cell)
    ideal = closed.ramp_excess_mw() * RAMP_WINDOW_S
    out.check(
        "the measured ramp excess is within 15 per cent of the declared excess energy",
        abs(ramp.energy_mj / ideal - 1.0) <= 0.15,
        f"{ramp.energy_mj:.6f} vs {ideal:.6f} mJ over {ramp.n_wakes} wake(s); the "
        "difference is the trapezoid straddling the step at the window edge and the "
        "steady level being estimated from the samples after it",
    )
    low = ramp_energy(campaign.require("w64_d010"))
    out.check(
        "the ramp is reported as NOT separable at duty 0.10 rather than estimated",
        not low.separable and low.energy_mj == 0.0,
        "the 0.2 s active phase is shorter than three settling windows",
    )
    worst = 0.0
    for grid_cell in campaign.grid_cells():
        analytic = duty_cycle_model(campaign, grid_cell.label, floor=floor)
        truth = ground_truth(grid_cell)
        worst = max(
            worst,
            abs(
                analytic.energy_per_useful_decision_mj
                / truth.energy_per_useful_decision_mj
                - 1.0
            ),
        )
    out.check(
        "the analytic model is within 10 per cent of ground truth on all 15 grid cells",
        worst <= 0.10,
        f"largest relative gap {worst:.4f}",
    )

    out.rule("6  the measured rail-accounting ratio against load")
    rail = rail_accounting_curve(campaign)
    for line in rail_lines(rail):
        out(f"  {line}")
    out("")
    out.check(
        "the fit covers every cell of the campaign",
        len(rail.cells) == len(campaign.cells),
        f"{len(rail.cells)} cell(s)",
    )
    out.check(
        "the ratio is between 0 and 1 on every cell",
        all(0.0 < value < 1.0 for value in rail.ratio),
        f"range {min(rail.ratio):.6f} to {max(rail.ratio):.6f}",
    )

    out.rule("7  baseline 3, ground truth, and the between-run interval")
    truth = ground_truth(cell)
    for line in baseline3_lines(truth):
        out(f"  {line}")
    out("")
    expected_total = closed.total_energy_mj(0.50, cycles)
    expected_accepted = closed.accepted(0.50, 64, cycles)
    out.check(
        "the ground-truth numerator is the closed-form whole-run energy",
        abs(truth.energy_mj / expected_total - 1.0) <= closed.CLOSED_FORM_REL,
        f"{truth.energy_mj:.6f} vs {expected_total:.6f} mJ",
    )
    out.check(
        "the ground-truth denominator is the closed-form accepted count",
        truth.accepted == expected_accepted,
        f"{truth.accepted} == {expected_accepted}",
    )
    spread = between_run_spread(campaign)
    for line in spread_lines(spread):
        out(f"  {line}")
    out("")
    out.check(
        "the interval is taken over four repeats of the reference cell",
        spread.n == 4 and len(spread.cells) == 4,
        f"cells {', '.join(spread.cells)}",
    )
    out.check(
        "the four byte-identical synthetic repeats give a spread of exactly zero",
        spread.sd_mj == 0.0,
        "this is the correct answer for this input and it is not worked around; the "
        "jittered fixture in tests/conftest.py exercises a positive spread",
    )

    out.rule("8  the headline, on synthetic data, with its closure residual")
    for variant in ("measured-active", "nominal-rail"):
        result = decompose(cell, variant)
        for line in result.lines():
            out(f"  {line}")
        product = result.product_of_named_factors()
        out.check(
            f"the three named factors multiply to F exactly ({variant})",
            abs(product / result.factor - 1.0) <= 1e-12,
            f"product {product:.12f} vs F {result.factor:.12f}; there is no residual "
            "term under the declared-phase partition",
        )
        out.check(
            f"the declared phases partition the run ({variant})",
            abs(result.partition_residual) <= 1e-12,
            f"partition residual {result.partition_residual:.3e}; the synthetic run "
            "ends exactly on a cycle boundary, so it is zero here. The measured run "
            "ends a few milliseconds later and tests/test_measured_campaign.py bounds "
            "that tail.",
        )
        out.check(
            f"the phase occupancy is 1 on the synthetic pipeline ({variant})",
            abs(result.occupancy.occupancy - 1.0) <= 1e-9,
            f"occupancy {result.occupancy.occupancy:.9f}; a DIAGNOSTIC, not a factor. "
            "On the measured campaign it is 0.999723.",
        )
        out("")

    out.rule("9  what this script does NOT establish")
    out(
        "  Nothing here is a measurement. The agreement above says the integrator, the "
        "window derivation and the decomposition compute what their docstrings say, on "
        "a trace whose truth is known. It says nothing about any board. The measured "
        "figures are in validate_measured_headline.py, validate_measured_baselines.py "
        "and validate_measured_learned.py, which read data/traces/ and refuse, exiting "
        "non-zero, if it is absent or incomplete."
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
