"""The worked example quoted in README.md, executed so the output cannot drift.

Everything this script prints appears in README.md under "A worked example" and
"Install and first run". It is re-run as the last step before hand-off and the
README text is replaced with what it printed, because two products of this
mission caught their own wrong README numbers exactly this way.
"""

from __future__ import annotations

from _bootstrap import Tee, bootstrap, header

bootstrap()

from pointahead import (  # noqa: E402
    BudgetInputs,
    build_budget,
    closed_form_point_ahead,
    decompose_point_ahead,
    leo_crossing_pair,
    reference_point_ahead,
    score_closed_form,
    verify_sensitivities,
)
from pointahead.report import (  # noqa: E402
    format_budget,
    format_decomposition,
    format_score,
    format_sensitivity_checks,
)
from pointahead.terminal import BeamContext  # noqa: E402


def main() -> int:
    tee = Tee("worked_example_output.txt")
    header(tee, "worked example: the code and the output quoted in README.md")

    tee.rule("the code")
    tee("""
    from pointahead import (
        BudgetInputs,
        build_budget,
        closed_form_point_ahead,
        decompose_point_ahead,
        leo_crossing_pair,
        reference_point_ahead,
        score_closed_form,
        verify_sensitivities,
    )
    from pointahead.terminal import BeamContext

    link = leo_crossing_pair()
    beam = BeamContext(wavelength_m=1550e-9, waist_m=0.025, label="1550 nm, 25 mm")
    inputs = BudgetInputs(
        sigma_velocity_per_axis_mps=0.05,
        ephemeris_age_s=300.0,
        light_time_iterations=1,
        beam=beam,
        angle_method="closed_form",
        loss_budget_db=1.0,
    )

    print(closed_form_point_ahead(link).total_urad)
    print(reference_point_ahead(link).total_urad)
    print(build_budget(link, inputs).dominant_term)
    """)

    link = leo_crossing_pair()
    beam = BeamContext(wavelength_m=1550e-9, waist_m=0.025, label="1550 nm, 25 mm")
    inputs = BudgetInputs(
        sigma_velocity_per_axis_mps=0.05,
        ephemeris_age_s=300.0,
        light_time_iterations=1,
        beam=beam,
        angle_method="closed_form",
        loss_budget_db=1.0,
    )

    tee.rule("the three printed lines")
    closed = closed_form_point_ahead(link)
    reference = reference_point_ahead(link)
    budget = build_budget(link, inputs)
    tee(f"  {closed.total_urad!r}")
    tee(f"  {reference.total_urad!r}")
    tee(f"  {budget.dominant_term!r}")

    tee.rule("the full reports")
    tee(format_decomposition(decompose_point_ahead(link)))
    tee("")
    tee(format_score(score_closed_form(link)))
    tee("")
    tee(format_budget(budget))
    tee("")
    tee(format_sensitivity_checks(verify_sensitivities(link, inputs)))

    tee.rule("the three facts the README leads with")
    tee(f"  point-ahead angle                  : {reference.total_urad:.6f} urad")
    tee(f"  dominant budget line               : {budget.dominant_term}")
    tee(f"  share of the total                 : {budget.dominant_fraction:.4f}")
    tee(f"  pointing loss on this beam         : {budget.beam.loss_db:.6f} dB")
    tee(f"  error affordable at 1 dB           : "
        f"{budget.beam.affordable_error_rad * 1.0e6:.6f} urad")
    tee(f"  headroom                           : {budget.beam.headroom:.4f} x")

    tee.check(
        "the closed form and the reference agree to 1e-8 of the angle",
        abs(closed.total_rad - reference.total_rad) / reference.total_rad < 1.0e-8,
        f"relative difference "
        f"{abs(closed.total_rad - reference.total_rad) / reference.total_rad:.3e}",
    )
    tee.check(
        "the budget total is dominated by a single line",
        budget.dominant_fraction > 0.5,
        f"{budget.dominant_term} at {budget.dominant_fraction:.4f}",
    )

    tee.summarise_failures()
    tee("")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
