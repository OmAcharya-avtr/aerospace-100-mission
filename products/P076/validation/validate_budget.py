"""Verify every analytic budget line against a numerical propagation of its input.

This is the check the product exists for. Each analytic line is a linearised
sensitivity or a bound; each is checked by perturbing exactly the input it is
declared for, re-running the whole reference pipeline, and differencing the
point-ahead offsets. A sensitivity expression that does not reproduce a
numerical perturbation is wrong, and perturbing is the only way to find out.

Two of the four analytic lines were wrong when first written and this check is
what found them. Both failures are recorded in ``validation/VALIDATION.md``
with the numbers, and both are pinned as regression tests:

1. ``ephemeris_staleness_velocity`` used ``2 |a_rel_perp| dt / c`` alone. For
   two terminals at equal radius under two-body gravity the relative
   acceleration is exactly radial, so that term is identically zero and the
   model predicted 2.087e-22 rad where the propagation measured 2.381e-08 rad.
2. ``ephemeris_staleness_geometry`` used ``dt^3 / 6`` for the cubic coefficient
   of the stale position error. The correct coefficient is ``dt^3 / 3`` --
   ``-1/6`` from the position expansion plus ``+1/2`` from carrying the stale
   velocity forward -- so the bound sat a factor of two below the measurement.

The surviving failure is also reported rather than tuned away: see section 4.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, bootstrap, header

bootstrap()

from pointahead.budget import (  # noqa: E402
    BudgetInputs,
    build_budget,
    direction_noise_floor_rad,
    relative_acceleration,
    relative_jerk,
    verify_sensitivities,
)
from pointahead.geometry import norm, transverse  # noqa: E402
from pointahead.report import format_budget, format_sensitivity_checks  # noqa: E402
from pointahead.scenario import DEFAULT_BEAM, NAMED_SCENARIOS, scenario_family  # noqa: E402


def main() -> int:
    tee = Tee("validate_budget_output.txt")
    header(tee, "attributed error budget: every analytic line against a propagation")

    tee.rule("1. the relative acceleration of an equal-radius pair is exactly radial")
    tee("  a_rel = -(mu / r^3) d when |r_local| = |r_remote|, so it is parallel to")
    tee("  the separation vector. This is why the first-order-only staleness model")
    tee("  predicted zero for the commonest crosslink geometry there is.")
    tee("")
    tee(f"  {'scenario':<18s} {'|a_rel|':>12s} {'|a_rel_perp|':>14s} {'ratio':>12s} "
        f"{'|jerk_perp|':>14s}")
    tee(f"  {'-' * 18} {'-' * 12} {'-' * 14} {'-' * 12} {'-' * 14}")
    for name, builder in NAMED_SCENARIOS.items():
        link = builder()
        acceleration = relative_acceleration(link)
        jerk = relative_jerk(link)
        perpendicular = norm(transverse(acceleration, link.line_of_sight))
        ratio = perpendicular / norm(acceleration)
        tee(
            f"  {name:<18s} {norm(acceleration):>12.6f} {perpendicular:>14.6e} "
            f"{ratio:>12.3e} {norm(transverse(jerk, link.line_of_sight)):>14.6e}"
        )
        if name in ("leo_crossing", "leo_trailing"):
            tee.check(
                f"{name} has a radial relative acceleration to 1e-14 relative",
                ratio < 1.0e-14,
                f"transverse fraction {ratio:.3e}",
            )

    tee.rule("2. the full budget for the three named geometries")
    for builder in NAMED_SCENARIOS.values():
        link = builder()
        inputs = BudgetInputs(0.05, 30.0, 1, DEFAULT_BEAM, "momentum_boost", "closed_form", 1.0)
        tee("")
        tee(format_budget(build_budget(link, inputs)))
        tee("")
        tee(format_sensitivity_checks(verify_sensitivities(link, inputs)))

    tee.rule("3. the dominant term changes with ephemeris age")
    tee("  This is the product's point in one table. At a fresh state the velocity")
    tee("  knowledge dominates; a minute later the staleness does and is four orders")
    tee("  larger; the model errors never matter at orbital speeds.")
    tee("")
    link = NAMED_SCENARIOS["leo_crossing"]()
    tee(f"  {'age / s':>8s} {'total / urad':>14s} {'dominant term':<30s} {'share':>8s} "
        f"{'loss / dB':>12s}")
    tee(f"  {'-' * 8} {'-' * 14} {'-' * 30} {'-' * 8} {'-' * 12}")
    for age in (0.0, 1.0, 10.0, 30.0, 60.0, 300.0, 900.0, 3600.0):
        budget = build_budget(link, BudgetInputs(0.05, age, 1, DEFAULT_BEAM))
        tee(
            f"  {age:>8.0f} {budget.total_rss_rad * 1.0e6:>14.6e} "
            f"{budget.dominant_term:<30s} {budget.dominant_fraction:>8.4f} "
            f"{budget.beam.loss_db:>12.6e}"
        )

    tee.rule("4. every analytic line checked across geometries and configurations")
    tee("  Four analytic lines x 33 geometries x 4 iteration caps x 3 ephemeris ages.")
    tee("  Equality checks use a 1 per cent relative tolerance; bound checks require")
    tee("  the analytic value to sit above the measurement with 1 per cent slack.")
    tee("")
    links = [(label, builder()) for label, builder in NAMED_SCENARIOS.items()]
    links += [(link.label, link) for link in scenario_family(760, 30)]
    tally: dict[str, dict[str, int]] = {}
    tightness: dict[str, list[float]] = {}
    failures: list[tuple[str, int, float, str, float, float]] = []
    for name, item in links:
        for caps in (0, 1, 2, 3):
            for age in (0.0, 30.0, 120.0):
                inputs = BudgetInputs(0.05, age, caps, DEFAULT_BEAM)
                for check in verify_sensitivities(item, inputs):
                    record = tally.setdefault(check.name, {"pass": 0, "fail": 0})
                    tightness.setdefault(check.name, []).append(check.relative_difference)
                    if check.passed:
                        record["pass"] += 1
                    else:
                        record["fail"] += 1
                        failures.append(
                            (name, caps, age, check.name, check.analytic_rad, check.numeric_rad)
                        )
    total = sum(record["pass"] + record["fail"] for record in tally.values())
    tee(f"  {'line':<32s} {'pass':>6s} {'fail':>6s} {'median tightness':>18s} "
        f"{'worst':>12s}")
    tee(f"  {'-' * 32} {'-' * 6} {'-' * 6} {'-' * 18} {'-' * 12}")
    for key in sorted(tally):
        values = np.array(tightness[key])
        tee(
            f"  {key:<32s} {tally[key]['pass']:>6d} {tally[key]['fail']:>6d} "
            f"{float(np.median(values)):>18.6e} {float(values.max()):>12.6e}"
        )
    tee(f"  {'-' * 32} {'-' * 6} {'-' * 6}")
    tee(f"  {'total':<32s} {total - len(failures):>6d} {len(failures):>6d}")

    tee("")
    if failures:
        tee(f"  FAILING COMBINATIONS: {len(failures)} of {total}")
        tee("")
        tee(f"  {'geometry':<14s} {'caps':>5s} {'age':>6s} {'line':<32s} "
            f"{'analytic':>13s} {'numeric':>13s}")
        tee(f"  {'-' * 14} {'-' * 5} {'-' * 6} {'-' * 32} {'-' * 13} {'-' * 13}")
        for name, caps, age, key, analytic, numeric in failures:
            tee(
                f"  {name:<14s} {caps:>5d} {age:>6.0f} {key:<32s} "
                f"{analytic:>13.6e} {numeric:>13.6e}"
            )
        caps_set = sorted({caps for _, caps, _, _, _, _ in failures})
        lines_set = sorted({key for _, _, _, key, _, _ in failures})
        worst_ratio = max(numeric / analytic for _, _, _, _, analytic, numeric in failures)
        tee("")
        tee(f"  all failures are on line(s)        : {lines_set}")
        tee(f"  all failures are at iteration caps : {caps_set}")
        tee(f"  worst numeric / analytic ratio     : {worst_ratio:.4f}")
        tee("  typical magnitude of both sides    : 1e-14 rad")
        tee(f"  float64 direction resolution       : "
            f"{direction_noise_floor_rad(links[0][1]):.6e} rad")
        tee("")
        tee("  Diagnosis, stated rather than tuned away. At iteration cap zero the")
        tee("  receive and transmit light-time truncation errors have opposite signs")
        tee("  and nearly cancel in the point-ahead offset, so the predicted error is")
        tee("  SMALLER than the arithmetic noise of the four unit directions it is")
        tee("  measured from. Both sides of the comparison are then quantisation")
        tee("  patterns of order 1e-14 rad and the bound holds or fails at random.")
        tee("  The honest statement is that the light-time iteration budget line")
        tee("  cannot be verified at iteration cap zero in float64 for these")
        tee("  geometries, and that at one or more iterations it verifies on every")
        tee("  combination measured. The engineering consequence is the opposite of")
        tee("  alarming: at cap zero the line is 1e-14 rad, which is nine orders")
        tee("  below the dominant term of every budget in section 3.")
        tee("")
        tee("  The floor constant was widened once, from 4 to 8 sqrt(3), when the")
        tee("  first derivation was found to ignore the per-component contribution")
        tee("  and the outer subtraction. It has not been widened again to make these")
        tee("  combinations pass.")
    else:
        tee("  FAILING COMBINATIONS: none")
    tee.check(
        "every analytic line verifies on every geometry and configuration measured",
        not failures,
        f"{len(failures)} of {total} combinations failed; see the diagnosis above",
    )
    tee.check(
        "no analytic line fails at one or more light-time iterations",
        all(caps == 0 for _, caps, _, _, _, _ in failures),
        f"failing iteration caps: "
        f"{sorted({caps for _, caps, _, _, _, _ in failures}) or 'none'}",
    )
    tee.check(
        "the light-time line is the only line that ever fails",
        {key for _, _, _, key, _, _ in failures} <= {"light_time_iteration_residual"},
        f"lines failing: {sorted({key for _, _, _, key, _, _ in failures}) or 'none'}",
    )
    tee.check(
        "the three analytic lines that enter a realistic budget all verify",
        all(
            tally[key]["fail"] == 0
            for key in (
                "relative_velocity_knowledge",
                "ephemeris_staleness_velocity",
                "ephemeris_staleness_geometry",
            )
        ),
        "velocity knowledge, staleness velocity and staleness geometry",
    )

    tee.rule("5. the beam context: the same total on two different beams")
    tee("  A point-ahead error is not a result until it is divided by a beam.")
    tee("")
    budget = build_budget(link, BudgetInputs(0.05, 300.0, 2, DEFAULT_BEAM))
    total = budget.total_rss_rad
    tee(f"  total point-ahead error            : {total * 1.0e6:.6f} urad")
    tee("")
    tee(f"  {'beam':<26s} {'divergence / urad':>19s} {'fraction':>12s} {'loss / dB':>12s} "
        f"{'headroom':>12s}")
    tee(f"  {'-' * 26} {'-' * 19} {'-' * 12} {'-' * 12} {'-' * 12}")
    from pointahead.terminal import BeamContext

    for wavelength, waist, label in (
        (1550.0e-9, 0.010, "1550 nm, 10 mm waist"),
        (1550.0e-9, 0.025, "1550 nm, 25 mm waist"),
        (1064.0e-9, 0.075, "1064 nm, 75 mm waist"),
        (1064.0e-9, 0.150, "1064 nm, 150 mm waist"),
    ):
        item = build_budget(link, BudgetInputs(0.05, 300.0, 2, BeamContext(wavelength, waist)))
        tee(
            f"  {label:<26s} {item.beam.divergence_half_angle_rad * 1.0e6:>19.6f} "
            f"{item.beam.error_fraction_of_beam:>12.6e} {item.beam.loss_db:>12.6e} "
            f"{item.beam.headroom:>12.4e}"
        )
    tee("")
    tee("  The total is identical in all four rows. The decibel cost spans three")
    tee("  orders of magnitude. That is the whole argument for carrying the beam")
    tee("  alongside the number.")

    tee.summarise_failures()
    tee("")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
