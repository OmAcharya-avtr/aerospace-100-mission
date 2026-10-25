"""The mission-profile ageing model, including the check it fails.

What the profile does
---------------------

A mission phase declares a duration, a per-block duty cycle and a per-block
stress factor, and a block's effective age over the mission is

    ``age_eff = sum over phases of duration * duty_cycle * stress_factor``

This is the cumulative-exposure model (Nelson, W., 1990, *Accelerated Testing:
Statistical Models, Test Plans, and Data Analyses*, Wiley, chapter 10). It is
**exact** for an exponential block, because a constant hazard rate makes the
cumulative hazard linear in age and the sum of phase contributions is the
integral.

The check this script fails
---------------------------

For a **non-exponential** block it is an approximation, and it is not the
approximation a reader is likely to assume. A reader who sees "stress factor
3.0" may reasonably expect the block's instantaneous hazard *rate* to be
tripled during that phase and the result to be the calendar-time integral

    ``H = integral_0^T s(t) h(t) dt``

Those two models are different whenever the shape parameter is not 1, and the
difference is **not small**. This script measures it and declares the check

    "the cumulative-exposure model agrees with calendar-time hazard rate
    integration to within 1 percent for a non-exponential block"

which **FAILS**, by a measured margin reported below. The check is declared and
published rather than removed, because a user would otherwise discover the
difference the hard way, and because the two models answer different physical
questions: cumulative exposure says the stress advances the block along its own
ageing curve, calendar integration says the stress multiplies its instantaneous
hazard at its current calendar age. For a wear-out process the first is
usually the intended one, and this package implements it; the point of the
failing check is that the choice is a modelling decision and the package makes
it rather than the user.

The rest
--------

The exactness for an exponential block is checked and passes. The refusal to
age a one-shot block is checked. Per-phase breakdown arithmetic is checked
against the effective ages it is supposed to decompose.
"""

from __future__ import annotations

import math

from _bootstrap import Tee, Timer, bootstrap, header

bootstrap()

from scipy.integrate import quad  # noqa: E402

from reliablock.cases import case, mission  # noqa: E402
from reliablock.diagram import Block, Diagram, Series  # noqa: E402
from reliablock.distributions import (  # noqa: E402
    Exponential,
    FixedProbability,
    Lognormal,
    Weibull,
)
from reliablock.mission import MissionPhase, MissionProfile  # noqa: E402
from reliablock.quantify import quantify  # noqa: E402

#: The tolerance a reader would plausibly assume between the two ageing models.
#: This script reports that it is NOT met, with the measured margin.
ASSUMED_AGREEMENT = 0.01


def check_exponential_exactness(tee: Tee) -> None:
    tee.rule("the model is exact for an exponential block")
    tee("")
    tee("  For a constant hazard rate, sum(duration * duty * stress) IS the")
    tee("  integral of the hazard rate over the mission, so the effective-age")
    tee("  model is exact rather than approximate. The reference is the integral,")
    tee("  evaluated by quad over the piecewise-constant stress profile.")
    tee("")
    tee(
        f"  {'rate /h':>10s}{'phases':>8s}{'effective H':>20s}"
        f"{'integral H':>20s}{'rel diff':>12s}"
    )
    tee(f"  {'-' * 70}")
    worst = 0.0
    for rate in (1e-6, 1e-5, 1e-4, 1e-3):
        phases = (
            MissionPhase("boost", 100.0, stress_factors={"x": 3.0}),
            MissionPhase("cruise", 2000.0, duty_cycles={"x": 0.25}),
            MissionPhase("science", 900.0, stress_factors={"x": 1.5}),
        )
        profile = MissionProfile(phases=phases)
        diagram = Diagram("one", Series(items=(Block(name="x", distribution=Exponential(rate)),)))
        effective = quantify(diagram, profile).hazard

        def rate_at(t: float, _r: float = rate) -> float:
            if t < 100.0:
                return 3.0 * _r
            if t < 2100.0:
                return 0.25 * _r
            return 1.5 * _r

        integral, _ = quad(rate_at, 0.0, 3000.0, points=[100.0, 2100.0], limit=200)
        difference = abs(effective - integral) / integral
        worst = max(worst, difference)
        tee(
            f"  {rate:>10.1e}{len(phases):>8d}{effective:>20.15f}"
            f"{integral:>20.15f}{difference:>12.3e}"
        )
    tee("")
    tee(f"  worst relative difference          : {worst:.6e}")
    tee.check(
        "the effective-age model is exact for an exponential block",
        worst <= 1e-12,
        f"worst {worst:.6e} against tolerance 1.0e-12",
    )


def check_non_exponential_divergence(tee: Tee) -> None:
    tee.rule("THE CHECK THIS SCRIPT FAILS: the two ageing models for a Weibull block")
    tee("")
    tee("  cumulative exposure (implemented): the stress advances the block")
    tee("    along its own ageing curve, so H = H_block(sum(duration * stress)).")
    tee("  calendar-time integration (not implemented): the stress multiplies")
    tee("    the instantaneous hazard rate at the block's calendar age, so")
    tee("    H = integral s(t) h(t) dt.")
    tee("")
    tee(f"  declared tolerance a reader would assume : {ASSUMED_AGREEMENT:.1%}")
    tee("")
    tee(
        f"  {'shape':>7s}{'scale h':>12s}{'stress':>8s}{'cum exposure H':>18s}"
        f"{'calendar H':>18s}{'rel diff':>12s}{'within 1%':>11s}"
    )
    tee(f"  {'-' * 86}")
    worst = 0.0
    within = 0
    cases = 0
    for shape in (0.6, 1.0, 1.8, 2.5, 3.5):
        for scale in (5.0e3, 5.0e4):
            for stress in (2.0, 4.0):
                block = Block(
                    name="x", distribution=Weibull(shape=shape, scale_hours=scale)
                )
                diagram = Diagram("one", Series(items=(block,)))
                profile = MissionProfile(
                    phases=(
                        MissionPhase("hot", 500.0, stress_factors={"x": stress}),
                        MissionPhase("cold", 500.0, stress_factors={"x": 1.0}),
                    )
                )
                exposure = quantify(diagram, profile).hazard

                def hazard_rate(
                    t: float, _s: float = shape, _e: float = scale, _f: float = stress
                ) -> float:
                    factor = _f if t < 500.0 else 1.0
                    if t <= 0.0:
                        return 0.0
                    return factor * (_s / _e) * (t / _e) ** (_s - 1.0)

                calendar, _ = quad(hazard_rate, 0.0, 1000.0, points=[500.0], limit=200)
                difference = abs(exposure - calendar) / calendar
                worst = max(worst, difference)
                cases += 1
                agrees = difference <= ASSUMED_AGREEMENT
                if agrees:
                    within += 1
                tee(
                    f"  {shape:>7.2f}{scale:>12.1e}{stress:>8.1f}{exposure:>18.9e}"
                    f"{calendar:>18.9e}{difference:>12.3e}{str(agrees):>11s}"
                )
    tee("")
    tee(f"  cases                              : {cases}")
    tee(f"  cases agreeing to within 1 percent : {within}")
    tee(f"  worst relative difference          : {worst:.6e}")
    tee(f"  worst, as a percentage             : {100.0 * worst:.2f} percent")
    tee("")
    tee("  The shape = 1.00 rows agree exactly, which is the exponential case and")
    tee("  is the reason the model is exact there. Every other shape diverges,")
    tee("  and the divergence grows with the shape and with the stress factor.")
    tee("")
    tee.check(
        "the cumulative-exposure model agrees with calendar-time hazard rate "
        "integration to within 1 percent for a non-exponential block",
        worst <= ASSUMED_AGREEMENT,
        f"worst {100.0 * worst:.2f} percent against an assumed {ASSUMED_AGREEMENT:.1%}; "
        f"{within} of {cases} cases agree",
    )
    tee("")
    tee("  THIS CHECK IS REPORTED AS FAILED AND IS NOT TUNED AWAY. The two")
    tee("  models answer different physical questions and this package")
    tee("  implements the cumulative-exposure one, which is named in")
    tee("  reliablock.mission's docstring and in the README under Limitations.")
    tee("  A user who wants calendar-time integration of a scaled hazard rate")
    tee("  needs a different package, and the figures above are how much the")
    tee("  choice is worth on a two-phase profile.")


def check_lognormal_divergence(tee: Tee) -> None:
    tee.rule("the same divergence for a lognormal block, for completeness")
    tee("")
    tee(
        f"  {'mu':>7s}{'sigma':>8s}{'stress':>8s}{'cum exposure H':>18s}"
        f"{'calendar H':>18s}{'rel diff':>12s}"
    )
    tee(f"  {'-' * 73}")
    worst = 0.0
    for mu in (8.0, 10.0):
        for sigma in (0.4, 0.8):
            for stress in (2.0, 4.0):
                distribution = Lognormal(log_median=mu, log_sigma=sigma)
                block = Block(name="x", distribution=distribution)
                diagram = Diagram("one", Series(items=(block,)))
                profile = MissionProfile(
                    phases=(
                        MissionPhase("hot", 500.0, stress_factors={"x": stress}),
                        MissionPhase("cold", 500.0, stress_factors={"x": 1.0}),
                    )
                )
                exposure = quantify(diagram, profile).hazard

                def hazard_rate(
                    t: float,
                    _d: Lognormal = distribution,
                    _f: float = stress,
                ) -> float:
                    factor = _f if t < 500.0 else 1.0
                    if t <= 0.0:
                        return 0.0
                    survival = _d.reliability(t)
                    if survival <= 0.0:
                        return 0.0
                    return factor * _d.density(t) / survival

                calendar, _ = quad(hazard_rate, 1e-6, 1000.0, points=[500.0], limit=200)
                difference = (
                    abs(exposure - calendar) / calendar if calendar > 0.0 else float("inf")
                )
                worst = max(worst, difference)
                tee(
                    f"  {mu:>7.1f}{sigma:>8.2f}{stress:>8.1f}{exposure:>18.9e}"
                    f"{calendar:>18.9e}{difference:>12.3e}"
                )
    tee("")
    tee(f"  worst relative difference          : {worst:.6e}")
    tee("")
    tee("  Reported as a measurement. No tolerance is declared for it, because")
    tee("  the two models are not meant to agree and declaring a tolerance would")
    tee("  imply that they are.")


def check_one_shot_refusal(tee: Tee) -> None:
    tee.rule("a one-shot block is not aged by a profile")
    tee("")
    tee("  A FixedProbability block's figure of merit is a probability per")
    tee("  actuation, not a rate. Multiplying it by a duty cycle is not a model")
    tee("  of anything, so a profile that tries is refused.")
    tee("")
    diagram = Diagram(
        "deploy", Series(items=(Block(name="d", distribution=FixedProbability(1.5e-3)),))
    )
    for label, phase in (
        ("duty cycle 0.5", MissionPhase("p", 100.0, duty_cycles={"d": 0.5})),
        ("stress factor 2.0", MissionPhase("p", 100.0, stress_factors={"d": 2.0})),
    ):
        profile = MissionProfile(phases=(phase,))
        try:
            profile.effective_ages(diagram)
            tee.check(f"{label} is refused", False, "it was accepted")
        except ValueError as error:
            tee(f"  {label}: {' '.join(str(error).split())[:200]}")
            tee.check(f"{label} on a one-shot block is refused", True)
    tee("")
    tee("  and the block's unreliability is unchanged by any mission duration:")
    unchanged = True
    for hours in (0.0, 1.0, 1e3, 1e6, 1e9):
        value = quantify(diagram, hours).unreliability
        tee(f"    {hours:>12.4g} h : Q = {value:.15e}")
        if abs(value - 1.5e-3) > 1e-15:
            unchanged = False
    tee.check("a one-shot block's unreliability is independent of mission duration", unchanged)


def check_breakdown(tee: Tee) -> None:
    tee.rule("the per-phase breakdown decomposes the effective ages it reports")
    tee("")
    diagram = case("smallsat_bus")
    profile = mission("leo_phased")
    ages = profile.effective_ages(diagram)
    rows = profile.breakdown(diagram)
    worst = 0.0
    for block in diagram.blocks():
        total = sum(r["effective_hours"] for r in rows if r["block"] == block.name)
        difference = abs(total - ages[block.name]) / max(ages[block.name], 1e-30)
        worst = max(worst, difference)
    tee(f"  blocks                             : {len(diagram.blocks())}")
    tee(f"  breakdown rows                     : {len(rows)}")
    tee(f"  worst relative difference          : {worst:.6e}")
    tee.check(
        "the per-phase breakdown sums to the reported effective age for every block",
        worst <= 1e-14,
        f"worst {worst:.6e}",
    )
    tee("")
    tee("  the phased profile against a flat one of the same elapsed time:")
    flat = mission("three_year")
    tee("")
    tee(f"  {'block':<26s}{'flat eff h':>14s}{'phased eff h':>16s}{'ratio':>10s}")
    tee(f"  {'-' * 66}")
    flat_ages = flat.effective_ages(diagram)
    for name in diagram.block_names():
        ratio = ages[name] / flat_ages[name] if flat_ages[name] else math.nan
        tee(f"  {name:<26s}{flat_ages[name]:>14.6g}{ages[name]:>16.6g}{ratio:>10.4f}")
    tee("")
    tee(f"  flat elapsed   : {flat.total_hours:.6g} h")
    tee(f"  phased elapsed : {profile.total_hours:.6g} h")
    tee.check(
        "both profiles declare the same elapsed time, so the comparison is fair",
        abs(flat.total_hours - profile.total_hours) < 1e-9,
        f"{flat.total_hours:.6g} against {profile.total_hours:.6g}",
    )


def main() -> int:
    tee = Tee("validate_mission_output.txt")
    header(tee, "the mission-profile ageing model, including the check it fails")
    tee("")
    tee("  This script contains the one declared check in this repository that")
    tee("  FAILS. It is published rather than removed, and the reason is in the")
    tee("  module docstring and in the README under Limitations.")

    timer = Timer()
    with timer:
        check_exponential_exactness(tee)
        check_non_exponential_divergence(tee)
        check_lognormal_divergence(tee)
        check_one_shot_refusal(tee)
        check_breakdown(tee)

    tee.rule("timing, declared unstable")
    tee("")
    tee("  | quantity | value | stability |")
    tee("  |---|---|---|")
    tee(f"  | whole script wall clock | {timer.elapsed_s:.2f} s | UNSTABLE, shared hardware |")

    tee.summarise_failures()
    tee("")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
