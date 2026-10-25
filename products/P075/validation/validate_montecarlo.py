"""Monte Carlo against the closed forms, with the budget derived from a target.

Why the budget is derived and not chosen
----------------------------------------

A round number of trials is not a statement about a check's power. What matters
is the width of the resulting confidence interval relative to the quantity
being checked: a simulation whose interval is wider than the effect it is
supposed to confirm confirms nothing. Each run below is therefore sized from a
declared target -- an interval half-width equal to a stated fraction of the
system **unreliability** -- through

    ``n = z^2 Q (1 - Q) / h^2``

evaluated at a pilot estimate of ``Q``, and the interval actually achieved is
reported rather than assumed. Both the budget and the achieved interval are in
the tables below.

Why the closed form is the reference
------------------------------------

For a declared diagram with declared distributions the reliability has a
correct answer. The simulation is scored against it, in units of the binomial
standard error, and any disagreement is reported as measured. The simulation is
not a better estimate of anything here; what it is, is an implementation that
follows the physical story -- the primary fails at ``u``, the switch is tried,
the standby has been ageing at a fraction of its rate -- with no algebra in
common with the closed form it is checked against. That is what makes the
agreement informative.

Compute budget
--------------

Two cores, shared with other work. Every run is seeded and batched, so the peak
memory is bounded by the batch size and not by the trial count. The whole
script's wall clock is in a table declared unstable.
"""

from __future__ import annotations

from _bootstrap import Tee, Timer, bootstrap, header

bootstrap()

from reliablock.cases import CASES, MISSIONS  # noqa: E402
from reliablock.commoncause import apply_common_cause, strip_common_cause  # noqa: E402
from reliablock.diagram import Diagram  # noqa: E402
from reliablock.mission import MissionProfile  # noqa: E402
from reliablock.montecarlo import simulate, trials_for_half_width  # noqa: E402
from reliablock.quantify import quantify  # noqa: E402

SEED = 20261025

#: Interval half-width target, as a fraction of the system unreliability.
#: 0.01 means the 95 percent interval is to be 1 percent of Q wide on each
#: side, so a closed form wrong by 5 percent of Q would sit five interval
#: half-widths away and could not be missed. The value is chosen from that
#: requirement and not for being round; the trial counts it implies are in the
#: tables below and are not round either.
TARGET_FRACTION = 0.01

#: Trials are capped so that the script stays inside the gate's 600 s per-script
#: limit on two shared cores. Where the cap binds, the achieved interval is
#: wider than the target and the table says so.
MAX_TRIALS = 20_000_000

PILOT_TRIALS = 50_000


def _sized_run(
    tee: Tee, label: str, diagram: Diagram, mission: MissionProfile
) -> tuple[bool, float, int, float]:
    """Pilot, size, run, and report. Returns (bracketed, sigma, trials, half_width)."""
    closed = quantify(diagram, mission)
    pilot = simulate(diagram, mission, PILOT_TRIALS, seed=SEED + 1)
    reference_q = max(pilot.unreliability, closed.unreliability, 1.0 / PILOT_TRIALS)
    wanted = trials_for_half_width(min(reference_q, 0.5), TARGET_FRACTION * reference_q)
    trials = min(wanted, MAX_TRIALS)
    result = simulate(diagram, mission, trials, seed=SEED)
    sigma = result.standard_errors_from(closed.reliability)
    bracketed = result.brackets(closed.reliability)
    tee("")
    tee(f"  {label}")
    tee(f"    closed form R                  : {closed.reliability:.12f}")
    tee(f"    closed form Q                  : {closed.unreliability:.9e}")
    tee(f"    pilot Q ({PILOT_TRIALS} trials)       : {pilot.unreliability:.9e}")
    tee(f"    target interval half-width     : {TARGET_FRACTION * reference_q:.6e}")
    tee(f"    trials needed for that target  : {wanted}")
    tee(f"    trials run (cap {MAX_TRIALS})  : {trials}")
    tee(f"    simulated R                    : {result.reliability:.12f}")
    tee(f"    Wilson 95 percent interval     : [{result.lower:.12f}, {result.upper:.12f}]")
    tee(f"    achieved half-width            : {result.half_width:.6e}")
    tee(f"    half-width as a fraction of Q  : {result.half_width / reference_q:.6f}")
    tee(f"    closed form inside interval    : {bracketed}")
    tee(f"    distance in standard errors    : {sigma:.4f}")
    tee(f"    signed difference R_mc - R_cf  : {result.reliability - closed.reliability:+.9e}")
    return bracketed, sigma, trials, result.half_width


def main() -> int:
    tee = Tee("validate_montecarlo_output.txt")
    header(tee, "Monte Carlo against the closed forms, budget derived from a target")
    tee("")
    tee(f"  seed                               : {SEED}")
    tee(f"  pilot trials                       : {PILOT_TRIALS}")
    tee(f"  target half-width                  : {TARGET_FRACTION:.3f} of the system Q")
    tee(f"  trial cap                          : {MAX_TRIALS}")
    tee("")
    tee("  The closed form is the reference throughout. The simulation shares no")
    tee("  algebra with it: standby is simulated by drawing the primary's")
    tee("  lifetime, trying the switch and ageing the spare at its declared")
    tee("  fraction; shared load by drawing the first failure and redrawing the")
    tee("  survivor's residual life at the full rate; a network by testing its")
    tee("  minimal path sets.")

    timer = Timer()
    outcomes: list[tuple[str, bool, float, int, float]] = []

    tee.rule("every built-in case over its default mission")
    with timer:
        for name in sorted(CASES):
            diagram = CASES[name]
            mission = MISSIONS["three_year"]
            bracketed, sigma, trials, half = _sized_run(
                tee, f"{name} over {mission.name}", diagram, mission
            )
            outcomes.append((name, bracketed, sigma, trials, half))

    tee.rule("the headline diagram, with and without the declared common cause")
    tee("")
    tee("  Both halves of the headline comparison are simulated independently, so")
    tee("  the difference between them is confirmed by simulation and not only by")
    tee("  arithmetic.")
    bus = CASES["smallsat_bus"]
    mission = MISSIONS["three_year"]
    with timer:
        for label, diagram in (
            ("independence assumed", strip_common_cause(bus)),
            ("declared common cause", apply_common_cause(bus)),
        ):
            bracketed, sigma, trials, half = _sized_run(tee, label, diagram, mission)
            outcomes.append((f"smallsat_bus: {label}", bracketed, sigma, trials, half))

    tee.rule("the phased mission profile")
    with timer:
        bracketed, sigma, trials, half = _sized_run(
            tee, "smallsat_bus over the LEO phased profile", bus, MISSIONS["leo_phased"]
        )
        outcomes.append(("smallsat_bus phased", bracketed, sigma, trials, half))

    tee.rule("summary")
    tee("")
    tee(
        f"  {'case':<44s}{'trials':>10s}{'half-width':>14s}{'sigma':>10s}{'inside':>8s}"
    )
    tee(f"  {'-' * 86}")
    for name, bracketed, sigma, trials, half in outcomes:
        tee(f"  {name:<44s}{trials:>10d}{half:>14.6e}{sigma:>10.4f}{str(bracketed):>8s}")

    outside = [name for name, bracketed, _, _, _ in outcomes if not bracketed]
    worst_sigma = max(sigma for _, _, sigma, _, _ in outcomes)
    tee("")
    tee(f"  runs                               : {len(outcomes)}")
    tee(f"  closed form outside the interval   : {len(outside)}")
    for name in outside:
        tee(f"    {name}")
    tee(f"  worst distance in standard errors  : {worst_sigma:.4f}")
    tee("")
    tee("  A 95 percent interval is expected to miss about 1 run in 20 by")
    tee("  construction, so a small number of misses is not evidence of a defect;")
    tee("  the distance in standard errors is. Any run beyond 4 sigma is reported")
    tee("  as a failed check below.")
    tee.check(
        "every simulation is within 4 standard errors of its closed form",
        worst_sigma < 4.0,
        f"worst {worst_sigma:.4f} standard errors over {len(outcomes)} runs",
    )
    tee.check(
        "the closed form lies inside at least 80 percent of the intervals",
        len(outside) <= max(1, len(outcomes) // 5),
        f"{len(outside)} of {len(outcomes)} outside",
    )

    tee.rule("timing, declared unstable")
    tee("")
    tee("  | quantity | value | stability |")
    tee("  |---|---|---|")
    tee(f"  | simulation wall clock | {timer.elapsed_s:.2f} s | UNSTABLE, shared hardware |")
    total_trials = sum(trials for _, _, _, trials, _ in outcomes)
    tee(f"  | total trials simulated | {total_trials} | stable, seeded |")

    tee.summarise_failures()
    tee("")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
