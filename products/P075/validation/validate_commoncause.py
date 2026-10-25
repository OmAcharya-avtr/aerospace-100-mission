"""The headline: reliability with and without a declared common-cause beta.

This is the product's reason to exist, so it is measured three ways: on a
realistic multi-group diagram, group by group across the parameter space, and
on the cut-set structure, which is the half of the answer that is not a
percentage.

It also maps a limitation the package would otherwise hide. The beta-factor
model has **exactly one** common-cause mode: the event that fails every member
of the group. For a parallel group (``k = 1``) that mode is the only way the
group can fail, so moving failure probability into it always increases the
group's unreliability and the model is conservative. For a k-of-n group with
``k >= 2`` the group fails once ``n - k + 1`` members have failed, which is
fewer than all of them, so the single all-members mode is "wasteful" and moving
probability into it can **reduce** the computed group unreliability. That is
arithmetically correct and it is a known inadequacy of the beta-factor model
for groups larger than two -- it is the reason NUREG/CR-4780 introduces the
multiple-Greek-letter and alpha-factor models, which carry a separate parameter
for each size of partial common-cause failure, and which are not implemented
here.

This script locates the crossing for every ``(k, n)`` it can, so that the
limitation is a measured boundary rather than a caveat.

Every beta used here is a declared input chosen to span a range. None of them
is data, and none is taken from any source.
"""

from __future__ import annotations

from _bootstrap import Tee, Timer, bootstrap, header

bootstrap()

from reliablock.cases import CASES, MISSIONS  # noqa: E402
from reliablock.commoncause import (  # noqa: E402
    apply_common_cause,
    group_beta_effects,
    strip_common_cause,
)
from reliablock.diagram import Block, Diagram, KOfN, Parallel  # noqa: E402
from reliablock.distributions import Exponential  # noqa: E402
from reliablock.montecarlo import simulate, trials_for_half_width  # noqa: E402
from reliablock.quantify import quantify  # noqa: E402
from reliablock.report import common_cause_comparison  # noqa: E402
from reliablock.structure import minimal_cut_sets  # noqa: E402

SEED = 20261025
HOURS = 1000.0


def _group(k: int, n: int, hazard: float, beta: float) -> Diagram:
    """A k-of-n group whose members each have cumulative hazard ``hazard`` at HOURS."""
    rate = hazard / HOURS
    items = tuple(
        Block(name=f"u{i}", distribution=Exponential(rate)) for i in range(n)
    )
    if k == 1:
        return Diagram("group", Parallel(items=items, beta_factor=beta, label="g"))
    return Diagram("group", KOfN(k=k, items=items, beta_factor=beta, label="g"))


def check_headline(tee: Tee) -> None:
    tee.rule("the headline, on the smallsat bus over three years")
    tee("")
    diagram = CASES["smallsat_bus"]
    mission = MISSIONS["three_year"]
    for line in common_cause_comparison(diagram, mission):
        tee(f"  {line}")

    independent = quantify(strip_common_cause(diagram), mission)
    with_cc = quantify(apply_common_cause(diagram), mission)
    ratio = with_cc.unreliability / independent.unreliability
    tee("")
    tee("  the headline numbers, isolated:")
    tee(f"    R assuming independence          : {independent.reliability:.12f}")
    tee(f"    R with the declared common cause : {with_cc.reliability:.12f}")
    tee(f"    Q assuming independence          : {independent.unreliability:.9e}")
    tee(f"    Q with the declared common cause : {with_cc.unreliability:.9e}")
    tee(f"    Q ratio                          : {ratio:.6f}")
    tee(
        f"    Q attributable to the assumption : "
        f"{with_cc.unreliability - independent.unreliability:.9e}"
    )
    singles_before = sum(
        1 for c in minimal_cut_sets(strip_common_cause(diagram)) if len(c) == 1
    )
    singles_after = sum(
        1 for c in minimal_cut_sets(apply_common_cause(diagram)) if len(c) == 1
    )
    tee(f"    single points of failure, before : {singles_before}")
    tee(f"    single points of failure, after  : {singles_after}")
    tee.check(
        "the declared common cause increases the system unreliability on this diagram",
        with_cc.unreliability > independent.unreliability,
        f"ratio {ratio:.6f}",
    )
    tee.check(
        "the declared common cause adds one single point of failure per beta group",
        singles_after - singles_before == len(diagram.declared_betas),
        f"{singles_after - singles_before} added for {len(diagram.declared_betas)} betas",
    )


def check_beta_sweep(tee: Tee) -> None:
    tee.rule("the effect of beta on a parallel group, swept")
    tee("")
    tee("  Two identical members, member cumulative hazard 0.03 over the mission")
    tee("  (so q = 0.029554 each). Every beta is a declared input.")
    tee("")
    tee(
        f"  {'beta':>8s}{'Q independent':>18s}{'Q with cc':>18s}{'ratio':>12s}"
        f"{'cc event q':>16s}{'order-1 cuts':>14s}"
    )
    tee(f"  {'-' * 86}")
    monotone = True
    previous = -1.0
    for beta in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        diagram = _group(1, 2, 0.03, beta)
        independent = quantify(strip_common_cause(diagram), HOURS).unreliability
        rewritten = apply_common_cause(diagram)
        with_cc = quantify(rewritten, HOURS).unreliability
        cc_blocks = [b for b in rewritten.blocks() if b.name.endswith("::common_cause")]
        cc_q = cc_blocks[0].distribution.unreliability(HOURS) if cc_blocks else 0.0
        singles = sum(1 for c in minimal_cut_sets(rewritten) if len(c) == 1)
        ratio = with_cc / independent if independent else float("inf")
        tee(
            f"  {beta:>8.2f}{independent:>18.9e}{with_cc:>18.9e}{ratio:>12.4f}"
            f"{cc_q:>16.9e}{singles:>14d}"
        )
        if with_cc < previous:
            monotone = False
        previous = with_cc
    tee("")
    tee.check(
        "a parallel group's unreliability is non-decreasing in beta",
        monotone,
        "swept over 8 declared beta values",
    )

    tee("")
    tee("  at beta = 1 the group must reduce to a single member, exactly:")
    rate = 0.03 / HOURS
    single = Exponential(rate).reliability(HOURS)
    worst = 0.0
    for n in (2, 3, 4, 5):
        diagram = _group(1, n, 0.03, 1.0)
        value = quantify(apply_common_cause(diagram), HOURS).reliability
        worst = max(worst, abs(value - single) / single)
        tee(f"    n = {n}: R = {value:.15f}, single member R = {single:.15f}")
    tee(f"    worst relative difference        : {worst:.6e}")
    tee.check(
        "beta = 1 reduces a group to one member",
        worst <= 1e-14,
        f"worst {worst:.6e} against tolerance 1.0e-14",
    )

    tee("")
    tee("  at beta = 0 the rewrite must reproduce the independent figure bit for bit,")
    tee("  because it creates no common-cause block at all:")
    identical = True
    for n in (2, 3, 4, 5):
        diagram = _group(1, n, 0.03, 0.0)
        a = quantify(apply_common_cause(diagram), HOURS).reliability
        b = quantify(strip_common_cause(diagram), HOURS).reliability
        if a != b:
            identical = False
            tee(f"    n = {n}: {a!r} against {b!r}")
    tee.check("beta = 0 reproduces the independent figure exactly", identical)


def check_crossing_map(tee: Tee) -> None:
    tee.rule("where the beta-factor model stops being conservative")
    tee("")
    tee("  For k = 1 the beta factor always increases Q. For k >= 2 it can")
    tee("  decrease Q, because the model's single common-cause mode fails all n")
    tee("  members while the group only needs n - k + 1 failures. The table")
    tee("  below locates the member cumulative hazard at which the ratio crosses")
    tee("  1, by bisection, at beta = 0.1.")
    tee("")
    tee(
        f"  {'k':>3s}{'n':>3s}{'needed':>8s}{'ratio at H=0.01':>18s}"
        f"{'ratio at H=3':>14s}{'crossing H':>14s}{'verdict':>26s}"
    )
    tee(f"  {'-' * 88}")
    beta = 0.1
    crossings: list[tuple[int, int, float]] = []
    # k = n is refused by Diagram.validate: it is a series group, not a
    # redundant one, and a beta factor on it is meaningless. That refusal was
    # added BECAUSE this sweep measured it -- at k = n the ratio is below 1 at
    # every hazard, so the beta factor made the group look better everywhere.
    # The refusal message quotes the measured ratio.
    for n in (2, 3, 4, 5, 6):
        for k in range(1, n):
            def ratio(hazard: float, _k: int = k, _n: int = n) -> float:
                diagram = _group(_k, _n, hazard, beta)
                independent = quantify(strip_common_cause(diagram), HOURS).unreliability
                with_cc = quantify(apply_common_cause(diagram), HOURS).unreliability
                return with_cc / independent if independent > 0.0 else float("inf")

            low_ratio = ratio(0.01)
            high_ratio = ratio(3.0)
            crossing = float("nan")
            if (low_ratio - 1.0) * (high_ratio - 1.0) < 0.0:
                lo, hi = 0.01, 3.0
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    if (ratio(lo) - 1.0) * (ratio(mid) - 1.0) <= 0.0:
                        hi = mid
                    else:
                        lo = mid
                crossing = 0.5 * (lo + hi)
                crossings.append((k, n, crossing))
            if crossing == crossing:
                verdict = "crosses: conservative below"
            elif low_ratio >= 1.0 and high_ratio >= 1.0:
                verdict = "conservative everywhere"
            else:
                verdict = "NON-CONSERVATIVE everywhere"
            tee(
                f"  {k:>3d}{n:>3d}{n - k + 1:>8d}{low_ratio:>18.6f}"
                f"{high_ratio:>14.6f}"
                + (f"{crossing:>14.6f}" if crossing == crossing else f"{'none':>14s}")
                + f"  {verdict:>26s}"
            )
    tee("")
    tee(f"  (k, n) pairs with a crossing       : {len(crossings)}")
    tee(f"  beta used                          : {beta} (a declared input, not data)")
    tee("")
    tee("  Reading the table: 'needed' is n - k + 1, the number of failures the")
    tee("  group needs. Where it equals n -- that is, k = 1 -- the beta factor is")
    tee("  conservative at every hazard and there is no crossing. Where it is")
    tee("  less than n, there is a hazard above which the beta factor REDUCES the")
    tee("  computed unreliability.")
    tee("")
    tee("  k = n is absent from the table because Diagram.validate now REFUSES a")
    tee("  beta factor there. That refusal was added because an earlier run of")
    tee("  this sweep measured k = n as non-conservative at every hazard -- a")
    tee("  ratio of 0.918939 at k = n = 6 -- which is correct arithmetic for a")
    tee("  series group and meaningless as a redundancy model. The refusal")
    tee("  message quotes that measured ratio.")
    no_crossing_at_k1 = all(k != 1 for k, _n, _h in crossings)
    tee.check(
        "a parallel group (k = 1) has no crossing: the beta factor is always conservative",
        no_crossing_at_k1,
        f"{sum(1 for k, _, _ in crossings if k == 1)} crossings found at k = 1",
    )
    tee.check(
        "a k-of-n group with k >= 2 does have a crossing, which is the limitation",
        len(crossings) > 0,
        f"{len(crossings)} crossings located",
    )
    tee("")
    tee("  reliablock.commoncause.group_beta_effects reports this per group, and")
    tee("  reliablock.report.common_cause_comparison prints a WARNING naming the")
    tee("  alpha-factor and multiple-Greek-letter models when any group crosses.")
    tee("  Neither is implemented here. The limitation is surfaced on the")
    tee("  declaration rather than left in a document.")


def check_group_effects(tee: Tee) -> None:
    tee.rule("per-group effects on the built-in cases")
    tee("")
    tee(
        f"  {'case':<20s}{'group':<18s}{'kind':<10s}{'beta':>7s}{'k-of-n':>8s}"
        f"{'Q indep':>14s}{'Q with cc':>14s}{'ratio':>10s}{'conserv':>9s}"
    )
    tee(f"  {'-' * 112}")
    non_conservative = 0
    total = 0
    for name in sorted(CASES):
        diagram = CASES[name]
        for effect in group_beta_effects(diagram, MISSIONS["three_year"]):
            total += 1
            if not effect.conservative:
                non_conservative += 1
            tee(
                f"  {name:<20s}{effect.label:<18s}{effect.kind:<10s}{effect.beta:>7.3g}"
                f"{f'{effect.k}-of-{effect.members}':>8s}"
                f"{effect.unreliability_independent:>14.6e}"
                f"{effect.unreliability_with_common_cause:>14.6e}"
                f"{effect.ratio:>10.4f}{str(effect.conservative):>9s}"
            )
    tee("")
    tee(f"  groups with a declared beta        : {total}")
    tee(f"  groups where beta DECREASES Q      : {non_conservative}")
    tee.check(
        "every declared beta group in the built-in cases is conservative at three years",
        non_conservative == 0,
        f"{non_conservative} of {total} are not",
    )


def check_monte_carlo(tee: Tee) -> None:
    tee.rule("both halves of the headline, simulated independently")
    tee("")
    tee("  The with-and-without comparison is confirmed by simulation as well as")
    tee("  by arithmetic, so the difference between the two columns does not rest")
    tee("  on a single code path.")
    tee("")
    diagram = CASES["smallsat_bus"]
    mission = MISSIONS["three_year"]
    rows: list[tuple[str, float, float, int, float, bool, float]] = []
    for label, structure in (
        ("independence assumed", strip_common_cause(diagram)),
        ("declared common cause", apply_common_cause(diagram)),
    ):
        closed = quantify(structure, mission)
        q = closed.unreliability
        trials = trials_for_half_width(q, 0.01 * q)
        result = simulate(structure, mission, trials, seed=SEED)
        rows.append(
            (
                label,
                closed.reliability,
                result.reliability,
                trials,
                result.half_width,
                result.brackets(closed.reliability),
                result.standard_errors_from(closed.reliability),
            )
        )
    tee(
        f"  {'half':<24s}{'closed form R':>18s}{'simulated R':>18s}"
        f"{'trials':>10s}{'half-width':>13s}{'inside':>8s}{'sigma':>8s}"
    )
    tee(f"  {'-' * 99}")
    for label, closed_r, mc_r, trials, half, inside, sigma in rows:
        tee(
            f"  {label:<24s}{closed_r:>18.12f}{mc_r:>18.12f}{trials:>10d}"
            f"{half:>13.6e}{str(inside):>8s}{sigma:>8.4f}"
        )
    tee("")
    simulated_gap = rows[0][2] - rows[1][2]
    closed_gap = rows[0][1] - rows[1][1]
    tee(f"  closed-form gap in R               : {closed_gap:.9e}")
    tee(f"  simulated gap in R                 : {simulated_gap:.9e}")
    tee(
        f"  gap difference                     : "
        f"{abs(simulated_gap - closed_gap):.9e}"
    )
    tee(
        f"  combined interval half-width       : "
        f"{rows[0][4] + rows[1][4]:.9e}"
    )
    tee.check(
        "the simulated gap between the two halves agrees with the closed-form gap",
        abs(simulated_gap - closed_gap) <= rows[0][4] + rows[1][4],
        f"gap difference {abs(simulated_gap - closed_gap):.6e} against combined "
        f"half-width {rows[0][4] + rows[1][4]:.6e}",
    )
    tee.check(
        "both halves are individually within 4 standard errors of their closed form",
        all(abs(sigma) < 4.0 for *_rest, sigma in rows),
        f"worst {max(abs(sigma) for *_rest, sigma in rows):.4f}",
    )


def check_refusals(tee: Tee) -> None:
    tee.rule("refusals")
    tee("")
    tee("  The beta-factor model is symmetric: one shared event fails the whole")
    tee("  group, and lambda_common = beta lambda needs one lambda. A group whose")
    tee("  members do not share one distribution has no single lambda, and taking")
    tee("  the mean, minimum or maximum gives three different answers with no")
    tee("  principled basis. It is refused.")
    tee("")
    mixed = Diagram(
        "mixed group",
        Parallel(
            items=(
                Block(name="a", distribution=Exponential(1.0e-5)),
                Block(name="b", distribution=Exponential(2.0e-5)),
            ),
            beta_factor=0.1,
            label="pair",
        ),
    )
    try:
        apply_common_cause(mixed)
        tee.check("a non-identical group is refused", False, "it was not refused")
    except ValueError as error:
        tee(f"  message: {error}")
        tee.check(
            "a non-identical group is refused and the message names the alternative model",
            "alpha-factor" in str(error),
        )
    tee("")
    tee("  A standby or shared-load node has no beta_factor field at all: it is")
    tee("  already a dependent-failure model, and compounding it with a beta")
    tee("  factor needs a combined Markov model this package does not have.")
    import dataclasses

    from reliablock.diagram import SharedLoad, Standby

    standby_fields = {f.name for f in dataclasses.fields(Standby)}
    shared_fields = {f.name for f in dataclasses.fields(SharedLoad)}
    tee(f"  standby fields     : {sorted(standby_fields)}")
    tee(f"  shared_load fields : {sorted(shared_fields)}")
    tee.check(
        "neither standby nor shared_load accepts a beta factor",
        "beta_factor" not in standby_fields and "beta_factor" not in shared_fields,
    )


def main() -> int:
    tee = Tee("validate_commoncause_output.txt")
    header(tee, "the headline: reliability with and without a declared common cause")
    tee("")
    tee("  Every beta factor used in this script is a declared input chosen to")
    tee("  span a range. None is data, and none is taken from any source. A beta")
    tee("  factor for real hardware comes from the programme's own common-cause")
    tee("  data, which this package does not have and does not pretend to.")

    timer = Timer()
    with timer:
        check_headline(tee)
        check_beta_sweep(tee)
        check_crossing_map(tee)
        check_group_effects(tee)
        check_monte_carlo(tee)
        check_refusals(tee)

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
