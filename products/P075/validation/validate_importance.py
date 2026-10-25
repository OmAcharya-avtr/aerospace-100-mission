"""Importance measures against finite-difference perturbation and their identities.

The reference for Birnbaum
--------------------------

``I_B(i) = dQ_sys / dq_i`` is a derivative, so the reference it is scored
against is a **central finite difference** of the system unreliability with
respect to one basic event's unreliability. That reference shares no expression
with the implementation: the implementation evaluates ``Q_sys`` twice, at
``q_i = 1`` and ``q_i = 0``, and subtracts; the reference evaluates it twice at
``q_i +- h`` and divides.

There is a second statement worth making here. ``Q_sys`` is **multilinear** in
the ``q_i`` for any coherent structure of independent components -- each ``q_i``
appears to at most the first power in every term -- so the derivative is
constant in ``q_i`` and the finite difference is *exact* for any step size, not
merely accurate for a small one. This script measures the agreement over five
decades of step size and reports that it does not degrade, which is a stronger
statement than agreement at one step.

The identities
--------------

``I_C(i) = (Q_sys - Q_sys(q_i = 0)) / Q_sys`` follows from the pivotal
decomposition, and ``I_FV(i) >= I_C(i)`` always. Both are checked. The
Fussell-Vesely figure is computed exactly, by enumerating the states of the
events appearing in the relevant cut sets, and is also compared with the
**rare-event approximation** (the sum of cut-set probabilities) so that the
difference between the two is on record: the approximation is what most tools
report and it is the usual source of a Fussell-Vesely figure above 1.
"""

from __future__ import annotations

from _bootstrap import Tee, Timer, bootstrap, header

bootstrap()

from reliablock.cases import CASES, MISSIONS  # noqa: E402
from reliablock.commoncause import apply_common_cause  # noqa: E402
from reliablock.diagram import Block, Diagram, KOfN, Parallel, Series  # noqa: E402
from reliablock.distributions import Exponential  # noqa: E402
from reliablock.importance import (  # noqa: E402
    boolean_system_unreliability,
    importance_table,
    pivot_unreliability,
    system_unreliability_at,
)
from reliablock.structure import minimal_cut_sets  # noqa: E402

STEPS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)

#: Declared tolerance on the finite-difference agreement, relative. The
#: difference is exact in exact arithmetic because Q_sys is multilinear, so the
#: only error is the cancellation in the numerator at small steps, and the
#: tolerance has to allow for it: at a step of 1e-7 the numerator loses about
#: seven digits.
FINITE_DIFFERENCE_TOLERANCE = 1e-6

#: Declared tolerance on the identities, relative.
IDENTITY_TOLERANCE = 1e-10


def _block(name: str, rate: float) -> Block:
    return Block(name=name, distribution=Exponential(rate))


def diagrams() -> dict[str, tuple[Diagram, float]]:
    """Boolean diagrams for which importance measures are defined, with a mission.

    A standby or shared-load node is excluded by construction: its
    quantification is not the independent-event arithmetic over its structure
    function, so ``dQ_sys/dq_i`` is not the derivative of the number the
    quantifier returns, and :func:`importance_table` refuses it. That refusal is
    checked at the end of this script.
    """
    out: dict[str, tuple[Diagram, float]] = {}
    out["series_parallel"] = (
        Diagram(
            "series parallel",
            Series(
                items=(
                    _block("single", 1.0e-4),
                    Parallel(items=(_block("a", 3.0e-4), _block("b", 3.0e-4)), label="pair"),
                ),
                label="chain",
            ),
        ),
        1000.0,
    )
    out["voting_triple"] = (CASES["voting_triple"], MISSIONS["three_year"].total_hours)
    out["three_of_four"] = (
        Diagram(
            "three of four",
            Series(
                items=(
                    KOfN(
                        k=3,
                        items=tuple(_block(f"u{i}", 1.0e-5) for i in range(4)),
                        label="wheels",
                    ),
                    _block("tail", 2.0e-6),
                ),
                label="chain",
            ),
        ),
        26298.0,
    )
    out["mixed_rates"] = (
        Diagram(
            "mixed rates",
            Series(
                items=(
                    Parallel(
                        items=(_block("p1", 1.0e-4), _block("p2", 5.0e-4)), label="pair"
                    ),
                    KOfN(
                        k=2,
                        items=(
                            _block("q1", 2.0e-4),
                            _block("q2", 3.0e-4),
                            _block("q3", 7.0e-4),
                        ),
                        label="triple",
                    ),
                ),
                label="chain",
            ),
        ),
        1000.0,
    )
    out["voting_triple_with_common_cause"] = (
        apply_common_cause(CASES["voting_triple"]),
        MISSIONS["three_year"].total_hours,
    )
    out["bridge"] = (
        __import__("reliablock.bridge", fromlist=["bridge_diagram"]).bridge_diagram(0.9),
        1.0,
    )
    return out


def check_finite_difference(tee: Tee) -> tuple[float, int]:
    """Score Birnbaum against a central finite difference. Returns (worst, cases)."""
    tee.rule("Birnbaum against central finite-difference perturbation")
    tee("")
    tee("  I_B(i) = dQ_sys / dq_i, scored against")
    tee("  [Q_sys(q_i + h) - Q_sys(q_i - h)] / (2h).")
    tee("")
    tee("  Q_sys is multilinear in the q_i for a coherent structure of")
    tee("  independent components, so the derivative is constant in q_i and the")
    tee("  finite difference is exact for ANY step in exact arithmetic. The only")
    tee("  error is the cancellation in the numerator, which grows as the step")
    tee("  shrinks -- which is why the agreement is measured across five decades")
    tee("  of step size rather than at one.")
    tee("")
    tee(f"  declared tolerance : {FINITE_DIFFERENCE_TOLERANCE:.1e} relative")
    tee("")
    tee(
        f"  {'diagram':<34s}{'step':>10s}{'events':>8s}"
        f"{'worst rel diff':>18s}"
    )
    tee(f"  {'-' * 70}")
    worst_overall = 0.0
    cases = 0
    for name, (diagram, hours) in sorted(diagrams().items()):
        base = {b.name: b.distribution.unreliability(hours) for b in diagram.blocks()}
        rows = importance_table(diagram, hours)
        for step in STEPS:
            worst = 0.0
            for row in rows:
                high = dict(base)
                low = dict(base)
                high[row.event] = min(base[row.event] + step, 1.0)
                low[row.event] = max(base[row.event] - step, 0.0)
                span = high[row.event] - low[row.event]
                if span <= 0.0:
                    continue
                difference = (
                    system_unreliability_at(diagram, high)
                    - system_unreliability_at(diagram, low)
                ) / span
                if row.birnbaum != 0.0:
                    worst = max(worst, abs(difference - row.birnbaum) / abs(row.birnbaum))
                else:
                    worst = max(worst, abs(difference))
                cases += 1
            worst_overall = max(worst_overall, worst)
            tee(f"  {name:<34s}{step:>10.1e}{len(rows):>8d}{worst:>18.6e}")
    tee("")
    tee(f"  comparisons                        : {cases}")
    tee(f"  worst relative difference          : {worst_overall:.6e}")
    tee.check(
        "Birnbaum importance matches a central finite difference across five decades of step",
        worst_overall <= FINITE_DIFFERENCE_TOLERANCE,
        f"worst {worst_overall:.6e} against tolerance {FINITE_DIFFERENCE_TOLERANCE:.1e}",
    )
    return worst_overall, cases


def check_exact_pivot(tee: Tee) -> float:
    """Birnbaum against the whole-interval pivot, which is exact for a multilinear Q."""
    tee.rule("Birnbaum against the whole-interval pivot, which is exact")
    tee("")
    tee("  Because Q_sys is multilinear, Q_sys(q_i = 1) - Q_sys(q_i = 0) IS the")
    tee("  derivative. This is the same relation the implementation uses, so it is")
    tee("  reported as a consistency check and not as independent evidence; the")
    tee("  independent evidence is the finite difference above.")
    tee("")
    worst = 0.0
    for name, (diagram, hours) in sorted(diagrams().items()):
        base = {b.name: b.distribution.unreliability(hours) for b in diagram.blocks()}
        for row in importance_table(diagram, hours):
            high = pivot_unreliability(diagram, hours, row.event, 1.0, base_unreliabilities=base)
            low = pivot_unreliability(diagram, hours, row.event, 0.0, base_unreliabilities=base)
            reference = high - low
            if reference != 0.0:
                worst = max(worst, abs(reference - row.birnbaum) / abs(reference))
        del name
    tee(f"  worst relative difference          : {worst:.6e}")
    tee.check("Birnbaum equals the whole-interval pivot", worst <= 1e-14, f"worst {worst:.6e}")
    return worst


def check_identities(tee: Tee) -> None:
    """Criticality identity, Fussell-Vesely ordering and ranges."""
    tee.rule("the identities")
    tee("")
    tee("  I_C(i) = (Q_sys - Q_sys(q_i = 0)) / Q_sys, from the pivotal")
    tee("  decomposition, and I_FV(i) >= I_C(i) always.")
    tee("")
    tee(f"  declared tolerance : {IDENTITY_TOLERANCE:.1e} relative")
    tee("")
    worst_criticality = 0.0
    fv_below_criticality = 0
    out_of_range = 0
    negative_birnbaum = 0
    rows_checked = 0
    for name, (diagram, hours) in sorted(diagrams().items()):
        base = {b.name: b.distribution.unreliability(hours) for b in diagram.blocks()}
        q_system = system_unreliability_at(diagram, base)
        for row in importance_table(diagram, hours):
            rows_checked += 1
            pinned = pivot_unreliability(
                diagram, hours, row.event, 0.0, base_unreliabilities=base
            )
            expected = (q_system - pinned) / q_system
            if expected != 0.0:
                worst_criticality = max(
                    worst_criticality, abs(row.criticality - expected) / abs(expected)
                )
            if row.fussell_vesely < row.criticality - 1e-12:
                fv_below_criticality += 1
            if not -1e-12 <= row.fussell_vesely <= 1.0 + 1e-12:
                out_of_range += 1
            if row.birnbaum < -1e-15:
                negative_birnbaum += 1
        del name
    tee(f"  rows checked                       : {rows_checked}")
    tee(f"  worst criticality identity error   : {worst_criticality:.6e}")
    tee(f"  rows with FV below criticality     : {fv_below_criticality}")
    tee(f"  rows with FV outside [0, 1]        : {out_of_range}")
    tee(f"  rows with negative Birnbaum        : {negative_birnbaum}")
    tee.check(
        "the criticality identity holds",
        worst_criticality <= IDENTITY_TOLERANCE,
        f"worst {worst_criticality:.6e} against tolerance {IDENTITY_TOLERANCE:.1e}",
    )
    tee.check(
        "Fussell-Vesely is at least criticality for every basic event",
        fv_below_criticality == 0,
        f"{fv_below_criticality} violations",
    )
    tee.check(
        "Fussell-Vesely lies in [0, 1] for every basic event",
        out_of_range == 0,
        f"{out_of_range} violations",
    )
    tee.check(
        "Birnbaum is non-negative, as a coherent structure requires",
        negative_birnbaum == 0,
        f"{negative_birnbaum} violations",
    )


def check_rare_event_gap(tee: Tee) -> None:
    """Exact Fussell-Vesely against the rare-event approximation."""
    tee.rule("exact Fussell-Vesely against the rare-event approximation")
    tee("")
    tee("  The rare-event approximation replaces P(union of cut sets containing i)")
    tee("  by the SUM of their probabilities, which overestimates. It is what most")
    tee("  tools report, and it is the usual reason a published Fussell-Vesely")
    tee("  figure exceeds 1. This package computes the union exactly, by")
    tee("  enumerating the states of the events involved. The gap is on record")
    tee("  here rather than left implicit.")
    tee("")
    tee(
        f"  {'diagram':<34s}{'event':<22s}{'exact FV':>14s}"
        f"{'rare-event FV':>16s}{'excess':>12s}"
    )
    tee(f"  {'-' * 98}")
    worst_excess = 0.0
    above_one = 0
    for name, (diagram, hours) in sorted(diagrams().items()):
        base = {b.name: b.distribution.unreliability(hours) for b in diagram.blocks()}
        q_system = system_unreliability_at(diagram, base)
        cuts = minimal_cut_sets(diagram)
        for row in importance_table(diagram, hours):
            relevant = [c for c in cuts if row.event in c]
            total = 0.0
            for cut in relevant:
                product = 1.0
                for event in sorted(cut):
                    product *= base[event]
                total += product
            approximate = total / q_system
            excess = approximate - row.fussell_vesely
            worst_excess = max(worst_excess, excess)
            if approximate > 1.0:
                above_one += 1
            if excess > 1e-4:
                tee(
                    f"  {name:<34s}{row.event:<22s}{row.fussell_vesely:>14.9f}"
                    f"{approximate:>16.9f}{excess:>12.3e}"
                )
        del name
    tee("")
    tee(f"  worst rare-event excess over exact : {worst_excess:.6e}")
    tee(f"  rare-event figures above 1         : {above_one}")
    tee.check(
        "the rare-event approximation never underestimates the exact Fussell-Vesely",
        worst_excess >= -1e-12,
        f"worst excess {worst_excess:.6e}",
    )
    tee(
        "  Note: the rare-event excess being small on these diagrams is a "
        "property of"
    )
    tee("  these diagrams, not a general result. It grows with the overlap between")
    tee("  cut sets and with the per-event unreliability.")


def check_quantification_agreement(tee: Tee) -> None:
    """Structural quantification against exhaustive state enumeration."""
    tee.rule("the structural quantification against exhaustive state enumeration")
    tee("")
    tee("  Importance measures are built on repeated evaluations of Q_sys, so an")
    tee("  error in Q_sys would show up in all of them. Q_sys is therefore also")
    tee("  checked here against a sum over all 2**n states.")
    tee("")
    tee(f"  {'diagram':<34s}{'events':>8s}{'structural Q':>20s}{'enumerated Q':>20s}{'rel':>12s}")
    tee(f"  {'-' * 96}")
    worst = 0.0
    for name, (diagram, hours) in sorted(diagrams().items()):
        base = {b.name: b.distribution.unreliability(hours) for b in diagram.blocks()}
        structural = system_unreliability_at(diagram, base)
        exhaustive = boolean_system_unreliability(diagram, base)
        difference = (
            abs(structural - exhaustive) / abs(exhaustive) if exhaustive else abs(structural)
        )
        worst = max(worst, difference)
        tee(
            f"  {name:<34s}{len(base):>8d}{structural:>20.15f}"
            f"{exhaustive:>20.15f}{difference:>12.3e}"
        )
    tee("")
    tee(f"  worst relative difference          : {worst:.6e}")
    tee.check(
        "system unreliability matches exhaustive state enumeration",
        worst <= 1e-12,
        f"worst {worst:.6e}",
    )


def check_refusals(tee: Tee) -> None:
    """The diagrams for which importance measures are refused, and the message."""
    tee.rule("refusals")
    tee("")
    tee("  Importance measures are not defined over a diagram whose")
    tee("  quantification is not the independent-event arithmetic over its")
    tee("  structure function. Returning a figure whose meaning differs from its")
    tee("  name would be worse than refusing, so these refuse.")
    tee("")
    for name in ("load_sharing_pair", "warm_standby", "smallsat_bus"):
        diagram = CASES[name]
        try:
            importance_table(diagram, 1000.0)
            tee.check(f"{name} is refused", False, "it was not refused")
        except ValueError as error:
            first = str(error).split(".")[0]
            tee(f"  {name:<24s} refused: {first}.")
            tee.check(f"{name} is refused with an explanation", True)


def main() -> int:
    tee = Tee("validate_importance_output.txt")
    header(tee, "importance measures against finite-difference perturbation")
    timer = Timer()
    with timer:
        check_finite_difference(tee)
        check_exact_pivot(tee)
        check_identities(tee)
        check_rare_event_gap(tee)
        check_quantification_agreement(tee)
        check_refusals(tee)

    tee.rule("a worked importance table")
    tee("")
    diagram, hours = diagrams()["voting_triple_with_common_cause"]
    tee(f"  {diagram.name} over {hours:.6g} hours")
    tee("")
    tee(
        f"  {'basic event':<28s}{'q':>14s}{'Birnbaum':>14s}"
        f"{'criticality':>14s}{'Fussell-Vesely':>16s}{'cuts':>6s}"
    )
    tee(f"  {'-' * 92}")
    for row in importance_table(diagram, hours):
        tee(
            f"  {row.event:<28s}{row.unreliability:>14.6e}{row.birnbaum:>14.6e}"
            f"{row.criticality:>14.6f}{row.fussell_vesely:>16.6f}{row.cut_sets:>6d}"
        )

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
