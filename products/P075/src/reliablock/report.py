"""Text reports: the headline comparison, the cut sets, the importance table.

Nothing in this module prints. Every function returns a list of lines, so a
caller decides where they go, and the same text appears in the CLI, the
examples and the validation output without a second formatting path that could
disagree with the first.

The headline report
-------------------

:func:`common_cause_comparison` is the report this package exists for. It runs
the same quantifier over two diagrams -- one with every declared beta factor
suppressed, one with every declared beta factor in the structure -- and puts
them side by side, with the ratio of the unreliabilities and the change in the
minimal cut sets. The change in the cut sets is the part that is usually
missing from a redundancy argument: a declared beta factor turns an ``n``-fold
redundant group into a structure with a cut set of size one, and that is a
qualitative statement about the design, not a percentage.
"""

from __future__ import annotations

from .bounds import esary_proschan_bounds
from .commoncause import apply_common_cause, group_beta_effects, strip_common_cause
from .diagram import Diagram
from .importance import importance_table
from .mission import MissionProfile
from .quantify import Quantity, quantify
from .structure import basic_events, minimal_cut_sets

__all__ = [
    "common_cause_comparison",
    "cut_set_report",
    "diagram_report",
    "importance_report",
    "mission_report",
    "quantity_tree",
]

_RULE = "-" * 78


def _mission(mission: MissionProfile | float) -> MissionProfile:
    """Coerce a duration to a single-phase profile."""
    return mission if isinstance(mission, MissionProfile) else MissionProfile.from_hours(mission)


def diagram_report(diagram: Diagram) -> list[str]:
    """Structural summary of a declared diagram: what was declared, and what it implies."""
    lines = [f"diagram            : {diagram.name}"]
    if diagram.description:
        lines.append(f"description        : {diagram.description}")
    kinds: dict[str, int] = {}
    for item in diagram.walk():
        if hasattr(item, "kind"):
            kinds[item.kind] = kinds.get(item.kind, 0) + 1
    lines.append(f"blocks             : {len(diagram.blocks())}")
    lines.append(f"basic events       : {len(basic_events(diagram))}")
    lines.append(
        "node kinds         : "
        + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()) if k != "block")
    )
    betas = diagram.declared_betas
    if betas:
        lines.append(
            "declared betas     : "
            + ", ".join(f"{name}={value:g}" for name, value in sorted(betas.items()))
        )
    else:
        lines.append(
            "declared betas     : none -- every redundant group is assumed independent"
        )
    lines.append(f"boolean throughout : {diagram.is_boolean}")
    if not diagram.is_boolean:
        lines.append(
            "                     a standby or shared_load node is present: its "
            "Boolean structure"
        )
        lines.append(
            "                     function and cut sets are exact, its "
            "quantification is not the"
        )
        lines.append(
            "                     independent-event arithmetic over them, and "
            "importance measures"
        )
        lines.append("                     are refused for this diagram.")
    for note in diagram.notes:
        lines.append(f"note               : {note}")
    return lines


def mission_report(diagram: Diagram, mission: MissionProfile | float) -> list[str]:
    """Per-block effective ages, and the phases that produced them."""
    profile = _mission(mission)
    ages = profile.effective_ages(diagram)
    lines = [
        f"mission            : {profile.name}",
        f"elapsed            : {profile.total_hours:.6g} h over {len(profile.phases)} phase(s)",
        "",
        f"  {'phase':<20s}{'hours':>12s}  {'block':<24s}{'duty':>8s}"
        f"{'stress':>8s}{'eff h':>14s}",
        f"  {_RULE[:76]}",
    ]
    for row in profile.breakdown(diagram):
        lines.append(
            f"  {row['phase']:<20s}{row['duration_hours']:>12.6g}  "
            f"{row['block']:<24s}{row['duty_cycle']:>8.3f}"
            f"{row['stress_factor']:>8.3f}{row['effective_hours']:>14.6g}"
        )
    lines.append("")
    lines.append(f"  {'block':<28s}{'effective age (h)':>20s}")
    lines.append(f"  {_RULE[:48]}")
    for name in sorted(ages):
        lines.append(f"  {name:<28s}{ages[name]:>20.6g}")
    return lines


def quantity_tree(quantity: Quantity, indent: int = 0) -> list[str]:
    """The per-node breakdown of a quantification, as an indented tree."""
    pad = "  " * indent
    label = quantity.label or quantity.kind
    lines = [
        f"{pad}{label:<{max(1, 36 - 2 * indent)}s} "
        f"R={quantity.reliability:.12f}  Q={quantity.unreliability:.6e}  "
        f"H={quantity.hazard:.6e}"
    ]
    for note in quantity.notes:
        lines.append(f"{pad}  . {note}")
    for part in quantity.parts:
        lines.extend(quantity_tree(part, indent + 1))
    return lines


def cut_set_report(diagram: Diagram, limit: int = 40) -> list[str]:
    """Minimal cut sets, grouped by order, with the singletons called out.

    A cut set of order one is a single point of failure. It is listed first and
    named as such, because that is the thing a reviewer is looking for and it is
    exactly what a declared common-cause beta introduces.
    """
    cuts = minimal_cut_sets(diagram)
    by_order: dict[int, list[frozenset[str]]] = {}
    for cut in cuts:
        by_order.setdefault(len(cut), []).append(cut)
    lines = [f"minimal cut sets   : {len(cuts)}"]
    singletons = by_order.get(1, [])
    if singletons:
        lines.append(
            f"single points of failure ({len(singletons)}): "
            + ", ".join(sorted(next(iter(c)) for c in singletons))
        )
    else:
        lines.append("single points of failure: none")
    lines.append("")
    shown = 0
    for order in sorted(by_order):
        lines.append(f"  order {order} ({len(by_order[order])}):")
        for cut in by_order[order]:
            if shown >= limit:
                lines.append(f"    ... {len(cuts) - shown} more not shown")
                return lines
            lines.append("    {" + ", ".join(sorted(cut)) + "}")
            shown += 1
    return lines


def importance_report(
    diagram: Diagram, mission: MissionProfile | float, limit: int = 20
) -> list[str]:
    """Importance table, most Fussell-Vesely-important first."""
    rows = importance_table(diagram, mission)
    lines = [
        "importance measures (Birnbaum, criticality, Fussell-Vesely; exact, not "
        "rare-event)",
        "",
        f"  {'basic event':<28s}{'q':>14s}{'Birnbaum':>14s}{'critical':>12s}"
        f"{'Fus-Ves':>12s}{'cuts':>6s}",
        f"  {_RULE[:86]}",
    ]
    for row in rows[:limit]:
        lines.append(
            f"  {row.event:<28s}{row.unreliability:>14.6e}{row.birnbaum:>14.6e}"
            f"{row.criticality:>12.6f}{row.fussell_vesely:>12.6f}{row.cut_sets:>6d}"
        )
    if len(rows) > limit:
        lines.append(f"  ... {len(rows) - limit} more")
    return lines


def common_cause_comparison(
    diagram: Diagram, mission: MissionProfile | float
) -> list[str]:
    """The headline: reliability with and without the declared common cause.

    Returns the two figures side by side, the ratio of their unreliabilities,
    the change in the minimal cut sets, and the Esary-Proschan bracket on each
    where the diagram admits one.

    A diagram with no declared beta factor gets a report that says so and says
    what the number therefore assumes, rather than an empty table. That case is
    the common one in practice and it is the one worth naming.
    """
    profile = _mission(mission)
    independent = strip_common_cause(diagram)
    with_cc = apply_common_cause(diagram)
    q_ind = quantify(independent, profile)
    q_cc = quantify(with_cc, profile)
    cuts_ind = minimal_cut_sets(independent)
    cuts_cc = minimal_cut_sets(with_cc)
    singles_ind = sorted(next(iter(c)) for c in cuts_ind if len(c) == 1)
    singles_cc = sorted(next(iter(c)) for c in cuts_cc if len(c) == 1)
    betas = diagram.declared_betas

    lines = [
        "reliability with and without the declared common cause",
        "",
        f"  mission                        : {profile.name}, "
        f"{profile.total_hours:.6g} h elapsed",
    ]
    if betas:
        lines.append(
            "  declared beta factors          : "
            + ", ".join(f"{k}={v:g}" for k, v in sorted(betas.items()))
        )
    else:
        lines.append("  declared beta factors          : none")
    lines.extend(
        [
            "",
            f"  {'':<32s}{'independence assumed':>24s}{'declared common cause':>24s}",
            f"  {_RULE}",
            f"  {'system reliability':<32s}{q_ind.reliability:>24.12f}"
            f"{q_cc.reliability:>24.12f}",
            f"  {'system unreliability':<32s}{q_ind.unreliability:>24.6e}"
            f"{q_cc.unreliability:>24.6e}",
            f"  {'cumulative hazard':<32s}{q_ind.hazard:>24.6e}{q_cc.hazard:>24.6e}",
            f"  {'minimal cut sets':<32s}{len(cuts_ind):>24d}{len(cuts_cc):>24d}",
            f"  {'single points of failure':<32s}{len(singles_ind):>24d}"
            f"{len(singles_cc):>24d}",
            f"  {_RULE}",
        ]
    )
    if q_ind.unreliability > 0.0:
        ratio = q_cc.unreliability / q_ind.unreliability
        lines.append(
            f"  unreliability ratio            : {ratio:.6f} "
            "(declared common cause / independence assumed)"
        )
        lines.append(
            f"  unreliability attributable to  : {q_cc.unreliability - q_ind.unreliability:.6e} "
            "absolute"
        )
        lines.append("  the independence assumption      ")
    new_singles = sorted(set(singles_cc) - set(singles_ind))
    if new_singles:
        lines.append("")
        lines.append(
            f"  the declared common cause introduces {len(new_singles)} single point(s) of"
        )
        lines.append("  failure that the independent diagram does not have:")
        for name in new_singles:
            lines.append(f"    {name}")
        lines.append(
            "  This is the qualitative half of the result and it is not a "
            "percentage: a group"
        )
        lines.append(
            "  whose members share a cause is not redundant against that cause, "
            "however many"
        )
        lines.append("  members it has.")
    elif not betas:
        lines.append("")
        lines.append(
            "  No beta factor is declared anywhere in this diagram, so the two "
            "columns are"
        )
        lines.append(
            "  identical by construction. That is not a finding about the "
            "hardware: it means"
        )
        lines.append(
            "  the declaration assumes every member of every redundant group "
            "fails"
        )
        lines.append(
            "  independently of every other. Declare a beta factor to find out "
            "what that"
        )
        lines.append("  assumption is worth.")
    effects = group_beta_effects(diagram, profile)
    if effects:
        lines.append("")
        lines.append("  per-group effect of each declared beta factor:")
        for effect in effects:
            lines.append(f"    {effect}")
        non_conservative = [e for e in effects if not e.conservative]
        if non_conservative:
            lines.append("")
            lines.append(
                f"  WARNING: {len(non_conservative)} group(s) above have a beta factor "
                "that DECREASES the"
            )
            lines.append(
                "  computed unreliability. That is arithmetically correct and it is a "
                "known"
            )
            lines.append(
                "  inadequacy of the beta-factor model, not a defect in this package: "
                "the model"
            )
            lines.append(
                "  has exactly one common-cause mode, the one that fails every member, "
                "and a"
            )
            lines.append(
                "  k-of-n group with k >= 2 fails on fewer failures than that. Use a "
                "model with"
            )
            lines.append(
                "  partial common-cause modes (multiple Greek letter, alpha factor) for "
                "such a"
            )
            lines.append(
                "  group; neither is implemented here. See reliablock.commoncause."
                "group_beta_effects."
            )

    for structure, quantity, name in (
        (independent, q_ind, "independence assumed"),
        (with_cc, q_cc, "declared common cause"),
    ):
        try:
            bracket = esary_proschan_bounds(structure, profile, quantity.reliability)
        except (ValueError, KeyError):  # pragma: no cover - dynamic nodes
            continue
        lines.append("")
        lines.append(f"  {name}: {bracket}")
    return lines
