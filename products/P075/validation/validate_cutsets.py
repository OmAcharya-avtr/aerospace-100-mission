"""Minimal cut sets against exhaustive state enumeration, on every small diagram.

Two routes with nothing in common
---------------------------------

:func:`reliablock.structure.minimal_cut_sets` combines each node's children's
cut sets structurally: the union for a series node, one cut from each child for
a parallel node, one cut from each of every ``(n - k + 1)``-subset for a
k-of-n node, the hypergraph dual of the path sets for a network.

:func:`reliablock.structure.enumerate_states` walks all ``2**n`` states,
evaluates the structure function on each, collects the states it calls a
failure, and keeps those from which restoring any single failed event restores
the system. It never looks at the structure's shape.

The two share no code path. This script requires them to produce **identical**
sets -- not a count, the sets themselves -- on every diagram small enough to
enumerate, and reports the state counts so that the enumeration's own size is
visible.

Three structural properties are also checked directly against the structure
function, which is the definition a cut set has to satisfy:

1. every reported cut set, when all its events fail, fails the system;
2. restoring any single event of a reported cut set restores the system, which
   is minimality;
3. no reported cut set contains another, which is minimality again, checked
   set-theoretically rather than through the structure function.
"""

from __future__ import annotations

from itertools import combinations

from _bootstrap import Tee, Timer, bootstrap, header

bootstrap()

from reliablock.bridge import BRIDGE_CUT_SETS, BRIDGE_PATH_SETS, bridge_diagram  # noqa: E402
from reliablock.cases import CASES  # noqa: E402
from reliablock.commoncause import apply_common_cause  # noqa: E402
from reliablock.diagram import (  # noqa: E402
    Block,
    Diagram,
    KOfN,
    Network,
    Parallel,
    Series,
    Standby,
)
from reliablock.distributions import Exponential, FixedProbability  # noqa: E402
from reliablock.structure import (  # noqa: E402
    MAX_ENUMERATION_EVENTS,
    basic_events,
    enumerate_states,
    minimal_cut_sets,
    minimal_path_sets,
    structure_function,
)


def _block(name: str, rate: float = 1e-5) -> Block:
    return Block(name=name, distribution=Exponential(rate))


def _one_shot(name: str) -> Block:
    return Block(name=name, distribution=FixedProbability(0.1))


def extra_diagrams() -> dict[str, Diagram]:
    """Diagrams built for this check, beyond the built-in cases.

    Chosen to cover every node kind and every combination the recursion has a
    separate branch for, including the two that are easiest to get wrong: a
    k-of-n group nested inside a parallel group, and a network that is not
    series-parallel.
    """
    out: dict[str, Diagram] = {}
    out["bridge"] = bridge_diagram(0.9)
    out["two_bridges"] = Diagram(
        "bridge then a link",
        Network(
            edges=(
                ("s", "a", _one_shot("e1")),
                ("s", "b", _one_shot("e2")),
                ("a", "b", _one_shot("e3")),
                ("a", "m", _one_shot("e4")),
                ("b", "m", _one_shot("e5")),
                ("m", "t", _one_shot("e6")),
            ),
            source="s",
            sink="t",
        ),
    )
    out["nested_parallel_of_series"] = Diagram(
        "nested",
        Parallel(
            items=(
                Series(items=(_block("a1"), _block("a2"))),
                Series(items=(_block("b1"), _block("b2"))),
            ),
            label="branches",
        ),
    )
    out["k_of_n_inside_parallel"] = Diagram(
        "k inside parallel",
        Parallel(
            items=(
                KOfN(k=2, items=(_block("x1"), _block("x2"), _block("x3")), label="xs"),
                _block("y"),
            ),
            label="either",
        ),
    )
    out["series_of_k_of_n"] = Diagram(
        "series of groups",
        Series(
            items=(
                KOfN(k=2, items=(_block("p1"), _block("p2"), _block("p3")), label="ps"),
                KOfN(k=3, items=(_block("q1"), _block("q2"), _block("q3")), label="qs"),
            ),
            label="chain",
        ),
    )
    out["two_cold_standbys"] = Diagram(
        "two spares",
        Standby(
            primary=_block("p"),
            standbys=(_block("s1"), _block("s2")),
            switch_reliability=0.99,
            label="sb",
        ),
    )
    out["standby_in_series"] = Diagram(
        "standby then a link",
        Series(
            items=(
                Standby(
                    primary=_block("p"),
                    standbys=(_block("s"),),
                    switch_reliability=0.99,
                    label="sb",
                ),
                _block("tail"),
            ),
            label="chain",
        ),
    )
    out["smallsat_bus_with_common_cause"] = apply_common_cause(CASES["smallsat_bus"])
    for name, diagram in CASES.items():
        out[f"case_{name}"] = diagram
    return out


def check_one(tee: Tee, name: str, diagram: Diagram) -> tuple[bool, int, int, int]:
    """Compare the two routes on one diagram. Returns (agree, events, cuts, states)."""
    events = basic_events(diagram)
    structural = minimal_cut_sets(diagram)
    if len(events) > MAX_ENUMERATION_EVENTS:
        tee(
            f"  {name:<36s}{len(events):>4d} events  "
            f"{len(structural):>4d} cuts  SKIPPED, above the 2**"
            f"{MAX_ENUMERATION_EVENTS} enumeration cap"
        )
        return True, len(events), len(structural), 0
    exhaustive = enumerate_states(diagram)
    agree = set(structural) == set(exhaustive["minimal_cut_sets"])
    tee(
        f"  {name:<36s}{len(events):>4d} events{exhaustive['states']:>9d} states"
        f"{exhaustive['failed_states']:>9d} failed{len(structural):>6d} cuts"
        f"{exhaustive['n_minimal']:>6d} cuts{'  SAME' if agree else '  DIFFER'}"
    )
    if not agree:
        only_structural = sorted(
            sorted(c) for c in set(structural) - set(exhaustive["minimal_cut_sets"])
        )
        only_exhaustive = sorted(
            sorted(c) for c in set(exhaustive["minimal_cut_sets"]) - set(structural)
        )
        for cut in only_structural:
            tee(f"      structural only: {{{', '.join(cut)}}}")
        for cut in only_exhaustive:
            tee(f"      exhaustive only: {{{', '.join(cut)}}}")
    return agree, len(events), len(structural), exhaustive["states"]


def check_properties(tee: Tee, diagrams: dict[str, Diagram]) -> tuple[int, int, int]:
    """Check the three defining properties directly against the structure function."""
    fails_when_cut = 0
    restored_when_minimal = 0
    no_superset = 0
    for diagram in diagrams.values():
        events = set(basic_events(diagram))
        cuts = minimal_cut_sets(diagram)
        for cut in cuts:
            if structure_function(diagram, events - cut):
                fails_when_cut += 1
            for element in sorted(cut):
                if not structure_function(diagram, (events - cut) | {element}):
                    restored_when_minimal += 1
        for first, second in combinations(cuts, 2):
            if first <= second or second <= first:
                no_superset += 1
    return fails_when_cut, restored_when_minimal, no_superset


def main() -> int:
    tee = Tee("validate_cutsets_output.txt")
    header(tee, "minimal cut sets: structural recursion against exhaustive enumeration")
    tee("")
    tee("  The two routes share no code path. The structural recursion combines")
    tee("  children's cut sets by node kind; the exhaustive route evaluates the")
    tee("  structure function on every one of 2**n states and keeps the minimal")
    tee("  failures. Agreement is required on the SETS, not on their count.")
    tee("")
    tee(f"  enumeration cap                    : 2**{MAX_ENUMERATION_EVENTS} states")

    diagrams = extra_diagrams()
    timer = Timer()

    tee.rule("per-diagram comparison")
    tee("")
    tee(
        f"  {'diagram':<36s}{'events':>10s}{'states':>15s}{'failed':>16s}"
        f"{'structural':>12s}{'exhaustive':>11s}"
    )
    tee(f"  {'-' * 102}")
    disagreements: list[str] = []
    total_states = 0
    with timer:
        for name in sorted(diagrams):
            agree, _events, _cuts, states = check_one(tee, name, diagrams[name])
            total_states += states
            if not agree:
                disagreements.append(name)
    tee("")
    tee(f"  diagrams compared                  : {len(diagrams)}")
    tee(f"  states enumerated in total         : {total_states}")
    tee(f"  disagreements                      : {len(disagreements)}")
    for name in disagreements:
        tee(f"    {name}")
    tee.check(
        "the structural and exhaustive cut sets are identical on every diagram",
        not disagreements,
        f"{len(disagreements)} of {len(diagrams)} disagree",
    )

    tee.rule("the defining properties, checked against the structure function")
    with timer:
        fails_when_cut, restored_when_minimal, no_superset = check_properties(tee, diagrams)
    tee("")
    tee(f"  cut sets that did NOT fail the system      : {fails_when_cut}")
    tee(f"  minimal cut sets NOT restored by one event : {restored_when_minimal}")
    tee(f"  reported cut sets containing another       : {no_superset}")
    tee.check(
        "every reported cut set fails the system when all its events fail",
        fails_when_cut == 0,
        f"{fails_when_cut} violations",
    )
    tee.check(
        "restoring any one event of a reported cut set restores the system",
        restored_when_minimal == 0,
        f"{restored_when_minimal} violations",
    )
    tee.check(
        "no reported cut set contains another",
        no_superset == 0,
        f"{no_superset} violations",
    )

    tee.rule("path sets, as the dual check")
    tee("")
    tee("  Every minimal path set must succeed, and removing any one of its")
    tee("  events must fail the system. The path sets are also what the")
    tee("  Esary-Proschan upper bound and the network cut-set dual are built")
    tee("  from, so an error in them would propagate.")
    path_failures = 0
    path_non_minimal = 0
    total_paths = 0
    with timer:
        for diagram in diagrams.values():
            for path in minimal_path_sets(diagram):
                total_paths += 1
                if not structure_function(diagram, path):
                    path_failures += 1
                for element in sorted(path):
                    if structure_function(diagram, path - {element}):
                        path_non_minimal += 1
    tee("")
    tee(f"  path sets checked                  : {total_paths}")
    tee(f"  path sets that did not succeed     : {path_failures}")
    tee(f"  path sets not minimal              : {path_non_minimal}")
    tee.check("every minimal path set succeeds", path_failures == 0, f"{path_failures} violations")
    tee.check(
        "removing any one event of a minimal path set fails the system",
        path_non_minimal == 0,
        f"{path_non_minimal} violations",
    )

    tee.rule("the bridge network's declared cut and path sets")
    tee("")
    tee("  reliablock.bridge declares the four minimal cut sets and the four")
    tee("  minimal path sets of the five-edge bridge as literals, derived in that")
    tee("  module's docstring. They are checked here against the enumeration.")
    bridge = bridge_diagram(0.9)
    enumerated_cuts = set(minimal_cut_sets(bridge))
    enumerated_paths = set(minimal_path_sets(bridge))
    exhaustive = enumerate_states(bridge)
    tee("")
    tee("  declared cut sets  : " + ", ".join(
        "{" + ", ".join(sorted(c)) + "}" for c in BRIDGE_CUT_SETS
    ))
    tee("  enumerated cut sets: " + ", ".join(
        "{" + ", ".join(sorted(c)) + "}" for c in sorted(enumerated_cuts, key=lambda s: sorted(s))
    ))
    tee("  declared path sets : " + ", ".join(
        "{" + ", ".join(sorted(c)) + "}" for c in BRIDGE_PATH_SETS
    ))
    tee("  enumerated paths   : " + ", ".join(
        "{" + ", ".join(sorted(c)) + "}" for c in sorted(enumerated_paths, key=lambda s: sorted(s))
    ))
    tee(
        f"  exhaustive states  : {exhaustive['states']}, "
        f"of which {exhaustive['failed_states']} fail"
    )
    tee.check(
        "the declared bridge cut sets match the enumeration",
        enumerated_cuts == set(BRIDGE_CUT_SETS) == set(exhaustive["minimal_cut_sets"]),
    )
    tee.check(
        "the declared bridge path sets match the enumeration",
        enumerated_paths == set(BRIDGE_PATH_SETS),
    )
    tee.check(
        "exactly half the bridge's 32 states fail, as a self-dual structure requires",
        exhaustive["failed_states"] * 2 == exhaustive["states"],
        f"{exhaustive['failed_states']} of {exhaustive['states']}",
    )

    tee.rule("what a declared common cause does to the cut sets")
    tee("")
    tee("  This is the qualitative half of the headline, stated as a cut-set")
    tee("  count rather than a percentage.")
    tee("")
    tee(
        f"  {'diagram':<30s}{'cuts':>8s}{'order 1':>10s}{'cuts':>8s}{'order 1':>10s}"
    )
    tee(f"  {'':<30s}{'(indep)':>8s}{'(indep)':>10s}{'(cc)':>8s}{'(cc)':>10s}")
    tee(f"  {'-' * 66}")
    from reliablock.commoncause import strip_common_cause

    for name in sorted(CASES):
        diagram = CASES[name]
        if not diagram.declared_betas:
            continue
        independent = minimal_cut_sets(strip_common_cause(diagram))
        with_cc = minimal_cut_sets(apply_common_cause(diagram))
        tee(
            f"  {name:<30s}{len(independent):>8d}"
            f"{sum(1 for c in independent if len(c) == 1):>10d}"
            f"{len(with_cc):>8d}{sum(1 for c in with_cc if len(c) == 1):>10d}"
        )
    tee("")
    tee("  Every declared beta factor adds exactly one cut set of order one, by")
    tee("  construction: the shared-cause event is a single basic event in series")
    tee("  with its group.")
    added_correctly = True
    for name in sorted(CASES):
        diagram = CASES[name]
        betas = diagram.declared_betas
        if not betas:
            continue
        independent = minimal_cut_sets(strip_common_cause(diagram))
        with_cc = minimal_cut_sets(apply_common_cause(diagram))
        before = sum(1 for c in independent if len(c) == 1)
        after = sum(1 for c in with_cc if len(c) == 1)
        if after - before != len(betas):
            added_correctly = False
            tee(f"    {name}: {len(betas)} betas but {after - before} new order-1 cut sets")
    tee.check(
        "each declared beta factor adds exactly one cut set of order one",
        added_correctly,
    )

    tee.rule("timing, declared unstable")
    tee("")
    tee("  | quantity | value | stability |")
    tee("  |---|---|---|")
    tee(f"  | enumeration wall clock | {timer.elapsed_s:.2f} s | UNSTABLE, shared hardware |")
    tee(f"  | states enumerated | {total_states} | stable |")

    tee.summarise_failures()
    tee("")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
