"""The worked example the README quotes, run end to end.

Everything the README shows under "A worked example" is produced by this
script, so the two cannot drift. The script loads a declared diagram from
``examples/diagrams/``, quantifies it over a declared mission profile, runs the
headline comparison, lists the cut sets, and prints the importance table where
the diagram admits one.
"""

from __future__ import annotations

from _bootstrap import Tee, Timer, bootstrap, header, relative

bootstrap()

from reliablock import __version__  # noqa: E402
from reliablock.commoncause import apply_common_cause, strip_common_cause  # noqa: E402
from reliablock.diagram import (  # noqa: E402
    Block,
    Diagram,
    KOfN,
    Series,
    load_diagram,  # noqa: E402
)
from reliablock.distributions import Exponential  # noqa: E402
from reliablock.mission import MissionProfile, load_mission  # noqa: E402
from reliablock.montecarlo import simulate, trials_for_half_width  # noqa: E402
from reliablock.quantify import quantify  # noqa: E402
from reliablock.report import (  # noqa: E402
    common_cause_comparison,
    cut_set_report,
    diagram_report,
    importance_report,
    mission_report,
    quantity_tree,
)
from reliablock.structure import minimal_cut_sets  # noqa: E402

DIAGRAM = "examples/diagrams/smallsat_bus.yaml"
MISSION = "examples/diagrams/three_year.yaml"
VOTING = "examples/diagrams/voting_triple.yaml"
SEED = 20261025


def main() -> int:
    tee = Tee("worked_example_output.txt")
    header(tee, f"worked example, reliablock {__version__}")
    tee("")
    tee("  Relative paths are used throughout. Every script in this directory")
    tee("  changes to the repository root first, derived from __file__, so a")
    tee("  relative path means the same thing whatever directory the script was")
    tee("  invoked from. See validation/_bootstrap.py.")
    tee("")
    tee(f"  diagram            : {DIAGRAM}")
    tee(f"  mission            : {MISSION}")

    timer = Timer()
    diagram = load_diagram(DIAGRAM)
    mission = load_mission(MISSION)

    tee.require(
        "the declared diagram loads and round-trips through YAML",
        load_diagram(DIAGRAM).to_yaml() == diagram.to_yaml(),
    )

    tee.rule("the declaration")
    tee("")
    for line in diagram_report(diagram):
        tee(f"  {line}")

    tee.rule("the mission profile")
    tee("")
    for line in mission_report(diagram, mission):
        tee(f"  {line}")

    tee.rule("closed-form quantification")
    with timer:
        result = quantify(diagram, mission)
    tee("")
    tee(f"  system reliability   : {result.reliability:.12f}")
    tee(f"  system unreliability : {result.unreliability:.9e}")
    tee(f"  cumulative hazard    : {result.hazard:.9e}")
    tee(f"  R + Q - 1            : {result.complement_residual:.3e}")
    tee("")
    tee("  per-node breakdown:")
    tee("")
    for line in quantity_tree(result):
        tee(f"  {line}")

    tee.rule("the headline: with and without the declared common cause")
    tee("")
    for line in common_cause_comparison(diagram, mission):
        tee(f"  {line}")

    tee.rule("minimal cut sets")
    tee("")
    for line in cut_set_report(apply_common_cause(diagram)):
        tee(f"  {line}")

    tee.rule("importance measures, on a diagram that admits them")
    tee("")
    tee("  The smallsat bus contains a standby node, whose quantification is not")
    tee("  the independent-event arithmetic over its structure function, so")
    tee("  importance measures are refused for it. The voting triple is Boolean")
    tee("  throughout and admits them.")
    tee("")
    try:
        importance_report(diagram, mission)
        tee.check("the bus is refused for importance measures", False, "it was not refused")
    except ValueError as error:
        tee(f"  smallsat bus: {error}")
        tee.check("the bus is refused for importance measures, with a reason", True)
    tee("")
    voting = load_diagram(VOTING)
    for line in importance_report(apply_common_cause(voting), mission):
        tee(f"  {line}")

    tee.rule("Monte Carlo, scored against the closed form")
    tee("")
    independent = strip_common_cause(diagram)
    with_cc = apply_common_cause(diagram)
    rows = []
    with timer:
        for label, structure in (
            ("independence assumed", independent),
            ("declared common cause", with_cc),
        ):
            closed = quantify(structure, mission)
            trials = trials_for_half_width(closed.unreliability, 0.01 * closed.unreliability)
            simulated = simulate(structure, mission, trials, seed=SEED)
            rows.append((label, closed, simulated, trials))
    tee(
        f"  {'half':<24s}{'closed form R':>18s}{'simulated R':>18s}"
        f"{'trials':>10s}{'half-width':>13s}{'sigma':>8s}"
    )
    tee(f"  {'-' * 91}")
    for label, closed, simulated, trials in rows:
        tee(
            f"  {label:<24s}{closed.reliability:>18.12f}{simulated.reliability:>18.12f}"
            f"{trials:>10d}{simulated.half_width:>13.6e}"
            f"{simulated.standard_errors_from(closed.reliability):>8.4f}"
        )
    tee("")
    for label, closed, simulated, _trials in rows:
        tee.check(
            f"the simulation of the {label} half is within 4 standard errors",
            simulated.standard_errors_from(closed.reliability) < 4.0,
            f"{simulated.standard_errors_from(closed.reliability):.4f}",
        )

    tee.rule("the short snippet the README quotes under 'A worked example'")
    tee("")
    tee("  The README's 15-line snippet, run verbatim, so that the figures it")
    tee("  shows are produced by a committed script and cannot drift from it.")
    tee("")
    wheels = KOfN(
        k=3,
        items=tuple(
            Block(name=f"wheel_{s}", distribution=Exponential(1.0e-6))
            for s in ("a", "b", "c", "d")
        ),
        beta_factor=0.08,
        label="reaction_wheels",
    )
    snippet = Diagram(
        "wheels behind one computer",
        Series(items=(Block(name="obc", distribution=Exponential(4.0e-7)), wheels)),
    )
    snippet_mission = MissionProfile.from_hours(26298.0)
    for label, structure in (
        ("independence assumed ", strip_common_cause(snippet)),
        ("declared common cause", apply_common_cause(snippet)),
    ):
        snippet_result = quantify(structure, snippet_mission)
        singles = [c for c in minimal_cut_sets(structure) if len(c) == 1]
        tee(
            f"  {label}  R = {snippet_result.reliability:.9f}  "
            f"Q = {snippet_result.unreliability:.6e}  "
            f"single points of failure = {len(singles)}"
        )
        tee("    singles: " + ", ".join(sorted(sorted(c)[0] for c in singles)))

    tee.rule("the numbers the README quotes")
    tee("")
    q_independent = quantify(independent, mission).unreliability
    q_with_cc = quantify(with_cc, mission).unreliability
    singles_before = sum(1 for c in minimal_cut_sets(independent) if len(c) == 1)
    singles_after = sum(1 for c in minimal_cut_sets(with_cc) if len(c) == 1)
    tee(f"  R assuming independence            : {quantify(independent, mission).reliability:.12f}")
    tee(f"  R with the declared common cause   : {quantify(with_cc, mission).reliability:.12f}")
    tee(f"  Q assuming independence            : {q_independent:.9e}")
    tee(f"  Q with the declared common cause   : {q_with_cc:.9e}")
    tee(f"  Q ratio                            : {q_with_cc / q_independent:.6f}")
    tee(f"  single points of failure, before   : {singles_before}")
    tee(f"  single points of failure, after    : {singles_after}")
    tee(f"  minimal cut sets, before           : {len(minimal_cut_sets(independent))}")
    tee(f"  minimal cut sets, after            : {len(minimal_cut_sets(with_cc))}")

    tee.rule("timing, declared unstable")
    tee("")
    tee("  | quantity | value | stability |")
    tee("  |---|---|---|")
    tee(
        f"  | quantification and simulation | {timer.elapsed_s:.2f} s "
        "| UNSTABLE, shared hardware |"
    )

    tee.summarise_failures()
    tee("")
    tee(f"  diagram read from  : {relative(DIAGRAM)}")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
