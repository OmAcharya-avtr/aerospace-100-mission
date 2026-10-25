"""The textbook bridge network: what is derived here and what is cited.

This script is the one the product specification asks for, and the honesty of
its sourcing matters more than its numbers, so the sourcing is stated first.

DERIVED IN THIS REPOSITORY, IN FULL
-----------------------------------

The closed form for the five-component bridge network under equal component
reliability ``p``:

    ``R = 2p^5 - 5p^4 + 2p^3 + 2p^2``

The derivation is written out in the module docstring of
:mod:`reliablock.bridge` and is reproduced in this script's output. It is
pivotal (conditional-probability) decomposition on the bridge edge:

    ``R = p * (1 - (1 - p)^2)^2 + (1 - p) * (1 - (1 - p^2)^2)``
      ``= p (4p^2 - 4p^3 + p^4) + (1 - p)(2p^2 - p^4)``
      ``= 2p^5 - 5p^4 + 2p^3 + 2p^2``

At ``p = 0.9`` it gives ``0.978480``. **That value is derived here, not
quoted.**

CITED FOR THE METHOD, NOT FOR A NUMBER
--------------------------------------

The bridge network and the conditional-probability decomposition of it are
standard material. Barlow, R. E. and Proschan, F. (1975), *Statistical Theory
of Reliability and Life Testing*, Holt, Rinehart and Winston, chapter 1, gives
the pivotal decomposition of a coherent structure. Billinton, R. and Allan,
R. N., *Reliability Evaluation of Engineering Systems*, 2nd edition, Plenum
Press, 1992, treats the bridge network and the conditional-probability method
applied to it.

NOT CITED
---------

**No page number and no published figure is quoted for the bridge result.**
This build had no access to either volume, and a page number that cannot be
checked is worse than none. The specification for this product allows a
derived-and-stated result in place of a citation that cannot be verified, and
that is what this is.

WHAT STANDS IN PLACE OF A CITATION
----------------------------------

Four independent routes to the same polynomial, every one of them reproducible
from this repository alone:

1. the derived closed form above, evaluated by Horner's rule;
2. the non-identical pivotal form, with all five edge reliabilities set equal;
3. inclusion-exclusion over the four minimal path sets, sixteen signed terms;
4. the package's own pivotal decomposition over the declared ``network`` node,
   which is the code path a user exercises.

plus a fifth that shares no algebra with any of them:

5. exhaustive enumeration of all ``2**5 = 32`` states against the Boolean
   structure function.

Four routes to one polynomial, and a fifth to its value, is a stronger
statement than an unverifiable page number would have been.
"""

from __future__ import annotations

from _bootstrap import Tee, Timer, bootstrap, header

bootstrap()

from reliablock.bounds import esary_proschan_bounds  # noqa: E402
from reliablock.bridge import (  # noqa: E402
    BRIDGE_CUT_SETS,
    BRIDGE_PATH_SETS,
    bridge_diagram,
    bridge_reliability_equal,
    bridge_reliability_inclusion_exclusion,
    bridge_reliability_pivotal,
)
from reliablock.importance import boolean_system_unreliability  # noqa: E402
from reliablock.montecarlo import simulate, trials_for_half_width  # noqa: E402
from reliablock.quantify import quantify  # noqa: E402
from reliablock.structure import (  # noqa: E402
    enumerate_states,
    minimal_cut_sets,
    minimal_path_sets,
)

#: Declared tolerance on agreement between the four algebraic routes.
TOLERANCE = 1e-13

SEED = 20261025


def _relative(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else abs(a)


def main() -> int:
    tee = Tee("validate_bridge_output.txt")
    header(tee, "the five-component bridge network: derived here, not quoted")

    tee.rule("sourcing, stated before any number")
    tee("")
    tee("  DERIVED IN THIS REPOSITORY:")
    tee("    R = 2p^5 - 5p^4 + 2p^3 + 2p^2, by pivotal decomposition on the")
    tee("    bridge edge e3:")
    tee("")
    tee("      with e3 working, vertices a and b merge, giving two parallel")
    tee("      pairs in series: (1 - (1-p)^2)^2 = 4p^2 - 4p^3 + p^4")
    tee("      with e3 failed, two disjoint paths in parallel:")
    tee("      1 - (1 - p^2)^2 = 2p^2 - p^4")
    tee("")
    tee("      R = p(4p^2 - 4p^3 + p^4) + (1-p)(2p^2 - p^4)")
    tee("        = 4p^3 - 4p^4 + p^5 + 2p^2 - p^4 - 2p^3 + p^5")
    tee("        = 2p^5 - 5p^4 + 2p^3 + 2p^2")
    tee("")
    tee("    At p = 0.9 this is 2(0.59049) - 5(0.6561) + 2(0.729) + 2(0.81)")
    tee("                     = 1.18098 - 3.2805 + 1.458 + 1.62 = 0.97848")
    tee("")
    tee("  CITED FOR THE METHOD ONLY:")
    tee("    Barlow, R. E. and Proschan, F. (1975), Statistical Theory of")
    tee("    Reliability and Life Testing, Holt, Rinehart and Winston, chapter 1,")
    tee("    for the pivotal decomposition of a coherent structure.")
    tee("    Billinton, R. and Allan, R. N., Reliability Evaluation of")
    tee("    Engineering Systems, 2nd edition, Plenum Press, 1992, for the bridge")
    tee("    network and the conditional-probability method applied to it.")
    tee("")
    tee("  NOT CITED:")
    tee("    No page number and no published figure. This build had no access to")
    tee("    either volume. A page number that cannot be checked is worse than")
    tee("    none, so none is given, and the derivation above stands in its place.")

    timer = Timer()

    tee.rule("route 1 to 4: four algebraic routes to the same polynomial")
    tee("")
    tee(
        f"  {'p':>8s}{'derived form':>22s}{'pivotal form':>22s}"
        f"{'incl-excl':>22s}{'package':>22s}{'worst rel':>12s}"
    )
    tee(f"  {'-' * 110}")
    worst_algebraic = 0.0
    with timer:
        for p in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0):
            derived = bridge_reliability_equal(p)
            pivotal = bridge_reliability_pivotal(p, p, p, p, p)
            inclusion = bridge_reliability_inclusion_exclusion(
                {f"e{i}": p for i in range(1, 6)}
            )
            package = quantify(bridge_diagram(p), 1.0).reliability
            worst = max(
                _relative(pivotal, derived),
                _relative(inclusion, derived),
                _relative(package, derived),
            )
            worst_algebraic = max(worst_algebraic, worst)
            tee(
                f"  {p:>8.3f}{derived:>22.15f}{pivotal:>22.15f}"
                f"{inclusion:>22.15f}{package:>22.15f}{worst:>12.3e}"
            )
    tee("")
    tee(f"  declared tolerance                 : {TOLERANCE:.1e} relative")
    tee(f"  worst relative difference          : {worst_algebraic:.6e}")
    tee.check(
        "all four algebraic routes agree",
        worst_algebraic <= TOLERANCE,
        f"worst {worst_algebraic:.6e} against tolerance {TOLERANCE:.1e}",
    )

    tee.rule("route 5: exhaustive enumeration of all 32 states")
    tee("")
    tee("  This route never forms the polynomial. It evaluates the Boolean")
    tee("  structure function on each of the 2**5 states and sums the probability")
    tee("  of the ones it calls a failure.")
    tee("")
    tee(f"  {'p':>8s}{'derived form':>22s}{'32-state sum':>22s}{'rel diff':>12s}")
    tee(f"  {'-' * 64}")
    worst_enumeration = 0.0
    with timer:
        for p in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
            diagram = bridge_diagram(p)
            unreliabilities = {f"e{i}": 1.0 - p for i in range(1, 6)}
            enumerated = 1.0 - boolean_system_unreliability(diagram, unreliabilities)
            derived = bridge_reliability_equal(p)
            difference = _relative(enumerated, derived)
            worst_enumeration = max(worst_enumeration, difference)
            tee(f"  {p:>8.3f}{derived:>22.15f}{enumerated:>22.15f}{difference:>12.3e}")
    tee("")
    tee(f"  worst relative difference          : {worst_enumeration:.6e}")
    tee.check(
        "exhaustive state enumeration agrees with the derived closed form",
        worst_enumeration <= TOLERANCE,
        f"worst {worst_enumeration:.6e}",
    )

    tee.rule("the structure: cut sets, path sets, state count")
    diagram = bridge_diagram(0.9)
    exhaustive = enumerate_states(diagram)
    tee("")
    tee("  minimal path sets, declared in reliablock.bridge:")
    for path in BRIDGE_PATH_SETS:
        tee(f"    {{{', '.join(sorted(path))}}}")
    tee("  minimal path sets, enumerated by reliablock.structure:")
    for path in minimal_path_sets(diagram):
        tee(f"    {{{', '.join(sorted(path))}}}")
    tee("")
    tee("  minimal cut sets, declared in reliablock.bridge:")
    for cut in BRIDGE_CUT_SETS:
        tee(f"    {{{', '.join(sorted(cut))}}}")
    tee("  minimal cut sets, enumerated by reliablock.structure:")
    for cut in minimal_cut_sets(diagram):
        tee(f"    {{{', '.join(sorted(cut))}}}")
    tee("")
    tee(f"  states                             : {exhaustive['states']}")
    tee(f"  failed states                      : {exhaustive['failed_states']}")
    tee(f"  minimal cut sets                    : {exhaustive['n_minimal']}")
    tee.check(
        "declared and enumerated path sets are identical",
        set(BRIDGE_PATH_SETS) == set(minimal_path_sets(diagram)),
    )
    tee.check(
        "declared and enumerated cut sets are identical",
        set(BRIDGE_CUT_SETS) == set(minimal_cut_sets(diagram)),
    )

    tee.rule("the bridge is not series-parallel")
    tee("")
    tee("  This is why the bridge is the right check. No nesting of series and")
    tee("  parallel groups reproduces it, and the two decompositions a reader")
    tee("  might reach for BRACKET the true answer rather than giving it.")
    tee("")
    tee(
        f"  {'p':>8s}{'two disjoint paths':>22s}{'exact bridge':>22s}"
        f"{'merged a and b':>22s}"
    )
    tee(f"  {'-' * 74}")
    bracketed = True
    for p in (0.1, 0.5, 0.9, 0.99):
        two_paths = 1.0 - (1.0 - p * p) ** 2
        merged = (1.0 - (1.0 - p) ** 2) ** 2
        exact = bridge_reliability_equal(p)
        tee(f"  {p:>8.3f}{two_paths:>22.15f}{exact:>22.15f}{merged:>22.15f}")
        if not two_paths < exact < merged:
            bracketed = False
    tee("")
    tee("  Treating e3 as always failed gives the lower figure; treating it as")
    tee("  always working gives the upper one. The exact answer is strictly")
    tee("  between them at every p in (0, 1), which is the definition of a")
    tee("  structure that is not series-parallel.")
    tee.check(
        "the exact bridge reliability lies strictly between the two series-parallel figures",
        bracketed,
    )

    tee.rule("self-duality")
    tee("")
    tee("  The bridge's minimal cut sets and minimal path sets have the same")
    tee("  shape -- two of size 2 and two of size 3 -- so the structure is")
    tee("  self-dual and R(p) + R(1-p) = 1 exactly. This is a property of the")
    tee("  polynomial that no part of the implementation was written to satisfy,")
    tee("  so it is a check on the polynomial itself.")
    tee("")
    tee(f"  {'p':>8s}{'R(p)':>22s}{'R(1-p)':>22s}{'sum - 1':>14s}")
    tee(f"  {'-' * 66}")
    worst_duality = 0.0
    for p in (0.0, 0.1, 0.25, 0.4, 0.5, 0.73, 0.9, 1.0):
        first = bridge_reliability_equal(p)
        second = bridge_reliability_equal(1.0 - p)
        residual = first + second - 1.0
        worst_duality = max(worst_duality, abs(residual))
        tee(f"  {p:>8.3f}{first:>22.15f}{second:>22.15f}{residual:>14.3e}")
    tee("")
    tee(f"  worst absolute residual            : {worst_duality:.6e}")
    tee.check(
        "the bridge polynomial is self-dual: R(p) + R(1-p) = 1",
        worst_duality <= 1e-15,
        f"worst {worst_duality:.6e} against tolerance 1.0e-15",
    )

    tee.rule("the Esary-Proschan bounds are not tight for the bridge")
    tee("")
    tee("  The bridge's cut sets share components, so the Esary-Proschan bounds")
    tee("  are strict rather than exact. That makes the bracket a real check")
    tee("  here, unlike on a series-parallel structure where it collapses to")
    tee("  zero width and tests only rounding.")
    tee("")
    for p in (0.5, 0.9, 0.99):
        declared = bridge_diagram(p)
        result = quantify(declared, 1.0)
        bracket = esary_proschan_bounds(declared, 1.0, result.reliability)
        tee(f"  p = {p:.2f}: {bracket}")
    tee("")
    widths = []
    inside = True
    for p in (0.5, 0.9, 0.99):
        declared = bridge_diagram(p)
        result = quantify(declared, 1.0)
        bracket = esary_proschan_bounds(declared, 1.0, result.reliability)
        widths.append(bracket.width)
        if not bracket.contains:
            inside = False
    tee(f"  narrowest bracket width            : {min(widths):.6e}")
    tee.check("the Esary-Proschan bounds bracket the exact bridge reliability", inside)
    tee.check(
        "the bracket has non-trivial width, so it is a real check",
        min(widths) > 1e-6,
        f"narrowest {min(widths):.6e}",
    )

    tee.rule("Monte Carlo, budget derived from a target half-width")
    tee("")
    p = 0.9
    declared = bridge_diagram(p)
    exact = bridge_reliability_equal(p)
    q = 1.0 - exact
    # 2 percent of Q rather than the 1 percent used in
    # validate_montecarlo.py: this script is one of seventeen that
    # probe_working_directories.py runs from five working directories, and the
    # measured cost of the tighter budget there was 30 s of that script's
    # 600 s gate allowance. The tight budget lives in validate_montecarlo.py,
    # which runs the same comparison on this diagram among others.
    target = 0.02 * q
    trials = trials_for_half_width(q, target)
    with timer:
        result = simulate(declared, 1.0, trials, seed=SEED)
    tee(f"  edge reliability                   : {p}")
    tee(f"  derived closed form                : {exact:.15f}")
    tee(f"  system unreliability               : {q:.9e}")
    tee(f"  target interval half-width         : {target:.6e} (2 percent of Q)")
    tee(f"  trials from n = z^2 Q(1-Q)/h^2     : {trials}")
    tee(f"  seed                               : {SEED}")
    tee(f"  simulated reliability              : {result.reliability:.12f}")
    tee(f"  Wilson 95 percent interval         : [{result.lower:.12f}, {result.upper:.12f}]")
    tee(f"  achieved half-width                : {result.half_width:.6e}")
    tee(f"  closed form inside the interval    : {result.brackets(exact)}")
    tee(f"  distance in standard errors        : {result.standard_errors_from(exact):.4f}")
    tee.check(
        "the simulated bridge reliability brackets the derived closed form",
        result.brackets(exact),
        f"{result.standard_errors_from(exact):.4f} standard errors",
    )

    tee.rule("non-identical edges")
    tee("")
    tee("  The derived form assumes equal edge reliability. The non-identical")
    tee("  pivotal form and inclusion-exclusion are compared against each other")
    tee("  and against the package, over five distinct edge reliabilities.")
    tee("")
    values = (0.91, 0.87, 0.95, 0.80, 0.99)
    mapping = {f"e{i + 1}": v for i, v in enumerate(values)}
    pivotal = bridge_reliability_pivotal(*values)
    inclusion = bridge_reliability_inclusion_exclusion(mapping)
    declared = bridge_diagram(
        edge_unreliabilities={k: 1.0 - v for k, v in mapping.items()}, mission_hours=1.0
    )
    package = quantify(declared, 1.0).reliability
    unreliabilities = {k: 1.0 - v for k, v in mapping.items()}
    enumerated = 1.0 - boolean_system_unreliability(declared, unreliabilities)
    tee("  edge reliabilities : " + ", ".join(f"{k}={v}" for k, v in mapping.items()))
    tee(f"  pivotal form                       : {pivotal:.15f}")
    tee(f"  inclusion-exclusion                : {inclusion:.15f}")
    tee(f"  package pivotal decomposition      : {package:.15f}")
    tee(f"  32-state enumeration               : {enumerated:.15f}")
    worst_mixed = max(
        _relative(inclusion, pivotal), _relative(package, pivotal), _relative(enumerated, pivotal)
    )
    tee(f"  worst relative difference          : {worst_mixed:.6e}")
    tee.check(
        "all four routes agree for non-identical edges",
        worst_mixed <= TOLERANCE,
        f"worst {worst_mixed:.6e}",
    )

    tee.rule("timing, declared unstable")
    tee("")
    tee("  | quantity | value | stability |")
    tee("  |---|---|---|")
    tee(f"  | whole script wall clock | {timer.elapsed_s:.2f} s | UNSTABLE, shared hardware |")
    tee(f"  | Monte Carlo trials | {trials} | stable, seeded |")

    tee.summarise_failures()
    tee("")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
