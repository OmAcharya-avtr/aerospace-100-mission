"""What computing in reliability space instead of unreliability space costs.

The claim being measured
------------------------

A reliability and an unreliability carry the same information in exact
arithmetic and completely different information in float64. This package
computes every small failure probability in **unreliability space** and derives
the reliability from it, never the other way round. This script measures what
the alternative costs, on each of the three expressions where it matters.

The three expressions
---------------------

**Parallel group.** ``Q = prod Q_i`` is exact, one rounding per factor. The
complementary route ``Q = 1 - prod(1 - Q_i)`` returns exactly ``0.0`` for any
group whose true unreliability falls below the machine epsilon, which for two
blocks means any block unreliability below about ``1.5e-8``. That is not a loss
of digits, it is a loss of the entire answer, and the measurement below shows
where the cliff is.

**Series chain.** ``Q = 1 - prod(1 - Q_i)`` evaluated directly carries an
absolute error of order ``eps`` whatever ``Q`` is, so its relative error is
``eps / Q``. Through hazard space, ``Q = -expm1(sum log1p(-Q_i))``, the relative
error stays at the ``eps`` level. For ten blocks at ``Q_i = 1e-12`` the
difference is about five significant figures, and -- the part that matters --
the naive figure falls **outside** the first- and second-order Bonferroni
bracket, which is an independently derived bound and not a second evaluation of
the same expression. A property test comparing the implementation with the same
expression would have passed.

**Bridge network.** The pivotal decomposition is carried out in unreliability
space, so a bridge of five highly reliable edges keeps its ``Q = 2q^2 + O(q^3)``
down to the subnormal range.

Why a bound and not a second implementation
-------------------------------------------

Two implementations of the same expression agree when the expression is wrong.
The Bonferroni bounds are derived by truncating inclusion-exclusion at
successive orders, both sides are sums of non-negative terms, and neither
inherits the cancellation. That is a different *class* of check and it is the
one that catches catastrophic cancellation.
"""

from __future__ import annotations

import math

from _bootstrap import Tee, Timer, bootstrap, header

bootstrap()

from reliablock.bounds import bonferroni_series_bounds  # noqa: E402
from reliablock.bridge import bridge_diagram  # noqa: E402
from reliablock.diagram import Block, Diagram, KOfN, Parallel  # noqa: E402
from reliablock.distributions import Exponential  # noqa: E402
from reliablock.numerics import (  # noqa: E402
    parallel_unreliability,
    series_unreliability,
    unreliability_from_hazard,
    weighted_binomial_survival,
)
from reliablock.quantify import quantify  # noqa: E402

EPS = 2.220446049250313e-16


def _block_at(name: str, q: float) -> Block:
    """A block whose unreliability at one hour is exactly ``q``."""
    return Block(name=name, distribution=Exponential(-math.log1p(-q)))


def check_parallel_cliff(tee: Tee) -> None:
    tee.rule("parallel: where the complementary route loses the whole answer")
    tee("")
    tee("  Q space : Q = prod Q_i, exact.")
    tee("  R space : Q = 1 - prod(1 - Q_i), which is 1 minus a number that")
    tee("            rounds to 1 as soon as Q falls below eps.")
    tee("")
    tee(
        f"  {'n':>3s}{'q each':>12s}{'Q in Q space':>18s}{'Q in R space':>18s}"
        f"{'rel error':>14s}{'digits lost':>13s}"
    )
    tee(f"  {'-' * 78}")
    total_loss = 0
    for n in (2, 3, 4):
        for q in (1e-2, 1e-4, 1e-6, 1e-8, 1e-9, 1e-12):
            exact = parallel_unreliability([q] * n)
            naive = 1.0 - (1.0 - q**n)
            if exact > 0.0:
                error = abs(naive - exact) / exact
            else:  # pragma: no cover - exact is positive for every q above
                error = 0.0
            digits = "all" if naive == 0.0 else f"{max(0.0, -math.log10(max(error, 1e-18))):.1f}"
            lost = "ALL" if naive == 0.0 else digits
            if naive == 0.0:
                total_loss += 1
            tee(
                f"  {n:>3d}{q:>12.1e}{exact:>18.6e}{naive:>18.6e}"
                f"{error:>14.3e}{lost:>13s}"
            )
    tee("")
    tee(f"  cases where the R-space route returned exactly 0.0 : {total_loss}")
    tee("")
    tee("  The 'digits lost' column is log10 of the relative error, so a value")
    tee("  of 4.0 means four significant figures survive. 'ALL' means the route")
    tee("  returned zero where the answer is positive.")
    tee.check(
        "the Q-space parallel product never returns zero for a positive answer",
        parallel_unreliability([1e-12, 1e-12]) == 1e-24,
        f"Q at q = 1e-12 is {parallel_unreliability([1e-12, 1e-12]):.6e}",
    )
    tee.check(
        "the R-space route does lose the answer, which is why it is not used",
        total_loss > 0,
        f"{total_loss} of 18 cases returned exactly 0.0",
    )

    tee("")
    tee("  the same thing through the package's own quantifier:")
    tee("")
    tee(f"  {'q each':>12s}{'package Q':>18s}{'package R':>22s}{'1 - R':>18s}")
    tee(f"  {'-' * 70}")
    for q in (1e-6, 1e-8, 1e-9, 1e-12):
        diagram = Diagram("pair", Parallel(items=(_block_at("a", q), _block_at("b", q))))
        result = quantify(diagram, 1.0)
        tee(
            f"  {q:>12.1e}{result.unreliability:>18.6e}{result.reliability:>22.17f}"
            f"{1.0 - result.reliability:>18.6e}"
        )
    tee("")
    tee("  The last column is what a package that carried only the reliability")
    tee("  would be able to report. At q = 1e-9 it is exactly zero and the")
    tee("  Q column is 1e-18.")


def check_series_bound(tee: Tee) -> None:
    tee.rule("series: the naive expression falls outside an independently derived bound")
    tee("")
    tee("  Bonferroni orders 1 and 2 bracket the union probability:")
    tee("    sum q_i - sum_{i<j} q_i q_j  <=  Q  <=  sum q_i")
    tee("  Both sides are sums of non-negative terms, so neither inherits the")
    tee("  cancellation in 1 - prod(1 - q_i).")
    tee("")
    tee(
        f"  {'n':>3s}{'q each':>10s}{'hazard-space Q':>20s}{'naive Q':>20s}"
        f"{'rel diff':>12s}{'in bound':>10s}{'naive in':>10s}"
    )
    tee(f"  {'-' * 85}")
    hazard_misses = 0
    naive_misses = 0
    cases = 0
    for n in (2, 5, 10, 50):
        for q in (1e-3, 1e-6, 1e-9, 1e-12, 1e-15):
            hazard = series_unreliability([q] * n)
            naive = 1.0 - (1.0 - q) ** n
            bracket_hazard = bonferroni_series_bounds([q] * n, hazard)
            bracket_naive = bonferroni_series_bounds([q] * n, naive)
            difference = abs(naive - hazard) / hazard if hazard else abs(naive)
            cases += 1
            if not bracket_hazard.contains:
                hazard_misses += 1
            if not bracket_naive.contains:
                naive_misses += 1
            tee(
                f"  {n:>3d}{q:>10.1e}{hazard:>20.12e}{naive:>20.12e}{difference:>12.3e}"
                f"{str(bracket_hazard.contains):>10s}{str(bracket_naive.contains):>10s}"
            )
    tee("")
    tee(f"  cases                              : {cases}")
    tee(f"  hazard-space figures outside bound : {hazard_misses}")
    tee(f"  naive figures outside bound        : {naive_misses}")
    tee.check(
        "the hazard-space series unreliability lies inside the Bonferroni bracket in every case",
        hazard_misses == 0,
        f"{hazard_misses} of {cases} outside",
    )
    tee.check(
        "the naive expression falls outside the bracket, which a same-expression "
        "property test could not detect",
        naive_misses > 0,
        f"{naive_misses} of {cases} outside",
    )

    tee("")
    tee("  the worked case the README quotes, in full:")
    q, n = 1.0e-12, 10
    hazard = series_unreliability([q] * n)
    naive = 1.0 - (1.0 - q) ** n
    bracket = bonferroni_series_bounds([q] * n, naive)
    tee("")
    tee(f"    blocks                           : {n}")
    tee(f"    per-block unreliability          : {q:.1e}")
    tee(f"    Bonferroni lower bound           : {bracket.lower:.15e}")
    tee(f"    Bonferroni upper bound           : {bracket.upper:.15e}")
    tee(f"    bracket width                    : {bracket.width:.6e}")
    tee(f"    bracket slack allowed            : {bracket.slack:.6e}")
    tee(f"    hazard-space Q (this package)    : {hazard:.15e}")
    tee(f"    naive 1 - prod(1 - q_i)          : {naive:.15e}")
    tee(f"    naive relative error             : {abs(naive - hazard) / hazard:.6e}")
    tee(f"    significant figures lost         : {-math.log10(abs(naive - hazard) / hazard):.2f}")
    tee(f"    naive inside the bracket         : {bracket.contains}")


def check_k_of_n_two_spaces(tee: Tee) -> None:
    tee.rule("k-of-n: R and Q from two separate non-negative convolutions")
    tee("")
    tee("  R = P(at least k survive) and Q = P(at least n-k+1 fail) are computed")
    tee("  by two separate recurrences over non-negative terms. Neither is 1")
    tee("  minus the other, so neither inherits the other's cancellation, and")
    tee("  their sum differs from 1 only by accumulated rounding -- which is")
    tee("  reported rather than hidden.")
    tee("")
    tee(
        f"  {'k':>3s}{'n':>3s}{'q each':>10s}{'Q (own space)':>20s}"
        f"{'1 - R':>20s}{'R + Q - 1':>14s}"
    )
    tee(f"  {'-' * 70}")
    worst_residual = 0.0
    for n in (3, 4, 5):
        for k in sorted({2, n - 1, n}):
            for q in (1e-3, 1e-6, 1e-9):
                items = tuple(_block_at(f"u{i}", q) for i in range(n))
                result = quantify(Diagram("knn", KOfN(k=k, items=items)), 1.0)
                naive = 1.0 - result.reliability
                worst_residual = max(worst_residual, abs(result.complement_residual))
                tee(
                    f"  {k:>3d}{n:>3d}{q:>10.1e}{result.unreliability:>20.6e}"
                    f"{naive:>20.6e}{result.complement_residual:>14.3e}"
                )
    tee("")
    tee(f"  worst R + Q - 1                    : {worst_residual:.6e}")
    tee.check(
        "R and Q from their own spaces sum to 1 to within accumulated rounding",
        worst_residual <= 1e-14,
        f"worst {worst_residual:.6e} against tolerance 1.0e-14",
    )

    tee("")
    tee("  the Q-space route against the leading-order expansion:")
    tee("  for a k-of-n group of identical blocks, Q = C(n, n-k+1) q^(n-k+1) + O(q^(n-k+2))")
    tee("")
    tee(
        f"  {'k':>3s}{'n':>3s}{'q each':>10s}{'Q computed':>20s}"
        f"{'leading term':>20s}{'ratio':>12s}"
    )
    tee(f"  {'-' * 68}")
    for n in (3, 4, 5):
        for k in sorted({2, n - 1}):
            q = 1e-9
            needed = n - k + 1
            computed = weighted_binomial_survival([q] * n, needed)
            leading = math.comb(n, needed) * q**needed
            tee(
                f"  {k:>3d}{n:>3d}{q:>10.1e}{computed:>20.6e}{leading:>20.6e}"
                f"{computed / leading:>12.9f}"
            )
    tee("")
    tee("  The ratio approaching 1 from below as q shrinks is the expansion being")
    tee("  correct; a Q computed as 1 - R would be zero in this whole table.")


def check_bridge_precision(tee: Tee) -> None:
    tee.rule("bridge: pivotal decomposition in Q space keeps a tiny answer")
    tee("")
    tee("  Q = 2q^2 + O(q^3) for the five-edge bridge at equal edge")
    tee("  unreliability q. The pivotal decomposition is carried out entirely in")
    tee("  unreliability space, and its terms are non-negative, so there is no")
    tee("  cancellation at any depth of the recursion.")
    tee("")
    tee(
        f"  {'q each':>10s}{'Q computed':>20s}{'2 q^2':>20s}{'ratio':>14s}"
        f"{'R':>22s}{'1 - R':>14s}"
    )
    tee(f"  {'-' * 100}")
    for q in (1e-3, 1e-5, 1e-7, 1e-9, 1e-11):
        diagram = bridge_diagram(
            edge_unreliabilities={f"e{i}": q for i in range(1, 6)}, mission_hours=1.0
        )
        result = quantify(diagram, 1.0)
        leading = 2.0 * q * q
        tee(
            f"  {q:>10.1e}{result.unreliability:>20.6e}{leading:>20.6e}"
            f"{result.unreliability / leading:>14.9f}{result.reliability:>22.17f}"
            f"{1.0 - result.reliability:>14.3e}"
        )
    tee("")
    tee("  The last two columns are the point: from q = 1e-9 down, the")
    tee("  reliability is exactly 1.0 and the only route to the answer is the")
    tee("  unreliability column.")
    q = 1e-11
    diagram = bridge_diagram(
        edge_unreliabilities={f"e{i}": q for i in range(1, 6)}, mission_hours=1.0
    )
    result = quantify(diagram, 1.0)
    tee.check(
        "the bridge keeps a Q of order 1e-22 while its R is exactly 1.0",
        result.unreliability > 0.0 and result.reliability == 1.0,
        f"Q = {result.unreliability:.6e}, R = {result.reliability!r}",
    )


def check_hazard_round_trip(tee: Tee) -> None:
    tee.rule("the hazard round trip in both directions")
    tee("")
    tee("  H -> Q -> H keeps full relative precision for any H. H -> R -> H")
    tee("  loses everything below eps, because R is exactly 1.0 there.")
    tee("")
    tee(
        f"  {'H':>12s}{'Q = -expm1(-H)':>20s}{'H from Q':>20s}{'rel':>12s}"
        f"{'R = exp(-H)':>22s}{'H from R':>12s}"
    )
    tee(f"  {'-' * 98}")
    worst = 0.0
    r_space_losses = 0
    for exponent in range(-18, 1, 3):
        h = 10.0**exponent
        q = unreliability_from_hazard(h)
        back_q = -math.log1p(-q)
        r = math.exp(-h)
        back_r = -math.log(r) if r < 1.0 else 0.0
        error = abs(back_q - h) / h
        worst = max(worst, error)
        if back_r == 0.0:
            r_space_losses += 1
        tee(f"  {h:>12.1e}{q:>20.6e}{back_q:>20.6e}{error:>12.3e}{r:>22.17f}{back_r:>12.3e}")
    tee("")
    tee(f"  worst relative round-trip error via Q : {worst:.6e}")
    tee(f"  hazards lost entirely via R           : {r_space_losses}")
    tee.check(
        "the H -> Q -> H round trip keeps full relative precision",
        worst <= 1e-14,
        f"worst {worst:.6e}",
    )
    tee.check(
        "the H -> R -> H round trip does lose small hazards, which is why Q is the carrier",
        r_space_losses > 0,
        f"{r_space_losses} of 7 hazards returned exactly 0.0",
    )


def main() -> int:
    tee = Tee("validate_precision_output.txt")
    header(tee, "what reliability space costs, measured")
    tee("")
    tee(f"  machine epsilon                    : {EPS:.6e}")
    tee("")
    tee("  Every figure below is a measurement of the package's own arithmetic")
    tee("  against an alternative route, or against an independently derived")
    tee("  bound. The point of the bound checks is that two implementations of")
    tee("  one expression agree when the expression is wrong.")

    timer = Timer()
    with timer:
        check_parallel_cliff(tee)
        check_series_bound(tee)
        check_k_of_n_two_spaces(tee)
        check_bridge_precision(tee)
        check_hazard_round_trip(tee)

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
