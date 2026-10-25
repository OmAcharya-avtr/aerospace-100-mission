"""Closed-form agreement for every structure, each against its own reference.

What is checked, and against what
---------------------------------

=============  ==========================================================
structure      independent reference
=============  ==========================================================
series         the exact hazard sum, and the two Bonferroni orders as an
               independently derived bracket on the unreliability
parallel       the exact product in unreliability space, and exhaustive
               state enumeration over the structure function
k-of-n         ``scipy.stats.binom.sf``, the binomial survival function,
               for identical blocks; exhaustive state enumeration for
               non-identical ones
standby        ``scipy.integrate.quad`` over the convolution integral
               ``R_p(t) + p_s int f_p(u) R_s(f u + t - u) du``
shared load    ``scipy.integrate.solve_ivp`` with DOP853 at rtol 1e-12 on
               the four-state Markov system
network        exhaustive state enumeration, and inclusion-exclusion over
               the minimal path sets
=============  ==========================================================

No reference above shares an expression with the implementation it checks.
``scipy`` is a separate implementation by other people; the enumeration walks
the structure function; the ODE solution never forms the closed form at all.

Tolerances are declared per check in the output and are not adjusted to make
anything pass.
"""

from __future__ import annotations

import math

from _bootstrap import Tee, Timer, bootstrap, header

bootstrap()

import numpy as np  # noqa: E402
from scipy.integrate import quad, solve_ivp  # noqa: E402
from scipy.stats import binom  # noqa: E402

from reliablock.bounds import bonferroni_series_bounds  # noqa: E402
from reliablock.bridge import (  # noqa: E402
    bridge_diagram,
    bridge_reliability_equal,
    bridge_reliability_inclusion_exclusion,
)
from reliablock.diagram import (  # noqa: E402
    Block,
    Diagram,
    KOfN,
    Parallel,
    Series,
    SharedLoad,
    Standby,
)
from reliablock.distributions import Exponential, Weibull  # noqa: E402
from reliablock.importance import boolean_system_unreliability  # noqa: E402
from reliablock.quantify import (  # noqa: E402
    quantify,
    shared_load_reliability,
    standby_reliability_cold_identical,
)

#: Declared tolerances, per structure. Stated here so that they appear once, in
#: the output, and are not quietly different per call site.
TOLERANCES = {
    "series": 1e-14,
    "parallel": 1e-14,
    "k_of_n": 1e-13,
    "standby_erlang": 1e-13,
    "standby_quadrature": 1e-10,
    "shared_load_ode": 1e-9,
    "network": 1e-13,
}

MISSION_HOURS = (10.0, 100.0, 1000.0, 10_000.0, 100_000.0)
RATES = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)


def _block(name: str, rate: float) -> Block:
    return Block(name=name, distribution=Exponential(rate))


def _relative(computed: float, reference: float) -> float:
    """Relative difference, or the absolute one when the reference is zero."""
    if reference == 0.0:
        return abs(computed)
    return abs(computed - reference) / abs(reference)


def check_series(tee: Tee) -> None:
    tee.rule("series: the exact hazard sum, and the Bonferroni bracket")
    tee("")
    tee("  R = prod R_i, so H = sum H_i exactly. The reference for the")
    tee("  reliability is the product of the block reliabilities; the reference")
    tee("  for the UNRELIABILITY is the first- and second-order Bonferroni")
    tee("  bracket, which is an independently derived bound and not a second")
    tee("  evaluation of the same expression.")
    tee("")
    tee(f"  declared tolerance : {TOLERANCES['series']:.1e} relative")
    tee("")
    tee(f"  {'n':>3s}{'rate /h':>12s}{'hours':>12s}{'R computed':>22s}{'rel diff':>12s}")
    tee(f"  {'-' * 61}")
    worst = 0.0
    bracket_failures = 0
    brackets = 0
    for n in (2, 5, 10):
        for rate in RATES:
            for hours in MISSION_HOURS:
                items = tuple(_block(f"b{i}", rate) for i in range(n))
                result = quantify(Diagram("chain", Series(items=items)), hours)
                reference = math.exp(-rate * hours) ** n
                difference = _relative(result.reliability, reference)
                worst = max(worst, difference)
                unreliabilities = [
                    Exponential(rate).unreliability(hours) for _ in range(n)
                ]
                bracket = bonferroni_series_bounds(unreliabilities, result.unreliability)
                brackets += 1
                if not bracket.contains:
                    bracket_failures += 1
                    tee(f"    BRACKET MISS n={n} rate={rate:g} hours={hours:g}: {bracket}")
                if n == 10 and hours == 10_000.0:
                    tee(
                        f"  {n:>3d}{rate:>12.3e}{hours:>12.4g}"
                        f"{result.reliability:>22.15f}{difference:>12.3e}"
                    )
    tee("")
    tee(f"  cases                              : {brackets}")
    tee(f"  worst relative difference in R     : {worst:.6e}")
    tee(f"  Bonferroni bracket misses          : {bracket_failures}")
    tee.check(
        "series reliability matches the product of block reliabilities",
        worst <= TOLERANCES["series"],
        f"worst {worst:.6e} against tolerance {TOLERANCES['series']:.1e}",
    )
    tee.check(
        "series unreliability lies inside the Bonferroni bracket in every case",
        bracket_failures == 0,
        f"{bracket_failures} of {brackets} outside",
    )


def check_parallel(tee: Tee) -> None:
    tee.rule("parallel: the exact product in Q space, and state enumeration")
    tee("")
    tee(f"  declared tolerance : {TOLERANCES['parallel']:.1e} relative")
    tee("")
    tee(f"  {'n':>3s}{'q each':>14s}{'Q computed':>18s}{'Q reference':>18s}{'rel diff':>12s}")
    tee(f"  {'-' * 65}")
    worst = 0.0
    for n in (2, 3, 4):
        for q in (1e-9, 1e-6, 1e-3, 1e-1, 0.5):
            rate = -math.log1p(-q)
            items = tuple(_block(f"b{i}", rate) for i in range(n))
            result = quantify(Diagram("group", Parallel(items=items)), 1.0)
            reference = q**n
            difference = _relative(result.unreliability, reference)
            worst = max(worst, difference)
            if n == 2:
                tee(
                    f"  {n:>3d}{q:>14.3e}{result.unreliability:>18.6e}"
                    f"{reference:>18.6e}{difference:>12.3e}"
                )
    tee("")
    tee(f"  worst relative difference in Q     : {worst:.6e}")
    tee.check(
        "parallel unreliability is the exact product of the block unreliabilities",
        worst <= TOLERANCES["parallel"],
        f"worst {worst:.6e} against tolerance {TOLERANCES['parallel']:.1e}",
    )

    tee("")
    tee("  against exhaustive state enumeration over the structure function:")
    worst_enumeration = 0.0
    for n in (2, 3, 4):
        items = tuple(_block(f"b{i}", 1e-4) for i in range(n))
        diagram = Diagram("group", Parallel(items=items))
        unreliabilities = {
            b.name: b.distribution.unreliability(1000.0) for b in diagram.blocks()
        }
        structural = quantify(diagram, 1000.0).unreliability
        exhaustive = boolean_system_unreliability(diagram, unreliabilities)
        worst_enumeration = max(worst_enumeration, _relative(structural, exhaustive))
    tee(f"  worst relative difference          : {worst_enumeration:.6e}")
    tee.check(
        "parallel quantification matches exhaustive state enumeration",
        worst_enumeration <= TOLERANCES["parallel"],
        f"worst {worst_enumeration:.6e}",
    )


def check_k_of_n(tee: Tee) -> None:
    tee.rule("k-of-n: against the binomial survival sum from scipy.stats")
    tee("")
    tee("  For identical blocks, R = sum_{i=k}^{n} C(n,i) p^i (1-p)^(n-i), which")
    tee("  is scipy.stats.binom.sf(k-1, n, p) -- a separate implementation by")
    tee("  other people, not a second evaluation of the same expression.")
    tee("")
    tee(f"  declared tolerance : {TOLERANCES['k_of_n']:.1e} relative")
    tee("")
    tee(f"  {'k':>3s}{'n':>3s}{'p':>14s}{'R computed':>20s}{'binom.sf':>20s}{'rel diff':>12s}")
    tee(f"  {'-' * 72}")
    worst = 0.0
    cases = 0
    for n in (2, 3, 4, 5, 6):
        for k in range(1, n + 1):
            for rate in RATES:
                for hours in (100.0, 1000.0, 10_000.0):
                    p = math.exp(-rate * hours)
                    items = tuple(_block(f"b{i}", rate) for i in range(n))
                    result = quantify(Diagram("knn", KOfN(k=k, items=items)), hours)
                    reference = float(binom.sf(k - 1, n, p))
                    difference = _relative(result.reliability, reference)
                    worst = max(worst, difference)
                    cases += 1
                    if n == 4 and k == 3 and hours == 1000.0 and rate == 1e-4:
                        tee(
                            f"  {k:>3d}{n:>3d}{p:>14.9f}{result.reliability:>20.15f}"
                            f"{reference:>20.15f}{difference:>12.3e}"
                        )
    tee("")
    tee(f"  cases                              : {cases}")
    tee(f"  worst relative difference          : {worst:.6e}")
    tee.check(
        "k-of-n reliability matches the binomial survival sum",
        worst <= TOLERANCES["k_of_n"],
        f"worst {worst:.6e} against tolerance {TOLERANCES['k_of_n']:.1e}",
    )

    tee("")
    tee("  k = n reduces to series and k = 1 to parallel, exactly:")
    worst_series = 0.0
    worst_parallel = 0.0
    for n in (2, 3, 4, 5):
        rate, hours = 2e-4, 1000.0
        items = tuple(_block(f"b{i}", rate) for i in range(n))
        knn_n = quantify(Diagram("a", KOfN(k=n, items=items)), hours).reliability
        chain = quantify(Diagram("b", Series(items=items)), hours).reliability
        knn_1 = quantify(Diagram("c", KOfN(k=1, items=items)), hours).reliability
        group = quantify(Diagram("d", Parallel(items=items)), hours).reliability
        worst_series = max(worst_series, _relative(knn_n, chain))
        worst_parallel = max(worst_parallel, _relative(knn_1, group))
    tee(f"  worst k=n versus series            : {worst_series:.6e}")
    tee(f"  worst k=1 versus parallel          : {worst_parallel:.6e}")
    tee.check(
        "k-of-n at k = n equals the series chain",
        worst_series <= TOLERANCES["k_of_n"],
        f"worst {worst_series:.6e}",
    )
    tee.check(
        "k-of-n at k = 1 equals the parallel group",
        worst_parallel <= TOLERANCES["k_of_n"],
        f"worst {worst_parallel:.6e}",
    )

    tee("")
    tee("  non-identical blocks against exhaustive state enumeration:")
    rates = (1e-4, 3e-4, 7e-4, 2e-4, 5e-4)
    worst_mixed = 0.0
    for n in (2, 3, 4, 5):
        for k in range(1, n + 1):
            items = tuple(_block(f"b{i}", rates[i]) for i in range(n))
            diagram = Diagram("mixed", KOfN(k=k, items=items))
            unreliabilities = {
                b.name: b.distribution.unreliability(1000.0) for b in diagram.blocks()
            }
            structural = quantify(diagram, 1000.0).unreliability
            exhaustive = boolean_system_unreliability(diagram, unreliabilities)
            worst_mixed = max(worst_mixed, _relative(structural, exhaustive))
    tee(f"  worst relative difference          : {worst_mixed:.6e}")
    tee.check(
        "non-identical k-of-n matches exhaustive state enumeration",
        worst_mixed <= TOLERANCES["k_of_n"],
        f"worst {worst_mixed:.6e}",
    )


def check_standby(tee: Tee) -> None:
    tee.rule("standby: against the convolution integral from scipy.integrate.quad")
    tee("")
    tee("  R(t) = R_p(t) + p_s int_0^t f_p(u) exp(-H_s(f u + t - u)) du")
    tee("")
    tee("  The Erlang closed form for identical cold standby units is scored")
    tee("  against this integral, evaluated by quad. The quadrature's own")
    tee("  reported absolute error bound is in the table, because an agreement")
    tee("  tighter than the reference's own error is not a measurement.")
    tee("")
    tee(f"  declared tolerance : {TOLERANCES['standby_quadrature']:.1e} relative")
    tee("")
    tee(
        f"  {'rate /h':>10s}{'hours':>10s}{'p_s':>7s}{'f':>6s}"
        f"{'closed form':>20s}{'quad':>20s}{'rel diff':>11s}{'quad err':>11s}"
    )
    tee(f"  {'-' * 95}")
    worst = 0.0
    cases = 0
    for rate in (1e-5, 1e-4, 1e-3):
        for hours in (100.0, 1000.0, 10_000.0):
            for switch in (1.0, 0.98, 0.5):
                for fraction in (0.0, 0.35, 1.0):
                    primary = Exponential(rate)
                    diagram = Diagram(
                        "sb",
                        Standby(
                            primary=_block("p", rate),
                            standbys=(_block("s", rate),),
                            switch_reliability=switch,
                            standby_hazard_fraction=fraction,
                            label="sb",
                        ),
                    )
                    closed = quantify(diagram, hours).reliability

                    def integrand(
                        u: float, _r: float = rate, _h: float = hours, _f: float = fraction
                    ) -> float:
                        age = _f * u + (_h - u)
                        return Exponential(_r).density(u) * Exponential(_r).reliability(age)

                    value, error = quad(integrand, 0.0, hours, limit=200)
                    reference = primary.reliability(hours) + switch * value
                    difference = _relative(closed, reference)
                    worst = max(worst, difference)
                    cases += 1
                    if rate == 1e-4 and hours == 1000.0 and switch == 0.98:
                        tee(
                            f"  {rate:>10.1e}{hours:>10.4g}{switch:>7.2f}{fraction:>6.2f}"
                            f"{closed:>20.15f}{reference:>20.15f}"
                            f"{difference:>11.3e}{switch * error:>11.3e}"
                        )
    tee("")
    tee(f"  cases                              : {cases}")
    tee(f"  worst relative difference          : {worst:.6e}")
    tee.check(
        "standby reliability matches the convolution integral",
        worst <= TOLERANCES["standby_quadrature"],
        f"worst {worst:.6e} against tolerance {TOLERANCES['standby_quadrature']:.1e}",
    )

    tee("")
    tee("  the Erlang form for m cold standby units against the Erlang series:")
    tee("")
    tee(f"  {'m':>3s}{'x = rate t':>14s}{'closed form':>20s}{'Erlang sum':>20s}{'rel diff':>12s}")
    tee(f"  {'-' * 69}")
    worst_erlang = 0.0
    for m in (1, 2, 3, 4):
        for x in (0.01, 0.1, 1.0, 5.0):
            closed = standby_reliability_cold_identical(x, 1.0, m, 1.0)
            reference = math.exp(-x) * sum(x**j / math.factorial(j) for j in range(m + 1))
            difference = _relative(closed, reference)
            worst_erlang = max(worst_erlang, difference)
            if x == 1.0:
                tee(f"  {m:>3d}{x:>14.4g}{closed:>20.15f}{reference:>20.15f}{difference:>12.3e}")
    tee("")
    tee(f"  worst relative difference          : {worst_erlang:.6e}")
    tee.check(
        "the m-standby closed form is the Erlang survival function",
        worst_erlang <= TOLERANCES["standby_erlang"],
        f"worst {worst_erlang:.6e} against tolerance {TOLERANCES['standby_erlang']:.1e}",
    )

    tee("")
    tee("  a Weibull primary and standby, closed form against quad:")
    shape, scale = 1.8, 4000.0
    worst_weibull = 0.0
    for hours in (500.0, 1000.0, 5000.0):
        diagram = Diagram(
            "weibull sb",
            Standby(
                primary=Block(name="p", distribution=Weibull(shape=shape, scale_hours=scale)),
                standbys=(
                    Block(name="s", distribution=Weibull(shape=shape, scale_hours=scale)),
                ),
                switch_reliability=0.99,
                label="sb",
            ),
        )
        closed = quantify(diagram, hours).reliability
        base = Weibull(shape=shape, scale_hours=scale)

        def integrand(u: float, _h: float = hours, _d: Weibull = base) -> float:
            return _d.density(u) * _d.reliability(_h - u)

        value, _error = quad(integrand, 0.0, hours, limit=200)
        reference = base.reliability(hours) + 0.99 * value
        worst_weibull = max(worst_weibull, _relative(closed, reference))
    tee(f"  worst relative difference          : {worst_weibull:.6e}")
    tee.check(
        "a Weibull standby pair matches the convolution integral",
        worst_weibull <= TOLERANCES["standby_quadrature"],
        f"worst {worst_weibull:.6e}",
    )


def check_shared_load(tee: Tee) -> None:
    tee.rule("shared load: against the four-state Markov ODE from scipy solve_ivp")
    tee("")
    tee("  The reference integrates the Markov system numerically with DOP853 at")
    tee("  rtol 1e-12 and never forms the closed form at all, so the two share")
    tee("  nothing but the rate matrix.")
    tee("")
    tee(f"  declared tolerance : {TOLERANCES['shared_load_ode']:.1e} relative")
    tee("")
    tee(
        f"  {'l1 /h':>10s}{'l2 /h':>10s}{'s':>6s}{'hours':>10s}"
        f"{'closed form':>20s}{'ODE':>20s}{'rel diff':>12s}"
    )
    tee(f"  {'-' * 88}")
    worst = 0.0
    cases = 0
    for l1 in (1e-5, 1e-4, 5e-4):
        for l2 in (1e-5, 1e-4, 5e-4):
            for s in (0.25, 0.5, 0.75, 1.0):
                for hours in (100.0, 1000.0, 5000.0):
                    closed = shared_load_reliability(l1, l2, s, hours)

                    def rhs(
                        _t: float,
                        y: np.ndarray,
                        a: float = l1,
                        b: float = l2,
                        factor: float = s,
                    ) -> list[float]:
                        both, only2, only1 = y
                        return [
                            -factor * (a + b) * both,
                            factor * a * both - b * only2,
                            factor * b * both - a * only1,
                        ]

                    solution = solve_ivp(
                        rhs,
                        (0.0, hours),
                        [1.0, 0.0, 0.0],
                        method="DOP853",
                        rtol=1e-12,
                        atol=1e-15,
                    )
                    reference = float(solution.y[:, -1].sum())
                    difference = _relative(closed, reference)
                    worst = max(worst, difference)
                    cases += 1
                    if l1 == 1e-4 and l2 == 5e-4 and s == 0.5 and hours == 1000.0:
                        tee(
                            f"  {l1:>10.1e}{l2:>10.1e}{s:>6.2f}{hours:>10.4g}"
                            f"{closed:>20.15f}{reference:>20.15f}{difference:>12.3e}"
                        )
    tee("")
    tee(f"  cases                              : {cases}")
    tee(f"  worst relative difference          : {worst:.6e}")
    tee.check(
        "shared-load reliability matches the Markov ODE solution",
        worst <= TOLERANCES["shared_load_ode"],
        f"worst {worst:.6e} against tolerance {TOLERANCES['shared_load_ode']:.1e}",
    )

    tee("")
    tee("  the exactly-degenerate case, where both denominators vanish:")
    tee("  s = 0.5 with l1 = l2 makes s(l1 + l2) = l2 exactly. The expm1 form")
    tee("  needs no threshold there; the limit is 1 + x times exp(-x).")
    worst_degenerate = 0.0
    for rate in (1e-5, 1e-4, 1e-3):
        for hours in (100.0, 1000.0, 10_000.0):
            closed = shared_load_reliability(rate, rate, 0.5, hours)
            x = rate * hours
            reference = math.exp(-x) * (1.0 + x)
            worst_degenerate = max(worst_degenerate, _relative(closed, reference))
    tee(f"  worst relative difference          : {worst_degenerate:.6e}")
    tee.check(
        "the degenerate shared-load case equals its analytic limit",
        worst_degenerate <= 1e-14,
        f"worst {worst_degenerate:.6e} against tolerance 1.0e-14",
    )

    tee("")
    tee("  sharing factor 1 makes the node a parallel pair:")
    worst_parallel = 0.0
    for rate in (1e-5, 1e-4):
        for hours in (100.0, 1000.0):
            diagram = Diagram(
                "no benefit",
                SharedLoad(
                    units=(_block("a", rate), _block("b", rate)), shared_rate_factor=1.0
                ),
            )
            closed = quantify(diagram, hours).reliability
            q = 1.0 - math.exp(-rate * hours)
            worst_parallel = max(worst_parallel, _relative(closed, 1.0 - q * q))
    tee(f"  worst relative difference          : {worst_parallel:.6e}")
    tee.check(
        "shared load at a sharing factor of 1 equals a parallel pair",
        worst_parallel <= 1e-12,
        f"worst {worst_parallel:.6e}",
    )


def check_network(tee: Tee) -> None:
    tee.rule("network: pivotal decomposition against enumeration and inclusion-exclusion")
    tee("")
    tee(f"  declared tolerance : {TOLERANCES['network']:.1e} relative")
    tee("")
    tee(
        f"  {'p':>8s}{'pivotal (package)':>24s}{'derived closed form':>24s}"
        f"{'incl-excl':>24s}{'rel diff':>12s}"
    )
    tee(f"  {'-' * 92}")
    worst = 0.0
    for p in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999):
        diagram = bridge_diagram(p)
        package = quantify(diagram, 1.0).reliability
        derived = bridge_reliability_equal(p)
        inclusion = bridge_reliability_inclusion_exclusion({f"e{i}": p for i in range(1, 6)})
        difference = max(_relative(package, derived), _relative(inclusion, derived))
        worst = max(worst, difference)
        tee(f"  {p:>8.3f}{package:>24.15f}{derived:>24.15f}{inclusion:>24.15f}{difference:>12.3e}")
    tee("")
    tee(f"  worst relative difference          : {worst:.6e}")
    tee.check(
        "network pivotal decomposition matches the derived closed form",
        worst <= TOLERANCES["network"],
        f"worst {worst:.6e} against tolerance {TOLERANCES['network']:.1e}",
    )

    tee("")
    tee("  against exhaustive state enumeration over all 32 states:")
    worst_enumeration = 0.0
    for p in (0.1, 0.5, 0.9, 0.99):
        diagram = bridge_diagram(p)
        unreliabilities = {f"e{i}": 1.0 - p for i in range(1, 6)}
        structural = quantify(diagram, 1.0).unreliability
        exhaustive = boolean_system_unreliability(diagram, unreliabilities)
        worst_enumeration = max(worst_enumeration, _relative(structural, exhaustive))
    tee(f"  worst relative difference          : {worst_enumeration:.6e}")
    tee.check(
        "network quantification matches exhaustive state enumeration",
        worst_enumeration <= TOLERANCES["network"],
        f"worst {worst_enumeration:.6e}",
    )


def main() -> int:
    tee = Tee("validate_structures_output.txt")
    header(tee, "closed-form agreement for every structure")
    tee("")
    tee("  Every reference below is an independent implementation, an exhaustive")
    tee("  enumeration, or an independently derived bound. None of them is a")
    tee("  second evaluation of the expression under test.")
    tee("")
    tee("  declared tolerances:")
    for name, tolerance in TOLERANCES.items():
        tee(f"    {name:<22s}{tolerance:.1e} relative")

    timer = Timer()
    with timer:
        check_series(tee)
        check_parallel(tee)
        check_k_of_n(tee)
        check_standby(tee)
        check_shared_load(tee)
        check_network(tee)

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
