"""Every defect Hypothesis found, pinned with its literal falsifying inputs.

``.hypothesis/`` is gitignored, so a counterexample found by one run is lost on
a fresh clone. Each of these is a defect that existed in this package, was
found by a property test and not by the hand-written suite, and is now fixed.
The literal inputs are in the test body so that the regression travels with the
repository.

Each test names which property test found it and what the symptom was.
"""

from __future__ import annotations

import math

import pytest

from reliablock.bounds import bonferroni_series_bounds, esary_proschan_bounds
from reliablock.diagram import Diagram, KOfN, Parallel, Series
from reliablock.distributions import Exponential
from reliablock.numerics import hazard_from_unreliability, weighted_binomial_survival
from reliablock.quantify import quantify, shared_load_reliability

from conftest import abs_, exponential_block, rel


def test_poisson_binomial_sum_clamped_to_one() -> None:
    """Found by ``test_k_equals_n_is_series``.

    The Poisson-binomial tail is a sum of non-negative float64 terms that add
    to 1 in exact arithmetic, so it can land a few ulps above 1. Unclamped it
    returned ``1.0000000000000002``, and ``hazard_from_unreliability`` then
    raised ``ValueError: q must lie in [0, 1]`` -- a crash, on a valid
    five-block k-of-n diagram at ``k = n``, not a rounding question.
    """
    rates = [0.0078125, 5e-05, 5e-05, 1.0000000000000003e-09, 1e-09]
    hours = 229415.5
    unreliabilities = [Exponential(r).unreliability(hours) for r in rates]
    survival = weighted_binomial_survival(unreliabilities, len(rates))
    assert survival <= 1.0
    # And the diagram it came from now quantifies instead of raising.
    items = tuple(exponential_block(f"u{i}", r) for i, r in enumerate(rates))
    result = quantify(Diagram("k", KOfN(k=len(rates), items=items)), hours)
    assert 0.0 <= result.unreliability <= 1.0
    # Q is exactly 1.0 here -- the first block's hazard over 229415 hours is
    # 1792, so the chain is certainly failed -- and an infinite cumulative
    # hazard is the correct answer for that, which numerics accepts.
    assert hazard_from_unreliability(result.unreliability) == math.inf


def test_esary_proschan_slack_follows_conditioning_not_operation_count() -> None:
    """Found by ``test_esary_proschan_bounds_bracket_a_nested_structure``.

    For a series-parallel structure no two cut sets share a component, so the
    Esary-Proschan bounds are exact and the bracket has zero width in exact
    arithmetic. The min-cut bound evaluates ``1 - prod q_i`` with every ``q_i``
    close to 1, whose absolute error is of order ``eps`` whatever the result,
    so its relative error is ``eps / (1 - prod q_i)``. At these inputs that is
    about 9e-13 and the exact reliability sits 2.7e-17 below the lower bound,
    while both a fixed ``8 eps`` slack and an operation-count slack allowed
    only about 3.5e-18.
    """
    rates_list = [0.0078125, 5.960464477539063e-08]
    hours = 762213.0
    inner = Parallel(
        items=tuple(exponential_block(f"u{i}", r) for i, r in enumerate(rates_list)),
        label="inner",
    )
    diagram = Diagram("nested", Series(items=(inner, exponential_block("tail", 1e-5))))
    result = quantify(diagram, hours)
    bracket = esary_proschan_bounds(diagram, hours, result.reliability)
    assert bracket.contains, str(bracket)
    # The slack is small in absolute terms, so the check still has power.
    assert bracket.slack < 1e-14


def test_esary_proschan_slack_at_half_a_million_hours() -> None:
    """The same defect at the first falsifying input Hypothesis reported.

    Pinned separately because the two inputs exercise different cut-set
    conditionings and the first one is the narrower margin.
    """
    rates_list = [0.0078125, 5.960464477539063e-08]
    for hours in (381107.0, 762213.0):
        inner = Parallel(
            items=tuple(exponential_block(f"u{i}", r) for i, r in enumerate(rates_list)),
            label="inner",
        )
        diagram = Diagram("nested", Series(items=(inner, exponential_block("tail", 1e-5))))
        result = quantify(diagram, hours)
        bracket = esary_proschan_bounds(diagram, hours, result.reliability)
        assert bracket.contains, f"{hours} h: {bracket}"


def test_bonferroni_slack_does_not_swamp_a_narrow_bracket() -> None:
    """Found by ``test_the_naive_expression_falls_outside_the_bound``.

    A slack floored at an absolute constant made the Bonferroni check vacuous:
    the bracket for ten blocks at ``q = 1e-12`` has width 4.5e-23 around 1e-11,
    and a slack of ``8 eps * max(scale, 1.0)`` is 1.8e-15 -- forty million
    times the width. Every value in the neighbourhood was reported inside,
    including the naive expression the check exists to catch.
    """
    q = 1.0e-12
    n = 10
    naive = 1.0 - (1.0 - q) ** n
    bracket = bonferroni_series_bounds([q] * n, naive)
    assert bracket.slack < bracket.width
    assert not bracket.contains, str(bracket)


def test_shared_load_is_continuous_through_its_degeneracy() -> None:
    """Found by ``test_degenerate_limit_is_continuous`` in the hand-written suite.

    The first implementation switched to an analytic limit below a declared
    relative distance from a vanishing denominator. The switch was a
    discontinuity, and the threshold would have had to be tuned to hide it. The
    expm1 form is exact at the degeneracy and needs no threshold, so this test
    checks smoothness across it rather than the width of a tolerance.
    """
    rate = 4.0e-4
    hours = 1000.0
    at_zero = shared_load_reliability(rate, rate, 0.5, hours)
    previous = at_zero
    for epsilon in (1e-16, 1e-13, 1e-10, 1e-7, 1e-4):
        value = shared_load_reliability(rate, rate * (1.0 + epsilon), 0.5, hours)
        assert abs(value - previous) < 10.0 * epsilon + 1e-15
        previous = value
    x = rate * hours
    assert at_zero == rel(math.exp(-x) * (1.0 + x), 1e-14)


def test_dangling_network_edge_is_detected() -> None:
    """Found while writing ``test_edge_on_no_path_is_rejected``.

    The first dangling-edge check tested reachability *with* the edge present.
    A reliability block diagram's network is undirected, so in a connected
    graph every vertex is reachable from every other and the test passed for a
    stub whose reliability could never enter the answer. The edge has to be
    removed before the reachability question is asked.
    """
    from reliablock.diagram import Block, Network
    from reliablock.distributions import FixedProbability

    def block(name: str) -> Block:
        return Block(name=name, distribution=FixedProbability(0.1))

    with pytest.raises(ValueError, match="lie on no"):
        Diagram(
            "stub",
            Network(
                edges=(("s", "t", block("e1")), ("s", "x", block("e2"))),
                source="s",
                sink="t",
            ),
        )


def test_poisson_binomial_recurrence_does_not_feed_itself() -> None:
    """Found by ``test_identical_blocks_match_the_binomial_survival``.

    The first recurrence updated the distribution array in place with a numpy
    view, so the row being written was partially read back as the previous row.
    It produced a plausible-looking number that disagreed with
    ``scipy.stats.binom.sf``, which is why the independent reference mattered
    more than the property.
    """
    from scipy.stats import binom

    for n in (2, 3, 5, 8):
        for k in range(1, n + 1):
            assert weighted_binomial_survival([0.9] * n, k) == abs_(
                float(binom.sf(k - 1, n, 0.9)), 1e-13
            )


def test_ages_with_time_is_not_a_positional_constructor_slot() -> None:
    """Found while smoke-testing the distributions.

    ``ages_with_time`` was first declared as a dataclass field on the base
    class, so it occupied the first positional slot in every subclass and
    ``Exponential(1e-5)`` set the flag and left the rate at its default of
    zero -- a block that never fails, reported without complaint. It is a
    ``ClassVar`` now.
    """
    assert Exponential(1.0e-5).rate_per_hour == 1.0e-5
    assert Exponential(1.0e-5).reliability(10_000.0) < 1.0
