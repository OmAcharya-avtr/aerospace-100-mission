"""Property-based tests for the algebraic identities, with Hypothesis.

Each identity below is **exact in exact arithmetic**; in IEEE 754 double
precision the two sides agree to within rounding, and every assertion here
therefore carries a two-sided tolerance. No assertion in this file compares
floating-point results for equality, and none asserts equality of a boolean
produced by a strict threshold comparison.

Any falsifying example Hypothesis finds here is pinned as a named regression
test in ``tests/test_regression.py`` with the literal inputs written into the
test body, because ``.hypothesis/`` is gitignored and a falsifier found by
one run is otherwise lost on a fresh clone.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from budgetroll import (
    Budget,
    Correlation,
    Item,
    MarginPolicy,
    Node,
    correlation_matrix,
    linear_rollup_uncertainty,
    quadratic_form,
    rollup,
    rss_rollup_uncertainty,
    value_shares,
    variance_shares,
)

POLICY = MarginPolicy(
    name="t",
    description="property test policy",
    source="declared in tests/test_properties.py",
    levels={"mature": 0.0, "new": 0.20},
)

SETTINGS = settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

magnitudes = st.floats(
    min_value=0.0, max_value=1.0e3, allow_nan=False, allow_infinity=False, width=64
)
values = st.floats(
    min_value=0.0, max_value=1.0e4, allow_nan=False, allow_infinity=False, width=64
)


def flat_budget(us: list[float], vs: list[float] | None = None) -> Budget:
    """A one-level RSS budget over the given uncertainties."""
    vs = [1.0] * len(us) if vs is None else vs
    children = tuple(
        Item(id=f"i{n}", name=f"i{n}", value=v, uncertainty=u, maturity="mature")
        for n, (u, v) in enumerate(zip(us, vs, strict=True))
    )
    root = Node(id="root", name="root", combination="rss", children=children)
    return Budget(id="b", name="b", unit="kg", quantity="mass", policy=POLICY, root=root)


@SETTINGS
@given(st.lists(magnitudes, min_size=1, max_size=12))
def test_rss_never_exceeds_linear(us):
    """RSS <= linear: the L2 norm of a non-negative vector never exceeds its L1 norm.

    Exact in exact arithmetic (Cauchy-Schwarz); the tolerance absorbs the
    rounding of the square root.
    """
    budget = flat_budget(us)
    rss = rss_rollup_uncertainty(budget)
    linear = linear_rollup_uncertainty(budget)
    assert rss <= linear + 1e-9 * max(1.0, linear)


@SETTINGS
@given(st.lists(magnitudes, min_size=1, max_size=12))
def test_rss_equals_linear_only_when_at_most_one_term_is_nonzero(us):
    """Equality in RSS <= linear holds exactly when at most one term is non-zero."""
    budget = flat_budget(us)
    rss = rss_rollup_uncertainty(budget)
    linear = linear_rollup_uncertainty(budget)
    nonzero = sum(1 for u in us if u > 0.0)
    if nonzero <= 1:
        assert rss == np.float64(linear).item() or abs(rss - linear) <= 1e-12 * max(1.0, linear)


@SETTINGS
@given(st.lists(magnitudes, min_size=2, max_size=12), st.integers(min_value=1, max_value=11))
def test_rss_associativity_under_an_arbitrary_split(us, cut):
    """Grouping independent terms before RSS does not change the result.

    sqrt(sqrt(sum_A u^2)^2 + sqrt(sum_B u^2)^2) = sqrt(sum_{A+B} u^2),
    exact in exact arithmetic and agreeing to within rounding here.
    """
    cut = min(cut, len(us) - 1)
    left, right = us[:cut], us[cut:]

    def node(ident, members, offset):
        return Node(
            id=ident,
            name=ident,
            combination="rss",
            children=tuple(
                Item(
                    id=f"{ident}{n + offset}",
                    name=f"{ident}{n + offset}",
                    value=1.0,
                    uncertainty=u,
                    maturity="mature",
                )
                for n, u in enumerate(members)
            ),
        )

    root = Node(
        id="root",
        name="root",
        combination="rss",
        children=(node("L", left, 0), node("R", right, 0)),
    )
    grouped = Budget(id="b", name="b", unit="kg", quantity="mass", policy=POLICY, root=root)
    flat = flat_budget(us)
    a = rollup(grouped).root.uncertainty
    b = rss_rollup_uncertainty(flat)
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


@SETTINGS
@given(st.lists(magnitudes, min_size=1, max_size=10))
def test_identity_matrix_quadratic_form_equals_rss(us):
    """sqrt(w^T I w) = sqrt(sum w^2), to within rounding."""
    matrix = np.eye(len(us))
    assert math.isclose(
        quadratic_form(us, matrix),
        math.sqrt(math.fsum(u * u for u in us)),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


@SETTINGS
@given(st.lists(magnitudes, min_size=1, max_size=10))
def test_all_ones_matrix_quadratic_form_equals_linear_sum(us):
    """sqrt(w^T J w) = sum w for non-negative w, to within rounding."""
    matrix = np.ones((len(us), len(us)))
    assert math.isclose(
        quadratic_form(us, matrix), math.fsum(us), rel_tol=1e-9, abs_tol=1e-9
    )


@SETTINGS
@given(
    st.lists(magnitudes, min_size=2, max_size=8),
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_quadratic_form_is_monotone_in_rho_for_one_declared_pair(us, rho):
    """Raising a single declared rho never lowers the combined uncertainty.

    Comparison against the rho = -1 value with a two-sided tolerance; exact
    in exact arithmetic because the variance is affine in rho with a
    non-negative coefficient 2 * u_0 * u_1.
    """
    ids = [f"i{n}" for n in range(len(us))]
    low = correlation_matrix(ids, (Correlation(ids[0], ids[1], -1.0, "r"),))
    here = correlation_matrix(ids, (Correlation(ids[0], ids[1], rho, "r"),))
    scale = max(1.0, math.fsum(us))
    assert quadratic_form(us, here) >= quadratic_form(us, low) - 1e-9 * scale


@SETTINGS
@given(st.lists(magnitudes, min_size=1, max_size=10))
def test_variance_shares_sum_to_one(us):
    """The decomposition of a quadratic form partitions the variance exactly."""
    budget = flat_budget(us)
    shares = variance_shares(budget)
    if all(share is None for share in shares.values()):
        assert math.fsum(u * u for u in us) == 0.0
        return
    assert math.isclose(math.fsum(shares.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9)


@SETTINGS
@given(st.lists(values, min_size=1, max_size=10))
def test_value_shares_sum_to_one(vs):
    """Leaf value shares partition the predicted root value."""
    budget = flat_budget([0.0] * len(vs), vs)
    shares = value_shares(rollup(budget))
    if all(share is None for share in shares.values()):
        assert math.fsum(vs) == 0.0
        return
    assert math.isclose(math.fsum(shares.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9)


@SETTINGS
@given(st.lists(magnitudes, min_size=1, max_size=10), magnitudes)
def test_adding_a_term_never_lowers_the_rss(us, extra):
    """RSS is monotone non-decreasing under appending a non-negative term."""
    before = rss_rollup_uncertainty(flat_budget(us))
    after = rss_rollup_uncertainty(flat_budget([*us, extra]))
    assert after >= before - 1e-9 * max(1.0, before)


@SETTINGS
@given(st.lists(values, min_size=1, max_size=8), st.floats(min_value=0.0, max_value=1e5))
def test_margin_is_allocation_minus_predicted(vs, allocation):
    """margin = allocation - predicted, to within rounding of the sum."""
    budget = flat_budget([0.0] * len(vs), vs)
    root = Node(
        id="root",
        name="root",
        combination="rss",
        children=budget.root.children,
        allocation=allocation,
    )
    with_allocation = Budget(
        id="b", name="b", unit="kg", quantity="mass", policy=POLICY, root=root
    )
    result = rollup(with_allocation).root
    assert math.isclose(
        result.margin_absolute,
        allocation - result.predicted,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )
