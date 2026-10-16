"""Pinned regression tests.

Every case in this file is a concrete input that was found by a search over
the implementation rather than reasoned about in advance, with the literal
values written into the test body. ``.hypothesis/`` is gitignored, so a
falsifying example found by one run is lost on a fresh clone unless it is
pinned here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from budgetroll import (
    Budget,
    Item,
    MarginPolicy,
    Node,
    quadratic_form,
    rollup,
    rss_rollup_uncertainty,
    run_checks,
    variance_shares,
)

POLICY = MarginPolicy(
    name="t",
    description="regression policy",
    source="declared in tests/test_regression.py",
    levels={"mature": 0.0},
)

#: The smallest 1 sigma value whose square is still a NORMAL double. Below
#: this the square is subnormal and loses significant bits.
SUBNORMAL_SQUARE_THRESHOLD = math.sqrt(2.2250738585072014e-308)

#: The value below which the square underflows to exactly zero, so the term
#: vanishes from every root-sum-square and every quadratic form.
TOTAL_UNDERFLOW_THRESHOLD = math.sqrt(5e-324)


def flat(us: list[float], values: list[float] | None = None, allocation: float | None = None):
    values = [1.0] * len(us) if values is None else values
    children = tuple(
        Item(id=f"i{n}", name=f"i{n}", value=v, uncertainty=u, maturity="mature")
        for n, (u, v) in enumerate(zip(us, values, strict=True))
    )
    root = Node(
        id="root", name="root", combination="rss", children=children, allocation=allocation
    )
    return Budget(id="b", name="B", unit="kg", quantity="mass", policy=POLICY, root=root)


def test_regression_quadratic_form_underflows_at_subnormal_weights():
    """Found by a 40000-example search over the all-ones identity.

    With four weights at exactly DBL_MIN = 2.2250738585072014e-308, the
    squares underflow to zero and ``sqrt(w^T J w)`` returns 0.0, while the
    arithmetic sum is 8.900295434028806e-308. The relative disagreement is
    1.0, a complete failure of the identity, and no tolerance on the
    relative error can rescue it. It is pinned here as the documented lower
    bound of the implementation: an uncertainty below about 1.49e-154 in the
    budget's unit has a subnormal square and starts losing bits, and one
    below about 2.22e-162 has a square that underflows to zero, so the term
    disappears from every root-sum-square and every quadratic form. The
    limitation is recorded in README.md.
    """
    weights = [2.2250738585072014e-308] * 4
    combined = quadratic_form(weights, np.ones((4, 4)))
    arithmetic_sum = math.fsum(weights)
    assert combined == 0.0
    assert arithmetic_sum == pytest.approx(8.900295434028806e-308, rel=1e-12)
    # The absolute disagreement is itself far below any engineering scale,
    # which is why the property test in test_properties.py carries an
    # absolute tolerance beside its relative one.
    assert abs(combined - arithmetic_sum) < 1e-300
    assert pytest.approx(1.4916681462400413e-154, rel=1e-12) == SUBNORMAL_SQUARE_THRESHOLD
    assert pytest.approx(2.2227587494850775e-162, rel=1e-12) == TOTAL_UNDERFLOW_THRESHOLD


def test_regression_rss_underflows_at_the_same_threshold():
    """The hierarchical RSS loses the same terms, for the same reason."""
    # 1e-170 squared is 1e-340, below the smallest subnormal double, so the
    # square is exactly zero and the term is lost.
    assert rss_rollup_uncertainty(flat([1e-170, 1e-170])) == 0.0
    # 1e-160 squared is 1e-320, subnormal but non-zero: the term survives
    # with reduced precision rather than disappearing.
    assert rss_rollup_uncertainty(flat([1e-160, 1e-160])) > 0.0
    # Well inside the normal range, nothing is lost: sqrt(2) * 1e-150.
    assert rss_rollup_uncertainty(flat([1e-150, 1e-150])) == pytest.approx(
        math.sqrt(2.0) * 1e-150, rel=1e-12
    )


def test_regression_margin_sign_at_the_rounding_boundary():
    """Three items of 0.1 kg against an allocation of 0.3 kg.

    0.1 + 0.1 + 0.1 evaluates to 0.30000000000000004, so the margin is
    -5.551115123125783e-17 and a strict ``margin < 0`` test reports the
    allocation as exceeded. It is not: the overrun is 5.6e-17 kg. BR003
    carries a rounding band for exactly this case and reports it as closed
    exactly, at warning severity, rather than as an error.
    """
    budget = flat([0.0, 0.0, 0.0], values=[0.1, 0.1, 0.1], allocation=0.3)
    result = rollup(budget).root
    assert result.predicted == pytest.approx(0.30000000000000004, rel=0, abs=1e-18)
    assert result.margin_absolute == pytest.approx(-5.551115123125783e-17, rel=1e-9)
    assert result.margin_absolute < 0.0  # the raw sign, stated for the record
    findings = [f for f in run_checks(budget) if f.code == "BR003"]
    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert "to within floating-point rounding" in findings[0].message


def test_regression_a_genuine_overrun_is_still_an_error():
    """The rounding band must not swallow a real overrun of engineering size."""
    budget = flat([0.0, 0.0], values=[0.2, 0.2], allocation=0.3)
    findings = [f for f in run_checks(budget) if f.code == "BR003"]
    assert len(findings) == 1
    assert findings[0].severity == "error"


def test_regression_worst_associativity_disagreement_found():
    """Worst hierarchical-against-flat RSS disagreement seen in 40000 draws.

    Uncertainties [1.9, 45.331874064628245, 1.9] split after the second term
    give 45.4114391559144 hierarchically and 45.41143915591439 flat: a
    relative disagreement of 3.13e-16, about 1.4 units in the last place.
    The identity is exact in exact arithmetic; this pins the measured size of
    the rounding so a future change that makes it worse is visible.
    """
    us = [1.9, 45.331874064628245, 1.9]
    left = Node(
        id="L",
        name="L",
        combination="rss",
        children=tuple(
            Item(id=f"l{n}", name=f"l{n}", value=1.0, uncertainty=u, maturity="mature")
            for n, u in enumerate(us[:2])
        ),
    )
    right = Node(
        id="R",
        name="R",
        combination="rss",
        children=(
            Item(id="r0", name="r0", value=1.0, uncertainty=us[2], maturity="mature"),
        ),
    )
    grouped = Budget(
        id="b",
        name="B",
        unit="kg",
        quantity="mass",
        policy=POLICY,
        root=Node(id="root", name="root", combination="rss", children=(left, right)),
    )
    hierarchical = rollup(grouped).root.uncertainty
    flat_value = rss_rollup_uncertainty(flat(us))
    assert hierarchical == pytest.approx(45.4114391559144, rel=1e-15)
    assert flat_value == pytest.approx(45.41143915591439, rel=1e-15)
    assert abs(hierarchical - flat_value) / flat_value < 1e-15


def test_regression_variance_shares_sum_drift_with_mixed_magnitudes():
    """Worst variance-share sum drift seen in the same search: 4.44e-16.

    The twelve literal uncertainties below span 152 down to 2.2e-313 and
    their shares sum to 1.0000000000000004 rather than 1. The identity is
    exact in exact arithmetic; the property test therefore compares with a
    tolerance and this pins the measured drift.
    """
    us = [
        152.0486537801309,
        270.6856638939313,
        1e-09,
        999.776050216058,
        561.2715430184469,
        2.2250738585e-313,
        238.0905804307201,
        2.220446049250313e-16,
        113.63643211986398,
        4.573555877470527e-63,
        238.0905804307201,
        188.05015820086228,
    ]
    shares = variance_shares(flat(us))
    total = math.fsum(v for v in shares.values() if v is not None)
    assert total == pytest.approx(1.0000000000000004, rel=1e-15)
    assert abs(total - 1.0) <= 1e-15
