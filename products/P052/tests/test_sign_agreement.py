r"""The core deliverable, as a property test.

The claim
---------
For every formula in the shipped fragment and every trace,

    ``robustness(formula, trace) < 0``  if and only if  ``not satisfies(formula, trace)``

equivalently ``robustness(...) >= 0 == satisfies(...)``, **with no tolerance
band**. The two sides are computed by two separate implementations:
:meth:`Formula.rho` does arithmetic and ``min``/``max``, :meth:`Formula.sat`
does comparisons and ``all``/``any``, and nothing is shared between them below
the term layer.

Hypothesis generates the formula and the trace together, so the test covers
arbitrary nesting of the operators rather than a hand-picked list. The
elementwise form is asserted as well as the value at ``t = 0``, because the
elementwise statement is the inductive one and a failure there localises the
defect.

What is deliberately excluded, and why
--------------------------------------
``nan`` is excluded because the trace constructor rejects it: a ``nan`` sample
makes every comparison false while leaving the arithmetic ``nan``, so the
equivalence would fail for a reason that has nothing to do with the semantics.
Infinities in the *robustness* are generated and included (they come from empty
time windows), and the equivalence must hold through them, which is why the
empty-window conventions were chosen as they were.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from falsifyloop.requirements import (
    Abs,
    Always,
    And,
    Difference,
    Eventually,
    Or,
    Predicate,
    Signal,
    robustness,
    satisfies,
)
from falsifyloop.traces import Trace

SIGNAL_NAMES = ("a", "b")
DT = 0.1
MAX_SAMPLES = 9

_finite = st.floats(
    min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False, width=32
)


@st.composite
def traces(draw: st.DrawFn) -> Trace:
    """A short uniformly sampled trace with both signals present."""
    n = draw(st.integers(min_value=2, max_value=MAX_SAMPLES))
    signals = {
        name: np.asarray(
            draw(st.lists(_finite, min_size=n, max_size=n)), dtype=float
        )
        for name in SIGNAL_NAMES
    }
    return Trace(np.arange(n) * DT, signals)


@st.composite
def terms(draw: st.DrawFn) -> object:
    """``Signal``, ``Difference``, or either wrapped in ``Abs``."""
    name = draw(st.sampled_from(SIGNAL_NAMES))
    base = draw(st.sampled_from([Signal, Difference]))(name)
    return Abs(base) if draw(st.booleans()) else base


def _bounds(draw: st.DrawFn) -> tuple[float, float]:
    """A window on the sample grid, ``lo <= hi``."""
    la = draw(st.integers(min_value=0, max_value=MAX_SAMPLES))
    extra = draw(st.integers(min_value=0, max_value=MAX_SAMPLES))
    return la * DT, (la + extra) * DT


@st.composite
def formulas(draw: st.DrawFn, depth: int = 2) -> object:
    """A formula of the shipped fragment, nested up to ``depth``."""
    if depth <= 0:
        kind = "predicate"
    else:
        kind = draw(
            st.sampled_from(["predicate", "and", "or", "always", "eventually"])
        )
    if kind == "predicate":
        return Predicate(
            draw(terms()),
            draw(st.sampled_from(["<=", ">="])),
            draw(st.floats(min_value=-60.0, max_value=60.0, allow_nan=False, width=32)),
            scale=draw(st.floats(min_value=0.25, max_value=4.0, allow_nan=False, width=32)),
        )
    if kind in {"and", "or"}:
        parts = draw(
            st.lists(formulas(depth=depth - 1), min_size=1, max_size=3)
        )
        return And(*parts) if kind == "and" else Or(*parts)
    lo, hi = _bounds(draw)
    inner = draw(formulas(depth=depth - 1))
    return Always(inner, lo, hi) if kind == "always" else Eventually(inner, lo, hi)


_SETTINGS = settings(
    max_examples=400,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@given(formula=formulas(), trace=traces())
@_SETTINGS
def test_robustness_is_negative_exactly_when_the_requirement_is_violated(
    formula, trace
) -> None:
    rho = robustness(formula, trace)
    sat = satisfies(formula, trace)
    assert (rho < 0.0) == (not sat), (
        f"sign disagreement at t=0: rho={rho!r}, satisfied={sat!r}, formula={formula}"
    )
    assert (rho >= 0.0) == sat


@given(formula=formulas(), trace=traces())
@_SETTINGS
def test_sign_agreement_holds_elementwise_not_just_at_time_zero(formula, trace) -> None:
    rho = formula.rho(trace)
    sat = formula.sat(trace)
    assert rho.shape == sat.shape == (trace.length,)
    np.testing.assert_array_equal(
        rho >= 0.0,
        sat,
        err_msg=f"elementwise sign disagreement for formula={formula}",
    )


@given(formula=formulas(), trace=traces())
@_SETTINGS
def test_boolean_semantics_never_returns_a_non_boolean(formula, trace) -> None:
    assert formula.sat(trace).dtype == np.bool_


@given(formula=formulas(), trace=traces())
@_SETTINGS
def test_robustness_is_never_nan(formula, trace) -> None:
    # Infinite robustness is legal and comes from empty time windows; nan is not,
    # because it would make the sign agreement meaningless.
    assert not np.any(np.isnan(formula.rho(trace)))


@given(
    formula=formulas(),
    trace=traces(),
    scale=st.floats(min_value=0.125, max_value=8.0, allow_nan=False, width=32),
)
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_positive_scaling_of_every_predicate_cannot_change_the_verdict(
    formula, trace, scale
) -> None:
    """Rescaling all predicate scales by a positive factor preserves the verdict.

    The robustness magnitude changes; the sign must not. This is the property
    that licenses reporting a dimensionless robustness without ever worrying that
    a normalisation flipped a result.
    """
    rescaled = _rescale(formula, scale)
    assert satisfies(rescaled, trace) == satisfies(formula, trace)
    assert (robustness(rescaled, trace) < 0.0) == (robustness(formula, trace) < 0.0)


def _rescale(formula, factor: float):
    """Rebuild ``formula`` with every predicate scale multiplied by ``factor``."""
    if isinstance(formula, Predicate):
        return Predicate(formula.term, formula.op, formula.bound, formula.scale * factor)
    if isinstance(formula, And):
        return And(*(_rescale(p, factor) for p in formula.parts))
    if isinstance(formula, Or):
        return Or(*(_rescale(p, factor) for p in formula.parts))
    if isinstance(formula, Always):
        return Always(_rescale(formula.inner, factor), formula.lo, formula.hi)
    if isinstance(formula, Eventually):
        return Eventually(_rescale(formula.inner, factor), formula.lo, formula.hi)
    raise TypeError(f"unhandled formula type {type(formula)!r}")


@given(
    trace=traces(),
    bound_lo=st.floats(min_value=-40.0, max_value=0.0, allow_nan=False, width=32),
    gap=st.floats(min_value=0.0, max_value=40.0, allow_nan=False, width=32),
)
@settings(max_examples=200, deadline=None)
def test_a_looser_upper_bound_is_never_harder_to_satisfy(trace, bound_lo, gap) -> None:
    """Monotonicity: raising the bound of a ``<=`` predicate cannot lose satisfaction.

    An algebraic identity of the semantics, and the one a falsification user
    relies on when they tighten a requirement to make an instance harder.
    """
    tight = Always(Predicate(Signal("a"), "<=", bound_lo), 0.0, 0.1)
    loose = Always(Predicate(Signal("a"), "<=", bound_lo + gap), 0.0, 0.1)
    assert robustness(loose, trace) >= robustness(tight, trace)
    if satisfies(tight, trace):
        assert satisfies(loose, trace)


@given(trace=traces())
@settings(max_examples=200, deadline=None)
def test_always_is_never_more_robust_than_eventually_on_the_same_window(trace) -> None:
    """``min <= max`` over a non-empty window, for every inner formula."""
    inner = Predicate(Abs(Signal("b")), "<=", 3.0)
    window = (0.0, 0.3)
    assert robustness(Always(inner, *window), trace) <= robustness(
        Eventually(inner, *window), trace
    )
