r"""The requirement language, its robustness semantics, and an independent
Boolean semantics that the robustness sign is property-tested against.

The fragment, completely
------------------------
Terms (evaluate to a real array of length ``T``, in the unit of the named
signal, or that unit per second for a difference):

===================== =========================================================
``Signal(name)``      ``s[i] = w_name(t_i)``
``Difference(name)``  backward difference, forward difference at the left edge:
                      ``d[0] = (x[1]-x[0])/dt``,
                      ``d[i] = (x[i]-x[i-1])/dt`` for ``i >= 1``.
                      Unit: unit of ``name`` per second.
``Abs(term)``         ``|term[i]|``
===================== =========================================================

Formulas (evaluate to a robustness array ``rho`` of length ``T``, and
independently to a Boolean array ``sat`` of length ``T``):

================================ ==============================================
``Predicate(term, "<=", c, s)``  ``rho[i] = (c - term[i]) / s``,
                                 ``sat[i] = term[i] <= c``
``Predicate(term, ">=", c, s)``  ``rho[i] = (term[i] - c) / s``,
                                 ``sat[i] = term[i] >= c``
``And(f1, ..., fn)``             ``rho[i] = min_k rho_k[i]``,
                                 ``sat[i] = all_k sat_k[i]``
``Or(f1, ..., fn)``              ``rho[i] = max_k rho_k[i]``,
                                 ``sat[i] = any_k sat_k[i]``
``Always(f, a, b)``              ``rho[i] = min_{j in W(i)} rho_f[j]``,
                                 ``sat[i] = all_{j in W(i)} sat_f[j]``;
                                 ``+inf`` / ``True`` on an empty window
``Eventually(f, a, b)``          ``rho[i] = max_{j in W(i)} rho_f[j]``,
                                 ``sat[i] = any_{j in W(i)} sat_f[j]``;
                                 ``-inf`` / ``False`` on an empty window
================================ ==============================================

``W(i)`` is the sample window ``[i + round(a/dt), i + round(b/dt)]`` intersected
with ``[0, T-1]``. The bounds ``a <= b`` are in seconds and **must be integer
multiples of the trace's sample interval**; a bound that is not is rejected with
a ``ValueError`` rather than silently rounded, because rounding would make the
reported window differ from the evaluated one.

The requirement of a whole trace is the value at ``i = 0``:
:func:`robustness` returns ``rho[0]``, :func:`satisfies` returns ``sat[0]``.

Why the sign agreement is exact, and why there is no negation
-------------------------------------------------------------
The deliverable property is

.. math:: \rho(\varphi, w) < 0 \iff w \not\models \varphi

equivalently ``rho >= 0`` iff satisfied, with **no tolerance band**. It holds
elementwise and therefore at ``i = 0``:

* For a predicate, ``(c - x)/s >= 0`` iff ``x <= c`` for ``s > 0``. In IEEE-754
  double arithmetic the difference of two distinct finite doubles is never
  rounded to zero (their separation is at least one ulp, hence at least one
  subnormal), so ``c - x`` has the sign of the real difference exactly, and
  division by a positive ``s`` preserves it.
* ``min`` is ``>= 0`` iff every argument is; ``max`` is ``>= 0`` iff some
  argument is. The windowed forms are the same statement over ``W(i)``.
* The empty-window conventions are chosen to match: ``+inf >= 0`` is ``True``
  and ``-inf >= 0`` is ``False``.

**The language has no negation operator, deliberately.** With a ``Not`` node the
equivalence would fail on the zero set: ``rho(Not f) = -rho(f) = 0`` when
``rho(f) = 0``, so ``Not f`` would be scored non-negative while being
unsatisfied. Instead the comparison operator comes in both directions, so every
formula is already in negation normal form and the equivalence is exact
including at the boundary. The cost is that ``Not`` and ``Until`` are not
expressible; see the README's alternatives table, where ``rtamt`` is named for
exactly this reason.

A note on units that the robustness value cannot protect you from
-----------------------------------------------------------------
``And``/``Or`` take a ``min``/``max`` across their arguments. If those arguments
are predicates on terms with different units -- degrees and degrees per second,
say -- the resulting robustness *magnitude* mixes units and is not interpretable
as a margin in any single unit. Its **sign is still exact**, which is all the
falsification search uses. Set ``scale`` on each predicate to the requirement's
own tolerance to make the arguments dimensionless and the magnitude comparable;
``scale`` is required to be strictly positive so it can never change a sign.

References
----------
Fainekos, G. E. and Pappas, G. J. (2009), "Robustness of temporal logic
specifications for continuous-time signals", Theoretical Computer Science
410(42), 4262-4291. The ``min``/``max`` robustness-degree semantics reproduced
here for the bounded fragment.

Donze, A. and Maler, O. (2010), "Robust satisfaction of temporal logic over
real-valued signals", FORMATS 2010, LNCS 6246. Time-bounded operators.

Maler, O. and Nickovic, D. (2004), "Monitoring temporal properties of continuous
signals", FORMATS/FTRTFT 2004, LNCS 3253. Signal temporal logic.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from .traces import Trace

#: Absolute tolerance on "is this time bound an integer multiple of dt".
BOUND_ATOL = 1e-9

_OPERATORS = ("<=", ">=")


# --------------------------------------------------------------------------- #
# Terms
# --------------------------------------------------------------------------- #
class Term(ABC):
    """A real-valued function of a trace, evaluated at every sample."""

    @abstractmethod
    def values(self, trace: Trace) -> np.ndarray:
        """Return the term's value at every sample, shape ``(T,)``."""

    @abstractmethod
    def signals(self) -> frozenset[str]:
        """Signal names this term reads."""

    @abstractmethod
    def __str__(self) -> str: ...


class Signal(Term):
    """The named signal itself. Unit: the unit of that signal.

    Parameters
    ----------
    name:
        Signal name, which must exist on every trace this term is evaluated on.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError(f"signal name must be a non-empty string, got {name!r}")
        self.name = name

    def values(self, trace: Trace) -> np.ndarray:
        return trace.signal(self.name)

    def signals(self) -> frozenset[str]:
        return frozenset({self.name})

    def __str__(self) -> str:
        return self.name


class Difference(Term):
    """First difference of a signal, in signal units per second.

    ``d[0] = (x[1] - x[0]) / dt`` and ``d[i] = (x[i] - x[i-1]) / dt`` for
    ``i >= 1``: a backward difference with a one-sided forward difference at the
    left edge, so that the derived signal has the same length as the trace and
    no sample carries a fabricated zero. The convention matters at ``i = 0``
    only, and is the standard one-sided first-order approximation; its
    truncation error is ``O(dt)`` for a smooth signal, which is the same order
    as the backward difference used everywhere else.

    Parameters
    ----------
    name:
        Signal to difference.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError(f"signal name must be a non-empty string, got {name!r}")
        self.name = name

    def values(self, trace: Trace) -> np.ndarray:
        x = trace.signal(self.name)
        d = np.empty_like(x)
        d[1:] = (x[1:] - x[:-1]) / trace.dt
        d[0] = d[1]
        return d

    def signals(self) -> frozenset[str]:
        return frozenset({self.name})

    def __str__(self) -> str:
        return f"d/dt({self.name})"


class Abs(Term):
    """Absolute value of a term. Unit: unchanged.

    Parameters
    ----------
    term:
        The term to take the magnitude of.
    """

    __slots__ = ("term",)

    def __init__(self, term: Term) -> None:
        if not isinstance(term, Term):
            raise TypeError(f"Abs expects a Term, got {type(term)!r}")
        self.term = term

    def values(self, trace: Trace) -> np.ndarray:
        return np.abs(self.term.values(trace))

    def signals(self) -> frozenset[str]:
        return self.term.signals()

    def __str__(self) -> str:
        return f"|{self.term}|"


# --------------------------------------------------------------------------- #
# Formulas
# --------------------------------------------------------------------------- #
class Formula(ABC):
    """A requirement over a trace, with a robustness and a Boolean semantics.

    The two semantics are implemented separately on purpose. :meth:`rho` does
    arithmetic and ``min``/``max``; :meth:`sat` does comparisons and
    ``all``/``any``. Nothing is shared between them below the term layer, so the
    Hypothesis property test that asserts their sign agreement is testing two
    independent implementations rather than one implementation twice.
    """

    @abstractmethod
    def rho(self, trace: Trace) -> np.ndarray:
        """Robustness at every sample, shape ``(T,)``, possibly infinite."""

    @abstractmethod
    def sat(self, trace: Trace) -> np.ndarray:
        """Boolean satisfaction at every sample, shape ``(T,)``, dtype bool."""

    @abstractmethod
    def signals(self) -> frozenset[str]:
        """Signal names this formula reads."""

    @abstractmethod
    def horizon(self) -> float:
        """Future time in seconds this formula needs beyond the sample it scores.

        The sum of the upper time bounds along the deepest nesting path. A trace
        shorter than ``horizon()`` will evaluate against clipped (or empty)
        windows, which is legal but means the verdict at ``t = 0`` was taken
        with less evidence than the requirement asks for; use
        :func:`check_horizon` to be told so.
        """

    @abstractmethod
    def __str__(self) -> str: ...


class Predicate(Formula):
    """``term op bound``, with robustness ``(bound - term)/scale`` or its negative.

    Parameters
    ----------
    term:
        The term being bounded.
    op:
        ``"<="`` or ``">="``. There is no strict form: with floating-point
        samples the strict and non-strict predicates differ only on a measure-zero
        set that no sampled simulation lands on reliably, and admitting only the
        non-strict form keeps the robustness sign exact at the boundary.
    bound:
        The bound, in the unit of ``term``.
    scale:
        Strictly positive normaliser, in the unit of ``term``. The robustness is
        divided by it, which makes the value dimensionless when ``scale`` is the
        requirement's own tolerance, and cannot change its sign.
    """

    __slots__ = ("bound", "op", "scale", "term")

    def __init__(self, term: Term, op: str, bound: float, scale: float = 1.0) -> None:
        if not isinstance(term, Term):
            raise TypeError(f"Predicate expects a Term, got {type(term)!r}")
        if op not in _OPERATORS:
            raise ValueError(f"op must be one of {_OPERATORS}, got {op!r}")
        bound = float(bound)
        if not math.isfinite(bound):
            raise ValueError(f"bound must be finite, got {bound}")
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"scale must be finite and strictly positive, got {scale}")
        self.term = term
        self.op = op
        self.bound = bound
        self.scale = scale

    def rho(self, trace: Trace) -> np.ndarray:
        x = self.term.values(trace)
        margin = (self.bound - x) if self.op == "<=" else (x - self.bound)
        return margin / self.scale

    def sat(self, trace: Trace) -> np.ndarray:
        x = self.term.values(trace)
        return (x <= self.bound) if self.op == "<=" else (x >= self.bound)

    def signals(self) -> frozenset[str]:
        return self.term.signals()

    def horizon(self) -> float:
        return 0.0

    def __str__(self) -> str:
        return f"({self.term} {self.op} {self.bound:g})"


class _Junction(Formula):
    """Shared validation for :class:`And` and :class:`Or`."""

    __slots__ = ("parts",)

    def __init__(self, *parts: Formula) -> None:
        if len(parts) < 1:
            raise ValueError(f"{type(self).__name__} needs at least one argument, got none")
        for part in parts:
            if not isinstance(part, Formula):
                raise TypeError(
                    f"{type(self).__name__} expects Formula arguments, got {type(part)!r}"
                )
        self.parts = tuple(parts)

    def signals(self) -> frozenset[str]:
        return frozenset().union(*(p.signals() for p in self.parts))

    def horizon(self) -> float:
        return max(p.horizon() for p in self.parts)


class And(_Junction):
    """Conjunction: robustness ``min``, satisfaction ``all``."""

    __slots__ = ()

    def rho(self, trace: Trace) -> np.ndarray:
        return np.min(np.stack([p.rho(trace) for p in self.parts]), axis=0)

    def sat(self, trace: Trace) -> np.ndarray:
        out = self.parts[0].sat(trace).copy()
        for part in self.parts[1:]:
            out &= part.sat(trace)
        return out

    def __str__(self) -> str:
        return "(" + " and ".join(str(p) for p in self.parts) + ")"


class Or(_Junction):
    """Disjunction: robustness ``max``, satisfaction ``any``."""

    __slots__ = ()

    def rho(self, trace: Trace) -> np.ndarray:
        return np.max(np.stack([p.rho(trace) for p in self.parts]), axis=0)

    def sat(self, trace: Trace) -> np.ndarray:
        out = self.parts[0].sat(trace).copy()
        for part in self.parts[1:]:
            out |= part.sat(trace)
        return out

    def __str__(self) -> str:
        return "(" + " or ".join(str(p) for p in self.parts) + ")"


def _offsets(trace: Trace, lo: float, hi: float) -> tuple[int, int]:
    """Convert a window ``[lo, hi]`` in seconds to integer sample offsets.

    Raises
    ------
    ValueError
        If either bound is not an integer multiple of ``trace.dt`` to within
        :data:`BOUND_ATOL` seconds. Silent rounding is refused because the
        window actually evaluated would then differ from the window the
        requirement states.
    """
    out = []
    for name, bound in (("lower", lo), ("upper", hi)):
        ratio = bound / trace.dt
        nearest = round(ratio)
        if abs(ratio - nearest) * trace.dt > BOUND_ATOL:
            raise ValueError(
                f"{name} time bound {bound:g} s is not an integer multiple of the trace "
                f"sample interval dt = {trace.dt:g} s (ratio {ratio:.12g}); choose a bound "
                "on the sample grid rather than letting the window be rounded"
            )
        out.append(int(nearest))
    return out[0], out[1]


class _Temporal(Formula):
    """Shared validation and window arithmetic for the bounded operators."""

    __slots__ = ("hi", "inner", "lo")

    def __init__(self, inner: Formula, lo: float, hi: float) -> None:
        if not isinstance(inner, Formula):
            raise TypeError(f"{type(self).__name__} expects a Formula, got {type(inner)!r}")
        lo, hi = float(lo), float(hi)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"time bounds must be finite, got [{lo}, {hi}]")
        if lo < 0.0:
            raise ValueError(f"lower time bound must be non-negative, got {lo}")
        if hi < lo:
            raise ValueError(f"time bounds must satisfy lo <= hi, got [{lo}, {hi}]")
        self.inner = inner
        self.lo = lo
        self.hi = hi

    def signals(self) -> frozenset[str]:
        return self.inner.signals()

    def horizon(self) -> float:
        return self.hi + self.inner.horizon()

    def _reduce(self, inner: np.ndarray, trace: Trace, identity, reducer) -> np.ndarray:
        """Apply ``reducer`` over the sample window of every index, vectorised.

        The window of index ``i`` is ``[i + la, i + lb]`` clipped to the trace.
        Because ``lo >= 0`` the left edge never needs clipping, so the only
        clipping is at the right, and padding the inner array with ``lb`` copies
        of the reducer's identity makes the clipped and empty windows come out
        exactly as the conventions in the class docstrings require: an empty
        window reduces over padding alone and yields the identity.

        Complexity ``O(T * (lb - la + 1))`` element visits with no Python-level
        loop over samples, which is what keeps a falsification run that
        evaluates one requirement per simulation inside the compute budget.
        """
        la, lb = _offsets(trace, self.lo, self.hi)
        n = trace.length
        width = lb - la + 1
        padded = np.concatenate([inner, np.full(lb, identity, dtype=inner.dtype)])
        view = np.lib.stride_tricks.sliding_window_view(padded, width)
        return reducer(view[la : la + n], axis=1)


class Always(_Temporal):
    """``always[lo, hi] inner``: ``min`` robustness, ``all`` satisfaction.

    An empty window -- which happens at samples whose window lies entirely past
    the end of the trace -- scores ``+inf`` and ``True``, the identity of
    ``min`` and ``all``. That is the vacuous-truth convention, and it is the
    reason :func:`check_horizon` exists: a requirement whose window never fit in
    the trace is satisfied for a reason that has nothing to do with the system.
    """

    __slots__ = ()

    def rho(self, trace: Trace) -> np.ndarray:
        return self._reduce(self.inner.rho(trace), trace, np.inf, np.min)

    def sat(self, trace: Trace) -> np.ndarray:
        return self._reduce(self.inner.sat(trace), trace, True, np.all)

    def __str__(self) -> str:
        return f"always[{self.lo:g},{self.hi:g}] {self.inner}"


class Eventually(_Temporal):
    """``eventually[lo, hi] inner``: ``max`` robustness, ``any`` satisfaction.

    An empty window scores ``-inf`` and ``False``, the identity of ``max`` and
    ``any``.
    """

    __slots__ = ()

    def rho(self, trace: Trace) -> np.ndarray:
        return self._reduce(self.inner.rho(trace), trace, -np.inf, np.max)

    def sat(self, trace: Trace) -> np.ndarray:
        return self._reduce(self.inner.sat(trace), trace, False, np.any)

    def __str__(self) -> str:
        return f"eventually[{self.lo:g},{self.hi:g}] {self.inner}"


# --------------------------------------------------------------------------- #
# Top-level evaluation
# --------------------------------------------------------------------------- #
def robustness(formula: Formula, trace: Trace) -> float:
    """Robustness of ``formula`` on ``trace``, in the unit of its terms.

    Negative exactly when the requirement is violated. May be ``+-inf`` when a
    time-bounded window is empty.
    """
    if not isinstance(formula, Formula):
        raise TypeError(f"expected a Formula, got {type(formula)!r}")
    if not isinstance(trace, Trace):
        raise TypeError(f"expected a Trace, got {type(trace)!r}")
    missing = formula.signals() - set(trace.names)
    if missing:
        raise KeyError(
            f"formula reads signals {sorted(missing)} that the trace does not have; "
            f"trace has {list(trace.names)}"
        )
    return float(formula.rho(trace)[0])


def satisfies(formula: Formula, trace: Trace) -> bool:
    """Whether ``trace`` satisfies ``formula``, by the Boolean semantics alone.

    Computed without reference to :func:`robustness`. The agreement
    ``robustness(f, w) >= 0 == satisfies(f, w)`` is the property tested in
    ``tests/test_sign_agreement.py``; it is not assumed anywhere in this module.
    """
    if not isinstance(formula, Formula):
        raise TypeError(f"expected a Formula, got {type(formula)!r}")
    if not isinstance(trace, Trace):
        raise TypeError(f"expected a Trace, got {type(trace)!r}")
    missing = formula.signals() - set(trace.names)
    if missing:
        raise KeyError(
            f"formula reads signals {sorted(missing)} that the trace does not have; "
            f"trace has {list(trace.names)}"
        )
    return bool(formula.sat(trace)[0])


def violated(formula: Formula, trace: Trace) -> bool:
    """``robustness(formula, trace) < 0``. The falsification objective's verdict."""
    return robustness(formula, trace) < 0.0


def check_horizon(formula: Formula, trace: Trace) -> None:
    """Raise if ``trace`` is too short to evaluate ``formula`` over full windows.

    Raises
    ------
    ValueError
        If ``formula.horizon()`` exceeds the trace duration. A verdict taken on
        a clipped window is legal under the semantics and misleading in a
        report, so the benchmark instances call this at construction.
    """
    need = formula.horizon()
    have = trace.duration
    if need > have + BOUND_ATOL:
        raise ValueError(
            f"formula needs {need:g} s of trace beyond t=0 but the trace is {have:g} s long; "
            "the verdict at t=0 would be taken over a clipped or empty window"
        )
