"""The maximal robust invariant set of a discrete-time linear system.

Problem.  For the autonomous discrete-time linear system with additive bounded
disturbance

    x(k+1) = A x(k) + w(k),    w(k) in W  for all k,                       (6)

subject to the state constraint `x(k) in X` for all `k`, a set `S subset X` is
**robustly invariant** when

    for all x in S, for all w in W:   A x + w in S.                        (7)

The **maximal robust invariant set** `S_inf subset X` is the union of all sets
satisfying (7), i.e. the set of initial states from which the constraint can
be satisfied for every admissible disturbance sequence, forever.  It is
computed by the one-step-set recursion

    Omega_0     = X
    Omega_{k+1} = Omega_k  &  Pre(Omega_k)                                 (8)
    Pre(S)      = {x : A x + W subset S} = {x : a_i^T A x <= b_i - h_W(a_i)}

where the halfspace form of `Pre` is identity (I2) of `setalgebra`, i.e.
`Pre(S) = A^{-1}(S (-) W)` written without inverting `A`.  The sequence (8) is
nested and decreasing; it terminates when `Omega_k subset Pre(Omega_k)`, at
which point `Omega_k` satisfies (7) and equals `S_inf`.

Sources.  Recursion (8) and the termination test: Gilbert and Tan, "Linear
systems with state and control constraints: the theory and application of
maximal output admissible sets", *IEEE Transactions on Automatic Control*
36(9), 1991, pp. 1008-1020; Kolmanovsky and Gilbert, "Theory and computation
of disturbance invariant sets for discrete-time linear systems",
*Mathematical Problems in Engineering* 4(4), 1998, pp. 317-367; Blanchini,
"Set invariance in control - a survey", *Automatica* 35(11), 1999,
pp. 1747-1768;
Borrelli, Bemporad and Morari, *Predictive Control for Linear and Hybrid
Systems*, Cambridge University Press, 2017.

**Which set this is.**  This module computes the **maximal** robust invariant
set inside `X`.  It does **not** compute the **minimal** robust positively
invariant set, the set `F_inf = sum_{i>=0} A^i W` of Rakovic, Kerrigan,
Kouramas and Mayne, "Invariant approximations of the minimal robust positively
invariant set", *IEEE Transactions on Automatic Control* 50(3), 2005,
pp. 406-410.  The two are different objects and are easy to confuse: the
maximal set is an outer object determined by `X` and shrinks as `W` grows; the
minimal set is an inner object determined by `A` and `W` alone and does not
involve `X` at all.  In one dimension with `A = [lambda]`, `|lambda| < 1`,
`W = [-w, w]` and `X = [-b, b]`, the minimal set is `[-w/(1-|lambda|),
w/(1-|lambda|)]` while the maximal set is `[-b, b]` when `b >= w/(1-|lambda|)`
and **empty** otherwise.  Section 2 of validation/VALIDATION.md carries that
arithmetic in full.

Termination.  (8) is not guaranteed to terminate in finitely many steps.  It
does when `A` is stable and `X` is bounded with `0` in the interior of
`X (-) W` (Gilbert and Tan 1991, for the undisturbed case; Kolmanovsky and
Gilbert 1998 Theorem 4.1 for the disturbed case), but the number of steps is
not bounded a priori and for eigenvalues near the unit circle it is large.
This module therefore takes an explicit iteration cap and **reports
non-convergence as a result rather than raising or silently returning the
last iterate as if it were the answer**.

Units: `A` is dimensionless (state to state), `X` and `W` are in state units.
Validity range: `A` any real `n x n` matrix; `X` any polyhedron; `W` any
polytope bounded along the row normals that appear.  Singular `A` is handled
(it produces degenerate rows in `Pre`, which are resolved explicitly).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .polytope import (
    DEFAULT_REDUNDANCY_TOL,
    ZERO_ROW_TOL,
    EmptyPolytopeError,
    Polytope,
    VertexEnumerationError,
)
from .setalgebra import intersect

__all__ = [
    "CONVERGED",
    "EMPTY",
    "ITERATION_CAP",
    "InvariantSetResult",
    "IterationRecord",
    "maximal_robust_invariant_set",
    "pre_set",
    "verify_robust_invariance",
]

#: `Omega_k` satisfied (7); the returned polytope is `S_inf`.
CONVERGED = "converged"
#: The recursion reached the empty set; no robust invariant set exists in `X`.
EMPTY = "empty"
#: The iteration cap was hit while the sequence was still shrinking.
ITERATION_CAP = "iteration_cap"


def pre_set(A: np.ndarray, S: Polytope, W: Polytope) -> Polytope:
    """`Pre(S) = {x : A x + W subset S}`, equation (8), in H-form.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        System matrix of (6), dimensionless.
    S : Polytope
        Target set, in state units.
    W : Polytope
        Disturbance set, in state units, bounded along `S`'s row normals.

    Returns
    -------
    Polytope
        Rows `a_i^T A` with right-hand sides `b_i - h_W(a_i)`.  Rows whose
        normal `a_i^T A` vanishes (possible when `A` is singular) are kept as
        degenerate rows `0 <= rhs`; `Polytope.remove_redundant` drops them
        when `rhs >= 0` and reports emptiness when `rhs < 0`.
    """
    Aa = np.asarray(A, dtype=float)
    if Aa.ndim != 2 or Aa.shape[0] != Aa.shape[1]:
        raise ValueError(f"A must be square, got shape {Aa.shape}")
    if Aa.shape[0] != S.dim:
        raise ValueError(f"A is {Aa.shape[0]}x{Aa.shape[1]} but S has dim {S.dim}")
    if W.dim != S.dim:
        raise ValueError(f"W has dim {W.dim} but S has dim {S.dim}")
    if not np.all(np.isfinite(Aa)):
        raise ValueError("A contains a non-finite entry")
    hW = W.support_many(S.A)
    return Polytope(S.A @ Aa, S.b - hW)


def verify_robust_invariance(
    A: np.ndarray, S: Polytope, W: Polytope, tol: float = 0.0
) -> tuple[bool, float]:
    """Exact test of (7) for a given `S`, with the worst facet margin.

    For every row `(c, e)` of `S`, (7) holds if and only if

        h_S(A^T c) + h_W(c) <= e                                           (9)

    which is one linear programme per facet and involves no sampling.

    Parameters
    ----------
    A, S, W
        As in (6)-(7).
    tol : float
        Absolute slack in `b` units, default `0.0`.  **The default is strict
        on purpose and will report `False` on a correct maximal set about
        half the time.**  A maximal set has spent all of its margin, so the
        true worst margin is exactly zero and the computed one lands on
        either side of zero at the 1e-16 level depending on the LP solver's
        rounding.  Pass `tol = 1e-9` when the question is "is this set
        invariant to engineering accuracy" and leave the default when the
        question is "what is the margin".  Measured instances of both signs
        are in section 7 of validation/VALIDATION.md.

    Returns
    -------
    (bool, float)
        Whether (7) holds to `tol`, and
        `max_i (h_S(A^T c_i) + h_W(c_i) - e_i)` in `b` units.
    """
    if not np.isfinite(tol) or tol < 0:
        raise ValueError(f"tol must be finite and non-negative, got {tol!r}")
    Aa = np.asarray(A, dtype=float)
    hW = W.support_many(S.A)
    normals = S.A @ Aa
    margins = np.empty(normals.shape[0])
    for i, c in enumerate(normals):
        if np.linalg.norm(c) <= ZERO_ROW_TOL:
            margins[i] = hW[i] - S.b[i]
            continue
        margins[i] = S.support(c) + hW[i] - S.b[i]
    worst = float(np.max(margins))
    return bool(worst <= tol), worst


@dataclass(frozen=True)
class IterationRecord:
    """One iteration of recursion (8), recorded for the growth tables.

    Attributes
    ----------
    k : int
        Iteration index, 1-based: `k` is the step that produced `Omega_k`.
    halfspaces_raw : int
        Rows of `Omega_{k-1} & Pre(Omega_{k-1})` before redundancy removal.
        This is the quantity that grows without bound if nothing is removed.
    halfspaces : int
        Rows after redundancy removal.  The difference from `halfspaces_raw`
        is the whole reason redundancy removal exists.
    vertices : int or None
        Vertex count of `Omega_k`, or `None` when vertex enumeration was
        disabled or exceeded its work budget.
    volume : float or None
        Lebesgue measure of `Omega_k` in (state unit)^n, or `None`.
    shrink_margin : float
        `max_i (h_{Omega_{k-1}}(a_i) - b_i)` over the rows of
        `Pre(Omega_{k-1})`: how far `Omega_{k-1}` sticks out of its own
        pre-set, in `b` units.  Monotonically approaches 0 from above and is
        the quantity the convergence test thresholds.
    """

    k: int
    halfspaces_raw: int
    halfspaces: int
    vertices: int | None
    volume: float | None
    shrink_margin: float


@dataclass(frozen=True)
class InvariantSetResult:
    """Outcome of :func:`maximal_robust_invariant_set`.

    Attributes
    ----------
    termination : str
        One of :data:`CONVERGED`, :data:`EMPTY`, :data:`ITERATION_CAP`.
    converged : bool
        True only for :data:`CONVERGED`.  :data:`EMPTY` is a definite answer
        but not a converged set; :data:`ITERATION_CAP` is not an answer at all
        and `polytope` is then only an **outer** bound on `S_inf`.
    iterations : int
        Number of executions of (8).
    polytope : Polytope or None
        `S_inf` when converged; the last iterate when the cap was hit (an
        outer bound, since (8) is decreasing); `None` when empty.
    history : tuple of IterationRecord
    convergence_tol : float
        Absolute tolerance, in `b` units, on the `shrink_margin` test.
    redundancy_tol : float
        The `tol` passed to :meth:`Polytope.remove_redundant` every iteration.
    max_iter : int
    """

    termination: str
    converged: bool
    iterations: int
    polytope: Polytope | None
    history: tuple[IterationRecord, ...] = field(default_factory=tuple)
    convergence_tol: float = 1e-9
    redundancy_tol: float = DEFAULT_REDUNDANCY_TOL
    max_iter: int = 50

    @property
    def is_empty(self) -> bool:
        """True when the maximal robust invariant set inside `X` is empty."""
        return self.termination == EMPTY

    def report(self) -> str:
        """A multi-line human-readable report, including non-convergence.

        Returns a string; this function does not print, so it is safe to call
        from library code.
        """
        lines = [
            "maximal robust invariant set",
            f"  termination        : {self.termination}",
            f"  iterations         : {self.iterations} (cap {self.max_iter})",
            f"  convergence tol    : {self.convergence_tol:.3e}",
            f"  redundancy tol     : {self.redundancy_tol:.3e}",
        ]
        if self.termination == CONVERGED and self.polytope is not None:
            lines.append(f"  facets of S_inf    : {self.polytope.n_halfspaces}")
        elif self.termination == EMPTY:
            lines.append("  S_inf is EMPTY: no state in X can be held in X for all")
            lines.append("  admissible disturbance sequences.")
        else:
            last = self.history[-1] if self.history else None
            lines.append(
                "  DID NOT CONVERGE. The iteration cap was reached while the"
            )
            lines.append(
                "  sequence was still shrinking, so the returned polytope is an"
            )
            lines.append(
                "  OUTER bound on S_inf, not S_inf. Raise max_iter or accept the"
            )
            lines.append("  outer bound explicitly.")
            if last is not None:
                lines.append(
                    f"  last shrink margin : {last.shrink_margin:.6e} "
                    f"(needs <= {self.convergence_tol:.1e})"
                )
                lines.append(f"  facets of bound    : {last.halfspaces}")
        if self.history:
            lines.append("  k  raw  facets  vertices  shrink_margin")
            for rec in self.history:
                vtx = "-" if rec.vertices is None else str(rec.vertices)
                lines.append(
                    f"  {rec.k:<3d}{rec.halfspaces_raw:<5d}{rec.halfspaces:<8d}"
                    f"{vtx:<10s}{rec.shrink_margin:.6e}"
                )
        return "\n".join(lines)


def maximal_robust_invariant_set(
    A: np.ndarray,
    X: Polytope,
    W: Polytope,
    *,
    max_iter: int = 50,
    convergence_tol: float = 1e-9,
    redundancy_tol: float = DEFAULT_REDUNDANCY_TOL,
    track_geometry: bool = True,
    vertex_budget: int = 200_000,
) -> InvariantSetResult:
    """Compute `S_inf`, the **maximal** robust invariant set of (6) inside `X`.

    This is recursion (8) with the termination test (9).  It is *not* the
    minimal robust positively invariant set of Rakovic et al. 2005; see the
    module docstring.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        System matrix of (6), dimensionless.  For an output-feedback loop pass
        the closed-loop matrix `A - B K`.
    X : Polytope
        State constraint set, in state units.  May be unbounded; the recursion
        still runs, but `volume` and `vertices` tracking will report `None`.
    W : Polytope
        Disturbance set, in state units.  Must be bounded along the row
        normals that appear in the recursion.
    max_iter : int
        Iteration cap for (8).  Hitting it returns
        `termination=ITERATION_CAP`; it does not raise.
    convergence_tol : float
        Absolute tolerance in `b` units on the margin of (9).  Declared
        criterion: stop at the first `k` with
        `max_i (h_{Omega_k}(a_i^T A) + h_W(a_i) - b_i) <= convergence_tol`
        over the rows of `Omega_k`.  With `convergence_tol = 0` the test is as
        exact as the LP solver; a positive value accepts an `Omega_k` that
        violates (7) by at most that much, so the returned set is then
        invariant only to that slack.
    redundancy_tol : float
        Passed to :meth:`Polytope.remove_redundant` each iteration.  Changing
        it can change the answer; see section 4 of validation/VALIDATION.md.
    track_geometry : bool
        Record vertex counts and volumes per iteration.  Costs a brute-force
        vertex enumeration per iteration and is the expensive part of a run.
    vertex_budget : int
        `max_bases` for the per-iteration vertex enumeration.  Exceeding it
        records `None` rather than raising, because growth that outruns the
        enumerator is itself a measurement this package reports.

    Returns
    -------
    InvariantSetResult

    Raises
    ------
    ValueError
        On shape mismatch, non-finite data, `max_iter < 1`, or negative
        tolerances.
    """
    Aa = np.asarray(A, dtype=float)
    if Aa.ndim != 2 or Aa.shape[0] != Aa.shape[1]:
        raise ValueError(f"A must be square, got shape {Aa.shape}")
    if Aa.shape[0] != X.dim:
        raise ValueError(f"A is {Aa.shape[0]}x{Aa.shape[0]} but X has dim {X.dim}")
    if W.dim != X.dim:
        raise ValueError(f"W has dim {W.dim} but X has dim {X.dim}")
    if int(max_iter) != max_iter or max_iter < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter!r}")
    for name, value in (
        ("convergence_tol", convergence_tol),
        ("redundancy_tol", redundancy_tol),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative, got {value!r}")

    omega = X.remove_redundant(redundancy_tol)
    history: list[IterationRecord] = []

    for k in range(1, int(max_iter) + 1):
        pre = pre_set(Aa, omega, W)
        # Termination test (9), evaluated on Pre's rows: Omega_k is robustly
        # invariant exactly when Omega_k is contained in its own pre-set.
        margins = np.empty(pre.n_halfspaces)
        for i in range(pre.n_halfspaces):
            c = pre.A[i]
            if np.linalg.norm(c) <= ZERO_ROW_TOL:
                margins[i] = -pre.b[i]
                continue
            try:
                margins[i] = omega.support(c) - pre.b[i]
            except EmptyPolytopeError:
                margins[i] = -np.inf
        shrink = float(np.max(margins))

        candidate = intersect(omega, pre)
        if candidate.is_empty(tol=0.0):
            history.append(
                IterationRecord(
                    k=k,
                    halfspaces_raw=candidate.n_halfspaces,
                    halfspaces=0,
                    vertices=0,
                    volume=0.0,
                    shrink_margin=shrink,
                )
            )
            return InvariantSetResult(
                termination=EMPTY,
                converged=False,
                iterations=k,
                polytope=None,
                history=tuple(history),
                convergence_tol=convergence_tol,
                redundancy_tol=redundancy_tol,
                max_iter=int(max_iter),
            )

        if shrink <= convergence_tol:
            vtx_c: int | None = None
            vol_c: float | None = None
            if track_geometry:
                try:
                    Vc = omega.vertices(max_bases=vertex_budget)
                    vtx_c = int(Vc.shape[0])
                    vol_c = float(omega.volume())
                except (VertexEnumerationError, EmptyPolytopeError):
                    vtx_c = None
                    vol_c = None
            history.append(
                IterationRecord(
                    k=k,
                    halfspaces_raw=candidate.n_halfspaces,
                    halfspaces=omega.n_halfspaces,
                    vertices=vtx_c,
                    volume=vol_c,
                    shrink_margin=shrink,
                )
            )
            return InvariantSetResult(
                termination=CONVERGED,
                converged=True,
                iterations=k,
                polytope=omega,
                history=tuple(history),
                convergence_tol=convergence_tol,
                redundancy_tol=redundancy_tol,
                max_iter=int(max_iter),
            )

        raw = candidate.n_halfspaces
        omega = candidate.remove_redundant(redundancy_tol)
        vtx: int | None = None
        vol: float | None = None
        if track_geometry:
            try:
                V = omega.vertices(max_bases=vertex_budget)
                vtx = int(V.shape[0])
                vol = float(omega.volume())
            except (VertexEnumerationError, EmptyPolytopeError):
                vtx = None
                vol = None
        history.append(
            IterationRecord(
                k=k,
                halfspaces_raw=raw,
                halfspaces=omega.n_halfspaces,
                vertices=vtx,
                volume=vol,
                shrink_margin=shrink,
            )
        )

    return InvariantSetResult(
        termination=ITERATION_CAP,
        converged=False,
        iterations=int(max_iter),
        polytope=omega,
        history=tuple(history),
        convergence_tol=convergence_tol,
        redundancy_tol=redundancy_tol,
        max_iter=int(max_iter),
    )
