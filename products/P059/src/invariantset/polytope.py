"""Convex polytopes in halfspace form, with exact support-function evaluation.

A polytope is stored in H-representation

    P = {x in R^n : A x <= b},   A in R^(m x n),  b in R^m

which is the representation every operation in this package needs except the
Minkowski sum.  The support function

    h_P(c) = sup_{x in P} c^T x                                            (1)

is the single primitive the rest of the package is built on: the Pontryagin
difference, the robust-invariance test and the convergence criterion of the
maximal-robust-invariant-set recursion are all statements about (1).

Reference for (1) and its properties (positive homogeneity, subadditivity,
Minkowski additivity): Rockafellar, *Convex Analysis*, Princeton University
Press, 1970; Schneider, *Convex Bodies: the Brunn-Minkowski Theory*,
Cambridge University Press, 1993.  Section numbers are deliberately not
quoted: no copy of either book was available in the build container to check
them against, and VALIDATION.md section 9 says so.

Units: this module is dimensionless.  The caller fixes the physical units of
the state space, and every tolerance argument below is in those same units
(absolute, not relative) unless the docstring says otherwise.

Validity range: exact convex polyhedra with rational-to-floating-point data.
The vertex enumeration is a brute-force basis enumeration and is only usable
for small `n` and `m` (see `vertices`); everything else is linear programming
and scales with the LP solver.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError

__all__ = [
    "DEFAULT_REDUNDANCY_TOL",
    "RANK_REL_TOL",
    "ZERO_ROW_TOL",
    "EmptyPolytopeError",
    "Polytope",
    "UnboundedDirectionError",
    "VertexEnumerationError",
]

#: Default absolute slack below which a halfspace is declared redundant.
#: This number is not innocent.  Section 4 of validation/VALIDATION.md shows a
#: system on which changing it from 1e-9 to 1e-2 changes the computed set.
DEFAULT_REDUNDANCY_TOL = 1e-9

#: Rows of `A` with norm below this are treated as the degenerate row `0 <= b`.
ZERO_ROW_TOL = 1e-12

#: Relative singular-value floor below which a basis of `n` rows is treated as
#: rank-deficient in `Polytope.vertices`.  Measured effect: see section 7 of
#: validation/VALIDATION.md.
RANK_REL_TOL = 1e-10

_LP_METHOD = "highs"


class UnboundedDirectionError(ValueError):
    """Raised when `h_P(c)` is `+inf`, i.e. `P` is unbounded along `c`."""


class EmptyPolytopeError(ValueError):
    """Raised when an operation needs a non-empty polytope and `P` is empty."""


class VertexEnumerationError(RuntimeError):
    """Raised when brute-force vertex enumeration exceeds its work budget."""


def _as_2d(A: Iterable) -> np.ndarray:
    arr = np.asarray(A, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"A must be 2-D, got shape {arr.shape}")
    return arr


class Polytope:
    """A convex polyhedron `{x : A x <= b}` in H-representation.

    Parameters
    ----------
    A : array_like, shape (m, n)
        Outward halfspace normals, one per row.  Not required to be normalised.
    b : array_like, shape (m,)
        Right-hand sides, in the same units as `A @ x`.
    box_data : tuple of ndarray, optional
        `(center, half_widths)` when the caller knows the polytope is the
        axis-aligned box `|x - center| <= half_widths`.  Set only by
        :meth:`from_box`; it enables the closed-form support function
        `h(c) = c.center + |c|.half_widths` instead of a linear programme.
        The closed form is checked against the generic LP in
        `validation/validate_support_algebra.py`.

    Raises
    ------
    ValueError
        If shapes disagree, or if `A` or `b` contains a non-finite entry.
    """

    __slots__ = ("_A", "_b", "_box")

    def __init__(self, A, b, *, box_data: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        Aa = _as_2d(A)
        bb = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
        if Aa.shape[0] != bb.shape[0]:
            raise ValueError(
                f"A has {Aa.shape[0]} rows but b has {bb.shape[0]} entries"
            )
        if Aa.shape[0] == 0:
            raise ValueError("a polytope needs at least one halfspace")
        if not np.all(np.isfinite(Aa)):
            raise ValueError("A contains a non-finite entry")
        if not np.all(np.isfinite(bb)):
            raise ValueError("b contains a non-finite entry")
        self._A = Aa
        self._b = bb
        self._box = box_data

    # ---------------------------------------------------------------- builders

    @classmethod
    def from_box(cls, center, half_widths) -> Polytope:
        """The axis-aligned box `|x_i - center_i| <= half_widths_i`.

        Parameters
        ----------
        center : array_like, shape (n,)
            Box centre, in state units.
        half_widths : array_like, shape (n,)
            Non-negative half-widths, in state units.  Zero is allowed and
            produces a lower-dimensional (flat) box.

        Raises
        ------
        ValueError
            If any half-width is negative or any entry is non-finite.
        """
        m = np.atleast_1d(np.asarray(center, dtype=float)).ravel()
        r = np.atleast_1d(np.asarray(half_widths, dtype=float)).ravel()
        if m.shape != r.shape:
            raise ValueError(f"center shape {m.shape} != half_widths shape {r.shape}")
        if m.size == 0:
            raise ValueError("a box needs at least one dimension")
        if not (np.all(np.isfinite(m)) and np.all(np.isfinite(r))):
            raise ValueError("box centre and half-widths must be finite")
        if np.any(r < 0.0):
            raise ValueError(f"half-widths must be non-negative, got {r.tolist()}")
        n = m.size
        eye = np.eye(n)
        A = np.vstack([eye, -eye])
        b = np.concatenate([m + r, -(m - r)])
        return cls(A, b, box_data=(m, r))

    @classmethod
    def from_bounds(cls, lower, upper) -> Polytope:
        """The box `lower <= x <= upper`, elementwise, in state units."""
        lo = np.atleast_1d(np.asarray(lower, dtype=float)).ravel()
        hi = np.atleast_1d(np.asarray(upper, dtype=float)).ravel()
        if lo.shape != hi.shape:
            raise ValueError(f"lower shape {lo.shape} != upper shape {hi.shape}")
        if np.any(hi < lo):
            raise ValueError("every upper bound must be >= the matching lower bound")
        return cls.from_box(0.5 * (lo + hi), 0.5 * (hi - lo))

    @classmethod
    def unit_box(cls, dim: int, radius: float = 1.0) -> Polytope:
        """The centred box of half-width `radius` in `dim` dimensions."""
        if int(dim) != dim or dim < 1:
            raise ValueError(f"dim must be a positive integer, got {dim!r}")
        if not np.isfinite(radius) or radius < 0:
            raise ValueError(f"radius must be finite and non-negative, got {radius!r}")
        return cls.from_box(np.zeros(int(dim)), np.full(int(dim), float(radius)))

    # ------------------------------------------------------------- properties

    @property
    def A(self) -> np.ndarray:
        """Halfspace normals, shape (m, n).  A read-only view."""
        view = self._A.view()
        view.flags.writeable = False
        return view

    @property
    def b(self) -> np.ndarray:
        """Halfspace right-hand sides, shape (m,).  A read-only view."""
        view = self._b.view()
        view.flags.writeable = False
        return view

    @property
    def dim(self) -> int:
        """Ambient dimension `n`."""
        return self._A.shape[1]

    @property
    def n_halfspaces(self) -> int:
        """Number of rows `m` currently stored, redundant rows included."""
        return self._A.shape[0]

    @property
    def is_box(self) -> bool:
        """True when this polytope was built as an axis-aligned box."""
        return self._box is not None

    def __repr__(self) -> str:
        kind = "box" if self.is_box else "polytope"
        return f"Polytope({kind}, dim={self.dim}, halfspaces={self.n_halfspaces})"

    # --------------------------------------------------------- support function

    def support(self, c) -> float:
        """Support function `h_P(c) = sup_{x in P} c^T x`, equation (1).

        Parameters
        ----------
        c : array_like, shape (n,)
            Direction, in units reciprocal to the state so that `h` carries
            the units of `c^T x`.

        Returns
        -------
        float
            The supremum.

        Raises
        ------
        UnboundedDirectionError
            If `P` is unbounded along `c`.
        EmptyPolytopeError
            If `P` is empty, in which case the supremum is `-inf`.
        ValueError
            If `c` has the wrong shape or a non-finite entry.
        """
        cc = np.atleast_1d(np.asarray(c, dtype=float)).ravel()
        if cc.shape[0] != self.dim:
            raise ValueError(f"direction has length {cc.shape[0]}, expected {self.dim}")
        if not np.all(np.isfinite(cc)):
            raise ValueError("direction contains a non-finite entry")
        if self._box is not None:
            m, r = self._box
            return float(cc @ m + np.abs(cc) @ r)
        res = linprog(-cc, A_ub=self._A, b_ub=self._b, bounds=(None, None), method=_LP_METHOD)
        if res.status == 2:
            raise EmptyPolytopeError("support function of an empty polytope is -inf")
        if res.status == 3:
            raise UnboundedDirectionError(
                f"polytope is unbounded along direction {cc.tolist()}"
            )
        if res.status != 0:
            raise RuntimeError(f"support LP failed with status {res.status}: {res.message}")
        return float(-res.fun)

    def support_many(self, C) -> np.ndarray:
        """Support function evaluated on every row of `C`, shape (k,)."""
        CC = _as_2d(C)
        if CC.shape[1] != self.dim:
            raise ValueError(f"C has {CC.shape[1]} columns, expected {self.dim}")
        if self._box is not None:
            m, r = self._box
            return CC @ m + np.abs(CC) @ r
        return np.array([self.support(row) for row in CC], dtype=float)

    # ------------------------------------------------------------- predicates

    def contains(self, x, tol: float = 1e-9) -> bool:
        """True when `A x <= b + tol` holds rowwise.  `tol` is in `b` units."""
        xx = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        if xx.shape[0] != self.dim:
            raise ValueError(f"point has length {xx.shape[0]}, expected {self.dim}")
        if tol < 0:
            raise ValueError(f"tol must be non-negative, got {tol!r}")
        return bool(np.all(self._A @ xx <= self._b + tol))

    def is_empty(self, tol: float = 1e-9) -> bool:
        """True when `{x : A x <= b}` has no point, by an LP feasibility test.

        `tol` is an absolute slack added to `b`, in `b` units: a polytope that
        is only `tol`-feasible is reported non-empty.
        """
        if tol < 0:
            raise ValueError(f"tol must be non-negative, got {tol!r}")
        norms = np.linalg.norm(self._A, axis=1)
        degenerate = norms <= ZERO_ROW_TOL
        if np.any(degenerate & (self._b < -tol)):
            return True
        keep = ~degenerate
        if not np.any(keep):
            return False
        res = linprog(
            np.zeros(self.dim),
            A_ub=self._A[keep],
            b_ub=self._b[keep] + tol,
            bounds=(None, None),
            method=_LP_METHOD,
        )
        return res.status == 2

    def is_bounded(self) -> bool:
        """True when `P` is bounded.

        Boundedness is decided exactly: `P` is bounded if and only if it is
        bounded along every signed coordinate direction, because that places
        it inside an axis-aligned box.  `2n` linear programmes.
        """
        if self._box is not None:
            return True
        eye = np.eye(self.dim)
        for sign in (1.0, -1.0):
            for row in eye:
                try:
                    self.support(sign * row)
                except UnboundedDirectionError:
                    return False
                except EmptyPolytopeError:
                    return True
        return True

    # --------------------------------------------------------------- reduction

    def remove_redundant(self, tol: float = DEFAULT_REDUNDANCY_TOL) -> Polytope:
        """Drop halfspaces that do not bound the set, by one LP per row.

        Row `i` is declared redundant when

            max {a_i^T x : a_j^T x <= b_j for all j != i}  <=  b_i + tol    (2)

        which is the standard exact test (Borrelli, Bemporad and Morari,
        *Predictive Control for Linear and Hybrid Systems*, Cambridge
        University Press, 2017) made inexact by `tol`.

        Parameters
        ----------
        tol : float
            Absolute slack in `b` units.  **This argument changes answers.**
            With `tol = 0` the test is exact up to LP accuracy and keeps rows
            that are redundant only to floating point.  With `tol` large it
            removes rows that genuinely bound the set, and the returned
            polytope is then a strict *superset* of the input.  See section 4
            of validation/VALIDATION.md for a measured case.

        Returns
        -------
        Polytope
            A polytope with a subset of the input rows.  Degenerate rows
            (`|a_i| <= ZERO_ROW_TOL`) with `b_i >= 0` are dropped as trivially
            satisfied; with `b_i < 0` the set is empty and the single
            infeasible row `0 <= b_i` is returned unchanged.

        Raises
        ------
        ValueError
            If `tol` is negative or non-finite.
        """
        if not np.isfinite(tol) or tol < 0:
            raise ValueError(f"tol must be finite and non-negative, got {tol!r}")
        norms = np.linalg.norm(self._A, axis=1)
        degenerate = norms <= ZERO_ROW_TOL
        if np.any(degenerate & (self._b < 0.0)):
            idx = int(np.flatnonzero(degenerate & (self._b < 0.0))[0])
            return Polytope(self._A[idx : idx + 1], self._b[idx : idx + 1])
        live = np.flatnonzero(~degenerate)
        if live.size == 0:
            return Polytope(np.zeros((1, self.dim)), np.array([0.0]))
        A = self._A[live]
        b = self._b[live]
        keep = np.ones(live.size, dtype=bool)
        for i in range(live.size):
            others = keep.copy()
            others[i] = False
            if not np.any(others):
                continue
            res = linprog(
                -A[i],
                A_ub=A[others],
                b_ub=b[others],
                bounds=(None, None),
                method=_LP_METHOD,
            )
            if res.status == 3:
                continue  # unbounded without row i, so row i is not redundant
            if res.status == 2:
                keep[i] = False  # the remaining rows are already infeasible
                continue
            if res.status != 0:
                raise RuntimeError(f"redundancy LP failed: {res.message}")
            if -res.fun <= b[i] + tol:
                keep[i] = False
        if not np.any(keep):
            keep[0] = True
        return Polytope(A[keep], b[keep])

    def normalised(self) -> Polytope:
        """Scale every non-degenerate row so that `|a_i| == 1`.

        Row scaling does not change the set, but it does make `tol` arguments
        comparable between rows, which is why the diagnostics use it.
        """
        norms = np.linalg.norm(self._A, axis=1)
        scale = np.where(norms > ZERO_ROW_TOL, norms, 1.0)
        return Polytope(self._A / scale[:, None], self._b / scale)

    # ------------------------------------------------------- vertices, volume

    def vertices(
        self,
        tol: float = 1e-9,
        max_bases: int = 200_000,
    ) -> np.ndarray:
        """Vertices of `P` by brute-force basis enumeration, shape (v, n).

        Every subset of `n` rows is solved as a square system; the solution is
        kept when it satisfies all `m` halfspaces to `tol`.  Duplicates are
        merged at `tol`.  Cost is `C(m, n)` dense solves, which is why
        `max_bases` exists.

        Parameters
        ----------
        tol : float
            Absolute feasibility and deduplication tolerance, in `b` units.
        max_bases : int
            Work budget.  `C(m, n)` above this raises rather than hanging.

        Raises
        ------
        VertexEnumerationError
            If `C(m, n) > max_bases`, or if `P` is unbounded (a polyhedron
            with an unbounded direction is not the convex hull of its
            vertices and the result would be misleading).
        EmptyPolytopeError
            If `P` is empty.
        """
        if self.is_empty(tol=0.0):
            raise EmptyPolytopeError("an empty polytope has no vertices")
        if not self.is_bounded():
            raise VertexEnumerationError(
                "vertex enumeration needs a bounded polytope; this one has an "
                "unbounded direction, so it is not the hull of its vertices"
            )
        n = self.dim
        norms = np.linalg.norm(self._A, axis=1)
        live = np.flatnonzero(norms > ZERO_ROW_TOL)
        A = self._A[live]
        b = self._b[live]
        m = A.shape[0]
        n_bases = 1
        for k in range(n):
            n_bases = n_bases * (m - k) // (k + 1)
        if n_bases > max_bases:
            raise VertexEnumerationError(
                f"C({m}, {n}) = {n_bases} bases exceeds max_bases={max_bases}; "
                "this polytope is too large for brute-force vertex enumeration"
            )
        out: list[np.ndarray] = []
        for idx in itertools.combinations(range(m), n):
            rows = list(idx)
            sub = A[rows]
            # Singularity is decided on the singular values rather than the
            # determinant: a determinant underflows or overflows on badly
            # scaled rows, which is exactly the case that arises after several
            # iterations of the recursion.  A basis is skipped when its
            # smallest singular value falls below RANK_REL_TOL times its
            # largest, so near-parallel facet pairs do not contribute
            # spurious vertices at 1e8 from the origin.
            sv = np.linalg.svd(sub, compute_uv=False)
            if sv[-1] <= RANK_REL_TOL * max(float(sv[0]), 1.0):
                continue
            try:
                x = np.linalg.solve(sub, b[rows])
            except np.linalg.LinAlgError:
                continue
            if np.all(A @ x <= b + tol):
                out.append(x)
        if not out:
            return np.zeros((0, n))
        V = np.array(out)
        kept: list[np.ndarray] = []
        for v in V:
            if not any(np.linalg.norm(v - k) <= max(tol, 1e-12) * 10.0 for k in kept):
                kept.append(v)
        return np.array(kept)

    def volume(self, tol: float = 1e-9) -> float:
        """Lebesgue measure of `P`, in (state unit)^n.

        1-D uses the interval length; `n >= 2` uses the Qhull volume of the
        enumerated vertices.  A lower-dimensional polytope has volume 0, and
        Qhull reports that by failing, which is caught and returned as 0.
        """
        if self.is_empty(tol=0.0):
            return 0.0
        V = self.vertices(tol=tol)
        if V.shape[0] == 0:
            return 0.0
        if self.dim == 1:
            return float(V.max() - V.min())
        if V.shape[0] <= self.dim:
            return 0.0
        try:
            return float(ConvexHull(V).volume)
        except (QhullError, ValueError):
            return 0.0
