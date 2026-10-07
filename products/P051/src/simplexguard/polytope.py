"""Convex polytopes in halfspace form and the support-function algebra.

Everything the runtime-assurance switching condition needs is in this module and
nothing else is. A polytope is stored as ``{x : A x <= b}`` with ``A`` of shape
``(m, n)`` and ``b`` of shape ``(m,)``; no vertex representation is maintained,
and no operation in this module converts between representations except
:meth:`Polytope.vertices_2d`, which exists for plotting and is restricted to
``n == 2``.

Definitions and sources
-----------------------
Support function (Rockafellar, *Convex Analysis*, Princeton, 1970, section 13;
Schneider, *Convex Bodies: the Brunn-Minkowski Theory*, CUP, 1993, section 1.7)

    h_P(c) = sup_{x in P} c^T x,                                           (1)

which for a bounded polytope is a linear programme and for an axis-aligned box
``[l, u]`` is the closed form

    h_box(c) = c^T m + |c|^T r,   m = (l+u)/2,   r = (u-l)/2.              (2)

Pontryagin (erosion) difference, written ``P (-) W``
(Kolmanovsky and Gilbert, "Theory and computation of disturbance invariant sets
for discrete-time linear systems", *Mathematical Problems in Engineering* 4(4),
1998, pp. 317-367; Blanchini and Miani, *Set-Theoretic Methods in Control*,
Birkhauser, 2008)

    P (-) W = {x : x + w in P for all w in W}
            = {x : a_i^T x <= b_i - h_W(a_i), i = 1..m}.                   (3)

Identity (3) is exact for a polytope in halfspace form; it is what makes the
switching condition in :mod:`simplexguard.guard` an exact computation rather
than a sampled one, and it is property-tested in
``tests/test_properties.py``.

Preimage under a linear map (same references)

    M^{-1} P = {x : M x in P} = {x : (A M) x <= b}.                        (4)

Units
-----
This module is dimensionless: it manipulates numbers. The physical units of the
coordinates are fixed by the caller and documented in
:mod:`simplexguard.plant`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

__all__ = ["Box", "Polytope", "box", "unit_normalise"]

_LP_METHOD = "highs"


def unit_normalise(matrix: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale each row of ``A x <= b`` to unit 2-norm in ``A``.

    Rows that are numerically zero are left untouched; a zero row with a
    negative right-hand side is an infeasible row and is preserved so that
    emptiness is detected rather than silently dropped.
    """
    matrix = np.asarray(matrix, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    norms = np.linalg.norm(matrix, axis=1)
    safe = np.where(norms > 0.0, norms, 1.0)
    return matrix / safe[:, None], rhs / safe


@dataclass(frozen=True)
class Polytope:
    """A convex polytope ``{x : A x <= b}``.

    Parameters
    ----------
    A :
        Shape ``(m, n)``, the outward facet normals, one per row.
    b :
        Shape ``(m,)``, the facet offsets.

    The constructor validates shapes and finiteness and raises ``ValueError``
    or ``TypeError`` on bad input. It does **not** check boundedness: an
    unbounded polytope is a legal object here, and :meth:`support` returns
    ``inf`` in an unbounded direction.
    """

    A: np.ndarray
    b: np.ndarray

    def __post_init__(self) -> None:
        a = np.asarray(self.A, dtype=float)
        rhs = np.asarray(self.b, dtype=float)
        if a.ndim != 2:
            raise ValueError(f"A must be 2-D, got shape {a.shape}")
        if rhs.ndim != 1:
            raise ValueError(f"b must be 1-D, got shape {rhs.shape}")
        if a.shape[0] != rhs.shape[0]:
            raise ValueError(f"A has {a.shape[0]} rows but b has {rhs.shape[0]} entries")
        if a.shape[0] == 0:
            raise ValueError("a polytope needs at least one halfspace")
        if not np.all(np.isfinite(a)):
            raise ValueError("A contains non-finite entries")
        if not np.all(np.isfinite(rhs)):
            raise ValueError("b contains non-finite entries")
        object.__setattr__(self, "A", a)
        object.__setattr__(self, "b", rhs)

    # ---------------------------------------------------------------- shape
    @property
    def dim(self) -> int:
        """Ambient dimension ``n``."""
        return int(self.A.shape[1])

    @property
    def n_halfspaces(self) -> int:
        """Number of rows ``m``, including any redundant ones."""
        return int(self.A.shape[0])

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"Polytope(dim={self.dim}, halfspaces={self.n_halfspaces})"

    # ------------------------------------------------------------- support
    def support(self, direction: np.ndarray) -> float:
        """Support function ``h_P(c)`` of equation (1), by linear programme.

        Returns ``+inf`` if the polytope is unbounded in ``direction`` and
        ``-inf`` if the polytope is empty, which is the convention
        ``sup`` over the empty set takes.
        """
        c = np.asarray(direction, dtype=float).ravel()
        if c.shape[0] != self.dim:
            raise ValueError(f"direction has length {c.shape[0]}, expected {self.dim}")
        if not np.all(np.isfinite(c)):
            raise ValueError("direction contains non-finite entries")
        res = linprog(
            -c,
            A_ub=self.A,
            b_ub=self.b,
            bounds=[(None, None)] * self.dim,
            method=_LP_METHOD,
        )
        if res.status == 2:  # infeasible: the polytope is empty
            return -np.inf
        if res.status == 3:  # unbounded
            return np.inf
        if not res.success:  # pragma: no cover - solver pathology
            raise RuntimeError(f"support LP failed: {res.message}")
        return float(-res.fun)

    def support_many(self, directions: np.ndarray) -> np.ndarray:
        """Support function evaluated at each row of ``directions``.

        The generic implementation is one linear programme per row. :class:`Box`
        overrides it with the closed form (2), which is the path the guard takes
        on a box disturbance set.
        """
        d = np.atleast_2d(np.asarray(directions, dtype=float))
        if d.shape[1] != self.dim:
            raise ValueError(f"directions have width {d.shape[1]}, expected {self.dim}")
        return np.array([self.support(row) for row in d], dtype=float)

    # ------------------------------------------------------- membership
    def residual(self, x: np.ndarray) -> np.ndarray:
        """Row-wise constraint residual ``A x - b``; non-positive inside."""
        v = np.asarray(x, dtype=float).ravel()
        if v.shape[0] != self.dim:
            raise ValueError(f"x has length {v.shape[0]}, expected {self.dim}")
        return self.A @ v - self.b

    def contains(self, x: np.ndarray, tol: float = 1e-9) -> bool:
        """True if ``A x <= b + tol`` row-wise."""
        return bool(np.all(self.residual(x) <= tol))

    def contains_many(self, xs: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        """Vectorised :meth:`contains` over the rows of ``xs``."""
        pts = np.atleast_2d(np.asarray(xs, dtype=float))
        if pts.shape[1] != self.dim:
            raise ValueError(f"points have width {pts.shape[1]}, expected {self.dim}")
        return np.all(pts @ self.A.T - self.b <= tol, axis=1)

    def slack(self, x: np.ndarray) -> float:
        """Smallest row slack ``min_i (b_i - a_i^T x)``; positive inside."""
        return float(np.min(-self.residual(x)))

    # ---------------------------------------------------------- algebra
    def intersect(self, other: Polytope) -> Polytope:
        """Stack the halfspaces of both polytopes. No redundancy removal."""
        if other.dim != self.dim:
            raise ValueError(f"dimension mismatch: {self.dim} and {other.dim}")
        return Polytope(np.vstack([self.A, other.A]), np.concatenate([self.b, other.b]))

    def erode(self, disturbance: Polytope) -> Polytope:
        """Pontryagin difference ``self (-) disturbance`` by identity (3).

        Exact: each facet offset is reduced by the support function of the
        disturbance set in that facet's normal direction. The result may be
        empty, which :meth:`is_empty` detects.
        """
        if disturbance.dim != self.dim:
            raise ValueError(f"dimension mismatch: {self.dim} and {disturbance.dim}")
        offsets = disturbance.support_many(self.A)
        if not np.all(np.isfinite(offsets)):
            raise ValueError("disturbance set is unbounded; erosion is not defined")
        return Polytope(self.A.copy(), self.b - offsets)

    def preimage(self, linear_map: np.ndarray) -> Polytope:
        """``{x : M x in self}`` by identity (4)."""
        m = np.asarray(linear_map, dtype=float)
        if m.ndim != 2 or m.shape[0] != self.dim:
            raise ValueError(f"linear_map must have {self.dim} rows, got shape {m.shape}")
        return Polytope(self.A @ m, self.b.copy())

    def scaled(self, factor: float) -> Polytope:
        """``factor * self`` for ``factor > 0`` (scaling about the origin)."""
        f = float(factor)
        if not np.isfinite(f) or f <= 0.0:
            raise ValueError(f"factor must be finite and positive, got {factor}")
        return Polytope(self.A.copy(), self.b * f)

    # -------------------------------------------------------- emptiness
    def chebyshev_radius(self) -> float:
        """Signed radius of the largest inscribed ball; negative when empty.

        Solves ``max r`` subject to ``a_i^T x + ||a_i|| r <= b_i`` with ``r``
        free, which is the standard linear programme for the Chebyshev centre of
        a polyhedron (Boyd and Vandenberghe, *Convex Optimization*, CUP, 2004,
        section 8.5.1). Leaving ``r`` free rather than bounding it below at zero
        is deliberate: the programme is then always feasible and the optimum is
        negative exactly when the polytope is empty, so the magnitude says how
        infeasible it is instead of only that it is. ``+inf`` is returned when
        the polytope contains arbitrarily large balls.
        """
        a, rhs = unit_normalise(self.A, self.b)
        n = self.dim
        cost = np.concatenate([np.zeros(n), [-1.0]])
        a_ub = np.column_stack([a, np.ones(a.shape[0])])
        res = linprog(
            cost,
            A_ub=a_ub,
            b_ub=rhs,
            bounds=[(None, None)] * n + [(None, None)],
            method=_LP_METHOD,
        )
        if res.status == 2:
            return -np.inf
        if res.status == 3:
            return np.inf
        if not res.success:  # pragma: no cover - solver pathology
            raise RuntimeError(f"Chebyshev LP failed: {res.message}")
        return float(-res.fun)

    def is_empty(self, tol: float = 1e-9) -> bool:
        """True if the polytope has no point with a ball of radius ``tol``."""
        return self.chebyshev_radius() <= tol

    def contains_polytope(self, other: Polytope, tol: float = 1e-9) -> bool:
        """True if ``other`` is a subset of ``self``, by one LP per facet."""
        if other.dim != self.dim:
            raise ValueError(f"dimension mismatch: {self.dim} and {other.dim}")
        for row, offset in zip(self.A, self.b, strict=True):
            s = other.support(row)
            if s == -np.inf:  # other is empty, so it is a subset
                return True
            if s > offset + tol:
                return False
        return True

    # ------------------------------------------------------- reduction
    def minimal(self, tol: float = 1e-10) -> Polytope:
        """Remove redundant halfspaces, one LP per row.

        A row ``j`` is redundant when the support of the polytope defined by the
        remaining kept rows, in direction ``a_j``, is already at most ``b_j``.
        Rows are tested in order against the current kept set, so an exactly
        duplicated row loses exactly one of its copies.
        """
        a, rhs = unit_normalise(self.A, self.b)
        keep = np.ones(a.shape[0], dtype=bool)
        for j in range(a.shape[0]):
            trial = keep.copy()
            trial[j] = False
            if not trial.any():
                continue
            s = Polytope(a[trial], rhs[trial]).support(a[j])
            if s == -np.inf:
                # Remaining rows are already infeasible; keep this row and stop
                # reducing, so emptiness stays visible.
                break
            if s <= rhs[j] + tol:
                keep[j] = False
        return Polytope(a[keep], rhs[keep])

    # ------------------------------------------------------ 2-D helpers
    def vertices_2d(self, tol: float = 1e-8) -> np.ndarray:
        """Counter-clockwise vertices, for plotting only. Requires ``dim == 2``.

        Enumerates all ``C(m, 2)`` facet intersections and keeps the feasible
        ones; this is ``O(m^2)`` and is not intended for anything but figures.
        """
        if self.dim != 2:
            raise ValueError(f"vertices_2d requires dim == 2, got {self.dim}")
        found: list[np.ndarray] = []
        for i, j in itertools.combinations(range(self.n_halfspaces), 2):
            m = np.array([self.A[i], self.A[j]])
            det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
            if abs(det) < 1e-12:
                continue
            v = np.linalg.solve(m, np.array([self.b[i], self.b[j]]))
            if np.all(self.A @ v - self.b <= tol):
                found.append(v)
        if len(found) < 3:
            return np.zeros((0, 2))
        pts = np.array(found)
        centre = pts.mean(axis=0)
        order = np.argsort(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]))
        pts = pts[order]
        kept = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p - kept[-1]) > 1e-9:
                kept.append(p)
        if len(kept) > 1 and np.linalg.norm(kept[0] - kept[-1]) <= 1e-9:
            kept.pop()
        return np.array(kept)

    def area_2d(self) -> float:
        """Area by the shoelace formula on :meth:`vertices_2d`. ``dim == 2``."""
        v = self.vertices_2d()
        if v.shape[0] < 3:
            return 0.0
        x, y = v[:, 0], v[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


class Box(Polytope):
    """An axis-aligned box ``{x : lower <= x <= upper}``.

    Carries the closed-form support function (2), so the erosion in
    :meth:`Polytope.erode` and the switching condition in
    :mod:`simplexguard.guard` cost one matrix product rather than one linear
    programme per facet. ``tests/test_properties.py`` checks the closed form
    against the generic LP path.
    """

    def __init__(self, lower: np.ndarray, upper: np.ndarray) -> None:
        lo = np.asarray(lower, dtype=float).ravel()
        hi = np.asarray(upper, dtype=float).ravel()
        if lo.shape != hi.shape:
            raise ValueError(f"lower has shape {lo.shape}, upper has shape {hi.shape}")
        if lo.size == 0:
            raise ValueError("a box needs at least one dimension")
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            raise ValueError("box bounds must be finite")
        if np.any(hi < lo):
            raise ValueError("box upper bound is below lower bound in some coordinate")
        n = lo.size
        a = np.vstack([np.eye(n), -np.eye(n)])
        rhs = np.concatenate([hi, -lo])
        super().__init__(a, rhs)
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)

    @property
    def centre(self) -> np.ndarray:
        """Box centre ``(lower + upper) / 2``."""
        return 0.5 * (self.lower + self.upper)

    @property
    def half_widths(self) -> np.ndarray:
        """Box half-widths ``(upper - lower) / 2``, non-negative."""
        return 0.5 * (self.upper - self.lower)

    def support(self, direction: np.ndarray) -> float:
        """Closed form (2). No linear programme."""
        c = np.asarray(direction, dtype=float).ravel()
        if c.shape[0] != self.dim:
            raise ValueError(f"direction has length {c.shape[0]}, expected {self.dim}")
        if not np.all(np.isfinite(c)):
            raise ValueError("direction contains non-finite entries")
        return float(c @ self.centre + np.abs(c) @ self.half_widths)

    def support_many(self, directions: np.ndarray) -> np.ndarray:
        """Vectorised closed form (2): one matrix product for all rows."""
        d = np.atleast_2d(np.asarray(directions, dtype=float))
        if d.shape[1] != self.dim:
            raise ValueError(f"directions have width {d.shape[1]}, expected {self.dim}")
        if not np.all(np.isfinite(d)):
            raise ValueError("directions contain non-finite entries")
        return d @ self.centre + np.abs(d) @ self.half_widths

    def scaled(self, factor: float) -> Box:
        """Scale the half-widths about the box centre by ``factor >= 0``."""
        f = float(factor)
        if not np.isfinite(f) or f < 0.0:
            raise ValueError(f"factor must be finite and non-negative, got {factor}")
        c, r = self.centre, self.half_widths * f
        return Box(c - r, c + r)

    def vertices(self) -> np.ndarray:
        """All ``2**n`` box vertices. Only call this for small ``n``."""
        n = self.dim
        if n > 16:
            raise ValueError(f"refusing to enumerate 2**{n} vertices")
        signs = np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)
        return self.lower + signs * (self.upper - self.lower)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"Box(lower={self.lower.tolist()}, upper={self.upper.tolist()})"


def box(half_widths: np.ndarray) -> Box:
    """A box centred at the origin with the given non-negative half-widths."""
    r = np.asarray(half_widths, dtype=float).ravel()
    if np.any(r < 0.0):
        raise ValueError("half-widths must be non-negative")
    return Box(-r, r)
