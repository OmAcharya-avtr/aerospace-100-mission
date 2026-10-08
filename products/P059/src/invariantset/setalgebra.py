"""Minkowski sum, Pontryagin difference and intersection of polytopes.

Definitions, for compact convex `P, Q` in `R^n`:

    Minkowski sum        P (+) Q = {p + q : p in P, q in Q}                (3)
    Pontryagin diff.     P (-) Q = {x in R^n : x + Q subset P}             (4)
    intersection         P & Q   = {x : x in P and x in Q}                 (5)

Identities this module relies on, each with a source:

(I1) `h_{P (+) Q}(c) = h_P(c) + h_Q(c)` for all `c`.
     Schneider, *Convex Bodies: the Brunn-Minkowski Theory*, Cambridge
     University Press, 1993 (Minkowski additivity of the support function).
     This is exact and is how (3) is checked here without ever enumerating
     the sum.

(I2) If `P = {x : A x <= b}` then `P (-) Q = {x : a_i^T x <= b_i - h_Q(a_i)}`.
     Kolmanovsky and Gilbert, "Theory and computation of disturbance invariant
     sets for discrete-time linear systems", *Mathematical Problems in
     Engineering* 4(4), 1998, pp. 317-367.  Exact in H-representation, which
     is why (4) is cheap and (3) is not.

(I3) `(P (-) Q) (+) Q  subset  P`, with equality only in special cases.
(I4) `(P (+) Q) (-) Q  =  P` for compact convex `P, Q`.

     (I3) and (I4) are the asymmetry that matters.  **The Pontryagin
     difference is not an inverse of the Minkowski sum.**  Eroding then
     dilating loses material and gives a subset, (I3); dilating then eroding
     recovers the original set exactly when it is convex, (I4).  Both are
     stated in Kolmanovsky and Gilbert 1998 and in Blanchini, "Set invariance
     in control - a survey", *Automatica* 35(11), 1999, pp. 1747-1768.  The identity
     users assume, `(P (-) Q) (+) Q = P`, is false; `tests/test_properties.py`
     tests (I3) and (I4) and `examples/set_algebra_gap.py` measures the area
     that (I3) loses.

Units: every function here is unit-agnostic and all three operations require
both arguments in the same state units and the same ambient dimension.

Validity range: compact convex polytopes.  `minkowski_sum` additionally needs
both arguments bounded and full-dimensional enough for Qhull in `n >= 2`; the
Pontryagin difference needs only `Q` bounded.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .polytope import (
    ZERO_ROW_TOL,
    EmptyPolytopeError,
    Polytope,
    UnboundedDirectionError,
    VertexEnumerationError,
)

__all__ = [
    "intersect",
    "is_subset",
    "minkowski_sum",
    "pontryagin_difference",
    "support_gap",
]


def intersect(P: Polytope, Q: Polytope) -> Polytope:
    """Intersection (5), by stacking halfspaces.  Exact and allocation-only.

    Raises
    ------
    ValueError
        If the dimensions differ.
    """
    if P.dim != Q.dim:
        raise ValueError(f"dimension mismatch: {P.dim} and {Q.dim}")
    return Polytope(np.vstack([P.A, Q.A]), np.concatenate([P.b, Q.b]))


def pontryagin_difference(P: Polytope, Q: Polytope) -> Polytope:
    """Pontryagin difference (4) via identity (I2).  Exact in H-form.

    Parameters
    ----------
    P, Q : Polytope
        Same ambient dimension and same state units.  `Q` must be bounded
        along every row normal of `P`.

    Returns
    -------
    Polytope
        Same row normals as `P`, right-hand sides `b_i - h_Q(a_i)`.  The
        result may be empty, which the caller detects with
        :meth:`Polytope.is_empty`; this function does not raise on emptiness
        because an empty robust-invariant set is a legitimate answer.

    Raises
    ------
    ValueError
        If the dimensions differ.
    UnboundedDirectionError
        If `Q` is unbounded along one of `P`'s row normals, in which case (4)
        is empty but for a reason the caller should see rather than silently
        receive as an empty set.
    """
    if P.dim != Q.dim:
        raise ValueError(f"dimension mismatch: {P.dim} and {Q.dim}")
    hQ = Q.support_many(P.A)
    return Polytope(P.A, P.b - hQ)


def minkowski_sum(P: Polytope, Q: Polytope, tol: float = 1e-9) -> Polytope:
    """Minkowski sum (3), by vertex enumeration and convex hull.

    There is no exact halfspace formula for (3): the sum of two
    H-representations requires the V-representation of both.  Vertices of `P`
    and `Q` are enumerated (`Polytope.vertices`, brute force), summed
    pairwise, and re-hulled.  Cost is `|V(P)| * |V(Q)|` points into Qhull, and
    the vertex enumeration itself is the expensive half.

    Parameters
    ----------
    P, Q : Polytope
        Bounded, same dimension, same units.
    tol : float
        Vertex enumeration and deduplication tolerance, in state units.

    Returns
    -------
    Polytope
        H-representation of the sum, from the Qhull facets, with rows
        normalised to unit normals.

    Raises
    ------
    ValueError
        If dimensions differ.
    VertexEnumerationError
        If either argument is unbounded or too large to enumerate, or if the
        summed point set is lower-dimensional than `n` so that Qhull cannot
        produce facets.
    """
    if P.dim != Q.dim:
        raise ValueError(f"dimension mismatch: {P.dim} and {Q.dim}")
    VP = P.vertices(tol=tol)
    VQ = Q.vertices(tol=tol)
    if VP.shape[0] == 0 or VQ.shape[0] == 0:
        raise EmptyPolytopeError("Minkowski sum with an empty polytope is empty")
    S = (VP[:, None, :] + VQ[None, :, :]).reshape(-1, P.dim)
    if P.dim == 1:
        lo = float(S.min())
        hi = float(S.max())
        return Polytope.from_bounds([lo], [hi])
    try:
        hull = ConvexHull(S)
    except QhullError as exc:
        raise VertexEnumerationError(
            "Qhull could not hull the summed vertices; the sum is probably "
            f"lower-dimensional than R^{P.dim} ({exc})"
        ) from exc
    # Qhull reports facets as A x + b0 <= 0, so A x <= -b0.
    A = np.asarray(hull.equations[:, :-1], dtype=float)
    b = -np.asarray(hull.equations[:, -1], dtype=float)
    norms = np.linalg.norm(A, axis=1)
    keep = norms > ZERO_ROW_TOL
    return Polytope(A[keep] / norms[keep, None], b[keep] / norms[keep])


def support_gap(P: Polytope, Q: Polytope, directions: np.ndarray) -> np.ndarray:
    """`h_Q(c) - h_P(c)` for each row `c` of `directions`.

    Non-positive everywhere is the support-function certificate for
    `Q subset P` restricted to the sampled directions; it is exact only when
    the directions include every facet normal of `P`, which
    :func:`is_subset` arranges.
    """
    return Q.support_many(directions) - P.support_many(directions)


def is_subset(Q: Polytope, P: Polytope, tol: float = 1e-9) -> tuple[bool, float]:
    """Exact test for `Q subset P`, with the worst margin.

    `Q subset P` if and only if `h_Q(a_i) <= b_i` for every row `(a_i, b_i)`
    of `P`.  This is exact, not sampled: a polytope is the intersection of its
    own halfspaces, so testing `P`'s facet normals is sufficient (Rockafellar
    1970).

    Returns
    -------
    (bool, float)
        Whether containment holds to `tol`, and `max_i (h_Q(a_i) - b_i)`,
        in `b` units.  A negative margin is slack; a positive margin is the
        amount by which `Q` sticks out of `P`.
    """
    if P.dim != Q.dim:
        raise ValueError(f"dimension mismatch: {P.dim} and {Q.dim}")
    if tol < 0:
        raise ValueError(f"tol must be non-negative, got {tol!r}")
    try:
        margins = Q.support_many(P.A) - P.b
    except EmptyPolytopeError:
        return True, -np.inf  # the empty set is a subset of everything
    except UnboundedDirectionError:
        return False, np.inf
    worst = float(np.max(margins))
    return bool(worst <= tol), worst
