"""Triangle matching: three observed stars against the catalogue pair table.

Given three observed directions with inter-star angles
``(t_ij, t_ik, t_jk)`` and an angular tolerance, find every catalogue triple
``(a, b, c)`` whose three separations match, so that ``a`` corresponds to the
first observed star, ``b`` to the second and ``c`` to the third:

.. math::
    |\\theta(a,b) - t_{ij}| \\le \\tau,\\quad
    |\\theta(a,c) - t_{ik}| \\le \\tau,\\quad
    |\\theta(b,c) - t_{jk}| \\le \\tau                              (Eq. T1)

Inter-star angles are invariant under rotation, which is what makes them
usable with no attitude prior at all -- the defining property of the
lost-in-space problem (Padgett & Kreutz-Delgado 1997; Spratling & Mortari
2009 survey).

The search is a three-way relational join over the pair table, done with array
operations rather than a Python loop:

1. range-search the sorted separations for pairs at ``t_ij`` -- candidates for
   ``(a, b)``, in both orientations;
2. for each such ``a``, range-search its adjacency at ``t_ik`` -- candidates
   for ``c``;
3. look up ``theta(b, c)`` directly and keep the rows that match ``t_jk``.

Steps 1 and 2 grow with the tolerance; the third is a lookup and does not.

Tolerance
---------
A pair separation is a difference of two measured directions, so if each
direction carries a per-axis centroid error ``sigma``, the separation error is
``sqrt(2) * sigma`` to first order in the transverse component along the
connecting arc. :func:`separation_tolerance` returns ``k * sqrt(2) * sigma``
with ``k = 3`` by default, a three-sigma gate. Setting ``k`` too small loses
true matches; too large multiplies the false ones, and Eq. T2 in
``validation/validate_identification.py`` measures both sides of that trade.
"""

from __future__ import annotations

import numpy as np

from .geometry import ARCSEC
from .pairtable import PairTable

__all__ = ["separation_tolerance", "triangle_candidates", "triangle_edge_angles"]


def separation_tolerance(centroid_sigma_arcsec: float, k_sigma: float = 3.0) -> float:
    """Angular match tolerance [rad] for a per-axis centroid sigma [arcsec].

    ``tau = k * sqrt(2) * sigma``. Returns a small positive floor (1e-9 rad)
    for zero noise, so that noise-free matching still tolerates round-off.
    """
    if centroid_sigma_arcsec < 0.0:
        raise ValueError(
            f"centroid_sigma_arcsec must be >= 0, got {centroid_sigma_arcsec}"
        )
    if k_sigma <= 0.0:
        raise ValueError(f"k_sigma must be > 0, got {k_sigma}")
    return max(k_sigma * np.sqrt(2.0) * centroid_sigma_arcsec * ARCSEC, 1e-9)


def triangle_edge_angles(
    vectors: np.ndarray, i: int, j: int, k: int
) -> tuple[float, float, float]:
    """The three inter-star angles ``(t_ij, t_ik, t_jk)`` [rad] of an observed triple."""
    v = np.asarray(vectors, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vectors must have shape (N, 3), got {v.shape}")
    for name, idx in (("i", i), ("j", j), ("k", k)):
        if not 0 <= idx < v.shape[0]:
            raise ValueError(f"index {name}={idx} out of range for {v.shape[0]} vectors")
    if len({i, j, k}) != 3:
        raise ValueError(f"i, j, k must be distinct, got ({i}, {j}, {k})")

    def sep(p: int, q: int) -> float:
        c = np.linalg.norm(np.cross(v[p], v[q]))
        return float(np.arctan2(c, float(v[p] @ v[q])))

    return sep(i, j), sep(i, k), sep(j, k)


def triangle_candidates(
    table: PairTable,
    t_ij: float,
    t_ik: float,
    t_jk: float,
    tolerance_rad: float,
    max_candidates: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Eq. T1. Catalogue triples matching three observed inter-star angles.

    Parameters
    ----------
    table
        The catalogue pair table.
    t_ij, t_ik, t_jk
        Observed inter-star angles [rad], in the order
        (first-second, first-third, second-third).
    tolerance_rad
        Half-width of the match window [rad].
    max_candidates
        If given and the join produces more than this many triples, the
        arrays are truncated. The full count is still returned, so a caller
        can tell "ambiguous" from "one match".

    Returns ``(a, b, c, residuals)`` where ``residuals`` is ``(m, 3)`` of
    signed ``theta_catalogue - theta_observed`` [rad] for the three edges in
    the same order as the inputs. ``m`` is the number returned, which may be
    less than the number found if ``max_candidates`` truncated it.
    """
    if tolerance_rad <= 0.0:
        raise ValueError(f"tolerance_rad must be > 0, got {tolerance_rad}")
    for name, value in (("t_ij", t_ij), ("t_ik", t_ik), ("t_jk", t_jk)):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0, got {value}")
    empty_i = np.empty(0, dtype=np.int64)
    if max_candidates is not None and max_candidates < 0:
        raise ValueError(f"max_candidates must be >= 0, got {max_candidates}")

    a, b = table.ordered_range(t_ij - tolerance_rad, t_ij + tolerance_rad)
    if a.size == 0:
        return empty_i, empty_i, empty_i, np.empty((0, 3))
    theta_ab = table.separation_lookup(a, b)

    rows, c = table.neighbours_range(a, t_ik - tolerance_rad, t_ik + tolerance_rad)
    if rows.size == 0:
        return empty_i, empty_i, empty_i, np.empty((0, 3))
    aa, bb = a[rows], b[rows]
    theta_ac = table.separation_lookup(aa, c)
    theta_bc = table.separation_lookup(bb, c)

    keep = (
        (c != aa)
        & (c != bb)
        & np.isfinite(theta_bc)
        & (np.abs(theta_bc - t_jk) <= tolerance_rad)
    )
    aa, bb, cc = aa[keep], bb[keep], c[keep]
    res = np.stack(
        [theta_ab[rows][keep] - t_ij, theta_ac[keep] - t_ik, theta_bc[keep] - t_jk], axis=1
    )
    if max_candidates is not None and aa.size > max_candidates:
        cut = int(max_candidates)
        aa, bb, cc, res = aa[:cut], bb[:cut], cc[:cut], res[:cut]
    return aa, bb, cc, res
