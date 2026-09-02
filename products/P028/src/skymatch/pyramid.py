"""Pyramid matching: the fourth star that turns a triangle into an identification.

A matched triangle is not an identification. Three inter-star angles inside a
tolerance window can be matched by an unrelated catalogue triple, and the
rate at which that happens is the whole difficulty of the problem: it rises
with the tolerance, with the catalogue density, and with the number of false
detections in the frame. Mortari, Samaan, Bruccoleri and Junkins' Pyramid
algorithm (*Navigation* 51(3), 171-183, 2004) resolves it by requiring a
fourth observed star to be consistent with the same catalogue triple:

.. math::
    |\\theta(a,d) - t_{ir}| \\le \\tau,\\quad
    |\\theta(b,d) - t_{jr}| \\le \\tau,\\quad
    |\\theta(c,d) - t_{kr}| \\le \\tau                              (Eq. Y1)

for some catalogue star ``d``. The fourth star adds three more angle
constraints against one more unknown, so the probability that a coincidental
triangle also passes is far smaller than the probability of the triangle
itself; that is the published argument for the algorithm, and Section 4 of
``validation/VALIDATION.md`` measures it here rather than assuming it.

The second thing the published algorithm contributes is the *order* the
observed triples are tried in. If some observed spots are false, a naive
``for i < j < k`` loop retries triples containing the same bad spot many times
before moving on. The Pyramid scan indexes triples by their gaps,

.. code-block:: text

    for dj in 1 .. n-2:
        for dk in 1 .. n-1-dj:
            for i in 0 .. n-1-dj-dk:
                j = i + dj;  k = j + dk

so consecutive attempts change which spots are involved as fast as possible.
:func:`pyramid_triple_order` implements it, and
``tests/test_pyramid.py`` checks it enumerates every triple exactly once.

Units: angles in radians throughout.
"""

from __future__ import annotations

import numpy as np

from .pairtable import PairTable

__all__ = ["confirm_with_fourth_star", "pyramid_triple_order"]


def pyramid_triple_order(n_stars: int) -> list[tuple[int, int, int]]:
    """The Pyramid gap-ordered scan over observed triples.

    Returns every ``(i, j, k)`` with ``0 <= i < j < k < n_stars`` exactly once,
    ordered so that successive attempts share as few spots as possible.
    Raises ``ValueError`` for ``n_stars < 3``.
    """
    if n_stars < 3:
        raise ValueError(f"need at least 3 stars to form a triple, got {n_stars}")
    triples: list[tuple[int, int, int]] = []
    for dj in range(1, n_stars - 1):
        for dk in range(1, n_stars - dj):
            for i in range(0, n_stars - dj - dk):
                triples.append((i, i + dj, i + dj + dk))
    return triples


def confirm_with_fourth_star(
    table: PairTable,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    t_ar: float | np.ndarray,
    t_br: float | np.ndarray,
    t_cr: float | np.ndarray,
    tolerance_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eq. Y1, vectorised over a whole array of candidate triangles.

    Parameters
    ----------
    table
        The catalogue pair table.
    a, b, c
        ``(m,)`` candidate catalogue triples.
    t_ar, t_br, t_cr
        Observed angles [rad] from the fourth spot to the first, second and
        third spots of the triple. Scalars, or ``(m,)`` arrays so that several
        fourth spots can be tested against several triangles in one call --
        which is how :func:`skymatch.identify.gather_candidates` uses it, and
        the reason the whole confirmation stage of a frame costs one join
        rather than one per fourth spot.
    tolerance_rad
        Match half-window [rad].

    Returns ``(n_matches, star_d, residual_rms)``, each ``(m,)``:

    * ``n_matches`` -- how many catalogue stars satisfy Eq. Y1 for that
      candidate. Zero means the fourth spot refutes it; more than one means
      the fourth spot does not settle it, which the Pyramid rule treats as
      *not* a confirmation rather than as a match;
    * ``star_d`` -- the matching catalogue index where ``n_matches == 1``,
      else -1;
    * ``residual_rms`` -- root-mean-square of the three residuals of Eq. Y1
      for that match [rad], ``nan`` where there is no unique match.
    """
    if tolerance_rad <= 0.0:
        raise ValueError(f"tolerance_rad must be > 0, got {tolerance_rad}")
    av = np.asarray(a, dtype=np.int64).reshape(-1)
    bv = np.asarray(b, dtype=np.int64).reshape(-1)
    cv = np.asarray(c, dtype=np.int64).reshape(-1)
    if not (av.shape == bv.shape == cv.shape):
        raise ValueError(
            f"a, b, c must have the same shape, got {av.shape}, {bv.shape}, {cv.shape}"
        )
    m = av.size
    n_matches = np.zeros(m, dtype=np.int64)
    star_d = np.full(m, -1, dtype=np.int64)
    residual_rms = np.full(m, np.nan)
    if m == 0:
        return n_matches, star_d, residual_rms
    tar = np.broadcast_to(np.asarray(t_ar, dtype=float), av.shape)
    tbr = np.broadcast_to(np.asarray(t_br, dtype=float), av.shape)
    tcr = np.broadcast_to(np.asarray(t_cr, dtype=float), av.shape)

    rows, d = table.neighbours_range(av, tar - tolerance_rad, tar + tolerance_rad)
    if rows.size == 0:
        return n_matches, star_d, residual_rms

    theta_ad = table.separation_lookup(av[rows], d)
    theta_bd = table.separation_lookup(bv[rows], d)
    theta_cd = table.separation_lookup(cv[rows], d)
    ok = (
        (d != av[rows])
        & (d != bv[rows])
        & (d != cv[rows])
        & np.isfinite(theta_bd)
        & np.isfinite(theta_cd)
        & (np.abs(theta_bd - tbr[rows]) <= tolerance_rad)
        & (np.abs(theta_cd - tcr[rows]) <= tolerance_rad)
    )
    rows, d = rows[ok], d[ok]
    if rows.size == 0:
        return n_matches, star_d, residual_rms
    res = np.sqrt(
        (
            (theta_ad[ok] - tar[rows]) ** 2
            + (theta_bd[ok] - tbr[rows]) ** 2
            + (theta_cd[ok] - tcr[rows]) ** 2
        )
        / 3.0
    )

    n_matches = np.bincount(rows, minlength=m).astype(np.int64)
    unique = n_matches == 1
    if np.any(unique):
        # For rows with exactly one match, the first (and only) hit is the one.
        first = np.full(m, -1, dtype=np.int64)
        order = np.argsort(rows, kind="stable")
        rows_sorted = rows[order]
        starts = np.searchsorted(rows_sorted, np.arange(m), side="left")
        has = n_matches > 0
        first[has] = order[starts[has]]
        sel = unique & has
        star_d[sel] = d[first[sel]]
        residual_rms[sel] = res[first[sel]]
    return n_matches, star_d, residual_rms
