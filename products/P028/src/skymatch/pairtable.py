"""The pair table: every catalogue star pair inside the field, indexed for range search.

Every feature-based star identification algorithm reduces to the same primitive:
*given an inter-star angle and a tolerance, list the catalogue pairs whose
separation falls in that window*. Mortari and Neta's k-vector (AAS 00-128,
2000) is the classical answer, an index built so the search is O(1) rather than
O(log n). This package uses the simpler sorted array plus binary search:
``numpy.searchsorted`` on the sorted separations, which is O(log n) per query
and, in NumPy, faster in practice at these table sizes than a Python-level
k-vector because the whole query vectorises. The k-vector's advantage is
constant-time bounds on a non-uniform distribution; nothing here needs that.

Three indexes are built over the same pairs, because the three queries the
matcher makes are different:

1. ``theta`` sorted ascending, with ``star_a`` / ``star_b`` alongside
   -- :meth:`PairTable.ordered_range`, "which pairs are this far apart?"
2. a per-star adjacency sorted by ``(star, theta)``
   -- :meth:`PairTable.neighbours_range`, "which stars are this far from star s?"
3. the pairs keyed by ``min(a,b) * N + max(a,b)``, sorted
   -- :meth:`PairTable.separation_lookup`, "how far apart are these two stars?"

Index 2 is searched with a composite key ``10 * star + theta``, which is
monotone because ``theta < pi < 10``, so a per-star range search becomes one
vectorised ``searchsorted`` over the whole array instead of a Python loop.
Float64 resolves that key to about ``3e-11`` rad for catalogues up to ``1e5``
stars, five orders below the smallest tolerance the package uses.

Size: the number of pairs closer than ``theta_max`` for ``N`` uniformly
distributed stars is

.. math:: P = \\binom{N}{2}\\,\\frac{1 - \\cos\\theta_{\\max}}{2}       (Eq. P1)

so the table is quadratic in catalogue size and the memory cost of raising the
magnitude limit is about ``10**(2*0.52) = 11x`` per magnitude. This is the real
constraint on a star-tracker catalogue, and it is measured in
``validation/validate_catalogue.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .catalogue import StarCatalogue
from .geometry import angular_separation

__all__ = ["PairTable", "expected_pair_count"]

_STAR_KEY_STRIDE = 10.0  # > pi, so 10*star + theta is monotone in (star, theta)


def expected_pair_count(n_stars: int, max_separation_rad: float) -> float:
    """Eq. P1: expected pair count for ``n_stars`` uniform stars within an angle."""
    if n_stars < 0:
        raise ValueError(f"n_stars must be >= 0, got {n_stars}")
    if not 0.0 <= max_separation_rad <= np.pi:
        raise ValueError(f"max_separation_rad must be in [0, pi], got {max_separation_rad}")
    return 0.5 * n_stars * (n_stars - 1) * 0.5 * (1.0 - np.cos(max_separation_rad))


class PairTable:
    """Range-searchable index of catalogue star pairs within ``max_separation_rad``.

    Build once per (catalogue, field of view); it is read-only afterwards.

    Parameters
    ----------
    catalogue
        The prepared catalogue.
    max_separation_rad
        Largest pair separation to store [rad]. Use
        ``CameraModel.max_separation_rad``; a smaller value silently makes
        wide star pairs unmatchable.
    """

    def __init__(self, catalogue: StarCatalogue, max_separation_rad: float) -> None:
        if not np.isfinite(max_separation_rad) or not 0.0 < max_separation_rad <= np.pi:
            raise ValueError(
                f"max_separation_rad must be in (0, pi], got {max_separation_rad}"
            )
        if catalogue.n_stars < 2:
            raise ValueError(f"catalogue needs at least 2 stars, got {catalogue.n_stars}")

        self.catalogue = catalogue
        self.max_separation_rad = float(max_separation_rad)
        n = catalogue.n_stars

        chord = 2.0 * np.sin(0.5 * self.max_separation_rad)
        tree = cKDTree(catalogue.vectors)
        pairs = tree.query_pairs(chord, output_type="ndarray")
        if pairs.size == 0:
            raise ValueError(
                f"no star pairs closer than {np.degrees(self.max_separation_rad):.3f} deg "
                f"in a catalogue of {n} stars; the field is too small or the catalogue too sparse"
            )
        lo = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
        hi = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
        theta = angular_separation(catalogue.vectors[lo], catalogue.vectors[hi])

        order = np.argsort(theta, kind="stable")
        self._theta = np.ascontiguousarray(theta[order])
        self._star_a = np.ascontiguousarray(lo[order])
        self._star_b = np.ascontiguousarray(hi[order])

        # Index 2: adjacency, sorted by (star, theta), searched by composite key.
        adj_star = np.concatenate([self._star_a, self._star_b])
        adj_other = np.concatenate([self._star_b, self._star_a])
        adj_theta = np.concatenate([self._theta, self._theta])
        adj_order = np.lexsort((adj_theta, adj_star))
        self._adj_other = np.ascontiguousarray(adj_other[adj_order])
        self._adj_key = np.ascontiguousarray(
            adj_star[adj_order].astype(np.float64) * _STAR_KEY_STRIDE + adj_theta[adj_order]
        )

        # Index 3: pair -> separation, keyed on the ordered index pair.
        keys = self._star_a * n + self._star_b
        key_order = np.argsort(keys, kind="stable")
        self._pair_key = np.ascontiguousarray(keys[key_order])
        self._pair_key_theta = np.ascontiguousarray(self._theta[key_order])
        self._n_stars = n

    # -- properties -------------------------------------------------------

    @property
    def n_pairs(self) -> int:
        """Number of stored pairs."""
        return int(self._theta.shape[0])

    @property
    def n_stars(self) -> int:
        """Number of stars in the underlying catalogue."""
        return self._n_stars

    @property
    def separations(self) -> np.ndarray:
        """Read-only view of the sorted pair separations [rad]."""
        view = self._theta.view()
        view.flags.writeable = False
        return view

    @property
    def nbytes(self) -> int:
        """Total bytes held by the three indexes."""
        arrays = (
            self._theta,
            self._star_a,
            self._star_b,
            self._adj_other,
            self._adj_key,
            self._pair_key,
            self._pair_key_theta,
        )
        return int(sum(a.nbytes for a in arrays))

    # -- queries ----------------------------------------------------------

    def ordered_range(self, lo_rad: float, hi_rad: float) -> tuple[np.ndarray, np.ndarray]:
        """Pairs with separation in ``[lo_rad, hi_rad]``, both orientations.

        Returns ``(a, b)`` index arrays of equal length ``2 * m``: the first
        ``m`` are the stored orientation and the second ``m`` are reversed, so
        a caller matching an ordered observed pair need not consider order.
        """
        if hi_rad < lo_rad:
            raise ValueError(f"hi_rad ({hi_rad}) must be >= lo_rad ({lo_rad})")
        i0 = int(np.searchsorted(self._theta, lo_rad, side="left"))
        i1 = int(np.searchsorted(self._theta, hi_rad, side="right"))
        a = self._star_a[i0:i1]
        b = self._star_b[i0:i1]
        return np.concatenate([a, b]), np.concatenate([b, a])

    def neighbours_range(
        self,
        stars: np.ndarray,
        lo_rad: float | np.ndarray,
        hi_rad: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """For each entry of ``stars``, its neighbours at separation in ``[lo, hi]``.

        Vectorised over ``stars``; one ``searchsorted`` pair for the whole
        array. ``lo_rad`` and ``hi_rad`` may be scalars or arrays broadcast
        against ``stars``, so a batch of different queries costs one call.
        Returns ``(rows, neighbours)``, where ``rows`` indexes back into
        ``stars`` and ``neighbours`` is the catalogue index of the neighbour,
        so ``stars[rows[k]]`` and ``neighbours[k]`` are a matching pair.
        """
        s = np.asarray(stars, dtype=np.int64).reshape(-1)
        if s.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        if np.any(s < 0) or np.any(s >= self._n_stars):
            raise ValueError("star index out of range for this catalogue")
        lo = np.broadcast_to(np.asarray(lo_rad, dtype=float), s.shape)
        hi = np.broadcast_to(np.asarray(hi_rad, dtype=float), s.shape)
        if np.any(hi < lo):
            raise ValueError("hi_rad must be >= lo_rad elementwise")
        base = s.astype(np.float64) * _STAR_KEY_STRIDE
        start = np.searchsorted(self._adj_key, base + np.maximum(lo, 0.0), side="left")
        stop = np.searchsorted(self._adj_key, base + hi, side="right")
        counts = np.maximum(stop - start, 0)
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        rows = np.repeat(np.arange(s.size, dtype=np.int64), counts)
        offsets = np.arange(total, dtype=np.int64) - np.repeat(
            np.cumsum(counts) - counts, counts
        )
        idx = np.repeat(start, counts) + offsets
        return rows, self._adj_other[idx]

    def separation_lookup(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Separation [rad] of each ``(a, b)`` pair, or ``nan`` if not in the table.

        ``nan`` means "these two stars are further apart than
        ``max_separation_rad``", which is the answer the matcher needs: such a
        pair can never both be in the field.
        """
        av = np.asarray(a, dtype=np.int64).reshape(-1)
        bv = np.asarray(b, dtype=np.int64).reshape(-1)
        if av.shape != bv.shape:
            raise ValueError(f"a {av.shape} and b {bv.shape} must have the same shape")
        if av.size == 0:
            return np.empty(0, dtype=float)
        if np.any(av < 0) or np.any(av >= self._n_stars):
            raise ValueError("star index out of range for this catalogue")
        if np.any(bv < 0) or np.any(bv >= self._n_stars):
            raise ValueError("star index out of range for this catalogue")
        key = np.minimum(av, bv) * self._n_stars + np.maximum(av, bv)
        idx = np.searchsorted(self._pair_key, key)
        clipped = np.clip(idx, 0, self._pair_key.shape[0] - 1)
        hit = (idx < self._pair_key.shape[0]) & (self._pair_key[clipped] == key) & (av != bv)
        return np.where(hit, self._pair_key_theta[clipped], np.nan)
