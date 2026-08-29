"""Subaperture layout and the Southwell / Fried slope-to-phase geometry matrices.

What a "geometry" is
--------------------
A Shack-Hartmann sensor measures two average wavefront gradients per
subaperture. A *reconstruction geometry* is the choice of where the estimated
phase points sit relative to those gradient measurements, together with the
linear relation assumed between them. W. H. Southwell, "Wave-front estimation
from wave-front slope measurements", *Journal of the Optical Society of
America* **70** (8), 998-1006 (1980), Fig. 1, sets out the three standard
choices; two of them are implemented here.

**Fried geometry** (Southwell 1980 Fig. 1b; D. L. Fried, "Least-square fitting
a wave-front distortion estimate to an array of phase-difference measurements",
*JOSA* **67** (3), 370-375, 1977). Phase points sit on the *corners* of the
subaperture cells, an ``(N+1) x (N+1)`` grid for ``N x N`` subapertures. The
gradient at a cell centre is modelled as the mean of the two phase differences
along the two opposite edges::

    u^x_cell = [ (p_{i+1,j}   - p_{i,j}  ) + (p_{i+1,j+1} - p_{i,j+1}) ] / 2
    u^y_cell = [ (p_{i,j+1}   - p_{i,j}  ) + (p_{i+1,j+1} - p_{i+1,j}) ] / 2

with ``p`` the phase at a corner [rad] and ``u = h * ds`` the phase difference
across one subaperture [rad]; ``h`` is the subaperture pitch [m] and ``s`` the
wavefront phase gradient [rad/m]. This is a direct forward model, ``u = G p``.

**Southwell geometry** (Southwell 1980 Fig. 1c, his Eq. 5). Phase points are
*co-located* with the gradient measurements, i.e. at the subaperture centres,
an ``N x N`` grid. The relation is the trapezoidal rule between neighbours::

    ( u^x_{i+1,j} + u^x_{i,j} ) / 2 = p_{i+1,j} - p_{i,j}
    ( u^y_{i,j+1} + u^y_{i,j} ) / 2 = p_{i,j+1} - p_{i,j}

which is *not* of the form ``u = G p``; it is ``E p = C u``. Both geometries are
therefore returned in the common form ``A p = B u`` so that a single
reconstructor implementation serves both.

Exactness (validity range of the discretisation)
------------------------------------------------
Both relations are algebraically exact when the phase is a polynomial of degree
<= 2 in ``(x, y)`` and the gradients are point samples at the nominal
measurement positions:

* Southwell: the trapezoidal rule ``(g(a) + g(b))/2 = (f(b) - f(a))/(b - a)``
  is exact when ``g = f'`` is linear, i.e. ``f`` quadratic.
* Fried: a first difference of a quadratic equals ``h`` times its derivative at
  the interval midpoint, and averaging the two edge midpoints of a cell gives
  the derivative at the cell centre because that derivative is linear in the
  transverse coordinate.

For higher-order phase both incur an ``O(h^2)`` truncation error, and neither
models the *area average* of the gradient that a real subaperture performs.
Those two model errors are quantified in `validation/VALIDATION.md`.

Null spaces
-----------
Both ``A`` matrices annihilate piston (only differences appear). The Fried
operator additionally has an almost-null "waffle" (checkerboard) mode, because
a checkerboard phase pattern produces zero mean edge difference in both
directions; this is the classical reason Fried-geometry least-squares
reconstructors need either regularisation or a waffle penalty
(Southwell 1980 Sec. V; Hardy 1998, *Adaptive Optics for Astronomical
Telescopes*, OUP, ch. 5). The measured singular spectra are reported in
`validation/VALIDATION.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["SubapertureGeometry", "GeometryMatrices", "build_geometry_matrices"]


@dataclass(frozen=True)
class SubapertureGeometry:
    """Square lenslet grid over a circular unobscured pupil.

    Parameters
    ----------
    n_sub : int
        Number of subapertures across one side of the square grid [-], >= 2.
    diameter : float
        Pupil diameter [m], > 0.
    fill_threshold : float
        Minimum fraction of a subaperture's area that must lie inside the pupil
        for it to be counted as illuminated [-], in ``(0, 1]``.

    Attributes
    ----------
    pitch : float
        Subaperture pitch ``diameter / n_sub`` [m].
    mask : ndarray of bool, shape (n_sub, n_sub)
        Illuminated subapertures, indexed ``[iy, ix]``.

    Notes
    -----
    Coordinates exposed by this class are **normalised pupil coordinates**:
    the pupil rim is at ``x**2 + y**2 = 1``, i.e. physical metres divided by
    ``diameter / 2``. That is the coordinate system the Zernike basis in
    :mod:`wavelab.zernike` expects.
    """

    n_sub: int
    diameter: float = 1.0
    fill_threshold: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.n_sub, bool) or not isinstance(self.n_sub, (int, np.integer)):
            raise TypeError(f"n_sub must be an integer, got {self.n_sub!r}")
        if self.n_sub < 2:
            raise ValueError(f"n_sub must be >= 2, got {self.n_sub}")
        d = float(self.diameter)
        if not np.isfinite(d) or d <= 0.0:
            raise ValueError(f"diameter must be finite and > 0 m, got {self.diameter!r}")
        f = float(self.fill_threshold)
        if not (0.0 < f <= 1.0):
            raise ValueError(f"fill_threshold must be in (0, 1], got {self.fill_threshold!r}")

    @property
    def pitch(self) -> float:
        """Subaperture pitch [m]."""
        return float(self.diameter) / int(self.n_sub)

    @property
    def scaled_slope_factor(self) -> float:
        """Factor converting a normalised-coordinate gradient to ``u`` [-].

        ``u = h * dphi/dx`` with ``x`` in metres. With normalised coordinates
        ``xn = x / (D/2)``, ``dphi/dx = (2/D) dphi/dxn``, so
        ``u = (2 h / D) dphi/dxn = (2 / n_sub) dphi/dxn``.
        """
        return 2.0 / int(self.n_sub)

    @property
    def mask(self) -> NDArray[np.bool_]:
        """Illuminated-subaperture mask, shape ``(n_sub, n_sub)``, ``[iy, ix]``.

        A cell counts as illuminated when the fraction of its area inside the
        unit circle is at least ``fill_threshold``. The fraction is evaluated
        by a fixed 8x8 midpoint quadrature per cell, so the mask is
        deterministic but the fill fraction itself is accurate only to about
        ``1/64`` of a cell.
        """
        n = int(self.n_sub)
        edges = np.linspace(-1.0, 1.0, n + 1)
        q = 8
        offs = (np.arange(q) + 0.5) / q
        cell = 2.0 / n
        frac = np.zeros((n, n), dtype=float)
        for iy in range(n):
            ys = edges[iy] + cell * offs
            for ix in range(n):
                xs = edges[ix] + cell * offs
                gx, gy = np.meshgrid(xs, ys, indexing="xy")
                frac[iy, ix] = np.mean(gx**2 + gy**2 <= 1.0)
        return frac >= float(self.fill_threshold) - 1e-12

    @property
    def n_valid_sub(self) -> int:
        """Number of illuminated subapertures [-]."""
        return int(np.count_nonzero(self.mask))

    @property
    def n_slopes(self) -> int:
        """Length of the slope vector, ``2 * n_valid_sub`` [-] (all x, then all y)."""
        return 2 * self.n_valid_sub

    def subaperture_centres(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Normalised ``(x, y)`` of the illuminated subaperture centres.

        Ordering is row-major over ``[iy, ix]`` restricted to the mask, and is
        the ordering of the slope vector.
        """
        n = int(self.n_sub)
        c = (np.arange(n) + 0.5) * (2.0 / n) - 1.0
        gx, gy = np.meshgrid(c, c, indexing="xy")
        m = self.mask
        return gx[m], gy[m]

    def southwell_points(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Normalised ``(x, y)`` of the Southwell phase points (= subaperture centres)."""
        return self.subaperture_centres()

    def corner_mask(self) -> NDArray[np.bool_]:
        """Active corner mask, shape ``(n_sub + 1, n_sub + 1)``, ``[iy, ix]``.

        A corner is active when it belongs to at least one illuminated cell.
        """
        n = int(self.n_sub)
        m = self.mask
        active = np.zeros((n + 1, n + 1), dtype=bool)
        for dy in (0, 1):
            for dx in (0, 1):
                active[dy : dy + n, dx : dx + n] |= m
        return active

    def fried_points(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Normalised ``(x, y)`` of the active Fried phase points (cell corners)."""
        n = int(self.n_sub)
        e = np.linspace(-1.0, 1.0, n + 1)
        gx, gy = np.meshgrid(e, e, indexing="xy")
        a = self.corner_mask()
        return gx[a], gy[a]

    def phase_points(self, geometry: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Normalised phase-point coordinates for ``"southwell"`` or ``"fried"``."""
        g = _check_geometry(geometry)
        return self.southwell_points() if g == "southwell" else self.fried_points()


def _check_geometry(geometry: str) -> str:
    if not isinstance(geometry, str):
        raise TypeError(f"geometry must be a string, got {geometry!r}")
    g = geometry.strip().lower()
    if g not in ("southwell", "fried"):
        raise ValueError(f"geometry must be 'southwell' or 'fried', got {geometry!r}")
    return g


@dataclass(frozen=True)
class GeometryMatrices:
    """The pair ``(A, B)`` defining ``A p = B u`` for one reconstruction geometry.

    Attributes
    ----------
    geometry : str
        ``"southwell"`` or ``"fried"``.
    a : ndarray, shape (n_eq, n_phase)
        Phase-side operator (dimensionless).
    b : ndarray, shape (n_eq, n_slopes)
        Slope-side operator (dimensionless). ``u`` is the phase difference
        across one subaperture, ``u = pitch * gradient`` [rad].
    row_subapertures : list[tuple[int, ...]]
        For each equation, the indices (into ``0 .. n_valid_sub - 1``) of the
        subapertures whose measurement it uses. Used to drop equations when
        subapertures drop out.
    n_phase : int
        Number of phase unknowns.
    """

    geometry: str
    a: NDArray[np.float64]
    b: NDArray[np.float64]
    row_subapertures: list[tuple[int, ...]]

    @property
    def n_phase(self) -> int:
        """Number of phase unknowns [-]."""
        return int(self.a.shape[1])

    @property
    def n_slopes(self) -> int:
        """Length of the slope vector [-]."""
        return int(self.b.shape[1])

    def active_rows(self, sub_available: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Boolean row selector keeping only equations whose subapertures are present.

        Parameters
        ----------
        sub_available : ndarray of bool, shape (n_valid_sub,)
            ``True`` where the subaperture delivered a usable measurement.
        """
        avail = np.asarray(sub_available, dtype=bool).ravel()
        n_sub = self.b.shape[1] // 2
        if avail.shape != (n_sub,):
            raise ValueError(f"sub_available must have shape ({n_sub},), got {avail.shape}")
        return np.array([all(avail[k] for k in row) for row in self.row_subapertures], dtype=bool)


def build_geometry_matrices(geom: SubapertureGeometry, geometry: str) -> GeometryMatrices:
    """Assemble ``A p = B u`` for the requested reconstruction geometry.

    Parameters
    ----------
    geom : SubapertureGeometry
        Lenslet layout.
    geometry : str
        ``"southwell"`` (Southwell 1980, Eq. 5) or ``"fried"``
        (Fried 1977; Southwell 1980, Fig. 1b).

    Returns
    -------
    GeometryMatrices

    Notes
    -----
    Slope-vector ordering is: all x-slopes of the illuminated subapertures in
    row-major mask order, then all y-slopes in the same order.
    """
    g = _check_geometry(geometry)
    if not isinstance(geom, SubapertureGeometry):
        raise TypeError(f"geom must be a SubapertureGeometry, got {type(geom).__name__}")
    return _southwell(geom) if g == "southwell" else _fried(geom)


def _southwell(geom: SubapertureGeometry) -> GeometryMatrices:
    n = int(geom.n_sub)
    mask = geom.mask
    idx = -np.ones((n, n), dtype=int)
    idx[mask] = np.arange(geom.n_valid_sub)
    ns = geom.n_valid_sub

    rows_a: list[NDArray[np.float64]] = []
    rows_b: list[NDArray[np.float64]] = []
    row_subs: list[tuple[int, ...]] = []

    def add(p_lo: int, p_hi: int, s_lo: int, s_hi: int, k_lo: int, k_hi: int) -> None:
        ra = np.zeros(ns)
        ra[p_hi] = 1.0
        ra[p_lo] = -1.0
        rb = np.zeros(2 * ns)
        rb[s_hi] = 0.5
        rb[s_lo] = 0.5
        rows_a.append(ra)
        rows_b.append(rb)
        row_subs.append((k_lo, k_hi))

    for iy in range(n):
        for ix in range(n - 1):
            if mask[iy, ix] and mask[iy, ix + 1]:
                k0, k1 = int(idx[iy, ix]), int(idx[iy, ix + 1])
                add(k0, k1, k0, k1, k0, k1)  # x-slopes occupy 0 .. ns-1
    for iy in range(n - 1):
        for ix in range(n):
            if mask[iy, ix] and mask[iy + 1, ix]:
                k0, k1 = int(idx[iy, ix]), int(idx[iy + 1, ix])
                add(k0, k1, ns + k0, ns + k1, k0, k1)  # y-slopes occupy ns .. 2ns-1

    return GeometryMatrices("southwell", np.array(rows_a), np.array(rows_b), row_subs)


def _fried(geom: SubapertureGeometry) -> GeometryMatrices:
    n = int(geom.n_sub)
    mask = geom.mask
    active = geom.corner_mask()
    cidx = -np.ones((n + 1, n + 1), dtype=int)
    cidx[active] = np.arange(int(np.count_nonzero(active)))
    n_phase = int(np.count_nonzero(active))
    ns = geom.n_valid_sub
    sidx = -np.ones((n, n), dtype=int)
    sidx[mask] = np.arange(ns)

    rows_a: list[NDArray[np.float64]] = []
    rows_b: list[NDArray[np.float64]] = []
    row_subs: list[tuple[int, ...]] = []

    for iy in range(n):
        for ix in range(n):
            if not mask[iy, ix]:
                continue
            k = int(sidx[iy, ix])
            c00 = int(cidx[iy, ix])
            c10 = int(cidx[iy, ix + 1])
            c01 = int(cidx[iy + 1, ix])
            c11 = int(cidx[iy + 1, ix + 1])

            ax = np.zeros(n_phase)
            ax[c10] += 0.5
            ax[c00] -= 0.5
            ax[c11] += 0.5
            ax[c01] -= 0.5
            bx = np.zeros(2 * ns)
            bx[k] = 1.0
            rows_a.append(ax)
            rows_b.append(bx)
            row_subs.append((k,))

            ay = np.zeros(n_phase)
            ay[c01] += 0.5
            ay[c00] -= 0.5
            ay[c11] += 0.5
            ay[c10] -= 0.5
            by = np.zeros(2 * ns)
            by[ns + k] = 1.0
            rows_a.append(ay)
            rows_b.append(by)
            row_subs.append((k,))

    return GeometryMatrices("fried", np.array(rows_a), np.array(rows_b), row_subs)
