"""Synthetic slopes-to-Zernike dataset generation for the modal benchmark and the ML model.

Pipeline per sample, entirely deterministic given a seed:

1. Draw a Kolmogorov phase screen (`wavelab.screens.kolmogorov_screen`) on an
   ``n_grid_screen x n_grid_screen`` unit-disc grid, with ``r0/D`` drawn
   uniformly from a caller-supplied range (more turbulence strengths seen in
   training than any single value).
2. Least-squares fit the screen to Noll-Zernike coefficients
   (`wavelab.zernike.fit_zernike`), excluding piston -- this is the "ground
   truth" `a_true` every reconstructor in this benchmark tries to recover.
3. Compute the noise-free subaperture-centre slope vector analytically,
   ``s = G_modal @ a_true`` (`wavelab.zernike.zernike_slope_matrix`) -- exact,
   no finite-screen-resolution discretization error, so validation §1
   (noise-free recovery) isolates the reconstructor's own numerical behaviour.
4. Add photon-shot-noise (`wavelab.noise.add_slope_noise`) at the requested
   flux.
5. Draw a subaperture dropout mask (`wavelab.noise.apply_dropout`) at the
   requested rate; slopes at inactive subapertures are zeroed in the returned
   array (`wavelab.ml` uses the mask as an explicit input feature; the
   regularized least-squares baseline in `wavelab.modal` instead drops the
   corresponding matrix rows exactly -- see `ModalReconstructor` docstring for
   why that is not "cheating" so much as a documented asymmetry).

Committed to the repository: no. Every array here regenerates deterministically
from an integer seed in well under a second per hundred samples (see
`DATASET_CARD.md`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .geometry import PupilGrid
from .noise import add_slope_noise, apply_dropout
from .screens import kolmogorov_screen
from .zernike import fit_zernike, unit_disc_grid, zernike_slope_matrix

__all__ = ["ModalGeometry", "SampleBatch", "build_modal_geometry", "generate_batch"]


@dataclass(frozen=True)
class ModalGeometry:
    """Fixed subaperture layout and interaction matrix shared by every sample.

    Parameters
    ----------
    noll_indices: Noll ``j`` values reconstructed (piston excluded).
    sub_x, sub_y: ``(n_sub,)`` subaperture centres, dimensionless pupil units.
    matrix: ``(2 * n_sub, n_modes)`` `wavelab.zernike.zernike_slope_matrix`.
    """

    noll_indices: list[int]
    sub_x: NDArray[np.float64]
    sub_y: NDArray[np.float64]
    matrix: NDArray[np.float64]

    @property
    def n_sub(self) -> int:
        return self.sub_x.size

    @property
    def n_modes(self) -> int:
        return len(self.noll_indices)


def build_modal_geometry(noll_indices: list[int], n_side: int) -> ModalGeometry:
    """Build a `ModalGeometry` from an ``n_side x n_side`` circular subaperture layout.

    Parameters
    ----------
    noll_indices: Noll ``j`` values to reconstruct (``1`` excluded).
    n_side: subapertures per side of the square layout before the circular
        mask, ``>= 3``.

    Returns
    -------
    `ModalGeometry` with ``n_sub`` = number of subapertures inside the unit
    disc.
    """
    if 1 in noll_indices:
        raise ValueError("piston (Noll j = 1) has zero gradient and cannot be reconstructed")
    grid = PupilGrid(n_side)
    if grid.n_active < 2 * len(noll_indices):
        raise ValueError(
            f"n_side={n_side} gives only {grid.n_active} active subapertures, "
            f"too few for {len(noll_indices)} modes (need >= {2 * len(noll_indices)})"
        )
    sub_x, sub_y = grid.active_coords()
    matrix = zernike_slope_matrix(list(noll_indices), sub_x, sub_y)
    return ModalGeometry(list(noll_indices), sub_x, sub_y, matrix)


@dataclass(frozen=True)
class SampleBatch:
    """A batch of synthetic slope/coefficient samples.

    Attributes
    ----------
    slopes: ``(n_samples, 2 * n_sub)`` noisy slopes, zero at inactive rows.
    active: ``(n_samples, n_sub)`` bool, True = subaperture kept.
    coeffs: ``(n_samples, n_modes)`` ground-truth Noll coefficients [rad].
    """

    slopes: NDArray[np.float64]
    active: NDArray[np.bool_]
    coeffs: NDArray[np.float64]

    def __len__(self) -> int:
        return self.slopes.shape[0]


def generate_batch(
    geometry: ModalGeometry,
    n_samples: int,
    photon_flux: float,
    dropout_rate: float,
    seed: int,
    r0_over_d_range: tuple[float, float] = (0.10, 0.35),
    n_grid_screen: int = 64,
    sigma_ref: float = 1.0,
    flux_ref: float = 100.0,
) -> SampleBatch:
    """Generate one batch of samples for a fixed (flux, dropout) operating point.

    Parameters
    ----------
    geometry: from `build_modal_geometry`.
    n_samples: batch size, ``>= 1``.
    photon_flux: photons per subaperture ``N`` [-], ``> 0``. See `wavelab.noise.slope_sigma`.
    dropout_rate: subaperture dropout probability [-], in ``[0, 1)``.
    seed: base integer seed; sample ``i`` uses screen seed ``seed * 2 + 2*i``
        and independent noise/dropout seed ``seed * 2 + 2*i + 1``, so
        re-running with the same `seed` is bit-for-bit reproducible and
        disjoint from any other `seed`.
    r0_over_d_range: ``(low, high)`` uniform range for each sample's Fried
        parameter as a fraction of the pupil diameter, ``0 < low < high``.
    n_grid_screen: phase-screen grid side, ``>= 8`` (kept small per the
        mission compute budget; default 64).
    sigma_ref, flux_ref: passed to `wavelab.noise.slope_sigma`.

    Returns
    -------
    `SampleBatch` of size `n_samples`.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise TypeError(f"n_samples must be an integer, got {n_samples!r}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    lo, hi = r0_over_d_range
    if not (0.0 < lo < hi):
        raise ValueError(f"r0_over_d_range must satisfy 0 < low < high, got {r0_over_d_range!r}")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError(f"seed must be an integer, got {seed!r}")

    x, y, mask = unit_disc_grid(n_grid_screen)
    xm, ym = x[mask], y[mask]

    n_sub = geometry.n_sub
    n_modes = geometry.n_modes
    slopes = np.empty((n_samples, 2 * n_sub), dtype=np.float64)
    active = np.empty((n_samples, n_sub), dtype=bool)
    coeffs = np.empty((n_samples, n_modes), dtype=np.float64)

    for i in range(n_samples):
        screen_rng = np.random.default_rng(int(seed) * 2 + 2 * i)
        r0d = screen_rng.uniform(lo, hi)
        screen = kolmogorov_screen(n_grid_screen, r0d, seed=int(seed) * 2 + 2 * i)
        a_true = fit_zernike(geometry.noll_indices, xm, ym, screen[mask])
        s_true = geometry.matrix @ a_true

        noise_rng = np.random.default_rng(int(seed) * 2 + 2 * i + 1)
        s_noisy = add_slope_noise(
            s_true, photon_flux, noise_rng, sigma_ref=sigma_ref, flux_ref=flux_ref
        )
        act = apply_dropout(n_sub, dropout_rate, noise_rng)
        row_mask = np.concatenate([act, act])
        s_noisy = np.where(row_mask, s_noisy, 0.0)

        slopes[i] = s_noisy
        active[i] = act
        coeffs[i] = a_true

    return SampleBatch(slopes=slopes, active=active, coeffs=coeffs)
