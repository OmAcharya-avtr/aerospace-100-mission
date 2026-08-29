"""Synthetic Kolmogorov phase screens via the Fourier-transform method.

Power spectral density (Kolmogorov, no inner/outer scale): Roddier, F. (1981),
"The effects of atmospheric turbulence in optical astronomy",
*Progress in Optics* **19**, 281-376, North-Holland::

    Phi_phi(f) = 0.023 * r0^(-5/3) * f^(-11/3)    [rad^2 per (1/length)^2]

with ``f`` the spatial frequency in cycles per unit length and ``r0`` the
Fried parameter (same length unit as ``f^-1``). Generation method: sample
Gaussian-random Fourier coefficients weighted by ``sqrt(Phi_phi)`` on a
regular frequency grid and inverse-transform -- McGlamery, B. L. (1976),
"Computer simulation studies of compensation of turbulence degraded images",
*Proc. SPIE* **74**, 225-233; the specific discretization used here (frequency
grid spacing, FFT normalization) follows J. D. Schmidt, *Numerical Simulation
of Optical Wave Propagation*, SPIE Press, 2010, Sec. 9.3, algorithm
``ft_phase_screen``.

Length unit: this module works in **normalised pupil-diameter units**
(``r0_over_d`` is ``r0`` as a fraction of the pupil diameter) rather than
physical metres, because WaveLab validates reconstruction algorithms, not a
particular telescope's turbulence strength; only the ratio ``r0/D`` enters the
algebra. Reported phase is in **radians**.

Known, documented limitation (not corrected here): the pure FFT method
systematically under-represents low spatial-frequency power (which is where
most Kolmogorov energy is) because the steep ``f^-11/3`` spectrum is
undersampled near ``f = 0`` on a finite grid -- Lane, R. G., Glindemann, A. &
Dainty, J. C. (1992), "Simulation of a Kolmogorov phase screen", *Waves in
Random Media* **2**, 209-224, who propose adding low-frequency "subharmonic"
terms to compensate. WaveLab does **not** implement subharmonics (keeps
generation cheap and the code simple); consequently large-scale content
(tip/tilt and low-order aberrations in particular) in these screens is
understated relative to true Kolmogorov statistics. This is stated in
README "Limitations" and in `DATASET_CARD.md`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["kolmogorov_screen"]

#: Roddier (1981) Kolmogorov phase PSD constant.
_PSD_CONST = 0.023


def kolmogorov_screen(
    n_grid: int, r0_over_d: float, seed: int, pupil_diameter: float = 2.0
) -> NDArray[np.float64]:
    """Generate one square Kolmogorov phase screen [rad].

    Parameters
    ----------
    n_grid: samples per side, ``>= 8``.
    r0_over_d: Fried parameter as a fraction of `pupil_diameter` [-], ``> 0``.
        Smaller values are stronger turbulence.
    seed: integer seed for `numpy.random.default_rng`; the same seed
        reproduces the same screen bit-for-bit.
    pupil_diameter: nominal pupil diameter in the same normalised length unit
        used elsewhere in WaveLab (default 2.0, matching the unit-disc
        convention ``x, y in [-1, 1]``). ``> 0``.

    Returns
    -------
    ``(n_grid, n_grid)`` real array, radians. Mean is not guaranteed to be
    zero (piston is present); callers that need a piston-free screen should
    subtract the mean over their pupil mask, which
    `wavelab.zernike.fit_zernike` does implicitly by excluding the piston
    basis function.
    """
    if isinstance(n_grid, bool) or not isinstance(n_grid, (int, np.integer)):
        raise TypeError(f"n_grid must be an integer, got {n_grid!r}")
    if n_grid < 8:
        raise ValueError(f"n_grid must be >= 8, got {n_grid}")
    r0d = float(r0_over_d)
    if not np.isfinite(r0d) or r0d <= 0.0:
        raise ValueError(f"r0_over_d must be finite and > 0, got {r0_over_d!r}")
    diam = float(pupil_diameter)
    if not np.isfinite(diam) or diam <= 0.0:
        raise ValueError(f"pupil_diameter must be finite and > 0, got {pupil_diameter!r}")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError(f"seed must be an integer, got {seed!r}")

    n = int(n_grid)
    r0 = r0d * diam
    delta = diam / n  # spatial sample spacing, normalised pupil units
    del_f = 1.0 / (n * delta)  # frequency grid spacing, cycles per unit length

    fx = (np.arange(n) - n // 2) * del_f
    fxx, fyy = np.meshgrid(fx, fx, indexing="xy")
    f = np.hypot(fxx, fyy)
    f[n // 2, n // 2] = np.inf  # exclude f = 0 (piston/DC) from the power law

    psd = _PSD_CONST * r0 ** (-5.0 / 3.0) * f ** (-11.0 / 3.0)

    rng = np.random.default_rng(int(seed))
    cn = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * np.sqrt(psd) * del_f
    screen = np.real(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(cn)))) * (n * del_f) ** 2
    return screen
