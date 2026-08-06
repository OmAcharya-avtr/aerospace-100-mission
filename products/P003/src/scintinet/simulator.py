"""Split-step phase-screen simulator for optical scintillation.

Method
------
Angular-spectrum (paraxial transfer-function) propagation between thin
Kolmogorov phase screens synthesised by FFT filtering of white noise.

References
----------
- J. D. Schmidt, *Numerical Simulation of Optical Wave Propagation with
  Examples in MATLAB*, SPIE Press, 2010 (split-step method, FFT phase
  screens, sampling constraints).
- J. W. Goodman, *Introduction to Fourier Optics*, 3rd ed., 2005
  (angular-spectrum propagation).
- Kolmogorov phase spectrum: Phi_phi(kappa) = 2*pi*k^2*dz*0.033*Cn^2 *
  kappa^(-11/3) (Andrews & Phillips 2005).

Known limitation
----------------
FFT-synthesised screens contain no power below the grid's fundamental
spatial frequency 2*pi/(N*dx); subharmonic augmentation (Lane et al. 1992)
is NOT implemented. Low-order aberrations (tilt, large-scale phase) are
therefore under-represented. Weak-regime scintillation is dominated by
Fresnel-scale eddies sqrt(L/k), which the grid resolves, so sigma_I^2 is
only mildly affected; do not use this simulator for beam-wander or
long-exposure phase statistics.

Sampling rules (documented, checked at run time)
------------------------------------------------
1. Fresnel-zone resolution:  dx <= sqrt(lambda * L) / 4
   (the intensity speckle scale ~sqrt(lambda*L) must span >= 4 samples).
2. Domain size:              N * dx >= 4 * sqrt(lambda * L)
   (grid side must hold several speckle cells for statistics).
3. Screen phase resolution:  r0_screen >= 2 * dx per screen, where
   r0 = (0.423 k^2 Cn^2 dz)^(-3/5) (Fried parameter, plane wave).
Violations raise ValueError.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rytov import _validate_scalar

__all__ = [
    "SimParams",
    "SimResult",
    "kolmogorov_phase_screen",
    "angular_spectrum_propagate",
    "simulate_scintillation",
]


@dataclass(frozen=True)
class SimParams:
    """Parameters for a scintillation simulation.

    Attributes
    ----------
    cn2 : float
        Refractive-index structure parameter [m^(-2/3)], >= 0.
    wavelength : float
        Wavelength [m].
    path_length : float
        Path length [m].
    aperture_diameters : tuple of float
        Circular receiver aperture diameters [m] for which the
        aperture-averaged scintillation index is evaluated.
    grid_size : int
        Grid points per side (power of 2 recommended).
    grid_width : float
        Physical side length of the grid [m].
    n_screens : int
        Number of phase screens (one per path segment, applied mid-segment).
    n_realizations : int
        Independent turbulence realizations to average over.
    """

    cn2: float
    wavelength: float
    path_length: float
    aperture_diameters: tuple[float, ...] = field(default_factory=tuple)
    grid_size: int = 256
    grid_width: float = 0.5
    n_screens: int = 8
    n_realizations: int = 8


@dataclass(frozen=True)
class SimResult:
    """Result of a scintillation simulation.

    Attributes
    ----------
    sigma_i2_point : float
        Point (single-pixel) scintillation index [dimensionless].
    sigma_i2_aperture : dict
        Aperture-averaged scintillation index per requested diameter [m].
    mean_intensity : float
        Mean intensity over the analysis window (energy check; 1.0 for a
        unit-amplitude input plane wave in a lossless simulation).
    params : SimParams
        Echo of the input parameters.
    seed : int
        Seed used for the random generator.
    """

    sigma_i2_point: float
    sigma_i2_aperture: dict[float, float]
    mean_intensity: float
    params: SimParams
    seed: int


def kolmogorov_phase_screen(
    rng: np.random.Generator,
    n: int,
    dx: float,
    cn2_dz: float,
    wavelength: float,
) -> np.ndarray:
    """Synthesise one Kolmogorov phase screen by FFT filtering (no subharmonics).

    Spectrum: Phi_phi(kappa) = 2*pi*k^2*(Cn^2 dz)*0.033*kappa^(-11/3)
    [rad^2 m^2], Andrews & Phillips 2005; FFT synthesis per Schmidt 2010.
    The kappa=0 (piston) component is set to zero. Frequencies below
    2*pi/(n*dx) are absent (documented low-frequency limitation).

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator (seeded by the caller for reproducibility).
    n : int
        Grid size per side.
    dx : float
        Grid sample spacing [m].
    cn2_dz : float
        Path-integrated structure parameter Cn^2 * dz for this screen
        [m^(1/3)].
    wavelength : float
        Wavelength [m].

    Returns
    -------
    numpy.ndarray
        (n, n) real phase screen [rad].
    """
    if n < 8:
        raise ValueError(f"grid size n must be >= 8, got {n}")
    if cn2_dz < 0.0:
        raise ValueError(f"cn2_dz must be >= 0, got {cn2_dz}")
    k = 2.0 * np.pi / wavelength
    dkappa = 2.0 * np.pi / (n * dx)  # [rad/m]
    kx = np.fft.fftfreq(n, d=dx) * 2.0 * np.pi
    kxx, kyy = np.meshgrid(kx, kx, indexing="ij")
    kappa = np.hypot(kxx, kyy)
    kappa[0, 0] = 1.0  # avoid divide-by-zero; piston removed below
    psd = 2.0 * np.pi * k**2 * cn2_dz * 0.033 * kappa ** (-11.0 / 3.0)
    psd[0, 0] = 0.0
    h = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    # Schmidt 2010 recipe: c = (randn + i*randn) * sqrt(Phi) * dkappa, then take
    # the real part of the inverse DFT. <|c|^2> = 2*Phi*dkappa^2 and the real
    # part carries half that power, so the screen variance integrates Phi
    # correctly over the resolved spectrum.
    c = h * np.sqrt(psd) * dkappa
    screen = np.real(np.fft.ifft2(c)) * n**2
    return screen


def angular_spectrum_propagate(
    field: np.ndarray,
    wavelength: float,
    dx: float,
    dz: float,
) -> np.ndarray:
    """Paraxial angular-spectrum propagation over distance dz.

    Transfer function H(kappa) = exp(-i * (kappa_x^2 + kappa_y^2) * dz / (2k))
    (Fresnel/paraxial approximation; Goodman 2005, Schmidt 2010). Unitary:
    total energy sum(|U|^2) is preserved exactly.

    Parameters
    ----------
    field : numpy.ndarray
        (n, n) complex field at the input plane.
    wavelength : float
        Wavelength [m].
    dx : float
        Sample spacing [m].
    dz : float
        Propagation distance [m] (may be 0).

    Returns
    -------
    numpy.ndarray
        (n, n) complex field at the output plane.
    """
    field = np.asarray(field, dtype=complex)
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError(f"field must be a square 2-D array, got shape {field.shape}")
    if dz == 0.0:
        return field.copy()
    n = field.shape[0]
    k = 2.0 * np.pi / wavelength
    kx = np.fft.fftfreq(n, d=dx) * 2.0 * np.pi
    kxx, kyy = np.meshgrid(kx, kx, indexing="ij")
    hh = np.exp(-1j * (kxx**2 + kyy**2) * dz / (2.0 * k))
    return np.fft.ifft2(np.fft.fft2(field) * hh)


def _check_sampling(p: SimParams) -> None:
    """Enforce the documented sampling rules; raise ValueError on violation."""
    dx = p.grid_width / p.grid_size
    fresnel = np.sqrt(p.wavelength * p.path_length)  # speckle scale [m]
    if dx > fresnel / 4.0:
        raise ValueError(
            f"sampling violation: dx={dx:.4g} m must be <= sqrt(lambda*L)/4="
            f"{fresnel / 4.0:.4g} m; increase grid_size or reduce grid_width"
        )
    if p.grid_width < 4.0 * fresnel:
        raise ValueError(
            f"sampling violation: grid_width={p.grid_width:.4g} m must be >= "
            f"4*sqrt(lambda*L)={4.0 * fresnel:.4g} m; enlarge grid_width"
        )
    if p.cn2 > 0.0:
        k = 2.0 * np.pi / p.wavelength
        dz = p.path_length / p.n_screens
        r0 = (0.423 * k**2 * p.cn2 * dz) ** (-3.0 / 5.0)  # Fried parameter [m]
        if r0 < 2.0 * dx:
            raise ValueError(
                f"sampling violation: per-screen r0={r0:.4g} m must be >= 2*dx="
                f"{2.0 * dx:.4g} m; use more screens or a finer grid"
            )
    for d in p.aperture_diameters:
        if d > p.grid_width / 4.0:
            raise ValueError(
                f"aperture diameter {d:.4g} m exceeds grid_width/4="
                f"{p.grid_width / 4.0:.4g} m; enlarge the grid"
            )


def _aperture_kernel(n: int, dx: float, diameter: float) -> np.ndarray:
    """FFT of a normalised circular-aperture kernel (area-average filter)."""
    x = (np.arange(n) - n // 2) * dx
    xx, yy = np.meshgrid(x, x, indexing="ij")
    mask = (np.hypot(xx, yy) <= diameter / 2.0).astype(float)
    total = mask.sum()
    if total < 1.0:
        # Sub-pixel aperture: treat as a point receiver (single pixel).
        mask[n // 2, n // 2] = 1.0
        total = 1.0
    return np.fft.fft2(np.fft.ifftshift(mask / total))


def simulate_scintillation(params: SimParams, seed: int) -> SimResult:
    """Run a seeded split-step scintillation simulation for a plane wave.

    A unit-amplitude plane wave is propagated through ``n_screens``
    Kolmogorov phase screens (each representing Cn^2 * L/n_screens, applied
    at segment midpoints: propagate dz/2, screen, propagate dz/2). The
    scintillation index sigma_I^2 = <I^2>/<I>^2 - 1 is estimated over the
    central half of the grid across all realizations; aperture-averaged
    indices are computed by circular-mean filtering the intensity before
    taking statistics.

    Parameters
    ----------
    params : SimParams
        Simulation parameters (units documented on the dataclass).
    seed : int
        Seed for numpy's default_rng; identical (params, seed) pairs give
        bit-identical results.

    Returns
    -------
    SimResult
        Point and aperture-averaged scintillation indices, mean intensity.
    """
    _validate_scalar("wavelength", params.wavelength)
    _validate_scalar("path_length", params.path_length)
    _validate_scalar("grid_width", params.grid_width)
    if params.cn2 < 0.0 or not np.isfinite(params.cn2):
        raise ValueError(f"cn2 must be finite and >= 0, got {params.cn2}")
    if params.n_screens < 1:
        raise ValueError(f"n_screens must be >= 1, got {params.n_screens}")
    if params.n_realizations < 1:
        raise ValueError(f"n_realizations must be >= 1, got {params.n_realizations}")
    for d in params.aperture_diameters:
        _validate_scalar("aperture_diameter", d)
    if not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
    _check_sampling(params)

    n = params.grid_size
    dx = params.grid_width / n
    dz = params.path_length / params.n_screens
    cn2_dz = params.cn2 * dz
    rng = np.random.default_rng(seed)
    lo, hi = n // 4, 3 * n // 4  # central analysis window (guards wrap-around)

    kernels = {d: _aperture_kernel(n, dx, d) for d in params.aperture_diameters}
    point_sum, point_sq, count = 0.0, 0.0, 0
    ap_sum = {d: 0.0 for d in params.aperture_diameters}
    ap_sq = {d: 0.0 for d in params.aperture_diameters}

    for _ in range(params.n_realizations):
        u = np.ones((n, n), dtype=complex)  # unit-amplitude plane wave
        for _ in range(params.n_screens):
            u = angular_spectrum_propagate(u, params.wavelength, dx, dz / 2.0)
            if cn2_dz > 0.0:
                u = u * np.exp(1j * kolmogorov_phase_screen(rng, n, dx, cn2_dz,
                                                            params.wavelength))
            u = angular_spectrum_propagate(u, params.wavelength, dx, dz / 2.0)
        intensity = np.abs(u) ** 2
        win = intensity[lo:hi, lo:hi]
        point_sum += win.sum()
        point_sq += (win**2).sum()
        count += win.size
        for d, ker in kernels.items():
            smoothed = np.real(np.fft.ifft2(np.fft.fft2(intensity) * ker))
            swin = smoothed[lo:hi, lo:hi]
            ap_sum[d] += swin.sum()
            ap_sq[d] += (swin**2).sum()

    mean_i = point_sum / count
    sigma_i2_point = point_sq / count / mean_i**2 - 1.0
    sigma_i2_ap = {}
    for d in params.aperture_diameters:
        m = ap_sum[d] / count
        sigma_i2_ap[d] = ap_sq[d] / count / m**2 - 1.0
    return SimResult(
        sigma_i2_point=float(sigma_i2_point),
        sigma_i2_aperture=sigma_i2_ap,
        mean_intensity=float(mean_i),
        params=params,
        seed=int(seed),
    )
