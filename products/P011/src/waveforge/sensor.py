"""Shack-Hartmann wavefront sensor model (geometric, slope level).

A Shack-Hartmann sensor divides the pupil into subapertures and measures, for
each one, the **average wavefront gradient** over that subaperture — the spot
displacement behind a lenslet of focal length ``f`` is ``Delta x = f * <dW/dx>``
(Hardy 1998, *Adaptive Optics for Astronomical Telescopes*, Sec. 5.2).  This
module works directly in slope units and does not form spot images; spot
formation and centroiding are the subject of the companion product P018
ShackSim and are cited rather than duplicated here.

Slope convention
----------------
Slopes are **phase gradients in rad/m**::

    s_x = < d(phi)/dx >_subaperture      [rad/m]

Multiply by ``lambda / (2 pi)`` to obtain the OPD gradient (dimensionless
angle of arrival in radians), and by ``f`` to obtain a spot displacement.

Measurement noise
-----------------
Centre-of-gravity slope-measurement variance, in units of ``(lambda / d_sub)^2``
of angle of arrival (G. Rousset, "Wave-front sensors", in *Adaptive Optics in
Astronomy*, ed. F. Roddier, Cambridge University Press 1999, Ch. 5, Eqs. 5.16
and 5.17; the same expressions appear in Nicolle et al., *Astron. Astrophys.*
**420**, 989-1000, 2004)::

    sigma^2_photon = (pi^2 / 2) * (1 / N_ph) * (X_T / X_D)^2
    sigma^2_read   = (pi^2 / 3) * (sigma_e^2 / N_ph^2) * (X_S^2 / X_D)^2

* ``N_ph`` — detected photoelectrons per subaperture per frame,
* ``X_T`` — FWHM of the actual spot in pixels,
* ``X_D`` — FWHM of the diffraction-limited spot in pixels,
* ``X_S`` — number of pixels per side of the centroiding window,
* ``sigma_e`` — detector read noise in electrons RMS per pixel.

Assumptions: unbiased centre of gravity on a square window, background
subtracted, Poisson photon statistics, Gaussian read noise, no spot truncation.
Validity: ``N_ph`` above a few photoelectrons; the expressions diverge as
``N_ph -> 0`` and there the centroid is undefined rather than merely noisy.

Converting to the slope unit used here: one ``lambda / d_sub`` of angle of
arrival is ``2 pi / d_sub`` rad/m of phase gradient, so
``sigma_slope = (2 pi / d_sub) * sigma``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pupil import PupilGrid

__all__ = ["ShackHartmann", "SlopeMeasurement"]


@dataclass(frozen=True)
class SlopeMeasurement:
    """Result of one WFS frame.

    Attributes
    ----------
    slopes:
        Concatenated ``[s_x (all valid subapertures), s_y (same order)]``
        in rad/m.  Dropped subapertures carry ``0.0``.
    valid:
        Boolean array, one entry per valid subaperture, ``False`` where the
        subaperture was dropped this frame (see ``dropout_probability``).
    noise_sigma:
        Per-slope one-sigma measurement noise actually applied [rad/m].
    """

    slopes: np.ndarray
    valid: np.ndarray
    noise_sigma: float


@dataclass
class ShackHartmann:
    """Geometric Shack-Hartmann sensor over a :class:`~waveforge.pupil.PupilGrid`.

    Parameters
    ----------
    pupil:
        Pupil sampling grid.  ``pupil.n_pix`` must be divisible by ``n_sub``.
    n_sub:
        Subapertures across the pupil diameter, ``>= 2``.
    fill_threshold:
        Minimum illuminated fraction for a subaperture to be used, in
        ``(0, 1]``.  Default 0.5 — the usual choice for a circular pupil.
    wavelength_m:
        Sensing wavelength [m], used only for the noise conversion.
    photon_flux:
        Detected photoelectrons per subaperture per frame, ``> 0``.  ``inf``
        means noiseless.
    read_noise_e:
        Detector read noise [e- RMS/pixel], ``>= 0``.
    pixels_per_sub:
        Centroiding window size ``X_S`` in pixels per side, ``>= 2``.
    spot_fwhm_pixels:
        Actual spot FWHM ``X_T`` in pixels, ``> 0``.
    diffraction_fwhm_pixels:
        Diffraction-limited spot FWHM ``X_D`` in pixels, ``> 0``.
    dropout_probability:
        Per-subaperture, per-frame probability that the measurement is lost
        (obscuration, dead lenslet, cosmic ray), in ``[0, 1)``.
    """

    pupil: PupilGrid
    n_sub: int = 8
    fill_threshold: float = 0.5
    wavelength_m: float = 1.55e-6
    photon_flux: float = float("inf")
    read_noise_e: float = 0.0
    pixels_per_sub: int = 6
    spot_fwhm_pixels: float = 2.0
    diffraction_fwhm_pixels: float = 2.0
    dropout_probability: float = 0.0
    _operator: np.ndarray = field(init=False, repr=False)
    _valid_sub: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.n_sub) != self.n_sub or self.n_sub < 2:
            raise ValueError(f"n_sub must be an integer >= 2, got {self.n_sub!r}")
        if self.pupil.n_pix % self.n_sub != 0:
            raise ValueError(
                f"pupil.n_pix ({self.pupil.n_pix}) must be divisible by n_sub ({self.n_sub})"
            )
        if not (0.0 < self.fill_threshold <= 1.0):
            raise ValueError(f"fill_threshold must lie in (0, 1], got {self.fill_threshold!r}")
        if not np.isfinite(self.wavelength_m) or self.wavelength_m <= 0.0:
            raise ValueError(f"wavelength_m must be finite and > 0, got {self.wavelength_m!r}")
        if np.isnan(self.photon_flux) or self.photon_flux <= 0.0:
            raise ValueError(f"photon_flux must be > 0, got {self.photon_flux!r}")
        if not np.isfinite(self.read_noise_e) or self.read_noise_e < 0.0:
            raise ValueError(f"read_noise_e must be finite and >= 0, got {self.read_noise_e!r}")
        if int(self.pixels_per_sub) != self.pixels_per_sub or self.pixels_per_sub < 2:
            raise ValueError(f"pixels_per_sub must be an integer >= 2, got {self.pixels_per_sub!r}")
        for name in ("spot_fwhm_pixels", "diffraction_fwhm_pixels"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value!r}")
        if not (0.0 <= self.dropout_probability < 1.0):
            raise ValueError(
                f"dropout_probability must lie in [0, 1), got {self.dropout_probability!r}"
            )
        self._build()

    # -- geometry ---------------------------------------------------------
    @property
    def subaperture_size_m(self) -> float:
        """Subaperture side length ``d_sub = D / n_sub`` [m]."""
        return self.pupil.diameter_m / self.n_sub

    @property
    def valid_subapertures(self) -> np.ndarray:
        """Boolean ``(n_sub, n_sub)`` array of illuminated subapertures."""
        return self._valid_sub

    @property
    def n_valid(self) -> int:
        """Number of illuminated subapertures."""
        return int(np.count_nonzero(self._valid_sub))

    @property
    def n_slopes(self) -> int:
        """Length of the slope vector (``2 * n_valid``)."""
        return 2 * self.n_valid

    @property
    def operator(self) -> np.ndarray:
        """Slope operator ``G`` with shape ``(n_slopes, pupil.n_valid)`` [1/m].

        ``slopes = G @ phase[mask]``.  Rows ``0 .. n_valid-1`` are ``s_x``,
        rows ``n_valid .. 2 n_valid - 1`` are ``s_y``.
        """
        return self._operator

    def _build(self) -> None:
        n = self.pupil.n_pix
        mask = self.pupil.mask
        step = n // self.n_sub
        d = self.pupil.sample_spacing_m
        index = np.full((n, n), -1, dtype=int)
        index[mask] = np.arange(self.pupil.n_valid)

        fill = np.zeros((self.n_sub, self.n_sub), dtype=float)
        for a in range(self.n_sub):
            for b in range(self.n_sub):
                block = mask[a * step : (a + 1) * step, b * step : (b + 1) * step]
                fill[a, b] = block.mean()
        valid = fill >= self.fill_threshold

        rows_x: list[np.ndarray] = []
        rows_y: list[np.ndarray] = []
        kept = np.zeros_like(valid)
        for a in range(self.n_sub):
            for b in range(self.n_sub):
                if not valid[a, b]:
                    continue
                row_x = np.zeros(self.pupil.n_valid)
                row_y = np.zeros(self.pupil.n_valid)
                count_x = count_y = 0
                for i in range(a * step, (a + 1) * step):
                    for j in range(b * step, (b + 1) * step):
                        if not mask[i, j]:
                            continue
                        if j - 1 >= 0 and j + 1 < n and mask[i, j - 1] and mask[i, j + 1]:
                            row_x[index[i, j + 1]] += 1.0
                            row_x[index[i, j - 1]] -= 1.0
                            count_x += 1
                        if i - 1 >= 0 and i + 1 < n and mask[i - 1, j] and mask[i + 1, j]:
                            row_y[index[i + 1, j]] += 1.0
                            row_y[index[i - 1, j]] -= 1.0
                            count_y += 1
                if count_x == 0 or count_y == 0:
                    continue
                rows_x.append(row_x / (2.0 * d * count_x))
                rows_y.append(row_y / (2.0 * d * count_y))
                kept[a, b] = True
        if not rows_x:
            raise ValueError("no subaperture passed the fill threshold; lower it or use fewer")
        self._valid_sub = kept
        self._operator = np.vstack([np.stack(rows_x), np.stack(rows_y)])

    # -- noise ------------------------------------------------------------
    def slope_noise_variance_lambda_over_d(self) -> tuple[float, float]:
        """Photon and read-noise slope variances in ``(lambda/d_sub)^2`` units."""
        xt, xd = self.spot_fwhm_pixels, self.diffraction_fwhm_pixels
        xs = float(self.pixels_per_sub)
        if np.isinf(self.photon_flux):
            return 0.0, 0.0
        photon = (np.pi**2 / 2.0) * (1.0 / self.photon_flux) * (xt / xd) ** 2
        read = (
            (np.pi**2 / 3.0)
            * (self.read_noise_e**2 / self.photon_flux**2)
            * (xs**2 / xd) ** 2
        )
        return float(photon), float(read)

    def slope_noise_sigma(self) -> float:
        """Total one-sigma slope-measurement noise [rad/m].

        ``sigma = (2 pi / d_sub) * sqrt(sigma^2_photon + sigma^2_read)`` with
        the two Rousset (1999) terms above.  Returns ``0.0`` for an ideal
        (infinite-flux, noiseless) sensor.
        """
        photon, read = self.slope_noise_variance_lambda_over_d()
        return float(2.0 * np.pi / self.subaperture_size_m * np.sqrt(photon + read))

    def noise_equivalent_angle_rad(self) -> float:
        """One-sigma angle-of-arrival measurement error [rad]."""
        return float(self.slope_noise_sigma() * self.wavelength_m / (2.0 * np.pi))

    # -- measurement ------------------------------------------------------
    def true_slopes(self, phase: np.ndarray) -> np.ndarray:
        """Noise-free average subaperture slopes from a pupil phase map [rad/m]."""
        phase = np.asarray(phase, dtype=float)
        if phase.shape != (self.pupil.n_pix, self.pupil.n_pix):
            raise ValueError(
                f"phase shape {phase.shape} does not match pupil "
                f"({self.pupil.n_pix}, {self.pupil.n_pix})"
            )
        return self._operator @ phase[self.pupil.mask]

    def measure(
        self,
        phase: np.ndarray,
        rng: np.random.Generator | int | None = None,
    ) -> SlopeMeasurement:
        """Measure slopes with photon noise, read noise and subaperture dropout.

        Dropped subapertures return zero slope and are flagged ``False`` in
        :attr:`SlopeMeasurement.valid`; a controller is expected to consult the
        flags rather than treat a zero as a real measurement.
        """
        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        slopes = self.true_slopes(phase)
        sigma = self.slope_noise_sigma()
        if sigma > 0.0:
            slopes = slopes + generator.normal(0.0, sigma, size=slopes.shape)
        valid = np.ones(self.n_valid, dtype=bool)
        if self.dropout_probability > 0.0:
            valid = generator.random(self.n_valid) >= self.dropout_probability
            slopes = slopes.copy()
            slopes[: self.n_valid][~valid] = 0.0
            slopes[self.n_valid :][~valid] = 0.0
        return SlopeMeasurement(slopes=slopes, valid=valid, noise_sigma=sigma)
