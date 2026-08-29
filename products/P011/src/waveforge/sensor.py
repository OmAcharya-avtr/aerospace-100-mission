"""Shack-Hartmann wavefront sensor model at the slope level.

Scope
-----
This is a **slope-level** model: each subaperture returns the spatially
averaged wavefront gradient over its footprint, plus a noise term drawn from an
analytic photon/read-noise expression. It deliberately does not form detector
images. The pixel-level companion (spot formation, thresholded centre of
gravity, correlation centroiding) is product **P018 ShackSim**; no code is
shared with it and none of its results are re-derived here.

Measurement principle
---------------------
A lenslet of clear aperture ``d`` samples the *average* wavefront gradient over
its footprint (Hardy, J. W. 1998, *Adaptive Optics for Astronomical
Telescopes*, Oxford University Press, ch. 5):

```
s_x = (1/A) * integral_A  dphi/dx  dA          [rad of phase per metre]
```

For a square subaperture this equals ``(<phi>_right_edge - <phi>_left_edge)/d``
by the divergence theorem. It is implemented here as the mean of a centred
finite-difference gradient over the subaperture samples, which is the same
quantity to second order in the grid pitch.

Units. Slopes are carried as **radians of optical phase per metre**. The ray
angle is ``theta = s * lambda / (2 pi)`` [rad], and the optical-path-difference
gradient is ``dW/dx = theta`` [m/m].

Noise
-----
Two independent contributions, both standard (Hardy 1998 ch. 5; Thomas, S.
et al. 2006, "Comparison of centroid computation algorithms in a Shack-Hartmann
sensor", *MNRAS* **371**, 323-336; Rousset, G. 1999, in *Adaptive Optics in
Astronomy*, ed. F. Roddier, Cambridge Univ. Press, ch. 5):

```
Var(theta_hat) = sigma_theta^2 / N_ph                       (photon noise)
               + sigma_e^2 * N_D^2 (N_D^2 - 1) / 12
                 * theta_pix^2 / N_ph^2                     (read noise)
```

with

* ``sigma_theta`` the angular Gaussian-equivalent width of the spot. The Airy
  core has ``FWHM = 1.0287938 lambda/d`` (Born & Wolf 1999, *Principles of
  Optics*, 7th ed., sec. 8.5.2), so ``sigma_theta = FWHM / 2.3548 =
  0.43689 lambda/d``;
* ``N_ph`` detected photo-electrons per subaperture per frame;
* ``sigma_e`` read noise [e- rms per pixel], ``N_D`` the centroiding window
  side in pixels, ``theta_pix`` the angular pixel scale [rad].

The first term is the Cramer-Rao bound for the centroid of a Gaussian spot; the
second is the variance a zero-mean per-pixel noise of variance ``sigma_e^2``
injects into a first-moment estimator whose lever arms are the pixel offsets
(``sum x_i^2 = N_D^2 (N_D^2-1)/12`` pixel^2 for an ``N_D x N_D`` window with
centred coordinates). *Validity:* high enough flux that the centroid is not
capture-limited; Thomas et al. 2006 and P018 both show the linearised
expression under-predicts badly below ~100 e- per subaperture. *Assumptions:*
background-subtracted, unsaturated, spot inside the window, no spot elongation.

Converting to phase slope: ``Var(s) = (2 pi / lambda)^2 Var(theta_hat)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .pupil import Pupil

__all__ = ["AIRY_FWHM_COEFF", "GAUSS_SIGMA_PER_FWHM", "ShackHartmann"]

# Born & Wolf (1999) sec. 8.5.2: the Airy pattern (2 J1(v)/v)^2 falls to half
# its peak at v = 1.61633, giving FWHM = 2 * 1.61633 / pi * lambda/d.
AIRY_FWHM_COEFF: float = 1.0287938
# sigma = FWHM / (2 sqrt(2 ln 2)).
GAUSS_SIGMA_PER_FWHM: float = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _check_positive(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0, got {v!r}")
    return v


@dataclass
class ShackHartmann:
    """Square Shack-Hartmann array sampling a :class:`~waveforge.pupil.Pupil`.

    Parameters
    ----------
    pupil:
        Pupil grid the sensor observes.
    n_sub:
        Subapertures across the pupil diameter [-]. Must be >= 2 and divide
        ``pupil.n_grid`` exactly, so each subaperture owns a whole number of
        grid samples.
    wavelength:
        Sensing wavelength [m]. Must be > 0.
    fill_threshold:
        Minimum fraction of a subaperture's samples that must lie inside the
        clear aperture for the subaperture to be used [-], in ``(0, 1]``.
        Default 0.5, the usual choice.
    n_pixels:
        Centroiding window side [pixels]. Default 4.
    pixel_scale:
        Angular pixel scale [rad]. Default ``None`` -> Nyquist sampling of the
        diffraction spot, ``lambda / (2 d_sub)``.

    Attributes
    ----------
    valid:
        Boolean ``(n_sub, n_sub)`` array of usable subapertures.
    n_valid:
        Number of usable subapertures [-].
    n_slopes:
        ``2 * n_valid`` -- the length of a measurement vector [-].
    """

    pupil: Pupil
    n_sub: int
    wavelength: float
    fill_threshold: float = 0.5
    n_pixels: int = 4
    pixel_scale: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pupil, Pupil):
            raise TypeError(f"pupil must be a Pupil, got {type(self.pupil).__name__}")
        n_sub = int(self.n_sub)
        if n_sub < 2:
            raise ValueError(f"n_sub must be >= 2, got {n_sub}")
        if self.pupil.n_grid % n_sub != 0:
            raise ValueError(
                f"n_sub ({n_sub}) must divide pupil.n_grid ({self.pupil.n_grid}) exactly"
            )
        self.n_sub = n_sub
        self.wavelength = _check_positive("wavelength", self.wavelength)
        ft = float(self.fill_threshold)
        if not (0.0 < ft <= 1.0):
            raise ValueError(f"fill_threshold must be in (0, 1], got {self.fill_threshold!r}")
        self.fill_threshold = ft
        npx = int(self.n_pixels)
        if npx < 2:
            raise ValueError(f"n_pixels must be >= 2, got {npx}")
        self.n_pixels = npx
        if self.pixel_scale is None:
            self.pixel_scale = self.wavelength / (2.0 * self.sub_size)
        else:
            self.pixel_scale = _check_positive("pixel_scale", self.pixel_scale)

        self._block = self.pupil.n_grid // n_sub
        fill = self._fill_fraction()
        self._fill = fill
        self.valid: NDArray[np.bool_] = fill >= self.fill_threshold
        if not np.any(self.valid):
            raise ValueError(
                "no subaperture reaches fill_threshold "
                f"{self.fill_threshold}; max fill is {fill.max():.3f}"
            )

    # ---------------------------------------------------------------- geometry
    @property
    def sub_size(self) -> float:
        """Subaperture side length ``d = D / n_sub`` [m]."""
        return self.pupil.diameter / self.n_sub

    @property
    def n_valid(self) -> int:
        """Number of usable subapertures [-]."""
        return int(np.count_nonzero(self.valid))

    @property
    def n_slopes(self) -> int:
        """Length of a slope measurement vector, ``2 * n_valid`` [-]."""
        return 2 * self.n_valid

    @property
    def fill_fraction(self) -> NDArray[np.float64]:
        """Per-subaperture illuminated fraction [-], shape ``(n_sub, n_sub)``."""
        return self._fill

    def _fill_fraction(self) -> NDArray[np.float64]:
        b = self._block
        m = self.pupil.mask.astype(np.float64)
        return m.reshape(self.n_sub, b, self.n_sub, b).mean(axis=(1, 3))

    def subaperture_centres(self) -> NDArray[np.float64]:
        """Centres of the valid subapertures, shape ``(n_valid, 2)`` [m] as ``(x, y)``."""
        d = self.sub_size
        axis = (np.arange(self.n_sub) - (self.n_sub - 1) / 2.0) * d
        xx, yy = np.meshgrid(axis, axis, indexing="xy")
        return np.column_stack([xx[self.valid], yy[self.valid]])

    # ------------------------------------------------------------- measurement
    def slopes(self, phase: NDArray[np.float64]) -> NDArray[np.float64]:
        """Noise-free average gradient per valid subaperture.

        Parameters
        ----------
        phase:
            ``(n_grid, n_grid)`` optical phase [rad]. Values outside the pupil
            are ignored (only samples inside the mask contribute).

        Returns
        -------
        ndarray
            Length ``2 * n_valid`` vector ``[s_x (all subaps), s_y (all subaps)]``
            in **rad of phase per metre**.
        """
        arr = np.asarray(phase, dtype=np.float64)
        n = self.pupil.n_grid
        if arr.shape != (n, n):
            raise ValueError(f"phase must have shape {(n, n)}, got {arr.shape}")
        dx = self.pupil.dx
        mask = self.pupil.mask
        filled = np.where(mask, arr, 0.0)
        gy, gx = np.gradient(filled, dx, dx)
        # Only average gradient samples whose 3-point stencil is fully inside
        # the pupil, otherwise the edge of the mask injects a spurious gradient.
        interior = mask.copy()
        interior[1:, :] &= mask[:-1, :]
        interior[:-1, :] &= mask[1:, :]
        interior[:, 1:] &= mask[:, :-1]
        interior[:, :-1] &= mask[:, 1:]
        b = self._block
        w = interior.astype(np.float64).reshape(self.n_sub, b, self.n_sub, b).sum(axis=(1, 3))
        w = np.where(w > 0, w, 1.0)
        sx = (
            np.where(interior, gx, 0.0).reshape(self.n_sub, b, self.n_sub, b).sum(axis=(1, 3)) / w
        )
        sy = (
            np.where(interior, gy, 0.0).reshape(self.n_sub, b, self.n_sub, b).sum(axis=(1, 3)) / w
        )
        return np.concatenate([sx[self.valid], sy[self.valid]])

    # -------------------------------------------------------------------- noise
    def spot_sigma_angle(self) -> float:
        """Gaussian-equivalent angular width of the diffraction spot [rad].

        ``sigma_theta = 1.0287938 * lambda / d / 2.3548``
        (Born & Wolf 1999, sec. 8.5.2).
        """
        return AIRY_FWHM_COEFF * self.wavelength / self.sub_size * GAUSS_SIGMA_PER_FWHM

    def noise_variance(self, n_photons: float, read_noise: float = 0.0) -> float:
        """Per-slope measurement noise variance [(rad/m)^2].

        Parameters
        ----------
        n_photons:
            Detected photo-electrons per subaperture per frame [-]. Must be > 0.
        read_noise:
            Detector read noise [e- rms per pixel]. Must be >= 0.

        Returns
        -------
        float
            Variance of one slope component, in (rad of phase per metre)^2.

        Notes
        -----
        See the module docstring for the expression and its validity range. The
        linearised read-noise term is optimistic below roughly 100 e- per
        subaperture (Thomas et al. 2006).
        """
        n_photons = _check_positive("n_photons", n_photons)
        rn = float(read_noise)
        if not np.isfinite(rn) or rn < 0.0:
            raise ValueError(f"read_noise must be >= 0, got {read_noise!r}")
        sigma_theta = self.spot_sigma_angle()
        npx = self.n_pixels
        lever = npx**2 * (npx**2 - 1) / 12.0
        var_theta = sigma_theta**2 / n_photons
        var_theta += rn**2 * lever * float(self.pixel_scale) ** 2 / n_photons**2
        return float((2.0 * np.pi / self.wavelength) ** 2 * var_theta)

    def measure(
        self,
        phase: NDArray[np.float64],
        n_photons: float | None = None,
        read_noise: float = 0.0,
        rng: np.random.Generator | int | None = None,
        dropout: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.float64]:
        """Noisy slope measurement.

        Parameters
        ----------
        phase:
            ``(n_grid, n_grid)`` optical phase [rad].
        n_photons:
            Detected photo-electrons per subaperture per frame. ``None``
            (default) means a noiseless measurement.
        read_noise:
            Read noise [e- rms per pixel].
        rng:
            ``numpy.random.Generator`` or seed.
        dropout:
            Optional boolean array of length ``n_valid``; True marks a
            subaperture whose measurement is lost this frame. Lost slopes are
            returned as 0.0, which is what a reconstructor sees when a
            subaperture is flagged and its measurement replaced by the
            zero-mean prior.

        Returns
        -------
        ndarray
            Length ``2 * n_valid`` slope vector [rad/m].
        """
        s = self.slopes(phase)
        if n_photons is not None:
            if not isinstance(rng, np.random.Generator):
                rng = np.random.default_rng(rng)
            sigma = np.sqrt(self.noise_variance(n_photons, read_noise))
            s = s + sigma * rng.standard_normal(s.shape)
        if dropout is not None:
            d = np.asarray(dropout, dtype=bool)
            if d.shape != (self.n_valid,):
                raise ValueError(
                    f"dropout must have shape {(self.n_valid,)}, got {d.shape}"
                )
            s = s.copy()
            s[: self.n_valid][d] = 0.0
            s[self.n_valid :][d] = 0.0
        return s
