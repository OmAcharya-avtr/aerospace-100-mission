"""Shack-Hartmann lenslet-array geometry.

Everything in this module is deterministic optical geometry: where the
subapertures are, which of them see light, how big a diffraction-limited spot
is behind one lenslet, and how a local wavefront gradient maps to a spot
displacement on the detector.

Coordinate conventions used throughout `shacksim`
-------------------------------------------------
* Pupil coordinates ``(X, Y)`` are in metres, measured from the pupil centre.
* Detector coordinates inside one subaperture are in **pixels**, measured from
  the geometric centre of that subaperture's pixel block, ``(n_pix - 1) / 2``.
* ``+x`` is increasing column index, ``+y`` is increasing row index. Images are
  stored ``image[row, col] == image[y, x]``.
* A "slope" is the **wavefront (OPD) gradient**, ``dW/dx`` in metres per metre,
  i.e. dimensionless, reported in radians of ray angle (the two are numerically
  equal in the small-angle limit; see `slope_to_displacement`).

References
----------
Hardy, J. W. (1998), *Adaptive Optics for Astronomical Telescopes*, Oxford
University Press — Shack-Hartmann geometry and noise, chapter 5.
Born, M. & Wolf, E. (1999), *Principles of Optics*, 7th ed., Cambridge
University Press — Airy pattern, section 8.5.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["LensletArray", "AIRY_FWHM_COEFF"]

# Full width at half maximum of the Airy pattern, in units of lambda*f/d.
# Born & Wolf (1999) section 8.5.2: the Airy pattern I(v) = (2 J1(v)/v)^2 with
# v = pi*d*r/(lambda*f) falls to half its peak at v = 1.61633..., so
# FWHM = 2 * 1.61633 / pi * lambda*f/d = 1.028787... * lambda*f/d.
AIRY_FWHM_COEFF: float = 1.0287938


def _check_positive(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0, got {v!r}")
    return v


@dataclass(frozen=True)
class LensletArray:
    """Geometry of a square Shack-Hartmann lenslet array with a circular pupil.

    Parameters
    ----------
    n_lenslets:
        Number of lenslets across one side of the square array [-]. >= 2.
    pitch:
        Lenslet pitch (centre-to-centre spacing, taken equal to the clear
        aperture of one lenslet) [m]. > 0.
    focal_length:
        Lenslet focal length [m]. > 0.
    pixels_per_sub:
        Detector pixels across one subaperture, per axis [-]. >= 4. The pixel
        size follows as ``pitch / pixels_per_sub`` — i.e. the detector is
        assumed exactly matched to the lenslet array with no gaps.
    wavelength:
        Optical wavelength [m]. > 0. Monochromatic model only.
    pupil_diameter:
        Diameter of the illuminated circular pupil [m]. Defaults to
        ``n_lenslets * pitch`` (pupil inscribed in the array). > 0.
    obscuration:
        Central obscuration as a fraction of ``pupil_diameter`` [-], in
        [0, 1). Subapertures whose centre falls inside the obscuration are
        marked invalid.
    fill_threshold:
        A subaperture is valid when the fraction of its area inside the
        annular pupil, estimated on a 5x5 subgrid, is >= this value [-],
        in (0, 1].

    Notes
    -----
    The model is an *idealized* geometry: perfectly square, perfectly aligned,
    identical lenslets, 100 % detector fill factor, no gaps, no crosstalk
    between subapertures. See README Limitations.
    """

    n_lenslets: int = 8
    pitch: float = 500e-6
    focal_length: float = 50e-3
    pixels_per_sub: int = 16
    wavelength: float = 633e-9
    pupil_diameter: float | None = None
    obscuration: float = 0.0
    fill_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.n_lenslets, (int, np.integer)) or isinstance(
            self.n_lenslets, (bool, np.bool_)
        ):
            raise TypeError(f"n_lenslets must be an int, got {type(self.n_lenslets).__name__}")
        if not isinstance(self.pixels_per_sub, (int, np.integer)) or isinstance(
            self.pixels_per_sub, (bool, np.bool_)
        ):
            raise TypeError(
                f"pixels_per_sub must be an int, got {type(self.pixels_per_sub).__name__}"
            )
        if self.n_lenslets < 2:
            raise ValueError(f"n_lenslets must be >= 2, got {self.n_lenslets}")
        if self.pixels_per_sub < 4:
            raise ValueError(
                f"pixels_per_sub must be >= 4 (a smaller stamp cannot support a centroid "
                f"or a correlation peak), got {self.pixels_per_sub}"
            )
        _check_positive("pitch", self.pitch)
        _check_positive("focal_length", self.focal_length)
        _check_positive("wavelength", self.wavelength)
        obs = float(self.obscuration)
        if not (0.0 <= obs < 1.0):
            raise ValueError(f"obscuration must be in [0, 1), got {obs!r}")
        fill = float(self.fill_threshold)
        if not (0.0 < fill <= 1.0):
            raise ValueError(f"fill_threshold must be in (0, 1], got {fill!r}")
        if self.pupil_diameter is not None:
            _check_positive("pupil_diameter", self.pupil_diameter)

    # ---------------------------------------------------------------- basics

    @property
    def diameter(self) -> float:
        """Illuminated pupil diameter [m]."""
        if self.pupil_diameter is None:
            return self.n_lenslets * self.pitch
        return float(self.pupil_diameter)

    @property
    def pixel_size(self) -> float:
        """Detector pixel pitch [m] = lenslet pitch / pixels per subaperture."""
        return self.pitch / self.pixels_per_sub

    @property
    def image_size(self) -> int:
        """Side length of the full detector frame [pixels]."""
        return self.n_lenslets * self.pixels_per_sub

    @property
    def pixel_angle(self) -> float:
        """Angular size of one detector pixel seen from the lenslet [rad].

        ``theta_pix = p / f`` in the small-angle (paraxial) limit. This is the
        conversion constant between a spot displacement in pixels and a
        wavefront slope in radians.
        """
        return self.pixel_size / self.focal_length

    # -------------------------------------------------------- spot formation

    @property
    def spot_fwhm(self) -> float:
        """Diffraction-limited spot FWHM behind one lenslet [m].

        ``FWHM = 1.0288 * lambda * f / d`` (Born & Wolf 1999, sec. 8.5.2, Airy
        pattern of a circular aperture of diameter ``d = pitch``). The
        **scaling** is the point: spot size grows linearly with wavelength and
        f-number ``f/d``. Valid for an unobscured, unaberrated, uniformly
        illuminated circular subaperture in the paraxial limit.
        """
        return AIRY_FWHM_COEFF * self.wavelength * self.focal_length / self.pitch

    @property
    def spot_fwhm_px(self) -> float:
        """Diffraction-limited spot FWHM [pixels]."""
        return self.spot_fwhm / self.pixel_size

    @property
    def spot_sigma_px(self) -> float:
        """Gaussian-equivalent spot RMS width [pixels].

        `shacksim` approximates the Airy core by a Gaussian of equal FWHM,
        ``sigma = FWHM / (2 sqrt(2 ln 2)) = FWHM / 2.3548``. The Gaussian has
        no rings and decays far faster than the true ``r^-3`` Airy envelope;
        this understates the flux in the wings (README Limitations).
        """
        return self.spot_fwhm_px / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    @property
    def max_slope(self) -> float:
        """Largest slope whose spot centre still lands inside the subaperture [rad].

        ``s_max = (n_pix / 2) * p / f``. Beyond this the spot leaves its own
        pixel block; the simulator does not model the resulting crosstalk into
        the neighbouring subaperture, it simply truncates.
        """
        return 0.5 * self.pixels_per_sub * self.pixel_angle

    # ----------------------------------------------------- slope conversions

    def slope_to_displacement(self, slope: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """Convert a wavefront slope [rad] to a spot displacement [pixels].

        Derivation (paraxial, single subaperture)
        ----------------------------------------
        Let the wavefront (optical path difference) over the subaperture be
        ``W(x, y)`` [m]. Averaged over the subaperture the local gradient is

            g_x = <dW/dx>  [m/m, dimensionless]

        A wavefront with gradient ``g_x`` is, to first order, a plane wave
        propagating at angle ``theta_x = g_x`` to the optical axis (the
        wavefront normal is tilted by exactly the gradient for small angles).
        A perfect lens of focal length ``f`` maps ray *angle* to focal-plane
        *position*:

            dx = f * tan(theta_x) ~= f * g_x   [m]

        Dividing by the pixel size ``p`` gives the displacement in pixels:

            dx_px = f * g_x / p = g_x / theta_pix

        so displacement is linear in slope with constant ``f / p``. This is the
        entire measurement principle of a Shack-Hartmann sensor
        (Hardy 1998, ch. 5). Validity: paraxial (``|g| << 1``), gradient
        approximately constant over the subaperture, spot inside the block.

        Parameters
        ----------
        slope: wavefront gradient(s) [rad]. Any shape.

        Returns
        -------
        Displacement(s) in pixels, same shape as ``slope``.
        """
        return np.asarray(slope, dtype=float) / self.pixel_angle

    def displacement_to_slope(self, displacement: NDArray[np.float64] | float):
        """Convert a spot displacement [pixels] to a wavefront slope [rad].

        Inverse of `slope_to_displacement`: ``g = dx_px * p / f``.
        """
        return np.asarray(displacement, dtype=float) * self.pixel_angle

    # --------------------------------------------------------- pupil mapping

    def subaperture_centres(self) -> NDArray[np.float64]:
        """Centres of all ``n_lenslets**2`` subapertures in pupil coordinates.

        Returns
        -------
        Array of shape ``(n_lenslets, n_lenslets, 2)`` holding ``(X, Y)`` in
        metres from the pupil centre, indexed ``[row, col]`` with row = y.
        """
        n = self.n_lenslets
        offs = (np.arange(n) - (n - 1) / 2.0) * self.pitch
        gx, gy = np.meshgrid(offs, offs, indexing="xy")
        return np.stack([gx, gy], axis=-1)

    def valid_mask(self) -> NDArray[np.bool_]:
        """Boolean ``(n_lenslets, n_lenslets)`` mask of illuminated subapertures.

        A subaperture counts as illuminated when at least `fill_threshold` of
        its area lies inside the annular pupil, estimated by sampling a 5x5
        grid of points over the subaperture square. This is the standard
        "fill fraction" criterion; the sampling makes it approximate near the
        pupil edge by design (a fully exact area integral would not change the
        physics and would obscure the criterion).
        """
        n = self.n_lenslets
        centres = self.subaperture_centres()
        k = 5
        sub = (np.arange(k) - (k - 1) / 2.0) / k * self.pitch
        sx, sy = np.meshgrid(sub, sub, indexing="xy")
        px = centres[..., 0][..., None, None] + sx
        py = centres[..., 1][..., None, None] + sy
        r = np.hypot(px, py)
        r_out = 0.5 * self.diameter
        r_in = 0.5 * self.diameter * float(self.obscuration)
        inside = (r <= r_out) & (r >= r_in)
        frac = inside.reshape(n, n, -1).mean(axis=-1)
        return frac >= float(self.fill_threshold)

    @property
    def n_valid(self) -> int:
        """Number of illuminated subapertures [-]."""
        return int(self.valid_mask().sum())

    def valid_centres(self) -> NDArray[np.float64]:
        """``(n_valid, 2)`` pupil coordinates [m] of the illuminated subapertures.

        Ordering is row-major over the ``(row, col)`` grid and is the ordering
        used for every slope vector returned by `shacksim`.
        """
        return self.subaperture_centres()[self.valid_mask()]

    def subaperture_slice(self, row: int, col: int) -> tuple[slice, slice]:
        """Row/column slices selecting subaperture ``(row, col)`` in a full frame."""
        n = self.n_lenslets
        if not (0 <= row < n and 0 <= col < n):
            raise ValueError(f"(row, col) = ({row}, {col}) outside a {n}x{n} array")
        p = self.pixels_per_sub
        return slice(row * p, (row + 1) * p), slice(col * p, (col + 1) * p)

    def summary(self) -> dict[str, float]:
        """Key derived quantities, for logging and figure captions (SI units)."""
        return {
            "diameter_m": self.diameter,
            "pixel_size_m": self.pixel_size,
            "pixel_angle_rad": self.pixel_angle,
            "spot_fwhm_m": self.spot_fwhm,
            "spot_fwhm_px": self.spot_fwhm_px,
            "spot_sigma_px": self.spot_sigma_px,
            "max_slope_rad": self.max_slope,
            "n_valid": float(self.n_valid),
        }
