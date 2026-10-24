"""Beam-divergence context: whether a point-ahead error matters.

A point-ahead error in radians is not a result. The same 5 microradian error is
10 per cent of the half-angle of a 50 microradian beam and 100 per cent of a 5
microradian one, and the loss it costs differs by orders of magnitude. Every
budget this package produces therefore carries the beam it was judged against
and converts the total into a fraction of the beam and a decibel loss.

Equations
---------
Far-field divergence of a circular Gaussian ``TEM00`` beam, measured as the
half-angle to the 1/e^2 intensity radius:

    theta_div = lambda / (pi w0)

with ``w0`` the 1/e^2 intensity waist radius at the aperture. Source: Siegman,
*Lasers*, University Science Books 1986, section 17.1 (Gaussian beam
propagation); also Andrews and Phillips, *Laser Beam Propagation through Random
Media*, 2nd ed., SPIE Press 2005, chapter 4. Valid in the paraxial far field,
``z >> pi w0^2 / lambda``, for an unobscured, untruncated, diffraction-limited
beam.

On-axis-normalised intensity at a radial angle ``theta`` from boresight:

    I(theta) / I(0) = exp(-2 theta^2 / theta_div^2)

which follows directly from the Gaussian transverse profile. The pointing loss
in decibels is ``-10 log10`` of that, i.e.

    L_dB = (20 / ln 10) * (theta / theta_div)^2 = 8.685889638... * theta^2 / theta_div^2

Limitations that matter for a real terminal and are not modelled here: aperture
truncation and central obscuration change the far-field pattern and the
optimum truncation ratio (Klein and Degnan, "Optical antenna gain. 1:
Transmitting antennas", *Applied Optics* 13(9), 1974); atmospheric turbulence
does not apply to a space-to-space crosslink but does to a space-to-ground
one; and a real terminal's jitter spectrum turns a static pointing error into a
distribution, which this module does not integrate over. The numbers here are
the diffraction-limited static case and are labelled as such in the README.
"""

from __future__ import annotations

import numpy as np

_DB_PER_NEPER_TWO = 20.0 / float(np.log(10.0))


def gaussian_divergence_half_angle(wavelength_m: float, waist_m: float) -> float:
    """Far-field 1/e^2 intensity divergence half-angle in radians.

    ``theta_div = wavelength / (pi * waist)``.

    Raises
    ------
    ValueError
        If either argument is non-positive or non-finite, or if the waist is
        smaller than the wavelength, where the paraxial far-field expression is
        not valid.
    """
    wavelength = float(wavelength_m)
    waist = float(waist_m)
    for name, value in (("wavelength_m", wavelength), ("waist_m", waist)):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value!r}")
    if waist < wavelength:
        raise ValueError(
            f"waist_m {waist!r} is below wavelength_m {wavelength!r}; the paraxial "
            "far-field Gaussian divergence is not valid in that regime"
        )
    return wavelength / (float(np.pi) * waist)


def rayleigh_range(wavelength_m: float, waist_m: float) -> float:
    """Rayleigh range ``pi w0^2 / lambda`` in metres.

    Reported alongside a divergence so a reader can check that the link range
    is in the far field, which is the regime the divergence expression assumes.
    """
    theta = gaussian_divergence_half_angle(wavelength_m, waist_m)
    return float(waist_m) / theta


def beam_fraction(error_rad: float, divergence_rad: float) -> float:
    """Pointing error as a fraction of the 1/e^2 divergence half-angle."""
    divergence = float(divergence_rad)
    if not np.isfinite(divergence) or divergence <= 0.0:
        raise ValueError(f"divergence_rad must be positive and finite, got {divergence_rad!r}")
    error = float(error_rad)
    if not np.isfinite(error) or error < 0.0:
        raise ValueError(f"error_rad must be non-negative and finite, got {error_rad!r}")
    return error / divergence


def pointing_loss_fraction(error_rad: float, divergence_rad: float) -> float:
    """On-axis-normalised intensity at ``error_rad`` off boresight, in ``(0, 1]``."""
    ratio = beam_fraction(error_rad, divergence_rad)
    return float(np.exp(-2.0 * ratio * ratio))


def pointing_loss_db(error_rad: float, divergence_rad: float) -> float:
    """Pointing loss in decibels at ``error_rad`` off boresight, non-negative."""
    ratio = beam_fraction(error_rad, divergence_rad)
    return _DB_PER_NEPER_TWO * ratio * ratio


def error_for_loss_db(loss_db: float, divergence_rad: float) -> float:
    """Pointing error in radians that costs exactly ``loss_db`` decibels.

    The inverse of :func:`pointing_loss_db`. This is the number a terminal
    designer actually wants: given the beam, how much total point-ahead error
    is affordable before the link budget line for pointing loss is spent.
    """
    loss = float(loss_db)
    if not np.isfinite(loss) or loss < 0.0:
        raise ValueError(f"loss_db must be non-negative and finite, got {loss_db!r}")
    divergence = float(divergence_rad)
    if not np.isfinite(divergence) or divergence <= 0.0:
        raise ValueError(f"divergence_rad must be positive and finite, got {divergence_rad!r}")
    return divergence * float(np.sqrt(loss / _DB_PER_NEPER_TWO))
