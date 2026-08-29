r"""Synthetic measurement generator: known ``Cn2(z)`` -> noisy sensor readings.

Two instruments are simulated on the same path.

**Scintillometer channels** (point receiver, and a finite receiver aperture).
Irradiance samples are drawn from the **gamma-gamma** distribution of Al-Habash,
Andrews & Phillips (2001), *Opt. Eng.* 40(8), 1554-1562: ``I = X Y`` with
``X ~ Gamma(alpha, 1/alpha)``, ``Y ~ Gamma(beta, 1/beta)`` and

``alpha = 1/(exp(sigma_lnX^2)-1)``, ``beta = 1/(exp(sigma_lnY^2)-1)``,

where ``sigma_lnX^2`` and ``sigma_lnY^2`` are exactly the two terms of the
Andrews-Phillips scintillation index (:func:`turbscope.scintillation.
log_irradiance_variances`).  Sampling this distribution therefore reproduces the
forward model's ``sigma_I^2`` *by construction*, and the estimator noise on
``sigma_I^2 = var(I)/mean(I)^2`` from ``N`` samples comes out of the physics
rather than being bolted on.  ``E[I] = 1`` for all parameter values, so the
normalisation carries no information.

**DIMM channel.**  Differential centroid displacements are Gaussian with variance
``sigma_true^2 + sigma_noise^2``; the sample variance of ``N`` frames is therefore
``(sigma_true^2 + sigma_noise^2) * chi^2_(N-1)/(N-1)``.  ``sigma_noise`` is a
centroid measurement-noise floor (photon and read noise), a real DIMM error term
noted by Tokovinin (2002), *PASP* 114, 1156-1166.  It is assumed to be calibrated
and is subtracted by the closed-form inversion.

Deliberately **not** modelled: finite exposure time (image-motion smearing),
scintillation-induced centroid noise, inner/outer-scale departures from
Kolmogorov, temporal correlation between samples (every sample is independent, so
``N`` here is an *effective independent* count, not a raw frame count), pointing
drift, and non-Kolmogorov exponents.  These are stated in the README limitations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._validate import check_count, check_positive
from .dimm import dimm_variance, fried_parameter
from .geometry import PathGeometry
from .scintillation import (
    aperture_parameter_sq,
    gamma_gamma_parameters,
    rytov_variance,
    scintillation_index,
)

__all__ = ["Measurement", "SensorSuite", "simulate_measurement"]

ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)


@dataclass(frozen=True)
class SensorSuite:
    """Instrument configuration for one synthetic site.

    Parameters
    ----------
    receiver_diameter_m
        Aperture diameter of the *aperture-averaged* scintillometer channel, m.
        The second scintillometer channel is always a point receiver (``d^2 = 0``).
    dimm_subaperture_m, dimm_baseline_m
        DIMM sub-aperture diameter and baseline, m.  ``baseline >= 2 x subaperture``.
    n_irradiance_samples
        Number of statistically independent irradiance samples per channel.
    n_dimm_frames
        Number of independent DIMM frames.
    dimm_noise_arcsec
        RMS differential-centroid measurement noise per frame, arcseconds.  Added
        in quadrature to the atmospheric variance; assumed calibrated.
    """

    receiver_diameter_m: float = 0.10
    dimm_subaperture_m: float = 0.06
    dimm_baseline_m: float = 0.20
    n_irradiance_samples: int = 1000
    n_dimm_frames: int = 500
    dimm_noise_arcsec: float = 0.05

    def __post_init__(self) -> None:
        check_positive("receiver_diameter_m", self.receiver_diameter_m, allow_zero=True)
        check_positive("dimm_subaperture_m", self.dimm_subaperture_m)
        check_positive("dimm_baseline_m", self.dimm_baseline_m)
        check_count("n_irradiance_samples", self.n_irradiance_samples, minimum=10)
        check_count("n_dimm_frames", self.n_dimm_frames, minimum=10)
        check_positive("dimm_noise_arcsec", self.dimm_noise_arcsec, allow_zero=True)
        if self.dimm_baseline_m < 2.0 * self.dimm_subaperture_m:
            raise ValueError(
                "dimm_baseline_m must be >= 2 x dimm_subaperture_m (Sarazin & Roddier 1990)"
            )

    @property
    def dimm_noise_variance_rad2(self) -> float:
        """Centroid-noise variance per component, rad^2."""
        return (self.dimm_noise_arcsec * ARCSEC_TO_RAD) ** 2


@dataclass(frozen=True)
class Measurement:
    """One synthetic observation of a path.

    All ``sigma_*`` fields are the values an instrument would *report*; the
    ``true_*`` fields are the noise-free forward-model values kept for validation.
    """

    sigma_i2_point: float
    sigma_i2_aperture: float
    sigma_l2_rad2: float
    sigma_t2_rad2: float
    true_sigma_i2_point: float
    true_sigma_i2_aperture: float
    true_sigma_l2_rad2: float
    true_sigma_t2_rad2: float
    true_beta0_sq: float
    true_r0_m: float
    aperture_d_sq: float


def _sample_scintillation_index(
    beta0_sq: float, d_sq: float, n: int, rng: np.random.Generator
) -> float:
    """Sample-variance estimate of ``sigma_I^2`` from ``n`` gamma-gamma irradiances."""
    alpha, beta = gamma_gamma_parameters(beta0_sq, d_sq)
    # Gamma(shape, scale=1/shape) has unit mean; the product has unit mean too.
    x = rng.gamma(alpha, 1.0 / alpha, size=n)
    y = rng.gamma(beta, 1.0 / beta, size=n)
    irradiance = x * y
    mean = float(irradiance.mean())
    if mean <= 0.0:  # pragma: no cover - gamma draws are strictly positive
        raise RuntimeError("degenerate irradiance sample")
    return float(irradiance.var(ddof=1) / (mean * mean))


def simulate_measurement(
    z_m: np.ndarray,
    cn2: np.ndarray,
    path: PathGeometry,
    suite: SensorSuite,
    rng: np.random.Generator,
) -> Measurement:
    """Generate one noisy multi-sensor observation of a known ``Cn2(z)`` path.

    Parameters
    ----------
    z_m, cn2
        The known profile, metres and m^(-2/3).  ``z_m[-1]`` must equal
        ``path.length_m``.
    path, suite
        Geometry and instrument configuration.
    rng
        A seeded :class:`numpy.random.Generator`; the function is deterministic
        given the generator state.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator, got {type(rng).__name__}")

    beta0_sq = rytov_variance(z_m, cn2, path)
    if beta0_sq <= 0.0:
        raise ValueError("beta_0^2 is zero: the path carries no turbulence to measure")
    d_sq = aperture_parameter_sq(suite.receiver_diameter_m, path)

    true_point = float(scintillation_index(beta0_sq, 0.0))
    true_aperture = float(scintillation_index(beta0_sq, d_sq))
    meas_point = _sample_scintillation_index(
        beta0_sq, 0.0, suite.n_irradiance_samples, rng
    )
    meas_aperture = _sample_scintillation_index(
        beta0_sq, d_sq, suite.n_irradiance_samples, rng
    )

    r0 = fried_parameter(z_m, cn2, path)
    true_l = dimm_variance(
        r0, path.wavelength_m, suite.dimm_subaperture_m, suite.dimm_baseline_m, "longitudinal"
    )
    true_t = dimm_variance(
        r0, path.wavelength_m, suite.dimm_subaperture_m, suite.dimm_baseline_m, "transverse"
    )
    noise = suite.dimm_noise_variance_rad2
    dof = suite.n_dimm_frames - 1
    meas_l = float((true_l + noise) * rng.chisquare(dof) / dof)
    meas_t = float((true_t + noise) * rng.chisquare(dof) / dof)

    return Measurement(
        sigma_i2_point=meas_point,
        sigma_i2_aperture=meas_aperture,
        sigma_l2_rad2=meas_l,
        sigma_t2_rad2=meas_t,
        true_sigma_i2_point=true_point,
        true_sigma_i2_aperture=true_aperture,
        true_sigma_l2_rad2=true_l,
        true_sigma_t2_rad2=true_t,
        true_beta0_sq=float(beta0_sq),
        true_r0_m=float(r0),
        aperture_d_sq=float(d_sq),
    )
