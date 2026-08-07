"""shacksim — Shack-Hartmann wavefront sensor simulator.

Lenslet-array geometry, per-subaperture spot formation with diffraction and
detector noise, and slope extraction by thresholded centre of gravity, by
correlation, and by a learned ensemble estimator with per-slope confidence.

Research-grade software. Not flight-qualified, not certified, not approved for
operational aerospace use.

Product P018 of the OPTIMA aerospace portfolio. Related work: P008 CentroidNet
covers single-spot subpixel centroiding on one detector window; this product is
the sensor-array level — array geometry, per-subaperture noise, and a
slope-vector output.
"""

from .geometry import AIRY_FWHM_COEFF, LensletArray
from .ml import MLSlopeEstimator
from .sensor import (
    extract_subapertures,
    generate_subaperture_dataset,
    simulate_frame,
    subaperture_spot,
)
from .slopes import (
    cog_displacement,
    cog_noise_sigma,
    cog_slopes,
    correlation_displacement,
    correlation_slopes,
    reference_template,
)
from .wavefront import defocus_slopes, random_slopes, slope_rms, tilt_slopes

__version__ = "0.1.0"

__all__ = [
    "AIRY_FWHM_COEFF",
    "LensletArray",
    "MLSlopeEstimator",
    "__version__",
    "cog_displacement",
    "cog_noise_sigma",
    "cog_slopes",
    "correlation_displacement",
    "correlation_slopes",
    "defocus_slopes",
    "extract_subapertures",
    "generate_subaperture_dataset",
    "random_slopes",
    "reference_template",
    "simulate_frame",
    "slope_rms",
    "subaperture_spot",
    "tilt_slopes",
]
