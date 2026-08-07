"""ZernKit -- Zernike polynomial toolkit for adaptive optics and wavefront analysis.

Conventions used throughout (read this before trusting any coefficient)
-----------------------------------------------------------------------
* **Two single-index orderings are supported and never mixed silently.**
  Noll (1976) indices are 1-based (``j = 1`` is piston); OSA/ANSI indices are
  0-based (``j = 0`` is piston). Convert explicitly with
  :func:`noll_to_osa` / :func:`osa_to_noll`; both are exact integer maps.
* **Sign convention:** ``m > 0`` selects ``cos(m theta)``, ``m < 0`` selects
  ``sin(|m| theta)``, with ``theta`` measured counter-clockwise from ``+x``.
* **Normalisation:** by default modes carry the Noll/ANSI factor
  ``sqrt(2(n+1))`` (``sqrt(n+1)`` for ``m = 0``), giving orthonormality under
  the ``1/pi`` area weight on the unit disc. Pass ``normalized=False`` for the
  unnormalised Born & Wolf form (unit peak). The two coefficient sets are not
  interchangeable.
* **Domain:** the unit disc ``rho <= 1`` only; circular unobscured pupils.
* **Units:** the library is unit-agnostic for the wavefront. Coefficients come
  out in whatever unit the sampled wavefront went in (waves, radians, metres).
  The turbulence statistics in :mod:`zernkit.statistics` are the exception and
  are in rad^2 for a given ``D / r0``.

See ``README.md`` for the equations with sources, and
``validation/VALIDATION.md`` for the numerical evidence.
"""

from __future__ import annotations

from .fitting import FitResult, fit_wavefront, mode_list, zernike_design_matrix
from .gradients import (
    zernike_gradient,
    zernike_gradient_noll,
    zernike_gradient_osa,
    zernike_slope_matrix,
)
from .indexing import (
    mode_name,
    nm_to_noll,
    nm_to_osa,
    noll_to_nm,
    noll_to_osa,
    osa_to_nm,
    osa_to_noll,
    radial_order_from_noll,
    validate_nm,
)
from .polynomials import (
    azimuthal_factor,
    normalization,
    radial_coefficients,
    radial_polynomial,
    unit_disc_grid,
    zernike,
    zernike_cartesian,
    zernike_noll,
    zernike_osa,
)
from .statistics import (
    KOLMOGOROV_PSD_CONSTANT,
    NOLL_PSD_CONSTANT,
    NOLL_TABLE_IV,
    coefficient_variance,
    coefficient_variance_noll,
    residual_variance,
    residual_variance_asymptotic,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # indexing
    "validate_nm",
    "noll_to_nm",
    "nm_to_noll",
    "osa_to_nm",
    "nm_to_osa",
    "noll_to_osa",
    "osa_to_noll",
    "radial_order_from_noll",
    "mode_name",
    # polynomials
    "normalization",
    "radial_coefficients",
    "radial_polynomial",
    "azimuthal_factor",
    "zernike",
    "zernike_cartesian",
    "zernike_noll",
    "zernike_osa",
    "unit_disc_grid",
    # gradients
    "zernike_gradient",
    "zernike_gradient_noll",
    "zernike_gradient_osa",
    "zernike_slope_matrix",
    # fitting
    "FitResult",
    "mode_list",
    "zernike_design_matrix",
    "fit_wavefront",
    # statistics
    "NOLL_PSD_CONSTANT",
    "KOLMOGOROV_PSD_CONSTANT",
    "NOLL_TABLE_IV",
    "coefficient_variance",
    "coefficient_variance_noll",
    "residual_variance",
    "residual_variance_asymptotic",
]
