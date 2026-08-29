r"""Forward models for irradiance scintillation (the scintillometer channel).

Weak-fluctuation (Rytov) theory
-------------------------------
For a spherical wave the log-irradiance variance is the Rytov double integral
(Andrews & Phillips 2005, Eq. 8.10, with the Kolmogorov spectrum
``Phi_n(kappa) = 0.033 Cn2 kappa^(-11/3)``):

.. math::
    \sigma_I^2 = 8\pi^2 k^2 L \int_0^1\!\!\int_0^\infty
        \kappa\,\Phi_n(\kappa,\xi)\,
        \bigl[1-\cos(L\kappa^2\xi(1-\xi)/k)\bigr]\,d\kappa\,d\xi .

The inner integral has the closed form
``0.5 a^(5/6) [-Gamma(-5/6) cos(5 pi/12)]`` with ``a = L xi(1-xi)/k``, giving

.. math::
    \beta_0^2 = C_R\,k^{7/6}\int_0^L C_n^2(z)\,z^{5/6}(1-z/L)^{5/6}\,dz,
    \qquad C_R = 8\pi^2(0.033)(0.5)\,[-\Gamma(-5/6)\cos(5\pi/12)] = 2.2522625\ldots

which is the coefficient quoted as ``2.25`` in the literature (0.10 % apart).  For
a uniform ``Cn2`` this reduces to ``beta_0^2 = 0.4967 Cn2 k^(7/6) L^(11/6)``, the
value quoted as ``0.5 Cn2 k^(7/6) L^(11/6)`` (Andrews & Phillips 2005, Eq. 8.13);
the plane-wave kernel ``(1-z/L)^(5/6)`` gives ``1.2285`` against the quoted
``1.23`` (Eq. 8.9).  All three constants are re-derived numerically in
``validation/validate_forward_models.py`` rather than copied.

Units: ``Cn2`` m^(-2/3); ``k`` rad/m; ``L`` m; ``beta_0^2`` and ``sigma_I^2``
dimensionless.  Validity of the *linear* relation: ``beta_0^2 < ~0.3`` (weak
fluctuations); see :func:`scintillation_index` for the all-regime model.

Strong fluctuations and saturation
----------------------------------
Beyond ``beta_0^2 ~ 1`` multiple scattering makes ``sigma_I^2`` grow more slowly
than ``beta_0^2``, peak, and then *decay* toward unity -- the saturation of
scintillation (Gracheva & Gurvich 1965; Andrews & Phillips 2005, ch. 9).  The
model implemented here is the Andrews-Phillips spherical-wave scintillation index
with aperture averaging (Andrews & Phillips 2005, Eq. 9.60; Andrews, Phillips,
Hopen & Al-Habash 1999, *Waves in Random Media* 9, 33-45):

.. math::
    \sigma_I^2(D) = \exp\Bigl[
        \frac{0.49\beta_0^2}{(1+0.18d^2+0.56\beta_0^{12/5})^{7/6}}
      + \frac{0.51\beta_0^2(1+0.69\beta_0^{12/5})^{-5/6}}
             {1+0.90d^2+0.62d^2\beta_0^{12/5}} \Bigr] - 1,
    \qquad d = \sqrt{kD^2/(4L)} .

The two exponent terms are the large-scale and small-scale log-irradiance
variances ``sigma_lnX^2`` and ``sigma_lnY^2`` of the gamma-gamma model
(Al-Habash, Andrews & Phillips 2001, *Opt. Eng.* 40(8), 1554-1562), which is what
:mod:`turbscope.measurements` samples from.

Validity: Kolmogorov spectrum, no inner/outer-scale effects, weak-to-strong
horizontal path, ``d`` from a circular receiver aperture.  The formula is an
interpolating fit, accurate to a few per cent against the asymptotic theory; it
is *not* exact.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize, special

from ._validate import check_geometry, check_path_samples, check_positive
from .geometry import PathGeometry, scintillation_weight, weight_normalisation

__all__ = [
    "RYTOV_COEFFICIENT",
    "aperture_parameter_sq",
    "gamma_gamma_parameters",
    "log_irradiance_variances",
    "rytov_variance",
    "rytov_variance_from_average",
    "saturation_peak",
    "scintillation_index",
]

# C_R = 8 pi^2 * 0.033 * 0.5 * [-Gamma(-5/6) cos(5 pi/12)]  -- derived, not copied.
RYTOV_COEFFICIENT: float = float(
    8.0
    * np.pi**2
    * 0.033
    * 0.5
    * (-special.gamma(-5.0 / 6.0) * np.cos(5.0 * np.pi / 12.0))
)

# Above this beta_0^2 the linear (weak-fluctuation) relation sigma_I^2 = beta_0^2
# is in error by more than ~10 %; see validation/validate_saturation.py.
WEAK_REGIME_BETA0_SQ = 0.3


def aperture_parameter_sq(receiver_diameter_m: float, path: PathGeometry) -> float:
    """Squared aperture-averaging parameter ``d^2 = k D^2 / (4 L)``, dimensionless.

    Source: Andrews & Phillips (2005) Eq. (9.60) and ch. 10.  ``d^2 = 0``
    corresponds to a point receiver.  ``D`` is the receiver aperture diameter in
    metres; ``d^2`` is the aperture area measured in Fresnel zones.
    """
    d = check_positive("receiver_diameter_m", receiver_diameter_m, allow_zero=True)
    if not isinstance(path, PathGeometry):
        raise TypeError(f"path must be a PathGeometry, got {type(path).__name__}")
    return path.k * d * d / (4.0 * path.length_m)


def rytov_variance_from_average(cn2_average: float, path: PathGeometry) -> float:
    """Rytov variance ``beta_0^2`` from a kernel-weighted path average of ``Cn2``.

    ``beta_0^2 = C_R * N_W * <Cn2>_W * k^(7/6) * L^(11/6)`` where ``N_W`` is the
    kernel normalisation (``B(11/6,11/6)`` spherical, ``6/11`` plane).  For a
    spherical wave ``C_R * N_W = 0.49670`` (literature value 0.5); for a plane
    wave ``1.22851`` (literature value 1.23).

    Parameters
    ----------
    cn2_average
        Scintillation-kernel weighted path average of ``Cn2``, m^(-2/3).
    path
        Propagation geometry.
    """
    c = check_positive("cn2_average", cn2_average, allow_zero=True)
    if not isinstance(path, PathGeometry):
        raise TypeError(f"path must be a PathGeometry, got {type(path).__name__}")
    norm = weight_normalisation("scintillation", path.geometry)
    return RYTOV_COEFFICIENT * norm * c * path.k ** (7.0 / 6.0) * path.length_m ** (11.0 / 6.0)


def rytov_variance(z_m: np.ndarray, cn2: np.ndarray, path: PathGeometry) -> float:
    """Rytov variance ``beta_0^2`` from a sampled ``Cn2(z)`` path profile.

    Evaluates ``C_R k^(7/6) int_0^L Cn2(z) W(z/L) L^(5/6) dz`` by composite
    Simpson's rule, with ``W`` the scintillation kernel of
    :func:`turbscope.geometry.scintillation_weight`.

    Units: ``z_m`` m, ``cn2`` m^(-2/3); returns a dimensionless variance.
    """
    z, c = check_path_samples(z_m, cn2)
    if not isinstance(path, PathGeometry):
        raise TypeError(f"path must be a PathGeometry, got {type(path).__name__}")
    ltot = path.length_m
    if abs(z[-1] - ltot) > 1e-6 * ltot:
        raise ValueError(
            f"path samples must span the geometry length: z[-1]={z[-1]:g} m but "
            f"path.length_m={ltot:g} m"
        )
    w = scintillation_weight(z / ltot, path.geometry)
    integral = float(integrate.simpson(w * c, x=z))
    return RYTOV_COEFFICIENT * path.k ** (7.0 / 6.0) * ltot ** (5.0 / 6.0) * integral


def log_irradiance_variances(
    beta0_sq: np.ndarray, aperture_d_sq: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Large- and small-scale log-irradiance variances ``(sigma_lnX^2, sigma_lnY^2)``.

    Andrews & Phillips (2005) Eq. (9.60):

    ``sigma_lnX^2 = 0.49 b / (1 + 0.18 d^2 + 0.56 b^(6/5))^(7/6)``
    ``sigma_lnY^2 = 0.51 b (1 + 0.69 b^(6/5))^(-5/6) / (1 + 0.90 d^2 + 0.62 d^2 b^(6/5))``

    with ``b = beta_0^2`` and ``b^(6/5) = beta_0^(12/5)``.  Both dimensionless.
    These are also the gamma-gamma shape parameters' generators
    (:func:`gamma_gamma_parameters`).
    """
    b = np.asarray(beta0_sq, dtype=float)
    if np.any(b < 0.0):
        raise ValueError("beta0_sq must be non-negative")
    if not np.all(np.isfinite(b)):
        raise ValueError("beta0_sq must be finite")
    d2 = check_positive("aperture_d_sq", aperture_d_sq, allow_zero=True)
    b65 = b ** (6.0 / 5.0)
    s_x = 0.49 * b / (1.0 + 0.18 * d2 + 0.56 * b65) ** (7.0 / 6.0)
    s_y = 0.51 * b * (1.0 + 0.69 * b65) ** (-5.0 / 6.0) / (1.0 + 0.90 * d2 + 0.62 * d2 * b65)
    return s_x, s_y


def scintillation_index(beta0_sq: np.ndarray, aperture_d_sq: float = 0.0) -> np.ndarray:
    """Irradiance scintillation index ``sigma_I^2``, dimensionless, all regimes.

    ``sigma_I^2 = exp(sigma_lnX^2 + sigma_lnY^2) - 1`` (Andrews & Phillips 2005,
    Eq. 9.60).  Reduces to ``beta_0^2`` as ``beta_0^2 -> 0`` and saturates toward
    1 as ``beta_0^2 -> inf``, passing through a maximum -- which is why the
    analytic inversion is multi-valued (see :mod:`turbscope.inversion`).

    Validity: spherical wave, Kolmogorov, no inner/outer scale, circular receiver
    aperture entering only through ``d^2 = k D^2 / (4 L)``.
    """
    s_x, s_y = log_irradiance_variances(beta0_sq, aperture_d_sq)
    return np.expm1(s_x + s_y)


def gamma_gamma_parameters(
    beta0_sq: float, aperture_d_sq: float = 0.0
) -> tuple[float, float]:
    """Gamma-gamma shape parameters ``(alpha, beta)``, dimensionless.

    ``alpha = 1/(exp(sigma_lnX^2) - 1)``, ``beta = 1/(exp(sigma_lnY^2) - 1)``
    (Al-Habash, Andrews & Phillips 2001, *Opt. Eng.* 40(8), 1554-1562).  With
    ``I = X*Y``, ``X ~ Gamma(alpha, 1/alpha)`` and ``Y ~ Gamma(beta, 1/beta)``, the
    resulting irradiance has exactly the scintillation index of
    :func:`scintillation_index`, which is why this is the sampling model used by
    :mod:`turbscope.measurements`.

    Raises ``ValueError`` if ``beta0_sq`` is zero (both parameters diverge; there
    is no fluctuation to sample).
    """
    b = check_positive("beta0_sq", beta0_sq)
    s_x, s_y = log_irradiance_variances(np.asarray(b), aperture_d_sq)
    alpha = 1.0 / np.expm1(float(s_x))
    beta = 1.0 / np.expm1(float(s_y))
    return float(alpha), float(beta)


def saturation_peak(aperture_d_sq: float = 0.0) -> tuple[float, float]:
    """Locate the maximum of ``sigma_I^2(beta_0^2)``.

    Returns ``(beta0_sq_at_peak, sigma_I_sq_peak)``.  Above the peak the
    scintillation index *decreases* with turbulence strength, so a measured
    ``sigma_I^2`` below the peak value but above the ``beta_0^2 -> inf`` asymptote
    corresponds to **two** turbulence strengths.  Found by bounded Brent
    minimisation of ``-sigma_I^2`` in ``log10(beta_0^2)`` over ``[-2, 4]``.
    """
    d2 = check_positive("aperture_d_sq", aperture_d_sq, allow_zero=True)

    def neg(log_b: float) -> float:
        return -float(scintillation_index(10.0**log_b, d2))

    res = optimize.minimize_scalar(neg, bounds=(-2.0, 4.0), method="bounded",
                                   options={"xatol": 1e-10})
    if not res.success:  # pragma: no cover - scipy bounded Brent does not fail here
        raise RuntimeError("saturation peak search failed to converge")
    b_peak = float(10.0**res.x)
    return b_peak, float(scintillation_index(b_peak, d2))


def uniform_cn2_from_beta0_sq(beta0_sq: float, path: PathGeometry) -> float:
    """Invert :func:`rytov_variance_from_average` for the weighted-average ``Cn2``.

    ``<Cn2>_W = beta_0^2 / (C_R N_W k^(7/6) L^(11/6))``, m^(-2/3).
    """
    b = check_positive("beta0_sq", beta0_sq, allow_zero=True)
    geom = check_geometry(path.geometry)
    norm = weight_normalisation("scintillation", geom)
    denom = RYTOV_COEFFICIENT * norm * path.k ** (7.0 / 6.0) * path.length_m ** (11.0 / 6.0)
    return b / denom
