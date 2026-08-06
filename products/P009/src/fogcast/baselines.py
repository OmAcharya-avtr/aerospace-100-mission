"""Empirical fog/aerosol optical attenuation baselines for free-space optical (FSO) links.

Both models relate meteorological visibility V (km) and optical wavelength lambda (nm)
to the specific attenuation of an optical beam (dB/km) through the Koschmieder relation
with a 2 % contrast threshold:

    sigma(V, lambda) = (3.912 / V) * (lambda / 550 nm)^(-q)      [1/km]
    alpha_dB         = (10 / ln 10) * sigma                       [dB/km]

where 3.912 = ln(1/0.02) and q is an empirical size-distribution exponent.

References
----------
- P. W. Kruse, L. D. McGlauchlin, R. B. McQuistan, "Elements of Infrared Technology:
  Generation, Transmission and Detection", Wiley, 1962.  (Kruse q(V) exponent.)
- I. I. Kim, B. McArthur, E. Korevaar, "Comparison of laser beam propagation at 785 nm
  and 1550 nm in fog and haze for optical wireless communications", Proc. SPIE
  vol. 4214, pp. 26-37, 2001.  (Kim piecewise q(V) modification for V < 6 km.)
- Koschmieder relation with 2 % contrast threshold: visibility V defined as the range
  at which image contrast drops to 2 % at 550 nm.

Units: visibility in km, wavelength in nm, attenuation in dB/km.

Assumptions and validity
------------------------
- Empirical models for haze/fog aerosol scattering in the visible / near-IR
  (approximately 500-2000 nm); they do not include molecular absorption lines,
  rain, or snow.
- Visibility is the 550 nm Koschmieder visibility (2 % contrast).
- Kruse's low-visibility branch (q = 0.585 V^(1/3), V < 6 km) was derived from haze
  data; Kim et al. (2001) argue it is NOT supported by fog measurements and propose
  q -> 0 for V <= 0.5 km, i.e. wavelength-INDEPENDENT attenuation in dense fog.
  The two models therefore disagree strongly for V < 1 km: Kruse predicts a longer
  wavelength (e.g. 1550 nm) suffers noticeably less attenuation than 550 nm, while
  Kim predicts no wavelength advantage in dense fog.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: ln(1/0.02): Koschmieder constant for the 2 % contrast visibility definition.
KOSCHMIEDER_2PCT = 3.912

#: Conversion from extinction in 1/km (nepers) to dB/km: 10 / ln(10).
DB_PER_NEPER = 10.0 / np.log(10.0)

#: Reference wavelength of the visibility definition (nm).
REFERENCE_WAVELENGTH_NM = 550.0

#: Supported wavelength range (nm) — visible / near-IR validity of both models.
WAVELENGTH_RANGE_NM = (500.0, 2000.0)

#: Supported visibility range (km). Below ~0.05 km the Koschmieder definition itself
#: becomes unreliable; above 100 km attenuation is dominated by molecular effects.
VISIBILITY_RANGE_KM = (0.05, 100.0)


def _validate_inputs(
    visibility_km: ArrayLike, wavelength_nm: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and broadcast visibility (km) and wavelength (nm) inputs."""
    v = np.asarray(visibility_km, dtype=float)
    lam = np.asarray(wavelength_nm, dtype=float)
    if np.any(~np.isfinite(v)) or np.any(~np.isfinite(lam)):
        raise ValueError("visibility_km and wavelength_nm must be finite numbers.")
    if np.any(v < VISIBILITY_RANGE_KM[0]) or np.any(v > VISIBILITY_RANGE_KM[1]):
        raise ValueError(
            f"visibility_km must be within {VISIBILITY_RANGE_KM} km "
            f"(Koschmieder visibility validity range); got values outside it."
        )
    if np.any(lam < WAVELENGTH_RANGE_NM[0]) or np.any(lam > WAVELENGTH_RANGE_NM[1]):
        raise ValueError(
            f"wavelength_nm must be within {WAVELENGTH_RANGE_NM} nm "
            f"(visible/near-IR validity of the Kim/Kruse models)."
        )
    v, lam = np.broadcast_arrays(v, lam)
    return v.astype(float), lam.astype(float)


def kruse_q(visibility_km: ArrayLike) -> NDArray[np.float64] | float:
    """Kruse size-distribution exponent q(V).

    Source: Kruse et al. 1962, "Elements of Infrared Technology".

        q = 1.6              for V > 50 km
        q = 1.3              for 6 km < V <= 50 km
        q = 0.585 V^(1/3)    for V <= 6 km

    Parameters
    ----------
    visibility_km : visibility V in km, within ``VISIBILITY_RANGE_KM``.

    Returns
    -------
    q (dimensionless), scalar or array matching the input shape.
    """
    v = np.asarray(visibility_km, dtype=float)
    q = np.select(
        [v > 50.0, v > 6.0],
        [1.6, 1.3],
        default=0.0,
    ) + np.where(v <= 6.0, 0.585 * np.cbrt(v), 0.0)
    return float(q) if np.isscalar(visibility_km) else q


def kim_q(visibility_km: ArrayLike) -> NDArray[np.float64] | float:
    """Kim size-distribution exponent q(V).

    Source: Kim, McArthur, Korevaar, Proc. SPIE 4214, 2001 (Eq. for q(V)).

        q = 1.6              for V > 50 km
        q = 1.3              for 6 km < V <= 50 km
        q = 0.16 V + 0.34    for 1 km < V <= 6 km
        q = V - 0.5          for 0.5 km < V <= 1 km
        q = 0                for V <= 0.5 km

    In dense fog (V <= 0.5 km) attenuation becomes wavelength-independent.

    Parameters
    ----------
    visibility_km : visibility V in km, within ``VISIBILITY_RANGE_KM``.

    Returns
    -------
    q (dimensionless), scalar or array matching the input shape.
    """
    v = np.asarray(visibility_km, dtype=float)
    q = np.select(
        [v > 50.0, v > 6.0, v > 1.0, v > 0.5],
        [1.6, 1.3, 0.16 * v + 0.34, v - 0.5],
        default=0.0,
    )
    return float(q) if np.isscalar(visibility_km) else q


def _attenuation_db_km(
    visibility_km: ArrayLike, wavelength_nm: ArrayLike, q: NDArray[np.float64]
) -> NDArray[np.float64] | float:
    """Common Koschmieder-based attenuation kernel, dB/km."""
    v, lam = _validate_inputs(visibility_km, wavelength_nm)
    sigma = (KOSCHMIEDER_2PCT / v) * (lam / REFERENCE_WAVELENGTH_NM) ** (-q)  # 1/km
    alpha = DB_PER_NEPER * sigma  # dB/km
    if np.isscalar(visibility_km) and np.isscalar(wavelength_nm):
        return float(alpha)
    return alpha


def kruse_attenuation_db_km(
    visibility_km: ArrayLike, wavelength_nm: ArrayLike
) -> NDArray[np.float64] | float:
    """Kruse-model specific attenuation (dB/km).

    alpha = (10/ln 10) * (3.912 / V) * (lambda/550 nm)^(-q_Kruse(V))

    Source: Kruse et al. 1962. Units: V in km, lambda in nm, output dB/km.
    Validity: V in [0.05, 100] km, lambda in [500, 2000] nm; haze/fog aerosol only.
    Known limitation: the low-visibility branch overestimates the long-wavelength
    advantage in fog (see Kim et al. 2001).
    """
    v, lam = _validate_inputs(visibility_km, wavelength_nm)
    q = np.asarray(kruse_q(v), dtype=float)
    return _attenuation_db_km(visibility_km, wavelength_nm, q)


def kim_attenuation_db_km(
    visibility_km: ArrayLike, wavelength_nm: ArrayLike
) -> NDArray[np.float64] | float:
    """Kim-model specific attenuation (dB/km).

    alpha = (10/ln 10) * (3.912 / V) * (lambda/550 nm)^(-q_Kim(V))

    Source: Kim, McArthur, Korevaar, Proc. SPIE 4214, 2001.
    Units: V in km, lambda in nm, output dB/km.
    Validity: V in [0.05, 100] km, lambda in [500, 2000] nm; haze/fog aerosol only.
    For V <= 0.5 km (dense fog) the result is wavelength-independent (q = 0).
    """
    v, lam = _validate_inputs(visibility_km, wavelength_nm)
    q = np.asarray(kim_q(v), dtype=float)
    return _attenuation_db_km(visibility_km, wavelength_nm, q)
