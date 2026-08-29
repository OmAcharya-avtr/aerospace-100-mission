"""Classical closed-form inversion to path-averaged Cn2, with uncertainty.

Both single-sensor weak-regime inversions used here
(:func:`turbscope.scintillometer.invert_cn2_weak`,
:func:`turbscope.dimm.invert_cn2_from_variance`) are **exactly linear** in
their respective measured quantity (Cn2 is directly proportional to sigma_I^2
for the scintillometer, and to the DIMM differential variance for DIMM --
verified algebraically in the docstrings below and by a Hypothesis property
test in ``tests/test_inversion.py``). Linear error propagation is therefore
*exact*, not a small-perturbation approximation, for a given relative
measurement noise standard deviation: a `k`% relative error on the
measurement is exactly a `k`% relative error on the inverted Cn2.

:func:`fuse_inverse_variance` combines independent single-sensor estimates
into one multi-sensor closed-form estimate using the standard inverse
-variance-weighted combination of independent unbiased estimators (e.g.
Bevington, P. R. and Robinson, D. K. (2003), *Data Reduction and Error
Analysis for the Physical Sciences*, 3rd ed., McGraw-Hill, Ch. 4). This is a
classical statistics result, not specific to atmospheric optics, and is the
non-learned "closed-form" comparator this product's :mod:`turbscope.model`
benchmarks the learned regressor against alongside the single-sensor
baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dimm import invert_cn2_from_variance
from .scintillometer import invert_cn2_weak, is_weak_regime, rytov_variance

__all__ = [
    "PointEstimate",
    "MultiSensorEstimate",
    "invert_dimm_with_uncertainty",
    "invert_scintillometer_weak_with_uncertainty",
    "fuse_inverse_variance",
    "multi_sensor_closed_form_estimate",
]


@dataclass(frozen=True)
class PointEstimate:
    """A single Cn2_path estimate with 1-sigma uncertainty.

    Attributes
    ----------
    cn2_path : float
        Point estimate, m^-2/3.
    cn2_std : float
        1-sigma uncertainty, m^-2/3 (from linear propagation of the
        measurement's relative noise -- see module docstring).
    source : str
        Which sensor/method produced this estimate.
    """

    cn2_path: float
    cn2_std: float
    source: str


def _validate_relative_std(relative_std: float) -> float:
    r = float(relative_std)
    if not np.isfinite(r) or r < 0.0:
        raise ValueError(f"relative_std must be finite and >= 0 (got {relative_std!r}).")
    return r


def invert_scintillometer_weak_with_uncertainty(
    sigma_i2_measured: float,
    sigma_i2_relative_std: float,
    path_length_m: float,
    wavelength_m: float,
    wave_type: str = "spherical",
) -> PointEstimate:
    """Weak-regime scintillometer inversion with propagated uncertainty.

    Because ``Cn2 = sigma_I^2 / (C_w k^{7/6} L^{11/6})`` is linear in
    ``sigma_I^2``, ``std(Cn2)/Cn2 == std(sigma_I^2)/sigma_I^2`` exactly.

    Parameters
    ----------
    sigma_i2_measured : float
        Measured scintillation index (>= 0).
    sigma_i2_relative_std : float
        Relative (fractional) 1-sigma measurement noise, >= 0.
    path_length_m, wavelength_m, wave_type
        As in :func:`turbscope.scintillometer.invert_cn2_weak`.

    Returns
    -------
    PointEstimate

    Notes
    -----
    This is the classical single-sensor closed-form baseline this product's
    learned model is benchmarked against. It applies weak-theory unchanged
    regardless of whether the true regime is weak or saturated -- the
    documented failure mode quantified in
    ``validation/saturation_regime.py``.
    """
    cn2 = invert_cn2_weak(sigma_i2_measured, path_length_m, wavelength_m, wave_type)
    rel = _validate_relative_std(sigma_i2_relative_std)
    return PointEstimate(cn2_path=cn2, cn2_std=abs(cn2) * rel, source="scintillometer_weak")


def invert_dimm_with_uncertainty(
    variance_rad2: float,
    variance_relative_std: float,
    path_length_m: float,
    wavelength_m: float,
    aperture_diam_m: float,
    separation_m: float,
    component: str = "longitudinal",
    zenith_deg: float = 0.0,
) -> PointEstimate:
    """DIMM inversion with propagated uncertainty.

    ``Cn2 = r0^{-5/3} / (0.423 k^2 sec(zeta) L)`` and
    ``r0 = D (var / (0.358 (lam/D)^2 bracket))^{-3/5}`` compose to
    ``Cn2 = var / (0.423 k^2 sec(zeta) L D^{5/3}) * (0.358 (lam/D)^2
    bracket)^{-1}`` -- i.e. **Cn2 is exactly linear in the measured
    variance** (verified in ``tests/test_inversion.py``), so
    ``std(Cn2)/Cn2 == std(var)/var`` exactly, as for the scintillometer.

    Parameters, returns and raised errors mirror
    :func:`turbscope.dimm.invert_cn2_from_variance`, plus
    ``variance_relative_std`` (relative 1-sigma noise, >= 0).
    """
    cn2 = invert_cn2_from_variance(
        variance_rad2, path_length_m, wavelength_m, aperture_diam_m, separation_m, component,
        zenith_deg,
    )
    rel = _validate_relative_std(variance_relative_std)
    return PointEstimate(cn2_path=cn2, cn2_std=abs(cn2) * rel, source=f"dimm_{component}")


def fuse_inverse_variance(estimates: list[PointEstimate]) -> PointEstimate:
    r"""Combine independent Cn2 estimates by inverse-variance weighting.

    .. math::

        \hat{C} = \frac{\sum_i C_i / \sigma_i^2}{\sum_i 1/\sigma_i^2}, \qquad
        \sigma_{\hat C}^2 = \left(\sum_i 1/\sigma_i^2\right)^{-1}

    the minimum-variance unbiased linear combination of independent unbiased
    estimators (Bevington & Robinson 2003, Ch. 4).

    Parameters
    ----------
    estimates : list of PointEstimate
        At least one estimate; each ``cn2_std`` must be > 0.

    Returns
    -------
    PointEstimate
        ``source`` is ``"fused(<sources>)"``.

    Raises
    ------
    ValueError
        If ``estimates`` is empty or any ``cn2_std`` is <= 0 or non-finite.
    """
    if not estimates:
        raise ValueError("estimates must be non-empty.")
    weights = []
    for e in estimates:
        if not np.isfinite(e.cn2_std) or e.cn2_std <= 0.0:
            raise ValueError(f"Every estimate needs cn2_std > 0 (got {e.cn2_std!r}).")
        weights.append(1.0 / e.cn2_std**2)
    weights_arr = np.asarray(weights, dtype=float)
    values = np.asarray([e.cn2_path for e in estimates], dtype=float)
    w_sum = float(np.sum(weights_arr))
    combined = float(np.sum(weights_arr * values) / w_sum)
    combined_std = float(np.sqrt(1.0 / w_sum))
    names = "+".join(e.source for e in estimates)
    return PointEstimate(cn2_path=combined, cn2_std=combined_std, source=f"fused({names})")


@dataclass(frozen=True)
class MultiSensorEstimate:
    """Combined closed-form multi-sensor estimate plus the inputs it was built
    from, for diagnostics.

    Attributes
    ----------
    fused : PointEstimate
        The inverse-variance combination of all individual estimates.
    individual : tuple of PointEstimate
        The per-sensor estimates that went into the fusion.
    weak_regime_scint : bool
        Whether the scintillometer measurement is self-consistent with the
        weak-fluctuation assumption (its own inverted Cn2 implies a Rytov
        variance below :data:`turbscope.constants.WEAK_REGIME_MAX_SIGMA_R2`).
        False is a warning flag, not an exception: the fused estimate is
        still returned, but should be treated with caution -- see
        ``validation/saturation_regime.py``.
    """

    fused: PointEstimate
    individual: tuple[PointEstimate, ...]
    weak_regime_scint: bool


def multi_sensor_closed_form_estimate(
    sigma_i2_measured: float,
    var_long_measured: float,
    var_trans_measured: float,
    path_length_m: float,
    scint_wavelength_m: float,
    scint_wave_type: str,
    dimm_wavelength_m: float,
    aperture_diam_m: float,
    separation_m: float,
    scint_relative_std: float,
    dimm_relative_std: float,
) -> MultiSensorEstimate:
    """Classical (non-learned) multi-sensor fusion: three closed-form
    single-sensor estimates, combined by inverse-variance weighting.

    This is a legitimate *classical* multi-sensor estimator -- distinct from
    both the single-sensor baseline (:func:`invert_scintillometer_weak_with_uncertainty`
    alone) and the learned model (:mod:`turbscope.model`). It is reported
    alongside both in ``validation/VALIDATION.md`` for context, but the
    mission-mandated comparator for the learned model is the single-sensor
    weak-inversion baseline.

    Parameters
    ----------
    sigma_i2_measured, var_long_measured, var_trans_measured : float
        Sensor readings (scintillometer, DIMM longitudinal, DIMM transverse).
    path_length_m : float
        Known path length, m.
    scint_wavelength_m, scint_wave_type, dimm_wavelength_m, aperture_diam_m,
    separation_m : as in the individual forward/inverse models.
    scint_relative_std, dimm_relative_std : float
        Assumed relative 1-sigma measurement noise for each sensor type.

    Returns
    -------
    MultiSensorEstimate
    """
    scint = invert_scintillometer_weak_with_uncertainty(
        sigma_i2_measured, scint_relative_std, path_length_m, scint_wavelength_m, scint_wave_type
    )
    dimm_long = invert_dimm_with_uncertainty(
        var_long_measured, dimm_relative_std, path_length_m, dimm_wavelength_m,
        aperture_diam_m, separation_m, "longitudinal",
    )
    dimm_trans = invert_dimm_with_uncertainty(
        var_trans_measured, dimm_relative_std, path_length_m, dimm_wavelength_m,
        aperture_diam_m, separation_m, "transverse",
    )
    individual = (scint, dimm_long, dimm_trans)
    fused = fuse_inverse_variance(list(individual))
    r_var = rytov_variance(scint.cn2_path, path_length_m, scint_wavelength_m, scint_wave_type)
    return MultiSensorEstimate(
        fused=fused, individual=individual, weak_regime_scint=bool(is_weak_regime(r_var))
    )
