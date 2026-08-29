r"""Closed-form inversion of sensor readings to a path-averaged ``Cn2``.

This module is the **classical baseline** of the product and was implemented and
validated before any learned model existed.

What each inversion actually returns
------------------------------------
Neither sensor measures ``Cn2`` at a point, and the two do not measure the same
average.  Assuming a uniform ``Cn2`` and inverting the forward model returns the
*kernel-weighted* path average exactly:

``Cn2_hat_scint = int W_sc(u) Cn2 du / int W_sc(u) du``,
``Cn2_hat_dimm  = int W_co(u) Cn2 du / int W_co(u) du``,

with ``W_sc`` and ``W_co`` from :mod:`turbscope.geometry`.  This is an algebraic
identity, exact to quadrature error, and is checked with Hypothesis in
``tests/test_properties.py``.  It also means that on a non-uniform path the two
inversions **disagree by construction** -- that disagreement is the "each sensor
has its own bias" of the problem statement, not a defect.

Uncertainty
-----------
* Scintillometer: the sampling variance of ``sigma_I^2 = var(I)/mean(I)^2`` from
  ``N`` independent gamma-gamma irradiances is obtained by the delta method from
  the exact gamma-gamma moments (Al-Habash, Andrews & Phillips 2001), then mapped
  through ``d beta_0^2 / d sigma_I^2``.  The interval is first-order Gaussian in
  ``log Cn2``; it is not valid where the mapping is singular (saturation).
* DIMM: the sample variance of ``N`` Gaussian differential displacements is
  exactly ``(sigma^2 + sigma_noise^2) chi^2_(N-1)/(N-1)``, so the interval is the
  exact chi-square interval with the calibrated noise floor subtracted.

Saturation, the documented failure regime
-----------------------------------------
``sigma_I^2(beta_0^2)`` rises, peaks near ``beta_0^2 ~ 7.3`` at
``sigma_I^2 ~ 1.69``, and decays to 1 (Andrews & Phillips 2005, ch. 9; the effect
was measured by Gracheva & Gurvich 1965).  Consequently:

* a measurement in ``(1, 1.69]`` has **two** solutions -- the inversion is
  multi-valued and no amount of care in the estimator fixes it;
* a measurement above the peak has **no** solution;
* the sensitivity ``d sigma_I^2 / d beta_0^2`` passes through zero at the peak, so
  the propagated uncertainty diverges there.

:func:`saturation_report` and :func:`scintillation_branches` expose all three
facts; the package never silently returns the weak-regime answer in that regime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

from ._validate import check_count, check_positive, check_probability
from .dimm import cn2_average_from_fried, r0_from_dimm_variance
from .geometry import PathGeometry
from .scintillation import (
    WEAK_REGIME_BETA0_SQ,
    aperture_parameter_sq,
    gamma_gamma_parameters,
    saturation_peak,
    scintillation_index,
    uniform_cn2_from_beta0_sq,
)

__all__ = [
    "Cn2Estimate",
    "SaturationReport",
    "invert_dimm",
    "invert_scintillation",
    "saturation_report",
    "scintillation_branches",
    "scintillation_index_relative_sigma",
]

_LOG_B_MIN = -12.0
_LOG_B_MAX = 4.0


@dataclass(frozen=True)
class Cn2Estimate:
    """A path-averaged ``Cn2`` estimate with an interval and provenance.

    Attributes
    ----------
    cn2, cn2_lower, cn2_upper
        Point estimate and interval bounds, m^(-2/3).
    coverage
        Nominal coverage of ``[cn2_lower, cn2_upper]``.
    kernel
        ``"scintillation"`` or ``"coherence"`` -- which weighted path average this
        number estimates.
    method
        Inversion used (``"weak"``, ``"saturation"``, ``"dimm"``).
    relative_sigma
        First-order relative standard deviation of the point estimate.
    branches
        All ``Cn2`` solutions consistent with the reading (length > 1 means the
        inversion is genuinely ambiguous).
    valid
        ``False`` when the reading falls outside the model's invertible range.
    notes
        Human-readable caveats attached by the inversion.
    """

    cn2: float
    cn2_lower: float
    cn2_upper: float
    coverage: float
    kernel: str
    method: str
    relative_sigma: float
    branches: tuple[float, ...] = ()
    valid: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ambiguous(self) -> bool:
        """True when more than one turbulence strength explains the reading."""
        return len(self.branches) > 1


@dataclass(frozen=True)
class SaturationReport:
    """Where the analytic scintillation inversion breaks, for one aperture."""

    aperture_d_sq: float
    beta0_sq_peak: float
    sigma_i2_peak: float
    sigma_i2_asymptote: float
    ambiguous_sigma_i2_range: tuple[float, float]
    beta0_sq_weak_limit: float


def scintillation_index_relative_sigma(
    beta0_sq: float, aperture_d_sq: float, n_samples: int
) -> float:
    """Relative standard deviation of the estimator ``sigma_I^2 = var(I)/mean(I)^2``.

    Delta method on the exact gamma-gamma moments
    ``m_n = prod_{j=1..n} (1 + j/alpha)(1 + j/beta)`` (Al-Habash, Andrews &
    Phillips 2001).  With ``m1 = 1``,

    ``Var(T) ~= [ (m4 - m2^2) + 4 m2^2 (m2 - 1) - 4 m2 (m3 - m2) ] / N``

    and the returned value is ``sqrt(Var(T)) / (m2 - 1)``.  Valid for
    ``N`` large enough that the estimator is approximately normal (``N >~ 100``).
    """
    n = check_count("n_samples", n_samples, minimum=10)
    alpha, beta = gamma_gamma_parameters(beta0_sq, aperture_d_sq)
    m2 = (1.0 + 1.0 / alpha) * (1.0 + 1.0 / beta)
    m3 = m2 * (1.0 + 2.0 / alpha) * (1.0 + 2.0 / beta)
    m4 = m3 * (1.0 + 3.0 / alpha) * (1.0 + 3.0 / beta)
    var_t = ((m4 - m2 * m2) + 4.0 * m2 * m2 * (m2 - 1.0) - 4.0 * m2 * (m3 - m2)) / n
    sigma_i2 = m2 - 1.0
    if sigma_i2 <= 0.0 or var_t <= 0.0:  # pragma: no cover - defensive
        raise ValueError("degenerate scintillation-index moments")
    return float(np.sqrt(var_t) / sigma_i2)


def scintillation_branches(
    sigma_i2: float, aperture_d_sq: float = 0.0, *, n_scan: int = 6401
) -> tuple[float, ...]:
    """All ``beta_0^2`` values whose scintillation index equals ``sigma_i2``.

    A log-spaced scan over ``beta_0^2 in [1e-12, 1e4]`` locates sign changes and
    Brent's method refines each one.  Returns an empty tuple when the reading
    exceeds the model's maximum attainable ``sigma_I^2`` (and, at the very bottom
    of the range, when ``sigma_i2`` is below ``1e-12`` -- there the weak-regime
    identity ``beta_0^2 = sigma_I^2`` is exact to better than 1e-9 anyway).
    """
    target = check_positive("sigma_i2", sigma_i2)
    d2 = check_positive("aperture_d_sq", aperture_d_sq, allow_zero=True)
    grid = np.logspace(_LOG_B_MIN, _LOG_B_MAX, int(n_scan))
    values = scintillation_index(grid, d2) - target
    roots: list[float] = []
    sign_change = np.nonzero(values[:-1] * values[1:] < 0.0)[0]
    for i in sign_change:
        roots.append(
            float(
                optimize.brentq(
                    lambda b: float(scintillation_index(b, d2)) - target,
                    grid[i],
                    grid[i + 1],
                    xtol=1e-12,
                    rtol=1e-12,
                )
            )
        )
    return tuple(sorted(roots))


def saturation_report(aperture_d_sq: float = 0.0) -> SaturationReport:
    """Summarise the invertible range of the scintillation forward model."""
    d2 = check_positive("aperture_d_sq", aperture_d_sq, allow_zero=True)
    b_peak, s_peak = saturation_peak(d2)
    asymptote = float(scintillation_index(10.0**_LOG_B_MAX, d2))
    return SaturationReport(
        aperture_d_sq=d2,
        beta0_sq_peak=b_peak,
        sigma_i2_peak=s_peak,
        sigma_i2_asymptote=asymptote,
        ambiguous_sigma_i2_range=(asymptote, s_peak),
        beta0_sq_weak_limit=WEAK_REGIME_BETA0_SQ,
    )


def invert_scintillation(
    sigma_i2: float,
    path: PathGeometry,
    *,
    n_samples: int = 1000,
    receiver_diameter_m: float = 0.0,
    method: str = "saturation",
    coverage: float = 0.90,
) -> Cn2Estimate:
    """Invert a scintillometer reading to the scintillation-weighted ``<Cn2>``.

    Parameters
    ----------
    sigma_i2
        Measured irradiance scintillation index, dimensionless.
    path
        Propagation geometry.
    n_samples
        Number of independent irradiance samples behind the reading; sets the
        statistical uncertainty.
    receiver_diameter_m
        Receiver aperture diameter, m (0 = point receiver).
    method
        ``"weak"``  -- the textbook linear inversion ``beta_0^2 = sigma_I^2``,
        valid only for ``beta_0^2 < 0.3``;
        ``"saturation"`` -- root-find the Andrews-Phillips index, returning the
        **lowest** branch as the point estimate and all branches in ``branches``.
    coverage
        Nominal interval coverage, e.g. 0.90.

    Returns
    -------
    Cn2Estimate
        ``valid=False`` when the reading is above the maximum attainable
        ``sigma_I^2`` (saturated beyond recovery).
    """
    s_meas = check_positive("sigma_i2", sigma_i2)
    if method not in ("weak", "saturation"):
        raise ValueError(f"method must be 'weak' or 'saturation', got {method!r}")
    cov = check_probability("coverage", coverage)
    n = check_count("n_samples", n_samples, minimum=10)
    d2 = aperture_parameter_sq(receiver_diameter_m, path)
    notes: list[str] = []

    if method == "weak":
        beta_hat = s_meas
        branches = (beta_hat,)
        dsigma_dbeta = 1.0
        valid = True
        if beta_hat > WEAK_REGIME_BETA0_SQ:
            notes.append(
                f"beta_0^2 = {beta_hat:.3g} exceeds the weak-fluctuation limit "
                f"{WEAK_REGIME_BETA0_SQ}; the linear inversion under-estimates Cn2"
            )
        if d2 > 0.0:
            notes.append(
                "the weak linear inversion ignores aperture averaging; use a point "
                "receiver or method='saturation'"
            )
    else:
        branches = scintillation_branches(s_meas, d2)
        if not branches and s_meas <= saturation_peak(d2)[1]:
            # below the scan floor: the weak-fluctuation identity is exact here
            branches = (s_meas,)
        if not branches:
            report = saturation_report(d2)
            return Cn2Estimate(
                cn2=float("nan"),
                cn2_lower=float("nan"),
                cn2_upper=float("nan"),
                coverage=cov,
                kernel="scintillation",
                method=method,
                relative_sigma=float("nan"),
                branches=(),
                valid=False,
                notes=(
                    f"sigma_I^2 = {s_meas:.4g} exceeds the maximum attainable "
                    f"{report.sigma_i2_peak:.4g} for d^2 = {d2:.4g}; no Cn2 explains "
                    "this reading under the Andrews-Phillips model",
                ),
            )
        beta_hat = branches[0]
        valid = True
        eps = 1e-4 * beta_hat
        dsigma_dbeta = float(
            (scintillation_index(beta_hat + eps, d2) - scintillation_index(beta_hat - eps, d2))
            / (2.0 * eps)
        )
        if len(branches) > 1:
            notes.append(
                f"reading is multi-valued: {len(branches)} branches at beta_0^2 = "
                + ", ".join(f"{b:.4g}" for b in branches)
                + "; the lowest branch is reported"
            )
        if beta_hat > WEAK_REGIME_BETA0_SQ:
            notes.append("beta_0^2 is outside the weak-fluctuation regime")

    rel_sigma_s = scintillation_index_relative_sigma(beta_hat, d2, n)
    if dsigma_dbeta <= 0.0:
        rel_sigma_beta = float("inf")
        notes.append("sensitivity d(sigma_I^2)/d(beta_0^2) is non-positive; interval unbounded")
    else:
        # delta method: sd(beta) = sd(sigma_I^2) / (d sigma_I^2 / d beta_0^2)
        rel_sigma_beta = rel_sigma_s * scintillation_index(beta_hat, d2) / (
            dsigma_dbeta * beta_hat
        )

    cn2 = uniform_cn2_from_beta0_sq(beta_hat, path)
    z = float(stats.norm.ppf(0.5 + 0.5 * cov))
    if np.isfinite(rel_sigma_beta):
        lower = cn2 * float(np.exp(-z * rel_sigma_beta))
        upper = cn2 * float(np.exp(+z * rel_sigma_beta))
    else:
        lower, upper = 0.0, float("inf")
    return Cn2Estimate(
        cn2=float(cn2),
        cn2_lower=float(lower),
        cn2_upper=float(upper),
        coverage=cov,
        kernel="scintillation",
        method=method,
        relative_sigma=float(rel_sigma_beta),
        branches=tuple(uniform_cn2_from_beta0_sq(b, path) for b in branches),
        valid=valid,
        notes=tuple(notes),
    )


def invert_dimm(
    variance_rad2: float,
    path: PathGeometry,
    *,
    subaperture_m: float,
    baseline_m: float,
    component: str = "longitudinal",
    n_frames: int = 500,
    noise_variance_rad2: float = 0.0,
    coverage: float = 0.90,
) -> Cn2Estimate:
    """Invert a DIMM variance to the coherence-weighted ``<Cn2>``.

    Chain: ``sigma^2 -> r_0`` (Sarazin & Roddier 1990) ``-> <Cn2>_co`` (Fried 1966).
    The calibrated centroid-noise variance is subtracted first.

    The interval is the **exact** chi-square interval for a sample variance of
    ``n_frames`` Gaussian differential displacements, with the noise floor
    subtracted from both bounds; no linearisation is involved.  ``valid=False``
    when the noise-corrected variance is non-positive, which is the real failure
    mode of a DIMM in good seeing.
    """
    var = check_positive("variance_rad2", variance_rad2)
    noise = check_positive("noise_variance_rad2", noise_variance_rad2, allow_zero=True)
    n = check_count("n_frames", n_frames, minimum=10)
    cov = check_probability("coverage", coverage)
    dof = n - 1
    notes: list[str] = []

    corrected = var - noise
    if corrected <= 0.0:
        return Cn2Estimate(
            cn2=float("nan"),
            cn2_lower=float("nan"),
            cn2_upper=float("nan"),
            coverage=cov,
            kernel="coherence",
            method="dimm",
            relative_sigma=float("nan"),
            branches=(),
            valid=False,
            notes=(
                f"measured variance {var:.4g} rad^2 is at or below the calibrated "
                f"centroid-noise floor {noise:.4g} rad^2; no seeing signal",
            ),
        )

    alpha = 1.0 - cov
    lo_scale = dof / stats.chi2.ppf(1.0 - alpha / 2.0, dof)
    hi_scale = dof / stats.chi2.ppf(alpha / 2.0, dof)
    var_lo = max(var * lo_scale - noise, 0.0)
    var_hi = var * hi_scale - noise
    if var_lo <= 0.0:
        notes.append("lower interval bound truncated at the centroid-noise floor")

    def to_cn2(v: float) -> float:
        r0 = r0_from_dimm_variance(v, path.wavelength_m, subaperture_m, baseline_m, component)
        return cn2_average_from_fried(r0, path)

    cn2 = to_cn2(corrected)
    cn2_lo = to_cn2(var_lo) if var_lo > 0.0 else 0.0
    cn2_hi = to_cn2(var_hi)
    rel_sigma = float(np.sqrt(2.0 / dof) * var / corrected)
    if noise > 0.0 and corrected < 3.0 * noise:
        notes.append(
            "atmospheric variance is less than 3x the noise floor; the estimate is "
            "noise-dominated"
        )
    return Cn2Estimate(
        cn2=float(cn2),
        cn2_lower=float(cn2_lo),
        cn2_upper=float(cn2_hi),
        coverage=cov,
        kernel="coherence",
        method="dimm",
        relative_sigma=rel_sigma,
        branches=(float(cn2),),
        valid=True,
        notes=tuple(notes),
    )
