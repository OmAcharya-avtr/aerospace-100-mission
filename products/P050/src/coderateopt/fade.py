"""Fade statistics for an atmospheric optical link, as availability functions.

Everything in this module answers exactly one question: given the link's
nominal (clear-sky) margin in dB, what is the probability that the
instantaneous received power stays at or above a stated threshold? That
probability is what the rate optimiser in :mod:`coderateopt.milp` consumes,
and nothing else about the channel is used.

Conventions and assumptions, stated once here because every number downstream
inherits them
-------------------------------------------------------------------------------
* ``I`` is normalised irradiance with ``E[I] = 1`` (mean-preserving
  normalisation). The dB fade is ``P_dB = 10 log10(I)``, so ``P_dB`` has a
  *negative* mean: the dB of the mean is not the mean of the dB.
  ``median_preserving=True`` switches the lognormal model to ``median(I) = 1``
  instead, which is the other convention in common use. The two differ by
  ``-2.1715 * sigma_lnI**2`` dB in the mean and change availability numbers,
  so the choice is explicit rather than defaulted silently.
* The scintillation index is ``sigma_I**2 = E[I**2]/E[I]**2 - 1``
  (Andrews & Phillips, *Laser Beam Propagation through Random Media*, 2nd ed.,
  SPIE Press, 2005). Both models below are parameterised by it.
* Fades are treated as a *marginal* distribution only. Correlation time,
  fade duration and fade-rate statistics are not modelled here; see the
  Limitations section of the README and ITU-R P.1623 ("Prediction method of
  fade dynamics on Earth-space paths") for the methodology that does model
  them. A marginal-only treatment answers long-run availability questions and
  does not answer "how long will an outage last".
* Validity: the lognormal model is a weak-turbulence model
  (``sigma_I**2 << 1``, conventionally below about 0.3). The gamma-gamma model
  is intended for weak-to-strong turbulence and is the one to use above that.
  Neither includes pointing jitter, beam wander or aperture averaging; those
  must be folded into the scintillation index before it is passed in.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import integrate, optimize, special, stats

__all__ = [
    "AvailabilityModel",
    "EmpiricalFade",
    "GammaGammaFade",
    "LognormalFade",
    "db_to_linear",
    "gamma_gamma_pdf",
    "linear_to_db",
    "scintillation_from_log_amplitude",
    "sigma_ln_i_from_scintillation",
]

_DB_PER_NEPER = 10.0 / math.log(10.0)  # 4.342944819032518


def linear_to_db(value: float | np.ndarray) -> float | np.ndarray:
    """Power ratio -> dB. ``10 log10(value)``. Units: dimensionless -> dB."""
    arr = np.asarray(value, dtype=float)
    if np.any(arr <= 0.0):
        raise ValueError("linear_to_db requires strictly positive power ratios")
    out = 10.0 * np.log10(arr)
    return float(out) if np.isscalar(value) or out.ndim == 0 else out


def db_to_linear(value: float | np.ndarray) -> float | np.ndarray:
    """dB -> power ratio. ``10 ** (value / 10)``. Units: dB -> dimensionless."""
    out = np.power(10.0, np.asarray(value, dtype=float) / 10.0)
    return float(out) if np.isscalar(value) or out.ndim == 0 else out


def sigma_ln_i_from_scintillation(scintillation_index: float) -> float:
    """Lognormal log-irradiance standard deviation from the scintillation index.

    ``sigma_lnI = sqrt(ln(1 + sigma_I**2))``, the inverse of
    ``sigma_I**2 = exp(sigma_lnI**2) - 1`` for a lognormal irradiance
    (Andrews & Phillips 2005, lognormal irradiance model). Units: dimensionless.
    """
    if not np.isfinite(scintillation_index) or scintillation_index <= 0.0:
        raise ValueError(
            f"scintillation_index must be finite and > 0, got {scintillation_index!r}"
        )
    return math.sqrt(math.log1p(scintillation_index))


def scintillation_from_log_amplitude(sigma_chi: float) -> float:
    """Scintillation index from the log-amplitude standard deviation.

    ``sigma_I**2 = exp(4 sigma_chi**2) - 1`` for lognormal irradiance, since
    ``I = exp(2 chi)`` (Andrews & Phillips 2005). Units: dimensionless.
    """
    if not np.isfinite(sigma_chi) or sigma_chi <= 0.0:
        raise ValueError(f"sigma_chi must be finite and > 0, got {sigma_chi!r}")
    return math.expm1(4.0 * sigma_chi**2)


class AvailabilityModel:
    """Base class: a marginal fade model that reports availability in dB terms.

    A subclass implements :meth:`exceedance_db`, the probability that the dB
    fade ``P_dB = 10 log10(I)`` is at or above a stated level.
    """

    #: Short label used in reports and CLI output.
    name: str = "availability-model"

    def exceedance_db(self, level_db: float | np.ndarray) -> float | np.ndarray:
        """``P[P_dB >= level_db]``. Units: dB in, probability out."""
        raise NotImplementedError

    def availability(
        self, margin_db: float | np.ndarray, threshold_db: float | np.ndarray
    ) -> float | np.ndarray:
        """``P[margin_db + P_dB >= threshold_db]``.

        Parameters
        ----------
        margin_db
            Nominal (clear-sky) received SNR or received power, dB, on the same
            scale as ``threshold_db``.
        threshold_db
            MODCOD demodulation threshold on that same scale, dB.

        Returns
        -------
        Probability in [0, 1], dimensionless.
        """
        level = np.asarray(threshold_db, dtype=float) - np.asarray(margin_db, dtype=float)
        return self.exceedance_db(level)

    def quantile_db(self, probability: float) -> float:
        """Fade level ``x`` in dB with ``P[P_dB >= x] = probability``.

        Solved by bisection on :meth:`exceedance_db`, so it works for any
        subclass without a closed-form inverse. Units: probability in, dB out.
        """
        if not 0.0 < probability < 1.0:
            raise ValueError(f"probability must be in (0, 1), got {probability!r}")
        lo, hi = -200.0, 50.0
        f = lambda x: float(np.asarray(self.exceedance_db(x))) - probability  # noqa: E731
        if f(lo) < 0.0 or f(hi) > 0.0:
            raise ValueError("requested quantile lies outside the searched -200..50 dB range")
        return float(optimize.brentq(f, lo, hi, xtol=1e-10, rtol=1e-12))

    def sample_db(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``size`` dB fades. Default implementation inverts the CDF."""
        u = rng.uniform(size=int(size))
        return np.array([self.quantile_db(float(p)) for p in u])


@dataclass(frozen=True)
class LognormalFade(AvailabilityModel):
    """Lognormal irradiance, the weak-turbulence marginal model.

    ``ln I ~ Normal(mu, sigma_lnI**2)`` with
    ``sigma_lnI**2 = ln(1 + scintillation_index)``, and ``mu`` fixed by the
    normalisation: ``mu = -sigma_lnI**2 / 2`` for ``E[I] = 1``, or ``mu = 0``
    for ``median(I) = 1``. Reference: Andrews & Phillips 2005, lognormal
    irradiance model; validity ``scintillation_index`` below about 0.3.

    Attributes
    ----------
    scintillation_index
        ``sigma_I**2``, dimensionless, > 0.
    median_preserving
        If True use ``median(I) = 1`` instead of ``E[I] = 1``.
    """

    scintillation_index: float
    median_preserving: bool = False
    name: str = "lognormal"

    def __post_init__(self) -> None:
        sigma_ln_i_from_scintillation(self.scintillation_index)

    @property
    def sigma_ln_i(self) -> float:
        """Standard deviation of ``ln I``, dimensionless."""
        return sigma_ln_i_from_scintillation(self.scintillation_index)

    @property
    def sigma_db(self) -> float:
        """Standard deviation of ``10 log10(I)``, dB."""
        return _DB_PER_NEPER * self.sigma_ln_i

    @property
    def mean_db(self) -> float:
        """Mean of ``10 log10(I)``, dB. Negative under mean-preserving scaling."""
        if self.median_preserving:
            return 0.0
        return -_DB_PER_NEPER * self.sigma_ln_i**2 / 2.0

    def exceedance_db(self, level_db: float | np.ndarray) -> float | np.ndarray:
        level = np.asarray(level_db, dtype=float)
        z = (self.mean_db - level) / self.sigma_db
        out = stats.norm.cdf(z)
        return float(out) if out.ndim == 0 else out

    def sample_db(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return self.mean_db + self.sigma_db * rng.standard_normal(int(size))


def gamma_gamma_pdf(irradiance: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Gamma-gamma irradiance pdf, ``E[I] = 1``.

    ``f(I) = 2 (alpha beta)**((alpha+beta)/2) / (Gamma(alpha) Gamma(beta))
             * I**((alpha+beta)/2 - 1) * K_{alpha-beta}(2 sqrt(alpha beta I))``

    Reference: Al-Habash, Andrews and Phillips, *Optical Engineering* 40(8),
    2001 (gamma-gamma irradiance model); also Andrews & Phillips 2005.
    ``alpha`` and ``beta`` are the effective numbers of large- and small-scale
    scatterers, dimensionless and > 0. Units: dimensionless in, density out.
    """
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError(f"alpha and beta must be > 0, got alpha={alpha!r}, beta={beta!r}")
    x = np.asarray(irradiance, dtype=float)
    out = np.zeros_like(x)
    pos = x > 0.0
    ab = alpha * beta
    log_pref = (
        math.log(2.0)
        + 0.5 * (alpha + beta) * math.log(ab)
        - special.gammaln(alpha)
        - special.gammaln(beta)
    )
    order = alpha - beta
    arg = 2.0 * np.sqrt(ab * x[pos])
    # kve = exp(arg) * kv, used to keep the Bessel factor in range for large arg.
    log_bessel = np.log(special.kve(order, arg)) - arg
    out[pos] = np.exp(log_pref + (0.5 * (alpha + beta) - 1.0) * np.log(x[pos]) + log_bessel)
    return out


@dataclass(frozen=True)
class GammaGammaFade(AvailabilityModel):
    """Gamma-gamma irradiance, weak-to-strong turbulence.

    The scintillation index of a gamma-gamma variate is
    ``sigma_I**2 = 1/alpha + 1/beta + 1/(alpha beta)`` (Al-Habash, Andrews and
    Phillips, *Optical Engineering* 40(8), 2001). Both moment identities are
    checked numerically in ``validation/validate_fade_moments.py`` rather than
    taken on trust.

    The CDF has no elementary form here, so exceedance is obtained by adaptive
    quadrature of :func:`gamma_gamma_pdf` (``scipy.integrate.quad``), which is
    accurate but roughly 10**4 times slower per evaluation than the lognormal
    model. For optimisation loops, build an :class:`EmpiricalFade` from samples
    or precompute a grid.
    """

    alpha: float
    beta: float
    name: str = "gamma-gamma"

    def __post_init__(self) -> None:
        if self.alpha <= 0.0 or self.beta <= 0.0:
            raise ValueError(
                f"alpha and beta must be > 0, got alpha={self.alpha!r}, beta={self.beta!r}"
            )

    @classmethod
    def from_scintillation(cls, scintillation_index: float, ratio: float = 1.0) -> GammaGammaFade:
        """Pick ``(alpha, beta)`` reproducing a scintillation index.

        ``ratio = beta / alpha`` in (0, 1]; ``ratio = 1`` gives the symmetric
        solution ``alpha = beta``. Solving
        ``sigma_I**2 = 1/alpha + 1/beta + 1/(alpha beta)`` with
        ``beta = ratio * alpha`` gives a quadratic in ``alpha``.
        """
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"ratio must be in (0, 1], got {ratio!r}")
        if not np.isfinite(scintillation_index) or scintillation_index <= 0.0:
            raise ValueError(
                f"scintillation_index must be finite and > 0, got {scintillation_index!r}"
            )
        # s * r * a**2 - (1 + r) * a - 1 = 0
        s, r = float(scintillation_index), float(ratio)
        a = (1.0 + r + math.sqrt((1.0 + r) ** 2 + 4.0 * s * r)) / (2.0 * s * r)
        return cls(alpha=a, beta=r * a)

    @property
    def scintillation_index(self) -> float:
        """``1/alpha + 1/beta + 1/(alpha beta)``, dimensionless."""
        return 1.0 / self.alpha + 1.0 / self.beta + 1.0 / (self.alpha * self.beta)

    def _density(self, t: float) -> float:
        return float(gamma_gamma_pdf(np.array([t]), self.alpha, self.beta)[0])

    def survival_linear(self, irradiance: float) -> float:
        """``P[I >= irradiance]`` by adaptive quadrature.

        The integral is taken over the *shorter* side of the mode: below
        ``I = 1`` the lower tail is integrated and subtracted, above it the
        upper tail is integrated directly to infinity. Integrating 0 to a
        large upper limit in one call makes ``quad`` miss the mass near the
        mode entirely and silently return a near-zero answer, which is how
        this function was wrong the first time it was written.
        """
        if irradiance <= 0.0:
            return 1.0
        if irradiance <= 1.0:
            lower, _ = integrate.quad(
                self._density, 0.0, irradiance, limit=200, epsabs=1e-13, epsrel=1e-11
            )
            return float(min(max(1.0 - lower, 0.0), 1.0))
        upper, _ = integrate.quad(
            self._density, irradiance, np.inf, limit=200, epsabs=1e-13, epsrel=1e-11
        )
        return float(min(max(upper, 0.0), 1.0))

    def exceedance_db(self, level_db: float | np.ndarray) -> float | np.ndarray:
        level = np.atleast_1d(np.asarray(level_db, dtype=float))
        out = np.array([self.survival_linear(float(db_to_linear(x))) for x in level])
        if np.ndim(level_db) == 0:
            return float(out[0])
        return out


@dataclass(frozen=True)
class EmpiricalFade(AvailabilityModel):
    """Availability straight from a sample of dB fades, no parametric model.

    Exceedance is the right-continuous empirical survival function of the
    supplied sample: ``P[P_dB >= x] = (#{s_i >= x}) / n``. The resolution floor
    is ``1/n``: with 10**4 samples nothing can be said about availability
    targets beyond 0.9999. That floor is reported by
    :attr:`resolution` and is the reason this class exists as an explicit
    alternative to the parametric models rather than as their replacement.
    """

    samples_db: np.ndarray
    name: str = "empirical"

    def __post_init__(self) -> None:
        arr = np.asarray(self.samples_db, dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("samples_db must be a non-empty 1-D array of dB fades")
        if not np.all(np.isfinite(arr)):
            raise ValueError("samples_db must be finite")
        object.__setattr__(self, "samples_db", np.sort(arr))

    @property
    def resolution(self) -> float:
        """Smallest resolvable non-zero probability, ``1 / n``."""
        return 1.0 / float(self.samples_db.size)

    def exceedance_db(self, level_db: float | np.ndarray) -> float | np.ndarray:
        level = np.asarray(level_db, dtype=float)
        n = self.samples_db.size
        idx = np.searchsorted(self.samples_db, level, side="left")
        out = (n - idx) / n
        return float(out) if out.ndim == 0 else out

    def sample_db(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.samples_db, size=int(size), replace=True)


def availability_callable(model: AvailabilityModel, margin_db: float) -> Callable[[float], float]:
    """Freeze a model and a margin into ``threshold_db -> availability``."""
    return lambda threshold_db: float(np.asarray(model.availability(margin_db, threshold_db)))
