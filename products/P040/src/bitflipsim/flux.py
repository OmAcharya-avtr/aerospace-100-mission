"""Single-event-upset rate model: flux x cross-section x bit count, and Poisson counts.

Rate model
----------
For a memory block of ``N`` bits exposed to an omnidirectional particle flux
``phi`` with an effective per-bit upset cross-section ``sigma``, the expected
upset rate is the elementary product

    lambda = phi * sigma * N                                              (1)

with units

    phi    : particles cm^-2 s^-1
    sigma  : cm^2 bit^-1
    N      : bit
    lambda : upset s^-1

Equation (1) is dimensional accounting, not a fitted model: it follows from
defining ``sigma`` as the upset probability per unit fluence per bit. Over an
exposure ``t`` seconds the expected count is

    mu = lambda * t                                                       (2)

and, if upsets in distinct bits are independent and the per-bit probability is
small, the count is Poisson with mean ``mu``:

    P(k) = exp(-mu) * mu**k / k!,    Var[k] = E[k] = mu                   (3)

The Poisson limit of the binomial is standard; this module verifies it
numerically in ``validation/validate_poisson_counts.py`` rather than asserting
it.

Assumptions and validity range
------------------------------
* ``sigma`` is a single effective cross-section. A real part has a cross-section
  that is a function of linear energy transfer (a Weibull fit to a heavy-ion
  test), and the correct rate is an integral of that curve over the LET spectrum
  of the environment. Equation (1) collapses that integral to one number and is
  therefore a first-order estimate only.
* Upsets are assumed single-bit and independent. Multiple-bit upsets from one
  particle (MBU), single-event functional interrupts, latch-up and total-dose
  effects are not modelled.
* No angular dependence, no shielding model, no proton-induced-versus-heavy-ion
  distinction, no orbit or solar-activity time dependence.
* The Poisson model requires ``phi * sigma * t << 1`` per bit, which the
  function :func:`poisson_validity` checks and reports.

Cross-sections and fluxes are **inputs supplied by the user from their own part
test data and environment model**. This module ships one illustrative flux
constant, named and documented as illustrative, so that the examples have a
number to run with. It is not attributed to any measured environment.

FIT conversion
--------------
A rate in FIT is failures per 10**9 device-hours, so

    lambda_FIT = lambda_per_s * 1e9 * 3600 = lambda_per_s * 3.6e12         (4)

which is arithmetic and is checked in the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Illustrative particle flux, particles cm^-2 s^-1. A round number chosen so
#: that the examples produce upset counts of order 1 to 100 over minutes of
#: exposure. It is NOT a measured or published environment figure and must not
#: be cited as one. Replace it with output from an environment model (for
#: example an AP-9/AE-9 or CREME run) before drawing any conclusion.
ILLUSTRATIVE_FLUX_PER_CM2_S: float = 1.0e3

#: Illustrative per-bit upset cross-section, cm^2 bit^-1. Same caveat: a round
#: number for demonstration, not a part measurement. Real SRAM saturated
#: cross-sections must come from a heavy-ion or proton test of the actual part.
ILLUSTRATIVE_CROSS_SECTION_CM2_PER_BIT: float = 1.0e-14

SECONDS_PER_FIT_WINDOW: float = 3.6e12
"""Seconds in 10**9 device-hours: ``1e9 * 3600``. Used by equation (4)."""


@dataclass(frozen=True)
class UpsetRate:
    """Result of equation (1), with its inputs retained.

    Attributes
    ----------
    flux_per_cm2_s:
        Particle flux, particles cm^-2 s^-1.
    cross_section_cm2_per_bit:
        Effective per-bit upset cross-section, cm^2 bit^-1.
    bit_count:
        Number of exposed bits, dimensionless count.
    rate_per_s:
        ``phi * sigma * N``, upsets s^-1.
    """

    flux_per_cm2_s: float
    cross_section_cm2_per_bit: float
    bit_count: int
    rate_per_s: float

    @property
    def rate_fit(self) -> float:
        """The same rate expressed in FIT (failures per 1e9 device-hours)."""
        return self.rate_per_s * SECONDS_PER_FIT_WINDOW

    @property
    def mean_time_between_upsets_s(self) -> float:
        """``1 / lambda`` in seconds; ``inf`` when the rate is zero."""
        return float("inf") if self.rate_per_s == 0.0 else 1.0 / self.rate_per_s

    def expected_upsets(self, exposure_s: float) -> float:
        """Equation (2): ``mu = lambda * t``, dimensionless count."""
        if exposure_s < 0.0:
            raise ValueError(f"exposure_s must be >= 0, got {exposure_s}")
        return self.rate_per_s * float(exposure_s)

    def exposure_for_expected_upsets(self, expected: float) -> float:
        """Invert equation (2): the exposure in seconds giving ``mu = expected``."""
        if expected < 0.0:
            raise ValueError(f"expected must be >= 0, got {expected}")
        if self.rate_per_s == 0.0:
            raise ValueError("rate is zero; no exposure yields a non-zero expectation")
        return float(expected) / self.rate_per_s


def upset_rate(
    flux_per_cm2_s: float,
    cross_section_cm2_per_bit: float,
    bit_count: int,
) -> UpsetRate:
    """Equation (1): expected upset rate of a bit block under a particle flux.

    Parameters
    ----------
    flux_per_cm2_s:
        Particle flux, particles cm^-2 s^-1. Must be >= 0.
    cross_section_cm2_per_bit:
        Effective per-bit upset cross-section, cm^2 bit^-1. Must be >= 0.
    bit_count:
        Number of exposed bits. Must be >= 0.

    Returns
    -------
    UpsetRate
        With ``rate_per_s`` in upsets per second.
    """
    if flux_per_cm2_s < 0.0:
        raise ValueError(f"flux_per_cm2_s must be >= 0, got {flux_per_cm2_s}")
    if cross_section_cm2_per_bit < 0.0:
        raise ValueError(
            f"cross_section_cm2_per_bit must be >= 0, got {cross_section_cm2_per_bit}"
        )
    if bit_count < 0:
        raise ValueError(f"bit_count must be >= 0, got {bit_count}")
    rate = float(flux_per_cm2_s) * float(cross_section_cm2_per_bit) * int(bit_count)
    return UpsetRate(
        flux_per_cm2_s=float(flux_per_cm2_s),
        cross_section_cm2_per_bit=float(cross_section_cm2_per_bit),
        bit_count=int(bit_count),
        rate_per_s=rate,
    )


def poisson_validity(rate: UpsetRate, exposure_s: float) -> dict[str, float | bool]:
    """Report whether the small-per-bit-probability condition holds.

    The Poisson model of equation (3) approximates a binomial over ``N`` bits
    with per-bit probability ``p = phi * sigma * t``. The approximation is good
    when ``p << 1``; this returns ``p``, ``mu`` and a boolean at the 1e-3
    threshold so that a caller outside the validity range is told, not guessed
    at.
    """
    per_bit = rate.flux_per_cm2_s * rate.cross_section_cm2_per_bit * float(exposure_s)
    return {
        "per_bit_probability": per_bit,
        "expected_upsets": rate.expected_upsets(exposure_s),
        "within_validity": bool(per_bit < 1e-3),
        "threshold": 1e-3,
    }


def sample_upset_counts(
    rate: UpsetRate,
    exposure_s: float,
    trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``trials`` Poisson upset counts for one exposure.

    Returns
    -------
    numpy.ndarray
        Shape ``(trials,)``, dtype int64, dimensionless counts.
    """
    if trials <= 0:
        raise ValueError(f"trials must be > 0, got {trials}")
    mu = rate.expected_upsets(exposure_s)
    return rng.poisson(lam=mu, size=int(trials)).astype(np.int64)


@dataclass(frozen=True)
class CountStatistics:
    """Sample statistics of an upset-count sample, with their sampling errors.

    For a Poisson sample of size ``n`` and mean ``mu``:

    * the standard error of the sample mean is ``sqrt(mu / n)``;
    * the standard error of the sample variance of a Poisson variate is
      ``sqrt((mu + 2 * mu**2) / n)``, from ``Var[S2] = (mu_4 - mu_2**2)/n``
      with the Poisson central moments ``mu_2 = mu`` and
      ``mu_4 = mu + 3 * mu**2``.

    Both expressions are derived in the docstring of :func:`count_statistics`
    and checked against a Monte Carlo in
    ``validation/validate_poisson_counts.py``.
    """

    trials: int
    expected: float
    sample_mean: float
    sample_variance: float
    mean_standard_error: float
    variance_standard_error: float

    @property
    def mean_z(self) -> float:
        """``(sample_mean - expected) / se(mean)``, dimensionless."""
        if self.mean_standard_error == 0.0:
            return 0.0
        return (self.sample_mean - self.expected) / self.mean_standard_error

    @property
    def variance_z(self) -> float:
        """``(sample_variance - expected) / se(variance)``, dimensionless."""
        if self.variance_standard_error == 0.0:
            return 0.0
        return (self.sample_variance - self.expected) / self.variance_standard_error


def count_statistics(counts: np.ndarray, expected: float) -> CountStatistics:
    """Sample mean and variance of ``counts`` with Poisson sampling errors.

    Derivation of the two standard errors, for a Poisson variate ``K`` with mean
    ``mu`` and a sample of size ``n``:

    * ``Var[mean] = Var[K]/n = mu/n``, so ``se(mean) = sqrt(mu/n)``.
    * The unbiased sample variance ``S2`` has
      ``Var[S2] = mu_4/n - mu_2**2 * (n-3)/(n(n-1))``. For large ``n`` this is
      ``(mu_4 - mu_2**2)/n``. Poisson central moments are ``mu_2 = mu`` and
      ``mu_4 = mu + 3 mu**2``, giving ``Var[S2] = (mu + 2 mu**2)/n`` and
      ``se(S2) = sqrt((mu + 2 mu**2)/n)``.

    Parameters
    ----------
    counts:
        Integer counts, shape ``(n,)``.
    expected:
        The model mean ``mu`` the counts are being tested against.
    """
    arr = np.asarray(counts, dtype=np.float64).ravel()
    n = arr.size
    if n < 2:
        raise ValueError(f"need at least 2 counts to form a variance, got {n}")
    if expected < 0.0:
        raise ValueError(f"expected must be >= 0, got {expected}")
    mu = float(expected)
    return CountStatistics(
        trials=int(n),
        expected=mu,
        sample_mean=float(arr.mean()),
        sample_variance=float(arr.var(ddof=1)),
        mean_standard_error=float(np.sqrt(mu / n)),
        variance_standard_error=float(np.sqrt((mu + 2.0 * mu * mu) / n)),
    )
