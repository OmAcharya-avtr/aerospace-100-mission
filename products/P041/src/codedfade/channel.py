"""Temporally correlated optical fading channel: filtered-Gaussian sample paths.

What this module is for
-----------------------
A memoryless fading model tells you the marginal distribution of the received
irradiance and nothing about *how long* a fade lasts. Interleaver depth is set
by fade duration, not by the marginal. This module therefore generates fading as
a **sample path** with a specified temporal correlation, so that fade duration
and level-crossing rate are emergent statistics of the realisation rather than
parameters handed to the model.

Correlation kernels
-------------------
Two kernels are offered, and the difference between them is not cosmetic.

``"exp"`` -- Gauss-Markov / Ornstein-Uhlenbeck. The underlying unit-variance
Gaussian process has autocorrelation

    R(t) = exp(-|t| / tau)                                                (1)

realised exactly in discrete time as the AR(1) recursion

    g[n] = rho * g[n-1] + sqrt(1 - rho**2) * w[n],   rho = exp(-1/(fs*tau))  (2)

with ``w`` i.i.d. standard normal. Equation (2) reproduces (1) exactly at the
sample instants (textbook AR(1) result, verified in
``validation/validate_channel_statistics.py``).

``"gauss"`` -- Gaussian kernel. The underlying process has

    R(t) = exp(-(t / tau)**2)                                             (3)

realised by convolving white Gaussian noise with a Gaussian window of standard
deviation ``s = tau / 2`` and renormalising to unit variance: the
autocorrelation of white noise filtered by ``w(t) = exp(-t**2/(2 s**2))`` is
proportional to ``exp(-t**2/(4 s**2))``, which equals (3) when ``s = tau/2``.

**Why both exist.** The ``exp`` kernel is the standard first-order model and is
the one most papers write down, but ``R(t) = exp(-|t|/tau)`` is not twice
differentiable at the origin, so the continuous-time process has no finite
mean-square derivative and its true level-crossing rate is **infinite**. In
discrete time the measured crossing rate therefore grows without bound as the
sample rate rises; it is a property of the sampling, not of the channel. The
``gauss`` kernel has ``R''(0) = -2/tau**2`` and a finite crossing rate, so Rice's
(1945) level-crossing formula applies to it and the measured rate converges as
``fs`` rises. ``validation/validate_crossing_convergence.py`` measures both.
Both facts are standard consequences of Rice (1945); neither is a claim about a
particular reference's page.

In both cases ``tau`` is defined as the 1/e point of the autocorrelation of the
**underlying Gaussian process**: ``R(tau) = 1/e`` by construction in (1) and (3).

Marginal distributions
----------------------
``"lognormal"`` -- weak-turbulence model. The irradiance ``I`` normalised to
``E[I] = 1`` is lognormal with log-irradiance variance

    sigma_lnI**2 = ln(1 + SI)                                             (4)

where ``SI`` is the scintillation index ``SI = Var[I]/E[I]**2``, and mean
``mu_lnI = -sigma_lnI**2 / 2`` so that ``E[I] = 1``. Then

    I[n] = exp(mu_lnI + sigma_lnI * g[n])                                 (5)

with ``g`` the unit-variance correlated Gaussian of (2) or (3). Equation (4) is
the elementary lognormal moment relation, not an empirical fit. The lognormal
irradiance model for weak optical turbulence is standard; see Andrews &
Phillips, *Laser Beam Propagation through Random Media* (2005).

**Reachable range.** Equation (6) applied to (7)-(9) is not monotone in
``sigma_R**2``: it rises to a maximum and falls again in the saturation regime.
:data:`GAMMA_GAMMA_SI_PEAK` records that maximum, computed by golden-section
search at import, and :func:`gamma_gamma_parameters_from_si` refuses targets above
it rather than silently returning the wrong branch. The lognormal marginal has no
such cap.

``"gammagamma"`` -- the Al-Habash, Andrews & Phillips (*Optical Engineering*
40(8), 2001) model, valid from weak to strong turbulence. ``I = X * Y`` with
``X ~ Gamma(alpha, 1/alpha)`` (large-scale) and ``Y ~ Gamma(beta, 1/beta)``
(small-scale), both unit mean, so ``E[I] = 1`` and

    SI = 1/alpha + 1/beta + 1/(alpha*beta)                                (6)

For a plane wave the parameters follow from the Rytov variance ``sigma_R**2``
through the commonly stated relations

    sigma_x**2 = 0.49 * sR2 / (1 + 1.11 * sR2**(6/5))**(7/6)              (7)
    sigma_y**2 = 0.51 * sR2 / (1 + 0.69 * sR2**(6/5))**(5/6)              (8)
    alpha = 1 / (exp(sigma_x**2) - 1),  beta = 1 / (exp(sigma_y**2) - 1)   (9)

(Al-Habash, Andrews & Phillips 2001; reproduced in Andrews & Phillips 2005.)
``rytov_to_gamma_gamma`` implements (7)-(9) and
``validation/validate_channel_statistics.py`` checks that (6) applied to the
resulting ``(alpha, beta)`` equals ``exp(sigma_x**2 + sigma_y**2) - 1``, which
is the identity (9) forces.

Temporal correlation of the gamma-gamma path is imposed through a **Gaussian
copula**: two independent correlated Gaussian paths are mapped to uniforms by
``Phi`` and then to gamma marginals by the gamma inverse CDF. This makes the
marginal exact and the correlation structure *approximate* -- the specified
``tau`` governs the latent Gaussian, and the measured 1/e correlation time of
the resulting log-irradiance differs from it. The measured value is reported by
``measured_correlation_time`` rather than assumed, and the discrepancy is
published in ``validation/VALIDATION.md``. No claim is made that the copula
reproduces a physical two-scale temporal spectrum.

Units and validity
------------------
* ``correlation_time_s`` : s. Must be > 0. Must satisfy ``fs * tau >= 4`` for
  the path to resolve the correlation at all; ``ChannelConfig`` raises below 2.
* ``sample_rate_hz`` : Hz (samples per second; in the link model one sample per
  channel symbol).
* ``scintillation_index`` : dimensionless, ``Var[I]/E[I]**2``, > 0.
* Irradiance ``I`` is normalised to unit mean; amplitude ``a = sqrt(I)``.
* Frozen-turbulence/stationary assumption: no pointing jitter, no beam
  wander as a separate process, no time-varying mean, no wavelength or path
  geometry. Those are inputs to ``scintillation_index`` and
  ``correlation_time_s``, not modelled here.
* The model is scalar and single-aperture: no spatial structure, so aperture
  averaging must be folded into ``scintillation_index`` by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import signal, special, stats

Kernel = Literal["exp", "gauss"]
Marginal = Literal["lognormal", "gammagamma"]


@dataclass(frozen=True)
class ChannelConfig:
    """Configuration of a correlated fading channel.

    Attributes
    ----------
    scintillation_index:
        ``Var[I]/E[I]**2``, dimensionless, > 0.
    correlation_time_s:
        1/e point of the autocorrelation of the underlying Gaussian process, s.
    sample_rate_hz:
        Sampling rate of the generated path, Hz.
    marginal:
        ``"lognormal"`` or ``"gammagamma"``.
    kernel:
        ``"exp"`` (Gauss-Markov) or ``"gauss"`` (differentiable).
    seed:
        Seed for ``numpy.random.default_rng``.
    """

    scintillation_index: float
    correlation_time_s: float
    sample_rate_hz: float
    marginal: Marginal = "lognormal"
    kernel: Kernel = "exp"
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.scintillation_index > 0:
            raise ValueError(
                f"scintillation_index must be > 0, got {self.scintillation_index!r}"
            )
        if not self.correlation_time_s > 0:
            raise ValueError(
                f"correlation_time_s must be > 0 s, got {self.correlation_time_s!r}"
            )
        if not self.sample_rate_hz > 0:
            raise ValueError(f"sample_rate_hz must be > 0 Hz, got {self.sample_rate_hz!r}")
        if self.marginal not in ("lognormal", "gammagamma"):
            raise ValueError(
                f"marginal must be 'lognormal' or 'gammagamma', got {self.marginal!r}"
            )
        if self.kernel not in ("exp", "gauss"):
            raise ValueError(f"kernel must be 'exp' or 'gauss', got {self.kernel!r}")
        if self.samples_per_correlation_time < 2.0:
            raise ValueError(
                "sample_rate_hz * correlation_time_s must be >= 2 for the path to "
                f"resolve the correlation; got {self.samples_per_correlation_time:.4f}. "
                "Raise sample_rate_hz or correlation_time_s."
            )

    @property
    def samples_per_correlation_time(self) -> float:
        """``L_c = tau * fs``: correlation time in samples (symbols). Dimensionless."""
        return float(self.correlation_time_s * self.sample_rate_hz)

    @property
    def log_irradiance_variance(self) -> float:
        """``sigma_lnI**2 = ln(1 + SI)``, equation (4). Dimensionless."""
        return float(np.log1p(self.scintillation_index))


def correlated_gaussian(
    n_samples: int,
    samples_per_correlation_time: float,
    rng: np.random.Generator,
    kernel: Kernel = "exp",
) -> np.ndarray:
    """Unit-variance, zero-mean correlated Gaussian path.

    Parameters
    ----------
    n_samples:
        Length of the returned path, samples.
    samples_per_correlation_time:
        ``L_c = tau * fs``, dimensionless, > 0.
    rng:
        Seeded generator.
    kernel:
        ``"exp"`` gives equation (1)-(2); ``"gauss"`` gives equation (3).

    Returns
    -------
    float64 array of shape ``(n_samples,)``, sample mean ~0, sample variance ~1.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples!r}")
    lc = float(samples_per_correlation_time)
    if not lc > 0:
        raise ValueError(f"samples_per_correlation_time must be > 0, got {lc!r}")

    if kernel == "exp":
        rho = float(np.exp(-1.0 / lc))
        white = rng.standard_normal(n_samples)
        out = np.empty(n_samples, dtype=np.float64)
        out[0] = white[0]
        scale = np.sqrt(1.0 - rho * rho)
        # AR(1) recursion, equation (2). Loop is O(n) and the dominant cost of
        # path generation; vectorising it requires a cumulative product that
        # overflows for small rho, so the explicit loop is kept.
        prev = out[0]
        for i in range(1, n_samples):
            prev = rho * prev + scale * white[i]
            out[i] = prev
        return out

    if kernel == "gauss":
        s = lc / 2.0
        half = int(np.ceil(5.0 * s))
        taps = np.exp(-0.5 * (np.arange(-half, half + 1) / s) ** 2)
        taps /= np.sqrt(np.sum(taps**2))  # unit-energy -> unit-variance output
        white = rng.standard_normal(n_samples + 2 * half)
        # fftconvolve, not np.convolve: at Lc = 400 the kernel is ~2000 taps and the
        # direct convolution is O(n * taps), which is minutes for a 4e6-sample path.
        out = signal.fftconvolve(white, taps, mode="valid")
        return out[:n_samples].astype(np.float64, copy=False)

    raise ValueError(f"kernel must be 'exp' or 'gauss', got {kernel!r}")


def rytov_to_gamma_gamma(rytov_variance: float) -> tuple[float, float]:
    """Plane-wave gamma-gamma shape parameters from the Rytov variance.

    Implements equations (7)-(9), Al-Habash, Andrews & Phillips (*Opt. Eng.*
    40(8), 2001).

    Parameters
    ----------
    rytov_variance:
        ``sigma_R**2``, dimensionless, > 0.

    Returns
    -------
    ``(alpha, beta)``, both dimensionless and > 0.
    """
    sr2 = float(rytov_variance)
    if not sr2 > 0:
        raise ValueError(f"rytov_variance must be > 0, got {rytov_variance!r}")
    sx2 = 0.49 * sr2 / (1.0 + 1.11 * sr2 ** (6.0 / 5.0)) ** (7.0 / 6.0)
    sy2 = 0.51 * sr2 / (1.0 + 0.69 * sr2 ** (6.0 / 5.0)) ** (5.0 / 6.0)
    alpha = 1.0 / (np.expm1(sx2))
    beta = 1.0 / (np.expm1(sy2))
    return float(alpha), float(beta)


def gamma_gamma_scintillation_index(alpha: float, beta: float) -> float:
    """``SI = 1/alpha + 1/beta + 1/(alpha*beta)``, equation (6). Dimensionless."""
    if not (alpha > 0 and beta > 0):
        raise ValueError(f"alpha and beta must be > 0, got {alpha!r}, {beta!r}")
    return float(1.0 / alpha + 1.0 / beta + 1.0 / (alpha * beta))


def _gamma_gamma_si_peak() -> tuple[float, float]:
    """``(sigma_R**2, SI)`` at the maximum of equation (6) along equations (7)-(9).

    Equation (6) applied to the plane-wave relations is **not monotone** in the
    Rytov variance: it rises to a maximum and then falls again, which is the
    saturation regime of the model. The inversion below is therefore restricted to
    the increasing branch, and this function locates its end. Found by
    golden-section search, so the number is computed rather than quoted.
    """
    lo, hi = 1.0, 100.0
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    f = lambda x: gamma_gamma_scintillation_index(*rytov_to_gamma_gamma(x))  # noqa: E731
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    for _ in range(200):
        if f(c) > f(d):
            b = d
        else:
            a = c
        c, d = b - phi * (b - a), a + phi * (b - a)
    peak = 0.5 * (a + b)
    return float(peak), float(f(peak))


#: ``(sigma_R**2, SI)`` at the end of the monotone-increasing branch of the
#: plane-wave gamma-gamma scintillation index. Beyond this Rytov variance the
#: model's scintillation index decreases again (the saturation regime), so
#: :func:`gamma_gamma_parameters_from_si` refuses targets above ``SI`` here rather
#: than returning the wrong branch.
GAMMA_GAMMA_SI_PEAK: tuple[float, float] = _gamma_gamma_si_peak()


def gamma_gamma_parameters_from_si(scintillation_index: float) -> tuple[float, float]:
    """``(alpha, beta)`` reproducing a target scintillation index.

    Equations (7)-(9) are inverted numerically for ``sigma_R**2`` so that (6)
    returns ``scintillation_index``. The inversion is restricted to the
    **monotone-increasing branch** of equation (6), which ends at
    :data:`GAMMA_GAMMA_SI_PEAK`; a bisection on ``log10(sigma_R**2)`` over
    ``[-4, log10(peak)]`` is then well posed.

    Targets above the peak are **not reachable** with the plane-wave relations and
    raise ``ValueError`` naming the reachable range. This is a property of the
    model, not a limitation of the solver: the plane-wave gamma-gamma
    scintillation index saturates. A caller who needs a higher scintillation index
    should use the lognormal marginal (which has no such cap) or supply
    ``(alpha, beta)`` from a spherical-wave or measured parameterisation, neither
    of which is implemented here.
    """
    si = float(scintillation_index)
    if not si > 0:
        raise ValueError(f"scintillation_index must be > 0, got {scintillation_index!r}")
    peak_sr2, peak_si = GAMMA_GAMMA_SI_PEAK
    hi = float(np.log10(peak_sr2))
    lo = -4.0
    f = lambda t: gamma_gamma_scintillation_index(*rytov_to_gamma_gamma(10.0**t)) - si  # noqa: E731
    if f(lo) > 0 or si > peak_si:
        raise ValueError(
            f"scintillation_index {si!r} is outside the reachable plane-wave "
            f"gamma-gamma range [{f(lo) + si:.6g}, {peak_si:.6g}] (the model's "
            f"scintillation index saturates at sigma_R^2 = {peak_sr2:.4f})"
        )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return rytov_to_gamma_gamma(10.0 ** (0.5 * (lo + hi)))


def generate_irradiance(config: ChannelConfig, n_samples: int) -> np.ndarray:
    """Correlated irradiance path normalised to unit mean.

    Parameters
    ----------
    config:
        Channel configuration. ``config.seed`` makes the result reproducible.
    n_samples:
        Path length, samples.

    Returns
    -------
    float64 array, ``I[n] > 0``, ensemble mean 1 by construction.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples!r}")
    rng = np.random.default_rng(config.seed)
    lc = config.samples_per_correlation_time

    if config.marginal == "lognormal":
        g = correlated_gaussian(n_samples, lc, rng, config.kernel)
        sigma = np.sqrt(config.log_irradiance_variance)
        return np.exp(-0.5 * sigma * sigma + sigma * g)

    alpha, beta = gamma_gamma_parameters_from_si(config.scintillation_index)
    gx = correlated_gaussian(n_samples, lc, rng, config.kernel)
    gy = correlated_gaussian(n_samples, lc, rng, config.kernel)
    # Gaussian copula: Phi(g) is uniform, gamma ppf maps it to the target
    # marginal. ndtr is the standard normal CDF.
    ux = np.clip(special.ndtr(gx), 1e-12, 1.0 - 1e-12)
    uy = np.clip(special.ndtr(gy), 1e-12, 1.0 - 1e-12)
    x = stats.gamma.ppf(ux, a=alpha, scale=1.0 / alpha)
    y = stats.gamma.ppf(uy, a=beta, scale=1.0 / beta)
    return np.asarray(x * y, dtype=np.float64)


def generate_amplitude(config: ChannelConfig, n_samples: int) -> np.ndarray:
    """Amplitude path ``a = sqrt(I)``, dimensionless, ``E[a**2] = 1``."""
    return np.sqrt(generate_irradiance(config, n_samples))


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased sample autocorrelation of ``x`` for lags ``0..max_lag``, normalised to 1 at lag 0.

    Uses the direct (biased, 1/N) estimator, which is the consistent choice for
    fitting a correlation time: the unbiased 1/(N-k) estimator is noisy at large
    lag and can exceed 1.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    if max_lag < 1 or max_lag >= x.size:
        raise ValueError(f"max_lag must be in [1, {x.size - 1}], got {max_lag!r}")
    c = x - x.mean()
    n = c.size
    denom = float(np.dot(c, c))
    out = np.empty(max_lag + 1, dtype=np.float64)
    out[0] = 1.0
    for k in range(1, max_lag + 1):
        out[k] = float(np.dot(c[: n - k], c[k:])) / denom
    return out


def measured_correlation_time(
    series: np.ndarray, sample_rate_hz: float, max_lag: int | None = None
) -> float:
    """1/e correlation time of ``log(series)``, estimated from the sample autocorrelation.

    The log is taken because both marginals are defined through a Gaussian in the
    log domain, so the log-domain autocorrelation is the quantity the model
    specifies. Linear interpolation between the bracketing lags gives a
    sub-sample estimate.

    Returns the correlation time in seconds, or ``nan`` if the autocorrelation
    does not fall to 1/e within ``max_lag``.
    """
    series = np.asarray(series, dtype=np.float64)
    if np.any(series <= 0):
        raise ValueError("series must be strictly positive to take its logarithm")
    if max_lag is None:
        max_lag = min(series.size - 1, 4096)
    acf = autocorrelation(np.log(series), max_lag)
    target = float(np.exp(-1.0))
    below = np.nonzero(acf < target)[0]
    if below.size == 0:
        return float("nan")
    k = int(below[0])
    if k == 0:
        return 0.0
    a0, a1 = acf[k - 1], acf[k]
    frac = (a0 - target) / (a0 - a1)
    return float((k - 1 + frac) / sample_rate_hz)
