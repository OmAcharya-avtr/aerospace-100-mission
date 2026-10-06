"""Correlated lognormal fading channel, and the exact discrete-time analytic
fade statistics of that channel.

Model
-----
Weak-to-moderate atmospheric scintillation is conventionally described by a
lognormal irradiance distribution; see L. C. Andrews and R. L. Phillips,
*Laser Beam Propagation through Random Media*, 2nd ed., SPIE Press (2005),
chapter on the probability distribution of irradiance. With a scintillation
index

    SI = var(I) / E[I]**2,

the lognormal log-irradiance variance that reproduces it is

    sigma_lnI**2 = ln(1 + SI),                                            (1)

and unit mean irradiance requires

    ln I = -sigma_lnI**2 / 2 + sigma_lnI * x,   x ~ N(0, 1).              (2)

Amplitude is ``a = sqrt(I)``, so the amplitude threshold ``T`` maps to a level
on the unit-variance Gaussian ``x``:

    u = (2 ln(T / sqrt(E[I])) + sigma_lnI**2 / 2) / sigma_lnI.            (3)

Temporal correlation is imposed on ``x`` by a first-order Gauss-Markov (AR(1))
recursion, the discrete-time sampling of an Ornstein-Uhlenbeck process with
correlation time ``tau``:

    rho = exp(-1 / (tau * fs)),                                           (4)
    x[0] = z[0],  x[n] = rho * x[n-1] + sqrt(1 - rho**2) * z[n],          (5)

with ``z`` i.i.d. standard normal. The initial condition in (5) is the
stationary one: ``x`` has unit variance at every index, including ``n = 0``.
The resulting autocorrelation is ``E[x[n] x[n+k]] = rho**|k|``, i.e. an
exponential autocorrelation with ``1/e`` length ``tau * fs`` samples.

This is a *model*, and a deliberately simple one. It reproduces the first-order
irradiance distribution and an exponential autocorrelation, and nothing else.
It does not reproduce the measured scintillation power spectrum (which rolls
off closer to ``f**(-8/3)`` in the inertial range than the AR(1) Lorentzian
does), it has no aperture averaging, no beam wander, no pointing jitter and no
inner- or outer-scale structure. Where the irradiance distribution itself is
wrong -- strong turbulence -- the gamma-gamma distribution of M. A. Al-Habash,
L. C. Andrews and R. L. Phillips, "Mathematical model for the irradiance
probability density function of a laser beam propagating through turbulent
media", *Optical Engineering* 40(8):1554-1562 (2001), is the standard
replacement, and this module does not implement it.

Why the analytic crossing rate here is a discrete-time quantity
---------------------------------------------------------------
Rice's continuous-time level-crossing rate requires the process to have a
finite-variance derivative. The Ornstein-Uhlenbeck process does not: its
spectrum is Lorentzian, its second spectral moment diverges, and its
continuous-time crossing rate of any level is infinite. Every crossing rate in
this module is therefore the *per-sample-pair* down-crossing probability times
``fs``,

    LCR = fs * P(x[n-1] >= u, x[n] < u),                                  (6)

which for the AR(1) model is an exact bivariate-normal orthant probability and
is computed here to about 1e-12. It is the right comparand for a sample
statistic measured on a sampled record at the same ``fs``, and it is *not* an
approximation to a continuous-time rate.

Units
-----
``fs`` Hz, ``tau`` s, ``si`` dimensionless, irradiance and amplitude in
arbitrary linear units with ``mean_irradiance`` in the irradiance unit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.signal import lfilter
from scipy.stats import multivariate_normal, norm

__all__ = [
    "LognormalSeries",
    "amplitude_threshold_to_gaussian_level",
    "analytic_down_crossing_probability",
    "analytic_level_crossing_rate",
    "analytic_mean_fade_duration",
    "analytic_outage_fraction",
    "ar1_unit_variance",
    "conditional_onset_probability",
    "correlation_length_samples",
    "gaussian_level_to_amplitude_threshold",
    "lognormal_amplitude_series",
    "rho_from_tau",
    "sigma_ln_i_from_si",
]


def sigma_ln_i_from_si(si: float) -> float:
    """``sigma_lnI = sqrt(ln(1 + SI))`` -- equation (1).

    Parameters
    ----------
    si:
        Scintillation index ``var(I) / E[I]**2``, dimensionless, ``>= 0``.

    Returns
    -------
    float
        Standard deviation of ``ln I``, dimensionless (nepers).
    """
    s = float(si)
    if not math.isfinite(s) or s < 0.0:
        raise ValueError(f"si (scintillation index) must be finite and >= 0, got {si!r}")
    return math.sqrt(math.log1p(s))


def rho_from_tau(tau_s: float, fs_hz: float) -> float:
    """``rho = exp(-1 / (tau * fs))`` -- equation (4).

    Parameters
    ----------
    tau_s:
        Correlation time in seconds, ``> 0``.
    fs_hz:
        Sample rate in Hz, ``> 0``.

    Returns
    -------
    float
        Lag-one autocorrelation of the log-amplitude Gaussian, in ``(0, 1)``.
    """
    tau = float(tau_s)
    fs = float(fs_hz)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"tau_s must be a positive finite correlation time, got {tau_s!r}")
    if not math.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"fs_hz must be a positive finite sample rate, got {fs_hz!r}")
    return math.exp(-1.0 / (tau * fs))


def correlation_length_samples(tau_s: float, fs_hz: float) -> float:
    """``tau * fs``: the ``1/e`` autocorrelation length in samples."""
    if not math.isfinite(float(tau_s)) or float(tau_s) <= 0.0:
        raise ValueError(f"tau_s must be a positive finite correlation time, got {tau_s!r}")
    if not math.isfinite(float(fs_hz)) or float(fs_hz) <= 0.0:
        raise ValueError(f"fs_hz must be a positive finite sample rate, got {fs_hz!r}")
    return float(tau_s) * float(fs_hz)


def ar1_unit_variance(z: NDArray[np.float64], rho: float) -> NDArray[np.float64]:
    """Stationary AR(1) filter of an i.i.d. standard-normal driving series.

    Implements equation (5) exactly:
    ``x[0] = z[0]``, ``x[n] = rho * x[n-1] + sqrt(1 - rho**2) * z[n]``.

    Parameters
    ----------
    z:
        Driving noise, one-dimensional, length ``>= 1``. Unit variance is
        assumed, not checked, since a caller may legitimately want a scaled
        process.
    rho:
        Lag-one autocorrelation, ``0 <= rho < 1``.

    Returns
    -------
    ndarray
        Same length as ``z``, unit variance when ``z`` has unit variance.

    Notes
    -----
    Implemented with :func:`scipy.signal.lfilter`, which gives
    ``s * z[0]`` at index 0 rather than ``z[0]``. The discrepancy
    ``d[n] = x[n] - x_lfilter[n]`` obeys ``d[0] = (1 - s) z[0]`` and
    ``d[n] = rho d[n-1]``, hence ``d[n] = (1 - s) z[0] rho**n``, which is added
    back in closed form. The result is bit-comparable with the direct
    recursion to a few times 1e-15 over two million samples
    (``validation/validate_series_construction.py``).
    """
    arr = np.asarray(z, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"z must be one-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("z must be non-empty")
    r = float(rho)
    if not math.isfinite(r) or not 0.0 <= r < 1.0:
        raise ValueError(f"rho must satisfy 0 <= rho < 1, got {rho!r}")
    s = math.sqrt(1.0 - r * r)
    x = lfilter([s], [1.0, -r], arr)
    if r == 0.0:
        x[0] = arr[0]
        return np.asarray(x, dtype=np.float64)
    n = np.arange(arr.size, dtype=np.float64)
    x = x + arr[0] * (1.0 - s) * np.exp(n * math.log(r))
    return np.asarray(x, dtype=np.float64)


@dataclass(frozen=True)
class LognormalSeries:
    """A correlated lognormal fading realisation and the parameters behind it.

    Attributes
    ----------
    amplitude:
        ``sqrt(I)``, arbitrary linear amplitude unit.
    irradiance:
        ``I``, mean ``mean_irradiance``.
    gaussian:
        The unit-variance AR(1) log-amplitude driver ``x`` of equation (5).
    fs_hz, tau_s, si, rho, sigma_ln_i, mean_irradiance, seed:
        The configuration, carried with the data so a result can never be
        reported without it.
    """

    amplitude: NDArray[np.float64]
    irradiance: NDArray[np.float64]
    gaussian: NDArray[np.float64]
    fs_hz: float
    tau_s: float
    si: float
    rho: float
    sigma_ln_i: float
    mean_irradiance: float
    seed: int

    @property
    def n_samples(self) -> int:
        return int(self.amplitude.size)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs_hz

    def describe(self) -> str:
        """Multi-line configuration block for validation output."""
        return "\n".join(
            [
                "Channel configuration:",
                "  model              : lognormal irradiance, AR(1) Gauss-Markov log-amplitude",
                f"  samples N          : {self.n_samples}",
                f"  sample rate fs     : {self.fs_hz!r} Hz",
                f"  correlation time   : {self.tau_s!r} s",
                f"  rho = exp(-1/(tau*fs)) : {self.rho!r}",
                f"  correlation length : {self.tau_s * self.fs_hz!r} samples",
                f"  scintillation index SI : {self.si!r}",
                f"  sigma_lnI**2 = ln(1+SI): {self.sigma_ln_i ** 2!r}",
                f"  sigma_lnI          : {self.sigma_ln_i!r}",
                f"  mean irradiance E[I]   : {self.mean_irradiance!r} (by construction)",
                (
                    f"  driving noise      : numpy.random.default_rng({self.seed})"
                    f".standard_normal({self.n_samples})"
                ),
            ]
        )


def lognormal_amplitude_series(
    n_samples: int,
    *,
    fs_hz: float,
    tau_s: float,
    si: float,
    seed: int,
    mean_irradiance: float = 1.0,
) -> LognormalSeries:
    """Generate a correlated lognormal amplitude series.

    The driving noise is drawn as a **single** call
    ``numpy.random.default_rng(seed).standard_normal(n_samples)``, so the
    series is fully determined by ``(n_samples, seed)`` and the four channel
    parameters. Drawing it in more than one call would change the stream and
    therefore the series.

    Parameters
    ----------
    n_samples:
        Record length, ``>= 1``.
    fs_hz:
        Sample rate, Hz.
    tau_s:
        Correlation time, s.
    si:
        Scintillation index, dimensionless.
    seed:
        Seed for ``numpy.random.default_rng``.
    mean_irradiance:
        ``E[I]``, in the irradiance unit. Default 1.0.

    Returns
    -------
    LognormalSeries
    """
    n = int(n_samples)
    if n < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples!r}")
    mi = float(mean_irradiance)
    if not math.isfinite(mi) or mi <= 0.0:
        raise ValueError(f"mean_irradiance must be positive and finite, got {mean_irradiance!r}")
    rho = rho_from_tau(tau_s, fs_hz)
    sigma = sigma_ln_i_from_si(si)
    z = np.random.default_rng(int(seed)).standard_normal(n)
    x = ar1_unit_variance(z, rho)
    log_i = math.log(mi) - sigma * sigma / 2.0 + sigma * x
    irradiance = np.exp(log_i)
    return LognormalSeries(
        amplitude=np.sqrt(irradiance),
        irradiance=irradiance,
        gaussian=x,
        fs_hz=float(fs_hz),
        tau_s=float(tau_s),
        si=float(si),
        rho=rho,
        sigma_ln_i=sigma,
        mean_irradiance=mi,
        seed=int(seed),
    )


def amplitude_threshold_to_gaussian_level(
    threshold_amplitude: float, si: float, *, mean_irradiance: float = 1.0
) -> float:
    """Equation (3): amplitude threshold ``T`` to Gaussian level ``u``.

    ``a < T`` if and only if ``x < u``, exactly, because the map from ``x`` to
    ``a`` is strictly increasing.
    """
    t = float(threshold_amplitude)
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError(f"threshold_amplitude must be positive and finite, got {t!r}")
    mi = float(mean_irradiance)
    if not math.isfinite(mi) or mi <= 0.0:
        raise ValueError(f"mean_irradiance must be positive and finite, got {mean_irradiance!r}")
    sigma = sigma_ln_i_from_si(si)
    if sigma == 0.0:
        raise ValueError("si = 0 is a non-fading channel; the Gaussian level is undefined")
    return (2.0 * math.log(t) - math.log(mi) + sigma * sigma / 2.0) / sigma


def gaussian_level_to_amplitude_threshold(
    level: float, si: float, *, mean_irradiance: float = 1.0
) -> float:
    """Inverse of :func:`amplitude_threshold_to_gaussian_level`."""
    sigma = sigma_ln_i_from_si(si)
    if sigma == 0.0:
        raise ValueError("si = 0 is a non-fading channel; the amplitude threshold is undefined")
    log_i = math.log(float(mean_irradiance)) - sigma * sigma / 2.0 + sigma * float(level)
    return math.exp(0.5 * log_i)


def analytic_down_crossing_probability(level: float, rho: float, *, method: str = "mvn") -> float:
    """``P(x[n-1] >= u, x[n] < u)`` for the stationary AR(1) process.

    Exact bivariate-normal orthant probability
    ``Phi(u) - Phi2(u, u; rho)``, equation (6) without the ``fs`` factor.

    Parameters
    ----------
    level:
        Gaussian level ``u``.
    rho:
        Lag-one autocorrelation, ``0 <= rho < 1``.
    method:
        ``"mvn"`` -- :func:`scipy.stats.multivariate_normal.cdf`.
        ``"quad"`` -- the one-dimensional integral
        ``int_u^inf phi(a) Phi((u - rho a) / sqrt(1 - rho**2)) da``
        by :func:`scipy.integrate.quad`. The two routes share no code inside
        SciPy and agree to about 1e-16 absolute
        (``validation/validate_analytic_lcr.py``), which is why both are kept.

    Returns
    -------
    float
        Probability in ``[0, 0.5]``.
    """
    u = float(level)
    r = float(rho)
    if not math.isfinite(u):
        raise ValueError(f"level must be finite, got {level!r}")
    if not math.isfinite(r) or not 0.0 <= r < 1.0:
        raise ValueError(f"rho must satisfy 0 <= rho < 1, got {rho!r}")
    if method == "quad":
        s = math.sqrt(1.0 - r * r)
        upper = u + 40.0

        def integrand(a: float) -> float:
            return float(norm.pdf(a) * norm.cdf((u - r * a) / s))

        value, _ = quad(integrand, u, upper, limit=400)
        return float(value)
    if method != "mvn":
        raise ValueError(f"method must be 'mvn' or 'quad', got {method!r}")
    phi2 = float(
        multivariate_normal.cdf([u, u], mean=[0.0, 0.0], cov=[[1.0, r], [r, 1.0]])
    )
    return float(norm.cdf(u)) - phi2


def analytic_outage_fraction(
    threshold_amplitude: float, si: float, *, mean_irradiance: float = 1.0
) -> float:
    """``P(a < T) = Phi(u)``, the stationary fraction of time in fade."""
    u = amplitude_threshold_to_gaussian_level(
        threshold_amplitude, si, mean_irradiance=mean_irradiance
    )
    return float(norm.cdf(u))


def analytic_level_crossing_rate(
    threshold_amplitude: float,
    *,
    si: float,
    tau_s: float,
    fs_hz: float,
    mean_irradiance: float = 1.0,
    method: str = "mvn",
) -> float:
    """Equation (6): ``fs * P(x[n-1] >= u, x[n] < u)``, in Hz.

    The discrete-time down-crossing rate of the AR(1) lognormal model at the
    stated sample rate. It is sample-rate dependent by construction, because
    the underlying continuous-time process has no finite crossing rate.
    """
    u = amplitude_threshold_to_gaussian_level(
        threshold_amplitude, si, mean_irradiance=mean_irradiance
    )
    rho = rho_from_tau(tau_s, fs_hz)
    return float(fs_hz) * analytic_down_crossing_probability(u, rho, method=method)


def analytic_mean_fade_duration(
    threshold_amplitude: float,
    *,
    si: float,
    tau_s: float,
    fs_hz: float,
    mean_irradiance: float = 1.0,
    method: str = "mvn",
) -> float:
    """``Phi(u) / LCR``, in seconds.

    The stationary-process form of the Rice relation: the fraction of time
    below the level divided by the rate at which excursions below it start.
    Exact in the limit of an infinite record for any stationary binary
    sequence, and therefore exact for this model; a finite record differs by
    its censoring and edge effects, which is what
    :attr:`linkoutage.fade.FadeStatistics.rice_relative_residual` measures.
    """
    rate = analytic_level_crossing_rate(
        threshold_amplitude,
        si=si,
        tau_s=tau_s,
        fs_hz=fs_hz,
        mean_irradiance=mean_irradiance,
        method=method,
    )
    if rate <= 0.0:
        return float("nan")
    return analytic_outage_fraction(
        threshold_amplitude, si, mean_irradiance=mean_irradiance
    ) / rate


def conditional_onset_probability(
    current_gaussian: NDArray[np.float64] | float,
    *,
    level: float,
    rho: float,
    horizon_samples: int,
    n_quadrature: int = 2000,
) -> NDArray[np.float64]:
    """``P(a fade begins within the next H samples | x[t])``, analytic.

    The expected number of down-crossings of ``u`` in ``(t, t + H]`` given
    ``x[t] = c`` is

        m(c) = sum_{k=1}^{H} P(x[t+k-1] >= u, x[t+k] < u | x[t] = c),

    each term an exact bivariate-normal orthant probability of the conditional
    law ``x[t+k-1] | c ~ N(rho**(k-1) c, 1 - rho**(2(k-1)))`` with
    ``x[t+k] = rho x[t+k-1] + sqrt(1 - rho**2) e``. Evaluated here as the
    one-dimensional integral

        P_k = int_u^inf N(a; rho**(k-1) c, 1 - rho**(2(k-1)))
                        Phi((u - rho a) / sqrt(1 - rho**2)) da

    on a fixed ``n_quadrature``-point trapezoidal grid over ``[u, u + 12]``,
    with the ``k = 1`` term taken in closed form because its conditional
    variance is zero. The onset probability is then the Poisson-clumping
    approximation

        P(onset within H | c) = 1 - exp(-m(c)).

    This is an approximation in exactly one place: clumping. It treats the
    down-crossings in the horizon as a Poisson count, which understates the
    probability when ``m`` is large because down-crossings of a correlated
    process arrive in bursts. It is exact to first order in ``m`` and is the
    analytic predictor benchmarked in :mod:`linkoutage.predictors`.

    Parameters
    ----------
    current_gaussian:
        Observed ``x[t]``, scalar or array.
    level:
        Gaussian level ``u``.
    rho:
        Lag-one autocorrelation.
    horizon_samples:
        ``H >= 1``.
    n_quadrature:
        Trapezoid points. 2000 gives about 1e-9 on the integral at
        ``rho = 0.995``; the default is deliberately generous because the
        function is evaluated on a grid and interpolated, not per sample.

    Returns
    -------
    ndarray
        Probabilities in ``[0, 1)``, same shape as ``current_gaussian``.
    """
    c = np.atleast_1d(np.asarray(current_gaussian, dtype=np.float64))
    u = float(level)
    r = float(rho)
    h = int(horizon_samples)
    if h < 1:
        raise ValueError(f"horizon_samples must be >= 1, got {horizon_samples!r}")
    if not math.isfinite(r) or not 0.0 <= r < 1.0:
        raise ValueError(f"rho must satisfy 0 <= rho < 1, got {rho!r}")
    if int(n_quadrature) < 16:
        raise ValueError(f"n_quadrature must be >= 16, got {n_quadrature!r}")
    s1 = math.sqrt(1.0 - r * r)

    # k = 1: x[t] is known exactly.
    m = np.where(c >= u, norm.cdf((u - r * c) / s1), 0.0)

    if h > 1:
        a = np.linspace(u, u + 12.0, int(n_quadrature))
        tail = norm.cdf((u - r * a) / s1)
        for k in range(2, h + 1):
            var = 1.0 - r ** (2 * (k - 1))
            sd = math.sqrt(var) if var > 0.0 else 0.0
            if sd <= 0.0:
                m = m + np.where(c >= u, norm.cdf((u - r * c) / s1), 0.0)
                continue
            mean = (r ** (k - 1)) * c
            dens = norm.pdf((a[None, :] - mean[:, None]) / sd) / sd
            m = m + np.trapezoid(dens * tail[None, :], a, axis=1)

    out = 1.0 - np.exp(-m)
    if np.isscalar(current_gaussian) or np.asarray(current_gaussian).ndim == 0:
        return np.asarray(out[0], dtype=np.float64)
    return np.asarray(out, dtype=np.float64)
