"""Fade statistics: sample definitions, and the analytic level-crossing baseline.

Sample definitions, stated exactly because a cross-check depends on them
----------------------------------------------------------------------
Let ``a[0..N-1]`` be an amplitude series sampled at ``fs`` Hz and ``a_th`` a
threshold amplitude. Define the indicator ``b[n] = (a[n] < a_th)``.

* A **down-crossing** is an index ``n >= 1`` with ``b[n-1] == False`` and
  ``b[n] == True``. The **level-crossing rate** is

      LCR = (number of down-crossings) / (N / fs)                       [s^-1]

  i.e. down-crossings per second over the whole record, including the parts of
  the record spent above the threshold.
* A **fade** is a maximal run of consecutive samples with ``b == True``. A fade
  is **complete** if it is preceded by at least one sample with ``b == False``
  and followed by at least one sample with ``b == False``. Runs touching either
  end of the record are **censored and excluded** from the duration statistics.
* The **mean fade duration** is the arithmetic mean of the complete-run lengths
  in samples, divided by ``fs``:

      MFD = mean(run_length_samples) / fs                               [s]

  A run of one sample has duration ``1/fs``, not zero.
* The **outage fraction** is ``mean(b)``, dimensionless.

These are sample statistics of one realisation. They are not estimates of a
continuous-time quantity and are **not** compared against any wall-clock
measurement anywhere in this package. Cross-check X1 of the batch specification
compares exactly ``LCR`` and ``MFD`` as defined above, on an identical seeded
series, against P049 LinkOutage.

Analytic baseline
-----------------
The amplitude series derives from a unit-variance Gaussian ``g`` through a
monotone map, so a fade in amplitude is an excursion of ``g`` below a
standardised level ``u``. For the lognormal marginal with
``ln I = mu + sigma * g``, ``mu = -sigma**2/2``:

    u = (2*ln(a_th) - mu) / sigma                                        (10)

**Continuous time, Gaussian kernel.** For a stationary, zero-mean,
unit-variance Gaussian process with twice-differentiable autocorrelation
``R(t)``, Rice's (1945) result gives the expected rate of down-crossings of
level ``u`` as

    N(u) = (1 / (2*pi)) * sqrt(-R''(0)) * exp(-u**2 / 2)                 (11)

For the Gaussian kernel ``R(t) = exp(-(t/tau)**2)``, ``R''(0) = -2/tau**2``, so

    N(u) = sqrt(2) / (2*pi*tau) * exp(-u**2 / 2)                         (12)
    MFD(u) = Phi(u) / N(u)                                               (13)

**Discrete time, Gauss-Markov kernel.** For ``R(t) = exp(-|t|/tau)``,
``R''(0)`` does not exist and (11) diverges: the continuous-time crossing rate
of an Ornstein-Uhlenbeck process is infinite. The sampled process is still
exactly tractable, because ``(g[n-1], g[n])`` is bivariate normal with
correlation ``rho = exp(-1/(fs*tau))``:

    P(down-cross at n) = Phi(u) - Phi_2(u, u; rho)                       (14)
    LCR = fs * (Phi(u) - Phi_2(u, u; rho))                               [s^-1]
    MFD = Phi(u) / LCR                                                    [s]  (15)

Equation (14) is exact for the sampled AR(1) path, and (15) follows from
renewal accounting (expected fraction of time below / expected entries per
second). Both are implemented here and verified against sample statistics in
``validation/validate_crossing_convergence.py``, which also shows (14) growing
without bound as ``fs`` rises while (12) converges.

**Fade-duration exceedance.** Level-crossing theory gives the *mean* duration,
not its distribution. The standard engineering closure is the memoryless one:

    P(T > t) ~ exp(-t / MFD)                                             (16)

which is the analytic baseline the learned predictor in ``predictor.py`` is
benchmarked against. Equation (16) is an approximation with no claim to
exactness; its measured error is reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats


@dataclass(frozen=True)
class FadeStatistics:
    """Sample fade statistics of one amplitude record.

    Attributes
    ----------
    threshold:
        Amplitude threshold used, dimensionless.
    n_samples:
        Record length, samples.
    sample_rate_hz:
        Sampling rate, Hz.
    down_crossings:
        Number of down-crossings as defined in the module docstring.
    level_crossing_rate_hz:
        ``down_crossings / (n_samples / fs)``, s^-1.
    complete_fades:
        Number of complete (uncensored) fade runs.
    censored_fades:
        Number of runs touching either end of the record, excluded from MFD.
    mean_fade_duration_s:
        Mean complete-run length / fs, s. ``nan`` if ``complete_fades == 0``.
    median_fade_duration_s:
        Median complete-run length / fs, s. ``nan`` if no complete fades.
    max_fade_duration_s:
        Longest complete run / fs, s. ``nan`` if no complete fades.
    outage_fraction:
        ``mean(a < threshold)``, dimensionless.
    durations_s:
        Complete-run durations, s, in order of occurrence.
    """

    threshold: float
    n_samples: int
    sample_rate_hz: float
    down_crossings: int
    level_crossing_rate_hz: float
    complete_fades: int
    censored_fades: int
    mean_fade_duration_s: float
    median_fade_duration_s: float
    max_fade_duration_s: float
    outage_fraction: float
    durations_s: np.ndarray


def fade_runs(below: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start indices and lengths of every maximal run of ``True`` in ``below``.

    Returns ``(starts, lengths)``, both int64 arrays of equal length.
    """
    below = np.asarray(below, dtype=bool)
    if below.ndim != 1:
        raise ValueError(f"below must be 1-D, got shape {below.shape}")
    if below.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    padded = np.concatenate(([False], below, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.nonzero(edges == 1)[0].astype(np.int64)
    ends = np.nonzero(edges == -1)[0].astype(np.int64)
    return starts, (ends - starts).astype(np.int64)


def fade_statistics(
    amplitude: np.ndarray, threshold: float, sample_rate_hz: float
) -> FadeStatistics:
    """Sample fade statistics, by the definitions in the module docstring.

    Parameters
    ----------
    amplitude:
        Amplitude series, dimensionless, same normalisation as ``threshold``.
    threshold:
        Amplitude threshold, > 0.
    sample_rate_hz:
        Sampling rate, Hz, > 0.
    """
    a = np.asarray(amplitude, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"amplitude must be 1-D, got shape {a.shape}")
    if a.size < 2:
        raise ValueError(f"amplitude must have at least 2 samples, got {a.size}")
    if not threshold > 0:
        raise ValueError(f"threshold must be > 0, got {threshold!r}")
    if not sample_rate_hz > 0:
        raise ValueError(f"sample_rate_hz must be > 0 Hz, got {sample_rate_hz!r}")

    below = a < threshold
    down = int(np.count_nonzero(below[1:] & ~below[:-1]))
    starts, lengths = fade_runs(below)
    if starts.size:
        complete_mask = (starts > 0) & (starts + lengths < a.size)
    else:
        complete_mask = np.zeros(0, dtype=bool)
    complete = lengths[complete_mask]
    durations = complete / float(sample_rate_hz)
    record_s = a.size / float(sample_rate_hz)
    return FadeStatistics(
        threshold=float(threshold),
        n_samples=int(a.size),
        sample_rate_hz=float(sample_rate_hz),
        down_crossings=down,
        level_crossing_rate_hz=down / record_s,
        complete_fades=int(complete.size),
        censored_fades=int(starts.size - complete.size),
        mean_fade_duration_s=float(durations.mean()) if durations.size else float("nan"),
        median_fade_duration_s=(
            float(np.median(durations)) if durations.size else float("nan")
        ),
        max_fade_duration_s=float(durations.max()) if durations.size else float("nan"),
        outage_fraction=float(np.mean(below)),
        durations_s=durations,
    )


def lognormal_standard_level(threshold_amplitude: float, scintillation_index: float) -> float:
    """Standardised Gaussian level ``u`` for an amplitude threshold, equation (10).

    Parameters
    ----------
    threshold_amplitude:
        Amplitude threshold with ``E[a**2] = 1``, > 0.
    scintillation_index:
        ``Var[I]/E[I]**2``, > 0.

    Returns
    -------
    ``u``, dimensionless.
    """
    if not threshold_amplitude > 0:
        raise ValueError(f"threshold_amplitude must be > 0, got {threshold_amplitude!r}")
    if not scintillation_index > 0:
        raise ValueError(f"scintillation_index must be > 0, got {scintillation_index!r}")
    sigma2 = float(np.log1p(scintillation_index))
    sigma = np.sqrt(sigma2)
    return float((2.0 * np.log(threshold_amplitude) + 0.5 * sigma2) / sigma)


def rice_crossing_rate_gauss_kernel(
    standard_level: float, correlation_time_s: float
) -> float:
    """Continuous-time down-crossing rate for the Gaussian kernel, equation (12).

    Units: s^-1. Valid only for ``kernel="gauss"``; for ``kernel="exp"`` the
    continuous-time rate is infinite and :func:`markov_crossing_rate` must be
    used instead.
    """
    if not correlation_time_s > 0:
        raise ValueError(f"correlation_time_s must be > 0 s, got {correlation_time_s!r}")
    u = float(standard_level)
    return float(np.sqrt(2.0) / (2.0 * np.pi * correlation_time_s) * np.exp(-0.5 * u * u))


def rice_mean_fade_duration_gauss_kernel(
    standard_level: float, correlation_time_s: float
) -> float:
    """Continuous-time mean fade duration for the Gaussian kernel, equation (13). Seconds."""
    u = float(standard_level)
    rate = rice_crossing_rate_gauss_kernel(u, correlation_time_s)
    if rate <= 0.0:
        return float("inf")
    return float(special.ndtr(u) / rate)


def markov_crossing_rate(
    standard_level: float, correlation_time_s: float, sample_rate_hz: float
) -> float:
    """Exact down-crossing rate of the **sampled** Gauss-Markov path, equation (14).

    Units: s^-1. Depends on ``sample_rate_hz`` by construction; this is the
    correct behaviour, because the continuous-time rate of an
    Ornstein-Uhlenbeck process is infinite.
    """
    if not correlation_time_s > 0:
        raise ValueError(f"correlation_time_s must be > 0 s, got {correlation_time_s!r}")
    if not sample_rate_hz > 0:
        raise ValueError(f"sample_rate_hz must be > 0 Hz, got {sample_rate_hz!r}")
    u = float(standard_level)
    rho = float(np.exp(-1.0 / (sample_rate_hz * correlation_time_s)))
    joint = float(
        stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]]).cdf([u, u])
    )
    p_down = float(special.ndtr(u)) - joint
    return float(sample_rate_hz * max(p_down, 0.0))


def markov_mean_fade_duration(
    standard_level: float, correlation_time_s: float, sample_rate_hz: float
) -> float:
    """Exact mean fade duration of the sampled Gauss-Markov path, equation (15). Seconds."""
    u = float(standard_level)
    rate = markov_crossing_rate(u, correlation_time_s, sample_rate_hz)
    if rate <= 0.0:
        return float("inf")
    return float(special.ndtr(u) / rate)


def exponential_exceedance(duration_s: np.ndarray | float, mean_fade_duration_s: float):
    """``P(T > t) = exp(-t / MFD)``, equation (16). Dimensionless probability.

    This is the analytic exceedance baseline. It is an approximation: the
    level-crossing result fixes only the mean.
    """
    if not mean_fade_duration_s > 0:
        raise ValueError(
            f"mean_fade_duration_s must be > 0 s, got {mean_fade_duration_s!r}"
        )
    t = np.asarray(duration_s, dtype=np.float64)
    if np.any(t < 0):
        raise ValueError("duration_s must be >= 0 s")
    return np.exp(-t / float(mean_fade_duration_s))


def required_interleaver_depth(
    mean_fade_duration_s: float, symbol_rate_hz: float, margin: float = 1.0
) -> int:
    """Interleaver depth that spreads a mean-length fade across distinct codewords.

    A block interleaver of depth ``D`` separates consecutive symbols of one
    codeword by ``D`` symbol periods, so a burst of ``L`` consecutive channel
    symbols deposits at most ``ceil(L / D)`` errors in any one codeword. Setting
    ``D = margin * MFD * Rs`` makes that at most ``ceil(1/margin)``.

    Parameters
    ----------
    mean_fade_duration_s:
        Mean fade duration, s, > 0.
    symbol_rate_hz:
        Channel symbol rate, Hz, > 0.
    margin:
        Multiplier on the mean fade duration, > 0. ``margin = 1`` sizes to the
        mean, which by construction is exceeded by roughly ``exp(-1)`` of fades
        under equation (16); size up accordingly.

    Returns
    -------
    Depth in symbols, >= 1.
    """
    if not mean_fade_duration_s > 0:
        raise ValueError(f"mean_fade_duration_s must be > 0 s, got {mean_fade_duration_s!r}")
    if not symbol_rate_hz > 0:
        raise ValueError(f"symbol_rate_hz must be > 0 Hz, got {symbol_rate_hz!r}")
    if not margin > 0:
        raise ValueError(f"margin must be > 0, got {margin!r}")
    return int(max(1, np.ceil(margin * mean_fade_duration_s * symbol_rate_hz)))
