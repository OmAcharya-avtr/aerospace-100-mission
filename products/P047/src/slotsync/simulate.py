"""Open-loop detector statistics and closed-loop Monte Carlo of the timing loop.

Two measurements live here, and the whole validation argument is the comparison
between them and the predictions in :mod:`slotsync.loop`:

* :func:`measure_ted_statistics` - hold the timing estimate fixed, add noise, and
  measure the detector output's mean and variance.  The variance at zero offset
  is ``sigma_n^2``, the one quantity besides ``K_d`` that the closed-form jitter
  prediction needs.  Fully vectorised.
* :func:`run_timing_loop` - close the loop and run it symbol by symbol, with the
  real detector, the real pulse shape, the real noise and the real nonlinear
  S-curve.  Reports the measured jitter variance and the measured cycle-slip
  count.

Neither function is given the other's answer.  ``K_d`` comes from
:func:`slotsync.scurve.scurve`, ``sigma_n^2`` from the open-loop measurement, the
prediction from :mod:`slotsync.loop`, and the closed-loop run never sees any of
them except through the loop coefficients.

Pulse evaluation inside the sequential loop
-------------------------------------------
The closed loop samples at instants the loop itself chooses, so the pulse has to
be evaluated at arbitrary times.  Calling into numpy per sample would dominate
the runtime, so the loop uses a precomputed table of the pulse with linear
interpolation between entries.  The interpolation error is bounded and measured:
``tests/test_simulate.py`` compares the table against the exact shape over a
dense grid and asserts the maximum absolute error, and the default table
(``2**14 + 1`` points across the support) keeps it near 1e-07 for every shape in
:mod:`slotsync.pulses`.  The open-loop measurement uses exact evaluation, so the
two paths differ by that bound and no more.

Compute budget
--------------
Measured on the build host (Python 3.13, two shared cores): a closed-loop run of
120000 symbols with the Gardner detector on a truncated Nyquist raised cosine
takes about 2 s, so a sweep of eight operating points fits inside the 3-minute
budget with room to spare.  Every validation script states its own runtime.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .loop import LoopDesign
from .pulses import PulseShape
from .stream import sample_noise_sigma
from .ted import (
    TedConfig,
    evaluate_ted,
    evaluate_ted_scalar,
    ted_decision_offsets,
    ted_fractional_positions,
    ted_sample_slots,
    ted_time_offsets,
)

__all__ = [
    "LoopRun",
    "TedStatistics",
    "measure_ted_autocovariance",
    "measure_ted_statistics",
    "pulse_table",
    "run_timing_loop",
]


def pulse_table(pulse: PulseShape, points: int = 16385) -> Callable[[float], float]:
    """Linear-interpolating lookup of ``pulse`` over its support, as a scalar callable.

    Returns a closure taking a time in symbol periods and returning the
    amplitude, or 0.0 outside the support.  ``points`` must be odd so that the
    pulse centre is a table entry.
    """
    if points % 2 == 0 or points < 257:
        raise ValueError(f"points must be odd and at least 257, got {points}")
    half = pulse.half_support
    grid = np.linspace(-half, half, points)
    values = pulse.amplitude(grid)
    step = 2.0 * half / (points - 1)
    table = values.tolist()
    last = points - 1

    def lookup(t: float) -> float:
        position = (t + half) / step
        index = int(position)
        if index < 0 or index >= last:
            if position == last:
                return table[last]
            return 0.0
        frac = position - index
        low = table[index]
        return low + frac * (table[index + 1] - low)

    return lookup


def _symbols(rng: np.random.Generator, count: int, alphabet: str) -> np.ndarray:
    bits = rng.integers(0, 2, size=count).astype(float)
    return 2.0 * bits - 1.0 if alphabet == "antipodal" else bits


@dataclass(frozen=True)
class TedStatistics:
    """Open-loop detector statistics at a fixed timing offset.

    Attributes
    ----------
    offset
        The fixed timing error held during the measurement, symbol periods.
    mean
        Mean detector output.  At ``offset = 0`` a non-zero value is a lock-point
        bias.
    variance
        Variance of the detector output: this is ``sigma_n^2``.  It contains both
        the channel-noise contribution and the detector's data-dependent
        self-noise, which is what the loop actually sees.
    self_noise_variance
        The same measurement repeated with the channel noise switched off, so the
        split between self-noise and channel noise is visible rather than
        assumed.
    """

    config: TedConfig
    pulse_name: str
    offset: float
    sample_snr_db: float
    samples: int
    mean: float
    variance: float
    self_noise_variance: float

    @property
    def channel_noise_variance(self) -> float:
        """``variance - self_noise_variance``: the part the channel contributed."""
        return self.variance - self.self_noise_variance


def measure_ted_statistics(
    config: TedConfig,
    pulse: PulseShape,
    *,
    sample_snr_db: float,
    offset: float = 0.0,
    samples: int = 200000,
    seed: int = 20261006,
    use_true_decisions: bool = False,
) -> TedStatistics:
    """Mean and variance of the detector output at a fixed timing offset.

    Vectorised: the symbol stream is turned into a sliding window and the pulse
    contributions become one matrix product, so 200000 samples take well under a
    second.

    Parameters
    ----------
    config, pulse
        Detector and pulse shape.
    sample_snr_db
        Per-sample SNR; see :mod:`slotsync.stream` for its exact definition.
    offset
        Fixed timing error to hold, symbol periods.
    samples
        Number of detector updates to average over.
    seed
        Seed for the data and the noise.
    use_true_decisions
        If true, the decision-directed detectors are given the transmitted
        symbols instead of sliced ones.  The difference between the two is the
        decision-error contribution, which this package reports rather than
        assumes away.

    Returns
    -------
    TedStatistics
    """
    if samples < 1000:
        raise ValueError(f"samples must be at least 1000, got {samples}")
    taps = np.asarray(ted_time_offsets(config), dtype=float)
    span = pulse.isi_span_symbols
    reach = span + 2
    width = 2 * reach + 1
    rng = np.random.default_rng(seed)
    data = _symbols(rng, samples + width, config.alphabet)
    windows = np.lib.stride_tricks.sliding_window_view(data, width)[:samples]
    relative = np.arange(-reach, reach + 1, dtype=float)
    weights = pulse.amplitude((offset + taps)[:, None] - relative[None, :])  # (n_taps, width)
    clean = windows @ weights.T  # (samples, n_taps)

    # One noise sample per physical instant, not one per tap: consecutive updates
    # share samples (see slotsync.ted.ted_sample_slots), and drawing fresh noise
    # per tap would make the detector noise artificially white.
    sigma = sample_noise_sigma(sample_snr_db)
    slots = ted_sample_slots(config)
    n_fractions = len(ted_fractional_positions(config))
    max_lag = max(lag for lag, _ in slots)
    pool = sigma * rng.standard_normal((samples + max_lag, n_fractions))
    columns = [pool[max_lag - lag : max_lag - lag + samples, index] for lag, index in slots]
    noisy = clean + np.stack(columns, axis=-1)

    needed = ted_decision_offsets(config)

    def decisions_for(values: np.ndarray) -> np.ndarray | None:
        if not needed:
            return None
        columns = []
        strobe_index = list(taps).index(0.0)
        for off in needed:
            if use_true_decisions:
                columns.append(windows[:, reach + off])
            elif off == 0:
                columns.append(_slice(values[:, strobe_index], config.alphabet))
            else:
                shifted = np.empty(values.shape[0])
                shifted[0] = 0.0
                shifted[1:] = _slice(values[:-1, strobe_index], config.alphabet)
                columns.append(shifted)
        return np.stack(columns, axis=-1)

    noisy_out = evaluate_ted(config, noisy, decisions_for(noisy))
    clean_out = evaluate_ted(config, clean, decisions_for(clean))
    # The first update has no previous decision, so it is dropped rather than
    # left to contribute a spurious 1/N to the variance.
    if any(off < 0 for off in needed):
        noisy_out = noisy_out[1:]
        clean_out = clean_out[1:]
    return TedStatistics(
        config=config,
        pulse_name=pulse.name,
        offset=float(offset),
        sample_snr_db=float(sample_snr_db),
        samples=int(noisy_out.size),
        mean=float(noisy_out.mean()),
        variance=float(noisy_out.var()),
        self_noise_variance=float(clean_out.var()),
    )


def _slice(values: np.ndarray, alphabet: str) -> np.ndarray:
    if alphabet == "antipodal":
        return np.where(values >= 0.0, 1.0, -1.0)
    return np.where(values >= 0.5, 1.0, 0.0)


def _slice_scalar(value: float, alphabet: str) -> float:
    if alphabet == "antipodal":
        return 1.0 if value >= 0.0 else -1.0
    return 1.0 if value >= 0.5 else 0.0


@dataclass
class LoopRun:
    """Result of one closed-loop Monte Carlo run.

    Attributes
    ----------
    error
        Wrapped timing error history after the discard period, symbol periods.
        "Wrapped" means the integer part has been removed, so a slip appears as a
        step in :attr:`slip_count` rather than as a ramp in the error.
    jitter_variance
        Variance of :attr:`error`, squared symbol periods.  This is the number
        compared against :func:`slotsync.loop.jitter_variance_closed_form`.
    slip_count
        Number of times the loop changed which symbol it was locked to.
    """

    config: TedConfig
    pulse_name: str
    design: LoopDesign
    sample_snr_db: float
    n_symbols: int
    discarded: int
    true_offset: float
    error: np.ndarray = field(repr=False)
    mean_error: float = 0.0
    jitter_variance: float = 0.0
    slip_count: int = 0
    diverged: bool = False
    jitter_variance_standard_error: float = 0.0
    mean_error_standard_error: float = 0.0

    @property
    def jitter_rms(self) -> float:
        """Root-mean-square timing jitter, symbol periods."""
        return math.sqrt(max(self.jitter_variance, 0.0))

    @property
    def jitter_variance_relative_error(self) -> float:
        """Standard error of :attr:`jitter_variance` as a fraction of it."""
        if self.jitter_variance <= 0.0:
            return float("inf")
        return self.jitter_variance_standard_error / self.jitter_variance

    @property
    def slip_rate_per_symbol(self) -> float:
        """Measured slips per symbol over the measured interval."""
        measured = self.error.size
        return self.slip_count / measured if measured else 0.0

    def summary(self) -> dict[str, object]:
        """Flat dictionary of the headline quantities."""
        return {
            "detector": self.config.label,
            "pulse": self.pulse_name,
            "B_n": self.design.noise_bandwidth,
            "zeta": self.design.damping,
            "K_d": self.design.detector_gain,
            "sample_snr_db": self.sample_snr_db,
            "symbols": self.n_symbols,
            "measured": int(self.error.size),
            "mean_error": self.mean_error,
            "jitter_variance": self.jitter_variance,
            "jitter_variance_standard_error": self.jitter_variance_standard_error,
            "jitter_rms": self.jitter_rms,
            "mean_error_standard_error": self.mean_error_standard_error,
            "slips": self.slip_count,
            "slip_rate_per_symbol": self.slip_rate_per_symbol,
            "diverged": self.diverged,
        }


def run_timing_loop(
    config: TedConfig,
    pulse: PulseShape,
    design: LoopDesign,
    *,
    n_symbols: int = 120000,
    sample_snr_db: float = 20.0,
    true_offset: float = 0.0,
    initial_error: float = 0.0,
    discard: int | None = None,
    seed: int = 20261006,
    table_points: int = 16385,
    use_true_decisions: bool = False,
    divergence_limit: float = 1000.0,
) -> LoopRun:
    """Close the loop and run it, one detector update per symbol.

    Parameters
    ----------
    config, pulse, design
        Detector, pulse shape and loop coefficients.  ``design`` must have been
        built with the ``K_d`` measured for *this* detector and *this* pulse.
    n_symbols
        Total symbols processed.
    sample_snr_db
        Per-sample SNR.
    true_offset
        The transmitter's timing offset the loop has to find, symbol periods.
    initial_error
        Initial value of ``tau_hat - true_offset``, symbol periods.  Used by the
        acquisition example.
    discard
        Symbols to discard before measuring, to let the loop settle.  Defaults to
        ``min(n_symbols // 4, max(2000, int(20 / B_n)))`` - twenty loop time
        constants or a quarter of the run, whichever is smaller.
    seed
        Seed for the data and the noise.
    table_points
        Pulse table resolution; see :func:`pulse_table`.
    use_true_decisions
        Give the decision-directed detectors the transmitted symbols instead of
        sliced ones.
    divergence_limit
        If ``|tau_hat - true_offset|`` exceeds this many symbol periods the run is
        stopped and flagged as diverged rather than producing a meaningless
        variance.  The default is deliberately loose: a loop at low loop SNR
        random-walks across many symbols by slipping, which is a measurement, not
        a divergence, and the wrapped error is still the right quantity there.
        Tighten it only to detect genuine numerical blow-up.

    Returns
    -------
    LoopRun
    """
    if n_symbols < 2000:
        raise ValueError(f"n_symbols must be at least 2000, got {n_symbols}")
    if not design.is_stable:
        raise ValueError(
            "the loop design is unstable; refusing to run a Monte Carlo that cannot "
            "have a stationary distribution"
        )
    if discard is None:
        discard = min(n_symbols // 4, max(2000, int(20.0 / design.noise_bandwidth)))
    if not 0 <= discard < n_symbols:
        raise ValueError(f"discard must lie in [0, n_symbols), got {discard}")

    taps = tuple(float(t) for t in ted_time_offsets(config))
    strobe_index = taps.index(0.0)
    needed = ted_decision_offsets(config)
    span = pulse.isi_span_symbols
    reach = span + 2

    rng = np.random.default_rng(seed)
    data = _symbols(rng, n_symbols + 2 * reach + 4, config.alphabet).tolist()
    sigma = sample_noise_sigma(sample_snr_db)
    slots = ted_sample_slots(config)
    n_fractions = len(ted_fractional_positions(config))
    max_lag = max(lag for lag, _ in slots)
    noise_pool = (sigma * rng.standard_normal((n_symbols + max_lag, n_fractions))).tolist()

    lookup = pulse_table(pulse, table_points)
    k1 = design.k_proportional
    k2 = design.k_integral
    alphabet = config.alphabet

    tau_hat = true_offset + initial_error
    velocity = 0.0
    previous_decision = 0.0
    previous_lock = 0
    slips = 0
    errors: list[float] = []
    diverged = False

    neighbourhood = range(-reach, reach + 1)
    for k in range(n_symbols):
        base = k + tau_hat
        values: list[float] = []
        for tap_index, tap in enumerate(taps):
            lag, fraction_index = slots[tap_index]
            t = base + tap
            centre = k + int(round(tap))
            total = 0.0
            for d in neighbourhood:
                m = centre + d
                if 0 <= m < len(data):
                    amplitude = data[m]
                    if amplitude != 0.0:
                        total += amplitude * lookup(t - m - true_offset)
            values.append(total + noise_pool[max_lag + k - lag][fraction_index])

        if needed:
            current = (
                data[k]
                if use_true_decisions
                else _slice_scalar(values[strobe_index], alphabet)
            )
            if len(needed) == 1:
                decisions = (current,)
            else:
                decisions = (previous_decision, current)
            previous_decision = current
        else:
            decisions = ()

        error_signal = evaluate_ted_scalar(config, tuple(values), decisions)
        velocity -= k2 * error_signal
        tau_hat += -k1 * error_signal + velocity

        raw = tau_hat - true_offset
        if abs(raw) > divergence_limit:
            diverged = True
            break
        lock = int(round(raw))
        if lock != previous_lock:
            slips += 1
            previous_lock = lock
        if k >= discard:
            errors.append(raw - lock)

    history = np.asarray(errors, dtype=float)
    run = LoopRun(
        config=config,
        pulse_name=pulse.name,
        design=design,
        sample_snr_db=float(sample_snr_db),
        n_symbols=int(n_symbols),
        discarded=int(discard),
        true_offset=float(true_offset),
        error=history,
        diverged=diverged,
    )
    if history.size:
        run.mean_error = float(history.mean())
        run.jitter_variance = float(history.var())
        mean_se, var_se = _batch_standard_errors(history, design.noise_bandwidth)
        run.mean_error_standard_error = mean_se
        run.jitter_variance_standard_error = var_se
    run.slip_count = int(slips)
    return run


def _batch_standard_errors(history: np.ndarray, noise_bandwidth: float) -> tuple[float, float]:
    """Batch-means standard errors of the mean and the variance of a correlated series.

    The timing-error history is correlated over roughly ``1 / (2 B_n)`` symbols, so
    the naive ``sigma / sqrt(N)`` understates the uncertainty by an order of
    magnitude.  The series is split into batches four correlation times long, the
    statistic is formed per batch, and the standard error of the batch mean is
    reported.  Returns ``(0.0, 0.0)`` when fewer than eight batches fit, because a
    standard error from fewer than eight batches is not worth printing.
    """
    correlation = max(1, int(round(4.0 / (2.0 * noise_bandwidth))))
    batches = history.size // correlation
    if batches < 8:
        return 0.0, 0.0
    usable = history[: batches * correlation].reshape(batches, correlation)
    batch_means = usable.mean(axis=1)
    centred = history - history.mean()
    squares = (centred[: batches * correlation] ** 2).reshape(batches, correlation)
    batch_variances = squares.mean(axis=1)
    mean_se = float(batch_means.std(ddof=1) / math.sqrt(batches))
    var_se = float(batch_variances.std(ddof=1) / math.sqrt(batches))
    return mean_se, var_se


def measure_ted_autocovariance(
    config: TedConfig,
    pulse: PulseShape,
    *,
    sample_snr_db: float,
    max_lag: int = 32,
    offset: float = 0.0,
    samples: int = 400000,
    seed: int = 20261006,
    use_true_decisions: bool = False,
) -> np.ndarray:
    """Autocovariance ``R_n[0 .. max_lag]`` of the detector output at a fixed offset.

    The detector output at zero timing error is a constant plus a zero-mean
    fluctuation; this measures that fluctuation's autocovariance.  ``R_n[0]`` is
    the same quantity as :attr:`TedStatistics.variance`.  Lags beyond zero are
    what distinguishes self-noise from channel noise: channel noise contributes
    only at the lags where consecutive updates share a sample, while self-noise
    is correlated over the pulse's whole inter-symbol span.

    Feeds :func:`slotsync.loop.jitter_variance_coloured`.
    """
    if max_lag < 0:
        raise ValueError(f"max_lag must be non-negative, got {max_lag}")
    taps = np.asarray(ted_time_offsets(config), dtype=float)
    span = pulse.isi_span_symbols
    reach = span + 2
    width = 2 * reach + 1
    rng = np.random.default_rng(seed)
    data = _symbols(rng, samples + width, config.alphabet)
    windows = np.lib.stride_tricks.sliding_window_view(data, width)[:samples]
    relative = np.arange(-reach, reach + 1, dtype=float)
    weights = pulse.amplitude((offset + taps)[:, None] - relative[None, :])
    clean = windows @ weights.T

    sigma = sample_noise_sigma(sample_snr_db)
    slots = ted_sample_slots(config)
    n_fractions = len(ted_fractional_positions(config))
    max_slot_lag = max(lag for lag, _ in slots)
    pool = sigma * rng.standard_normal((samples + max_slot_lag, n_fractions))
    columns = [
        pool[max_slot_lag - lag : max_slot_lag - lag + samples, index] for lag, index in slots
    ]
    noisy = clean + np.stack(columns, axis=-1)

    needed = ted_decision_offsets(config)
    decisions = None
    if needed:
        strobe_index = int(np.where(taps == 0.0)[0][0])
        columns_d = []
        for off in needed:
            if use_true_decisions:
                columns_d.append(windows[:, reach + off])
            elif off == 0:
                columns_d.append(_slice(noisy[:, strobe_index], config.alphabet))
            else:
                shifted = np.empty(samples)
                shifted[0] = 0.0
                shifted[1:] = _slice(noisy[:-1, strobe_index], config.alphabet)
                columns_d.append(shifted)
        decisions = np.stack(columns_d, axis=-1)

    out = evaluate_ted(config, noisy, decisions)
    if any(off < 0 for off in needed):
        out = out[1:]
    centred = out - out.mean()
    n = centred.size
    result = np.empty(max_lag + 1)
    for j in range(max_lag + 1):
        result[j] = float(np.dot(centred[: n - j], centred[j:]) / n)
    return result
