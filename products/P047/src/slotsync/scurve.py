"""Detector S-curves and the detector gain K_d measured from them.

The **S-curve** of a timing-error detector is the mean detector output as a
function of the true timing offset, with the noise averaged out:

    ``S(eps) = E[ e | timing error = eps ]``

The expectation is over the data.  In this module it is an **exact** ensemble
average: because every pulse shape in :mod:`slotsync.pulses` has finite support,
only a finite window of symbols can reach any of the detector's samples, so the
average is computed by enumerating every data pattern in that window rather than
by Monte Carlo.  For a window of ``n`` symbols that is ``2**n`` patterns; when
``n`` exceeds ``max_exact_symbols`` the function falls back to a seeded Monte
Carlo average and says so in :attr:`SCurve.exact`.

The quantity the loop analysis needs is the **detector gain**

    ``K_d = dS/d(eps)`` at ``eps = 0``

in units of detector output per symbol period of timing error.  It is measured
here by a least-squares straight line through the origin over a small window of
the computed S-curve, and cross-checked by a central difference.  It is never
asserted: :mod:`slotsync.loop` takes ``K_d`` as an argument and every example and
validation script passes it the measured value.  GNU Radio's Symbol Sync block
documents the same quantity under the name "Expected TED Gain", described there
as "the slope of the TED's S-curve at timing offset tau = 0", which is the
definition used here.

What the S-curve also tells you
-------------------------------
* :attr:`SCurve.linear_halfwidth` - how far the S-curve stays within 10 % of the
  straight line ``K_d * eps``.  Outside it the loop's linearised analysis stops
  applying, and the jitter prediction degrades accordingly.
* :attr:`SCurve.peak_offset` - where the S-curve flattens, i.e. the largest
  timing error the detector still reports at full strength.
* :attr:`SCurve.reversal_offset` - where it crosses zero again.  Beyond that
  point the detector pushes the loop the **wrong way**; this is the boundary
  that makes a cycle slip possible and it sets the ``+-boundary`` used by
  :func:`slotsync.loop.cycle_slip_rate_rice`.
* :attr:`SCurve.self_noise_at_origin` - the pattern-to-pattern standard
  deviation of the detector output at zero offset.  It does not go away as the
  channel noise goes to zero; it is the detector's self-noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .pulses import PulseShape
from .stream import antipodal_patterns, ook_patterns, sample_matrix
from .ted import TedConfig, evaluate_ted, ted_decision_offsets, ted_time_offsets

__all__ = ["SCurve", "default_offsets", "scurve"]

_FIT_HALFWIDTH = 0.05
_LINEARITY_TOLERANCE = 0.10


def default_offsets(halfwidth: float = 0.75, points: int = 301) -> np.ndarray:
    """Symmetric offset grid in symbol periods, guaranteed to contain exactly 0.

    ``points`` must be odd so that the origin is a grid point; the gain
    measurement and the lock-point bias both read the value there.
    """
    if points % 2 == 0 or points < 11:
        raise ValueError(f"points must be odd and at least 11, got {points}")
    if not 0.05 < halfwidth <= 2.0:
        raise ValueError(f"halfwidth must lie in (0.05, 2.0] symbol periods, got {halfwidth!r}")
    return np.linspace(-halfwidth, halfwidth, points)


@dataclass(frozen=True)
class SCurve:
    """A computed S-curve and the quantities read off it.

    Attributes
    ----------
    offsets
        Timing offsets, symbol periods.
    values
        Mean detector output at each offset, dimensionless.
    dispersion
        Standard deviation of the detector output over the data ensemble at each
        offset, dimensionless.  This is detector self-noise; no channel noise is
        present in an S-curve.
    gain
        Detector gain ``K_d``, detector output per symbol period of timing
        error, from a least-squares line through the origin.
    gain_central_difference
        The same gain from a two-point central difference, as a cross-check.
    bias
        ``S(0)``.  A non-zero value means the detector's zero crossing is not at
        the true symbol centre, so the loop locks with a static timing offset.
    exact
        True when the ensemble average enumerated every data pattern.
    """

    config: TedConfig
    pulse_name: str
    offsets: np.ndarray
    values: np.ndarray
    dispersion: np.ndarray
    gain: float
    gain_central_difference: float
    bias: float
    exact: bool
    pattern_count: int
    symbol_window: int

    @property
    def self_noise_at_origin(self) -> float:
        """Standard deviation of the detector output over the data, at ``eps = 0``."""
        return float(self.dispersion[self._origin_index()])

    def _origin_index(self) -> int:
        return int(np.argmin(np.abs(self.offsets)))

    @property
    def linear_halfwidth(self) -> float:
        """Largest ``r`` with ``|S(eps) - K_d eps| <= 10 % |K_d eps|`` for all ``|eps| <= r``.

        Returns 0.0 when the first off-origin grid point already violates the
        tolerance, which happens when the gain is effectively zero.
        """
        if self.gain == 0.0:
            return 0.0
        order = np.argsort(np.abs(self.offsets))
        best = 0.0
        for idx in order:
            eps = float(self.offsets[idx])
            if eps == 0.0:
                continue
            straight = self.gain * eps
            if abs(float(self.values[idx]) - straight) > _LINEARITY_TOLERANCE * abs(straight):
                break
            best = abs(eps)
        return best

    @property
    def peak_offset(self) -> float:
        """Positive offset at which ``|S|`` is largest: where the S-curve flattens."""
        positive = self.offsets > 0.0
        if not positive.any():
            return float("nan")
        sub = np.abs(self.values[positive])
        return float(self.offsets[positive][int(np.argmax(sub))])

    @property
    def peak_value(self) -> float:
        """``S`` at :attr:`peak_offset`."""
        positive = self.offsets > 0.0
        sub = np.abs(self.values[positive])
        return float(self.values[positive][int(np.argmax(sub))])

    @property
    def reversal_offset(self) -> float | None:
        """Smallest positive offset where ``S`` changes sign, or ``None`` if it does not.

        Found by linear interpolation between the two bracketing grid points.
        Beyond this offset the detector drives the loop away from lock.
        """
        positive = np.where(self.offsets > 0.0)[0]
        if positive.size == 0:
            return None
        sign0 = math.copysign(1.0, self.gain) if self.gain != 0.0 else 1.0
        for a, b in zip(positive[:-1], positive[1:], strict=False):
            va, vb = float(self.values[a]), float(self.values[b])
            if va * sign0 > 0.0 >= vb * sign0:
                if va == vb:
                    return float(self.offsets[b])
                frac = va / (va - vb)
                return float(self.offsets[a] + frac * (self.offsets[b] - self.offsets[a]))
        return None

    def summary(self) -> dict[str, object]:
        """Flat dictionary of the headline quantities, for printing and serialising."""
        reversal = self.reversal_offset
        return {
            "detector": self.config.label,
            "alphabet": self.config.alphabet,
            "pulse": self.pulse_name,
            "exact": self.exact,
            "patterns": self.pattern_count,
            "symbol_window": self.symbol_window,
            "K_d": self.gain,
            "K_d_central_difference": self.gain_central_difference,
            "bias_at_zero": self.bias,
            "self_noise_at_zero": self.self_noise_at_origin,
            "linear_halfwidth_symbols": self.linear_halfwidth,
            "peak_offset_symbols": self.peak_offset,
            "peak_value": self.peak_value,
            "reversal_offset_symbols": reversal,
        }


def scurve(
    config: TedConfig,
    pulse: PulseShape,
    offsets: np.ndarray | None = None,
    *,
    max_exact_symbols: int = 14,
    trials: int = 20000,
    seed: int = 20261006,
    chunk: int = 512,
) -> SCurve:
    """Compute a detector S-curve, exactly where the pattern count permits.

    Parameters
    ----------
    config
        Detector configuration.
    pulse
        Pulse shape.  Its finite support fixes the symbol window that has to be
        enumerated.
    offsets
        Timing offsets in symbol periods; defaults to :func:`default_offsets`.
    max_exact_symbols
        Enumerate all ``2**n`` patterns while the window is at most this many
        symbols, otherwise draw ``trials`` random patterns with ``seed``.
    trials, seed
        Monte Carlo fallback size and seed.
    chunk
        Patterns evaluated per block, to bound peak memory.  Does not change the
        result.

    Returns
    -------
    SCurve
        With the gain, bias, self-noise and validity range already read off.

    Notes
    -----
    The decisions handed to the decision-directed detectors are the **true**
    symbols, so this is the correct-decision S-curve.  A real receiver's
    decisions are wrong some of the time and its S-curve is correspondingly
    shallower; the README says so under Limitations.
    """
    eps = default_offsets() if offsets is None else np.asarray(offsets, dtype=float)
    if eps.ndim != 1 or eps.size < 3:
        raise ValueError(f"offsets must be a 1-D array of at least 3 points, got shape {eps.shape}")

    taps = np.asarray(ted_time_offsets(config), dtype=float)
    times = eps[:, None] + taps[None, :]
    span = pulse.isi_span_symbols
    m_lo = int(math.floor(times.min() - span))
    m_hi = int(math.ceil(times.max() + span))
    m_index = np.arange(m_lo, m_hi + 1)
    window = int(m_index.size)

    matrix = sample_matrix(times, m_index.astype(float), pulse)  # (n_eps, n_taps, n_m)

    exact = window <= max_exact_symbols
    if exact:
        patterns = (
            antipodal_patterns(window)
            if config.alphabet == "antipodal"
            else ook_patterns(window)
        )
    else:
        rng = np.random.default_rng(seed)
        bits = rng.integers(0, 2, size=(trials, window)).astype(float)
        patterns = 2.0 * bits - 1.0 if config.alphabet == "antipodal" else bits

    decision_offsets = ted_decision_offsets(config)
    decision_columns = [off - m_lo for off in decision_offsets]
    for col in decision_columns:
        if not 0 <= col < window:
            raise ValueError(
                "the symbol window does not cover a required decision; this is an "
                "internal inconsistency between the pulse support and the detector"
            )

    n_eps = eps.size
    total = patterns.shape[0]
    accum = np.zeros(n_eps)
    accum_sq = np.zeros(n_eps)
    for start in range(0, total, chunk):
        block = patterns[start : start + chunk]
        samples = np.einsum("otm,pm->opt", matrix, block, optimize=True)
        if decision_columns:
            dec = block[:, decision_columns]  # (p, n_dec)
            dec_b = np.broadcast_to(dec[None, :, :], (n_eps, block.shape[0], len(decision_columns)))
            out = evaluate_ted(config, samples, dec_b)
        else:
            out = evaluate_ted(config, samples)
        accum += out.sum(axis=1)
        accum_sq += (out * out).sum(axis=1)

    mean = accum / total
    variance = np.maximum(accum_sq / total - mean * mean, 0.0)
    dispersion = np.sqrt(variance)

    fit = np.abs(eps) <= _FIT_HALFWIDTH
    if fit.sum() < 3:
        fit = np.abs(eps) <= np.sort(np.abs(eps))[2]
    denom = float(np.sum(eps[fit] ** 2))
    gain = float(np.sum(eps[fit] * mean[fit]) / denom) if denom > 0.0 else 0.0

    origin = int(np.argmin(np.abs(eps)))
    if 0 < origin < n_eps - 1:
        central = float(
            (mean[origin + 1] - mean[origin - 1]) / (eps[origin + 1] - eps[origin - 1])
        )
    else:
        central = gain

    return SCurve(
        config=config,
        pulse_name=pulse.name,
        offsets=eps,
        values=mean,
        dispersion=dispersion,
        gain=gain,
        gain_central_difference=central,
        bias=float(mean[origin]),
        exact=exact,
        pattern_count=int(total),
        symbol_window=window,
    )
