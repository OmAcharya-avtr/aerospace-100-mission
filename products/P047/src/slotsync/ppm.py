"""Slot-clock synchronisation for M-ary pulse-position modulation.

What carries over from binary symbol timing, and what does not
-------------------------------------------------------------
**Carries over.**

* The whole chain of this package: a detector, its S-curve, the slope of that
  S-curve at the origin as the gain ``K_d``, a second-order loop designed from
  ``K_d`` and a normalised noise bandwidth, a jitter variance predicted from the
  detector's output noise, and a boundary-crossing estimate of the slip rate.
  Every formula in :mod:`slotsync.loop` applies unchanged with the symbol period
  ``T`` read as the **slot** period ``T_s``.
* The early-late family.  An energy early-late gate on the slot that carries the
  pulse is the natural slot-timing detector, and it is the one implemented here.

**Does not carry over.**

1. **Update rate.** There is one pulse per symbol, so there is one usable
   detector update per ``M`` slots, not one per slot.  A loop whose normalised
   bandwidth is ``B_n`` **per update** has an equivalent slot-rate bandwidth of
   ``B_n / M``: the loop is ``M`` times slower in slot time than the same number
   would suggest for a continuously modulated stream.  :func:`equivalent_slot_bandwidth`
   does that conversion and every figure in this module states which of the two
   it is using.
2. **Duty cycle.** Only ``1`` slot in ``M`` carries energy.  The detector sees
   signal on one slot and nothing on the other ``M - 1``, so for a given
   per-sample SNR the detector output variance per update does not fall with
   ``M`` while the gain stays fixed - the loop SNR degrades.
   :func:`duty_cycle_penalty_db` reports the measured penalty rather than an
   assumed ``10 log10(M)``.
3. **Gardner and Mueller-Mueller do not transfer.** Both are built on the
   symbol-to-symbol transitions of a linearly modulated stream: Gardner needs a
   midpoint sample between two *consecutive data symbols*, and Mueller-Mueller
   needs the product of a data decision with a neighbouring strobe.  In PPM the
   "decision" is a slot **index**, not an amplitude, and consecutive slots are
   not independent data symbols - they are constrained to contain exactly one
   pulse.  Neither detector is implemented for the slot clock, and this package
   does not pretend they are.
4. **A slot slip is a symbol error, not a loop-level cycle slip.** The receiver
   chooses the pulse slot by index.  If the slot clock drifts past half a slot,
   the maximum-energy slot index changes, the loop re-acquires on the new slot
   and the loop itself shows no slip - but the decoded symbol is wrong.  So the
   quantity a PPM link cares about is not slips per second, it is the symbol
   error floor that slot jitter creates.  :func:`slot_index_error_rate` measures
   that directly from a closed-loop run.
5. **The symbol boundary is not observable from the slot clock.** Slot timing
   locks the loop to the slot grid; which of the ``M`` slot positions begins a
   symbol is a separate ``M``-fold ambiguity that needs frame synchronisation
   (a marker, a guard time, or a dead-time constraint).  Nothing in this module
   resolves it, and a receiver built on this module alone would decode the right
   slot pattern against the wrong frame.

Units
-----
Everything in this module is in **slot periods**.  ``offset``, ``jitter`` and
``delta`` are fractions of a slot; ``B_n`` is cycles per loop **update**, i.e.
per symbol, unless a name says otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .loop import LoopDesign
from .pulses import PulseShape
from .simulate import LoopRun, _batch_standard_errors, pulse_table
from .stream import sample_noise_sigma
from .ted import TedConfig

__all__ = [
    "PpmConfig",
    "PpmLoopRun",
    "PpmSlotCurve",
    "duty_cycle_penalty_db",
    "equivalent_slot_bandwidth",
    "measure_ppm_slot_statistics",
    "ppm_slot_autocovariance",
    "ppm_slot_scurve",
    "run_ppm_slot_loop",
    "slot_index_error_rate",
]


@dataclass(frozen=True)
class PpmConfig:
    """M-ary PPM slot-clock configuration.

    Parameters
    ----------
    order
        ``M``, the number of slots per symbol.  At least 2.
    delta
        Early-late gate half-spacing in **slot** periods, in ``(0, 0.5]``.
    known_slot
        If true, the detector is told which slot carries the pulse; if false it
        picks the maximum-energy slot, which is what a receiver does.  The
        difference between the two is the slot-decision penalty, which this
        module measures.
    """

    order: int = 4
    delta: float = 0.25
    known_slot: bool = False

    def __post_init__(self) -> None:
        if self.order < 2:
            raise ValueError(f"order must be at least 2 slots per symbol, got {self.order}")
        if not 0.0 < self.delta <= 0.5:
            raise ValueError(f"delta must lie in (0, 0.5] slot periods, got {self.delta!r}")

    @property
    def label(self) -> str:
        """Short label for figures and tables."""
        mode = "known slot" if self.known_slot else "max-energy slot"
        return f"{self.order}-PPM slot early-late (d={self.delta:g}, {mode})"


def equivalent_slot_bandwidth(noise_bandwidth_per_update: float, order: int) -> float:
    """Convert a per-update loop bandwidth to cycles per **slot**.

    ``B_slot = B_update / M``.  One update per symbol and ``M`` slots per symbol,
    so a loop that looks fast in update time is ``M`` times slower in slot time.
    """
    if order < 2:
        raise ValueError(f"order must be at least 2, got {order}")
    if not 0.0 < noise_bandwidth_per_update < 0.5:
        raise ValueError(
            f"noise_bandwidth_per_update must lie in (0, 0.5), got {noise_bandwidth_per_update!r}"
        )
    return noise_bandwidth_per_update / order


@dataclass(frozen=True)
class PpmSlotCurve:
    """Exact slot-clock S-curve and the gain read off it."""

    config: PpmConfig
    pulse_name: str
    offsets: np.ndarray
    values: np.ndarray
    gain: float
    gain_central_difference: float
    bias: float
    pattern_count: int

    @property
    def peak_offset(self) -> float:
        """Positive offset where ``|S|`` is largest, slot periods."""
        positive = self.offsets > 0.0
        return float(self.offsets[positive][int(np.argmax(np.abs(self.values[positive])))])

    @property
    def reversal_offset(self) -> float | None:
        """Smallest positive offset where ``S`` returns to zero, or ``None``.

        For a return-to-zero slot pulse narrower than one slot there may be no
        reversal inside half a slot at all: the detector output simply decays as
        the pulse leaves the gates.  That is a real property of the slot clock and
        it is why a PPM slot slip behaves differently from a symbol-timing cycle
        slip; see the module docstring.
        """
        positive = np.where(self.offsets > 0.0)[0]
        sign0 = math.copysign(1.0, self.gain) if self.gain != 0.0 else 1.0
        for a, b in zip(positive[:-1], positive[1:], strict=False):
            va, vb = float(self.values[a]), float(self.values[b])
            if va * sign0 > 0.0 >= vb * sign0:
                if va == vb:
                    return float(self.offsets[b])
                return float(
                    self.offsets[a]
                    + (va / (va - vb)) * (self.offsets[b] - self.offsets[a])
                )
        return None

    def summary(self) -> dict[str, object]:
        """Flat dictionary of the headline quantities."""
        return {
            "configuration": self.config.label,
            "pulse": self.pulse_name,
            "patterns": self.pattern_count,
            "K_d_per_slot": self.gain,
            "K_d_central_difference": self.gain_central_difference,
            "bias_at_zero": self.bias,
            "peak_offset_slots": self.peak_offset,
            "reversal_offset_slots": self.reversal_offset,
        }


def ppm_slot_scurve(
    config: PpmConfig,
    pulse: PulseShape,
    offsets: np.ndarray | None = None,
    *,
    neighbours: int = 1,
    fit_halfwidth: float = 0.02,
) -> PpmSlotCurve:
    """Exact slot-clock S-curve, enumerating every neighbouring pulse position.

    The detector is the squared early-late gate on the slot carrying the pulse,
    ``e = early**2 - late**2``, with the slot known (the ideal case).  The
    expectation is over the pulse positions of the current symbol and of
    ``neighbours`` symbols either side, so it is exact for a slot pulse whose
    support reaches no further than ``neighbours * M`` slots.

    Returns
    -------
    PpmSlotCurve
    """
    if neighbours < 0:
        raise ValueError(f"neighbours must be non-negative, got {neighbours}")
    order = config.order
    if pulse.half_support > neighbours * order + 0.5:
        raise ValueError(
            f"pulse support {pulse.half_support} slots reaches beyond the enumerated "
            f"window of {neighbours} neighbouring symbols at order {order}; increase "
            "neighbours"
        )
    eps = (
        np.linspace(-0.5, 0.5, 201)
        if offsets is None
        else np.asarray(offsets, dtype=float)
    )
    total_symbols = 2 * neighbours + 1
    total_slots = total_symbols * order
    centre_symbol = neighbours

    neighbour_positions = np.array(
        np.meshgrid(*[np.arange(order)] * (total_symbols - 1), indexing="ij")
    ).reshape(total_symbols - 1, -1).T if total_symbols > 1 else np.zeros((1, 0), dtype=int)

    slot_index = np.arange(total_slots, dtype=float)
    accumulator = np.zeros(eps.size)
    count = 0
    other_symbols = [s for s in range(total_symbols) if s != centre_symbol]
    for position in range(order):
        on_slot = centre_symbol * order + position
        times = on_slot + eps[:, None] + np.array([-config.delta, config.delta])[None, :]
        weights = pulse.amplitude(times[..., None] - slot_index)  # (n_eps, 2, n_slots)
        for row in neighbour_positions:
            amplitudes = np.zeros(total_slots)
            amplitudes[on_slot] = 1.0
            for symbol, pos in zip(other_symbols, row, strict=True):
                amplitudes[symbol * order + int(pos)] = 1.0
            gates = weights @ amplitudes  # (n_eps, 2)
            accumulator += gates[:, 0] ** 2 - gates[:, 1] ** 2
            count += 1

    mean = accumulator / count
    fit = np.abs(eps) <= fit_halfwidth
    if fit.sum() < 3:
        fit = np.abs(eps) <= np.sort(np.abs(eps))[2]
    denominator = float(np.sum(eps[fit] ** 2))
    gain = float(np.sum(eps[fit] * mean[fit]) / denominator) if denominator > 0.0 else 0.0
    origin = int(np.argmin(np.abs(eps)))
    central = (
        float((mean[origin + 1] - mean[origin - 1]) / (eps[origin + 1] - eps[origin - 1]))
        if 0 < origin < eps.size - 1
        else gain
    )
    return PpmSlotCurve(
        config=config,
        pulse_name=pulse.name,
        offsets=eps,
        values=mean,
        gain=gain,
        gain_central_difference=central,
        bias=float(mean[origin]),
        pattern_count=count,
    )


def _ppm_positions(rng: np.random.Generator, symbols: int, order: int) -> np.ndarray:
    return rng.integers(0, order, size=symbols)


def _ppm_slot_gates(
    positions: np.ndarray,
    order: int,
    delta: float,
    lookup,
    reach: int,
    offset: float,
    noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Early and late gate samples for every slot of every symbol.

    Returns two arrays of shape ``(symbols, order)``.  Pure helper: the caller
    supplies the noise so the open-loop and closed-loop paths can share it.
    """
    symbols = positions.size
    early = np.zeros((symbols, order))
    late = np.zeros((symbols, order))
    for k in range(symbols):
        for j in range(order):
            slot = k * order + j
            for sign, store in ((-1.0, early), (1.0, late)):
                t = slot + offset + sign * delta
                total = 0.0
                for dk in range(-reach, reach + 1):
                    kk = k + dk
                    if 0 <= kk < symbols:
                        on = kk * order + int(positions[kk])
                        total += lookup(t - on)
                store[k, j] = total
    early += noise[:, :, 0]
    late += noise[:, :, 1]
    return early, late


def _ppm_detector_output(
    early: np.ndarray, late: np.ndarray, positions: np.ndarray, known_slot: bool
) -> tuple[np.ndarray, np.ndarray]:
    """``(e, chosen_slot)``: the slot early-late output and the slot index used."""
    if known_slot:
        chosen = positions.astype(int)
    else:
        chosen = np.argmax(early**2 + late**2, axis=1)
    rows = np.arange(early.shape[0])
    e = early[rows, chosen] ** 2 - late[rows, chosen] ** 2
    return e, chosen


def measure_ppm_slot_statistics(
    config: PpmConfig,
    pulse: PulseShape,
    *,
    sample_snr_db: float,
    offset: float = 0.0,
    symbols: int = 20000,
    seed: int = 20261006,
) -> dict[str, float]:
    """Open-loop mean, variance and slot-decision error of the PPM slot detector.

    Returns a dictionary with ``mean``, ``variance`` (this is ``sigma_n^2`` per
    loop update), ``self_noise_variance``, and ``slot_error_rate`` - the fraction
    of symbols whose maximum-energy slot is not the transmitted one at this
    offset and SNR.
    """
    if symbols < 1000:
        raise ValueError(f"symbols must be at least 1000, got {symbols}")
    order = config.order
    rng = np.random.default_rng(seed)
    positions = _ppm_positions(rng, symbols, order)
    lookup = pulse_table(pulse)
    reach = int(math.ceil(pulse.half_support / order)) + 1
    sigma = sample_noise_sigma(sample_snr_db)
    noise = sigma * rng.standard_normal((symbols, order, 2))
    zero = np.zeros_like(noise)
    early, late = _ppm_slot_gates(positions, order, config.delta, lookup, reach, offset, noise)
    clean_early, clean_late = _ppm_slot_gates(
        positions, order, config.delta, lookup, reach, offset, zero
    )
    out, chosen = _ppm_detector_output(early, late, positions, config.known_slot)
    clean_out, _ = _ppm_detector_output(clean_early, clean_late, positions, config.known_slot)
    return {
        "mean": float(out.mean()),
        "variance": float(out.var()),
        "self_noise_variance": float(clean_out.var()),
        "slot_error_rate": float(np.mean(chosen != positions)),
        "symbols": float(symbols),
    }


def ppm_slot_autocovariance(
    config: PpmConfig,
    pulse: PulseShape,
    *,
    sample_snr_db: float,
    max_lag: int = 8,
    offset: float = 0.0,
    symbols: int = 20000,
    seed: int = 20261006,
) -> np.ndarray:
    """Autocovariance ``R_n[0 .. max_lag]`` of the PPM slot detector output, per update.

    Feeds :func:`slotsync.loop.jitter_variance_coloured` exactly as the symbol-timing
    version does.
    """
    order = config.order
    rng = np.random.default_rng(seed)
    positions = _ppm_positions(rng, symbols, order)
    lookup = pulse_table(pulse)
    reach = int(math.ceil(pulse.half_support / order)) + 1
    sigma = sample_noise_sigma(sample_snr_db)
    noise = sigma * rng.standard_normal((symbols, order, 2))
    early, late = _ppm_slot_gates(positions, order, config.delta, lookup, reach, offset, noise)
    out, _ = _ppm_detector_output(early, late, positions, config.known_slot)
    centred = out - out.mean()
    n = centred.size
    return np.array(
        [float(np.dot(centred[: n - j], centred[j:]) / n) for j in range(max_lag + 1)]
    )


def duty_cycle_penalty_db(
    config: PpmConfig,
    pulse: PulseShape,
    *,
    sample_snr_db: float,
    reference_gain: float,
    symbols: int = 20000,
    seed: int = 20261006,
) -> float:
    """The PPM slot detector's jitter factor ``sigma_n^2 / K_d^2``, in dB.

    Returns ``10 log10( (sigma_n^2 / K_d^2) * reference_gain^2 )``: the zero point
    is a reference detector with unit output variance and gain ``reference_gain``.
    The jitter variance is proportional to ``sigma_n^2 / K_d^2`` at a fixed loop
    bandwidth, so a difference in this number between two configurations is exactly
    the difference in their loop SNR in dB.

    The caller states the reference.  Nothing about the PPM order is assumed: the
    often-quoted ``10 log10(M)`` duty-cycle rule is **not** used here, and
    ``validation/validate_ppm_slot.py`` shows the measured order dependence is
    nothing like it at high SNR and steeper than it at low SNR.
    """
    stats = measure_ppm_slot_statistics(
        config, pulse, sample_snr_db=sample_snr_db, symbols=symbols, seed=seed
    )
    curve = ppm_slot_scurve(config, pulse, offsets=np.linspace(-0.05, 0.05, 41))
    if curve.gain == 0.0:
        raise ValueError("the measured PPM slot gain is zero; no penalty can be formed")
    ratio = (stats["variance"] / curve.gain**2) * reference_gain**2
    return float(10.0 * math.log10(ratio)) if ratio > 0.0 else float("-inf")


@dataclass
class PpmLoopRun(LoopRun):
    """A closed-loop PPM slot run: a :class:`slotsync.simulate.LoopRun` plus slot errors."""

    slot_error_rate: float = 0.0
    order: int = 0

    def ppm_summary(self) -> dict[str, object]:
        """Flat dictionary including the slot-index error rate."""
        base = self.summary()
        base["order"] = self.order
        base["slot_error_rate"] = self.slot_error_rate
        base["B_slot"] = equivalent_slot_bandwidth(self.design.noise_bandwidth, self.order)
        return base


def run_ppm_slot_loop(
    config: PpmConfig,
    pulse: PulseShape,
    design: LoopDesign,
    *,
    n_symbols: int = 40000,
    sample_snr_db: float = 20.0,
    true_offset: float = 0.0,
    initial_error: float = 0.0,
    discard: int | None = None,
    seed: int = 20261006,
    divergence_limit: float = 20.0,
) -> PpmLoopRun:
    """Close the slot-timing loop on an M-ary PPM stream, one update per symbol.

    The loop state is a slot-clock offset in slot periods; the detector is the
    squared early-late gate on the chosen slot.  ``design.noise_bandwidth`` is
    cycles per **update**; :func:`equivalent_slot_bandwidth` gives the slot-rate
    figure.

    Returns
    -------
    PpmLoopRun
        With the measured jitter in slot periods, the slot-index error rate and
        the number of times the loop changed which slot it was locked to.
    """
    if n_symbols < 2000:
        raise ValueError(f"n_symbols must be at least 2000, got {n_symbols}")
    if not design.is_stable:
        raise ValueError("the loop design is unstable; refusing to run")
    if discard is None:
        discard = min(n_symbols // 4, max(1000, int(20.0 / design.noise_bandwidth)))

    order = config.order
    delta = config.delta
    rng = np.random.default_rng(seed)
    positions = _ppm_positions(rng, n_symbols, order).tolist()
    sigma = sample_noise_sigma(sample_snr_db)
    noise = (sigma * rng.standard_normal((n_symbols, order, 2))).tolist()
    lookup = pulse_table(pulse)
    reach = int(math.ceil(pulse.half_support / order)) + 1

    k1 = design.k_proportional
    k2 = design.k_integral
    tau_hat = true_offset + initial_error
    velocity = 0.0
    previous_lock = 0
    slips = 0
    slot_errors = 0
    errors: list[float] = []
    diverged = False

    for k in range(n_symbols):
        best_energy = -1.0
        best_slot = 0
        best_pair = (0.0, 0.0)
        for j in range(order):
            slot = k * order + j
            gates = []
            for sign in (-1.0, 1.0):
                t = slot + tau_hat + sign * delta
                total = 0.0
                for dk in range(-reach, reach + 1):
                    kk = k + dk
                    if 0 <= kk < n_symbols:
                        on = kk * order + positions[kk]
                        total += lookup(t - on - true_offset)
                gates.append(total)
            early = gates[0] + noise[k][j][0]
            late = gates[1] + noise[k][j][1]
            energy = early * early + late * late
            if energy > best_energy:
                best_energy = energy
                best_slot = j
                best_pair = (early, late)
        if config.known_slot:
            chosen = positions[k]
            if chosen != best_slot:
                slot = k * order + chosen
                pair = []
                for sign in (-1.0, 1.0):
                    t = slot + tau_hat + sign * delta
                    total = 0.0
                    for dk in range(-reach, reach + 1):
                        kk = k + dk
                        if 0 <= kk < n_symbols:
                            on = kk * order + positions[kk]
                            total += lookup(t - on - true_offset)
                    pair.append(total)
                best_pair = (
                    pair[0] + noise[k][chosen][0],
                    pair[1] + noise[k][chosen][1],
                )
        else:
            chosen = best_slot
        if chosen != positions[k]:
            slot_errors += 1
        early, late = best_pair
        error_signal = early * early - late * late
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
    run = PpmLoopRun(
        config=TedConfig("early-late", "ook", delta, "square"),
        pulse_name=pulse.name,
        design=design,
        sample_snr_db=float(sample_snr_db),
        n_symbols=int(n_symbols),
        discarded=int(discard),
        true_offset=float(true_offset),
        error=history,
        diverged=diverged,
        order=order,
        slot_error_rate=slot_errors / max(n_symbols, 1),
    )
    if history.size:
        run.mean_error = float(history.mean())
        run.jitter_variance = float(history.var())
        mean_se, var_se = _batch_standard_errors(history, design.noise_bandwidth)
        run.mean_error_standard_error = mean_se
        run.jitter_variance_standard_error = var_se
    run.slip_count = int(slips)
    return run


def slot_index_error_rate(run: PpmLoopRun) -> float:
    """Fraction of symbols whose decoded slot index was wrong, from a closed-loop run.

    This is the quantity a PPM link cares about: slot jitter shows up as a symbol
    error floor, not as a loop cycle-slip rate.  See point 4 of the module
    docstring.
    """
    return run.slot_error_rate
