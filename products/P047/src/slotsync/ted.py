"""Configuration and dispatch layer over the raw detectors.

:mod:`slotsync.detectors` holds the detectors as pure functions with different
signatures, because they genuinely need different samples.  This module wraps
them in one configuration object (:class:`TedConfig`) plus two functions:

* :func:`ted_time_offsets` - which sample instants the detector needs, in
  symbol periods **relative to the loop's current timing estimate**;
* :func:`evaluate_ted` - the detector output given those samples.

Everything downstream - the exact S-curve in :mod:`slotsync.scurve`, the closed
loop in :mod:`slotsync.simulate`, the PPM slot clock in :mod:`slotsync.ppm` -
goes through this layer, so the S-curve that sets the loop gain is produced by
the same code path that the loop then runs.  That is the point: a detector gain
measured by one implementation and used by another is not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detectors import DETECTOR_NAMES, early_late, early_late_dd, gardner, mueller_muller

__all__ = [
    "ALPHABETS",
    "EARLY_LATE_FORMS",
    "TedConfig",
    "evaluate_ted",
    "evaluate_ted_scalar",
    "symbol_variance",
    "ted_decision_offsets",
    "ted_fractional_positions",
    "ted_sample_slots",
    "ted_time_offsets",
]

#: Data alphabets.  ``antipodal`` is ``+-1`` (electrical NRZ); ``ook`` is ``0``/``1``
#: (intensity modulation, the optical case).
ALPHABETS: tuple[str, ...] = ("antipodal", "ook")

#: Early-late arrangements.  ``dd`` multiplies by the data decision, ``square``
#: differences the squared gates (non-data-aided), ``plain`` differences the raw
#: gates and is only meaningful for unipolar data.
EARLY_LATE_FORMS: tuple[str, ...] = ("dd", "square", "plain")


@dataclass(frozen=True)
class TedConfig:
    """What detector to run, on what alphabet, with what gate spacing.

    Parameters
    ----------
    detector
        One of :data:`slotsync.detectors.DETECTOR_NAMES`.
    alphabet
        ``"antipodal"`` or ``"ook"``.
    delta
        Early-late gate half-spacing in symbol periods.  Ignored by the other
        detectors.  Must lie in ``(0, 0.5]``: at ``0.5`` the gates sit on the
        neighbouring symbol boundaries.
    form
        Early-late arrangement; ``"auto"`` picks ``"dd"`` for antipodal data and
        ``"square"`` for OOK.
    """

    detector: str = "early-late"
    alphabet: str = "antipodal"
    delta: float = 0.25
    form: str = "auto"

    def __post_init__(self) -> None:
        if self.detector not in DETECTOR_NAMES:
            raise ValueError(
                f"unknown detector {self.detector!r}; choose one of {list(DETECTOR_NAMES)}"
            )
        if self.alphabet not in ALPHABETS:
            raise ValueError(
                f"unknown alphabet {self.alphabet!r}; choose one of {list(ALPHABETS)}"
            )
        if not 0.0 < self.delta <= 0.5:
            raise ValueError(
                f"delta must lie in (0, 0.5] symbol periods, got {self.delta!r}"
            )
        if self.form not in ("auto", *EARLY_LATE_FORMS):
            raise ValueError(
                f"unknown early-late form {self.form!r}; choose one of "
                f"{['auto', *EARLY_LATE_FORMS]}"
            )

    @property
    def resolved_form(self) -> str:
        """The early-late arrangement actually used, with ``auto`` resolved."""
        if self.form != "auto":
            return self.form
        return "dd" if self.alphabet == "antipodal" else "square"

    @property
    def label(self) -> str:
        """Short human-readable label for figures and tables."""
        if self.detector == "early-late":
            return f"early-late ({self.resolved_form}, d={self.delta:g})"
        return self.detector

    @property
    def samples_per_symbol(self) -> int:
        """Distinct sample instants the detector consumes per symbol."""
        return len(ted_time_offsets(self))


def ted_time_offsets(config: TedConfig) -> tuple[float, ...]:
    """Sample instants the detector needs, in symbol periods.

    Offsets are measured from the loop's current timing estimate for the
    **current** symbol, so ``0.0`` is the current strobe and ``-1.0`` is the
    previous strobe.
    """
    if config.detector == "early-late":
        return (-config.delta, 0.0, config.delta)
    if config.detector == "gardner":
        return (-1.0, -0.5, 0.0)
    return (-1.0, 0.0)


def ted_decision_offsets(config: TedConfig) -> tuple[int, ...]:
    """Symbol indices, relative to the current symbol, whose decisions are needed.

    Empty for detectors that are not decision-directed.
    """
    if config.detector == "mueller-muller":
        return (-1, 0)
    if config.detector == "early-late" and config.resolved_form == "dd":
        return (0,)
    return ()


def evaluate_ted(
    config: TedConfig, samples: np.ndarray, decisions: np.ndarray | None = None
) -> np.ndarray:
    """Detector output from samples taken at :func:`ted_time_offsets`.

    Parameters
    ----------
    config
        Detector configuration.
    samples
        Array whose **last axis** indexes :func:`ted_time_offsets`, in that
        order.  Leading axes are arbitrary and are broadcast through.
    decisions
        Array whose last axis indexes :func:`ted_decision_offsets`, in that
        order.  Required when that tuple is non-empty.

    Returns
    -------
    numpy.ndarray
        Detector output, shape equal to ``samples.shape[:-1]``, positive when
        sampling late.
    """
    s = np.asarray(samples, dtype=float)
    offsets = ted_time_offsets(config)
    if s.shape[-1] != len(offsets):
        raise ValueError(
            f"samples last axis must have length {len(offsets)} for detector "
            f"{config.detector!r}, got {s.shape[-1]}"
        )
    needed = ted_decision_offsets(config)
    if needed:
        if decisions is None:
            raise ValueError(
                f"detector {config.detector!r} (form {config.resolved_form!r}) is "
                f"decision-directed and needs decisions at symbol offsets {list(needed)}"
            )
        d = np.asarray(decisions, dtype=float)
        if d.shape[-1] != len(needed):
            raise ValueError(
                f"decisions last axis must have length {len(needed)}, got {d.shape[-1]}"
            )
    if config.detector == "early-late":
        form = config.resolved_form
        if form == "dd":
            assert decisions is not None  # guarded above
            return early_late_dd(np.asarray(decisions)[..., 0], s[..., 0], s[..., 2])
        return early_late(s[..., 0], s[..., 2], square=(form == "square"))
    if config.detector == "gardner":
        return gardner(s[..., 0], s[..., 1], s[..., 2])
    assert decisions is not None  # mueller-muller is decision-directed
    d = np.asarray(decisions, dtype=float)
    return mueller_muller(d[..., 0], d[..., 1], s[..., 0], s[..., 1])


def symbol_variance(alphabet: str) -> float:
    """Second moment ``E[a^2]`` of the alphabet, dimensionless.

    ``1.0`` for antipodal ``+-1``; ``0.5`` for equiprobable OOK ``0``/``1``.
    Used by the hand-computable Mueller-Mueller known-answer test.
    """
    if alphabet == "antipodal":
        return 1.0
    if alphabet == "ook":
        return 0.5
    raise ValueError(f"unknown alphabet {alphabet!r}; choose one of {list(ALPHABETS)}")


def evaluate_ted_scalar(
    config: TedConfig, samples: tuple[float, ...], decisions: tuple[float, ...] = ()
) -> float:
    """Scalar fast path for :func:`evaluate_ted`, for the sequential closed loop.

    The closed-loop simulation in :mod:`slotsync.simulate` runs one update per
    symbol and cannot be vectorised, so it needs a float-in float-out detector.
    This function implements exactly the same algebra on Python floats.  The
    duplication is a performance compromise and it is covered by a test:
    ``tests/test_ted.py`` asserts bit-for-bit equality with
    :func:`evaluate_ted` over a pseudo-random grid of inputs for every detector
    and form, so the two paths cannot drift apart silently.
    """
    offsets = ted_time_offsets(config)
    if len(samples) != len(offsets):
        raise ValueError(
            f"samples must have {len(offsets)} entries for detector {config.detector!r}, "
            f"got {len(samples)}"
        )
    needed = ted_decision_offsets(config)
    if len(decisions) != len(needed):
        raise ValueError(
            f"decisions must have {len(needed)} entries for detector {config.detector!r} "
            f"(form {config.resolved_form!r}), got {len(decisions)}"
        )
    if config.detector == "early-late":
        form = config.resolved_form
        if form == "dd":
            return decisions[0] * (samples[0] - samples[2])
        if form == "square":
            return samples[0] * samples[0] - samples[2] * samples[2]
        return samples[0] - samples[2]
    if config.detector == "gardner":
        return samples[1] * (samples[2] - samples[0])
    return decisions[1] * samples[0] - decisions[0] * samples[1]


def ted_sample_slots(config: TedConfig) -> tuple[tuple[int, int], ...]:
    """Map each detector tap onto a physical sample instant.

    A detector tap at offset ``t`` symbol periods from the current symbol's
    strobe sits at absolute time ``k + tau_hat + t``, which is the same physical
    instant as tap ``t + 1`` of symbol ``k - 1``.  Consecutive detector updates
    therefore **share samples**, and a simulation that draws fresh noise for each
    tap of each symbol would make the detector noise whiter than it is.

    This function returns, for each tap in :func:`ted_time_offsets` order, a pair
    ``(lag, fraction_index)``: the instant is the one belonging to symbol
    ``k - lag`` at the fractional position
    ``ted_fractional_positions(config)[fraction_index]``.  The simulation draws
    one noise sample per instant and reads it through this map.

    Example
    -------
    Gardner's taps ``(-1.0, -0.5, 0.0)`` give ``((1, 0), (1, 1), (0, 0))``: the
    previous strobe is symbol ``k-1``'s strobe, so every strobe sample is used by
    two consecutive updates and the detector noise is correlated at lag one.  The
    early-late gate at ``delta = 0.25`` shares nothing; at ``delta = 0.5`` it
    shares one sample per update.
    """
    positions = ted_fractional_positions(config)
    slots: list[tuple[int, int]] = []
    for tap in ted_time_offsets(config):
        lag = int(-np.floor(tap + 1.0e-12))
        fraction = tap + lag
        index = int(np.argmin([abs(fraction - p) for p in positions]))
        slots.append((lag, index))
    return tuple(slots)


def ted_fractional_positions(config: TedConfig) -> tuple[float, ...]:
    """Distinct fractional sample positions within a symbol, ascending.

    One noise sample per entry per symbol is all a simulation needs; see
    :func:`ted_sample_slots`.
    """
    seen: list[float] = []
    for tap in ted_time_offsets(config):
        fraction = tap + int(-np.floor(tap + 1.0e-12))
        if not any(abs(fraction - s) < 1.0e-9 for s in seen):
            seen.append(fraction)
    return tuple(sorted(seen))
