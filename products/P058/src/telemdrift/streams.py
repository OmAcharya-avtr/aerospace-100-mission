"""Seeded synthetic univariate telemetry streams with declared change types.

All streams are generated from ``numpy.random.Generator`` instances created with
an explicit integer seed, so every figure in this repository regenerates
bit-for-bit from the seed printed beside it.

Units
-----
The channel is dimensionless and standardised: the pre-change distribution is
``N(0, 1)``. A real telemetry channel (for example a transponder output power in
dBm, or a battery bus voltage in V) is standardised by the caller with its own
declared nominal mean and nominal standard deviation before entering this
package. Change magnitudes below are therefore in units of the *pre-change*
standard deviation, which is the only scale on which detector thresholds
transfer between channels.

Model and its assumptions
-------------------------
The baseline process is independent and identically distributed Gaussian noise:

    x_t ~ N(mu_t, sigma_t^2),   x_t independent of x_s for t != s.

Independence is an assumption, not a property of spacecraft telemetry. Real
housekeeping channels are sampled faster than their physical time constants and
are strongly autocorrelated. ``ar1_stationary`` generates a first-order
autoregressive stream for exactly this reason, and
``validation/validate_autocorrelation.py`` measures what autocorrelation does to
a threshold calibrated under the i.i.d. assumption. The measured answer is that
it destroys it.

Validity range
--------------
``sigma > 0``; ``0 <= abs(phi) < 1`` for the AR(1) stream. Change magnitudes are
unrestricted but the detectors in this package are designed for shifts of order
0.25 to 4 pre-change standard deviations, which is the range the validation
scripts cover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CHANGE_TYPES",
    "ChangeSpec",
    "ar1_stationary",
    "change_stream",
    "stationary",
    "transient_spike",
]

#: The four change types the benchmark scores. ``transient`` is the negative
#: control: it is a short excursion the channel recovers from, and a detector
#: that alarms on it has produced a false alarm, not a detection.
CHANGE_TYPES = ("mean_step", "variance_step", "drift_ramp", "transient")


@dataclass(frozen=True)
class ChangeSpec:
    """A declared change injected into a stationary stream.

    Parameters
    ----------
    kind:
        One of :data:`CHANGE_TYPES`.
    magnitude:
        Interpretation depends on ``kind``, all in pre-change standard
        deviations (dimensionless):

        * ``mean_step``   - the size of the step added to the mean.
        * ``variance_step`` - the multiplicative factor applied to the standard
          deviation, so ``magnitude = 2.0`` doubles sigma. Must be positive.
        * ``drift_ramp``  - the slope in standard deviations per sample.
        * ``transient``   - the peak amplitude of the excursion.
    duration:
        Used by ``transient`` only: the number of samples the excursion lasts.
        Ignored by the other kinds.
    """

    kind: str
    magnitude: float
    duration: int = 20

    def __post_init__(self) -> None:
        if self.kind not in CHANGE_TYPES:
            raise ValueError(f"kind must be one of {CHANGE_TYPES}, got {self.kind!r}")
        if self.kind == "variance_step" and self.magnitude <= 0.0:
            raise ValueError("variance_step magnitude is a sigma multiplier and must be > 0")
        if self.kind == "transient" and self.duration < 1:
            raise ValueError("transient duration must be >= 1 sample")


def _rng(seed: int | np.random.Generator) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def stationary(length: int, seed: int | np.random.Generator, sigma: float = 1.0) -> np.ndarray:
    """Return ``length`` i.i.d. samples from ``N(0, sigma^2)``.

    Parameters
    ----------
    length:
        Number of samples, must be >= 1.
    seed:
        Integer seed or an existing ``numpy.random.Generator``.
    sigma:
        Standard deviation (dimensionless, in pre-change sigma units), > 0.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    return _rng(seed).normal(0.0, sigma, size=length)


def ar1_stationary(
    length: int,
    seed: int | np.random.Generator,
    phi: float = 0.8,
    sigma: float = 1.0,
) -> np.ndarray:
    """Return an AR(1) stream with unit marginal variance when ``sigma == 1``.

    ``x_t = phi x_{t-1} + e_t`` with ``e_t ~ N(0, sigma^2 (1 - phi^2))``, started
    from its stationary distribution ``N(0, sigma^2)`` so there is no burn-in
    transient. The marginal distribution is therefore identical to
    :func:`stationary`; only the dependence structure differs, which is what
    makes it the right control for an i.i.d.-calibrated threshold.

    Parameters
    ----------
    phi:
        Lag-one autocorrelation, ``abs(phi) < 1``.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    if not -1.0 < phi < 1.0:
        raise ValueError("phi must satisfy abs(phi) < 1 for a stationary AR(1)")
    g = _rng(seed)
    innov_sd = sigma * np.sqrt(1.0 - phi * phi)
    out = np.empty(length)
    out[0] = g.normal(0.0, sigma)
    noise = g.normal(0.0, innov_sd, size=length - 1)
    for t in range(1, length):
        out[t] = phi * out[t - 1] + noise[t - 1]
    return out


def change_stream(
    pre_length: int,
    post_length: int,
    spec: ChangeSpec,
    seed: int | np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Return ``(stream, change_index)`` for one declared change.

    The first ``pre_length`` samples are ``N(0, 1)``. The change takes effect at
    index ``change_index = pre_length``, meaning sample ``pre_length`` is the
    first sample drawn from the post-change distribution.

    Returns
    -------
    stream:
        Array of length ``pre_length + post_length``.
    change_index:
        Index of the first post-change sample.
    """
    if pre_length < 1 or post_length < 1:
        raise ValueError("pre_length and post_length must both be >= 1")
    g = _rng(seed)
    pre = g.normal(0.0, 1.0, size=pre_length)
    if spec.kind == "mean_step":
        post = g.normal(spec.magnitude, 1.0, size=post_length)
    elif spec.kind == "variance_step":
        post = g.normal(0.0, spec.magnitude, size=post_length)
    elif spec.kind == "drift_ramp":
        ramp = spec.magnitude * np.arange(1, post_length + 1, dtype=float)
        post = g.normal(0.0, 1.0, size=post_length) + ramp
    elif spec.kind == "transient":
        post = g.normal(0.0, 1.0, size=post_length)
        bump = min(spec.duration, post_length)
        post[:bump] += spec.magnitude
    else:  # pragma: no cover - ChangeSpec validates kind
        raise ValueError(f"unhandled kind {spec.kind!r}")
    return np.concatenate([pre, post]), pre_length


def transient_spike(
    pre_length: int,
    post_length: int,
    amplitude: float,
    duration: int,
    seed: int | np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Convenience wrapper: a stream whose only event is a transient excursion.

    A detector alarm anywhere in this stream is a false alarm. The channel
    returns to ``N(0, 1)`` after ``duration`` samples, so there is no change in
    the sense the detectors are scored on.
    """
    return change_stream(
        pre_length,
        post_length,
        ChangeSpec("transient", amplitude, duration),
        seed,
    )
