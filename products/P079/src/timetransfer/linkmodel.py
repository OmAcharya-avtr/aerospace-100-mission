"""The seeded link model: a delay floor per direction, queueing, and outages.

The delay decomposition
-----------------------
Each direction of the link has a delay

    d = f + q                                                              [s]

with ``f`` a fixed floor (propagation plus deterministic serialisation and
device latency) and ``q >= 0`` a variable queueing component. The **asymmetry**
of one exchange is

    A = d_ab - d_ba = (f_ab - f_ba) + (q_ab - q_ba)                        [s]

and it splits into a static part and a variable part. That split is the whole
reason this package exists in the form it does:

* The **static** part ``f_ab - f_ba`` is not observable from timestamps. The
  four timestamps of a two-way exchange give two independent combinations,
  ``d_ab + theta`` and ``d_ba - theta``; their sum is the round trip and their
  difference is ``2*theta + (f_ab - f_ba) + (q_ab - q_ba)``. A static asymmetry
  and a constant offset enter identically, so no estimator -- learned or not --
  can separate them without external information. This is the standard result
  behind the symmetric-path assumption in IEEE Std 1588-2019 (section 7.4.2,
  mean path delay) and in NTP (Mills, "Internet time synchronization: the
  Network Time Protocol", *IEEE Transactions on Communications*
  39(10):1482-1493, 1991, section III).
* The **variable** part ``q_ab - q_ba`` *is* partly observable, because the
  per-exchange queueing excursion above each direction's own floor shows up in
  the dwell times. That is what the learned corrector in
  :mod:`timetransfer.asym` is given to work with, and the honest statement of
  its ceiling.

Queueing distribution
---------------------
``q`` is drawn from a gamma distribution with shape ``k`` and scale
``theta_s``, so its mean is ``k * theta_s`` and it is non-negative and
right-skewed, which is the qualitative shape of measured packet-delay
distributions. The gamma is a modelling choice, not a measurement: it is
chosen because it is non-negative, has a tunable tail, and reduces to the
exponential at ``k = 1`` (the M/M/1 waiting time). No parameter in this module
is fitted to a real network trace, and nothing here should be read as a claim
about one.

Outages
-------
Two independent mechanisms, because they behave differently:

* **Bernoulli loss** -- each exchange is independently lost with probability
  ``loss_probability``. Models isolated drops.
* **Scheduled outage windows** -- a list of ``(start_s, end_s)`` intervals in
  which every exchange is lost. Models an antenna handover or a keyhole, and
  is the case the recovery procedure in ``docs/DEPLOYMENT.md`` is written for,
  because a burst of consecutive losses is what breaks a running estimate.

Validity range
--------------
A floor per direction that is constant over the acquisition, a stationary
queueing distribution, and losses that are independent of delay. A congested
link violates all three at once; this model cannot represent delay-correlated
loss, and ``validation/validate_asymmetry_benchmark.py`` therefore reports a
result about this generator and not about a network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .errors import ConfigurationError

__all__ = ["LinkModel", "LinkSpec"]


@dataclass(frozen=True)
class LinkSpec:
    """Declared parameters of a simulated two-way link.

    Attributes
    ----------
    floor_ab_s, floor_ba_s:
        Fixed delay floor in each direction [s]. Their difference is the static
        asymmetry, which is unobservable from timestamps alone.
    jitter_shape_ab, jitter_shape_ba:
        Gamma shape ``k`` of the queueing component in each direction,
        dimensionless, must be > 0.
    jitter_scale_ab_s, jitter_scale_ba_s:
        Gamma scale of the queueing component in each direction [s], >= 0. The
        mean queueing delay is ``shape * scale``.
    turnaround_s:
        Time the remote node holds the request before replying [s], as seen on
        the remote clock. Enters ``t3 - t2`` and cancels from the offset
        estimator exactly.
    loss_probability:
        Probability an individual exchange is lost, in [0, 1).
    outage_windows:
        Intervals ``(start_s, end_s)`` of local reference time in which every
        exchange is lost.
    """

    floor_ab_s: float = 5.0e-3
    floor_ba_s: float = 5.0e-3
    jitter_shape_ab: float = 2.0
    jitter_shape_ba: float = 2.0
    jitter_scale_ab_s: float = 2.0e-4
    jitter_scale_ba_s: float = 2.0e-4
    turnaround_s: float = 1.0e-4
    loss_probability: float = 0.0
    outage_windows: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        for name in ("floor_ab_s", "floor_ba_s", "turnaround_s"):
            value = float(getattr(self, name))
            if value < 0.0 or not np.isfinite(value):
                raise ConfigurationError(f"{name} must be finite and >= 0, got {value!r}")
        for name in ("jitter_shape_ab", "jitter_shape_ba"):
            value = float(getattr(self, name))
            if not (value > 0.0) or not np.isfinite(value):
                raise ConfigurationError(f"{name} must be finite and > 0, got {value!r}")
        for name in ("jitter_scale_ab_s", "jitter_scale_ba_s"):
            value = float(getattr(self, name))
            if value < 0.0 or not np.isfinite(value):
                raise ConfigurationError(f"{name} must be finite and >= 0, got {value!r}")
        p = float(self.loss_probability)
        if not (0.0 <= p < 1.0):
            raise ConfigurationError(f"loss_probability must be in [0, 1), got {p!r}")
        for window in self.outage_windows:
            if len(window) != 2:
                raise ConfigurationError(f"outage window must be a pair, got {window!r}")
            start, end = float(window[0]), float(window[1])
            if not (end > start):
                raise ConfigurationError(f"outage window must satisfy end > start, got {window!r}")

    @property
    def static_asymmetry_s(self) -> float:
        """``floor_ab_s - floor_ba_s`` [s]: the unobservable part of ``A``."""
        return float(self.floor_ab_s) - float(self.floor_ba_s)

    @property
    def static_bias_s(self) -> float:
        """``static_asymmetry_s / 2`` [s]: the bias it puts on the symmetric estimator.

        Derived in closed form in :func:`timetransfer.twoway.asymmetry_bias`
        and verified in ``validation/validate_symmetric_bias.py``.
        """
        return 0.5 * self.static_asymmetry_s

    @property
    def mean_queue_asymmetry_s(self) -> float:
        """Expected ``q_ab - q_ba`` [s] from the declared gamma parameters."""
        return float(self.jitter_shape_ab) * float(self.jitter_scale_ab_s) - float(
            self.jitter_shape_ba
        ) * float(self.jitter_scale_ba_s)

    def in_outage(self, t_s: float) -> bool:
        """Whether local reference time ``t_s`` [s] falls in a scheduled outage."""
        return any(float(a) <= float(t_s) < float(b) for a, b in self.outage_windows)


@dataclass
class LinkModel:
    """A seeded realisation of a :class:`LinkSpec`.

    Parameters
    ----------
    spec:
        Declared link parameters.
    seed:
        Seed for the ``numpy`` PCG64 generator driving queueing and loss.
    """

    spec: LinkSpec
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)
    draws: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._rng = np.random.Generator(np.random.PCG64(int(self.seed)))

    def draw(self, t_s: float) -> tuple[float, float, bool]:
        """Draw one exchange's delays at local reference time ``t_s``.

        Randomness is consumed in a fixed order -- loss, forward queue, reverse
        queue -- on **every** call including a lost one, so the realised
        sequence depends only on the seed and the number of calls. That is what
        makes a dropped exchange not shift the rest of the realisation.

        Returns
        -------
        tuple
            ``(d_ab_s, d_ba_s, lost)``. When ``lost`` is ``True`` the delays
            are still returned, so a caller logging a loss can record what the
            exchange would have measured; they must not be used as data.
        """
        u = float(self._rng.random())
        q_ab = float(
            self._rng.gamma(float(self.spec.jitter_shape_ab), float(self.spec.jitter_scale_ab_s))
        )
        q_ba = float(
            self._rng.gamma(float(self.spec.jitter_shape_ba), float(self.spec.jitter_scale_ba_s))
        )
        self.draws += 1
        lost = bool(u < float(self.spec.loss_probability)) or self.spec.in_outage(t_s)
        return float(self.spec.floor_ab_s) + q_ab, float(self.spec.floor_ba_s) + q_ba, lost

    def snapshot(self) -> dict[str, object]:
        """Complete state, including the generator's bit state, for recovery."""
        return {"draws": self.draws, "bit_generator_state": self._rng.bit_generator.state}

    def restore(self, snapshot: dict[str, object]) -> None:
        """Put the model back into a state captured by :meth:`snapshot`."""
        self.draws = int(snapshot["draws"])  # type: ignore[arg-type]
        self._rng.bit_generator.state = snapshot["bit_generator_state"]
