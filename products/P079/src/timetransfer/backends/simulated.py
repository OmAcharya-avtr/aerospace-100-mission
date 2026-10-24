"""The simulation engine and the simulated backend.

Where the simulation actually lives
-----------------------------------
:class:`ExchangeEngine` is the single implementation of the simulated physics:
a :class:`~timetransfer.clockmodel.ClockModel`, a
:class:`~timetransfer.linkmodel.LinkModel`, a timestamping-noise generator and
a quantiser. :class:`SimulatedBackend` is a thin HAL wrapper around it, and
:class:`~timetransfer.backends.stubs.LoopbackDriver` is a thin *driver* wrapper
around the same class. Parity between the simulated and the device path is
therefore a property of the code layout, not of a coincidence, and the parity
test checks that the layout has not been broken.

Modelling decisions, stated rather than buried
----------------------------------------------
* **Three independent generators.** Clock noise, link queueing and
  timestamping noise are driven by three PCG64 generators seeded
  ``seed``, ``seed + 1`` and ``seed + 2``. Separating them means changing the
  jitter does not reshuffle the clock realisation, which is what makes the
  sensitivity studies in ``validation/`` interpretable.
* **Randomness is consumed on every attempt, including a lost one.** The link
  draw, the clock advance and the four timestamping draws all happen before
  loss is acted on, so a dropped exchange does not shift the realisation of
  the ones after it.
* **The clock's stochastic phase is evaluated once per exchange**, at the
  remote node's reception epoch, and held across the turnaround. At the default
  ``q1 = 1e-20 s^2/s`` and a turnaround of ``1e-4 s`` the neglected white-FM
  increment has standard deviation ``sqrt(q1*dt) = 1e-12 s``, three orders
  below the 1 ns timestamp quantisation. The *deterministic* polynomial is
  evaluated at each timestamp's own epoch, so skew and drift are exact.
* **Quantisation is applied last**, by rounding each timestamp to an integer
  multiple of ``resolution_ns``. This is the only place a timestamp becomes
  inexact, and it is the term :func:`timetransfer.timebase.offset_quantisation_uncertainty`
  accounts for.

Units
-----
All model times are seconds; all emitted timestamps are integer nanoseconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..clockmodel import ClockModel, ClockSpec
from ..errors import ConfigurationError
from ..exchange import Exchange
from ..hal import BackendInfo, ExchangeOutcome, ExchangeTruth
from ..linkmodel import LinkModel, LinkSpec
from ..timebase import NS_PER_S, VirtualTimebase

__all__ = ["ExchangeEngine", "SimulatedBackend", "SimulationSpec"]


@dataclass(frozen=True)
class SimulationSpec:
    """Everything needed to make a reproducible acquisition.

    Attributes
    ----------
    clock:
        Remote clock parameters.
    link:
        Link delay parameters.
    period_s:
        Nominal interval between exchange attempts [s], must be > 0.
    start_s:
        Local reference time of the first attempt [s], must be >= 0.
    timestamp_noise_s:
        Standard deviation of the independent white-phase timestamping noise
        applied to each of the four timestamps [s], >= 0. This is the
        white-PM term; it is measurement noise, not clock state.
    resolution_ns:
        Timestamp quantisation step [ns], >= 1.
    seed:
        Base seed; three generators are derived from it.
    """

    clock: ClockSpec = field(default_factory=ClockSpec)
    link: LinkSpec = field(default_factory=LinkSpec)
    period_s: float = 1.0
    start_s: float = 0.0
    timestamp_noise_s: float = 0.0
    resolution_ns: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if not (float(self.period_s) > 0.0) or not np.isfinite(self.period_s):
            raise ConfigurationError(f"period_s must be finite and > 0, got {self.period_s!r}")
        if float(self.start_s) < 0.0:
            raise ConfigurationError(f"start_s must be >= 0, got {self.start_s!r}")
        if float(self.timestamp_noise_s) < 0.0:
            raise ConfigurationError(
                f"timestamp_noise_s must be >= 0, got {self.timestamp_noise_s!r}"
            )
        if int(self.resolution_ns) < 1:
            raise ConfigurationError(f"resolution_ns must be >= 1, got {self.resolution_ns!r}")


class ExchangeEngine:
    """The one implementation of the simulated exchange physics.

    Parameters
    ----------
    spec:
        Simulation parameters.
    """

    def __init__(self, spec: SimulationSpec) -> None:
        self.spec = spec
        self._clock = ClockModel(spec.clock, seed=int(spec.seed))
        self._link = LinkModel(spec.link, seed=int(spec.seed) + 1)
        self._ts_rng = np.random.Generator(np.random.PCG64(int(spec.seed) + 2))
        self.attempts = 0

    @property
    def clock(self) -> ClockModel:
        """The remote clock realisation."""
        return self._clock

    @property
    def link(self) -> LinkModel:
        """The link realisation."""
        return self._link

    def _quantise_ns(self, t_s: float) -> int:
        """Round a time in seconds to an integer multiple of the resolution [ns]."""
        res = int(self.spec.resolution_ns)
        ticks = round(float(t_s) * NS_PER_S / res)
        return int(ticks) * res

    def step(self, seq: int) -> ExchangeOutcome:
        """Attempt exchange ``seq`` and return its outcome.

        Randomness is consumed in a fixed order on every call: link draw,
        clock advance, four timestamping draws. Loss is acted on only after
        all of it, so the realisation does not depend on which attempts
        survived.
        """
        spec = self.spec
        t_send = float(spec.start_s) + int(seq) * float(spec.period_s)
        d_ab, d_ba, lost = self._link.draw(t_send)
        t_b_recv = t_send + d_ab
        noise_offset_s = self._clock.offset_at(t_b_recv)
        held_noise_s = noise_offset_s - float(spec.clock.deterministic_offset_s(t_b_recv))
        t_b_send = t_b_recv + float(spec.link.turnaround_s)
        t_a_recv = t_b_send + d_ba
        sigma = float(spec.timestamp_noise_s)
        draws = self._ts_rng.normal(0.0, 1.0, size=4) * sigma if sigma > 0.0 else np.zeros(4)
        self.attempts += 1
        if lost:
            return ExchangeOutcome(seq=int(seq), exchange=None, lost=True, truth=None)
        t1 = self._quantise_ns(t_send + float(draws[0]))
        t2 = self._quantise_ns(
            t_b_recv + float(spec.clock.deterministic_offset_s(t_b_recv)) + held_noise_s
            + float(draws[1])
        )
        t3 = self._quantise_ns(
            t_b_send + float(spec.clock.deterministic_offset_s(t_b_send)) + held_noise_s
            + float(draws[2])
        )
        t4 = self._quantise_ns(t_a_recv + float(draws[3]))
        t_mid = 0.5 * (t_send + t_a_recv)
        truth = ExchangeTruth(
            offset_s=float(spec.clock.deterministic_offset_s(t_mid)) + held_noise_s,
            asymmetry_s=float(d_ab - d_ba),
            forward_delay_s=float(d_ab),
            reverse_delay_s=float(d_ba),
        )
        return ExchangeOutcome(
            seq=int(seq),
            exchange=Exchange(seq=int(seq), t1_ns=t1, t2_ns=t2, t3_ns=t3, t4_ns=t4),
            lost=False,
            truth=truth,
        )

    def snapshot(self) -> dict[str, object]:
        """Complete engine state, including all three generator bit states."""
        return {
            "attempts": self.attempts,
            "clock": self._clock.snapshot(),
            "link": self._link.snapshot(),
            "ts_rng": self._ts_rng.bit_generator.state,
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        """Put the engine back into a state captured by :meth:`snapshot`."""
        self.attempts = int(snapshot["attempts"])  # type: ignore[arg-type]
        self._clock.restore(snapshot["clock"])  # type: ignore[arg-type]
        self._link.restore(snapshot["link"])  # type: ignore[arg-type]
        self._ts_rng.bit_generator.state = snapshot["ts_rng"]


class SimulatedBackend:
    """HAL backend over an :class:`ExchangeEngine`. Never hardware.

    Parameters
    ----------
    spec:
        Simulation parameters.
    name:
        Label recorded in :attr:`~timetransfer.hal.BackendInfo.name`.
    """

    def __init__(self, spec: SimulationSpec, *, name: str = "simulated-clock") -> None:
        self._engine = ExchangeEngine(spec)
        self._timebase = VirtualTimebase(resolution_ns=int(spec.resolution_ns))
        self._open = False
        self._name = str(name)

    @property
    def engine(self) -> ExchangeEngine:
        """The engine; exposed so a validation script can read the truth."""
        return self._engine

    @property
    def info(self) -> BackendInfo:
        spec = self._engine.spec
        return BackendInfo(
            kind="simulated",
            name=self._name,
            driver="",
            detail={
                "period_s": f"{spec.period_s:.9g}",
                "resolution_ns": str(int(spec.resolution_ns)),
                "seed": str(int(spec.seed)),
                "static_asymmetry_s": f"{spec.link.static_asymmetry_s:.9g}",
                "skew_s_per_s": f"{spec.clock.skew_s_per_s:.9g}",
            },
        )

    @property
    def timebase(self) -> VirtualTimebase:
        return self._timebase

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> None:
        """Mark the backend open. A simulated backend is always available."""
        self._open = True

    def close(self) -> None:
        """Mark the backend closed. Idempotent."""
        self._open = False

    def next_exchange(self, seq: int) -> ExchangeOutcome:
        """Attempt exchange ``seq``, advancing the virtual timebase to its end."""
        outcome = self._engine.step(seq)
        spec = self._engine.spec
        target_ns = round((float(spec.start_s) + (int(seq) + 1) * float(spec.period_s)) * NS_PER_S)
        delta = target_ns - self._timebase.now_ns()
        if delta > 0:
            self._timebase.advance_ns(delta)
        return outcome

    def snapshot(self) -> dict[str, object]:
        return {
            "engine": self._engine.snapshot(),
            "timebase_ns": self._timebase.now_ns(),
            "open": self._open,
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        self._engine.restore(snapshot["engine"])  # type: ignore[arg-type]
        self._timebase.set_ns(int(snapshot["timebase_ns"]))  # type: ignore[arg-type]
        self._open = bool(snapshot["open"])
