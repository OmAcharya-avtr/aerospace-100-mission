"""End-to-end latency accounting: propagation, serialisation and queueing.

Decomposition
-------------
The one-way end-to-end delay of a message over a multi-hop route is

    T = sum over hops of ( T_prop + T_serialise + T_queue + T_process )     (1)

with

* ``T_prop = range / c`` -- propagation, where ``c`` is the vacuum speed of
  light.  For an optical ISL in vacuum this is exact; for the ground leg the
  group delay in the troposphere and ionosphere adds a term that is small
  compared with the slot granularity used here and is NOT modelled.
* ``T_serialise = message_bits / rate`` -- store-and-forward serialisation.
* ``T_queue`` -- queueing delay from the model below.
* ``T_process`` -- a caller-supplied constant per hop [s].

Queueing models
---------------
Both are standard single-server results (Kleinrock 1975, "Queueing Systems,
Volume 1: Theory", Wiley, Ch. 2 and 3; also Bertsekas & Gallager 1992, "Data
Networks", 2nd ed., Prentice Hall, Ch. 3):

* M/M/1 -- Poisson arrivals at rate ``lambda``, exponential service at rate
  ``mu``, utilisation ``rho = lambda / mu < 1``::

      W_mm1 = 1 / (mu - lambda) = (1 / mu) / (1 - rho)                     (2)

  ``W`` is the mean time in system (waiting plus service).
* M/D/1 -- Poisson arrivals, deterministic service time ``1 / mu``.  The
  Pollaczek-Khinchine formula with zero service variance gives::

      W_md1 = (1 / mu) * (1 + rho / (2 (1 - rho)))                         (3)

  M/D/1 is the right default for a fixed-length-frame space link, and gives
  exactly half the M/M/1 *waiting* time at the same utilisation.

Validity: both assume Poisson arrivals, an infinite buffer, a single server
and a stationary regime with ``rho < 1``.  A contact window is none of those
things over its whole duration -- it opens, it closes, and the arrival process
during a dump is usually not Poisson.  The models are used here for *relative*
comparison of route designs and sensitivity to utilisation, not as an absolute
delay prediction, and both functions raise rather than return a number when
``rho >= 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .frames import SPEED_OF_LIGHT_KM_S
from .graph import TeEdge
from .routing import Route

__all__ = [
    "mm1_mean_delay_s",
    "md1_mean_delay_s",
    "HopDelay",
    "RouteDelay",
    "route_delay",
]


def _check_rho(arrival_rate_bps: float, service_rate_bps: float) -> float:
    if arrival_rate_bps < 0.0:
        raise ValueError(f"arrival_rate_bps must be >= 0, got {arrival_rate_bps}")
    if service_rate_bps <= 0.0:
        raise ValueError(f"service_rate_bps must be > 0, got {service_rate_bps}")
    rho = arrival_rate_bps / service_rate_bps
    if rho >= 1.0:
        raise ValueError(
            f"utilisation rho = {rho:.4f} >= 1: the queue is unstable and the "
            f"stationary mean delay is infinite. Reduce the offered load or raise the "
            f"link rate.")
    return rho


def mm1_mean_delay_s(arrival_rate_bps: float, service_rate_bps: float,
                     message_bits: float) -> float:
    """M/M/1 mean time in system [s], Eq. (2), for messages of ``message_bits``.

    Service rate is expressed in bit/s and converted to messages per second by
    ``mu = service_rate_bps / message_bits``; arrivals likewise.
    """
    if message_bits <= 0.0:
        raise ValueError(f"message_bits must be > 0, got {message_bits}")
    rho = _check_rho(arrival_rate_bps, service_rate_bps)
    service_time_s = message_bits / service_rate_bps
    return service_time_s / (1.0 - rho)


def md1_mean_delay_s(arrival_rate_bps: float, service_rate_bps: float,
                     message_bits: float) -> float:
    """M/D/1 mean time in system [s], Eq. (3)."""
    if message_bits <= 0.0:
        raise ValueError(f"message_bits must be > 0, got {message_bits}")
    rho = _check_rho(arrival_rate_bps, service_rate_bps)
    service_time_s = message_bits / service_rate_bps
    return service_time_s * (1.0 + rho / (2.0 * (1.0 - rho)))


@dataclass(frozen=True)
class HopDelay:
    """Per-hop delay breakdown [s]."""

    src: str
    dst: str
    kind: str
    propagation_s: float
    serialisation_s: float
    queueing_s: float
    processing_s: float

    @property
    def total_s(self) -> float:
        """Sum of the four terms [s]."""
        return (self.propagation_s + self.serialisation_s
                + self.queueing_s + self.processing_s)


@dataclass(frozen=True)
class RouteDelay:
    """End-to-end delay of a route, Eq. (1)."""

    hops: tuple[HopDelay, ...]
    model: str

    @property
    def total_s(self) -> float:
        """End-to-end one-way delay [s]."""
        return sum(h.total_s for h in self.hops)

    @property
    def propagation_s(self) -> float:
        """Total propagation delay [s]."""
        return sum(h.propagation_s for h in self.hops)

    @property
    def queueing_s(self) -> float:
        """Total queueing delay [s]."""
        return sum(h.queueing_s for h in self.hops)

    def format_table(self) -> str:
        """Fixed-width breakdown table (str)."""
        head = (f"{'hop':<20}{'kind':<7}{'prop[ms]':>10}{'ser[ms]':>10}"
                f"{'queue[ms]':>11}{'proc[ms]':>10}{'total[ms]':>11}")
        lines = [f"End-to-end delay ({self.model})", "=" * len(head), head, "-" * len(head)]
        for h in self.hops:
            lines.append(f"{h.src + '->' + h.dst:<20}{h.kind:<7}"
                         f"{h.propagation_s * 1e3:>10.3f}{h.serialisation_s * 1e3:>10.3f}"
                         f"{h.queueing_s * 1e3:>11.3f}{h.processing_s * 1e3:>10.3f}"
                         f"{h.total_s * 1e3:>11.3f}")
        lines.append("-" * len(head))
        lines.append(f"{'TOTAL':<20}{'':<7}{self.propagation_s * 1e3:>10.3f}"
                     f"{'':>10}{self.queueing_s * 1e3:>11.3f}{'':>10}"
                     f"{self.total_s * 1e3:>11.3f}")
        return "\n".join(lines)


def route_delay(route: Route, message_bits: float,
                link_rate_bps: dict[tuple[str, str], float] | float,
                link_range_km: dict[tuple[str, str], float] | float,
                offered_load_fraction: float = 0.0,
                processing_s: float = 0.0,
                model: str = "md1") -> RouteDelay:
    """End-to-end delay of ``route``, Eq. (1).

    Parameters
    ----------
    route : a :class:`~constellink.routing.Route`.
    message_bits : message size [bits], > 0.
    link_rate_bps : per-link rate [bit/s], either a scalar applied to every
        transmit hop or a dict keyed by the link tuple as stored on the edge.
    link_range_km : per-link range [km], scalar or dict, same convention.
    offered_load_fraction : utilisation ``rho`` offered to each transmit hop,
        in ``[0, 1)``.  Zero means an empty queue, in which case the queueing
        term reduces to one service time (M/D/1) and the delay is the
        store-and-forward floor.
    processing_s : per-hop processing delay [s], >= 0.
    model : ``"md1"`` or ``"mm1"``.

    Hold edges contribute only their slot duration as a queueing term: a
    message waiting in a buffer for a future contact is delayed by the wait,
    not by a service process.
    """
    if message_bits <= 0.0:
        raise ValueError(f"message_bits must be > 0, got {message_bits}")
    if not 0.0 <= offered_load_fraction < 1.0:
        raise ValueError(
            f"offered_load_fraction must be in [0, 1), got {offered_load_fraction}")
    if processing_s < 0.0:
        raise ValueError(f"processing_s must be >= 0, got {processing_s}")
    if model not in ("md1", "mm1"):
        raise ValueError(f"model must be 'md1' or 'mm1', got {model!r}")

    def rate_of(e: TeEdge) -> float:
        if isinstance(link_rate_bps, dict):
            if e.link is None or e.link not in link_rate_bps:
                raise KeyError(f"no rate given for link {e.link}")
            return float(link_rate_bps[e.link])
        return float(link_rate_bps)

    def range_of(e: TeEdge) -> float:
        if isinstance(link_range_km, dict):
            if e.link is None or e.link not in link_range_km:
                raise KeyError(f"no range given for link {e.link}")
            return float(link_range_km[e.link])
        return float(link_range_km)

    delay_fn = md1_mean_delay_s if model == "md1" else mm1_mean_delay_s
    hops: list[HopDelay] = []
    for e in route.hops:
        if e.kind == "hold":
            hops.append(HopDelay(src=e.src[0], dst=e.dst[0], kind="hold",
                                 propagation_s=0.0, serialisation_s=0.0,
                                 queueing_s=e.cost_s, processing_s=0.0))
            continue
        rate = rate_of(e)
        if rate <= 0.0:
            raise ValueError(f"link rate must be > 0 for hop {e.link}, got {rate}")
        rng = range_of(e)
        t_prop = rng / SPEED_OF_LIGHT_KM_S
        t_ser = message_bits / rate
        total_in_system = delay_fn(offered_load_fraction * rate, rate, message_bits)
        t_queue = max(0.0, total_in_system - t_ser)
        hops.append(HopDelay(src=e.src[0], dst=e.dst[0], kind="tx",
                             propagation_s=t_prop, serialisation_s=t_ser,
                             queueing_s=t_queue, processing_s=processing_s))
    return RouteDelay(hops=tuple(hops), model=model)
