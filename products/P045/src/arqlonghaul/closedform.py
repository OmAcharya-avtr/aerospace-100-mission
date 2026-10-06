"""Closed-form ARQ throughput expressions, and the regime where each is exact.

Source for all four expressions: S. Lin and D. J. Costello, Jr., *Error Control
Coding*, 2nd ed., Prentice Hall, 2004, chapter on ARQ error control; the same
results appear in A. S. Tanenbaum, *Computer Networks*, as the sliding-window
efficiency derivations.  They are reproduced here with their derivations in the
docstrings so that the assumptions are visible, because the assumptions are
exactly what the bursty-channel part of this package goes on to break.

Throughput is reported as *normalised throughput* (equivalently goodput
efficiency) eta: the long-run number of frames delivered in order per frame
slot of elapsed time.  eta = 1 means every slot carried a frame that was
accepted first time.  Multiply by ``payload_bits / frame_time_s`` for
information bits per second.

Common symbols:

    p   frame error probability, dimensionless in [0, 1)
    N   slots per stop-and-wait cycle, = 1 + RTT/T_f (see link.py eq. 1)
    W   send window, frames

Validity, stated once and then relied on:

    * frame errors are statistically independent from frame to frame,
    * the reverse channel is error free and its delay is inside N,
    * frames are of constant length and the channel is always busy when the
      protocol permits a transmission,
    * selective repeat additionally assumes buffers large enough that the send
      window never blocks on an outstanding frame (the "infinite buffer" case).

``protocols.py`` simulates the state machines without any of these assumptions.
``validation/validate_closed_form.py`` measures the agreement in the regime
where they do hold; that measured agreement is the licence to believe the
simulator where they do not.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "slots_per_cycle",
    "sw_throughput",
    "gbn_throughput",
    "sr_throughput",
    "expected_transmissions",
    "window_knee",
    "throughput",
]

_PROTOCOLS = ("stop_and_wait", "go_back_n", "selective_repeat")


def _check_p(p: float) -> float:
    if not 0.0 <= p < 1.0:
        raise ValueError(f"frame error probability must be in [0, 1), got {p}")
    return float(p)


def _check_n(n: float) -> float:
    if not n >= 1.0:
        raise ValueError(f"N must be >= 1 slot, got {n}")
    return float(n)


def _check_w(w: int) -> int:
    if int(w) != w or w < 1:
        raise ValueError(f"window must be a positive integer number of frames, got {w}")
    return int(w)


def slots_per_cycle(rtt_s: float, frame_time_s: float) -> float:
    """N = 1 + RTT/T_f, dimensionless.

    Args:
        rtt_s: Round-trip feedback latency, seconds.
        frame_time_s: Frame transmission time, seconds.
    """
    if frame_time_s <= 0:
        raise ValueError(f"frame_time_s must be > 0, got {frame_time_s}")
    if rtt_s < 0:
        raise ValueError(f"rtt_s must be >= 0, got {rtt_s}")
    return 1.0 + rtt_s / frame_time_s


def expected_transmissions(p: float) -> float:
    """Mean transmissions per frame under independent errors, = 1/(1-p).

    The geometric mean of the number of attempts.  This is the quantity that
    correlation breaks: on a bursty channel the attempts of one frame are not
    independent draws, and the long-run mean can be larger or smaller depending
    on how the retransmission spacing N compares with the burst length.
    """
    return 1.0 / (1.0 - _check_p(p))


def sw_throughput(p: float, n: float) -> float:
    """Stop-and-wait normalised throughput, eta = (1-p)/N.

    Derivation: one attempt occupies exactly N slots (one slot of transmission
    then N-1 slots of waiting for the acknowledgement).  A fraction (1-p) of
    attempts deliver a frame.  Hence (1-p) frames per N slots.

    Note the structural consequence used later: eta is *linear* in p, so the
    stationary marginal frame error rate is a sufficient statistic and
    stop-and-wait throughput is insensitive to error correlation.  That is a
    prediction, and ``validation/validate_burst_channel.py`` tests it.
    """
    return (1.0 - _check_p(p)) / _check_n(n)


def gbn_throughput(p: float, n: float, window: int | None = None) -> float:
    """Go-back-N normalised throughput, eta = (1-p)/(1 - p + N p).

    Derivation: let E be the expected number of slots consumed per frame
    accepted.  With probability (1-p) the frame is accepted after its single
    slot; with probability p the whole window in flight is discarded, costing N
    slots, and the frame is attempted again:

        E = (1-p)(1) + p(N + E)  =>  E = 1 + Np/(1-p)

    and eta = 1/E = (1-p)/(1 - p + N p).

    Args:
        p: Frame error probability.
        n: Slots per cycle N.
        window: Send window in frames.  Go-back-N only transmits continuously
            when ``window >= N``; for a smaller window the sender stalls and
            this expression does not apply.  If ``window`` is given and is less
            than ``ceil(N)``, a :class:`ValueError` is raised rather than a
            number that is quietly wrong.

    Raises:
        ValueError: if ``window`` is given and is smaller than ``ceil(N)``.
    """
    p = _check_p(p)
    n = _check_n(n)
    if window is not None:
        window = _check_w(window)
        if window < math.ceil(n):
            raise ValueError(
                f"go-back-N closed form requires window >= ceil(N) = {math.ceil(n)} "
                f"for continuous transmission; got window={window}. "
                "Simulate it with protocols.simulate_go_back_n instead."
            )
    return (1.0 - p) / (1.0 - p + n * p)


def sr_throughput(p: float, n: float, window: int | None = None) -> float:
    """Selective-repeat normalised throughput.

    Two regimes, both exact under the stated assumptions:

        window >= N :  eta = 1 - p            (error limited)
        window <  N :  eta = (1-p) W / N      (window limited)

    and together ``eta = (1-p) min(1, W/N)``.

    Derivation of the window-limited branch: *assume* the sender keeps W frames
    outstanding at all times and learns their fate N slots later, so it makes W
    transmissions per N slots whatever they are (new frames or retransmissions),
    and a fraction (1-p) of transmissions deliver.  Stop-and-wait is the W = 1
    case, and ``sr_throughput(p, N, 1) == sw_throughput(p, N)`` exactly.

    **Both branches are upper bounds for 1 < W, and the window-limited branch is
    exact only at W = 1.**  That assumption -- a window that never blocks -- is
    false for real selective repeat, which cannot advance its window base past
    an unacknowledged frame: a frame being retransmitted holds its sequence
    number, the window fills with unacknowledged frames behind it, and the
    sender idles.  Measured in ``validation/validate_closed_form.py``: this
    expression **overstates the simulated goodput by 145.1 per cent at
    N = 200, p = 0.2, W = N/2** and by 98.5 per cent at N = 60, p = 0.2,
    W = N/2.  Against the other branch, the simulated protocol falls
    **56.6 per cent short of 1-p at W = N**, the textbook "one bandwidth-delay
    product plus one frame" window, at p = 0.2, and comes within 1 per cent of
    it only at about W = 6N.
    Treat this function as the no-blocking ideal and
    :func:`arqlonghaul.protocols.simulate_selective_repeat` as the answer.
    """
    p = _check_p(p)
    n = _check_n(n)
    if window is None:
        return 1.0 - p
    window = _check_w(window)
    return (1.0 - p) * min(1.0, window / n)


def throughput(protocol: str, p: float, n: float, window: int | None = None) -> float:
    """Dispatch to the closed form for ``protocol``.

    Args:
        protocol: One of ``"stop_and_wait"``, ``"go_back_n"``,
            ``"selective_repeat"``.
        p: Frame error probability.
        n: Slots per cycle N.
        window: Send window in frames, where the expression takes one.

    Raises:
        ValueError: on an unknown protocol name.
    """
    if protocol == "stop_and_wait":
        return sw_throughput(p, n)
    if protocol == "go_back_n":
        return gbn_throughput(p, n, window)
    if protocol == "selective_repeat":
        return sr_throughput(p, n, window)
    raise ValueError(f"unknown protocol {protocol!r}; expected one of {_PROTOCOLS}")


def window_knee(n: float) -> float:
    """Window, in frames, at which the window stops being the binding constraint.

    For selective repeat the two branches of :func:`sr_throughput` meet at
    W = N exactly, so the knee is at N frames and the sensitivity of throughput
    to window size drops to zero above it.  Returned as a float because N is
    not in general an integer.
    """
    return _check_n(n)


def sr_window_sweep(p: float, n: float, windows: np.ndarray) -> np.ndarray:
    """Vector of selective-repeat throughputs over ``windows``, dimensionless."""
    p = _check_p(p)
    n = _check_n(n)
    w = np.asarray(windows, dtype=float)
    if np.any(w < 1):
        raise ValueError("windows must all be >= 1 frame")
    return (1.0 - p) * np.minimum(1.0, w / n)
