"""Window sizing against the bandwidth-delay product, in frames and in bytes.

The textbook rule is "size the window to one bandwidth-delay product plus one
frame", which is correct and incomplete: it is the window at which the *window*
stops being the binding constraint, after which throughput is set by the error
rate and more window buys nothing.  This module locates that transition and
reports what is on each side of it, in both frames and bytes, with the assumed
frame size stated in every return value.

Selective repeat has the clean form.  From :mod:`closedform`,

    eta_SR(W) = (1 - p) min(1, W/N)                                        (6)

so the two branches meet at W = N exactly, the slope with respect to W is
(1-p)/N below the knee and zero above it, and the knee position does not depend
on p at all -- only its height does.  That is the useful statement: the window
you need is a property of the link geometry, and the throughput you then get is
a property of the channel.

Go-back-N is different and worse.  It needs ``W >= ceil(N)`` merely to transmit
continuously, and above that its throughput is ``(1-p)/(1-p+Np)``, which for
large N is approximately ``1/(Np)`` -- so on a long link a go-back-N window of
exactly one BDP does not get you a BDP's worth of throughput.  The functions
here report both, so the comparison is visible rather than asserted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .closedform import gbn_throughput, sr_throughput
from .link import LinkParams

__all__ = [
    "WindowSizing",
    "size_window",
    "window_sweep",
    "marginal_gain",
]


@dataclass(frozen=True)
class WindowSizing:
    """Window-sizing answer for one link and one frame error rate.

    Attributes:
        frame_bits: Assumed frame length, bits.  Stated because every frame
            count below is meaningless without it.
        n: Slots per cycle N, dimensionless.
        bdp_bits: Bandwidth-delay product, bits.
        bdp_frames: Bandwidth-delay product, frames.
        knee_frames: Window at which the window stops binding (selective
            repeat), frames.
        knee_bytes: The same window expressed as a send buffer, bytes.
        gbn_min_frames: Smallest go-back-N window permitting continuous
            transmission, frames.
        fer: Frame error probability used for the throughput figures.
        sr_goodput_at_knee: Selective-repeat normalised throughput at the knee.
        gbn_goodput_at_knee: Go-back-N normalised throughput at the same window.
        sr_bps_at_knee: Selective-repeat information throughput, bits per second.
        gbn_bps_at_knee: Go-back-N information throughput, bits per second.
    """

    frame_bits: int
    n: float
    bdp_bits: float
    bdp_frames: float
    knee_frames: int
    knee_bytes: float
    gbn_min_frames: int
    fer: float
    sr_goodput_at_knee: float
    gbn_goodput_at_knee: float
    sr_bps_at_knee: float
    gbn_bps_at_knee: float

    def as_dict(self) -> dict[str, float | int]:
        """Flat dict of the fields, for printing and serialisation."""
        return {
            "frame_bits": self.frame_bits,
            "N": self.n,
            "bdp_bits": self.bdp_bits,
            "bdp_bytes": self.bdp_bits / 8.0,
            "bdp_frames": self.bdp_frames,
            "knee_frames": self.knee_frames,
            "knee_bytes": self.knee_bytes,
            "gbn_min_frames": self.gbn_min_frames,
            "fer": self.fer,
            "sr_goodput_at_knee": self.sr_goodput_at_knee,
            "gbn_goodput_at_knee": self.gbn_goodput_at_knee,
            "sr_bps_at_knee": self.sr_bps_at_knee,
            "gbn_bps_at_knee": self.gbn_bps_at_knee,
        }


def size_window(link: LinkParams, fer: float) -> WindowSizing:
    """Locate the window-sizing knee for ``link`` and report both sides of it.

    Args:
        link: Link geometry.
        fer: Frame error probability, dimensionless in [0, 1).

    Returns:
        A :class:`WindowSizing`.

    Raises:
        ValueError: if ``fer`` is outside [0, 1).
    """
    if not 0.0 <= fer < 1.0:
        raise ValueError(f"fer must be in [0, 1), got {fer}")
    n = link.slots_per_cycle
    knee = max(1, math.ceil(n))
    sr = sr_throughput(fer, n, knee)
    gbn = gbn_throughput(fer, n, knee)
    scale = link.payload / link.frame_time_s
    return WindowSizing(
        frame_bits=link.frame_bits,
        n=n,
        bdp_bits=link.bdp_bits,
        bdp_frames=link.bdp_frames,
        knee_frames=knee,
        knee_bytes=link.window_bytes(knee),
        gbn_min_frames=knee,
        fer=float(fer),
        sr_goodput_at_knee=sr,
        gbn_goodput_at_knee=gbn,
        sr_bps_at_knee=sr * scale,
        gbn_bps_at_knee=gbn * scale,
    )


def window_sweep(
    n: float, fer: float, windows: np.ndarray
) -> dict[str, np.ndarray]:
    """Closed-form throughput against window for both sliding-window protocols.

    Go-back-N is only defined for ``W >= ceil(N)``; below that the returned
    array holds ``nan``, because a number there would be wrong rather than
    approximate.

    Args:
        n: Slots per cycle N.
        fer: Frame error probability.
        windows: Window sizes in frames, 1-D, all >= 1.

    Returns:
        ``{"window": ..., "selective_repeat": ..., "go_back_n": ...}``.
    """
    w = np.asarray(windows, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("windows must be a non-empty 1-D array")
    if np.any(w < 1):
        raise ValueError("windows must all be >= 1 frame")
    sr = np.array([sr_throughput(fer, n, int(x)) for x in w])
    gbn = np.array(
        [
            gbn_throughput(fer, n) if x >= math.ceil(n) else np.nan
            for x in w
        ]
    )
    return {"window": w, "selective_repeat": sr, "go_back_n": gbn}


def marginal_gain(n: float, fer: float, window: int) -> float:
    """Selective-repeat throughput gained per extra frame of window.

    ``(1-p)/N`` below the knee and exactly 0 at or above it.  This is the number
    that says when to stop buying buffer.

    Args:
        n: Slots per cycle N.
        fer: Frame error probability.
        window: Current window, frames.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if not 0.0 <= fer < 1.0:
        raise ValueError(f"fer must be in [0, 1), got {fer}")
    return 0.0 if window >= n else (1.0 - fer) / n
