"""Optional TLE entry point: get a state, not a covariance.

A TLE gives a state. It does not give a covariance, and a collision probability
computed from a covariance that did not come from orbit determination is a
number about an assumption, not about a conjunction. This module therefore
propagates TLEs to a state and stops; the caller must supply the covariance,
and this package will not invent one.

Propagation is delegated to the ``sgp4`` package (Vallado's reference C++
implementation behind a Python wrapper), which is the mature tool and is
credited as such in README.md. Positions and velocities are returned in the
TEME frame of the TLE epoch, converted to metres and metres per second.

``validation/validate_tle.py`` checks this module's propagation against the
official SGP4 verification vectors ``SGP4-VER.TLE`` and ``tcppver.out`` that
ship inside the installed ``sgp4`` package, which are the test cases published
with Vallado, D. A., Crawford, P., Hujsak, R., and Kelso, T. S. (2006),
"Revisiting Spacetrack Report #3", AIAA 2006-6753.

Units: metres, metres per second, seconds unless a name says minutes.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sgp4.api import Satrec

from .frames import rtn_basis

_KM_TO_M = 1000.0


def state_from_tle(
    line1: str, line2: str, minutes_from_epoch: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """TEME position and velocity from a two-line element set.

    Parameters
    ----------
    line1, line2
        The two TLE lines, as published.
    minutes_from_epoch
        Offset from the TLE epoch, minutes. Negative values propagate backwards.

    Returns
    -------
    tuple of ndarray
        ``(position_m, velocity_mps)``, each shape ``(3,)``, in the TEME frame
        of the TLE epoch.

    Raises
    ------
    ValueError
        If ``sgp4`` reports a propagation error. The SGP4 error code is
        included in the message, since codes 1 to 6 mean different things and
        a caller needs to know which.
    """
    satellite = Satrec.twoline2rv(line1, line2)
    jd = satellite.jdsatepoch
    fr = satellite.jdsatepochF + minutes_from_epoch / 1440.0
    error, position_km, velocity_kmps = satellite.sgp4(jd, fr)
    if error != 0:
        raise ValueError(
            f"sgp4 propagation failed with error code {error} at "
            f"{minutes_from_epoch} minutes from epoch"
        )
    return (
        np.asarray(position_km, dtype=float) * _KM_TO_M,
        np.asarray(velocity_kmps, dtype=float) * _KM_TO_M,
    )


def synthetic_secondary(
    position_m: NDArray[np.float64],
    velocity_mps: NDArray[np.float64],
    *,
    offset_rtn_m: tuple[float, float, float],
    velocity_rotation_rad: float,
    rotation_axis: str = "radial",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A declared synthetic secondary state built from a primary state.

    This is **not** a catalogue object. It is a way to manufacture a close
    approach with declared geometry from a real primary state, so that the
    tooling can be exercised without a second element set. The offset is
    applied in the primary's radial/along-track/cross-track frame and the
    secondary's velocity is the primary's velocity rotated by
    ``velocity_rotation_rad`` about the named axis of that frame. The resulting
    relative speed is ``2 |v_perp| sin(rotation / 2)`` where ``v_perp`` is the
    component of the primary velocity perpendicular to the rotation axis, so it
    equals ``2 |v| sin(rotation / 2)`` only when the velocity is perpendicular
    to that axis. The measured value for the shipped example is in
    ``validation/validate_tle_output.txt``.

    Parameters
    ----------
    position_m, velocity_mps
        Primary state, metres and metres per second.
    offset_rtn_m
        Offset of the secondary in the radial, along-track and cross-track
        directions, metres.
    velocity_rotation_rad
        Rotation applied to the velocity, radians.
    rotation_axis
        ``"radial"``, ``"along_track"`` or ``"cross_track"``.

    Returns
    -------
    tuple of ndarray
        ``(position_m, velocity_mps)`` of the synthetic secondary.
    """
    basis = rtn_basis(position_m, velocity_mps)
    axes = {"radial": 0, "along_track": 1, "cross_track": 2}
    if rotation_axis not in axes:
        raise ValueError(
            f"rotation_axis must be one of {sorted(axes)}, got {rotation_axis!r}"
        )
    offset = np.asarray(offset_rtn_m, dtype=float).reshape(-1)
    if offset.shape != (3,):
        raise ValueError("offset_rtn_m must have 3 components")
    axis = basis[:, axes[rotation_axis]]
    angle = float(velocity_rotation_rad)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    velocity = np.asarray(velocity_mps, dtype=float)
    rotated = (
        velocity * cos_a
        + np.cross(axis, velocity) * sin_a
        + axis * (axis @ velocity) * (1.0 - cos_a)
    )
    return np.asarray(position_m, dtype=float) + basis @ offset, rotated


@dataclass(frozen=True)
class ClosestApproach:
    """Result of a numerical closest-approach search.

    Attributes
    ----------
    time_s
        Time of minimum separation, seconds, in the same timebase as the
        callables passed to :func:`find_closest_approach`.
    separation_m
        Minimum separation found, metres.
    coarse_separation_m
        Separation at the best coarse grid node, metres, so that the gain from
        refinement is visible.
    evaluations
        Number of state-function evaluations used.
    """

    time_s: float
    separation_m: float
    coarse_separation_m: float
    evaluations: int


def find_closest_approach(
    state_a: Callable[[float], tuple[NDArray[np.float64], NDArray[np.float64]]],
    state_b: Callable[[float], tuple[NDArray[np.float64], NDArray[np.float64]]],
    t_start_s: float,
    t_end_s: float,
    *,
    coarse_steps: int = 400,
    golden_iterations: int = 60,
) -> ClosestApproach:
    """Minimum separation between two propagated objects over a time interval.

    A uniform coarse scan locates the bracketing interval and golden-section
    search refines it. The scan step must be fine enough that the true minimum
    is not skipped; ``validation/validate_tle.py`` measures the sensitivity of
    the result to ``coarse_steps`` instead of assuming it.

    Parameters
    ----------
    state_a, state_b
        Callables mapping a time in seconds to ``(position_m, velocity_mps)``.
    t_start_s, t_end_s
        Search interval, seconds, with ``t_end_s > t_start_s``.
    coarse_steps
        Number of uniform intervals in the coarse scan.
    golden_iterations
        Golden-section iterations applied to the bracketing interval.

    Returns
    -------
    ClosestApproach
    """
    if not t_end_s > t_start_s:
        raise ValueError(f"t_end_s must exceed t_start_s, got {t_start_s!r} and {t_end_s!r}")
    if coarse_steps < 2:
        raise ValueError(f"coarse_steps must be at least 2, got {coarse_steps!r}")
    counter = {"n": 0}

    def separation(t: float) -> float:
        counter["n"] += 1
        pa, _ = state_a(t)
        pb, _ = state_b(t)
        return float(np.linalg.norm(pb - pa))

    times = np.linspace(t_start_s, t_end_s, coarse_steps + 1)
    values = np.array([separation(float(t)) for t in times])
    best = int(np.argmin(values))
    coarse_value = float(values[best])
    low = float(times[max(best - 1, 0)])
    high = float(times[min(best + 1, coarse_steps)])
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = low, high
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = separation(c), separation(d)
    for _ in range(golden_iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = separation(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = separation(d)
        if abs(b - a) < 1e-9:
            break
    t_best = 0.5 * (a + b)
    return ClosestApproach(
        time_s=float(t_best),
        separation_m=separation(t_best),
        coarse_separation_m=coarse_value,
        evaluations=counter["n"],
    )
