"""Light-time solution by fixed-point iteration, with the residual reported.

The problem
-----------
A range measurement is not instantaneous.  For a **down-leg** (satellite
transmits, station receives at epoch ``t_r``) the light time ``tau`` satisfies

    c * tau = | r_sat(t_r - tau) - r_sta(t_r) |                        (1)

and for an **up-leg** (station transmits at ``t_t``, satellite receives)

    c * tau = | r_sat(t_t + tau) - r_sta(t_t) |.                       (2)

Both are solved here by the standard fixed-point iteration -- substitute the
current ``tau`` on the right, read off the next ``tau`` -- which is the
classical light-time solution used for radiometric observables (Moyer,
*Formulation for Observed and Computed Values of Deep Space Network Data Types
for Navigation*, JPL Deep Space Communications and Navigation Series
Monograph 2, 2000).

Why it converges, and how fast
------------------------------
Differentiating the right-hand side of (1) with respect to ``tau`` gives a
map whose derivative is ``-rho_dot/c`` in magnitude, so the iteration is a
contraction with rate ``|rho_dot|/c``.  For a LEO pass ``|rho_dot| <= 7.1
km/s``, giving a contraction factor of about 2.4e-5: each iteration gains
roughly 4.6 decimal digits, so from a zero initial guess the residual reaches
double-precision noise in three or four iterations.  The measured iteration
count and final residual for a representative pass are in
``validation/validate_lighttime.py`` -- this module reports both on every
call rather than saying "converged".

Sign convention
---------------
``tau`` is always **positive**: a duration, not an offset.  The direction is
carried by which function you call and by
:attr:`LightTimeSolution.emission_time_s` / ``reception_time_s``, which are
ordered ``emission_time_s < reception_time_s`` for both legs.  The
light-time-corrected range-rates for the two legs of a two-way measurement
then go into :func:`dopplerkit.doppler.two_way_doppler_two_leg_hz` in the
order (up, down).

Assumptions and validity
------------------------
* Straight-line propagation at exactly ``c`` in vacuum.  No tropospheric or
  ionospheric delay, no Shapiro (gravitational) delay, no antenna
  phase-centre or transponder group delay.  For a LEO link the neglected
  tropospheric zenith delay alone is of order 2.3 m (about 8 ns), which is
  far larger than the 1e-6 m convergence tolerance used here -- the
  tolerance describes how well the *vacuum* equation is solved, not the
  accuracy of the physical light time.
* The position callables must be valid over ``[t - 1 s, t + 1 s]`` around the
  requested epoch (a LEO light time is under 10 ms).
* Positions in metres, times in seconds, in one common inertial frame.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .constants import C_M_S

PositionFn = Callable[[float], np.ndarray]
"""Callable mapping time [s] to an inertial position vector [m], shape (3,)."""


@dataclass(frozen=True)
class LightTimeSolution:
    """Result of a light-time iteration, with its convergence evidence.

    Fields
    ------
    light_time_s
        Solved light time tau [s], always positive.
    range_m
        The geometric range ``c * tau`` [m] consistent with the solution.
    iterations
        Number of fixed-point updates performed.
    residual_m
        ``| c * tau - |rho(tau)| |`` at the returned ``tau`` [m].  This is the
        quantity the tolerance is applied to, and it is reported on every
        solution so a caller never has to trust the word "converged".
    converged
        True if ``residual_m <= tol_m`` within ``max_iter``.
    emission_time_s, reception_time_s
        The two epochs [s], ordered ``emission_time_s < reception_time_s``.
    leg
        ``"up"`` (station transmits) or ``"down"`` (satellite transmits).
    """

    light_time_s: float
    range_m: float
    iterations: int
    residual_m: float
    converged: bool
    emission_time_s: float
    reception_time_s: float
    leg: str

    def as_dict(self) -> dict[str, object]:
        """Plain-dict view, JSON-serialisable, units in the key names."""
        return {
            "light_time_s": self.light_time_s,
            "range_m": self.range_m,
            "iterations": self.iterations,
            "residual_m": self.residual_m,
            "converged": self.converged,
            "emission_time_s": self.emission_time_s,
            "reception_time_s": self.reception_time_s,
            "leg": self.leg,
        }


def _validate(tol_m: float, max_iter: int) -> None:
    if not tol_m > 0.0:
        raise ValueError(f"tol_m must be > 0, got {tol_m}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")


def _norm(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    for name, vec in (("satellite", av), ("station", bv)):
        if vec.shape != (3,):
            raise ValueError(
                f"{name} position callable must return shape (3,), got {vec.shape}"
            )
    return float(np.linalg.norm(av - bv))


def down_leg_light_time(
    satellite_position_fn: PositionFn,
    station_position_fn: PositionFn,
    reception_time_s: float,
    tol_m: float = 1e-6,
    max_iter: int = 50,
) -> LightTimeSolution:
    """Solve ``c tau = |r_sat(t_r - tau) - r_sta(t_r)|`` for the down-leg.

    The satellite transmits at ``reception_time_s - tau``; the station
    receives at ``reception_time_s``.

    Convention: ``tau > 0``; emission precedes reception.  The returned
    ``range_m`` is ``c * tau``, the light-time-consistent geometric range, not
    the instantaneous range at the reception epoch.

    Direction: down-leg (spacecraft to ground), one way.

    Units: times [s], positions [m], ``tol_m`` [m], returns a
    :class:`LightTimeSolution` with [s] and [m] fields.

    Reference: classical light-time solution, Moyer (2000), Monograph 2.
    Vacuum straight-line propagation only -- see the module docstring for what
    is dropped.  Does not raise on non-convergence; check
    :attr:`LightTimeSolution.converged` and ``residual_m``.
    """
    _validate(tol_m, max_iter)
    t_r = float(reception_time_s)
    r_sta = station_position_fn(t_r)
    tau = 0.0
    iterations = 0
    residual = float("inf")
    for _ in range(max_iter):
        rng = _norm(satellite_position_fn(t_r - tau), r_sta)
        tau_new = rng / C_M_S
        iterations += 1
        residual = abs(C_M_S * tau_new - _norm(satellite_position_fn(t_r - tau_new), r_sta))
        tau = tau_new
        if residual <= tol_m:
            break
    return LightTimeSolution(
        light_time_s=tau,
        range_m=C_M_S * tau,
        iterations=iterations,
        residual_m=residual,
        converged=residual <= tol_m,
        emission_time_s=t_r - tau,
        reception_time_s=t_r,
        leg="down",
    )


def up_leg_light_time(
    satellite_position_fn: PositionFn,
    station_position_fn: PositionFn,
    transmission_time_s: float,
    tol_m: float = 1e-6,
    max_iter: int = 50,
) -> LightTimeSolution:
    """Solve ``c tau = |r_sat(t_t + tau) - r_sta(t_t)|`` for the up-leg.

    The station transmits at ``transmission_time_s``; the satellite receives
    at ``transmission_time_s + tau``.

    Convention, units, reference and limitations: as
    :func:`down_leg_light_time`, with the leg reversed.  ``tau > 0`` and
    emission still precedes reception.
    """
    _validate(tol_m, max_iter)
    t_t = float(transmission_time_s)
    r_sta = station_position_fn(t_t)
    tau = 0.0
    iterations = 0
    residual = float("inf")
    for _ in range(max_iter):
        rng = _norm(satellite_position_fn(t_t + tau), r_sta)
        tau_new = rng / C_M_S
        iterations += 1
        residual = abs(C_M_S * tau_new - _norm(satellite_position_fn(t_t + tau_new), r_sta))
        tau = tau_new
        if residual <= tol_m:
            break
    return LightTimeSolution(
        light_time_s=tau,
        range_m=C_M_S * tau,
        iterations=iterations,
        residual_m=residual,
        converged=residual <= tol_m,
        emission_time_s=t_t,
        reception_time_s=t_t + tau,
        leg="up",
    )


@dataclass(frozen=True)
class TwoWayLightTime:
    """The two legs of a coherent two-way measurement, in transmission order.

    ``up`` is solved first from the ground transmit epoch; ``down`` is then
    solved backwards from the ground receive epoch
    ``t_transmit + tau_up + tau_down``, so the two legs are the legs of one
    physical round trip and not two independent approximations.

    Fields: ``up`` and ``down`` (:class:`LightTimeSolution`),
    ``round_trip_s`` [s] = ``up.light_time_s + down.light_time_s``,
    ``transmit_time_s`` and ``receive_time_s`` [s] at the ground station.
    """

    up: LightTimeSolution
    down: LightTimeSolution
    round_trip_s: float
    transmit_time_s: float
    receive_time_s: float


def two_way_light_time(
    satellite_position_fn: PositionFn,
    station_position_fn: PositionFn,
    transmission_time_s: float,
    tol_m: float = 1e-6,
    max_iter: int = 50,
) -> TwoWayLightTime:
    """Solve both legs of a two-way measurement from the ground transmit epoch.

    Procedure: solve the up-leg from ``transmission_time_s``; take the
    transponder epoch as ``transmission_time_s + tau_up``; then solve the
    down-leg *backwards* from a provisional ground receive epoch, iterating the
    receive epoch until the round trip is self-consistent to ``tol_m``.

    Convention: both light times positive, legs returned in transmission
    order (up, then down).  Feed ``up`` and ``down`` range-rates to
    :func:`dopplerkit.doppler.two_way_doppler_two_leg_hz` in that order.

    Units: [s], [m].  Reference: Moyer (2000), Monograph 2, two-way light-time
    solution.  Same vacuum-only limitations as the single-leg functions.
    """
    _validate(tol_m, max_iter)
    t_t = float(transmission_time_s)
    up = up_leg_light_time(
        satellite_position_fn, station_position_fn, t_t, tol_m=tol_m, max_iter=max_iter
    )
    round_trip = 2.0 * up.light_time_s
    down = up  # replaced below; keeps the type checker honest if max_iter == 0
    for _ in range(max_iter):
        down = down_leg_light_time(
            satellite_position_fn,
            station_position_fn,
            t_t + round_trip,
            tol_m=tol_m,
            max_iter=max_iter,
        )
        new_round_trip = up.light_time_s + down.light_time_s
        if abs(new_round_trip - round_trip) * C_M_S <= tol_m:
            round_trip = new_round_trip
            break
        round_trip = new_round_trip
    return TwoWayLightTime(
        up=up,
        down=down,
        round_trip_s=round_trip,
        transmit_time_s=t_t,
        receive_time_s=t_t + round_trip,
    )


__all__ = [
    "LightTimeSolution",
    "PositionFn",
    "TwoWayLightTime",
    "down_leg_light_time",
    "two_way_light_time",
    "up_leg_light_time",
]
