"""An independent closed-form solution of equation (1), for validation only.

Nothing in :mod:`twindiverge.plant`, :mod:`twindiverge.identifiability`,
:mod:`twindiverge.attribution`, :mod:`twindiverge.profile` or
:mod:`twindiverge.learned` uses this module. It exists so that the
Runge-Kutta integrator that every number in this repository rests on can be
scored against mathematics the integrator does not use, rather than against a
second numerical integration that would share its assumptions.

The derivation
--------------
Over a segment on which the current is constant, equation (1) is autonomous
and separable. Write ``q = T_c - k_t * i`` and

    equation (18)   J d(omega)/dt = -(c_d*omega^2 + b*omega + q)

Both sides divide by the quadratic and integrate. Let
``D = b^2 - 4*c_d*q`` be its discriminant.

**Real roots,** ``D > 0`` (which is the driven case here, because ``q`` is
negative when the motor torque exceeds the Coulomb torque): with
``r+ = (-b + sqrt(D)) / (2*c_d)`` and ``r- = (-b - sqrt(D)) / (2*c_d)``,
partial fractions give

    equation (19)   (omega - r+) / (omega - r-) = k * exp(s*t),
                    k = (omega_0 - r+) / (omega_0 - r-),
                    s = -c_d * (r+ - r-) / J

and solving for ``omega``,

    equation (20)   omega(t) = (r+ - r- * k * exp(s*t)) / (1 - k * exp(s*t)).

**Complex roots,** ``D < 0`` (the coasting case, where ``q = T_c > 0`` and
``b^2 = 1.44e-10`` is smaller than ``4*c_d*T_c = 2.0e-10``): with
``d = sqrt(-D)``,

    equation (21)   omega(t) = (d * tan(phi - t*d/(2*J)) - b) / (2*c_d),
                    phi = arctan((2*c_d*omega_0 + b) / d).

**Zero discriminant** is a measure-zero case that the declared parameters do
not realise; it raises rather than being approximated, because a silent
fallback to one of the two branches would be a wrong answer presented as a
reference.

A campaign with a current step is handled segment by segment, equation (20)
over the driven segment and equation (21) over the coasting one, with the
driven segment's terminal rate as the coasting segment's initial condition.
Each segment is exact, so the chained result is exact to within rounding.

Validity
--------
Same as equation (1): one sign of rotation, inside the declared rate window.
Equation (21) has poles where its tangent does, which happens when the wheel
would reach zero rate; the caller is outside the plant's declared validity
window before that, and :func:`exact_rate` raises if it is approached.

Measured agreement with the integrator is reported in
``validation/validate_plant_output.txt``.
"""

from __future__ import annotations

import math

import numpy as np

from .plant import Campaign, WheelParameters

__all__ = ["exact_campaign", "exact_rate"]


def exact_rate(
    parameters: WheelParameters,
    current_a: float,
    initial_rate_rad_s: float,
    times_s: np.ndarray,
) -> np.ndarray:
    """Equation (20) or (21): the exact rate over a constant-current segment, rad/s.

    ``times_s`` is measured from the start of the segment and need not start at
    zero. Returns an array of the same shape.
    """
    inertia, torque_constant, coulomb, viscous, drag = parameters.as_array()
    grid = np.asarray(times_s, dtype=float)
    if np.any(grid < 0.0):
        raise ValueError("segment times must not be negative")
    constant = coulomb - torque_constant * float(current_a)
    discriminant = viscous * viscous - 4.0 * drag * constant
    if discriminant == 0.0:
        raise ValueError(
            "equation (18) has a repeated root at these parameters, which neither "
            "equation (20) nor equation (21) covers; the declared parameters do not "
            "realise this case and it is raised rather than approximated"
        )
    if discriminant > 0.0:
        root = math.sqrt(discriminant)
        upper = (-viscous + root) / (2.0 * drag)
        lower = (-viscous - root) / (2.0 * drag)
        ratio = (initial_rate_rad_s - upper) / (initial_rate_rad_s - lower)
        decay = -drag * (upper - lower) / inertia
        growth = ratio * np.exp(decay * grid)
        if np.any(np.abs(1.0 - growth) < 1.0e-300):
            raise ValueError("equation (20) is singular over this time span")
        return (upper - lower * growth) / (1.0 - growth)
    spread = math.sqrt(-discriminant)
    phase = math.atan((2.0 * drag * initial_rate_rad_s + viscous) / spread)
    argument = phase - grid * spread / (2.0 * inertia)
    if np.any(argument <= -0.5 * math.pi + 1.0e-9):
        raise ValueError(
            "equation (21) approaches its pole over this time span: the wheel would "
            "reach zero rate, which is outside the declared validity window of "
            "equation (1)"
        )
    return (spread * np.tan(argument) - viscous) / (2.0 * drag)


def exact_campaign(parameters: WheelParameters, campaign: Campaign) -> np.ndarray:
    """The exact rate at every sample time of ``campaign``, rad/s.

    The driven and coasting segments are solved separately and chained through
    the driven segment's terminal rate, which is itself computed from equation
    (20) and not from a sample of it.
    """
    times = campaign.times_s()
    driven = times < campaign.drive_duration_s - 1.0e-12
    rates = np.empty_like(times)
    if driven.any():
        rates[driven] = exact_rate(
            parameters, campaign.drive_current_a, campaign.initial_rate_rad_s, times[driven]
        )
    if (~driven).any():
        handover = float(
            exact_rate(
                parameters,
                campaign.drive_current_a,
                campaign.initial_rate_rad_s,
                np.array([campaign.drive_duration_s]),
            )[0]
        ) if campaign.drive_duration_s > 0.0 else campaign.initial_rate_rad_s
        rates[~driven] = exact_rate(
            parameters, 0.0, handover, times[~driven] - campaign.drive_duration_s
        )
    return rates
