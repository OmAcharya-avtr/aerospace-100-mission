"""The declared digital twin: a reaction-wheel assembly observed through its rate.

The plant
---------
One state, the wheel rate ``omega`` in rad/s, driven by a logged motor current
``i(t)`` in A:

    equation (1)    J * d(omega)/dt = k_t * i(t) - T_c - b * omega
                                      - c_d * omega**2

with, in SI units throughout,

    J     wheel plus rotor moment of inertia            kg m^2
    k_t   motor torque constant                         N m / A
    T_c   Coulomb (speed-independent) friction torque   N m
    b     viscous friction coefficient                  N m s / rad
    c_d   quadratic drag coefficient                    N m s^2 / rad^2

The friction decomposition into a speed-independent term, a term linear in
speed and a term quadratic in speed is the standard one; see
Armstrong-Helouvry, Dupont & Canudas de Wit 1994, "A survey of models,
analysis tools and compensation methods for the control of machines with
friction", Automatica 30(7), 1083-1138, section 2, for the Coulomb-plus-viscous
form, and Bialke 1998, "High fidelity mathematical modeling of reaction wheel
performance", 21st Annual AAS Guidance and Control Conference, AAS 98-063, for
the same decomposition applied to a spacecraft reaction wheel together with the
motor torque constant of equation (1).

**The coefficient values declared in** :data:`NOMINAL` **are not taken from
either source and are not a measurement of any wheel.** They are a declared
fiction with an engineering shape, chosen so that the identifiability structure
this product exists to demonstrate is visible at a realistic sensor noise
level. Nothing in this repository is a measurement of real hardware.

Assumptions and validity range
------------------------------
- **One sign of rotation only.** Equation (1) carries the Coulomb torque as a
  constant ``-T_c`` rather than ``-T_c * sign(omega)``, which is correct only
  for ``omega > 0``. The declared validity window is
  ``OMEGA_MIN <= omega <= OMEGA_MAX`` (20 to 400 rad/s) and
  :func:`simulate` raises :class:`ValueError` when a trajectory leaves it. A
  wheel zero crossing is outside this model: that is exactly the regime where
  the Coulomb term is hardest and a smoothed or Stribeck form would be needed,
  and it is named in README.md under Limitations rather than approximated here.
- **Rigid wheel, rigid mount, no cogging, no thermal dependence.** The motor
  torque constant is a constant.
- **The current is logged exactly.** Current-sensor error is not modelled; it
  would add a sixth parameter collinear with ``k_t`` on a constant-current
  segment, which is a different degeneracy from the one studied here.
- **Zero-order hold on the current.** ``i(t)`` is piecewise constant between
  the declared switch times of the excitation profile, and the current is
  sampled ONCE PER STEP at that step's left endpoint and used for all four
  Runge-Kutta stages. Evaluating it per stage instead costs the integrator
  three orders of accuracy at the switching step, which was measured during
  this build: with the current evaluated at each stage's own time, the fourth
  stage of the step ENDING at the switch sees the new current, the global error
  at the end of a 20 s driven segment was 4.102564e-02 rad/s at a 0.1 s step,
  and the measured convergence order over steps from 0.4 s to 0.025 s was
  1.0000 instead of 4. Sampling once per step is also the physically correct
  reading of a zero-order hold: the commanded current is constant across a
  sample interval by construction.

Integration
-----------
Classical explicit fourth-order Runge-Kutta, Butcher 2016, "Numerical Methods
for Ordinary Differential Equations", 3rd ed., Wiley, section 232. A fixed step
is used deliberately: an adaptive step would make the trajectory, and therefore
the residual, depend on the parameter values through the step sequence as well
as through equation (1), and the sensitivity matrix of
:func:`simulate_with_sensitivities` would then be the sensitivity of a
different map from the one being analysed.

Sensitivities
-------------
The forward (direct) sensitivity equations are integrated alongside the state
by the same Runge-Kutta scheme, in **logarithmic parameters** ``p_j =
log(theta_j)``, so that

    equation (2)    s_j(t) = d(omega)/d(p_j) = theta_j * d(omega)/d(theta_j)

is dimensionless and is the change in rate produced by a unit *relative*
change in parameter ``j``. The direct method is Dickinson & Gelinas 1976,
"Sensitivity analysis of ordinary differential equation systems - a direct
method", Journal of Computational Physics 21(2), 123-143. Differentiating
equation (1),

    equation (3)    d(s_j)/dt = (df/d(omega)) * s_j + g_j,
                    df/d(omega) = (-b - 2 * c_d * omega) / J,
                    g = (-f, k_t*i/J, -T_c/J, -b*omega/J, -c_d*omega**2/J)

with ``s_j(0) = 0`` for every ``j`` because the initial rate is a declared
boundary condition and not a function of the parameters.

The reaction-torque channel is the net torque on the wheel,

    equation (4)    tau(t) = k_t * i(t) - T_c - b * omega - c_d * omega**2
                           = J * d(omega)/dt                       [N m]

which is what a torque sensor between the wheel housing and its mount reads.
Its sensitivities follow by the chain rule,

    equation (5)    d(tau)/d(p_j) = h_j + (d(tau)/d(omega)) * s_j,
                    h = (0, k_t*i, -T_c, -b*omega, -c_d*omega**2),
                    d(tau)/d(omega) = -b - 2 * c_d * omega

and note that ``h_0 = 0``: equation (4) contains no ``J``.

The two exact identities this product rests on
----------------------------------------------
Both are proved here on paper and confirmed numerically by
``validation/validate_identifiability.py``; their agreement is itself one of
the validation checks.

**Identity A (the rate channel cannot see the absolute torque scale).** Write
``theta_S = (J, k_t, T_c, b, c_d)`` and let every one of the five be scaled by
the same factor, ``theta -> lambda * theta``. Equation (1) divided by ``J``
depends on the parameters only through ``k_t/J``, ``T_c/J``, ``b/J`` and
``c_d/J``, every one of which is invariant under that scaling. Hence
``omega(t)`` is unchanged for every ``lambda > 0`` and every excitation, and
differentiating at ``lambda = 1`` gives

    equation (6)    sum_j s_j(t) = 0 for all t,

i.e. the sensitivity matrix of the rate channel annihilates the direction
``v_scale = (1, 1, 1, 1, 1) / sqrt(5)`` in logarithmic parameters, **exactly,
for any excitation and any record length**. The same statement follows
directly from equation (3): summing ``g`` over ``j`` gives
``-f + (k_t*i - T_c - b*omega - c_d*omega**2)/J = -f + f = 0``, so the sum of
the sensitivities obeys a homogeneous linear equation with zero initial
condition.

**Identity B (the torque channel can).** Summing ``h`` over ``j`` gives
``k_t*i - T_c - b*omega - c_d*omega**2 = tau(t)``, so by equation (5) and
identity A,

    equation (7)    sum_j d(tau)/d(p_j) = tau(t),

which is non-zero wherever the net torque is. Adding a reaction-torque channel
therefore removes the degeneracy of identity A, and the sensitivity of the
scale direction in that channel is *exactly the torque signal itself*.

**Corollary (an unexcited parameter).** If ``i(t) = 0`` for the whole record
then ``g_1 = 0`` and ``h_1 = 0`` identically, so ``s_1(t) = 0``: ``k_t`` is
exactly unidentifiable from a coast-down record in either channel. Combined
with identity A the null space of a coast-down rate record is the
two-dimensional span of ``e_{k_t}`` and ``(1, 0, 1, 1, 1) / 2``.

In exact arithmetic equations (6) and (7) are identities. In floating point
they hold to within rounding, and nothing in this package asserts them
exactly; the measured residuals are reported in
``validation/validate_identifiability_output.txt``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

__all__ = [
    "CAMPAIGNS",
    "NOMINAL",
    "OMEGA_0",
    "OMEGA_MAX",
    "OMEGA_MIN",
    "PARAMETER_NAMES",
    "PARAMETER_UNITS",
    "SCALE_DIRECTION",
    "Campaign",
    "Trajectory",
    "WheelParameters",
    "perturb",
    "simulate",
    "simulate_with_sensitivities",
]

#: Parameter names, in the order used by every vector and matrix here.
PARAMETER_NAMES: Final[tuple[str, ...]] = ("J", "k_t", "T_c", "b", "c_d")

#: SI units of each parameter, printed beside every estimate.
PARAMETER_UNITS: Final[tuple[str, ...]] = (
    "kg m^2",
    "N m / A",
    "N m",
    "N m s / rad",
    "N m s^2 / rad^2",
)

#: Declared validity window of equation (1), rad/s. One sign of rotation only.
OMEGA_MIN: Final[float] = 20.0
OMEGA_MAX: Final[float] = 400.0

#: Declared initial wheel rate of every campaign, rad/s.
OMEGA_0: Final[float] = 100.0

#: The exact null direction of the rate channel, equation (6), in logarithmic
#: parameters. Unit-norm. Derived on paper above; confirmed numerically by
#: ``validation/validate_identifiability.py``, never used to build the
#: sensitivity matrix it is compared against.
SCALE_DIRECTION: Final[np.ndarray] = np.full(5, 1.0 / math.sqrt(5.0))


@dataclass(frozen=True)
class WheelParameters:
    """The five parameters of equation (1), in SI units.

    Values are declared, not measured. See the module docstring.
    """

    inertia_kg_m2: float = 1.30e-2
    torque_constant_nm_per_a: float = 3.20e-2
    coulomb_nm: float = 2.50e-3
    viscous_nm_s: float = 1.20e-5
    drag_nm_s2: float = 2.00e-8

    def __post_init__(self) -> None:
        for name, value in zip(PARAMETER_NAMES, self.as_array(), strict=True):
            if not math.isfinite(value):
                raise ValueError(f"parameter {name} must be finite, got {value!r}")
            if value <= 0.0:
                raise ValueError(
                    f"parameter {name} must be strictly positive (the logarithmic "
                    f"parameterisation of equation (2) requires it), got {value!r}"
                )

    def as_array(self) -> np.ndarray:
        """The five parameters as a float array in :data:`PARAMETER_NAMES` order."""
        return np.array(
            [
                self.inertia_kg_m2,
                self.torque_constant_nm_per_a,
                self.coulomb_nm,
                self.viscous_nm_s,
                self.drag_nm_s2,
            ],
            dtype=float,
        )

    @classmethod
    def from_array(cls, values: np.ndarray) -> WheelParameters:
        """Build from a five-element array in :data:`PARAMETER_NAMES` order."""
        array = np.asarray(values, dtype=float)
        if array.shape != (5,):
            raise ValueError(f"expected 5 parameters, got shape {array.shape}")
        return cls(*(float(value) for value in array))


#: The declared nominal parameter set: what the twin is built with.
NOMINAL: Final[WheelParameters] = WheelParameters()


def perturb(base: WheelParameters, delta_log: np.ndarray) -> WheelParameters:
    """Apply a relative perturbation in logarithmic parameters, equation (2).

    ``delta_log[j]`` is the change in ``log(theta_j)``, so a value of 0.05 is a
    5.127 per cent increase (``exp(0.05) - 1``) and not a 5 per cent one. The
    logarithmic form is used throughout because it keeps every parameter
    strictly positive and makes the sensitivity matrix dimensionless.
    """
    array = np.asarray(delta_log, dtype=float)
    if array.shape != (5,):
        raise ValueError(f"expected 5 perturbation components, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("every perturbation component must be finite")
    return WheelParameters.from_array(base.as_array() * np.exp(array))


@dataclass(frozen=True)
class Campaign:
    """A declared excitation and sampling schedule: what telemetry exists.

    ``drive_current_a`` is held for ``drive_duration_s`` seconds from the start
    of the record and is zero afterwards. The step is a zero-order hold and
    ``drive_duration_s`` is an exact multiple of ``sample_period_s``, so no
    integration step straddles the discontinuity.
    """

    name: str
    sample_period_s: float
    samples: int
    drive_current_a: float
    drive_duration_s: float
    initial_rate_rad_s: float = OMEGA_0

    def __post_init__(self) -> None:
        if self.sample_period_s <= 0.0:
            raise ValueError(
                f"sample_period_s must be strictly positive, got {self.sample_period_s!r}"
            )
        if self.samples < 2:
            raise ValueError(f"samples must be at least 2, got {self.samples!r}")
        if self.drive_duration_s < 0.0:
            raise ValueError(
                f"drive_duration_s must not be negative, got {self.drive_duration_s!r}"
            )
        ratio = self.drive_duration_s / self.sample_period_s
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(
                "drive_duration_s must be an exact multiple of sample_period_s so "
                f"that no step straddles the current step: got {self.drive_duration_s} "
                f"/ {self.sample_period_s}"
            )

    @property
    def duration_s(self) -> float:
        """Record length in seconds."""
        return (self.samples - 1) * self.sample_period_s

    def times_s(self) -> np.ndarray:
        """Sample times, seconds from the start of the record."""
        return np.arange(self.samples, dtype=float) * self.sample_period_s

    def current_a(self, times_s: np.ndarray | None = None) -> np.ndarray:
        """The logged motor current at each sample time, A."""
        grid = self.times_s() if times_s is None else np.asarray(times_s, dtype=float)
        return np.where(grid < self.drive_duration_s - 1e-12, self.drive_current_a, 0.0)

    def _current_at(self, time_s: float) -> float:
        """Zero-order-hold current at an arbitrary time, A."""
        return self.drive_current_a if time_s < self.drive_duration_s - 1e-12 else 0.0


#: The three declared campaigns. Their identifiability structure is the subject
#: of this product and is measured in
#: ``validation/validate_identifiability_output.txt``:
#:
#: - ``full``     200 s, 40 s driven then 160 s coasting. Rate channel rank 4 of
#:                5; the only null direction is the exact scale direction of
#:                equation (6).
#: - ``short``    20 s, 10 s driven then 10 s coasting. Same structural rank 4,
#:                but the viscous and drag coefficients are PRACTICALLY
#:                unidentifiable over so narrow a rate excursion. This campaign
#:                exists to separate practical from structural.
#: - ``coast``    60 s, never driven. Rank 3 of 5: the scale direction plus the
#:                torque constant, which the excitation never excites
#:                (corollary to identity B).
CAMPAIGNS: Final[dict[str, Campaign]] = {
    "full": Campaign("full", 0.1, 2001, 1.0, 40.0),
    "short": Campaign("short", 0.1, 201, 1.0, 10.0),
    "coast": Campaign("coast", 0.1, 601, 0.0, 0.0),
}


@dataclass(frozen=True)
class Trajectory:
    """A simulated record: rate, net torque, and optionally their sensitivities.

    ``rate_rad_s`` and ``torque_nm`` have shape ``(samples,)``;
    ``rate_sensitivity`` and ``torque_sensitivity``, when present, have shape
    ``(samples, 5)`` and are derivatives with respect to the LOGARITHM of each
    parameter, equation (2), so they are in rad/s and N m respectively.
    """

    campaign: Campaign
    parameters: WheelParameters
    times_s: np.ndarray
    rate_rad_s: np.ndarray
    torque_nm: np.ndarray
    rate_sensitivity: np.ndarray | None = None
    torque_sensitivity: np.ndarray | None = None

    @property
    def rate_excursion_rad_s(self) -> float:
        """Peak-to-peak rate excursion over the record, rad/s."""
        return float(self.rate_rad_s.max() - self.rate_rad_s.min())


def _derivative(
    rate: float, current: float, values: tuple[float, float, float, float, float]
) -> float:
    """Right-hand side of equation (1), rad/s^2. Scalar path, no numpy.

    ``current`` is the zero-order-hold current of the step being taken, not of
    the stage's own time. See the module docstring.
    """
    inertia, torque_constant, coulomb, viscous, drag = values
    return (
        torque_constant * current - coulomb - viscous * rate - drag * rate * rate
    ) / inertia


def _joint_derivative(
    rate: float,
    sensitivity: tuple[float, float, float, float, float],
    current: float,
    values: tuple[float, float, float, float, float],
) -> tuple[float, tuple[float, float, float, float, float]]:
    """Equations (1) and (3) together. Scalar arithmetic: this is the hot loop."""
    inertia, torque_constant, coulomb, viscous, drag = values
    net = torque_constant * current - coulomb - viscous * rate - drag * rate * rate
    rate_dot = net / inertia
    jacobian = (-viscous - 2.0 * drag * rate) / inertia
    forcing = (
        -rate_dot,
        torque_constant * current / inertia,
        -coulomb / inertia,
        -viscous * rate / inertia,
        -drag * rate * rate / inertia,
    )
    return rate_dot, tuple(
        jacobian * sensitivity[j] + forcing[j] for j in range(5)
    )  # type: ignore[return-value]


def _check_window(rate: float, campaign: Campaign) -> None:
    if not (OMEGA_MIN <= rate <= OMEGA_MAX):
        raise ValueError(
            f"campaign {campaign.name!r} left the declared validity window of "
            f"equation (1): rate {rate:.6g} rad/s is outside "
            f"[{OMEGA_MIN}, {OMEGA_MAX}]. Equation (1) carries the Coulomb torque "
            "as a constant and is valid for one sign of rotation only; a zero "
            "crossing needs a smoothed friction model this package does not ship."
        )


def simulate(
    parameters: WheelParameters, campaign: Campaign | str = "full"
) -> Trajectory:
    """Integrate equation (1) over ``campaign``. Rate and net torque only.

    Returns a :class:`Trajectory` whose ``rate_sensitivity`` is ``None``. Use
    :func:`simulate_with_sensitivities` when the sensitivity matrix is wanted;
    this function is the one used to generate residual data, where it is called
    many times and the sensitivities are not needed.

    Raises :class:`ValueError` if the trajectory leaves the declared validity
    window ``[OMEGA_MIN, OMEGA_MAX]`` rad/s.
    """
    plan = CAMPAIGNS[campaign] if isinstance(campaign, str) else campaign
    values = tuple(float(value) for value in parameters.as_array())
    step = plan.sample_period_s
    rate = float(plan.initial_rate_rad_s)
    rates = np.empty(plan.samples, dtype=float)
    for index in range(plan.samples):
        time_s = index * step
        _check_window(rate, plan)
        rates[index] = rate
        if index == plan.samples - 1:
            break
        held = plan._current_at(time_s)
        first = _derivative(rate, held, values)  # type: ignore[arg-type]
        second = _derivative(rate + 0.5 * step * first, held, values)  # type: ignore[arg-type]
        third = _derivative(rate + 0.5 * step * second, held, values)  # type: ignore[arg-type]
        fourth = _derivative(rate + step * third, held, values)  # type: ignore[arg-type]
        rate = rate + step * (first + 2.0 * second + 2.0 * third + fourth) / 6.0
    _, torque_constant, coulomb, viscous, drag = values
    current = plan.current_a()
    torque = (
        torque_constant * current - coulomb - viscous * rates - drag * rates * rates
    )
    return Trajectory(plan, parameters, plan.times_s(), rates, torque)


def simulate_with_sensitivities(
    parameters: WheelParameters, campaign: Campaign | str = "full"
) -> Trajectory:
    """Integrate equations (1) and (3), then form equation (5).

    The returned ``rate_sensitivity`` and ``torque_sensitivity`` have shape
    ``(samples, 5)`` and are derivatives with respect to ``log(theta_j)``.
    """
    plan = CAMPAIGNS[campaign] if isinstance(campaign, str) else campaign
    values = tuple(float(value) for value in parameters.as_array())
    step = plan.sample_period_s
    rate = float(plan.initial_rate_rad_s)
    sensitivity = (0.0, 0.0, 0.0, 0.0, 0.0)
    rates = np.empty(plan.samples, dtype=float)
    sensitivities = np.empty((plan.samples, 5), dtype=float)
    for index in range(plan.samples):
        time_s = index * step
        _check_window(rate, plan)
        rates[index] = rate
        sensitivities[index] = sensitivity
        if index == plan.samples - 1:
            break
        held = plan._current_at(time_s)
        rate_1, sens_1 = _joint_derivative(rate, sensitivity, held, values)  # type: ignore[arg-type]
        rate_2, sens_2 = _joint_derivative(
            rate + 0.5 * step * rate_1,
            tuple(sensitivity[j] + 0.5 * step * sens_1[j] for j in range(5)),  # type: ignore[arg-type]
            held,
            values,  # type: ignore[arg-type]
        )
        rate_3, sens_3 = _joint_derivative(
            rate + 0.5 * step * rate_2,
            tuple(sensitivity[j] + 0.5 * step * sens_2[j] for j in range(5)),  # type: ignore[arg-type]
            held,
            values,  # type: ignore[arg-type]
        )
        rate_4, sens_4 = _joint_derivative(
            rate + step * rate_3,
            tuple(sensitivity[j] + step * sens_3[j] for j in range(5)),  # type: ignore[arg-type]
            held,
            values,  # type: ignore[arg-type]
        )
        rate = rate + step * (rate_1 + 2.0 * rate_2 + 2.0 * rate_3 + rate_4) / 6.0
        sensitivity = tuple(  # type: ignore[assignment]
            sensitivity[j]
            + step * (sens_1[j] + 2.0 * sens_2[j] + 2.0 * sens_3[j] + sens_4[j]) / 6.0
            for j in range(5)
        )
    _, torque_constant, coulomb, viscous, drag = values
    current = plan.current_a()
    torque = (
        torque_constant * current - coulomb - viscous * rates - drag * rates * rates
    )
    # Equation (5). h_0 is zero: equation (4) contains no J.
    explicit = np.column_stack(
        [
            np.zeros_like(rates),
            torque_constant * current,
            np.full_like(rates, -coulomb),
            -viscous * rates,
            -drag * rates * rates,
        ]
    )
    torque_jacobian = (-viscous - 2.0 * drag * rates)[:, None]
    torque_sensitivity = explicit + torque_jacobian * sensitivities
    return Trajectory(
        plan,
        parameters,
        plan.times_s(),
        rates,
        torque,
        rate_sensitivity=sensitivities,
        torque_sensitivity=torque_sensitivity,
    )
