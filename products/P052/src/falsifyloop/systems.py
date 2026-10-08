r"""The simulator under test: a single-axis attitude-hold loop with a
rate-limited actuator and a sinusoidal gust.

What this model is, and what it is not
--------------------------------------
It is a **synthetic benchmark closed loop**, dimensionally consistent, with the
structure that single-axis attitude control is normally written in: second-order
rigid-body attitude dynamics with aerodynamic damping, a proportional-derivative
attitude controller, a first-order actuator with a slew-rate limit and a
deflection limit, and an additive sinusoidal disturbance in angular
acceleration.

**No parameter value here was identified from any aircraft and none is traceable
to flight data.** The nominal values in :class:`LoopParameters` are declared
constants of this benchmark, chosen so the nominal closed loop is stable and
lightly damped and so the declared search box contains both satisfying and
violating settings. Treating any number below as a property of a real vehicle
would be a misreading. The *structure* is standard and is cited; the *values*
are this package's own.

Equations (continuous time)
---------------------------
State: attitude ``theta`` [deg], attitude rate ``q`` [deg/s], actuator
deflection ``delta`` [deg].

.. math::
    \dot\theta &= q \\
    \dot q &= M_q\, q + M_\delta\, \delta + w(t) \\
    \dot\delta &= \mathrm{clip}\!\left(\frac{\delta_\mathrm{cmd} - \delta}
                   {\tau},\; -\dot\delta_\mathrm{max},\;
                   +\dot\delta_\mathrm{max}\right)

with the controller and disturbance

.. math::
    e &= \theta_\mathrm{ref} - \theta \\
    \delta_\mathrm{cmd} &= \mathrm{clip}(K_p e - K_d q,\;
                            -\delta_\mathrm{max},\; +\delta_\mathrm{max}) \\
    w(t) &= A_g \sin(2\pi f_g t)

Units: ``M_q`` [1/s], ``M_delta`` [deg/s^2 per deg], ``tau`` [s],
``rate_limit`` [deg/s], ``deflection_limit`` [deg], ``Kp`` [deg/deg],
``Kd`` [deg per deg/s], ``A_g`` [deg/s^2], ``f_g`` [Hz].

Validity range of the model as written: small-angle single-axis motion with the
attitude treated as decoupled from translation, no sensor noise, no measurement
delay, no actuator dead-band, and gravity and airspeed effects folded into the
constant ``M_q`` and ``M_delta``. Outside small angles the single-axis
decoupling is not even approximately right and the model says nothing.

Discretisation
--------------
Explicit (forward) Euler at a fixed ``dt``, with the saturations applied to the
post-update state. Euler is used deliberately: the vector field is
**non-smooth** because of the two clips, so a higher-order scheme does not
deliver its order across a saturation event, and the lower per-step cost buys
simulations, which is the resource this whole package measures. The local
truncation error is ``O(dt^2)`` per step and the global error ``O(dt)``
(Butcher 2016). Two consequences are measured rather than asserted, in
``validation/validate_simulator.py``:

* against the exact zero-order-hold matrix-exponential solution of the
  *unsaturated* linear loop, where an analytic answer exists;
* against the same Euler scheme at ``dt/8``, as a self-convergence check in the
  saturated regime where no analytic answer exists.

Explicit Euler is stable on this plant only while ``dt`` is small against the
fastest time constant, which here is the actuator ``tau``; ``dt = 0.005`` s
against ``tau >= 0.025`` s gives ``dt/tau <= 0.2``. ``simulate`` rejects a
``dt`` that exceeds ``tau``, which is where the scheme stops being meaningful at
all.

References for the structure (not for the values)
-------------------------------------------------
Etkin, B. and Reid, L. D. (1996), *Dynamics of Flight: Stability and Control*,
3rd ed., Wiley. Single-axis attitude dynamics with aerodynamic damping and
control-surface effectiveness as a linearised second-order system.

Stevens, B. L., Lewis, F. L. and Johnson, E. N. (2015), *Aircraft Control and
Simulation*, 3rd ed., Wiley. First-order actuator models with rate and
deflection limits in flight-control simulation.

Franklin, G. F., Powell, J. D. and Emami-Naeini, A. (2015), *Feedback Control of
Dynamic Systems*, 7th ed., Pearson. Proportional-derivative attitude control and
the second-order response parameters quoted by :meth:`LoopParameters.nominal_modes`.

Butcher, J. C. (2016), *Numerical Methods for Ordinary Differential Equations*,
3rd ed., Wiley. Order of the explicit Euler method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .traces import Trace

#: Sample interval of the shipped benchmark, in seconds.
DEFAULT_DT = 0.005
#: Simulated horizon of the shipped benchmark, in seconds.
DEFAULT_HORIZON = 2.0


@dataclass(frozen=True)
class LoopParameters:
    """Declared constants of the benchmark loop. Not identified from any vehicle.

    Attributes
    ----------
    pitch_damping:
        ``M_q`` [1/s]. Negative for an aerodynamically damped airframe.
    control_effectiveness:
        ``M_delta`` [deg/s^2 per deg of deflection]. Strictly positive.
    actuator_tau:
        First-order actuator time constant [s]. Strictly positive.
    rate_limit:
        Actuator slew-rate limit [deg/s]. Strictly positive.
    deflection_limit:
        Actuator deflection limit [deg]. Strictly positive.
    kp:
        Proportional attitude gain [deg of command per deg of error].
    kd:
        Rate-feedback gain [deg of command per deg/s].
    """

    pitch_damping: float = -1.2
    control_effectiveness: float = 18.0
    actuator_tau: float = 0.05
    rate_limit: float = 120.0
    deflection_limit: float = 20.0
    kp: float = 1.4
    kd: float = 0.35

    def __post_init__(self) -> None:
        checks = {
            "control_effectiveness": self.control_effectiveness,
            "actuator_tau": self.actuator_tau,
            "rate_limit": self.rate_limit,
            "deflection_limit": self.deflection_limit,
            "kp": self.kp,
        }
        for name, value in checks.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and strictly positive, got {value}")
        for name, value in (("pitch_damping", self.pitch_damping), ("kd", self.kd)):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if self.pitch_damping > 0.0:
            raise ValueError(
                f"pitch_damping (M_q) must be non-positive for a damped airframe, "
                f"got {self.pitch_damping}"
            )

    def nominal_modes(self) -> tuple[float, float]:
        """Undamped natural frequency [rad/s] and damping ratio of the linear loop.

        From the unsaturated, zero-lag closed loop
        ``theta_ddot = M_q q + M_delta (Kp e - Kd q)``, which is
        ``theta_ddot + (M_delta Kd - M_q) theta_dot + M_delta Kp theta =
        M_delta Kp theta_ref``, so ``wn = sqrt(M_delta Kp)`` [rad/s] and
        ``zeta = (M_delta Kd - M_q) / (2 wn)``. Valid only while neither clip
        binds and while the actuator lag is negligible against ``1/wn``; it is a
        design sanity figure, not a prediction of the simulated response.
        """
        wn = math.sqrt(self.control_effectiveness * self.kp)
        zeta = (self.control_effectiveness * self.kd - self.pitch_damping) / (2.0 * wn)
        return wn, zeta


@dataclass(frozen=True)
class LoopInput:
    """One point of the search space: the falsifier's decision variables.

    Attributes
    ----------
    step_amplitude:
        Commanded attitude step [deg], applied at ``t = 0`` from rest.
    kp_factor:
        Multiplier on the nominal ``Kp``. Dimensionless, strictly positive.
    kd_factor:
        Multiplier on the nominal ``Kd``. Dimensionless, strictly positive.
    tau_factor:
        Multiplier on the nominal actuator time constant. Dimensionless,
        strictly positive.
    gust_amplitude:
        Sinusoidal gust amplitude in angular acceleration [deg/s^2],
        non-negative.
    gust_frequency:
        Gust frequency [Hz], non-negative.
    """

    step_amplitude: float
    kp_factor: float
    kd_factor: float
    tau_factor: float
    gust_amplitude: float
    gust_frequency: float

    #: Order of the decision variables in the flat vector form.
    FIELDS: tuple[str, ...] = (
        "step_amplitude",
        "kp_factor",
        "kd_factor",
        "tau_factor",
        "gust_amplitude",
        "gust_frequency",
    )

    def __post_init__(self) -> None:
        values = self.to_array()
        if not np.all(np.isfinite(values)):
            raise ValueError(f"all loop inputs must be finite, got {values.tolist()}")
        for name in ("kp_factor", "kd_factor", "tau_factor"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be strictly positive, got {getattr(self, name)}")
        for name in ("gust_amplitude", "gust_frequency"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")

    def to_array(self) -> np.ndarray:
        """Flat vector in :data:`FIELDS` order, shape ``(6,)``."""
        return np.array([getattr(self, f) for f in self.FIELDS], dtype=float)

    @classmethod
    def from_array(cls, vector: np.ndarray) -> LoopInput:
        """Build from a flat vector in :data:`FIELDS` order.

        Raises
        ------
        ValueError
            If ``vector`` does not have exactly ``len(FIELDS)`` entries.
        """
        arr = np.asarray(vector, dtype=float).ravel()
        if arr.size != len(cls.FIELDS):
            raise ValueError(f"expected {len(cls.FIELDS)} decision variables, got {arr.size}")
        return cls(*(float(v) for v in arr))


def simulate(
    loop_input: LoopInput,
    parameters: LoopParameters | None = None,
    dt: float = DEFAULT_DT,
    horizon: float = DEFAULT_HORIZON,
) -> Trace:
    """Simulate the closed loop and return its trace.

    Parameters
    ----------
    loop_input:
        The decision variables.
    parameters:
        Declared loop constants; :class:`LoopParameters` defaults if omitted.
    dt:
        Euler step [s]. Must be strictly positive and no larger than the
        effective actuator time constant ``actuator_tau * tau_factor``.
    horizon:
        Simulated duration [s]. Must be at least ``2 * dt``.

    Returns
    -------
    Trace
        Signals, all at the same ``T = floor(horizon/dt) + 1`` samples:

        ======== =========================================== ==========
        name     meaning                                     unit
        ======== =========================================== ==========
        theta    attitude                                    deg
        q        attitude rate                               deg/s
        delta    actuator deflection                         deg
        cmd      commanded deflection, after its clip        deg
        error    ``theta_ref - theta``                       deg
        over     ``theta - step_amplitude``, overshoot above target  deg
        ======== =========================================== ==========

    Raises
    ------
    TypeError
        If ``loop_input`` or ``parameters`` is of the wrong type.
    ValueError
        On a non-positive ``dt``, a ``dt`` above the effective actuator time
        constant (where explicit Euler stops being meaningful), or a horizon
        shorter than two steps.

    Notes
    -----
    Deterministic: no random number is drawn. The same ``loop_input`` always
    yields the same trace, which is what makes the seeded benchmark instances
    reproducible.
    """
    if not isinstance(loop_input, LoopInput):
        raise TypeError(f"loop_input must be a LoopInput, got {type(loop_input)!r}")
    if parameters is None:
        parameters = LoopParameters()
    if not isinstance(parameters, LoopParameters):
        raise TypeError(f"parameters must be a LoopParameters, got {type(parameters)!r}")
    dt = float(dt)
    horizon = float(horizon)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and strictly positive, got {dt}")
    if not math.isfinite(horizon) or horizon < 2.0 * dt:
        raise ValueError(f"horizon must be at least 2*dt = {2.0 * dt:g} s, got {horizon}")

    tau = parameters.actuator_tau * loop_input.tau_factor
    if dt > tau:
        raise ValueError(
            f"dt = {dt:g} s exceeds the effective actuator time constant tau = {tau:g} s; "
            "explicit Euler on the actuator lag is not meaningful there"
        )

    n = int(math.floor(horizon / dt)) + 1
    mq = parameters.pitch_damping
    mdelta = parameters.control_effectiveness
    rate_limit = parameters.rate_limit
    deflection_limit = parameters.deflection_limit
    kp = parameters.kp * loop_input.kp_factor
    kd = parameters.kd * loop_input.kd_factor
    ref = loop_input.step_amplitude
    gust_amp = loop_input.gust_amplitude
    gust_omega = 2.0 * math.pi * loop_input.gust_frequency

    theta_out = np.empty(n)
    q_out = np.empty(n)
    delta_out = np.empty(n)
    cmd_out = np.empty(n)

    theta = 0.0
    q = 0.0
    delta = 0.0
    for k in range(n):
        error = ref - theta
        cmd = kp * error - kd * q
        if cmd > deflection_limit:
            cmd = deflection_limit
        elif cmd < -deflection_limit:
            cmd = -deflection_limit
        theta_out[k] = theta
        q_out[k] = q
        delta_out[k] = delta
        cmd_out[k] = cmd
        if k == n - 1:
            break
        gust = gust_amp * math.sin(gust_omega * (k * dt))
        q_next = q + dt * (mq * q + mdelta * delta + gust)
        theta_next = theta + dt * q
        slew = (cmd - delta) / tau
        if slew > rate_limit:
            slew = rate_limit
        elif slew < -rate_limit:
            slew = -rate_limit
        delta_next = delta + dt * slew
        if delta_next > deflection_limit:
            delta_next = deflection_limit
        elif delta_next < -deflection_limit:
            delta_next = -deflection_limit
        theta, q, delta = theta_next, q_next, delta_next

    times = np.arange(n) * dt
    return Trace(
        times,
        {
            "theta": theta_out,
            "q": q_out,
            "delta": delta_out,
            "cmd": cmd_out,
            "error": ref - theta_out,
            "over": theta_out - ref,
        },
    )


def simulate_linear_zoh(
    loop_input: LoopInput,
    parameters: LoopParameters | None = None,
    dt: float = DEFAULT_DT,
    horizon: float = DEFAULT_HORIZON,
) -> Trace:
    """Exact zero-order-hold solution of the **unsaturated** loop, for validation.

    Drops both clips and solves ``x_dot = A x + b`` exactly over each sample by
    the matrix exponential, which is the analytic reference
    ``validation/validate_simulator.py`` measures :func:`simulate` against. The
    gust is held constant over each sample, which is itself a zero-order-hold
    approximation of the sinusoid and is why the comparison in that script is
    run at ``gust_amplitude = 0`` for the exact leg.

    The state is ``[theta, q, delta]`` and

    .. math::
        A = \\begin{bmatrix} 0 & 1 & 0 \\\\
                             0 & M_q & M_\\delta \\\\
                             -K_p/\\tau & -K_d/\\tau & -1/\\tau \\end{bmatrix},
        \\quad b = \\begin{bmatrix} 0 \\\\ w \\\\ K_p \\theta_r/\\tau \\end{bmatrix}

    Returns the same signal names as :func:`simulate`. Valid only while neither
    clip would have bound; the caller is responsible for staying in that regime
    and ``validate_simulator.py`` checks that it did.
    """
    from scipy.linalg import expm

    if parameters is None:
        parameters = LoopParameters()
    dt = float(dt)
    n = int(math.floor(float(horizon) / dt)) + 1
    tau = parameters.actuator_tau * loop_input.tau_factor
    kp = parameters.kp * loop_input.kp_factor
    kd = parameters.kd * loop_input.kd_factor
    a = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, parameters.pitch_damping, parameters.control_effectiveness],
            [-kp / tau, -kd / tau, -1.0 / tau],
        ]
    )
    states = np.empty((n, 3))
    x = np.zeros(3)
    states[0] = x
    omega = 2.0 * math.pi * loop_input.gust_frequency
    for k in range(n - 1):
        gust = loop_input.gust_amplitude * math.sin(omega * (k * dt))
        b = np.array([0.0, gust, kp * loop_input.step_amplitude / tau])
        block = np.zeros((4, 4))
        block[:3, :3] = a
        block[:3, 3] = b
        exp_block = expm(block * dt)
        x = exp_block[:3, :3] @ x + exp_block[:3, 3]
        states[k + 1] = x
    theta, q, delta = states[:, 0], states[:, 1], states[:, 2]
    cmd = kp * (loop_input.step_amplitude - theta) - kd * q
    return Trace(
        np.arange(n) * dt,
        {
            "theta": theta,
            "q": q,
            "delta": delta,
            "cmd": cmd,
            "error": loop_input.step_amplitude - theta,
            "over": theta - loop_input.step_amplitude,
        },
    )
