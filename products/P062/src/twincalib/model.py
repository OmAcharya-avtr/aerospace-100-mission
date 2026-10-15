"""Twin models: the physics whose parameters are to be calibrated.

Two models are provided.

``LinearNodeModel``
    One lumped thermal node with linear conduction to a fixed sink. It has a
    closed-form solution and closed-form parameter sensitivities, and it is
    structurally non-identifiable from a single temperature record in a way that
    can be written down exactly. It exists so that the finite-difference
    sensitivity machinery and the identifiability analysis have a known answer
    to be checked against, and it is cheap enough to use in coverage studies.

``ThermalTwinModel``
    Two lumped nodes -- an avionics unit conductively coupled to a radiator
    panel that radiates to deep space -- with six parameters. This is the twin
    the package is built around.

Governing equations, units, assumptions and validity ranges are stated on each
class. Both models are integrated with a fixed-step classical Runge-Kutta
scheme (Butcher 2016, section 23), and the observation times must be integer
multiples of the integration step.

References
----------
Gilmore, D. G. (ed.), *Spacecraft Thermal Control Handbook, Volume I:
Fundamentals and Environments*, 2nd ed., AIAA, 2002. Chapter 1 gives the lumped
thermal-network formulation used here; Chapter 4 covers radiator sizing.

Incropera, F. P. and DeWitt, D. P., *Fundamentals of Heat and Mass Transfer*,
Wiley. The lumped-capacitance method and the Biot-number validity criterion
Bi = h L / k < 0.1 (Chapter 5).

Butcher, J. C., *Numerical Methods for Ordinary Differential Equations*, 3rd
ed., Wiley, 2016. The classical fourth-order Runge-Kutta scheme and its order.

Madhusudana, C. V., *Thermal Contact Conductance*, 2nd ed., Springer, 2014. The
temperature dependence of interface conductance, which is the physical basis of
the declared model discrepancy in :mod:`twincalib.scenarios`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .constants import T_DEEP_SPACE

__all__ = [
    "Inputs",
    "LinearNodeModel",
    "ThermalTwinModel",
    "TwinModel",
    "rk4_steps",
]


@dataclass(frozen=True)
class Inputs:
    """Known, measured exogenous inputs on the integration grid.

    Attributes
    ----------
    times:
        Integration grid, seconds, strictly increasing and uniformly spaced.
    dissipation:
        Electrical dissipation of the unit, W, one value per entry of ``times``.
        Held constant over each integration step (zero-order hold), which is how
        a commanded duty cycle actually behaves.
    sun:
        Illumination fraction in [0, 1], dimensionless, one value per entry of
        ``times``. 1 is full sun on the radiator, 0 is eclipse. Zero-order hold.
    """

    times: np.ndarray
    dissipation: np.ndarray
    sun: np.ndarray

    def __post_init__(self) -> None:
        t = np.asarray(self.times, dtype=float)
        if t.ndim != 1 or t.size < 2:
            raise ValueError("times must be a 1-D array with at least 2 entries")
        dt = np.diff(t)
        if np.any(dt <= 0.0):
            raise ValueError("times must be strictly increasing")
        if not np.allclose(dt, dt[0], rtol=1e-9, atol=0.0):
            raise ValueError("times must be uniformly spaced")
        for name in ("dissipation", "sun"):
            v = np.asarray(getattr(self, name), dtype=float)
            if v.shape != t.shape:
                raise ValueError(f"{name} must have the same shape as times, got {v.shape}")
        s = np.asarray(self.sun, dtype=float)
        if np.any(s < 0.0) or np.any(s > 1.0):
            raise ValueError("sun must lie in [0, 1]")
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "dissipation", np.asarray(self.dissipation, dtype=float))
        object.__setattr__(self, "sun", s)

    @property
    def dt(self) -> float:
        """Integration step, seconds."""
        return float(self.times[1] - self.times[0])

    @property
    def n_steps(self) -> int:
        """Number of integration steps, dimensionless."""
        return int(self.times.size - 1)


def rk4_steps(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Integrate ``dy/dt = rhs(t, y)`` on a uniform grid with classical RK4.

    Parameters
    ----------
    rhs:
        Right-hand side. ``t`` in seconds, ``y`` in the state's own units.
    y0:
        Initial state, shape ``(n,)``.
    times:
        Uniform grid, seconds, shape ``(m,)``.

    Returns
    -------
    numpy.ndarray
        States on the grid, shape ``(m, n)``.

    Notes
    -----
    Fourth-order accurate and explicit (Butcher 2016, section 23). The scheme is
    conditionally stable: for the shipped designs the shortest time constant is
    about 270 s and the step is 30 s, giving a step-to-time-constant ratio of
    0.11. The measured convergence order is in
    ``validation/validate_model_output.txt``.
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(y0, dtype=float)
    out = np.empty((t.size, y.size), dtype=float)
    out[0] = y
    for i in range(t.size - 1):
        h = t[i + 1] - t[i]
        k1 = rhs(t[i], y)
        k2 = rhs(t[i] + 0.5 * h, y + 0.5 * h * k1)
        k3 = rhs(t[i] + 0.5 * h, y + 0.5 * h * k2)
        k4 = rhs(t[i] + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[i + 1] = y
    return out


class TwinModel:
    """Interface shared by the twin models.

    Subclasses declare ``parameter_names``, ``state_names`` and
    ``observable_names``, and implement :meth:`simulate`.
    """

    parameter_names: tuple[str, ...] = ()
    parameter_units: tuple[str, ...] = ()
    state_names: tuple[str, ...] = ()
    observable_names: tuple[str, ...] = ()

    @property
    def n_parameters(self) -> int:
        """Number of model parameters, dimensionless."""
        return len(self.parameter_names)

    def check_parameters(self, theta: Sequence[float]) -> np.ndarray:
        """Validate and return a parameter vector as a float array.

        Raises
        ------
        ValueError
            If the length is wrong or any entry is non-positive. Every parameter
            of both shipped models is a physically positive quantity -- a heat
            capacity, a conductance, a radiative coefficient or a heat load --
            so a non-positive value is rejected rather than silently integrated.
        """
        v = np.asarray(theta, dtype=float).ravel()
        if v.size != self.n_parameters:
            raise ValueError(
                f"expected {self.n_parameters} parameters "
                f"{self.parameter_names}, got {v.size}"
            )
        if not np.all(np.isfinite(v)):
            raise ValueError("parameters must all be finite")
        bad = [n for n, x in zip(self.parameter_names, v, strict=True) if x <= 0.0]
        if bad:
            raise ValueError(f"these parameters must be strictly positive: {bad}")
        return v

    def simulate(
        self, theta: Sequence[float], inputs: Inputs, initial_state: Sequence[float]
    ) -> np.ndarray:
        raise NotImplementedError

    def observe(
        self,
        theta: Sequence[float],
        inputs: Inputs,
        initial_state: Sequence[float],
        channels: Sequence[int],
        sample_stride: int = 1,
    ) -> np.ndarray:
        """Simulate and return the observed channels, stacked channel-major.

        Parameters
        ----------
        channels:
            Indices into ``state_names`` that are measured.
        sample_stride:
            Keep every ``sample_stride``-th point of the integration grid. The
            observation grid is therefore a subset of the integration grid, so
            no interpolation is ever done.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(channels) * n_samples,)`` in K, channel 0 first.
        """
        if sample_stride < 1:
            raise ValueError("sample_stride must be >= 1")
        ch = [int(c) for c in channels]
        if not ch:
            raise ValueError("at least one channel must be observed")
        if any(c < 0 or c >= len(self.state_names) for c in ch):
            raise ValueError(f"channels must index {self.state_names}")
        states = self.simulate(theta, inputs, initial_state)
        kept = states[::sample_stride, :]
        return np.concatenate([kept[:, c] for c in ch])


class LinearNodeModel(TwinModel):
    """One lumped node with linear conduction to a sink of fixed temperature.

    Governing equation
    ------------------
    ``C dT/dt = q - k (T - T_sink)``

    with ``T`` in K, ``C`` in J/K, ``k`` in W/K, ``q`` in W and ``T_sink`` in K.
    Source: the lumped-capacitance energy balance, Incropera and DeWitt Chapter
    5, equation 5.2, with a constant internal generation term.

    Closed-form solution (``q``, ``k``, ``C``, ``T_sink`` constant)::

        T(t) = T_sink + q/k + (T0 - T_sink - q/k) * exp(-k t / C)

    Assumptions: spatially uniform node temperature (Biot number below 0.1),
    constant properties, no radiation, no phase change, sink temperature known
    exactly. Validity range: any ``T`` for which the material properties are
    constant; the shipped design stays within 250-340 K.

    Structural non-identifiability, exactly
    ---------------------------------------
    The solution depends on the three parameters only through ``q/k`` and
    ``k/C``. Scaling ``(C, k, q)`` by a common factor leaves both invariant, so
    ``T(t)`` is unchanged for every ``t``. The sensitivity matrix in
    log-parameter coordinates therefore has an exact null direction
    ``(1, 1, 1)/sqrt(3)`` and rank 2 out of 3, for any observation design, any
    ``T0`` and any sink temperature. This is the known answer the identifiability
    module is tested against (``validation/validate_identifiability.py``).
    """

    parameter_names = ("C", "k", "q")
    parameter_units = ("J/K", "W/K", "W")
    state_names = ("T",)
    observable_names = ("T",)

    def __init__(self, t_sink: float = 280.0) -> None:
        """Create the model.

        Parameters
        ----------
        t_sink:
            Sink temperature, K. Treated as known, not calibrated.
        """
        if not np.isfinite(t_sink) or t_sink <= 0.0:
            raise ValueError("t_sink must be a positive, finite temperature in K")
        self.t_sink = float(t_sink)

    def simulate(
        self, theta: Sequence[float], inputs: Inputs, initial_state: Sequence[float]
    ) -> np.ndarray:
        """Integrate the node. Returns shape ``(len(inputs.times), 1)`` in K."""
        cap, cond, load = self.check_parameters(theta)
        y0 = np.asarray(initial_state, dtype=float).ravel()
        if y0.size != 1:
            raise ValueError("initial_state must have exactly 1 entry (T in K)")
        t_sink = self.t_sink

        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            return np.array([(load - cond * (y[0] - t_sink)) / cap])

        return rk4_steps(rhs, y0, inputs.times)

    def analytic_solution(
        self, theta: Sequence[float], times: np.ndarray, t0_state: float
    ) -> np.ndarray:
        """Closed-form ``T(t)`` in K, shape ``(len(times),)``.

        Exact for constant ``q`` and ``k``; used as the reference the integrator
        is measured against.
        """
        cap, cond, load = self.check_parameters(theta)
        t = np.asarray(times, dtype=float)
        steady = self.t_sink + load / cond
        return steady + (float(t0_state) - steady) * np.exp(-cond * (t - t[0]) / cap)

    def analytic_sensitivity(
        self, theta: Sequence[float], times: np.ndarray, t0_state: float
    ) -> np.ndarray:
        """Closed-form ``dT/dtheta``, shape ``(len(times), 3)``.

        Units: column ``C`` in K per (J/K), column ``k`` in K per (W/K), column
        ``q`` in K per W. Differentiating the closed form above with
        ``a = q/k``, ``lam = k/C`` and ``d0 = T0 - T_sink - a``::

            dT/dq = (1 - E)/k
            dT/dk = -(q/k^2)(1 - E) + d0 * E * (t/C)   [sign: E' wrt k]
            dT/dC = -d0 * E * (k t / C^2)

        where ``E = exp(-lam (t - t0))``. Derived in
        ``validation/validate_sensitivity.py``, which checks it against central
        finite differences of :meth:`analytic_solution` and of :meth:`simulate`.
        """
        cap, cond, load = self.check_parameters(theta)
        t = np.asarray(times, dtype=float) - float(times[0])
        e = np.exp(-cond * t / cap)
        a = load / cond
        d0 = float(t0_state) - self.t_sink - a
        d_q = (1.0 - e) / cond
        d_k = -(load / cond**2) * (1.0 - e) - d0 * e * (t / cap)
        d_c = d0 * e * (cond * t / cap**2)
        return np.column_stack([d_c, d_k, d_q])


class ThermalTwinModel(TwinModel):
    """Two lumped nodes: an avionics unit on a radiator panel.

    Governing equations
    -------------------
    ::

        C1 dT1/dt = P_dis(t) - k12 (T1 - T2)
        C2 dT2/dt = k12 (T1 - T2) + q_sun s(t) + q_par
                    - h_rad (T2**4 - T_space**4)

    Symbols, units and sources:

    ==========  ========  ====================================================
    symbol      unit      meaning
    ==========  ========  ====================================================
    ``T1``      K         unit (node 1) temperature
    ``T2``      K         radiator panel (node 2) temperature
    ``C1``      J/K       unit heat capacity, ``m c_p``
    ``C2``      J/K       panel heat capacity
    ``k12``     W/K       unit-to-panel interface conductance
    ``h_rad``   W/K^4     ``eps * sigma * A_rad``, grey-body radiator group
    ``q_sun``   W         absorbed solar flux at full illumination, ``alpha S A``
    ``q_par``   W         parasitic heat load into the panel from the rest of
                          the bus, constant
    ``P_dis``   W         measured electrical dissipation of the unit (input)
    ``s``       -         illumination fraction in [0, 1] (input)
    ==========  ========  ====================================================

    The conduction term is Fourier's law in lumped-network form and the
    radiation term is the grey-body Stefan-Boltzmann law with
    ``sigma = 5.670374419e-8 W/m^2/K^4``; both as set out in Gilmore 2002
    Chapter 1. The sink temperature is the cosmic microwave background,
    ``T_space = 2.72548 K`` (Fixsen 2009).

    Assumptions
    -----------
    1. Each node is isothermal (lumped capacitance, Biot number below 0.1).
    2. Constant ``c_p``, emissivity and absorptivity over the temperature range.
    3. The radiator sees only deep space -- no Earth infrared, no albedo, no
       self-viewing. A real radiator in low Earth orbit sees all three, so
       ``q_par`` in a real fit absorbs them and is not purely parasitic.
    4. No conduction path from either node to anything else.
    5. ``P_dis`` and ``s`` are known exactly; measurement error is on ``T``
       only.

    Validity range
    --------------
    200-350 K for the shipped designs. Below about 100 K the constant-``c_p``
    assumption fails badly for aluminium; above about 400 K the constant-
    emissivity assumption and the no-phase-change assumption become doubtful for
    normal spacecraft coatings. The model has no provision for heaters,
    thermostats, louvres, heat pipes or phase-change material.

    Structural non-identifiability, exactly
    ---------------------------------------
    ``q_sun`` and ``q_par`` enter only through ``q_sun s(t) + q_par``. If
    ``s(t)`` is constant at ``s0`` over the whole record then only the
    combination ``q_sun s0 + q_par`` is identifiable, and the exact null
    direction of the sensitivity matrix in log-parameter coordinates is
    proportional to ``(q_par, -s0 q_sun)`` in those two coordinates and zero in
    the other four. That is the ``full-sun`` design in
    :mod:`twincalib.scenarios`, and it is the deliberately non-identifiable
    configuration the package is required to report as such.
    """

    parameter_names = ("C1", "C2", "k12", "h_rad", "q_sun", "q_par")
    parameter_units = ("J/K", "J/K", "W/K", "W/K^4", "W", "W")
    state_names = ("T1", "T2")
    observable_names = ("T1", "T2")

    def __init__(
        self,
        t_space: float = T_DEEP_SPACE,
        conductance_slope: float = 0.0,
        t_reference: float = 290.0,
    ) -> None:
        """Create the model.

        Parameters
        ----------
        t_space:
            Radiative sink temperature, K.
        conductance_slope:
            Dimensionless slope ``beta`` of a declared *discrepancy* term that
            makes the interface conductance temperature dependent::

                k12_eff = k12 * (1 + beta * (T1 - t_reference) / 100 K)

            ``beta = 0`` is the calibration model: the form the fitter assumes.
            A non-zero ``beta`` is used only to generate truth data that the
            calibration model cannot represent at any parameter value, so that
            the residual model has something real to learn. Interface
            conductance does vary with temperature in reality, through the
            temperature dependence of the bulk conductivity and of the contact
            pressure (Madhusudana 2014, Chapter 4); the linear form and the
            magnitude here are declared, not measured.
        t_reference:
            Reference temperature of the discrepancy term, K.
        """
        if not np.isfinite(t_space) or t_space < 0.0:
            raise ValueError("t_space must be a finite, non-negative temperature in K")
        if not np.isfinite(conductance_slope):
            raise ValueError("conductance_slope must be finite")
        if not np.isfinite(t_reference) or t_reference <= 0.0:
            raise ValueError("t_reference must be a positive, finite temperature in K")
        self.t_space = float(t_space)
        self.conductance_slope = float(conductance_slope)
        self.t_reference = float(t_reference)

    @property
    def has_discrepancy(self) -> bool:
        """True when a non-zero conductance slope makes this a truth model."""
        return self.conductance_slope != 0.0

    def simulate(
        self, theta: Sequence[float], inputs: Inputs, initial_state: Sequence[float]
    ) -> np.ndarray:
        """Integrate both nodes. Returns shape ``(len(inputs.times), 2)`` in K.

        The exogenous inputs are held over each step at their left-hand value
        (zero-order hold), so the right-hand side is piecewise constant in ``t``
        and the Runge-Kutta stages inside a step all see the same input.
        """
        c1, c2, k12, h_rad, q_sun, q_par = self.check_parameters(theta)
        y0 = np.asarray(initial_state, dtype=float).ravel()
        if y0.size != 2:
            raise ValueError("initial_state must have exactly 2 entries (T1, T2 in K)")
        times = inputs.times
        p_dis = inputs.dissipation
        sun = inputs.sun
        t_space4 = self.t_space**4
        beta = self.conductance_slope
        t_ref = self.t_reference
        values_for_tau = (c1, c2, k12, h_rad, q_sun, q_par)
        out = np.empty((times.size, 2), dtype=float)
        out[0] = y0
        t1, t2 = float(y0[0]), float(y0[1])
        # A fixed-step explicit scheme can diverge if the design's step is long
        # against the fastest time constant. The overflow that announces that is
        # expected, is detected after the loop and is turned into a ValueError
        # naming the stability limit, so the warning itself is silenced here
        # rather than printed from library code.
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(times.size - 1):
                h = float(times[i + 1] - times[i])
                p_i = float(p_dis[i])
                s_i = float(sun[i])
                forcing = q_sun * s_i + q_par

                def f(
                    a: float, b: float, p_i: float = p_i, forcing: float = forcing
                ) -> tuple[float, float]:
                    k_eff = k12 * (1.0 + beta * (a - t_ref) / 100.0) if beta else k12
                    flow = k_eff * (a - b)
                    return (
                        (p_i - flow) / c1,
                        (flow + forcing - h_rad * (b**4 - t_space4)) / c2,
                    )

                a1, b1 = f(t1, t2)
                a2, b2 = f(t1 + 0.5 * h * a1, t2 + 0.5 * h * b1)
                a3, b3 = f(t1 + 0.5 * h * a2, t2 + 0.5 * h * b2)
                a4, b4 = f(t1 + h * a3, t2 + h * b3)
                t1 = t1 + (h / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
                t2 = t2 + (h / 6.0) * (b1 + 2.0 * b2 + 2.0 * b3 + b4)
                out[i + 1, 0] = t1
                out[i + 1, 1] = t2
        if not np.all(np.isfinite(out)):
            tau_fast = self.time_constants(values_for_tau, float(y0[1]))[0]
            raise ValueError(
                "the integration diverged: the fixed-step explicit scheme is "
                f"unstable at dt = {float(times[1] - times[0]):g} s for a fastest "
                f"time constant of about {tau_fast:.3g} s. Reduce the design's dt "
                "and raise sample_stride to keep the same observation interval."
            )
        return out

    def steady_state(
        self, theta: Sequence[float], dissipation: float, sun: float
    ) -> tuple[float, float]:
        """Algebraic equilibrium ``(T1, T2)`` in K for constant inputs.

        Setting both derivatives to zero gives, exactly::

            h_rad (T2**4 - T_space**4) = P_dis + q_sun s + q_par
            T1 = T2 + P_dis / k12

        so ``T2`` is a fourth root and ``T1`` follows. Used as an independent
        reference for the integrator in ``validation/validate_model.py``. Raises
        ``ValueError`` if the total heat input is non-positive, where no
        equilibrium above the sink temperature exists.
        """
        _c1, _c2, k12, h_rad, q_sun, q_par = self.check_parameters(theta)
        if not 0.0 <= sun <= 1.0:
            raise ValueError("sun must lie in [0, 1]")
        total = float(dissipation) + q_sun * float(sun) + q_par
        if total <= 0.0:
            raise ValueError("total heat input must be positive for an equilibrium")
        t2 = (total / h_rad + self.t_space**4) ** 0.25
        return (t2 + float(dissipation) / k12, t2)

    def time_constants(self, theta: Sequence[float], t2_about: float) -> tuple[float, float]:
        """Linearised time constants ``(tau_fast, tau_slow)`` in seconds.

        Linearising the radiative term about ``T2 = t2_about`` gives an
        effective radiative conductance ``h_lin = 4 h_rad T2**3`` in W/K. The
        two-node network then has eigenvalues of the 2x2 matrix

        ``A = [[-k12/C1, k12/C1], [k12/C2, -(k12 + h_lin)/C2]]``

        and the returned pair is ``-1/lambda`` for each, smallest first. Valid
        only while the excursion about ``t2_about`` is small enough that
        ``4 h_rad T2**3`` is a good approximation of ``d/dT2 (h_rad T2**4)``;
        a 10 K excursion about 280 K changes ``h_lin`` by about 11 per cent.
        """
        c1, c2, k12, h_rad, _q_sun, _q_par = self.check_parameters(theta)
        if not np.isfinite(t2_about) or t2_about <= 0.0:
            raise ValueError("t2_about must be a positive, finite temperature in K")
        h_lin = 4.0 * h_rad * float(t2_about) ** 3
        a = np.array([[-k12 / c1, k12 / c1], [k12 / c2, -(k12 + h_lin) / c2]])
        lam = np.linalg.eigvals(a).real
        taus = np.sort(-1.0 / lam)
        return (float(taus[0]), float(taus[1]))
