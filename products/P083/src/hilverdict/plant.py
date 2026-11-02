"""The declared plant, its controller, and the divergence metric the verdict uses.

What is declared here
---------------------
A hardware-in-the-loop verdict is only ever a statement about a declared
plant. This module holds that declaration in one place so that it can be
argued with: the continuous dynamics, the controller, the sample period, the
actuator limit, and the weighted norm in which a trajectory difference is
called large or small.

The dynamics
------------
A two-state longitudinal short-period model with a cubic pitching moment, the
standard textbook shape for an airframe whose static stability falls away with
incidence (Etkin & Reid 1996, *Dynamics of Flight: Stability and Control*, 3rd
edition, chapter 6, for the linear short-period form; Stengel 2004, *Flight
Dynamics*, Princeton, chapter 5, for the cubic pitching-moment extension). The
coefficients are **declared, not measured**: they are a fiction with an
engineering shape, chosen so the closed loop is lightly damped at a realistic
short-period frequency, and no number here is attributed to any real airframe.

```
alpha_dot = z_a alpha + q + z_d u                                        (1)
q_dot     = m_a alpha + m_a3 alpha^3 + m_q q + m_d u                     (2)
```

Units: ``alpha`` rad, ``q`` rad/s, ``u`` rad (an equivalent elevator
deflection), ``z_a``, ``z_d``, ``m_q`` in 1/s, ``m_a``, ``m_d`` in 1/s^2 and
``m_a3`` in 1/(s^2 rad^2). Validity: ``|alpha| <= ALPHA_VALID_RAD`` and
``|u| <= u_max``; outside either, the cubic term is an extrapolation and the
actuator model is wrong, and this package refuses to certify rather than
extrapolating (see :mod:`hilverdict.certificate`).

The controller
--------------
A trim feedforward plus a constant-gain regulator, sampled at ``period_s`` and
held over the period (zero-order hold):

```
u_k = u_trim(alpha_cmd) + K (x_k - x_trim(alpha_cmd)),  saturated to +-u_max  (3)
```

The trim is solved in closed form from (1)-(2) at ``alpha_dot = q_dot = 0``:

```
u_trim  = -( alpha_c (m_a - m_q z_a) + m_a3 alpha_c^3 ) / (m_d - m_q z_d)     (4)
q_trim  = -z_a alpha_c - z_d u_trim                                          (5)
```

Equation (4) is exact for the declared cubic plant, which is why the
feedforward is not a linearisation and does not degrade with amplitude.

The metric
----------
A trajectory difference in ``(alpha, q)`` cannot be measured by a plain
Euclidean norm, because the two components carry different units. The
divergence metric is therefore a weighted norm whose weights are the
**declared engineering tolerances**:

```
D = max over sample instants of  sqrt( (d_alpha/alpha_tol)^2 + (d_q/q_tol)^2 ) (6)
```

so ``D >= 1`` means "this difference exceeded the declared tolerance" and the
verdict threshold is the dimensionless number 1. The weights are a declaration
about what the hardware-in-the-loop result is for, not a property of the
plant, and every bound in this package depends on them: a tighter
``alpha_tol`` relative to ``q_tol`` makes the metric more anisotropic, the
closed-loop map more non-normal in that metric, and the analytic bound looser.
That dependence is measured in ``validation/validate_certificate.py`` rather
than left as a remark.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

__all__ = [
    "ALPHA_VALID_RAD",
    "DEFAULT_METRIC",
    "SHORT_PERIOD",
    "Controller",
    "DivergenceMetric",
    "LoopModel",
    "Plant",
    "SamplingLaw",
    "zoh_matrices",
]

#: Declared validity limit on incidence, rad. The cubic pitching moment of
#: equation (2) is a declared shape fitted to nothing; beyond this amplitude it
#: is an extrapolation and no claim in this package applies.
ALPHA_VALID_RAD = np.deg2rad(10.0)


@dataclass(frozen=True)
class Plant:
    """The declared continuous plant of equations (1)-(2).

    Attributes
    ----------
    z_alpha, z_delta
        Lift-curve and control terms of equation (1), 1/s.
    m_alpha, m_alpha3, m_q, m_delta
        Pitching-moment terms of equation (2); ``m_alpha`` and ``m_delta`` in
        1/s^2, ``m_alpha3`` in 1/(s^2 rad^2), ``m_q`` in 1/s.
    u_max_rad
        Actuator saturation limit, rad. Symmetric.
    name
        Declared label, carried into every report.
    """

    z_alpha: float
    z_delta: float
    m_alpha: float
    m_alpha3: float
    m_q: float
    m_delta: float
    u_max_rad: float
    name: str = "declared-short-period"

    def derivative(self, state: np.ndarray, control: float) -> np.ndarray:
        """Right-hand side of equations (1)-(2), in rad/s and rad/s^2.

        ``state`` is ``(alpha, q)`` in rad and rad/s and may carry a leading
        batch axis; ``control`` is the applied deflection in rad.

        A single state delegates to :meth:`derivative_scalar`, so there is
        exactly ONE arithmetic path for the quantity the integrator uses and
        the two cannot drift apart. The batch path is a separate expression in
        numpy and agrees with it only to within rounding -- measured at
        3.6e-15 absolute over 20000 random states during this product's build,
        a few units in the last place on a quantity of order 1, which is the
        ordinary consequence of evaluating one formula two ways. The tests
        state that tolerance rather than asserting bit-equality.
        """
        array = np.asarray(state, dtype=float)
        if array.ndim == 1:
            first, second = self.derivative_scalar(
                float(array[0]), float(array[1]), control
            )
            return np.array([first, second], dtype=float)
        alpha = array[..., 0]
        rate = array[..., 1]
        d_alpha = self.z_alpha * alpha + rate + self.z_delta * control
        d_rate = (
            self.m_alpha * alpha
            + self.m_alpha3 * alpha**3
            + self.m_q * rate
            + self.m_delta * control
        )
        return np.stack([d_alpha, d_rate], axis=-1)

    def derivative_scalar(
        self, alpha: float, rate: float, control: float
    ) -> tuple[float, float]:
        """Equations (1)-(2) on plain floats, for the integrator's inner loop.

        This is the single arithmetic path for the right-hand side at one
        state: :meth:`derivative` delegates to it. It exists because the
        integrator evaluates it four times per substep and the array form's
        allocation dominated the cost of every re-simulation in this package --
        measured at 12.4 microseconds per evaluation through the array path
        against 0.83 through this one, a factor of fifteen on the product's
        dominant cost.
        """
        return (
            self.z_alpha * alpha + rate + self.z_delta * control,
            self.m_alpha * alpha
            + self.m_alpha3 * alpha**3
            + self.m_q * rate
            + self.m_delta * control,
        )

    def jacobian(self, alpha: float = 0.0) -> np.ndarray:
        """State Jacobian of (1)-(2) at incidence ``alpha`` (rad), units 1/s.

        The ``(1, 0)`` entry carries the cubic term's contribution
        ``3 m_alpha3 alpha^2``, which is the only way the declared
        nonlinearity enters a linearisation.
        """
        return np.array(
            [
                [self.z_alpha, 1.0],
                [self.m_alpha + 3.0 * self.m_alpha3 * float(alpha) ** 2, self.m_q],
            ],
            dtype=float,
        )

    @property
    def input_matrix(self) -> np.ndarray:
        """Input Jacobian ``dF/du`` of (1)-(2), shape (2, 1), units 1/s and 1/s^2."""
        return np.array([[self.z_delta], [self.m_delta]], dtype=float)

    def trim(self, alpha_command: float) -> tuple[np.ndarray, float]:
        """Exact trim state and control of equations (4)-(5).

        Parameters
        ----------
        alpha_command
            Commanded incidence, rad.

        Returns
        -------
        (state, control)
            ``state`` is ``(alpha_c, q_trim)`` in rad and rad/s; ``control`` is
            ``u_trim`` in rad. The solution is exact for the cubic plant, not a
            linearisation.
        """
        alpha_c = float(alpha_command)
        denominator = self.m_delta - self.m_q * self.z_delta
        if denominator == 0.0:
            raise ValueError(
                "the declared plant has m_delta == m_q * z_delta, so equation (4) "
                "has no solution and the trim feedforward does not exist"
            )
        numerator = alpha_c * (self.m_alpha - self.m_q * self.z_alpha) + (
            self.m_alpha3 * alpha_c**3
        )
        control = -numerator / denominator
        rate = -self.z_alpha * alpha_c - self.z_delta * control
        return np.array([alpha_c, rate], dtype=float), float(control)

    def saturate(self, control: float) -> float:
        """Clip ``control`` (rad) to the declared actuator limit."""
        return float(np.clip(control, -self.u_max_rad, self.u_max_rad))


@dataclass(frozen=True)
class SamplingLaw:
    """The sampled-data timing declaration of the loop.

    Attributes
    ----------
    period_s
        Controller period ``T``, s.
    deadline_s
        Declared deadline within the period, s. An iteration whose measured
        execution time exceeds this has missed; ``deadline_s <= period_s``.
    substeps
        Runge-Kutta substeps per period used by the integrator. Declared, not
        adaptive: an adaptive step would make the reference trajectory depend
        on the perturbation and the counterfactual would stop being a
        counterfactual.
    """

    period_s: float
    deadline_s: float
    substeps: int = 4

    def __post_init__(self) -> None:
        if self.period_s <= 0.0:
            raise ValueError(f"period_s must be positive, got {self.period_s}")
        if not 0.0 < self.deadline_s <= self.period_s:
            raise ValueError(
                "deadline_s must lie in (0, period_s]; got "
                f"deadline_s={self.deadline_s}, period_s={self.period_s}"
            )
        if self.substeps < 1:
            raise ValueError(f"substeps must be at least 1, got {self.substeps}")


@dataclass(frozen=True)
class Controller:
    """The sampled regulator of equation (3).

    Attributes
    ----------
    gain
        Row of feedback gains ``K`` applied to ``(alpha, q)``; units rad/rad
        and rad/(rad/s).
    """

    gain: tuple[float, float]

    @property
    def gain_row(self) -> np.ndarray:
        """``K`` as a ``(1, 2)`` array."""
        return np.array([self.gain], dtype=float)

    def command(self, plant: Plant, state: np.ndarray, alpha_command: float) -> float:
        """Equation (3): trim feedforward plus regulator, saturated, in rad."""
        return self.command_with_saturation(plant, state, alpha_command)[0]

    def command_with_saturation(
        self, plant: Plant, state: np.ndarray, alpha_command: float
    ) -> tuple[float, bool]:
        """Equation (3) together with whether the actuator limit was reached.

        Returns ``(command_rad, saturated)``. The flag is what
        :func:`hilverdict.certificate.certify_run` refuses on: the certificate
        rests on a linear input map, and a clipped command is not one.
        """
        trim_state, trim_control = plant.trim(alpha_command)
        error = np.asarray(state, dtype=float) - trim_state
        raw = trim_control + float((self.gain_row @ error)[0])
        limited = plant.saturate(raw)
        return limited, bool(limited != raw)


@dataclass(frozen=True)
class DivergenceMetric:
    """The weighted norm of equation (6).

    Attributes
    ----------
    alpha_tolerance_rad
        Declared incidence tolerance, rad. A difference of exactly this much
        in incidence alone scores 1.
    rate_tolerance_rad_s
        Declared pitch-rate tolerance, rad/s.
    """

    alpha_tolerance_rad: float
    rate_tolerance_rad_s: float

    def __post_init__(self) -> None:
        if self.alpha_tolerance_rad <= 0.0 or self.rate_tolerance_rad_s <= 0.0:
            raise ValueError(
                "both tolerances must be positive; got "
                f"alpha={self.alpha_tolerance_rad}, rate={self.rate_tolerance_rad_s}"
            )

    @property
    def scaling(self) -> np.ndarray:
        """The diagonal scaling ``S`` of equation (6), shape (2, 2)."""
        return np.diag(
            [1.0 / self.alpha_tolerance_rad, 1.0 / self.rate_tolerance_rad_s]
        )

    @property
    def inverse_scaling(self) -> np.ndarray:
        """``S^-1``, shape (2, 2)."""
        return np.diag([self.alpha_tolerance_rad, self.rate_tolerance_rad_s])

    @property
    def anisotropy(self) -> float:
        """Ratio of the two weights; 1 means an isotropic metric."""
        return float(
            max(
                self.rate_tolerance_rad_s / self.alpha_tolerance_rad,
                self.alpha_tolerance_rad / self.rate_tolerance_rad_s,
            )
        )

    def norm(self, difference: np.ndarray) -> np.ndarray:
        """Equation (6) without the maximum: the per-sample weighted norm.

        ``difference`` has shape ``(..., 2)`` in rad and rad/s; the result has
        shape ``(...)`` and is dimensionless.
        """
        array = np.asarray(difference, dtype=float)
        scaled = array @ self.scaling.T
        return np.sqrt(np.sum(scaled**2, axis=-1))


#: The declared plant every number in this repository is measured against.
#: Coefficients are a declared fiction with an engineering shape: the linear
#: part gives a short period at omega_n = 3.3047 rad/s and zeta = 0.42364, and
#: the cubic term makes static stability fall away with incidence.
SHORT_PERIOD = Plant(
    z_alpha=-1.20,
    z_delta=-0.18,
    m_alpha=-9.0,
    m_alpha3=42.0,
    m_q=-1.60,
    m_delta=-22.0,
    u_max_rad=np.deg2rad(40.0),
)

#: The declared divergence metric: 0.01 deg of incidence, 2 deg/s of pitch
#: rate, so anisotropy 200. Declared this way because a hardware-in-the-loop
#: fidelity claim about incidence tracking is read at hundredths of a degree
#: while the rate channel only has to agree to degrees per second. The
#: consequence is not cosmetic and is not hidden: the anisotropy is what makes
#: the closed-loop map non-normal in this metric, and the analytic bound's
#: looseness is a direct function of it. The dependence is swept and reported
#: in ``validation/validate_certificate.py``: the median ratio of true
#: divergence to bound falls from 0.93 at anisotropy 25 to 0.35 at 200.
DEFAULT_METRIC = DivergenceMetric(
    alpha_tolerance_rad=np.deg2rad(0.01),
    rate_tolerance_rad_s=np.deg2rad(2.0),
)


def zoh_matrices(
    jacobian: np.ndarray, input_matrix: np.ndarray, duration_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-order-hold discretisation over ``duration_s`` by one exponential.

    Returns ``(A_d, B_d)`` with

    ```
    A_d = exp(A h),      B_d = integral_0^h exp(A s) ds  B                  (7)
    ```

    computed as a single matrix exponential of the block matrix
    ``[[A, B], [0, 0]]``, which is Van Loan's method (Van Loan 1978,
    "Computing integrals involving the matrix exponential", *IEEE
    Transactions on Automatic Control* 23(3), 395-404, section II). The block
    form is used rather than ``A^-1 (A_d - I) B`` because the latter needs
    ``A`` invertible and loses accuracy when ``A`` is near singular, which a
    rigid-body channel with a free integration always is.

    Units: ``A`` in 1/s, ``B`` in the plant's per-control units, ``duration_s``
    in s.
    """
    matrix = np.asarray(jacobian, dtype=float)
    column = np.asarray(input_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"jacobian must be square, got shape {matrix.shape}")
    if column.ndim != 2 or column.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"input_matrix must have shape ({matrix.shape[0]}, m), got {column.shape}"
        )
    if duration_s < 0.0:
        raise ValueError(f"duration_s must be non-negative, got {duration_s}")
    states = matrix.shape[0]
    inputs = column.shape[1]
    block = np.zeros((states + inputs, states + inputs), dtype=float)
    block[:states, :states] = matrix
    block[:states, states:] = column
    exponential = expm(block * float(duration_s))
    return exponential[:states, :states], exponential[:states, states:]


@dataclass(frozen=True)
class LoopModel:
    """Plant, controller, sampling law and metric, bundled as one declaration."""

    plant: Plant = SHORT_PERIOD
    controller: Controller = Controller(gain=(0.20, 1.44))
    sampling: SamplingLaw = SamplingLaw(period_s=0.010, deadline_s=0.010)
    metric: DivergenceMetric = DEFAULT_METRIC
    operating_alpha_rad: float = 0.0
    _cache: dict[str, np.ndarray] = field(
        default_factory=dict, repr=False, compare=False
    )

    def monodromy(self) -> np.ndarray:
        """The closed-loop one-period map ``Phi = A_d + B_d K`` at the operating point.

        Dimensionless. This is the linearisation the analytic certificate of
        :mod:`hilverdict.certificate` is a proof about.
        """
        if "monodromy" not in self._cache:
            jacobian = self.plant.jacobian(self.operating_alpha_rad)
            discrete, input_discrete = zoh_matrices(
                jacobian, self.plant.input_matrix, self.sampling.period_s
            )
            self._cache["monodromy"] = discrete + input_discrete @ (
                self.controller.gain_row
            )
        return self._cache["monodromy"]

    def scaled_monodromy(self) -> np.ndarray:
        """``S Phi S^-1``: the one-period map in the units of the metric."""
        scaling = self.metric.scaling
        return scaling @ self.monodromy() @ self.metric.inverse_scaling

    def scaled_jacobian(self) -> np.ndarray:
        """``S A S^-1`` at the operating point, 1/s."""
        scaling = self.metric.scaling
        return (
            scaling
            @ self.plant.jacobian(self.operating_alpha_rad)
            @ self.metric.inverse_scaling
        )

    def scaled_input_matrix(self) -> np.ndarray:
        """``S B``, shape (2, 1): the input Jacobian in the units of the metric."""
        return self.metric.scaling @ self.plant.input_matrix

    def spectral_radius(self) -> float:
        """Largest closed-loop eigenvalue magnitude; below 1 is asymptotic stability."""
        return float(np.max(np.abs(np.linalg.eigvals(self.monodromy()))))
