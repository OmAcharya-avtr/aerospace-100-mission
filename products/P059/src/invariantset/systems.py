"""The example systems this package is validated on.

Every system is a `(A, X, W)` triple for equation (6) of `invariant`, with the
units of each entry stated.  Four of the six have a hand-computable answer and
are used as known-answer fixtures in `tests/test_known_answers.py`; two exist
to exercise the pathologies (facet growth and non-convergence).

**The numbers in `attitude_loop` are illustrative**, chosen so that the
maximal robust invariant set is a non-trivial proper subset of `X`.  They are
not a measurement of any vehicle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .polytope import Polytope

__all__ = ["LinearSystem", "SYSTEMS", "get_system", "system_names"]


@dataclass(frozen=True)
class LinearSystem:
    """A named `x(k+1) = A x(k) + w(k)`, `w in W`, `x in X` instance.

    Attributes
    ----------
    name : str
    A : ndarray, shape (n, n)
        Dimensionless state-transition matrix (closed-loop where relevant).
    X : Polytope
        State constraint set, in the units named by `state_units`.
    W : Polytope
        Disturbance set, same units as `X`.
    state_units : tuple of str
    description : str
    expected : str
        One line stating the hand-computed or measured expectation, so a
        reader can see what the fixture is for without opening the tests.
    """

    name: str
    A: np.ndarray
    X: Polytope
    W: Polytope
    state_units: tuple[str, ...]
    description: str
    expected: str

    @property
    def dim(self) -> int:
        """Ambient dimension."""
        return self.X.dim


def _scalar_invariant_equals_x() -> LinearSystem:
    # 1-D, lambda = 0.8, w_max = 0.1, b = 1.0.
    # Minimal RPI half-width  = w/(1-|lambda|) = 0.1/0.2 = 0.5
    # X half-width b = 1.0 >= 0.5, so X is already robustly invariant and the
    # MAXIMAL set is X itself: [-1, 1], not [-0.5, 0.5].
    return LinearSystem(
        name="scalar_invariant_equals_x",
        A=np.array([[0.8]]),
        X=Polytope.from_box([0.0], [1.0]),
        W=Polytope.from_box([0.0], [0.1]),
        state_units=("dimensionless",),
        description="lambda=0.8, |w|<=0.1, |x|<=1.0",
        expected="S_inf = X = [-1, 1]; converges at k=1; the minimal RPI set is "
        "[-0.5, 0.5], a different object",
    )


def _scalar_empty() -> LinearSystem:
    # 1-D, lambda = 0.9, w_max = 0.2, b = 1.0.
    # w/(1-|lambda|) = 0.2/0.1 = 2.0 > 1.0 = b, so S_inf is EMPTY.
    # r_{k+1} = (r_k - w)/lambda, fixed point r* = 2, so r_k = 2 - (10/9)^k.
    # (10/9)^7 = 2.090751 > 2, hence r_7 < 0 and emptiness is detected at k=7.
    return LinearSystem(
        name="scalar_empty",
        A=np.array([[0.9]]),
        X=Polytope.from_box([0.0], [1.0]),
        W=Polytope.from_box([0.0], [0.2]),
        state_units=("dimensionless",),
        description="lambda=0.9, |w|<=0.2, |x|<=1.0",
        expected="S_inf is empty, detected at k=7; r_k = 2 - (10/9)^k",
    )


def _nilpotent_2d() -> LinearSystem:
    # A = [[0,1],[0,0]] maps (x1,x2) -> (x2,0).  Hand trace in
    # tests/test_known_answers.py: S_inf = {|x1|<=1, |x2|<=0.9}, k=2.
    return LinearSystem(
        name="nilpotent_2d",
        A=np.array([[0.0, 1.0], [0.0, 0.0]]),
        X=Polytope.from_box([0.0, 0.0], [1.0, 1.0]),
        W=Polytope.from_box([0.0, 0.0], [0.1, 0.1]),
        state_units=("dimensionless", "dimensionless"),
        description="singular A, box X of half-width 1, box W of half-width 0.1",
        expected="S_inf = {|x1|<=1, |x2|<=0.9}, converges at k=2, area 3.6",
    )


def _decoupled_2d() -> LinearSystem:
    # Diagonal A, so the recursion decouples into two independent 1-D
    # recursions with thresholds w_i/(1-lambda_i) = 0.1/0.5 = 0.2 <= 1.
    return LinearSystem(
        name="decoupled_2d",
        A=np.diag([0.5, 0.5]),
        X=Polytope.from_box([0.0, 0.0], [1.0, 1.0]),
        W=Polytope.from_box([0.0, 0.0], [0.1, 0.1]),
        state_units=("dimensionless", "dimensionless"),
        description="A = 0.5 I, box X of half-width 1, box W of half-width 0.1",
        expected="S_inf = X = [-1,1]^2, converges at k=1, area 4.0",
    )


def attitude_loop_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """The illustrative attitude loop: `(A_plant, B, K, dt, a_w)`.

    Plant: a rigid single-axis attitude channel modelled as a double
    integrator in `[theta, theta_dot]` with units `[rad, rad/s]`, discretised
    exactly by zero-order hold at `dt = 0.05 s`:

        A_plant = [[1, dt], [0, 1]],   B = [[dt^2/2], [dt]]                (10)

    which is the exact ZOH discretisation of `theta_ddot = u` (standard
    result; e.g. Franklin, Powell and Workman, *Digital Control of Dynamic
    Systems*, 3rd ed., Addison-Wesley, 1997).  Valid for small
    angles about a single axis, rigid body, no actuator lag, and a control
    input held constant over each sample.

    Gain: `K = [8.0, 3.8] [1/s^2, 1/s]`, placing the closed-loop poles of
    `A_plant - B K` at `0.9 +/- 0.1j` (characteristic polynomial
    `z^2 - 1.8 z + 0.82`).  Derived by hand in validation/VALIDATION.md
    section 3; it is a design choice, not a measurement.

    Disturbance: an unmodelled angular acceleration of magnitude
    `a_w = 0.12 rad/s^2` held over one sample enters the state as the box with
    half-widths `[dt^2/2 * a_w, dt * a_w] = [1.5e-4 rad, 6.0e-3 rad/s]`.
    """
    dt = 0.05
    a_w = 0.12
    A_plant = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5 * dt * dt], [dt]])
    K = np.array([[8.0, 3.8]])
    return A_plant, B, K, dt, a_w


def _attitude_loop() -> LinearSystem:
    A_plant, B, K, dt, a_w = attitude_loop_matrices()
    A_cl = A_plant - B @ K
    # State constraints |theta| <= 0.30 rad, |theta_dot| <= 0.50 rad/s, plus
    # the input constraint |u| = |K x| <= 3.0 rad/s^2 written as two state
    # halfspaces, which is what makes X non-box and the answer non-trivial.
    A_X = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [K[0, 0], K[0, 1]],
            [-K[0, 0], -K[0, 1]],
        ]
    )
    b_X = np.array([0.30, 0.30, 0.50, 0.50, 3.0, 3.0])
    W = Polytope.from_box([0.0, 0.0], [0.5 * dt * dt * a_w, dt * a_w])
    return LinearSystem(
        name="attitude_loop",
        A=A_cl,
        X=Polytope(A_X, b_X),
        W=W,
        state_units=("rad", "rad/s"),
        description=(
            "single-axis attitude double integrator, dt=0.05 s, K=[8.0, 3.8], "
            "closed-loop poles 0.9+/-0.1j; |theta|<=0.30 rad, "
            "|theta_dot|<=0.50 rad/s, |u|<=3.0 rad/s^2; disturbance "
            "0.12 rad/s^2 for one sample"
        ),
        expected="converges; facet and vertex counts measured in VALIDATION.md section 5",
    )


def _slow_pair() -> LinearSystem:
    # Two poles at 0.999 with coupling: the one-step recursion shrinks by a
    # factor close to 1 each iteration, so it needs far more than 50 steps.
    # This fixture exists to produce the non-convergence report.
    return LinearSystem(
        name="slow_pair",
        A=np.array([[0.999, 0.05], [0.0, 0.999]]),
        X=Polytope.from_box([0.0, 0.0], [1.0, 1.0]),
        W=Polytope.from_box([0.0, 0.0], [0.002, 0.002]),
        state_units=("dimensionless", "dimensionless"),
        description="repeated pole at 0.999 with coupling 0.05, |w|<=0.002",
        expected="does NOT converge within 40 iterations; see VALIDATION.md section 6",
    )


def _rotation(theta: float) -> np.ndarray:
    """Plane rotation by `theta` radians, used only to build fixtures."""
    c, sn = np.cos(theta), np.sin(theta)
    return np.array([[c, -sn], [sn, c]])


def _damped_rotation_2d() -> LinearSystem:
    # A damped rotation turns every box facet into a rotated facet, so each
    # iteration contributes genuinely new non-redundant rows.  This fixture
    # exists to measure facet growth, not because it is hand-computable.
    return LinearSystem(
        name="damped_rotation_2d",
        A=0.995 * _rotation(0.2),
        X=Polytope.from_box([0.0, 0.0], [1.0, 1.0]),
        W=Polytope.from_box([0.0, 0.0], [0.002, 0.002]),
        state_units=("dimensionless", "dimensionless"),
        description="A = 0.995 R(0.2 rad), box X half-width 1, box W half-width 0.002",
        expected="converges at k=8 with 32 facets and 32 vertices; raw rows 8k "
        "before redundancy removal (measured, VALIDATION.md section 5)",
    )


def _damped_rotation_3d() -> LinearSystem:
    # Same purpose in three dimensions, where the vertex count grows at twice
    # the facet rate: +8 vertices against +4 facets per iteration (measured).
    theta = 0.2
    R = np.eye(3)
    R[:2, :2] = _rotation(theta)
    return LinearSystem(
        name="damped_rotation_3d",
        A=np.diag([0.99, 0.99, 0.95]) @ R,
        X=Polytope.from_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        W=Polytope.from_box([0.0, 0.0, 0.0], [0.003, 0.003, 0.003]),
        state_units=("dimensionless",) * 3,
        description="A = diag(0.99, 0.99, 0.95) R_z(0.2 rad), box X and W in 3-D",
        expected="converges at k=7 with 30 facets and 56 vertices; vertices grow at "
        "twice the facet rate (measured, VALIDATION.md section 5)",
    )


SYSTEMS: dict[str, LinearSystem] = {
    s.name: s
    for s in (
        _scalar_invariant_equals_x(),
        _scalar_empty(),
        _nilpotent_2d(),
        _decoupled_2d(),
        _attitude_loop(),
        _damped_rotation_2d(),
        _damped_rotation_3d(),
        _slow_pair(),
    )
}


def system_names() -> list[str]:
    """Sorted names of the built-in systems."""
    return sorted(SYSTEMS)


def get_system(name: str) -> LinearSystem:
    """Look up a built-in system by name.

    Raises
    ------
    KeyError
        With the list of valid names in the message.
    """
    try:
        return SYSTEMS[name]
    except KeyError:
        raise KeyError(
            f"unknown system {name!r}; available: {', '.join(system_names())}"
        ) from None
