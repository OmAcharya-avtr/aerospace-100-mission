"""Control laws, the perpendicular-torque constraint, and momentum behaviour.

The property tests here are the core physics claims of the package:

* the magnetic torque is *always* perpendicular to B (the controllability gap);
* rotational kinetic energy is non-increasing under ideal B-dot, for **any**
  inertia tensor;
* the angular-momentum magnitude is non-increasing under ideal B-dot for an
  **isotropic** inertia;
* the angular-momentum magnitude is **not** monotone for a non-isotropic
  inertia - a claim that is proved by an explicit worked counterexample rather
  than asserted.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from detumblesim.control import (
    BDotController,
    CrossProductController,
    ideal_bdot_torque,
    magnetic_torque,
    perpendicular_component,
)

coord = st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False)
vec3 = st.lists(coord, min_size=3, max_size=3)
pos = st.floats(0.1, 5.0, allow_nan=False, allow_infinity=False)


def _unit_or_none(v):
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    return None if n < 1e-6 else a / n


class TestBDotController:
    def test_known_answer(self):
        # m = -k dB/dt with k = 2e5 and dB/dt = (1e-6, -2e-6, 0):
        # m = (-0.2, 0.4, 0.0) A m^2.
        c = BDotController(gain=2.0e5)
        assert np.allclose(
            c.command([3e-5, 0.0, 0.0], [1e-6, -2e-6, 0.0]), [-0.2, 0.4, 0.0]
        )

    def test_normalised_variant(self):
        c = BDotController(gain=1.0, normalise_by_field=True)
        m = c.command([0.0, 0.0, 4e-5], [4e-6, 0.0, 0.0])
        assert np.allclose(m, [-0.1, 0.0, 0.0])

    def test_normalised_variant_returns_zero_in_zero_field(self):
        c = BDotController(gain=1.0, normalise_by_field=True)
        assert np.allclose(c.command([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]), 0.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_rejects_bad_gain(self, bad):
        with pytest.raises(ValueError, match="gain"):
            BDotController(gain=bad)

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="shape"):
            BDotController(gain=1.0).command([1e-5, 0, 0], [1.0, 2.0])


class TestCrossProductController:
    def test_known_answer_torque_is_minus_k_omega_perp(self):
        c = CrossProductController(gain=1e-4)
        b = np.array([1e-5, 2e-5, -1e-5])
        w = np.array([0.1, -0.05, 0.2])
        torque = magnetic_torque(c.command(b, None, w), b)
        assert np.allclose(torque, -1e-4 * perpendicular_component(w, b), atol=1e-18)

    def test_requires_omega(self):
        with pytest.raises(ValueError, match="omega_body"):
            CrossProductController(gain=1.0).command([1e-5, 0.0, 0.0])

    def test_zero_field_gives_zero_command(self):
        c = CrossProductController(gain=1.0)
        assert np.allclose(c.command([0.0, 0.0, 0.0], None, [1.0, 0.0, 0.0]), 0.0)

    def test_rejects_bad_shapes(self):
        with pytest.raises(ValueError, match="shape"):
            CrossProductController(gain=1.0).command([1e-5, 0.0], None, [1.0, 0.0, 0.0])

    @pytest.mark.parametrize("bad", [0.0, -3.0])
    def test_rejects_bad_gain(self, bad):
        with pytest.raises(ValueError, match="gain"):
            CrossProductController(gain=bad)


class TestPerpendicularTorque:
    def test_magnetic_torque_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="shape"):
            magnetic_torque([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_perpendicular_component_of_zero_direction(self):
        v = np.array([1.0, 2.0, 3.0])
        assert np.allclose(perpendicular_component(v, [0.0, 0.0, 0.0]), v)

    @given(m=vec3, b=vec3)
    @settings(max_examples=100, deadline=None)
    def test_torque_is_always_perpendicular_to_b(self, m, b):
        # This is the controllability gap in one line: L . B == 0 identically.
        torque = magnetic_torque(m, b)
        scale = float(np.linalg.norm(m) * np.linalg.norm(b) ** 2) + 1e-30
        assert abs(float(torque @ np.asarray(b))) <= 1e-12 * scale


class TestIdealBDotEnergyAndMomentum:
    def test_zero_field_gives_zero_torque(self):
        assert np.allclose(ideal_bdot_torque([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0), 0.0)

    def test_known_answer(self):
        # L = -k |B|^2 omega_perp.  With B = (0, 0, 2), omega = (3, 0, 5),
        # omega_perp = (3, 0, 0) and k = 0.25, so L = -0.25 * 4 * (3, 0, 0)
        #                                          = (-3, 0, 0).
        assert np.allclose(
            ideal_bdot_torque([3.0, 0.0, 5.0], [0.0, 0.0, 2.0], 0.25), [-3.0, 0.0, 0.0]
        )

    @given(w=vec3, b=vec3, jx=pos, jy=pos, jz=pos, k=pos)
    @settings(max_examples=200, deadline=None)
    def test_kinetic_energy_never_increases_for_any_inertia(self, w, b, jx, jy, jz, k):
        # dT/dt = omega . L = -k |B|^2 |omega_perp|^2 <= 0, always.
        bb = np.asarray(b, dtype=float)
        if float(np.linalg.norm(bb)) < 1e-6:
            return
        if jx + jy < jz or jy + jz < jx or jz + jx < jy:
            return
        ww = np.asarray(w, dtype=float)
        torque = ideal_bdot_torque(ww, bb, k)
        de = float(ww @ torque)
        scale = k * float(bb @ bb) * float(ww @ ww) + 1e-30
        assert de <= 1e-12 * scale

    @given(w=vec3, b=vec3, j=pos, k=pos)
    @settings(max_examples=200, deadline=None)
    def test_momentum_magnitude_never_increases_for_isotropic_inertia(self, w, b, j, k):
        # For J = j I, H = j omega, so d|H|/dt = j (omega . L) / |omega| <= 0.
        bb = np.asarray(b, dtype=float)
        ww = np.asarray(w, dtype=float)
        if float(np.linalg.norm(bb)) < 1e-6 or float(np.linalg.norm(ww)) < 1e-6:
            return
        h = j * ww
        dh = float(h @ ideal_bdot_torque(ww, bb, k))
        scale = j * k * float(bb @ bb) * float(ww @ ww) + 1e-30
        assert dh <= 1e-12 * scale


class TestMomentumIsNotMonotoneWhenAsymmetric:
    """The stated property fails for a non-isotropic inertia; here is why.

    With ``L = -k |B|^2 (omega - (omega.B_hat) B_hat)`` the momentum magnitude
    obeys ``d|H|/dt = (H . L)/|H|``, and

        H . L = -k |B|^2 [ omega^T J omega - (omega.B_hat)(B_hat^T J omega) ]

    The bracket is positive for every field direction only when ``J`` is
    isotropic.  Maximising ``(omega.B_hat)(B_hat . J omega) - omega^T J omega``
    over unit ``B_hat`` in the plane spanned by ``omega`` and ``J omega``
    gives a maximum eigenvalue proportional to ``(1 - J3)^2`` for
    ``J = diag(1, 1, J3)`` and ``omega = (a, 0, c)``, which is strictly
    positive whenever ``J3 != 1`` and ``a, c != 0``.
    """

    def test_explicit_counterexample(self):
        j = np.diag([1.0, 1.0, 4.0])
        w = np.array([1.0, 0.0, 1.0])
        jw = j @ w
        # Choose B_hat in the (omega, J omega) plane at the maximising angle.
        best = None
        for psi in np.linspace(0.0, np.pi, 20001):
            bh = np.array([np.cos(psi), 0.0, np.sin(psi)])
            val = float(w @ bh) * float(bh @ jw) - float(w @ jw)
            if best is None or val > best[0]:
                best = (val, bh)
        value, bh = best
        assert value > 0.0
        torque = ideal_bdot_torque(w, bh, 1.0)
        assert float(jw @ torque) > 0.0, "momentum magnitude must rise here"

    def test_analytic_maximum_matches_the_derivation(self):
        # For J = diag(1, 1, J3), omega = (a, 0, c), the maximum of the bracket
        # over B_hat in that plane is
        #   (-(J3 c^2 + a^2) + sqrt((J3 c^2 - a^2)^2 + a^2 c^2 (1 + J3)^2)) / 2
        # and the discriminant identity gives
        #   (J3 c^2 - a^2)^2 + a^2 c^2 (1 + J3)^2 - (J3 c^2 + a^2)^2
        #     = a^2 c^2 (1 - J3)^2 > 0 for J3 != 1.
        a, c, j3 = 1.0, 1.0, 4.0
        lhs = (j3 * c**2 - a**2) ** 2 + a**2 * c**2 * (1 + j3) ** 2
        rhs = (j3 * c**2 + a**2) ** 2
        assert np.isclose(lhs - rhs, a**2 * c**2 * (1 - j3) ** 2)
        analytic = 0.5 * (-(j3 * c**2 + a**2) + np.sqrt(lhs))
        j = np.diag([1.0, 1.0, j3])
        w = np.array([a, 0.0, c])
        jw = j @ w
        best = max(
            float(w @ bh) * float(bh @ jw) - float(w @ jw)
            for bh in (
                np.array([np.cos(p), 0.0, np.sin(p)])
                for p in np.linspace(0.0, np.pi, 200001)
            )
        )
        assert np.isclose(best, analytic, rtol=1e-6)

    @given(w=vec3, b=vec3, jx=pos, jy=pos, jz=pos, k=pos)
    @settings(max_examples=200, deadline=None)
    def test_energy_still_falls_even_where_momentum_rises(self, w, b, jx, jy, jz, k):
        bb, ww = np.asarray(b, dtype=float), np.asarray(w, dtype=float)
        if float(np.linalg.norm(bb)) < 1e-6:
            return
        if jx + jy < jz or jy + jz < jx or jz + jx < jy:
            return
        torque = ideal_bdot_torque(ww, bb, k)
        scale = k * float(bb @ bb) * float(ww @ ww) + 1e-30
        assert float(ww @ torque) <= 1e-12 * scale
