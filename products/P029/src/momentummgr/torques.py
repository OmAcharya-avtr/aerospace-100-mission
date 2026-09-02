"""The four environmental disturbance torques, implemented from the cited physics.

Every function returns a torque in the frame its inputs were given in (in practice the
body frame), in newton metres. Sign convention throughout: ``T = offset x force`` with
``offset`` the vector from the centre of mass to the centre of pressure.

These are the standard first-order forms found in Wertz, *Spacecraft Attitude
Determination and Control*, Sidi, *Spacecraft Dynamics and Control*, and Markley and
Crassidis, *Fundamentals of Spacecraft Attitude Determination and Control*. There is no
new physics here. They exist in this package so that the momentum accumulation this
package manages can be cross-checked against an independent implementation of the same
environment (P027 ``disturbtorque``) without either code sharing a line with the other.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _validate as _v
from .constants import MU_EARTH, SRP_PRESSURE_1AU

__all__ = [
    "gravity_gradient_torque",
    "gravity_gradient_worst_case",
    "aerodynamic_force",
    "aerodynamic_torque",
    "srp_force",
    "srp_torque",
    "residual_dipole_torque",
]


def gravity_gradient_torque(
    inertia: ArrayLike,
    nadir_unit_body: ArrayLike,
    radius_m: float,
    mu: float = MU_EARTH,
) -> NDArray[np.float64]:
    r"""Gravity-gradient torque on a rigid body [N m].

    .. math:: \mathbf{T}_{gg} = 3 n^2\, \hat{\mathbf{u}} \times (\mathbf{I}\,
              \hat{\mathbf{u}}), \qquad n^2 = \mu / R^3

    with :math:`\hat{\mathbf{u}}` the unit nadir direction in the body frame and
    :math:`\mathbf{I}` the inertia tensor about the centre of mass in the same frame.
    Written in terms of the mean motion ``n`` because that is the form in which the
    result is usually quoted alongside the orbit rate.

    Source
        Wertz, *Spacecraft Attitude Determination and Control*; Hughes, *Spacecraft
        Attitude Dynamics*; Sidi, *Spacecraft Dynamics and Control*.
    Units
        ``inertia`` kg m^2, ``nadir_unit_body`` dimensionless unit vector, ``radius_m``
        m, ``mu`` m^3 s^-2; return N m.
    Assumptions
        Rigid body; first-order expansion of the field across the vehicle; point-mass
        central field (no J2); no flexible or slosh contribution.
    Validity
        Error is O((L/R)^2) with L the vehicle's largest dimension: about 2e-12 relative
        for a 10 m vehicle at 500 km, so the accuracy is set by the inertia tensor, not
        by this expression. Vanishes when nadir lies along a principal axis.
    """
    inert = _v.as_inertia_matrix(inertia)
    u = _v.as_unit_vector(nadir_unit_body, "nadir_unit_body")
    r = _v.positive(radius_m, "radius_m")
    m = _v.positive(mu, "mu")
    n_sq = m / r**3
    return 3.0 * n_sq * np.cross(u, inert @ u)


def gravity_gradient_worst_case(
    i_min: float, i_max: float, radius_m: float, mu: float = MU_EARTH
) -> float:
    r"""Worst-case planar gravity-gradient torque magnitude [N m].

    .. math:: T_{max} = \tfrac{3}{2}\, \frac{\mu}{R^3}\, |I_{max} - I_{min}|

    the maximum of :math:`\tfrac{3\mu}{2R^3}\Delta I \sin 2\theta`, attained 45 deg off
    nadir. This is the first-cut sizing expression of Larson & Wertz, *Space Mission
    Analysis and Design*. Units: kg m^2, m, m^3 s^-2 in; N m out. Assumptions and
    validity as :func:`gravity_gradient_torque`, plus a diagonal inertia and a
    single-axis offset.
    """
    a = _v.positive(i_min, "i_min")
    b = _v.positive(i_max, "i_max")
    r = _v.positive(radius_m, "radius_m")
    m = _v.positive(mu, "mu")
    return float(1.5 * m / r**3 * abs(b - a))


def aerodynamic_force(
    density_kg_m3: float,
    v_rel: ArrayLike,
    drag_coefficient: float,
    area_m2: float,
) -> NDArray[np.float64]:
    r"""Free-molecular aerodynamic force [N].

    .. math:: \mathbf{F} = -\tfrac{1}{2}\rho\, C_d\, A\, |\mathbf{v}_{rel}|\,
              \mathbf{v}_{rel}

    i.e. magnitude :math:`\tfrac12 \rho C_d A v^2` directed opposite the relative wind.

    Source
        The free-molecular drag law used for disturbance sizing in Wertz, *Spacecraft
        Attitude Determination and Control*, and Larson & Wertz, *Space Mission Analysis
        and Design*; the cannonball form in Vallado.
    Units
        kg m^-3, m s^-1 (shape ``(3,)`` or ``(N, 3)``), dimensionless, m^2; return N.
    Assumptions
        Free-molecular flow; all incident momentum transferred along the flow, so **lift
        is neglected** and the force is anti-parallel to the wind; one constant projected
        area, hence no attitude-dependent projection and no self-shadowing.
    Validity
        Above roughly 150 km. Cd carries about +/-20 % and the density model above
        400 km a factor of several over a solar cycle; both are linear in the force.
    """
    rho = _v.non_negative(density_kg_m3, "density_kg_m3")
    cd = _v.non_negative(drag_coefficient, "drag_coefficient")
    area = _v.non_negative(area_m2, "area_m2")
    v = np.asarray(v_rel, dtype=float)
    if v.shape[-1] != 3:
        raise ValueError(f"v_rel must have trailing dimension 3, got shape {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError("v_rel must be finite")
    speed = np.linalg.norm(v, axis=-1)
    return -0.5 * rho * cd * area * speed[..., None] * v


def aerodynamic_torque(
    density_kg_m3: float,
    v_rel_body: ArrayLike,
    drag_coefficient: float,
    area_m2: float,
    cp_offset_m: ArrayLike,
) -> NDArray[np.float64]:
    r"""Aerodynamic disturbance torque [N m], ``T = r_cp x F_aero``.

    For normal incidence this reduces to the textbook scalar
    :math:`T = \tfrac12 \rho C_d A v^2 (c_{pa} - c_g)`. Source, units, assumptions and
    validity as :func:`aerodynamic_force`, with the additional simplification that the
    centre of pressure is fixed in the body frame; in reality it moves with the flow
    direction, which is one reason this is a sizing estimate.
    """
    offset = _v.as_vector3(cp_offset_m, "cp_offset_m")
    return np.cross(offset, aerodynamic_force(density_kg_m3, v_rel_body, drag_coefficient, area_m2))


def srp_force(
    sun_unit_body: ArrayLike,
    area_m2: float,
    reflectance: float,
    distance_au: float = 1.0,
    illuminated: bool = True,
    pressure_1au: float = SRP_PRESSURE_1AU,
) -> NDArray[np.float64]:
    r"""Solar radiation pressure force [N].

    .. math:: \mathbf{F} = -\frac{P_{1AU}}{d_{AU}^2}\, A_s\, (1 + q)\, \hat{\mathbf{s}}

    with :math:`\hat{\mathbf{s}}` from the spacecraft toward the Sun, so the force is
    anti-sunward.

    Source
        Wertz, *Spacecraft Attitude Determination and Control*, and Larson & Wertz,
        *Space Mission Analysis and Design*: ``F = (Phi/c) A (1 + q) cos i``. The
        incidence cosine is taken as already contained in ``area_m2``, the *projected*
        sunlit area.
    Units
        m^2, dimensionless q in [0, 1], AU, N m^-2; return N.
    Assumptions
        One lumped surface with one reflectance; specular-plus-absorbed only, so no
        Lambertian 2/3 factor and no re-radiation torque; no self-shadowing; **no Earth
        albedo and no Earth infrared**, which together reach roughly a third of the
        direct solar pressure in LEO and are simply absent here.
    Validity
        Any heliocentric distance; ``illuminated=False`` zeroes the force in eclipse.
    """
    s = _v.as_unit_vector(sun_unit_body, "sun_unit_body")
    area = _v.non_negative(area_m2, "area_m2")
    q = _v.in_range(reflectance, "reflectance", 0.0, 1.0)
    d = _v.positive(distance_au, "distance_au")
    p = _v.non_negative(pressure_1au, "pressure_1au")
    if not illuminated:
        return np.zeros(3)
    return -(p / d**2) * area * (1.0 + q) * s


def srp_torque(
    sun_unit_body: ArrayLike,
    area_m2: float,
    reflectance: float,
    cp_offset_m: ArrayLike,
    distance_au: float = 1.0,
    illuminated: bool = True,
    pressure_1au: float = SRP_PRESSURE_1AU,
) -> NDArray[np.float64]:
    """Solar radiation pressure torque [N m], ``T = r_cps x F_srp``.

    Source, units, assumptions and validity as :func:`srp_force`.
    """
    offset = _v.as_vector3(cp_offset_m, "cp_offset_m")
    return np.cross(
        offset, srp_force(sun_unit_body, area_m2, reflectance, distance_au, illuminated, pressure_1au)
    )


def residual_dipole_torque(dipole_body_am2: ArrayLike, b_body_t: ArrayLike) -> NDArray[np.float64]:
    r"""Torque from a residual magnetic dipole in the geomagnetic field [N m].

    .. math:: \mathbf{T} = \mathbf{m} \times \mathbf{B}

    Source
        The magnetic dipole torque law, used for spacecraft disturbance estimation in
        Wertz, *Spacecraft Attitude Determination and Control*, and for magnetorquer
        design in Sidi, *Spacecraft Dynamics and Control*.
    Units
        A m^2 and tesla in, N m out (1 A m^2 T = 1 N m exactly). Accepts ``(3,)`` or
        ``(N, 3)`` for the field.
    Assumptions
        One lumped residual dipole fixed in the body frame; no induced moment, no
        eddy-current or hysteresis damping. Residual dipole is among the least
        well-known spacecraft parameters before magnetic testing, and the torque is
        linear in it.
    Validity
        Wherever the field model is valid; the centred-dipole field model, not this
        expression, dominates the error.
    """
    m = _v.as_vector3(dipole_body_am2, "dipole_body_am2")
    b = np.asarray(b_body_t, dtype=float)
    if b.shape[-1] != 3:
        raise ValueError(f"b_body_t must have trailing dimension 3, got shape {b.shape}")
    if not np.all(np.isfinite(b)):
        raise ValueError("b_body_t must be finite")
    return np.cross(m, b)
