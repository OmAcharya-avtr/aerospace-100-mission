"""The four environmental disturbance torque models.

Every function here returns a torque vector in the frame in which its inputs were
given (in practice the body frame), in newton metres. Nothing in this module knows
about orbits or time; the orbital sweep lives in :mod:`disturbtorque.profile`.

Sign convention: a torque is ``offset x force``, with ``offset`` the vector from the
centre of mass to the centre of pressure.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _validate as _v
from .constants import MU_EARTH, SRP_PRESSURE_1AU

__all__ = [
    "gravity_gradient_torque",
    "gravity_gradient_max_magnitude",
    "gravity_gradient_planar",
    "aerodynamic_torque",
    "aerodynamic_force",
    "solar_radiation_torque",
    "solar_radiation_force",
    "magnetic_torque",
]


# --------------------------------------------------------------------------------
# 1. Gravity gradient
# --------------------------------------------------------------------------------
def gravity_gradient_torque(
    inertia: ArrayLike,
    nadir_unit_body: ArrayLike,
    radius_m: float,
    mu: float = MU_EARTH,
) -> NDArray[np.float64]:
    r"""Gravity-gradient torque on a rigid body [N m].

    .. math:: \mathbf{T} = \frac{3\mu}{R^3}\, \hat{\mathbf{u}} \times (\mathbf{I}\,
              \hat{\mathbf{u}})

    where :math:`\hat{\mathbf{u}}` is the unit vector from the spacecraft toward the
    Earth centre (nadir), expressed in the body frame, and :math:`\mathbf{I}` is the
    inertia tensor about the centre of mass in the same frame.

    Source
        Wertz, *Spacecraft Attitude Determination and Control* (1978); Hughes,
        *Spacecraft Attitude Dynamics*; reproduced in Larson & Wertz, *Space Mission
        Analysis and Design*.
    Units
        ``inertia`` kg m^2, ``nadir_unit_body`` dimensionless unit vector, ``radius_m``
        m, ``mu`` m^3 s^-2; return N m.
    Assumptions
        Rigid body; first-order expansion of the gravity field over the vehicle, i.e.
        the vehicle's largest dimension is far smaller than the orbit radius; point-mass
        central field (no J2); no flexible or fuel-slosh contribution.
    Validity
        Exact to O((L/R)^2) with L the vehicle size. For a 10 m spacecraft at 500 km the
        neglected term is of order (10 / 6.9e6)^2 ~ 2e-12 relative, so the modelling
        error here is set entirely by the inertia tensor's own accuracy.

    Notes
    -----
    The torque vanishes when nadir is along a principal axis, and for a body whose
    principal axes are the body axes it takes the planar form
    ``3 mu / (2 R^3) (Iz - Iy) sin(2 theta)`` for a rotation ``theta`` about the body x
    axis; see :func:`gravity_gradient_planar` and
    :func:`gravity_gradient_max_magnitude`.
    """
    inert = _v.as_inertia_matrix(inertia)
    u = _v.as_unit_vector(nadir_unit_body, "nadir_unit_body")
    r = _v.positive(radius_m, "radius_m")
    m = _v.positive(mu, "mu")
    return (3.0 * m / r**3) * np.cross(u, inert @ u)


def gravity_gradient_planar(
    i_yy: float, i_zz: float, theta_rad: ArrayLike, radius_m: float, mu: float = MU_EARTH
) -> NDArray[np.float64]:
    r"""Planar gravity-gradient torque magnitude about the body x axis [N m].

    .. math:: T_x = \frac{3\mu}{2R^3}\,(I_{zz} - I_{yy})\,\sin 2\theta

    for a body whose principal axes are the body axes, tilted by ``theta`` about x from
    nadir-along-z. This is the form quoted for first-cut sizing in Larson & Wertz,
    *Space Mission Analysis and Design*, whose worst case is ``theta = 45 deg``.

    Units: moments of inertia kg m^2, ``theta_rad`` rad, ``radius_m`` m, return N m.
    Assumptions and validity: as :func:`gravity_gradient_torque`, plus a diagonal
    inertia tensor and a single-axis offset.
    """
    iyy = _v.positive(i_yy, "i_yy")
    izz = _v.positive(i_zz, "i_zz")
    r = _v.positive(radius_m, "radius_m")
    m = _v.positive(mu, "mu")
    theta = np.asarray(theta_rad, dtype=float)
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta_rad must be finite")
    return (3.0 * m / (2.0 * r**3)) * (izz - iyy) * np.sin(2.0 * theta)


def gravity_gradient_max_magnitude(
    i_yy: float, i_zz: float, radius_m: float, mu: float = MU_EARTH
) -> float:
    r"""Worst-case planar gravity-gradient torque [N m], attained at 45 deg off nadir.

    .. math:: T_{max} = \frac{3\mu}{2R^3}\,|I_{zz} - I_{yy}|

    Source: Larson & Wertz, *Space Mission Analysis and Design*, worst-case disturbance
    torque estimate. Units: kg m^2, m, m^3 s^-2 in; N m out. Assumptions and validity as
    :func:`gravity_gradient_planar`.
    """
    iyy = _v.positive(i_yy, "i_yy")
    izz = _v.positive(i_zz, "i_zz")
    r = _v.positive(radius_m, "radius_m")
    m = _v.positive(mu, "mu")
    return float(3.0 * m / (2.0 * r**3) * abs(izz - iyy))


# --------------------------------------------------------------------------------
# 2. Aerodynamic
# --------------------------------------------------------------------------------
def aerodynamic_force(
    density_kg_m3: float,
    v_rel_body: ArrayLike,
    drag_coefficient: float,
    area_m2: float,
) -> NDArray[np.float64]:
    r"""Free-molecular aerodynamic force on the vehicle [N].

    .. math:: \mathbf{F} = -\tfrac{1}{2}\,\rho\, C_d\, A\, |\mathbf{v}|\,\mathbf{v}

    i.e. magnitude :math:`\tfrac12 \rho C_d A v^2` directed opposite the relative wind.

    Source
        The free-molecular drag law used for disturbance-torque estimation in Larson &
        Wertz, *Space Mission Analysis and Design*, and Wertz, *Spacecraft Attitude
        Determination and Control*; the same form appears in Vallado, *Fundamentals of
        Astrodynamics and Applications*, as the cannonball drag model.
    Units
        ``density_kg_m3`` kg m^-3, ``v_rel_body`` m s^-1 in the body frame,
        ``drag_coefficient`` dimensionless, ``area_m2`` m^2; return N.
    Assumptions
        Free-molecular flow; all incident momentum transferred along the flow direction,
        so **lift is neglected** and the force is purely anti-parallel to the relative
        wind; a single constant projected area, so no attitude-dependent projection and
        no self-shadowing; no thermal (re-emission) contribution beyond what is lumped
        into Cd.
    Validity
        Above roughly 150 km, where the Knudsen number is large. Cd near 2.2 carries
        about +/-20 % uncertainty, and the density model above 400 km carries a factor of
        several between solar minimum and maximum; both propagate linearly into the
        force.
    """
    rho = _v.non_negative(density_kg_m3, "density_kg_m3")
    cd = _v.non_negative(drag_coefficient, "drag_coefficient")
    area = _v.non_negative(area_m2, "area_m2")
    v = np.atleast_2d(np.asarray(v_rel_body, dtype=float))
    if v.shape[-1] != 3:
        raise ValueError(f"v_rel_body must have trailing dimension 3, got shape {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError("v_rel_body must be finite")
    speed = np.linalg.norm(v, axis=-1)
    f = -0.5 * rho * cd * area * speed[:, None] * v
    return f[0] if np.ndim(v_rel_body) == 1 else f


def aerodynamic_torque(
    density_kg_m3: float,
    v_rel_body: ArrayLike,
    drag_coefficient: float,
    area_m2: float,
    cp_offset_m: ArrayLike,
) -> NDArray[np.float64]:
    r"""Aerodynamic disturbance torque [N m].

    .. math:: \mathbf{T} = (\mathbf{r}_{cp} - \mathbf{r}_{cm}) \times \mathbf{F}_{aero}

    with the force from :func:`aerodynamic_force`. For a normal-incidence flat plate this
    reduces to the textbook scalar form :math:`T = \tfrac12 \rho C_d A v^2 (c_{pa} -
    c_g)`.

    Source, units, assumptions and validity: as :func:`aerodynamic_force`, plus a
    centre-of-pressure offset that is treated as fixed in the body frame. In reality the
    aerodynamic centre of pressure moves with the flow direction; holding it fixed is the
    standard sizing simplification and is why this is a worst-case estimate.
    """
    offset = _v.as_vector3(cp_offset_m, "cp_offset_m")
    f = aerodynamic_force(density_kg_m3, v_rel_body, drag_coefficient, area_m2)
    return np.cross(offset, f)


# --------------------------------------------------------------------------------
# 3. Solar radiation pressure
# --------------------------------------------------------------------------------
def solar_radiation_force(
    sun_unit_body: ArrayLike,
    area_m2: float,
    reflectance: float,
    distance_au: float = 1.0,
    illuminated: bool = True,
    pressure_1au: float = SRP_PRESSURE_1AU,
) -> NDArray[np.float64]:
    r"""Solar radiation pressure force [N].

    .. math:: \mathbf{F} = -\frac{\Phi}{c}\,\frac{1}{d_{AU}^2}\,A_s\,(1+q)\,
              \hat{\mathbf{s}}

    where :math:`\hat{\mathbf{s}}` points from the spacecraft toward the Sun, so the
    force is anti-sunward.

    Source
        Larson & Wertz, *Space Mission Analysis and Design*, and Wertz, *Spacecraft
        Attitude Determination and Control*: F = (F_s/c) A_s (1+q) cos(i). Here the
        incidence cosine is taken as already contained in ``area_m2``, which is the
        *projected* sunlit area.
    Units
        ``area_m2`` m^2, ``reflectance`` dimensionless in [0, 1], ``distance_au`` AU,
        ``pressure_1au`` N m^-2; return N.
    Assumptions
        A single lumped surface with one reflectance factor; specular-plus-absorbed
        model only, so a diffuse (Lambertian) surface's 2/3 factor and any re-radiation
        torque are not represented; no self-shadowing; **no Earth albedo and no Earth
        infrared**, which together can reach roughly a third of the direct solar
        pressure in low orbit and are simply absent from this model.
    Validity
        Any sunlit heliocentric distance; ``illuminated=False`` zeroes the force in
        eclipse. The (1+q) form bounds the true force for a flat plate: q = 0 gives pure
        absorption, q = 1 gives perfect specular reflection at normal incidence.
    """
    s = _v.as_unit_vector(sun_unit_body, "sun_unit_body")
    area = _v.non_negative(area_m2, "area_m2")
    q = _v.in_range(reflectance, "reflectance", 0.0, 1.0)
    d = _v.positive(distance_au, "distance_au")
    p = _v.non_negative(pressure_1au, "pressure_1au")
    if not illuminated:
        return np.zeros(3)
    return -(p / d**2) * area * (1.0 + q) * s


def solar_radiation_torque(
    sun_unit_body: ArrayLike,
    area_m2: float,
    reflectance: float,
    cp_offset_m: ArrayLike,
    distance_au: float = 1.0,
    illuminated: bool = True,
    pressure_1au: float = SRP_PRESSURE_1AU,
) -> NDArray[np.float64]:
    r"""Solar radiation pressure torque [N m].

    ``T = (r_cps - r_cm) x F_srp`` with the force from :func:`solar_radiation_force`.
    Source, units, assumptions and validity: as :func:`solar_radiation_force`.
    """
    offset = _v.as_vector3(cp_offset_m, "cp_offset_m")
    f = solar_radiation_force(
        sun_unit_body, area_m2, reflectance, distance_au, illuminated, pressure_1au
    )
    return np.cross(offset, f)


# --------------------------------------------------------------------------------
# 4. Residual magnetic dipole
# --------------------------------------------------------------------------------
def magnetic_torque(dipole_body_am2: ArrayLike, b_field_body_t: ArrayLike) -> NDArray[np.float64]:
    r"""Torque from a residual magnetic dipole in the geomagnetic field [N m].

    .. math:: \mathbf{T} = \mathbf{m} \times \mathbf{B}

    Source
        The magnetic dipole torque law; used for spacecraft disturbance estimation in
        Wertz, *Spacecraft Attitude Determination and Control*, and Larson & Wertz,
        *Space Mission Analysis and Design*, in the worst-case scalar form T = D B.
    Units
        ``dipole_body_am2`` A m^2, ``b_field_body_t`` tesla; return N m
        (1 A m^2 * 1 T = 1 N m exactly).
    Assumptions
        A single lumped residual dipole fixed in the body frame; no induced or
        current-loop-dependent moment, no eddy-current or hysteresis damping torque.
        Residual dipole is one of the least well known spacecraft parameters before
        magnetic testing; an order-of-magnitude uncertainty in it is normal, and the
        torque is linear in it.
    Validity
        Anywhere the field model is valid; the field model here is a centred dipole and
        is the dominant error source, not this expression.
    """
    m = _v.as_vector3(dipole_body_am2, "dipole_body_am2")
    b = np.atleast_2d(np.asarray(b_field_body_t, dtype=float))
    if b.shape[-1] != 3:
        raise ValueError(f"b_field_body_t must have trailing dimension 3, got shape {b.shape}")
    if not np.all(np.isfinite(b)):
        raise ValueError("b_field_body_t must be finite")
    t = np.cross(m, b)
    return t[0] if np.ndim(b_field_body_t) == 1 else t
