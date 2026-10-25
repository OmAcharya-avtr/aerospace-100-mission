"""The independent force model, and the potential it is the gradient of.

This is "baseline 2" of the product: a numerical force model whose terms are
each declared, so that its divergence from SGP4 is attributable to a stated
list of physical differences rather than to an unspecified "high fidelity"
propagator. It is **not** a precision orbit propagator and no claim is made
that it is closer to truth than SGP4 is. It is a second opinion built on
different mathematics, and the rate at which the two opinions separate is the
quantity this package measures.

Terms, each switchable, each with its source:

``two_body``
    ``-mu r / |r|^3``.
``zonal``
    Zonal harmonics ``J2``, ``J3``, ``J4`` of the model in
    :class:`orbitdrift.constants.GravityModel`, evaluated as the analytic
    gradient of :func:`zonal_potential`. ``tests/test_forces.py`` checks the
    returned acceleration against a central finite difference of that
    potential, which is a different expression and therefore a real check
    rather than a restatement. Tesseral and sectoral terms are **not**
    modelled; for a non-resonant orbit their effect is periodic rather than
    secular, and for a resonant one it is not, which is stated in README.md
    under Limitations.
``drag``
    ``-0.5 (C_D A / m) rho |v_rel| v_rel`` with ``v_rel = v - omega x r``.
    Density from :func:`atmospheric_density_kg_m3`, the piecewise exponential
    model of Vallado 2013 Table 8-4 (originally CIRA-72 and the US Standard
    Atmosphere 1976). The model is **static**: it has no solar-activity, no
    diurnal and no geomagnetic dependence, so for a sun-synchronous orbit in
    high solar activity the true density can differ from it by a factor of
    several. ``C_D A / m`` comes from the TLE ``B*`` via
    :func:`cd_area_over_mass_m2_kg`, which carries its own assumption.
``third_body``
    Point-mass Sun and Moon, ``mu_b (d/|d|^3 - r_b/|r_b|^3)`` with
    ``d = r_b - r``, the standard form (Montenbruck and Gill 2000, *Satellite
    Orbits*, eq. 3.37) written so that the indirect term is not lost to
    cancellation at small ``|r|``. Body positions from the low-precision
    analytic series in :mod:`orbitdrift.frames`.

Not modelled at all, and therefore part of the declared divergence:
solar radiation pressure, Earth albedo and infrared, solid and ocean tides,
relativistic corrections, and the tesseral field. Solar radiation pressure is
the largest of these for a high area-to-mass object and its omission is
recorded in README.md under Limitations rather than hidden here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    BSTAR_REFERENCE_DENSITY_KG_M2_ER,
    MU_MOON_KM3_S2,
    MU_SUN_KM3_S2,
    WGS72,
    GravityModel,
)
from .frames import (
    earth_rotation_vector_rad_s,
    geodetic_altitude_km,
    moon_position_teme_km,
    sun_position_teme_km,
    teme_to_ecef,
)

# Vallado 2013, Table 8-4: piecewise exponential atmosphere. Columns are the
# base altitude h0 (km), the nominal density rho0 at h0 (kg/m^3) and the scale
# height H (km). Density is rho0 * exp(-(h - h0) / H) in the band starting at
# h0. Below 0 km and above 1000 km the end bands are extrapolated, which for
# the altitudes this package propagates means the 1000 km band is used for
# everything above it; at those altitudes the drag acceleration is below
# 1e-14 km/s^2 and the extrapolation cannot matter.
_ATMOSPHERE_TABLE = np.array(
    [
        [0.0, 1.225, 7.249],
        [25.0, 3.899e-2, 6.349],
        [30.0, 1.774e-2, 6.682],
        [40.0, 3.972e-3, 7.554],
        [50.0, 1.057e-3, 8.382],
        [60.0, 3.206e-4, 7.714],
        [70.0, 8.770e-5, 6.549],
        [80.0, 1.905e-5, 5.799],
        [90.0, 3.396e-6, 5.382],
        [100.0, 5.297e-7, 5.877],
        [110.0, 9.661e-8, 7.263],
        [120.0, 2.438e-8, 9.473],
        [130.0, 8.484e-9, 12.636],
        [140.0, 3.845e-9, 16.149],
        [150.0, 2.070e-9, 22.523],
        [180.0, 5.464e-10, 29.740],
        [200.0, 2.789e-10, 37.105],
        [250.0, 7.248e-11, 45.546],
        [300.0, 2.418e-11, 53.628],
        [350.0, 9.518e-12, 53.298],
        [400.0, 3.725e-12, 58.515],
        [450.0, 1.585e-12, 60.828],
        [500.0, 6.967e-13, 63.822],
        [600.0, 1.454e-13, 71.835],
        [700.0, 3.614e-14, 88.667],
        [800.0, 1.170e-14, 124.64],
        [900.0, 5.245e-15, 181.05],
        [1000.0, 3.019e-15, 268.00],
    ]
)

ATMOSPHERE_MODEL_NAME = "Vallado Table 8-4 piecewise exponential (static, no solar index)"


def atmospheric_density_kg_m3(altitude_km: np.ndarray | float) -> np.ndarray:
    """Density from the declared piecewise exponential model, kg/m^3.

    Parameters
    ----------
    altitude_km
        Geodetic height above the WGS84 ellipsoid.

    Notes
    -----
    The model is continuous within each band and **discontinuous at the band
    boundaries**, by construction of the published table: it is a set of 28
    independent exponential fits, not a single smooth function. The largest
    jump across a boundary over 100-1000 km is measured and reported in
    ``validation/validate_atmosphere.py``. A discontinuous density makes the
    right-hand side of the equations of motion discontinuous, which costs a
    fixed-step Runge-Kutta method its formal order at the instant of crossing.
    That is why the step-refinement study in
    ``validation/validate_step_refinement.py`` is run with drag switched off
    for the order measurement and separately with drag on, and both numbers
    are published.
    """
    h = np.asarray(altitude_km, dtype=float)
    base = _ATMOSPHERE_TABLE[:, 0]
    rho0 = _ATMOSPHERE_TABLE[:, 1]
    scale = _ATMOSPHERE_TABLE[:, 2]
    index = np.clip(np.searchsorted(base, h, side="right") - 1, 0, base.size - 1)
    return rho0[index] * np.exp(-(h - base[index]) / scale[index])


def cd_area_over_mass_m2_kg(bstar: np.ndarray | float) -> np.ndarray:
    """Recover the drag parameter ``C_D A / m`` in m^2/kg from a TLE ``B*``.

    From ``B* = (1/2) C_D (A/m) rho_0`` with ``rho_0`` the reference density in
    :data:`orbitdrift.constants.BSTAR_REFERENCE_DENSITY_KG_M2_ER`, the product
    that drag actually needs is

        C_D A / m = 2 B* / rho_0 = 12.7389 B*   m^2/kg,

    which requires no assumed drag coefficient at all, because ``C_D`` appears
    on both sides and cancels. Splitting the product into a separate ``C_D``
    and ``A/m`` would require an assumption; the product does not, and this
    package never makes that split.

    What remains an assumption is ``rho_0`` itself and the premise that an
    operational ``B*`` is a physical ballistic parameter. It is not: orbit
    determination lets ``B*`` absorb whatever else the fit could not explain,
    including negative values for objects whose semi-major axis was observed
    to rise. A negative ``B*`` is passed through unchanged, giving a negative
    drag parameter and therefore an along-track thrust, because clipping it
    would silently change the dynamics of the object the user supplied. The
    sensitivity of every divergence number in this repository to a factor-of-
    two error in this quantity is measured in
    ``validation/validate_bstar_sensitivity.py``.
    """
    return 2.0 * np.asarray(bstar, dtype=float) / BSTAR_REFERENCE_DENSITY_KG_M2_ER


def zonal_potential(
    r_km: np.ndarray, gravity: GravityModel = WGS72
) -> np.ndarray:
    """Specific gravitational potential energy ``V`` with zonal terms, km^2/s^2.

    ``V(r) = -mu/|r| + mu * sum_n J_n Re^n P_n(z/|r|) / |r|^(n+1)``

    for ``n = 2, 3, 4``, with ``P_n`` the Legendre polynomials. The sign
    convention is that acceleration is ``-grad V``; the same convention and
    the same ``J_n`` sign as Spacetrack Report #3, where ``J2`` is positive.
    :func:`zonal_acceleration_km_s2` is the analytic gradient of exactly this
    function and ``tests/test_forces.py`` finite-differences it to check.
    """
    r = np.asarray(r_km, dtype=float)
    rn = np.linalg.norm(r, axis=-1)
    u = r[..., 2] / rn
    total = -gravity.mu_km3_s2 / rn
    legendre = {
        2: 0.5 * (3.0 * u**2 - 1.0),
        3: 0.5 * (5.0 * u**3 - 3.0 * u),
        4: 0.125 * (35.0 * u**4 - 30.0 * u**2 + 3.0),
    }
    for degree, coefficient in gravity.zonal_coefficients:
        total = total + (
            gravity.mu_km3_s2
            * coefficient
            * gravity.radius_km**degree
            * legendre[degree]
            / rn ** (degree + 1)
        )
    return total


def _legendre_and_derivative(degree: int, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if degree == 2:
        return 0.5 * (3.0 * u**2 - 1.0), 3.0 * u
    if degree == 3:
        return 0.5 * (5.0 * u**3 - 3.0 * u), 0.5 * (15.0 * u**2 - 3.0)
    if degree == 4:
        return 0.125 * (35.0 * u**4 - 30.0 * u**2 + 3.0), 0.125 * (140.0 * u**3 - 60.0 * u)
    raise ValueError(f"zonal degree {degree} is not implemented (2, 3 and 4 only)")


def zonal_acceleration_km_s2(
    r_km: np.ndarray, gravity: GravityModel = WGS72, *, two_body: bool = True
) -> np.ndarray:
    """Acceleration ``-grad V`` from :func:`zonal_potential`, km/s^2.

    With ``u = z/r`` and ``T_n = mu J_n Re^n P_n(u) r^-(n+1)``,

        dT_n/dx = -mu J_n Re^n [ P_n'(u) z x r^-(n+4) + (n+1) P_n(u) x r^-(n+3) ]
        dT_n/dz = mu J_n Re^n [ P_n'(u) (r^2 - z^2) r^-(n+4)
                                - (n+1) P_n(u) z r^-(n+3) ]

    and ``y`` follows ``x`` by symmetry. For ``n = 2`` this reduces to the
    familiar ``(3/2) J2 mu Re^2 r^-5 [x(5 z^2/r^2 - 1), y(...), z(5 z^2/r^2 - 3)]``.

    Setting ``two_body=False`` returns the perturbing part alone, which is what
    the finite-difference check needs in order to see the small terms at all.
    """
    r = np.asarray(r_km, dtype=float)
    rn = np.linalg.norm(r, axis=-1)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    u = z / rn
    out = np.zeros_like(r)
    if two_body:
        out = out - gravity.mu_km3_s2 * r / rn[..., None] ** 3
    for degree, coefficient in gravity.zonal_coefficients:
        p_n, dp_n = _legendre_and_derivative(degree, u)
        factor = gravity.mu_km3_s2 * coefficient * gravity.radius_km**degree
        common = dp_n * z * rn ** (-(degree + 4)) + (degree + 1) * p_n * rn ** (-(degree + 3))
        out[..., 0] += factor * x * common
        out[..., 1] += factor * y * common
        out[..., 2] -= factor * (
            dp_n * (rn**2 - z**2) * rn ** (-(degree + 4))
            - (degree + 1) * p_n * z * rn ** (-(degree + 3))
        )
    return out


def drag_acceleration_km_s2(
    r_km: np.ndarray,
    v_km_s: np.ndarray,
    jd_ut1: np.ndarray | float,
    cd_area_over_mass_m2_kg: np.ndarray | float,
) -> np.ndarray:
    """Atmospheric drag acceleration in TEME, km/s^2.

    ``a = -0.5 (C_D A / m) rho |v_rel| v_rel`` with ``v_rel`` the velocity
    relative to an atmosphere co-rotating rigidly with the Earth.

    The unit bookkeeping is written out because getting it wrong by a factor
    of 1000 is the easiest mistake in this module, and the first version of
    this function made it -- in the wrong direction, so the drag term was
    1e-6 of its true size and had no visible effect at all. ``C_D A / m`` is
    m^2/kg and ``rho`` is kg/m^3, so their product is m^-1. ``|v_rel| v_rel``
    is km^2/s^2 here, so the whole product is ``m^-1 km^2 s^-2``. Since
    ``1 m^-1 = 1000 km^-1``, that is ``1000 km s^-2``, and the numeric
    product is therefore **multiplied** by 1000 to give km/s^2.

    The defect survived its own unit test, because that test recomputed the
    same expression and compared. What found it was a negative control --
    switching the term off and requiring the answer to change -- and then an
    order-of-magnitude check against the drag acceleration computed
    independently in SI units. Both are now in ``tests/test_forces.py`` and
    ``tests/test_divergence.py``.

    Co-rotation is an assumption: thermospheric winds reach of order 100 m/s,
    which against an orbital speed of 7.5 km/s is a 1.3 per cent error in
    ``|v_rel|`` and so about 2.7 per cent in the drag magnitude.
    """
    r = np.asarray(r_km, dtype=float)
    v = np.asarray(v_km_s, dtype=float)
    altitude = geodetic_altitude_km(teme_to_ecef(r, jd_ut1))
    rho = atmospheric_density_kg_m3(altitude)
    v_rel = v - np.cross(earth_rotation_vector_rad_s(), r)
    speed = np.linalg.norm(v_rel, axis=-1)
    coefficient = np.asarray(cd_area_over_mass_m2_kg, dtype=float)
    return -0.5 * (coefficient * rho * speed * 1000.0)[..., None] * v_rel


def third_body_acceleration_km_s2(
    r_km: np.ndarray,
    jd_tt: np.ndarray | float,
    bodies: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Luni-solar point-mass acceleration in TEME, km/s^2.

    ``a = sum_b mu_b ( (r_b - r)/|r_b - r|^3 - r_b/|r_b|^3 )``
    (Montenbruck and Gill 2000, eq. 3.37). The two terms nearly cancel for
    ``|r| << |r_b|`` and the expression is kept in this form, rather than
    expanded, because the cancellation is between quantities of order
    ``mu_b/|r_b|^2`` while the result is of order ``2 mu_b |r| / |r_b|^3`` --
    a relative cancellation of ``|r|/|r_b|``, about 1e-4 for LEO against the
    Sun. In double precision that costs four of sixteen digits, which is
    tolerable and is **measured** in ``validation/validate_third_body.py``
    against a series expansion of the same quantity that has no cancellation
    in it. That check is included specifically because a property test
    comparing this function to itself could not find such a loss.
    """
    r = np.asarray(r_km, dtype=float)
    total = np.zeros_like(r)
    if bodies is None:
        bodies = (sun_position_teme_km(jd_tt), moon_position_teme_km(jd_tt))
    for position, mu_body in ((bodies[0], MU_SUN_KM3_S2), (bodies[1], MU_MOON_KM3_S2)):
        body = np.asarray(position, dtype=float)
        if body.ndim < r.ndim:
            body = np.broadcast_to(body, r.shape)
        delta = body - r
        total = total + mu_body * (
            delta / np.linalg.norm(delta, axis=-1, keepdims=True) ** 3
            - body / np.linalg.norm(body, axis=-1, keepdims=True) ** 3
        )
    return total


@dataclass(frozen=True)
class ForceModel:
    """Which terms are on, and the constants they use.

    Attributes
    ----------
    gravity
        Gravity model. The default is WGS72, the model SGP4 uses, so that the
        divergence measured against SGP4 is not contaminated by a constants
        mismatch.
    zonal
        Include ``J2``, ``J3``, ``J4``.
    drag
        Include atmospheric drag.
    third_body
        Include luni-solar point-mass terms.
    """

    gravity: GravityModel = WGS72
    zonal: bool = True
    drag: bool = True
    third_body: bool = True

    @property
    def is_conservative(self) -> bool:
        """``True`` when every active term derives from a time-independent potential.

        Drag is dissipative and the luni-solar terms are explicitly
        time-dependent, so either one makes the specific energy a
        non-constant. Only with both off are energy and the polar component of
        angular momentum conserved quantities, and
        ``validation/validate_conservation.py`` tests that claim in both
        directions.
        """
        return not self.drag and not self.third_body

    def label(self) -> str:
        """Short human-readable description of the active terms."""
        terms = ["two-body"]
        if self.zonal:
            terms.append("J2-J4")
        if self.drag:
            terms.append("drag")
        if self.third_body:
            terms.append("luni-solar")
        return f"{self.gravity.name}: " + " + ".join(terms)

    def acceleration_km_s2(
        self,
        r_km: np.ndarray,
        v_km_s: np.ndarray,
        jd: np.ndarray | float,
        cd_area_over_mass: np.ndarray | float,
        bodies: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Total acceleration, km/s^2, for state(s) of shape ``(..., 3)``.

        ``cd_area_over_mass`` is ``C_D A / m`` in m^2/kg, as returned by
        :func:`cd_area_over_mass_m2_kg`. It is ignored when ``drag`` is off.
        ``bodies`` is an optional pre-computed ``(sun, moon)`` position pair.
        """
        if self.zonal:
            total = zonal_acceleration_km_s2(r_km, self.gravity, two_body=True)
        else:
            r = np.asarray(r_km, dtype=float)
            total = (
                -self.gravity.mu_km3_s2 * r / np.linalg.norm(r, axis=-1, keepdims=True) ** 3
            )
        if self.drag:
            total = total + drag_acceleration_km_s2(r_km, v_km_s, jd, cd_area_over_mass)
        if self.third_body:
            total = total + third_body_acceleration_km_s2(r_km, jd, bodies)
        return total
