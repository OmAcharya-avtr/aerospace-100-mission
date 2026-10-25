"""Closed-form two-body propagation and element conversions.

This module exists to give the numerical integrator something to be checked
*against* rather than merely compared with. A property test that asserts two
numerical propagators agree shares their expression and cannot catch
catastrophic cancellation in it; a check against the closed-form Kepler
solution is a different class of evidence, because the closed form is derived
from the conic geometry and never evaluates the equations of motion at all.
``validation/validate_kepler_reference.py`` is that check.

The solver is Newton-Raphson on Kepler's equation for elliptical orbits, with
the starting guess and the convergence criterion stated. Hyperbolic and
parabolic orbits are rejected rather than handled: nothing in this package
propagates an escape trajectory, and silently returning a wrong answer for one
would be worse than refusing.

References
----------
Battin, R. H. (1999). *An Introduction to the Mathematics and Methods of
Astrodynamics*, revised edition, AIAA. Chapter 4 for the conic relations.
Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications*, 4th
ed. Algorithm 9 (``RV2COE``), Algorithm 10 (``COE2RV``) and Algorithm 2
(``KepEqtnE``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import WGS72
from .errors import PropagationError

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class ClassicalElements:
    """Classical orbital elements. Angles in radians, ``a`` and ``p`` in km."""

    semi_major_axis_km: float
    eccentricity: float
    inclination_rad: float
    raan_rad: float
    arg_perigee_rad: float
    true_anomaly_rad: float

    @property
    def semi_latus_rectum_km(self) -> float:
        return self.semi_major_axis_km * (1.0 - self.eccentricity**2)


def rv_to_elements(
    r_km: np.ndarray, v_km_s: np.ndarray, mu_km3_s2: float = WGS72.mu_km3_s2
) -> ClassicalElements:
    """Classical elements from a state vector (Vallado Algorithm 9).

    Accuracy of the recovered inclination near zero
    ------------------------------------------------
    Inclination is recovered as ``arccos(h_z / |h|)``, and
    ``d(arccos x)/dx = -1/sin(i)``, so a relative error of a few machine
    epsilons in the ratio becomes an absolute error of about ``eps / sin(i)``
    in the angle. For a nearly equatorial orbit that is not small: at
    ``i = 6e-08`` radian the floor is about 3e-08 radian, which is half the
    inclination itself, and the recovered value of a round trip through
    :func:`elements_to_rv` departs from the input by about 2e-09.

    This is a property of the formula and not a defect in it. The alternative,
    ``arctan2(|node|, h_z)``, is better conditioned near zero but worse near
    90 degrees, and this package's population is inclined. The limitation is
    stated rather than worked around, and ``tests/test_kepler.py`` derives its
    tolerance from the conditioning rather than choosing one.

    Raises
    ------
    PropagationError
        If the orbit is not elliptical (``e >= 1`` or non-positive energy
        denominator), or if the state is non-finite.
    """
    r = np.asarray(r_km, dtype=float)
    v = np.asarray(v_km_s, dtype=float)
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v))):
        raise PropagationError("state vector is not finite")
    rn = float(np.linalg.norm(r))
    vn = float(np.linalg.norm(v))
    if rn <= 0.0:
        raise PropagationError("position vector has zero magnitude")
    h = np.cross(r, v)
    hn = float(np.linalg.norm(h))
    if hn <= 0.0:
        raise PropagationError("angular momentum is zero; the orbit is degenerate")
    node = np.array([-h[1], h[0], 0.0])
    node_n = float(np.linalg.norm(node))
    energy = 0.5 * vn**2 - mu_km3_s2 / rn
    if energy >= 0.0:
        raise PropagationError(
            f"specific energy {energy:.6e} km^2/s^2 is non-negative; this module "
            "propagates elliptical orbits only"
        )
    a = -mu_km3_s2 / (2.0 * energy)
    e_vec = (np.cross(v, h) / mu_km3_s2) - r / rn
    ecc = float(np.linalg.norm(e_vec))
    if ecc >= 1.0:
        raise PropagationError(f"eccentricity {ecc:.6f} is not elliptical")
    inc = float(np.arccos(np.clip(h[2] / hn, -1.0, 1.0)))

    if node_n > 1e-12:
        raan = float(np.arctan2(node[1], node[0]) % _TWO_PI)
        if ecc > 1e-12:
            cos_argp = float(np.dot(node, e_vec) / (node_n * ecc))
            argp = float(np.arccos(np.clip(cos_argp, -1.0, 1.0)))
            if e_vec[2] < 0.0:
                argp = _TWO_PI - argp
        else:
            argp = 0.0
    else:
        # Equatorial: the node line is undefined. Measure the argument of
        # perigee from the x axis instead and set the node to zero, which is
        # the usual convention and is stated here because it changes the
        # meaning of two of the six returned numbers.
        raan = 0.0
        argp = float(np.arctan2(e_vec[1], e_vec[0]) % _TWO_PI) if ecc > 1e-12 else 0.0

    if ecc > 1e-12:
        cos_nu = float(np.dot(e_vec, r) / (ecc * rn))
        nu = float(np.arccos(np.clip(cos_nu, -1.0, 1.0)))
        if float(np.dot(r, v)) < 0.0:
            nu = _TWO_PI - nu
    else:
        reference = node / node_n if node_n > 1e-12 else np.array([1.0, 0.0, 0.0])
        cos_nu = float(np.dot(reference, r) / rn)
        nu = float(np.arccos(np.clip(cos_nu, -1.0, 1.0)))
        if r[2] < 0.0 and inc > 1e-12:
            nu = _TWO_PI - nu

    return ClassicalElements(a, ecc, inc, raan, argp, nu)


def elements_to_rv(
    elements: ClassicalElements, mu_km3_s2: float = WGS72.mu_km3_s2
) -> tuple[np.ndarray, np.ndarray]:
    """State vector from classical elements (Vallado Algorithm 10)."""
    p = elements.semi_latus_rectum_km
    ecc = elements.eccentricity
    nu = elements.true_anomaly_rad
    r_mag = p / (1.0 + ecc * np.cos(nu))
    r_pqw = np.array([r_mag * np.cos(nu), r_mag * np.sin(nu), 0.0])
    v_pqw = np.sqrt(mu_km3_s2 / p) * np.array([-np.sin(nu), ecc + np.cos(nu), 0.0])

    c_o, s_o = np.cos(elements.raan_rad), np.sin(elements.raan_rad)
    c_i, s_i = np.cos(elements.inclination_rad), np.sin(elements.inclination_rad)
    c_w, s_w = np.cos(elements.arg_perigee_rad), np.sin(elements.arg_perigee_rad)
    rot = np.array(
        [
            [c_o * c_w - s_o * s_w * c_i, -c_o * s_w - s_o * c_w * c_i, s_o * s_i],
            [s_o * c_w + c_o * s_w * c_i, -s_o * s_w + c_o * c_w * c_i, -c_o * s_i],
            [s_w * s_i, c_w * s_i, c_i],
        ]
    )
    return rot @ r_pqw, rot @ v_pqw


def mean_to_true_anomaly(mean_anomaly_rad: float, ecc: float, tol: float = 1e-13) -> float:
    """Solve Kepler's equation ``M = E - e sin E`` and return true anomaly.

    Newton-Raphson from the standard starting guess ``E0 = M`` for ``e < 0.8``
    and ``E0 = pi`` above it (Vallado Algorithm 2). Convergence is on the
    increment, ``|dE| < tol``, with a hard cap of 80 iterations. The returned
    angle is in ``[0, 2*pi)``.

    Raises
    ------
    PropagationError
        If the iteration does not converge within 80 steps, or ``ecc`` is not
        in ``[0, 1)``.
    """
    if not (0.0 <= ecc < 1.0):
        raise PropagationError(f"eccentricity {ecc} is not in [0, 1)")
    m = float(mean_anomaly_rad) % _TWO_PI
    e_anom = m if ecc < 0.8 else np.pi
    for _ in range(80):
        denominator = 1.0 - ecc * np.cos(e_anom)
        step = (m - e_anom + ecc * np.sin(e_anom)) / denominator
        e_anom += step
        if abs(step) < tol:
            break
    else:
        raise PropagationError(
            f"Kepler's equation did not converge for M={m:.6f}, e={ecc:.6f}"
        )
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(0.5 * e_anom),
        np.sqrt(1.0 - ecc) * np.cos(0.5 * e_anom),
    )
    return float(nu % _TWO_PI)


def propagate_two_body(
    r0_km: np.ndarray,
    v0_km_s: np.ndarray,
    times_s: np.ndarray,
    mu_km3_s2: float = WGS72.mu_km3_s2,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact two-body propagation of one state to many times.

    The conic is fixed once from the initial state; each requested time only
    advances the mean anomaly. No differential equation is integrated, so this
    is a genuinely independent reference for a numerical propagator run with
    only the two-body term enabled.

    Returns
    -------
    (positions, velocities)
        Arrays of shape ``(len(times_s), 3)``, km and km/s.
    """
    elements = rv_to_elements(r0_km, v0_km_s, mu_km3_s2)
    ecc = elements.eccentricity
    nu0 = elements.true_anomaly_rad
    e_anom0 = 2.0 * np.arctan2(
        np.sqrt(1.0 - ecc) * np.sin(0.5 * nu0), np.sqrt(1.0 + ecc) * np.cos(0.5 * nu0)
    )
    m0 = e_anom0 - ecc * np.sin(e_anom0)
    n = np.sqrt(mu_km3_s2 / elements.semi_major_axis_km**3)

    times = np.atleast_1d(np.asarray(times_s, dtype=float))
    positions = np.empty((times.size, 3))
    velocities = np.empty((times.size, 3))
    for index, t in enumerate(times):
        nu = mean_to_true_anomaly(m0 + n * t, ecc)
        r, v = elements_to_rv(
            ClassicalElements(
                elements.semi_major_axis_km,
                ecc,
                elements.inclination_rad,
                elements.raan_rad,
                elements.arg_perigee_rad,
                nu,
            ),
            mu_km3_s2,
        )
        positions[index] = r
        velocities[index] = v
    return positions, velocities


def orbital_period_s(
    r_km: np.ndarray, v_km_s: np.ndarray, mu_km3_s2: float = WGS72.mu_km3_s2
) -> float:
    """Keplerian period of the osculating orbit, seconds."""
    elements = rv_to_elements(r_km, v_km_s, mu_km3_s2)
    return float(_TWO_PI * np.sqrt(elements.semi_major_axis_km**3 / mu_km3_s2))
