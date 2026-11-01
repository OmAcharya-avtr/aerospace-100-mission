"""An exact rotation-composition forward model, and the bound that checks it.

Why this module exists
----------------------
:mod:`pointmodel.terms` writes the design matrix as nine closed-form columns,
each the first-order sky-plane signature of one physical misalignment. Nothing
in that module can detect a wrong sign or a wrong trigonometric factor, because
every check available inside it shares the algebra being checked. So this
module builds the same mount a different way: as an exact composition of
rotation matrices, with no small-angle approximation anywhere, and the two are
compared.

The comparison is not "two implementations agree to a tolerance I chose". It is
a check against a **bound derived from different mathematics**: the operator-
norm Taylor remainder of the matrix exponential, which the design matrix does
not use and which fixes the tolerance rather than leaving it to be picked.

The exact model
---------------
Rotations, right-handed, in the local north-east-up frame of equation (7):

    Rx(t), Ry(t), Rz(t)                                                   (11)

The beam direction for encoder readings ``(A, E)`` and coefficients ``theta``
is the composition, innermost first:

    v = Tilt(AN, AW) . Rz(A + IA) . Rx(-NPAE) . Ry(-(E + dE_el)) . Rz(CA) . xhat
                                                                          (12)
    dE_el = IE + TF cos(E) + TS sin(E) + RF rho_hat(E)                    (13)

with ``Tilt(AN, AW)`` the exact Rodrigues rotation by ``theta = hypot(AN, AW)``
about the unit axis along ``[AW, -AN, 0]``, and ``xhat = [1, 0, 0]``. Every
angle in (12) and (13) is in radians; the coefficients arrive in arcseconds and
are converted once.

The order of (12) is the physical order of the mechanism: collimation is fixed
in the beam, the elevation rotation carries it, the elevation axis sits on the
azimuth structure at the non-perpendicularity angle, the azimuth rotation
carries all of that, and the whole mount sits on a base whose vertical axis is
tilted. Reordering any adjacent pair changes the result at second order in the
coefficients and not at first order, which is exactly the size of the remainder
bounded below.

The bound
---------
Write (12) as ``v = G . prod_i exp(C_i) . xhat`` where ``G`` collects the two
nominal rotations ``Rz(A)`` and ``Ry(-E)``, which are not small, and the ``C_i``
are the skew-symmetric generators of the five small rotations: azimuth index,
elevation offset (13), non-perpendicularity, collimation, and base tilt.
Conjugating a generator by an orthogonal matrix preserves its spectral norm,
so with

    s = sum_i ||C_i||  =  |IA| + |dE_el| + |NPAE| + |CA| + hypot(AN, AW)   (14)

in radians, and using ``||exp(C) - I - C|| <= ||C||^2 exp(||C||) / 2``
(immediate from the series) together with the product expansion
``prod (I + C_i + E_i)``, every term of total degree two or more is bounded by

    B(s) = exp( s + s^2 exp(s) / 2 ) - 1 - s.                              (15)

Since ``|xhat| = 1`` and ``G`` is orthogonal,

    || v_exact - (v_nominal + dX * e_X + dE * e_E) ||_2  <=  B(s)          (16)

where ``dX`` and ``dE`` are the first-order residuals from the design matrix of
:mod:`pointmodel.terms` in radians and ``e_X``, ``e_E`` are the orthonormal
tangent vectors at ``(A, E)``,

    e_X = [-sin(A), cos(A), 0],
    e_E = [-sin(E) cos(A), -sin(E) sin(A), cos(E)].                        (17)

Equation (16) is a genuine bound, not a tolerance: it has no fitted constant
and no slack term. Its slack follows the conditioning of the problem through
``s^2`` in (15) — double the misalignments and the admissible error quadruples
— rather than through an absolute number. For small ``s`` it is
``B(s) ~ s^2``, so at a total misalignment of 100 arcsec it admits
2.35e-7 radians and at 1000 arcsec it admits 2.35e-5 radians.

Measured in ``validation/validate_linearisation.py``: the worst observed ratio
of the left side of (16) to ``B(s)`` is reported there, is between 0 and 1 by
construction if the design matrix is right, and sits near 0.44 across four
decades of misalignment amplitude. A wrong sign on any column takes that ratio
far above 1 immediately, which is checked as a negative control.
"""

from __future__ import annotations

import numpy as np

from .refraction import STANDARD_PRESSURE_HPA, STANDARD_TEMPERATURE_C, refraction_shape
from .terms import N_TERMS, TERM_INDEX, design_matrix

__all__ = [
    "ARCSEC_PER_RAD",
    "RAD_PER_ARCSEC",
    "beam_direction",
    "exact_residuals",
    "linearisation_bound",
    "linearisation_error",
    "sky_vector",
    "tangent_basis",
    "total_small_angle",
]

#: Radians per arcsecond and its reciprocal.
RAD_PER_ARCSEC = np.pi / (180.0 * 3600.0)
ARCSEC_PER_RAD = 1.0 / RAD_PER_ARCSEC


def _rot_x(angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])


def _rot_y(angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])


def _rot_z(angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Exact rotation by ``angle`` radians about ``axis`` (need not be unit)."""
    vector = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or angle == 0.0:
        return np.eye(3)
    unit = vector / norm
    cross = np.array(
        [
            [0.0, -unit[2], unit[1]],
            [unit[2], 0.0, -unit[0]],
            [-unit[1], unit[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def sky_vector(azimuth_rad: float, elevation_rad: float) -> np.ndarray:
    """Equation (7): the unit sky vector in north-east-up, dimensionless."""
    return np.array(
        [
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.cos(elevation_rad) * np.sin(azimuth_rad),
            np.sin(elevation_rad),
        ]
    )


def tangent_basis(azimuth_rad: float, elevation_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Equation (17): the orthonormal cross-elevation and elevation tangents."""
    cross = np.array([-np.sin(azimuth_rad), np.cos(azimuth_rad), 0.0])
    elev = np.array(
        [
            -np.sin(elevation_rad) * np.cos(azimuth_rad),
            -np.sin(elevation_rad) * np.sin(azimuth_rad),
            np.cos(elevation_rad),
        ]
    )
    return cross, elev


def _elevation_offset_rad(
    coefficients: np.ndarray,
    elevation_rad: float,
    pressure_hpa: float,
    temperature_c: float,
) -> float:
    """Equation (13) in radians, from coefficients in arcseconds."""
    index = TERM_INDEX
    rho = float(refraction_shape(np.array([elevation_rad]), pressure_hpa, temperature_c)[0])
    arcsec = (
        coefficients[index["IE"]]
        + coefficients[index["TF"]] * np.cos(elevation_rad)
        + coefficients[index["TS"]] * np.sin(elevation_rad)
        + coefficients[index["RF"]] * rho
    )
    return float(arcsec) * RAD_PER_ARCSEC


def beam_direction(
    azimuth_rad: float,
    elevation_rad: float,
    coefficients: np.ndarray,
    pressure_hpa: float = STANDARD_PRESSURE_HPA,
    temperature_c: float = STANDARD_TEMPERATURE_C,
) -> np.ndarray:
    """Equation (12): the exact beam direction, a unit vector, dimensionless.

    ``coefficients`` is a length-9 vector in arcseconds in the column order of
    :data:`pointmodel.terms.TERM_NAMES`. No small-angle approximation is made:
    every rotation is a full matrix and the composition is exact to floating
    point.
    """
    values = np.asarray(coefficients, dtype=float)
    if values.shape != (N_TERMS,):
        raise ValueError(f"expected a length-{N_TERMS} coefficient vector, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("coefficients contain a non-finite value")
    index = TERM_INDEX
    vector = _rot_z(values[index["CA"]] * RAD_PER_ARCSEC) @ np.array([1.0, 0.0, 0.0])
    effective_el = elevation_rad + _elevation_offset_rad(
        values, elevation_rad, pressure_hpa, temperature_c
    )
    vector = _rot_y(-effective_el) @ vector
    vector = _rot_x(-values[index["NPAE"]] * RAD_PER_ARCSEC) @ vector
    vector = _rot_z(azimuth_rad + values[index["IA"]] * RAD_PER_ARCSEC) @ vector
    tilt_north = values[index["AN"]] * RAD_PER_ARCSEC
    tilt_east = values[index["AW"]] * RAD_PER_ARCSEC
    magnitude = float(np.hypot(tilt_north, tilt_east))
    vector = _rodrigues(np.array([tilt_east, -tilt_north, 0.0]), magnitude) @ vector
    return vector


def exact_residuals(
    azimuth_rad: np.ndarray,
    elevation_rad: np.ndarray,
    coefficients: np.ndarray,
    pressure_hpa: float = STANDARD_PRESSURE_HPA,
    temperature_c: float = STANDARD_TEMPERATURE_C,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact ``(dX, dE)`` residuals in arcseconds from equation (12).

    These are the residuals an ideal instrument with exactly these
    misalignments would produce, computed without any linearisation: the
    azimuth difference is taken as an angle difference wrapped to
    ``(-pi, pi]`` and multiplied by ``cos(E)`` per equation (8), and the
    elevation difference is taken directly.
    """
    azimuth = np.atleast_1d(np.asarray(azimuth_rad, dtype=float))
    elevation = np.atleast_1d(np.asarray(elevation_rad, dtype=float))
    if azimuth.size != elevation.size:
        raise ValueError(
            f"azimuth has {azimuth.size} entries but elevation has {elevation.size}"
        )
    cross = np.empty(azimuth.size, dtype=float)
    elev = np.empty(azimuth.size, dtype=float)
    for position in range(azimuth.size):
        vector = beam_direction(
            float(azimuth[position]),
            float(elevation[position]),
            coefficients,
            pressure_hpa,
            temperature_c,
        )
        azimuth_out = float(np.arctan2(vector[1], vector[0]))
        elevation_out = float(np.arcsin(np.clip(vector[2], -1.0, 1.0)))
        delta_az = (azimuth_out - float(azimuth[position]) + np.pi) % (2.0 * np.pi) - np.pi
        cross[position] = delta_az * np.cos(float(elevation[position])) * ARCSEC_PER_RAD
        elev[position] = (elevation_out - float(elevation[position])) * ARCSEC_PER_RAD
    return cross, elev


def total_small_angle(
    coefficients: np.ndarray,
    elevation_rad: float,
    pressure_hpa: float = STANDARD_PRESSURE_HPA,
    temperature_c: float = STANDARD_TEMPERATURE_C,
) -> float:
    """Equation (14): the sum of the five small rotation magnitudes, radians."""
    values = np.asarray(coefficients, dtype=float)
    index = TERM_INDEX
    return float(
        abs(values[index["IA"]]) * RAD_PER_ARCSEC
        + abs(_elevation_offset_rad(values, elevation_rad, pressure_hpa, temperature_c))
        + abs(values[index["NPAE"]]) * RAD_PER_ARCSEC
        + abs(values[index["CA"]]) * RAD_PER_ARCSEC
        + float(np.hypot(values[index["AN"]], values[index["AW"]])) * RAD_PER_ARCSEC
    )


def linearisation_bound(total_angle_rad: float) -> float:
    """Equation (15): ``B(s)``, dimensionless, an upper bound on a vector norm.

    ``B`` is derived from the operator-norm Taylor remainder of the matrix
    exponential and the expansion of a product of such exponentials. It carries
    no fitted constant. It is monotonically increasing and ``B(0) = 0``.
    """
    s = float(total_angle_rad)
    if s < 0.0:
        raise ValueError(f"total_angle_rad must be non-negative, got {s}")
    return float(np.exp(s + 0.5 * s * s * np.exp(s)) - 1.0 - s)


def linearisation_error(
    azimuth_rad: np.ndarray,
    elevation_rad: np.ndarray,
    coefficients: np.ndarray,
    pressure_hpa: float = STANDARD_PRESSURE_HPA,
    temperature_c: float = STANDARD_TEMPERATURE_C,
) -> tuple[np.ndarray, np.ndarray]:
    """Left and right sides of equation (16), per observation, dimensionless.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(error, bound)``: the Euclidean distance between the exact beam
        vector of (12) and the first-order reconstruction from the design
        matrix, and the bound ``B(s)`` of (15). Both are dimensionless vector
        norms. ``error <= bound`` must hold for every observation; any
        violation is a defect in :mod:`pointmodel.terms`, in this module, or in
        the derivation of (15), and is reported rather than absorbed.
    """
    azimuth = np.atleast_1d(np.asarray(azimuth_rad, dtype=float))
    elevation = np.atleast_1d(np.asarray(elevation_rad, dtype=float))
    values = np.asarray(coefficients, dtype=float)
    cross_block, elev_block = design_matrix(azimuth, elevation, pressure_hpa, temperature_c)
    errors = np.empty(azimuth.size, dtype=float)
    bounds = np.empty(azimuth.size, dtype=float)
    for position in range(azimuth.size):
        exact = beam_direction(
            float(azimuth[position]),
            float(elevation[position]),
            values,
            pressure_hpa,
            temperature_c,
        )
        nominal = sky_vector(float(azimuth[position]), float(elevation[position]))
        tangent_cross, tangent_elev = tangent_basis(
            float(azimuth[position]), float(elevation[position])
        )
        linear = (
            nominal
            + float(cross_block[position] @ values) * RAD_PER_ARCSEC * tangent_cross
            + float(elev_block[position] @ values) * RAD_PER_ARCSEC * tangent_elev
        )
        errors[position] = float(np.linalg.norm(exact - linear))
        bounds[position] = linearisation_bound(
            total_small_angle(
                values, float(elevation[position]), pressure_hpa, temperature_c
            )
        )
    return errors, bounds
