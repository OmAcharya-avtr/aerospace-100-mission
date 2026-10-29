"""The observability verdict. Computed before any estimate, and returned as output.

This is the module the product exists for. Every estimator in
:mod:`aligncal.batch` and :mod:`aligncal.recursive` takes an
:class:`ObservabilityReport` as a required argument, so the verdict cannot be
produced after the estimate it gates.

The Gramian
-----------
From equation (7) of :mod:`aligncal.record`, the residual of secondary ``j`` is
``r_j(t) = [w(t)]x theta_j + e_j`` with ``e_j`` white of variance ``s_j^2`` per
axis. The Fisher information of ``theta_j``, which is also the least-squares
normal matrix and the observability Gramian of the time-invariant state
``theta_j``, is

    (8)   G_j = (1 / s_j^2) sum_t [w_t]x^T [w_t]x
              = (1 / s_j^2) sum_t ( |w_t|^2 I - w_t w_t^T )

using identity (3) of :mod:`aligncal.geometry`. Units: s^-2 per rad^2, so that
``G^-1`` carries rad^2 and its square root is an angle.

The structural result this product reports
------------------------------------------
Write the normalised rate second moment

    (9)   M = ( sum_t w_t w_t^T ) / ( sum_t |w_t|^2 ),   trace(M) = 1

Then from (8), ``G_j = (W / s_j^2) (I - M)`` with ``W = sum_t |w_t|^2`` in
rad^2/s^2. ``G_j`` and ``M`` therefore share eigenvectors exactly, and

    (10)  lambda_i(G_j) = (W / s_j^2) ( 1 - p_i ),   p_i = eigenvalue of M

so the **least observable** alignment direction is the **most excited**
rotation axis, and a record whose motion is a rotation about one fixed axis
``n`` has ``p = (1, 0, 0)`` in the basis ``(n, .., ..)`` and ``lambda = 0``
along ``n``. The unobservable direction is the rotation axis itself. That is
the sentence this product puts in front of the estimate, and (10) is exercised
as a property test.

The noise floor, which is why a rank test on the measured Gramian is wrong
--------------------------------------------------------------------------
``w(t)`` is not available. The Gramian is formed from the **measured**
reference rate ``w_A = w + n_A``, and the noise contributes

    (11)  E[ |n|^2 I - n n^T ] = (3 sigma_A^2 - sigma_A^2) I = 2 sigma_A^2 I

per sample, in every direction including the unobservable one. So the measured
Gramian of an exactly single-axis record is **not singular**: its smallest
eigenvalue sits at about ``2 n sigma_A^2 / s_j^2``, and a bare rank test on it
reports three observable degrees of freedom for a record that physically
supports two. The floor of (11) is computed and every eigenvalue is tested
against it. With the floor gate removed, a single-axis record is declared
fully observable on every seed measured; that control is in
``validation/validate_controls.py``.

The verdict is three-valued per degree of freedom:

``unobservable``
    ``lambda_i <= floor_factor * 2 n sigma_A^2 / s_j^2``. The apparent
    information is indistinguishable from what the reference sensor's own noise
    would produce on a record with no excitation at all about that axis.
``marginal``
    above the floor, but the Cramer-Rao standard deviation ``1/sqrt(lambda_i)``
    exceeds the declared requirement.
``observable``
    above the floor and within the declared requirement.

The Cramer-Rao bound
--------------------
For the Gaussian linear model (7) the Cramer-Rao bound on any unbiased
estimator of ``theta_j`` is ``G_j^-1``, so the bound on the component along a
Gramian eigenvector ``v_i`` is

    (12)  sigma_CRB,i = lambda_i^(-1/2)   radians

and the bound on the component along a canonical body axis ``k`` is
``sqrt( (G_j^+)_kk )`` restricted to the retained subspace. Both are reported.
Source: Kay, S. M. (1993), *Fundamentals of Statistical Signal Processing:
Estimation Theory*, Prentice Hall, section 3.9, the Cramer-Rao bound for the
general linear Gaussian model. In the unobservable directions the bound is
infinite and this module returns ``inf`` rather than a large number.

Independent brackets on the smallest eigenvalue
-----------------------------------------------
``lambda_min`` is computed by :func:`numpy.linalg.eigh`. It is independently
bracketed, by mathematics the eigensolver does not use, with

    (13)  min_k ( G_kk - sum_{m != k} |G_km| )  <=  lambda_min  <=  min_k G_kk

the left from Gershgorin's circle theorem (Gershgorin, S. A., 1931, *Izv. Akad.
Nauk SSSR Ser. Mat.* 6, 749-754; stated in Horn, R. A. and Johnson, C. R.,
2013, *Matrix Analysis*, 2nd ed., Cambridge, Theorem 6.1.1), the right from the
Rayleigh quotient at the canonical basis vectors (Horn and Johnson, Theorem
4.2.2). Diagonal dominance and Rayleigh quotients share no machinery with a
tridiagonal-reduction eigensolver, so this is a bound check in the sense the
mission's build guide requires, and it is run on every call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .geometry import angle_between, skew
from .record import AlignmentRecord

__all__ = [
    "AXIS_NAMES",
    "DEFAULT_FLOOR_FACTOR",
    "DEFAULT_REQUIRED_SIGMA_RAD",
    "DegreeOfFreedom",
    "ObservabilityReport",
    "SensorObservability",
    "assess",
    "dominant_rotation_axis",
    "eigenvalue_bracket",
    "excitation_spectrum",
    "gramian",
    "noise_floor_eigenvalue",
]

Verdict = Literal["observable", "marginal", "unobservable"]

#: Canonical body axes of the reference frame, used to name directions.
AXIS_NAMES: tuple[str, str, str] = ("x", "y", "z")

#: An eigenvalue must exceed this multiple of the noise floor (11) to count as
#: excitation rather than as the reference sensor's own noise. 4.0 is a declared
#: policy choice and is labelled as such wherever it is reported; the measured
#: sensitivity of the verdict to it is swept in
#: ``validation/validate_observability.py``.
DEFAULT_FLOOR_FACTOR: float = 4.0

#: Declared alignment-knowledge requirement, radians, one sigma. 1e-3 rad is
#: 206.26 arcsec. A declared requirement is the only thing that makes
#: "observable" mean anything; this default is a placeholder for the reader's
#: own number and is never presented as a specification.
DEFAULT_REQUIRED_SIGMA_RAD: float = 1.0e-3


def gramian(record: AlignmentRecord, name: str) -> np.ndarray:
    """Observability Gramian ``G_j`` of equation (8) for secondary ``name``.

    Formed from the **measured** reference rate, which is the only one
    available, so it carries the noise floor of (11). Units s^-2 per rad^2,
    shape ``(3, 3)``, symmetric positive semi-definite up to rounding.
    """
    rates = record.reference_rates
    weight = 1.0 / (record.pair_sigma(name) ** 2)
    squared = float(np.sum(rates * rates))
    outer = rates.T @ rates
    matrix = weight * (squared * np.eye(3) - outer)
    return 0.5 * (matrix + matrix.T)


def excitation_spectrum(record: AlignmentRecord) -> tuple[np.ndarray, np.ndarray, float]:
    """Eigenvalues and eigenvectors of ``M`` in (9), and ``W = sum |w_t|^2``.

    Returns ``(p, V, W)`` with ``p`` descending, ``V`` columns the matching unit
    eigenvectors, and ``W`` in rad^2/s^2. ``p`` sums to 1 and is the excitation
    richness of the record: ``(1, 0, 0)`` is a single-axis rotation and
    ``(1/3, 1/3, 1/3)`` is isotropic.
    """
    rates = record.reference_rates
    total = float(np.sum(rates * rates))
    if total <= 0.0:
        raise ValueError(
            "the reference sensor recorded no motion at all, so no alignment degree of "
            "freedom is excited and the excitation spectrum is undefined"
        )
    moment = (rates.T @ rates) / total
    moment = 0.5 * (moment + moment.T)
    values, vectors = np.linalg.eigh(moment)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order], total


def dominant_rotation_axis(record: AlignmentRecord) -> np.ndarray:
    """Unit vector of the most excited rotation axis, from ``M`` in (9).

    This is the axis the record rotated about most, and by (10) it is the
    direction the alignment is least observable in. The sign is fixed so the
    largest-magnitude component is positive, because ``M`` is invariant under
    ``n -> -n`` and an unstable sign would make every printed report differ.
    """
    _, vectors, _ = excitation_spectrum(record)
    axis = vectors[:, 0]
    if axis[int(np.argmax(np.abs(axis)))] < 0.0:
        axis = -axis
    return axis


def noise_floor_eigenvalue(record: AlignmentRecord, name: str) -> float:
    """Gramian eigenvalue the reference noise alone would produce, equation (11).

    ``2 n sigma_A^2 / s_j^2``, dimensionless in the units of (8). This is the
    level below which an apparent degree of freedom is the reference sensor's
    own noise rather than motion.
    """
    sigma = record.reference.noise_sigma
    return float(2.0 * len(record) * sigma * sigma / (record.pair_sigma(name) ** 2))


def eigenvalue_bracket(matrix: np.ndarray) -> tuple[float, float]:
    """Independent bracket ``(lower, upper)`` on ``lambda_min``, equation (13).

    Gershgorin on the left, the Rayleigh quotient at the canonical basis
    vectors on the right. Neither uses an eigensolver. The lower bound may be
    negative, which is informative rather than wrong: it says diagonal
    dominance alone cannot separate this matrix from a singular one.
    """
    array = np.asarray(matrix, dtype=float)
    if array.shape != (3, 3):
        raise ValueError(f"eigenvalue_bracket() needs a (3, 3) matrix, got {array.shape}")
    diagonal = np.diag(array)
    off = np.sum(np.abs(array), axis=1) - np.abs(diagonal)
    return float(np.min(diagonal - off)), float(np.min(diagonal))


def _render_sigma(value: float) -> str:
    """One-sigma in arcseconds, or the word ``unbounded``."""
    if not np.isfinite(value):
        return "unbounded"
    return f"{value * 206264.80624709636:.2f} arcsec"


@dataclass(frozen=True)
class DegreeOfFreedom:
    """One alignment degree of freedom: an eigendirection of the Gramian.

    Attributes
    ----------
    index:
        0 for the best-observed direction, 2 for the worst.
    direction:
        Unit vector in the reference frame, shape ``(3,)``.
    eigenvalue:
        ``lambda_i`` of (8), s^-2 per rad^2.
    crb_sigma_rad:
        ``lambda_i^(-1/2)`` of (12), radians. ``inf`` when the direction is
        unobservable.
    excitation_share:
        ``p_i`` of (9): the share of the record's rotational energy about this
        direction. The unobservable direction is the one with the largest
        share, by (10).
    floor_ratio:
        ``lambda_i`` divided by the noise floor of (11). At or below
        ``floor_factor`` the direction is declared unobservable.
    verdict:
        ``"observable"``, ``"marginal"`` or ``"unobservable"``.
    nearest_axis:
        Name of the canonical body axis closest to ``direction``, with the
        angle to it, for a human-readable description of which rotation the
        direction corresponds to.
    nearest_axis_angle_deg:
        Angle between ``direction`` and ``nearest_axis``, degrees.
    """

    index: int
    direction: np.ndarray
    eigenvalue: float
    crb_sigma_rad: float
    excitation_share: float
    floor_ratio: float
    verdict: Verdict
    nearest_axis: str
    nearest_axis_angle_deg: float

    @property
    def crb_sigma_arcsec(self) -> float:
        """The bound of (12) in arcseconds, or ``inf``."""
        return self.crb_sigma_rad * 180.0 * 3600.0 / np.pi

    def describe(self) -> str:
        """One line naming the rotation this degree of freedom corresponds to."""
        axis = (
            f"the {self.nearest_axis} axis"
            if self.nearest_axis_angle_deg < 5.0
            else f"{self.nearest_axis_angle_deg:.1f} deg from the {self.nearest_axis} axis"
        )
        bound = (
            "unbounded"
            if not np.isfinite(self.crb_sigma_rad)
            else f"{self.crb_sigma_arcsec:.2f} arcsec"
        )
        return (
            f"dof {self.index}: rotation about {axis} "
            f"[{', '.join(f'{c:+.4f}' for c in self.direction)}] -- "
            f"{self.verdict}, CRB {bound}, excitation share {self.excitation_share:.6f}, "
            f"floor ratio {self.floor_ratio:.4g}"
        )


@dataclass(frozen=True)
class SensorObservability:
    """The observability verdict for one secondary sensor."""

    name: str
    gramian: np.ndarray
    degrees_of_freedom: tuple[DegreeOfFreedom, ...]
    noise_floor: float
    floor_factor: float
    required_sigma_rad: float
    dominant_axis: np.ndarray
    unobservable_axis_angle_deg: float
    condition_number: float
    bracket: tuple[float, float]
    bracket_consistent: bool
    samples: int
    rotational_energy: float

    @property
    def rank(self) -> int:
        """Number of degrees of freedom above the noise floor of (11)."""
        return sum(1 for dof in self.degrees_of_freedom if dof.verdict != "unobservable")

    @property
    def rank_deficient(self) -> bool:
        """True when fewer than three degrees of freedom clear the noise floor."""
        return self.rank < 3

    @property
    def observable_mask(self) -> np.ndarray:
        """Boolean ``(3,)`` over :attr:`degrees_of_freedom`, floor-clearing ones True."""
        return np.array(
            [dof.verdict != "unobservable" for dof in self.degrees_of_freedom], dtype=bool
        )

    @property
    def basis(self) -> np.ndarray:
        """``(3, 3)`` matrix whose columns are the degree-of-freedom directions."""
        return np.column_stack([dof.direction for dof in self.degrees_of_freedom])

    @property
    def unobservable_directions(self) -> tuple[DegreeOfFreedom, ...]:
        """Degrees of freedom the record cannot support, in report order."""
        return tuple(d for d in self.degrees_of_freedom if d.verdict == "unobservable")

    @property
    def marginal_directions(self) -> tuple[DegreeOfFreedom, ...]:
        """Degrees of freedom above the floor but outside the declared requirement."""
        return tuple(d for d in self.degrees_of_freedom if d.verdict == "marginal")

    def axis_crb_sigma_rad(self) -> np.ndarray:
        """Per-canonical-axis Cramer-Rao one-sigma, radians, shape ``(3,)``.

        ``sqrt(diag(G^+))`` with the pseudo-inverse taken over the retained
        subspace only. An axis with any projection onto an unobservable
        direction gets ``inf``: the bound on that axis really is unbounded, and
        a large finite number there would be a silent wrong answer.
        """
        basis = self.basis
        values = np.array([d.eigenvalue for d in self.degrees_of_freedom], dtype=float)
        mask = self.observable_mask
        out = np.zeros(3)
        for axis in range(3):
            projection = basis[axis, :]
            if np.any(np.abs(projection[~mask]) > 1.0e-12):
                out[axis] = np.inf
                continue
            out[axis] = float(
                np.sqrt(np.sum(projection[mask] ** 2 / values[mask])) if np.any(mask) else np.inf
            )
        return out

    def project_observable(self, vector: np.ndarray) -> np.ndarray:
        """Project ``vector`` onto the span of the floor-clearing directions."""
        array = np.asarray(vector, dtype=float).reshape(3)
        basis = self.basis[:, self.observable_mask]
        if basis.shape[1] == 0:
            return np.zeros(3)
        return basis @ (basis.T @ array)

    def project_unobservable(self, vector: np.ndarray) -> np.ndarray:
        """Project ``vector`` onto the span of the directions below the floor."""
        return np.asarray(vector, dtype=float).reshape(3) - self.project_observable(vector)

    def summary(self) -> list[str]:
        """The verdict as text, unobservable directions first."""
        lines = [
            f"sensor {self.name}: rank {self.rank} of 3"
            + (" -- RANK DEFICIENT" if self.rank_deficient else ""),
            f"  samples {self.samples}, rotational energy {self.rotational_energy:.6g} rad^2/s^2",
            f"  noise floor (eq 11) {self.noise_floor:.6g}, "
            f"declared floor factor {self.floor_factor:g}, "
            f"declared requirement {self.required_sigma_rad:.3g} rad",
            f"  Gramian condition number {self.condition_number:.6g}",
            f"  lambda_min bracket (eq 13) [{self.bracket[0]:.6g}, {self.bracket[1]:.6g}], "
            f"consistent: {self.bracket_consistent}",
        ]
        if self.unobservable_directions:
            lines.append(
                "  the record does NOT support these degrees of freedom at any data volume:"
            )
            for dof in self.unobservable_directions:
                lines.append(f"    {dof.describe()}")
            lines.append(
                f"    the dominant rotation axis of this record is "
                f"[{', '.join(f'{c:+.4f}' for c in self.dominant_axis)}]; the worst-observed "
                f"direction lies {self.unobservable_axis_angle_deg:.4f} deg from it"
            )
        else:
            lines.append("  every degree of freedom clears the noise floor of eq (11)")
        for dof in self.degrees_of_freedom:
            lines.append(f"    {dof.describe()}")
        axis_sigma = self.axis_crb_sigma_rad()
        rendered = ", ".join(
            f"{AXIS_NAMES[k]} " + _render_sigma(axis_sigma[k]) for k in range(3)
        )
        lines.append(f"  per-body-axis CRB one sigma: {rendered}")
        return lines


@dataclass(frozen=True)
class ObservabilityReport:
    """Observability of every secondary in a record. The first output of every call.

    Attributes
    ----------
    sensors:
        Mapping from secondary name to :class:`SensorObservability`.
    excitation:
        ``p`` of equation (9), descending. Shared by every sensor because it is
        a property of the motion, not of the instrument.
    dominant_axis:
        Unit vector of the most excited rotation axis.
    """

    sensors: dict[str, SensorObservability]
    excitation: np.ndarray
    dominant_axis: np.ndarray
    samples: int
    duration_s: float
    provenance: str

    def __getitem__(self, name: str) -> SensorObservability:
        if name not in self.sensors:
            raise KeyError(f"unknown secondary {name!r}; have {sorted(self.sensors)}")
        return self.sensors[name]

    @property
    def any_rank_deficient(self) -> bool:
        """True when any secondary has a degree of freedom below the noise floor."""
        return any(s.rank_deficient for s in self.sensors.values())

    @property
    def excitation_entropy(self) -> float:
        """Shannon entropy of ``p`` in (9), in bits, range ``[0, log2 3]``.

        0 bits is a single-axis rotation and 1.585 bits is isotropic motion.
        This is the one scalar summary of excitation richness used in the
        feature set of :mod:`aligncal.features`.
        """
        share = np.clip(self.excitation, 1.0e-300, None)
        return float(-np.sum(share * np.log2(share)))

    def summary(self) -> list[str]:
        """The whole verdict as text, structural facts first."""
        lines = [
            "OBSERVABILITY VERDICT (computed before any estimate)",
            f"  provenance        : {self.provenance}",
            f"  samples           : {self.samples}, duration {self.duration_s:.6g} s",
            "  excitation shares : "
            + ", ".join(f"p{i} {v:.6f}" for i, v in enumerate(self.excitation)),
            f"  excitation entropy: {self.excitation_entropy:.6f} bits of a possible 1.584963",
            "  dominant rotation axis: ["
            + ", ".join(f"{c:+.4f}" for c in self.dominant_axis)
            + "]",
        ]
        for name in sorted(self.sensors):
            lines.extend(self.sensors[name].summary())
        return lines


def assess(
    record: AlignmentRecord,
    *,
    required_sigma_rad: float = DEFAULT_REQUIRED_SIGMA_RAD,
    floor_factor: float = DEFAULT_FLOOR_FACTOR,
) -> ObservabilityReport:
    """Compute the observability verdict for every secondary. Call this first.

    Parameters
    ----------
    record:
        The measurements.
    required_sigma_rad:
        Declared alignment-knowledge requirement, one sigma, radians. A degree
        of freedom above the noise floor whose Cramer-Rao bound (12) exceeds
        this is reported ``marginal``, not ``observable``.
    floor_factor:
        Multiple of the noise floor (11) an eigenvalue must exceed to count as
        excitation. Dimensionless, must be at least 1.

    Returns
    -------
    ObservabilityReport
        Degrees of freedom ordered best-observed first, so index 2 is the one
        to read.

    Raises
    ------
    ValueError
        If the record contains no motion, or either threshold is unusable.
    """
    if not np.isfinite(required_sigma_rad) or required_sigma_rad <= 0.0:
        raise ValueError(
            f"required_sigma_rad must be finite and > 0 rad, got {required_sigma_rad!r}"
        )
    if not np.isfinite(floor_factor) or floor_factor < 1.0:
        raise ValueError(
            "floor_factor must be at least 1: a factor below 1 declares a direction "
            "observable on less information than the reference noise supplies, got "
            f"{floor_factor!r}"
        )
    shares, _, energy = excitation_spectrum(record)
    axis = dominant_rotation_axis(record)
    sensors: dict[str, SensorObservability] = {}
    for name in record.names:
        sensors[name] = _assess_one(record, name, axis, required_sigma_rad, floor_factor)
    return ObservabilityReport(
        sensors=sensors,
        excitation=shares,
        dominant_axis=axis,
        samples=len(record),
        duration_s=record.duration_s,
        provenance=record.provenance,
    )


def _assess_one(
    record: AlignmentRecord,
    name: str,
    dominant: np.ndarray,
    required_sigma_rad: float,
    floor_factor: float,
) -> SensorObservability:
    matrix = gramian(record, name)
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    floor = noise_floor_eigenvalue(record, name)
    energy = float(np.sum(record.reference_rates * record.reference_rates))
    weight = 1.0 / (record.pair_sigma(name) ** 2)
    scale = energy * weight

    dofs: list[DegreeOfFreedom] = []
    for index in range(3):
        value = float(values[index])
        direction = vectors[:, index]
        if direction[int(np.argmax(np.abs(direction)))] < 0.0:
            direction = -direction
        ratio = value / floor if floor > 0.0 else np.inf
        if ratio <= floor_factor or value <= 0.0:
            verdict: Verdict = "unobservable"
            sigma = np.inf
        else:
            sigma = float(value ** -0.5)
            verdict = "observable" if sigma <= required_sigma_rad else "marginal"
        # p_i from (10): lambda = scale (1 - p), so p = 1 - lambda / scale.
        share = float(np.clip(1.0 - value / scale, 0.0, 1.0)) if scale > 0.0 else 0.0
        nearest = int(np.argmax(np.abs(direction)))
        dofs.append(
            DegreeOfFreedom(
                index=index,
                direction=direction,
                eigenvalue=value,
                crb_sigma_rad=sigma,
                excitation_share=share,
                floor_ratio=float(ratio),
                verdict=verdict,
                nearest_axis=AXIS_NAMES[nearest],
                nearest_axis_angle_deg=float(
                    np.degrees(angle_between(direction, np.eye(3)[nearest]))
                ),
            )
        )

    lower, upper = eigenvalue_bracket(matrix)
    smallest = float(values[-1])
    low_slack = 1.0e-9 * max(1.0, abs(lower))
    high_slack = 1.0e-9 * max(1.0, abs(upper))
    consistent = bool(lower - low_slack <= smallest <= upper + high_slack)
    largest = float(values[0])
    condition = float(largest / smallest) if smallest > 0.0 else np.inf
    worst = dofs[-1].direction
    try:
        separation = float(np.degrees(angle_between(worst, dominant)))
    except ValueError:  # pragma: no cover - dominant axis is a unit vector
        separation = float("nan")
    separation = min(separation, 180.0 - separation)
    return SensorObservability(
        name=name,
        gramian=matrix,
        degrees_of_freedom=tuple(dofs),
        noise_floor=floor,
        floor_factor=float(floor_factor),
        required_sigma_rad=float(required_sigma_rad),
        dominant_axis=dominant,
        unobservable_axis_angle_deg=separation,
        condition_number=condition,
        bracket=(lower, upper),
        bracket_consistent=consistent,
        samples=len(record),
        rotational_energy=energy,
    )


def sensitivity_block(record: AlignmentRecord) -> np.ndarray:
    """Stacked sensitivity ``H(t) = [w_A(t)]x`` of (7), shape ``(n, 3, 3)``, 1/s.

    Exposed because every estimator in this package needs it and it should have
    exactly one definition.
    """
    return skew(record.reference_rates)
