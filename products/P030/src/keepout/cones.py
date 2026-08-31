"""Exclusion cones and the keep-out violation test.

An exclusion cone is a spherical cap on the sky: a unit axis and a half-angle.
A boresight direction violates the cone when the angle between boresight and
axis is smaller than the half-angle.

Angles are radians. Directions are unit vectors in a single common frame --
usually the spacecraft body frame, so the boresight is a fixed vector and the
cone axes move as the attitude changes; the geometry is identical if you
instead express both in an inertial frame.

References
----------
J. R. Wertz (ed.), *Spacecraft Attitude Determination and Control*, Reidel
    (1978), Sec. 5.2 and Ch. 11 -- cone/angle attitude geometry and sensor
    field-of-view exclusion.
D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.,
    Microcosm Press (2013), Sec. 11.7 -- sensor viewing geometry and Earth
    angular size.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import angular_separation, cap_solid_angle, unit

__all__ = ["ExclusionCone", "KeepOutSet", "body_exclusion_cone"]


@dataclass(frozen=True)
class ExclusionCone:
    """A forbidden cone of directions.

    Parameters
    ----------
    axis : array_like
        Cone axis, shape ``(3,)``; normalised on construction. Points at the
        centre of the excluded region (e.g. towards the Sun).
    half_angle : float
        Cone half-angle [rad], ``0 <= half_angle <= pi``. A boresight closer to
        ``axis`` than this is a violation.
    name : str
        Label used in violation reports, e.g. ``"sun"``.

    Notes
    -----
    Frozen and hashable-by-value except for the array field; treat instances as
    immutable. ``half_angle = 0`` is legal and excludes only the exact axis
    direction (measure zero).
    """

    axis: NDArray[np.float64]
    half_angle: float
    name: str = "cone"

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", unit(self.axis).reshape(3))
        ha = float(self.half_angle)
        if not np.isfinite(ha):
            raise ValueError("half_angle must be finite")
        if not 0.0 <= ha <= np.pi:
            raise ValueError(f"half_angle must lie in [0, pi] rad, got {ha}")
        object.__setattr__(self, "half_angle", ha)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")

    @property
    def half_angle_deg(self) -> float:
        """Cone half-angle [deg]."""
        return float(np.degrees(self.half_angle))

    @property
    def solid_angle(self) -> float:
        """Solid angle of the excluded cap [sr], ``2 pi (1 - cos(half_angle))``."""
        return float(cap_solid_angle(self.half_angle))

    def separation(self, boresight: ArrayLike) -> NDArray[np.float64] | float:
        """Angle between ``boresight`` and the cone axis [rad]."""
        return angular_separation(boresight, self.axis)

    def margin(self, boresight: ArrayLike) -> NDArray[np.float64] | float:
        """Clearance angle [rad]: ``separation - half_angle``.

        Positive means the boresight is outside the cone by that many radians;
        negative means it is inside by that much. The zero crossing is the cone
        boundary, which makes this the function to bisect when finding the
        instant a violation starts or ends.
        """
        return self.separation(boresight) - self.half_angle

    def contains(self, boresight: ArrayLike) -> NDArray[np.bool_] | bool:
        """``True`` where the boresight lies strictly inside the cone.

        The boundary (``separation == half_angle``) is *not* a violation, so a
        boresight exactly on the limit is reported as allowed. Floating-point
        equality on the boundary is not meaningful; use :meth:`margin` and your
        own tolerance if a boundary decision matters.
        """
        out = self.margin(boresight) < 0.0
        return bool(out) if np.ndim(out) == 0 else out

    def rotated(self, rotation: ArrayLike) -> ExclusionCone:
        """Return the same cone with its axis rotated by a ``(3, 3)`` matrix."""
        r = np.asarray(rotation, dtype=float)
        if r.shape != (3, 3):
            raise ValueError(f"rotation must have shape (3, 3), got {r.shape}")
        return replace(self, axis=r @ self.axis)


def body_exclusion_cone(
    name: str,
    body_direction: ArrayLike,
    body_angular_radius: float,
    exclusion_half_angle: float,
    reference: str = "limb",
) -> ExclusionCone:
    """Build a body exclusion cone from an instrument specification.

    Instrument keep-out angles are quoted two ways and the difference is the
    body's angular radius, which for the Earth in low orbit is about 70 deg --
    far too large to be absorbed by a margin.

    - ``reference="limb"``: the specified angle is the minimum permitted angle
      between the boresight and the nearest point of the body's *limb*. The cone
      half-angle is then ``body_angular_radius + exclusion_half_angle``.
    - ``reference="center"``: the specified angle is measured to the body's
      *centre* and is used as the cone half-angle unchanged.

    Parameters
    ----------
    name : str
        Cone label.
    body_direction : array_like
        Unit vector towards the body centre, shape ``(3,)``.
    body_angular_radius : float
        Apparent angular radius of the body [rad], ``>= 0``; see
        :func:`keepout.bodies.angular_radius`.
    exclusion_half_angle : float
        Instrument keep-out angle [rad], ``>= 0``.
    reference : {"limb", "center"}
        Which convention ``exclusion_half_angle`` follows.

    Returns
    -------
    ExclusionCone

    Notes
    -----
    Assumptions: the body is a sphere, the spacecraft is a point, and the
    instrument aperture is a point on that point. Baffle geometry, stray-light
    scattering off structure, and any dependence of the keep-out angle on
    illumination are not modelled -- see README limitations.
    """
    if body_angular_radius < 0.0:
        raise ValueError("body_angular_radius must be >= 0 rad")
    if exclusion_half_angle < 0.0:
        raise ValueError("exclusion_half_angle must be >= 0 rad")
    if reference == "limb":
        total = body_angular_radius + exclusion_half_angle
    elif reference == "center":
        total = exclusion_half_angle
    else:
        raise ValueError(f"reference must be 'limb' or 'center', got {reference!r}")
    return ExclusionCone(axis=body_direction, half_angle=min(total, np.pi), name=name)


@dataclass(frozen=True)
class KeepOutSet:
    """A set of exclusion cones tested together.

    Parameters
    ----------
    cones : sequence of ExclusionCone
        The cones. Names need not be unique but usually are.
    """

    cones: tuple[ExclusionCone, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        cones = tuple(self.cones)
        for c in cones:
            if not isinstance(c, ExclusionCone):
                raise TypeError(f"every element must be an ExclusionCone, got {type(c).__name__}")
        object.__setattr__(self, "cones", cones)

    def __len__(self) -> int:
        return len(self.cones)

    def __iter__(self):
        return iter(self.cones)

    def __getitem__(self, i: int) -> ExclusionCone:
        return self.cones[i]

    @property
    def names(self) -> tuple[str, ...]:
        """Cone names, in order."""
        return tuple(c.name for c in self.cones)

    def with_cone(self, cone: ExclusionCone) -> KeepOutSet:
        """Return a new set with ``cone`` appended."""
        return KeepOutSet(self.cones + (cone,))

    def margins(self, boresight: ArrayLike) -> NDArray[np.float64]:
        """Per-cone clearance angles [rad], shape ``(..., n_cones)``.

        Empty set returns shape ``(..., 0)``.
        """
        b = unit(boresight)
        if not self.cones:
            return np.empty(b.shape[:-1] + (0,), dtype=float)
        return np.stack([np.asarray(c.margin(b), dtype=float) for c in self.cones], axis=-1)

    def margin(self, boresight: ArrayLike) -> NDArray[np.float64] | float:
        """Worst-case clearance [rad]: the minimum over all cones.

        Positive means every cone is cleared, and the value is the distance to
        the nearest cone boundary. An empty set returns ``+inf``.
        """
        m = self.margins(boresight)
        if m.shape[-1] == 0:
            return np.full(m.shape[:-1], np.inf) if m.ndim > 1 else float("inf")
        out = np.min(m, axis=-1)
        return float(out) if np.ndim(out) == 0 else out

    def is_allowed(self, boresight: ArrayLike) -> NDArray[np.bool_] | bool:
        """``True`` where the boresight violates no cone."""
        m = self.margins(boresight)
        if m.shape[-1] == 0:
            return np.ones(m.shape[:-1], dtype=bool) if m.ndim > 1 else True
        out = np.all(m >= 0.0, axis=-1)
        return bool(out) if np.ndim(out) == 0 else out

    def violations(self, boresight: ArrayLike) -> tuple[str, ...]:
        """Names of the cones a single boresight violates, worst first.

        Parameters
        ----------
        boresight : array_like
            A single direction, shape ``(3,)``.

        Returns
        -------
        tuple of str
            Empty if the direction is allowed. Ordered by increasing margin, so
            the deepest violation comes first.
        """
        b = unit(boresight)
        if b.shape != (3,):
            raise ValueError(f"violations() takes a single direction (3,), got {b.shape}")
        m = self.margins(b)
        idx = [i for i in np.argsort(m) if m[i] < 0.0]
        return tuple(self.cones[i].name for i in idx)

    def rotated(self, rotation: ArrayLike) -> KeepOutSet:
        """Return the whole set with every cone axis rotated by ``(3, 3)``."""
        return KeepOutSet(tuple(c.rotated(rotation) for c in self.cones))
