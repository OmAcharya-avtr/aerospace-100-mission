"""Relativistic aberration of a light direction, by four independent routes.

Aberration is the change in the apparent direction of propagation of a photon
between two inertial frames in relative motion. For an optical crosslink it
enters twice and at first order in ``beta = v/c``: the direction a terminal
*measures* the incoming beam to come from is aberrated by its own velocity, and
the direction it must *point* to deliver a beam along a required inertial
direction is aberrated by the same velocity in the opposite sense. Both are
about 25 microradians for a terminal in low Earth orbit, and they add into the
point-ahead angle. They are not corrections to a smaller effect.

Equation (exact, special relativity). Let ``p`` be the unit propagation
direction of a photon in frame S, and let S' move with velocity ``v = c beta``
with respect to S. Then in S' the propagation direction is, in the
angle-relation form,

    cos(theta') = (cos(theta) - beta) / (1 - beta cos(theta))
    sin(theta') = sin(theta) / (gamma (1 - beta cos(theta)))

with ``theta`` measured from ``beta_hat`` and the azimuth about ``beta_hat``
unchanged. Source: Rindler, *Relativity: Special, General and Cosmological*,
2nd ed., Oxford University Press 2006, section 3.6 (aberration). The equivalent
first-order result ``p' = p - beta_perp`` is the classical stellar aberration of
Bradley 1729 and is the form used in the closed-form baseline.

Validity: the formulae are exact for ``0 <= |beta| < 1`` and any inclination;
they assume flat spacetime and no refractive medium. Gravitational light
bending is not modelled and is noted as a limitation in the README.

Four routes are implemented, and the validation suite cross-checks them against
each other because they are the only aberration reference this build has:

1. :func:`aberrate_first_order` -- the classical first-order vector form.
2. :func:`aberrate_angle_relation` -- the exact angle relation above.
3. :func:`aberrate_momentum_boost` -- Lorentz boost of the photon
   four-momentum ``k = (1, p)``.
4. :func:`aberrate_event_worldline` -- Lorentz transformation of the spacetime
   *coordinates* of two events on the photon worldline, taking the direction of
   the transformed spatial separation.

Routes 2, 3 and 4 are exact and are separate derivations: one works from the
trigonometric relation, one from a four-vector in momentum space, one from the
coordinate transformation of events with its own time dilation bookkeeping.
Agreement between them is an implementation check, not a physics check; the
physics is checked by the geometric invariants in
:mod:`pointahead.angle` and by the iterated light-time reference.

**Deviation recorded here as well as in validation/VALIDATION.md:** the batch
specification for this product asks for the aberration terms to be cross-checked
against ``astropy``. ``astropy`` is not installed in the environment this build
was measured in and installing it was not permitted, so no number anywhere in
this repository comes from ``astropy``. These four routes plus the iterated
light-time reference replace that cross-check.
"""

from __future__ import annotations

import numpy as np

from .geometry import C_LIGHT, angle_between, as_vector, unit


def _check_beta(beta: object) -> tuple[np.ndarray, float]:
    """Validate a velocity-over-c vector, returning ``(beta, |beta|)``."""
    b = as_vector(beta, "beta")
    magnitude = float(np.linalg.norm(b))
    if magnitude >= 1.0:
        raise ValueError(
            f"|beta| must be < 1 for a massive frame, got {magnitude!r}; "
            "divide a velocity in m/s by pointahead.geometry.C_LIGHT"
        )
    return b, magnitude


def beta_from_velocity(velocity_mps: object) -> np.ndarray:
    """Convert a velocity in m/s to the dimensionless ``beta = v / c``."""
    return as_vector(velocity_mps, "velocity_mps") / C_LIGHT


def lorentz_gamma(beta_magnitude: float) -> float:
    """Lorentz factor ``1 / sqrt(1 - beta^2)`` for a scalar ``|beta|``."""
    magnitude = float(beta_magnitude)
    if not 0.0 <= magnitude < 1.0:
        raise ValueError(f"beta_magnitude must be in [0, 1), got {beta_magnitude!r}")
    return 1.0 / float(np.sqrt(1.0 - magnitude * magnitude))


def aberrate_first_order(direction: object, beta: object) -> np.ndarray:
    """First-order aberration of a propagation direction. Route 1.

    ``p' = normalise(p - (beta - (beta . p) p))``, i.e. the direction is
    displaced by the component of ``beta`` transverse to it, against the boost.
    Error against the exact routes is ``O(beta^2)`` in general, and
    ``O(beta^3)`` for a direction exactly transverse to the boost, where the
    second-order term vanishes and the residual is
    ``arcsin(beta) - arctan(beta) = beta^3 / 2``. Both scalings are measured,
    not asserted: see ``validation/validate_aberration.py``.
    """
    p = unit(direction, "direction")
    b, _ = _check_beta(beta)
    transverse_beta = b - float(np.dot(b, p)) * p
    return unit(p - transverse_beta, "aberrated direction")


def aberrate_angle_relation(direction: object, beta: object) -> np.ndarray:
    """Exact aberration from the trigonometric relation. Route 2.

    Decomposes ``p`` into a component along ``beta_hat`` and a transverse
    component, maps the polar angle with
    ``cos(theta') = (cos(theta) - beta) / (1 - beta cos(theta))`` and
    ``sin(theta') = sin(theta) / (gamma (1 - beta cos(theta)))``, and
    reassembles. The azimuth about ``beta_hat`` is unchanged, which is why only
    two components are needed.
    """
    p = unit(direction, "direction")
    b, magnitude = _check_beta(beta)
    if magnitude == 0.0:
        return p
    b_hat = b / magnitude
    cos_theta = float(np.dot(p, b_hat))
    perpendicular = p - cos_theta * b_hat
    sin_theta = float(np.linalg.norm(perpendicular))
    denominator = 1.0 - magnitude * cos_theta
    cos_prime = (cos_theta - magnitude) / denominator
    if sin_theta == 0.0:
        return float(np.sign(cos_prime)) * b_hat
    gamma = lorentz_gamma(magnitude)
    sin_prime = sin_theta / (gamma * denominator)
    e_perp = perpendicular / sin_theta
    return unit(cos_prime * b_hat + sin_prime * e_perp, "aberrated direction")


def aberrate_momentum_boost(direction: object, beta: object) -> np.ndarray:
    """Exact aberration from a boost of the photon four-momentum. Route 3.

    Takes ``k = (k0, kvec) = (1, p)``, which is null because ``|p| = 1``, and
    applies the standard boost to a frame moving with velocity ``c beta``:

        k0'   = gamma (k0 - beta . kvec)
        kpar' = gamma (kpar - beta k0)
        kperp'= kperp

    then returns ``kvec' / k0'``. The result is automatically a unit vector
    because the boosted four-vector is still null; it is renormalised anyway so
    that a rounding error cannot leak a non-unit direction into a downstream
    angle.
    """
    p = unit(direction, "direction")
    b, magnitude = _check_beta(beta)
    if magnitude == 0.0:
        return p
    b_hat = b / magnitude
    gamma = lorentz_gamma(magnitude)
    k_parallel = float(np.dot(p, b_hat))
    k_perpendicular = p - k_parallel * b_hat
    k0_prime = gamma * (1.0 - magnitude * k_parallel)
    k_vector_prime = gamma * (k_parallel - magnitude) * b_hat + k_perpendicular
    return unit(k_vector_prime / k0_prime, "aberrated direction")


def aberrate_event_worldline(
    direction: object, beta: object, baseline_m: float = 1.0e6
) -> np.ndarray:
    """Exact aberration from a coordinate transformation of two events. Route 4.

    Places two events on the photon worldline in S -- the origin
    ``(t=0, x=0)`` and ``(t = L/c, x = L p)`` -- transforms both with the full
    Lorentz transformation of spacetime coordinates,

        t'    = gamma (t - beta . x / c)
        xpar' = gamma (xpar - beta c t)
        xperp'= xperp

    and returns the direction of the transformed spatial separation. This route
    never forms a four-momentum and carries its own time-dilation bookkeeping,
    so it exercises different arithmetic from route 3 while computing the same
    physical quantity.

    ``baseline_m`` is the separation between the two events and cancels exactly
    in the result; it is exposed so a test can confirm that it does.
    """
    p = unit(direction, "direction")
    b, magnitude = _check_beta(beta)
    length = float(baseline_m)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"baseline_m must be a positive finite length, got {baseline_m!r}")
    if magnitude == 0.0:
        return p
    b_hat = b / magnitude
    t_event = length / C_LIGHT
    x_event = length * p
    x_parallel = float(np.dot(x_event, b_hat))
    x_perpendicular = x_event - x_parallel * b_hat
    gamma = lorentz_gamma(magnitude)
    x_parallel_prime = gamma * (x_parallel - magnitude * C_LIGHT * t_event)
    separation = x_parallel_prime * b_hat + x_perpendicular
    return unit(separation, "aberrated direction")


ABERRATION_ROUTES = {
    "first_order": aberrate_first_order,
    "angle_relation": aberrate_angle_relation,
    "momentum_boost": aberrate_momentum_boost,
    "event_worldline": aberrate_event_worldline,
}
"""Named aberration routes. ``first_order`` is approximate; the other three are
exact and are expected to agree with each other to rounding."""

EXACT_ROUTES = ("angle_relation", "momentum_boost", "event_worldline")


def aberrate(direction: object, beta: object, route: str = "momentum_boost") -> np.ndarray:
    """Aberrate ``direction`` by ``beta`` using the named route.

    Parameters
    ----------
    direction
        Unit propagation direction of the photon in the source frame.
    beta
        Velocity of the target frame over c, as a 3-vector.
    route
        One of ``ABERRATION_ROUTES``. The default is the exact four-momentum
        boost.
    """
    try:
        function = ABERRATION_ROUTES[route]
    except KeyError:
        raise ValueError(
            f"unknown aberration route {route!r}; expected one of {sorted(ABERRATION_ROUTES)}"
        ) from None
    return function(direction, beta)


def aberration_angle(direction: object, beta: object, route: str = "momentum_boost") -> float:
    """Angular displacement in radians produced by aberrating ``direction``.

    For a direction transverse to the boost this is close to ``|beta|``; for a
    direction along the boost it is zero.
    """
    return angle_between(unit(direction, "direction"), aberrate(direction, beta, route))


def route_disagreement(direction: object, beta: object) -> dict[str, float]:
    """Pairwise angular disagreement in radians between all four routes.

    Returned keyed by ``"a_vs_b"``. Used by the validation scripts, which have
    no external aberration implementation to compare against.
    """
    results = {name: aberrate(direction, beta, name) for name in ABERRATION_ROUTES}
    names = list(ABERRATION_ROUTES)
    out: dict[str, float] = {}
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            out[f"{first}_vs_{second}"] = angle_between(results[first], results[second])
    return out
