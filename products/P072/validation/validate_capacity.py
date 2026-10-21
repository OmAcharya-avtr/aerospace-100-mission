"""Contact-time accounting against an independent integral of rate over window.

The capacity of a window is equation (1) of :mod:`downlinkpick.contacts`,
``C = integral of R(t) dt`` in bits. The library evaluates it in closed form as
the trapezoidal sum over the piecewise-linear breakpoints, which is exact for a
linear integrand on each segment in exact arithmetic.

Checks:

1. the closed-form capacity against ``scipy.integrate.quad``, adaptive
   Gauss-Kronrod quadrature of the same callable, which shares no code with the
   trapezoidal sum;
2. the closed-form capacity against a fine midpoint Riemann sum, a second
   independent route with a different error mechanism;
3. additivity: the capacity of a window equals the sum of the capacities of its
   two halves, which is a property of the integral and not of the profile;
4. the flat-profile closed form ``rate * duration``, which is the one case with
   an elementary answer;
5. the inverse, :meth:`ContactWindow.time_for_bits`, round-tripped through
   :meth:`ContactWindow.delivered_bits`;
6. the extension used by :mod:`downlinkpick.marginal` adds exactly the integral
   of the held final rate, ``final_rate * extra_s``.

Every comparison is relative and two-sided. None of them asserts bit-exact
equality: the trapezoidal sum, the quadrature and the Riemann sum accumulate
different rounding over different numbers of terms, so what is guaranteed is
agreement to within rounding.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

from downlinkpick.cases import load_case  # noqa: E402
from downlinkpick.contacts import (  # noqa: E402
    ContactWindow,
    constant_rate_profile,
    rate_profile_from_elevation,
)

#: Relative tolerance for the quadrature comparison. ``scipy.integrate.quad``
#: subdivides adaptively and does not place nodes on the profile kinks, so its
#: error is set by its own tolerance and not by double-precision rounding.
QUAD_TOLERANCE_REL = 1.0e-7

#: Floor on the relative tolerance for the midpoint Riemann comparison. The
#: midpoint rule is exact on each linear segment, so for a continuous profile
#: the only error is rounding over 4e5 terms. A profile containing a *rate step*
#: is different: one sub-interval straddles the discontinuity and the midpoint
#: value there is wrong by up to the step height, so the error of the reference
#: is bounded by ``sum(step heights) * h`` with ``h`` the sub-interval width.
#: That bound is computed per profile by :func:`riemann_tolerance` and the floor
#: below applies where there is no step. The disagreement at a step is a
#: property of the reference, not of the library, which is why it is bounded
#: rather than tuned away.
RIEMANN_TOLERANCE_REL = 1.0e-9

#: Safety factor on the derived step bound, dimensionless.
RIEMANN_BOUND_FACTOR = 4.0

#: Relative tolerance for the identities that hold in exact arithmetic:
#: additivity, the flat-profile closed form, and the extension increment.
IDENTITY_TOLERANCE_REL = 1.0e-12

RIEMANN_POINTS = 400_001

PROFILES = (
    ("flat 50 Mbit/s, 300 s", 0.0, 300.0, constant_rate_profile(0.0, 300.0, 5.0e7)),
    (
        "sine floor 0.22, 9 points, 480 s",
        0.0,
        480.0,
        rate_profile_from_elevation(0.0, 480.0, 1.2e8, floor_fraction=0.22, points=9),
    ),
    (
        "sine floor 0.0, 25 points, 612 s",
        120.0,
        732.0,
        rate_profile_from_elevation(120.0, 732.0, 8.0e7, floor_fraction=0.0, points=25),
    ),
    (
        "irregular breakpoints with a step",
        0.0,
        431.0,
        (
            (0.0, 1.0e7),
            (37.0, 2.3e8),
            (111.5, 1.1e8),
            (111.5, 2.6e8),
            (260.0, 3.05e8),
            (400.0, 7.0e7),
            (431.0, 1.0e7),
        ),
    ),
    (
        "zero-rate opening segment",
        10.0,
        250.0,
        ((10.0, 0.0), (60.0, 0.0), (155.0, 9.0e7), (250.0, 2.0e7)),
    ),
)


def riemann_tolerance(
    breakpoints: tuple[tuple[float, float], ...], start: float, end: float, capacity: float
) -> tuple[float, float]:
    """Derived relative tolerance for the midpoint reference, and the step bound.

    Returns ``(tolerance, absolute step bound in bits)``. The bound is
    ``RIEMANN_BOUND_FACTOR * sum(step heights) * h``, where a step is a pair of
    breakpoints sharing a time and ``h`` is the sub-interval width.
    """
    width = (end - start) / (RIEMANN_POINTS - 1)
    steps = sum(
        abs(later[1] - earlier[1])
        for earlier, later in zip(breakpoints, breakpoints[1:], strict=False)
        if later[0] == earlier[0]
    )
    bound = RIEMANN_BOUND_FACTOR * steps * width
    return max(RIEMANN_TOLERANCE_REL, bound / capacity), bound


def midpoint_riemann(window: ContactWindow, points: int) -> float:
    """Midpoint Riemann sum of the rate over the window, in bits."""
    edges = np.linspace(window.start_s, window.end_s, points)
    centres = 0.5 * (edges[:-1] + edges[1:])
    rates = np.array([window.rate_bps(float(t)) for t in centres])
    return float(np.sum(rates * np.diff(edges)))


def main() -> int:
    out = Tee("validate_capacity_output.txt")
    out.rule("downlinkpick validation: contact-time accounting")
    for line in environment_lines():
        out(line)
    failures = 0

    out.rule("1 and 2. closed form against two independent integrals of rate over window")
    out("reference 1: scipy.integrate.quad, adaptive Gauss-Kronrod, shares no code")
    out(f"reference 2: midpoint Riemann sum over {RIEMANN_POINTS - 1} sub-intervals")
    out(f"tolerances : quad {QUAD_TOLERANCE_REL:.1e} relative; Riemann "
        f"max({RIEMANN_TOLERANCE_REL:.1e}, {RIEMANN_BOUND_FACTOR:.0f} * sum(step heights) * h "
        "/ capacity), derived per profile")
    out("")
    out(
        f"{'profile':>38s} {'closed form [bit]':>19s} {'quad rel':>11s} "
        f"{'Riemann rel':>12s} {'Riemann tol':>12s}"
    )
    worst_quad = 0.0
    worst_riemann_margin = 0.0
    for label, start, end, breakpoints in PROFILES:
        window = ContactWindow("probe", start, end, breakpoints)
        closed = window.capacity_bits()
        integrated, _error = quad(window.rate_bps, start, end, limit=500)
        riemann = midpoint_riemann(window, RIEMANN_POINTS)
        quad_rel = abs(integrated - closed) / closed
        riemann_rel = abs(riemann - closed) / closed
        tolerance, _bound = riemann_tolerance(tuple(breakpoints), start, end, closed)
        worst_quad = max(worst_quad, quad_rel)
        worst_riemann_margin = max(worst_riemann_margin, riemann_rel / tolerance)
        out(
            f"{label:>38s} {closed:19.10e} {quad_rel:11.3e} {riemann_rel:12.3e} "
            f"{tolerance:12.3e}"
        )
        if quad_rel > QUAD_TOLERANCE_REL:
            failures += 1
            out(f"  FAIL quad disagreement {quad_rel:.3e} above {QUAD_TOLERANCE_REL:.1e}")
        if riemann_rel > tolerance:
            failures += 1
            out(f"  FAIL Riemann disagreement {riemann_rel:.3e} above {tolerance:.3e}")
    out("")
    out(f"worst quad relative difference         : {worst_quad:.6e}")
    out(f"worst Riemann difference / tolerance  : {worst_riemann_margin:.6e}")

    out.rule("3. additivity of the capacity integral over a split window")
    out("identity: delivered_bits(a, c) == delivered_bits(a, b) + delivered_bits(b, c)")
    out(f"tolerance: {IDENTITY_TOLERANCE_REL:.1e} relative")
    out("")
    out(f"{'profile':>38s} {'split at [s]':>12s} {'relative residual':>18s}")
    worst_additivity = 0.0
    for label, start, end, breakpoints in PROFILES:
        window = ContactWindow("probe", start, end, breakpoints)
        for fraction in (0.17, 0.5, 0.83):
            middle = start + fraction * (end - start)
            whole = window.delivered_bits(start, end)
            parts = window.delivered_bits(start, middle) + window.delivered_bits(middle, end)
            residual = abs(whole - parts) / whole
            worst_additivity = max(worst_additivity, residual)
            if fraction == 0.5:
                out(f"{label:>38s} {middle:12.3f} {residual:18.3e}")
            if residual > IDENTITY_TOLERANCE_REL:
                failures += 1
                out(f"  FAIL additivity residual {residual:.3e} at split {middle:.3f} s")
    out("")
    out(f"worst additivity residual over 15 splits: {worst_additivity:.6e}")

    out.rule("4. flat profile against the elementary closed form rate * duration")
    out(f"tolerance: {IDENTITY_TOLERANCE_REL:.1e} relative")
    out("")
    out(f"{'rate [bit/s]':>14s} {'duration [s]':>13s} {'rate*duration':>18s} {'rel':>11s}")
    worst_flat = 0.0
    for rate in (1.0e6, 5.0e7, 1.2e9):
        for duration in (37.0, 300.0, 912.5):
            window = ContactWindow(
                "probe", 0.0, duration, constant_rate_profile(0.0, duration, rate)
            )
            expected = rate * duration
            residual = abs(window.capacity_bits() - expected) / expected
            worst_flat = max(worst_flat, residual)
            out(f"{rate:14.3e} {duration:13.3f} {expected:18.10e} {residual:11.3e}")
            if residual > IDENTITY_TOLERANCE_REL:
                failures += 1
                out(f"  FAIL flat-profile residual {residual:.3e}")
    out("")
    out(f"worst flat-profile residual: {worst_flat:.6e}")

    out.rule("5. time_for_bits inverted against delivered_bits")
    out("identity: delivered_bits(start, start + time_for_bits(b)) == b")
    out("tolerance: 1.0e-9 relative on the bit count")
    out("")
    out(f"{'profile':>38s} {'fraction':>9s} {'seconds':>11s} {'rel':>11s}")
    worst_inverse = 0.0
    for label, start, end, breakpoints in PROFILES:
        window = ContactWindow("probe", start, end, breakpoints)
        capacity = window.capacity_bits()
        for fraction in (0.05, 0.25, 0.5, 0.9, 1.0):
            target = fraction * capacity
            seconds = window.time_for_bits(target)
            delivered = window.delivered_bits(start, start + seconds)
            residual = abs(delivered - target) / max(target, 1.0)
            worst_inverse = max(worst_inverse, residual)
            if fraction == 0.5:
                out(f"{label:>38s} {fraction:9.2f} {seconds:11.5f} {residual:11.3e}")
            if residual > 1.0e-9:
                failures += 1
                out(f"  FAIL inverse residual {residual:.3e} at fraction {fraction}")
    out("")
    out(f"worst inverse residual over 25 points: {worst_inverse:.6e}")

    out.rule("6. window extension adds exactly the held final rate times the step")
    out("identity: extended(d).capacity - capacity == final_rate * d")
    out(f"tolerance: {IDENTITY_TOLERANCE_REL:.1e} relative")
    out("")
    out(f"{'profile':>38s} {'step [s]':>9s} {'expected [bit]':>17s} {'rel':>11s}")
    worst_extension = 0.0
    for label, start, end, breakpoints in PROFILES:
        window = ContactWindow("probe", start, end, breakpoints)
        for step in (1.0, 10.0, 120.0):
            expected = window.final_rate_bps * step
            if expected == 0.0:
                continue
            added = window.extended(step).capacity_bits() - window.capacity_bits()
            residual = abs(added - expected) / expected
            worst_extension = max(worst_extension, residual)
            if step == 10.0:
                out(f"{label:>38s} {step:9.1f} {expected:17.8e} {residual:11.3e}")
            if residual > IDENTITY_TOLERANCE_REL:
                failures += 1
                out(f"  FAIL extension residual {residual:.3e} at step {step} s")
    out("")
    out(f"worst extension residual: {worst_extension:.6e}")

    out.rule("7. the shipped cases, capacity against quadrature")
    out(f"{'case':>36s} {'window':>8s} {'closed form [bit]':>19s} {'quad rel':>11s}")
    worst_case_rel = 0.0
    for name in (
        "examples/cases/three_pass_declared.yaml",
        "examples/cases/missing_declarations.yaml",
        "examples/cases/single_short_pass.yaml",
    ):
        case = load_case(name)
        for window in case.plan:
            closed = window.capacity_bits()
            integrated, _error = quad(window.rate_bps, window.start_s, window.end_s, limit=500)
            relative = abs(integrated - closed) / closed
            worst_case_rel = max(worst_case_rel, relative)
            out(f"{name.split('/')[-1]:>36s} {window.name:>8s} {closed:19.10e} {relative:11.3e}")
            if relative > QUAD_TOLERANCE_REL:
                failures += 1
                out(f"  FAIL case capacity disagreement {relative:.3e}")
    out("")
    out(f"worst shipped-case relative difference: {worst_case_rel:.6e}")

    out.rule("summary")
    out(f"failures: {failures}")
    out(f"output  : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
