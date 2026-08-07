"""Path-weighted turbulence integrals and the quadrature behind them.

Every metric in this package is a power of a weighted integral of the form

.. math::

    \\mu[W] = \\int_{h_0}^{H} C_n^2(h)\\,W(h)\\,dh

taken along the *vertical*; the zenith-angle dependence is applied afterwards,
explicitly and per quantity, in :mod:`atmoprofile.metrics`.  Keeping the two
separate is deliberate: it is the only way to state the sec(zeta) power of each
result rather than folding an airmass factor into an opaque constant.

Two integration methods are offered:

``"quad"`` (default)
    Adaptive Gauss-Kronrod (QUADPACK via :func:`scipy.integrate.quad`), given
    the profile's declared breakpoints so that piecewise models such as SLC are
    integrated exactly across their discontinuities.  The integrand is rescaled
    to O(1) before integration because Cn^2 integrals are ~1e-12 in SI units,
    far below QUADPACK's default absolute tolerance.

``"simpson"``
    Composite Simpson on a uniform grid of ``n_nodes`` points, ignoring
    breakpoints.  Provided for grid-refinement convergence studies and for
    users who want a fixed, transparent rule.  On a discontinuous profile it
    converges at a reduced rate; that is measured in ``validation/``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad, simpson

from ._validate import check_altitude_range, check_choice
from .profiles import Cn2Profile
from .wind import WindProfile

__all__ = [
    "INTEGRATION_METHODS",
    "weighted_integral",
    "turbulence_moment",
    "wind_weighted_moment",
    "effective_turbulence_height",
    "ConvergenceRecord",
    "grid_convergence",
]

INTEGRATION_METHODS: tuple[str, ...] = ("quad", "simpson")

_QUAD_EPSREL = 1e-10
_QUAD_EPSABS = 1e-12  # applied to the *rescaled* O(1) integrand


def _resolve_limits(
    profile: Cn2Profile, h_ground: float, h_top: float | None
) -> tuple[float, float]:
    top = profile.h_max if h_top is None else h_top
    return check_altitude_range(
        h_ground,
        top,
        profile_h_min=profile.h_min,
        profile_h_max=profile.h_max,
        profile_name=profile.name,
    )


def _integrate(
    integrand: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    method: str,
    n_nodes: int,
    breakpoints: tuple[float, ...],
) -> float:
    """Integrate a scalar callable over [lo, hi] with the requested rule."""
    if method == "quad":
        # Rescale so QUADPACK's absolute tolerance is meaningful: Cn^2 integrals
        # are O(1e-12) in SI units, while the default epsabs is 1.49e-8.
        probe = np.linspace(lo, hi, 257)
        vals = np.abs([integrand(float(h)) for h in probe])
        scale = float(np.max(vals)) * (hi - lo)
        if not np.isfinite(scale) or scale <= 0.0:
            return 0.0
        # Integrate panel by panel between the declared breakpoints.  Each panel
        # is smooth, so the adaptive rule converges without fighting the jumps
        # (which is what causes QUADPACK round-off warnings on piecewise or
        # interpolated profiles).
        edges = [lo, *sorted(b for b in breakpoints if lo < b < hi), hi]
        total = 0.0
        for left, right in zip(edges[:-1], edges[1:], strict=True):
            part, _ = quad(
                lambda h: integrand(h) / scale,
                left,
                right,
                limit=400,
                epsabs=_QUAD_EPSABS,
                epsrel=_QUAD_EPSREL,
            )
            total += part
        return float(total * scale)

    if n_nodes < 3:
        raise ValueError(f"n_nodes must be >= 3 for Simpson's rule, got {n_nodes}")
    n = int(n_nodes) | 1  # Simpson needs an odd number of nodes
    grid = np.linspace(lo, hi, n)
    y = np.array([integrand(float(h)) for h in grid])
    return float(simpson(y, x=grid))


def weighted_integral(
    profile: Cn2Profile,
    weight: Callable[[float], float] | None = None,
    *,
    h_ground: float = 0.0,
    h_top: float | None = None,
    method: str = "quad",
    n_nodes: int = 2001,
) -> float:
    """Evaluate ``integral Cn^2(h) W(h) dh`` along the vertical.

    Parameters
    ----------
    profile:
        Cn^2 profile, m^(-2/3), altitude in metres.
    weight:
        Dimensionless (or otherwise) weight W(h); ``None`` means W = 1.
    h_ground, h_top:
        Integration limits in metres.  ``h_top=None`` uses the profile's upper
        support limit.
    method:
        ``"quad"`` or ``"simpson"`` (see the module docstring).
    n_nodes:
        Number of Simpson nodes; ignored by ``"quad"``.

    Returns
    -------
    float
        The integral, in m^(-2/3) times metres times the units of W.
    """
    if not isinstance(profile, Cn2Profile):
        raise TypeError(f"profile must be a Cn2Profile, got {type(profile).__name__}")
    meth = check_choice("method", method, INTEGRATION_METHODS)
    lo, hi = _resolve_limits(profile, h_ground, h_top)

    if weight is None:

        def integrand(h: float) -> float:
            return float(profile(h))

    else:

        def integrand(h: float) -> float:
            return float(profile(h)) * float(weight(h))

    return _integrate(
        integrand, lo, hi, method=meth, n_nodes=n_nodes, breakpoints=profile.breakpoints
    )


def turbulence_moment(
    profile: Cn2Profile,
    power: float = 0.0,
    *,
    h_ground: float = 0.0,
    h_top: float | None = None,
    method: str = "quad",
    n_nodes: int = 2001,
) -> float:
    r"""Turbulence moment :math:`\mu_m = \int C_n^2(h)\,(h-h_0)^m\,dh`.

    The three moments used by this package are

    * ``m = 0``    -> Fried parameter (Fried 1966),
    * ``m = 5/3``  -> isoplanatic angle (Fried 1982),
    * ``m = 5/6``  -> Rytov variance (Andrews & Phillips 2005).

    Parameters
    ----------
    power:
        Moment order m (dimensionless).
    h_ground:
        Altitude of the observer, m; the moment arm is measured from here,
        because it is the *distance from the receiver* along the path that
        weights anisoplanatism and scintillation.

    Returns
    -------
    float
        :math:`\mu_m` in units of m^(m - 2/3 + 1) = m^(m + 1/3).
    """
    m = float(power)
    if not np.isfinite(m):
        raise ValueError(f"power must be finite, got {power!r}")
    if m == 0.0:
        return weighted_integral(
            profile, None, h_ground=h_ground, h_top=h_top, method=method, n_nodes=n_nodes
        )
    h0 = float(h_ground)
    return weighted_integral(
        profile,
        lambda h: max(h - h0, 0.0) ** m,
        h_ground=h_ground,
        h_top=h_top,
        method=method,
        n_nodes=n_nodes,
    )


def wind_weighted_moment(
    profile: Cn2Profile,
    wind: WindProfile,
    *,
    power: float = 5.0 / 3.0,
    h_ground: float = 0.0,
    h_top: float | None = None,
    method: str = "quad",
    n_nodes: int = 2001,
) -> float:
    r"""Wind-weighted moment :math:`\int C_n^2(h)\,v(h)^{p}\,dh`, default p = 5/3.

    This is the integral in Greenwood's bandwidth formula.  Units:
    m^(-2/3) * (m/s)^p * m = m^(1/3) (m/s)^p; with p = 5/3 the bracket in
    :func:`atmoprofile.metrics.greenwood_frequency` becomes s^(-5/3).
    """
    if not isinstance(wind, WindProfile):
        raise TypeError(f"wind must be a WindProfile, got {type(wind).__name__}")
    p = float(power)
    lo, hi = _resolve_limits(profile, h_ground, profile.h_max if h_top is None else h_top)
    if lo < wind.h_min - 1e-6 or hi > wind.h_max + 1e-6:
        raise ValueError(
            f"integration range [{lo:g}, {hi:g}] m exceeds the support "
            f"[{wind.h_min:g}, {wind.h_max:g}] m of wind profile {wind.name!r}"
        )
    return weighted_integral(
        profile,
        lambda h: float(wind(h)) ** p,
        h_ground=h_ground,
        h_top=h_top,
        method=method,
        n_nodes=n_nodes,
    )


def effective_turbulence_height(
    profile: Cn2Profile,
    *,
    h_ground: float = 0.0,
    h_top: float | None = None,
    method: str = "quad",
    n_nodes: int = 2001,
) -> float:
    r"""Effective turbulence height :math:`\bar h = [\mu_{5/3}/\mu_0]^{3/5}`, in metres.

    This is the altitude at which a single equivalent layer would produce the
    same anisoplanatism as the real profile; it is the quantity in the familiar
    relation theta0 = 0.314 r0 / h_bar (Roddier 1981; Hardy 1998), which this
    package reproduces to numerical precision (see the tests).

    Units: metres, measured from ``h_ground`` along the vertical (not along the
    slant path - the slant path enters the metrics as an explicit sec(zeta)).
    """
    mu0 = turbulence_moment(
        profile, 0.0, h_ground=h_ground, h_top=h_top, method=method, n_nodes=n_nodes
    )
    mu53 = turbulence_moment(
        profile, 5.0 / 3.0, h_ground=h_ground, h_top=h_top, method=method, n_nodes=n_nodes
    )
    if mu0 <= 0.0:
        raise ValueError("mu_0 is zero or negative; cannot form an effective height")
    return float((mu53 / mu0) ** 0.6)


@dataclass(frozen=True)
class ConvergenceRecord:
    """One row of a grid-refinement study.

    Attributes
    ----------
    n_nodes:
        Number of Simpson nodes used.
    value:
        Quantity evaluated on that grid (units of whatever was evaluated).
    rel_change:
        Relative change from the previous (coarser) grid; ``nan`` for the first
        row.
    rel_error_vs_reference:
        Relative difference from the reference value supplied by the caller
        (normally the adaptive-quadrature result).
    """

    n_nodes: int
    value: float
    rel_change: float
    rel_error_vs_reference: float


def grid_convergence(
    evaluate: Callable[[int], float],
    n_nodes_list: tuple[int, ...] | list[int],
    reference: float,
) -> list[ConvergenceRecord]:
    """Run a grid-refinement study of ``evaluate(n_nodes)``.

    Parameters
    ----------
    evaluate:
        Callable taking a Simpson node count and returning the quantity of
        interest.
    n_nodes_list:
        Increasing node counts.
    reference:
        Reference value (e.g. the adaptive-quadrature result) used for the
        relative-error column.

    Returns
    -------
    list of ConvergenceRecord
    """
    if reference == 0.0:
        raise ValueError("reference must be non-zero to form relative errors")
    out: list[ConvergenceRecord] = []
    previous: float | None = None
    for n in n_nodes_list:
        val = float(evaluate(int(n)))
        rel_change = float("nan") if previous is None else abs(val - previous) / abs(previous)
        out.append(
            ConvergenceRecord(
                n_nodes=int(n),
                value=val,
                rel_change=rel_change,
                rel_error_vs_reference=abs(val - reference) / abs(reference),
            )
        )
        previous = val
    return out
