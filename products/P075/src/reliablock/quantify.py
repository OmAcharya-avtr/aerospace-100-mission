"""Closed-form quantification of a declared diagram.

Everything here is a closed form. Monte Carlo (:mod:`reliablock.montecarlo`)
exists only to be scored *against* these numbers, never the other way round:
for a declared diagram with declared distributions the reliability has a correct
answer, and a simulation that disagrees with it is a defect in the simulation
or in the closed form, not a better estimate.

Every result carries both a reliability and an unreliability, and each is
computed in the space that keeps it. The :class:`Quantity` docstring says which
is which per structure, and ``validation/validate_precision_output.txt``
measures what the alternative costs.

Closed forms, with their derivations
------------------------------------

**Series.** ``H = sum H_i`` exactly, so ``R = exp(-H)`` and
``Q = -expm1(-H)``. The additivity of cumulative hazard over series
composition is immediate from ``R = prod R_i``.

**Parallel.** ``Q = prod Q_i`` exactly; ``R = 1 - Q``. For a redundant group
``Q`` is the small quantity, and it is the one that is exact here.

**k-of-n.** ``R`` is the Poisson-binomial survival ``P(at least k survive)``
and ``Q`` is ``P(at least n - k + 1 fail)``, each evaluated by its own exact
convolution recurrence over non-negative terms. Neither is derived from the
other, so neither inherits the other's cancellation. For identical blocks both
reduce to the binomial sum ``sum_{i=k}^{n} C(n,i) p^i q^(n-i)``, which is the
closed form the structure is validated against in
``validation/validate_structures.py``.

**Standby with switching reliability.** For ``m`` identical cold standby units
behind one identical primary, each with constant rate ``lambda``, and a
changeover that succeeds with probability ``p_s`` independently at each call:

    ``R(t) = exp(-lambda t) * sum_{j=0}^{m} (p_s lambda t)^j / j!``

Derivation. With perfect switching the system life is the sum of ``m + 1``
independent exponentials, so it is Erlang-distributed with shape ``m + 1``, and
``R(t) = sum_{j=0}^{m} exp(-lambda t) (lambda t)^j / j!`` is the standard Erlang
survival. The ``j``-th term is the event that exactly ``j`` changeovers have
been required, and each of those ``j`` changeovers must have succeeded, so the
term carries ``p_s^j``. Setting ``m = 1`` gives the textbook
``R = exp(-lambda t) (1 + p_s lambda t)``.

For one standby unit with arbitrary distributions and a warm-standby hazard
fraction ``f``:

    ``R(t) = R_p(t) + p_s * integral_0^t f_p(u) exp(-H_s(f u + t - u)) du``

Derivation. The system survives either because the primary survives to ``t``,
or because the primary fails at some ``u < t``, the changeover succeeds, and
the standby survives from ``u`` to ``t``. The standby has waited for ``u``
accruing hazard at a fraction ``f`` of its operating rate, so its effective age
at ``t`` is ``f u + (t - u)``. ``f = 0`` is cold standby and recovers the
ordinary convolution ``integral f_p(u) R_s(t - u) du``; ``f = 1`` is hot
standby, for which the effective age is ``t`` and the expression collapses to
the parallel-with-switch result. Both limits are asserted as property tests.
The integral is evaluated with ``scipy.integrate.quad``; its reported absolute
error bound is carried in :attr:`Quantity.notes`.

**Shared load, two units.** Each unit's declared rate is its full-load rate
``lambda_i``; while both survive each carries ``s lambda_i`` for a declared
sharing factor ``s`` in ``(0, 1]``. With ``L = s (lambda_1 + lambda_2)``:

    ``R(t) = exp(-L t)
             + s lambda_1 (exp(-lambda_2 t) - exp(-L t)) / (L - lambda_2)
             + s lambda_2 (exp(-lambda_1 t) - exp(-L t)) / (L - lambda_1)``

Derivation. The four-state Markov chain has states (both alive), (only 1
alive), (only 2 alive), (down). The sojourn in (both alive) is exponential with
rate ``L``. Leaving it, unit 1 is the one that failed with probability
``s lambda_1 / L``, after which unit 2 carries the whole load at
``lambda_2``. Integrating the survival of the second unit over the first
failure time ``u``,
``integral_0^t s lambda_1 exp(-L u) exp(-lambda_2 (t - u)) du``, gives the
second term; the third is the mirror image. Where a denominator vanishes the
limit ``s lambda_i t exp(-L t)`` is used, and the switch between the two is at a
declared relative distance so that the expression is continuous rather than
merely defined. ``s = 1`` makes the node a parallel pair, which is asserted as
a property test.

**Common cause.** Not a structure: see :mod:`reliablock.commoncause`. A
declared beta factor is applied by rewriting the diagram, so the comparison
with and without it runs the same quantifier over two diagrams rather than two
code paths over one diagram.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from scipy.integrate import quad

from .diagram import (
    Block,
    Diagram,
    KOfN,
    Network,
    Node,
    Parallel,
    Series,
    SharedLoad,
    Standby,
)
from .distributions import Exponential, FailureDistribution
from .mission import MissionProfile
from .numerics import (
    hazard_from_unreliability,
    parallel_unreliability,
    reliability_from_hazard,
    series_hazard,
    unreliability_from_hazard,
    weighted_binomial_survival,
)
from .structure import switch_event_name

__all__ = [
    "Quantity",
    "quantify",
    "shared_load_reliability",
    "standby_reliability_cold_identical",
    "standby_reliability_general",
]

def _expm1_over_x(x: float) -> float:
    """``expm1(x) / x``, with its removable singularity at ``x = 0`` filled in.

    This is the whole of the shared-load degeneracy problem. The closed form
    contains ``(exp(-a t) - exp(-L t)) / (L - a)``, which is ``0 / 0`` when the
    two rates coincide and catastrophically cancelling just before that. The
    algebraically identical form

        ``t exp(-L t) * expm1((L - a) t) / ((L - a) t)``

    is exact for every argument including zero, so there is no threshold to
    choose, no analytic limit to switch to, and no discontinuity to measure.
    Replacing a switched-threshold implementation with this one is why
    :func:`shared_load_reliability` carries no tuning constant: a threshold
    picked to make a continuity test pass is a tuned tolerance, which this
    repository does not do.
    """
    if x == 0.0:
        return 1.0
    return math.expm1(x) / x


@dataclass
class Quantity:
    """A reliability figure with the unreliability that was not derived from it.

    ``reliability`` and ``unreliability`` are each computed in the space that
    keeps them, so they sum to 1 only to within rounding. Where they disagree
    by more than rounding the structure is ill-conditioned in one of the two
    spaces, and :attr:`complement_residual` reports it rather than hiding it.
    """

    reliability: float
    unreliability: float
    #: Cumulative hazard ``-log R``, dimensionless. Exact for a series chain.
    hazard: float
    #: Human-readable structure label, for the per-node breakdown.
    label: str = ""
    #: Declared node kind.
    kind: str = ""
    #: Per-child quantities, in declaration order.
    parts: tuple[Quantity, ...] = ()
    #: Free-text notes: quadrature error bounds, degeneracy limits taken,
    #: and any statement that a quantification is not the Boolean one.
    notes: tuple[str, ...] = ()
    #: Extra named numbers a caller may want: switch reliability actually
    #: applied, the sharing factor, the common-cause beta.
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def complement_residual(self) -> float:
        """``R + Q - 1``. Rounding-level for a well-conditioned structure."""
        return self.reliability + self.unreliability - 1.0

    def __str__(self) -> str:
        name = self.label or self.kind
        return f"{name}: R={self.reliability:.12f} Q={self.unreliability:.6e}"


def _leaf(block: Block, age_hours: float) -> Quantity:
    """Quantity for a leaf block at an effective age in hours."""
    hazard = block.distribution.cumulative_hazard(age_hours)
    return Quantity(
        reliability=reliability_from_hazard(hazard),
        unreliability=unreliability_from_hazard(hazard),
        hazard=hazard,
        label=block.name,
        kind="block",
        detail={"effective_age_hours": float(age_hours)},
    )


def quantify(
    diagram: Diagram,
    mission: MissionProfile | float,
    *,
    block_ages: dict[str, float] | None = None,
) -> Quantity:
    """Reliability of ``diagram`` over ``mission``, as a closed form.

    Parameters
    ----------
    diagram
        The declared diagram.
    mission
        A :class:`~reliablock.mission.MissionProfile`, or a bare duration in
        hours, which is treated as a single phase at full duty and unit stress.
    block_ages
        Optional override of each block's effective age in hours, keyed by
        block name. Used by :mod:`reliablock.importance` to pivot a single
        block, and by the validation scripts to drive a structure at a
        declared component reliability rather than through a distribution.

    Returns
    -------
    Quantity
        The system figure, with the per-node breakdown in
        :attr:`Quantity.parts`.
    """
    profile = MissionProfile.from_hours(mission) if not isinstance(mission, MissionProfile) else (
        mission
    )
    ages = profile.effective_ages(diagram) if block_ages is None else dict(block_ages)
    switches = {}
    for index, item in enumerate(diagram.walk()):
        if isinstance(item, Standby):
            switches[id(item)] = switch_event_name(item, index)
    return _quantify_node(diagram.root, ages, profile, switches)


def _quantify_node(
    item: Node | Block,
    ages: dict[str, float],
    profile: MissionProfile,
    switches: dict[int, str],
) -> Quantity:
    """Recursive closed-form quantification."""
    if isinstance(item, Block):
        if item.name not in ages:
            raise KeyError(
                f"no effective age for block {item.name!r}; "
                f"known blocks are {sorted(ages)}"
            )
        return _leaf(item, ages[item.name])

    parts = tuple(_quantify_node(c, ages, profile, switches) for c in _structural_children(item))

    if isinstance(item, Series):
        hazard = series_hazard(p.hazard for p in parts)
        return Quantity(
            reliability=reliability_from_hazard(hazard),
            unreliability=unreliability_from_hazard(hazard),
            hazard=hazard,
            label=item.label or "series",
            kind="series",
            parts=parts,
        )

    if isinstance(item, Parallel):
        unreliability = parallel_unreliability(p.unreliability for p in parts)
        return Quantity(
            reliability=1.0 - unreliability,
            unreliability=unreliability,
            hazard=hazard_from_unreliability(unreliability),
            label=item.label or "parallel",
            kind="parallel",
            parts=parts,
            notes=(
                "Q = prod Q_i is exact; R = 1 - Q is the derived figure and is "
                "the one that loses precision when Q is below eps.",
            ),
        )

    if isinstance(item, KOfN):
        n = len(parts)
        reliability = weighted_binomial_survival([p.reliability for p in parts], item.k)
        unreliability = weighted_binomial_survival(
            [p.unreliability for p in parts], n - item.k + 1
        )
        return Quantity(
            reliability=reliability,
            unreliability=unreliability,
            hazard=hazard_from_unreliability(unreliability),
            label=item.label or f"{item.k}-of-{n}",
            kind="k_of_n",
            parts=parts,
            notes=(
                f"R = P(at least {item.k} of {n} survive) and "
                f"Q = P(at least {n - item.k + 1} of {n} fail) are computed by two "
                "separate non-negative convolutions; neither is 1 minus the other.",
            ),
            detail={"k": float(item.k), "n": float(n)},
        )

    if isinstance(item, SharedLoad):
        return _quantify_shared_load(item, ages, parts)

    if isinstance(item, Standby):
        return _quantify_standby(item, ages, parts)

    if isinstance(item, Network):
        return _quantify_network(item, parts)

    raise TypeError(f"cannot quantify {type(item).__name__}")  # pragma: no cover


def _structural_children(item: Node) -> tuple[Node | Block, ...]:
    """Children in the order the quantifier expects them."""
    return item.children


# ------------------------------------------------------------------- standby


def standby_reliability_cold_identical(
    rate_per_hour: float, t_hours: float, n_standby: int, switch_reliability: float
) -> float:
    """Erlang closed form for ``n_standby`` identical cold standby units.

    ``R(t) = exp(-lambda t) sum_{j=0}^{m} (p_s lambda t)^j / j!``

    Units: ``rate_per_hour`` in 1/hour, ``t_hours`` in hours, the result
    dimensionless in ``[0, 1]``. Derivation in the module docstring.
    """
    if rate_per_hour < 0.0:
        raise ValueError(f"rate_per_hour must be >= 0, got {rate_per_hour}")
    if t_hours < 0.0:
        raise ValueError(f"t_hours must be >= 0, got {t_hours}")
    if n_standby < 1:
        raise ValueError(f"n_standby must be >= 1, got {n_standby}")
    if not 0.0 <= switch_reliability <= 1.0:
        raise ValueError(f"switch_reliability must lie in [0, 1], got {switch_reliability}")
    x = rate_per_hour * t_hours
    scaled = switch_reliability * x
    total = 0.0
    term = 1.0
    for j in range(n_standby + 1):
        if j:
            term *= scaled / j
        total += term
    return reliability_from_hazard(x) * total


def standby_reliability_general(
    primary: FailureDistribution,
    standby: FailureDistribution,
    t_hours: float,
    switch_reliability: float,
    standby_hazard_fraction: float,
) -> tuple[float, float]:
    """Convolution-integral reliability for one standby unit, any distributions.

    ``R(t) = R_p(t) + p_s integral_0^t f_p(u) exp(-H_s(f u + t - u)) du``

    Returns ``(reliability, quadrature_absolute_error_bound)``. The error bound
    is the one ``scipy.integrate.quad`` reports, carried into
    :attr:`Quantity.notes` rather than discarded: a convolution-based figure
    whose quadrature error is not stated is not a checkable number.
    """
    if t_hours < 0.0:
        raise ValueError(f"t_hours must be >= 0, got {t_hours}")
    if not 0.0 <= switch_reliability <= 1.0:
        raise ValueError(f"switch_reliability must lie in [0, 1], got {switch_reliability}")
    if not 0.0 <= standby_hazard_fraction <= 1.0:
        raise ValueError(
            f"standby_hazard_fraction must lie in [0, 1], got {standby_hazard_fraction}"
        )
    base = primary.reliability(t_hours)
    if t_hours == 0.0 or switch_reliability == 0.0:
        return base, 0.0
    fraction = standby_hazard_fraction

    def integrand(u: float) -> float:
        age = fraction * u + (t_hours - u)
        return primary.density(u) * standby.reliability(age)

    value, error = quad(integrand, 0.0, t_hours, limit=200)
    return base + switch_reliability * value, switch_reliability * float(error)


def _quantify_standby(
    item: Standby, ages: dict[str, float], parts: tuple[Quantity, ...]
) -> Quantity:
    """Quantify a standby node by the applicable closed form."""
    assert item.primary is not None
    primary = item.primary
    standbys = item.standbys
    if not isinstance(primary, Block) or not all(isinstance(s, Block) for s in standbys):
        raise ValueError(
            f"standby node {item.label or 'standby'!r}: this package quantifies a "
            "standby group over blocks, not over nested structures. A nested "
            "standby branch needs a convolution over the branch's own density, "
            "which is not implemented."
        )
    duration = ages[primary.name]
    cold = item.standby_hazard_fraction == 0.0
    identical_exponential = (
        isinstance(primary.distribution, Exponential)
        and all(isinstance(s.distribution, Exponential) for s in standbys)
        and all(
            s.distribution.rate_per_hour == primary.distribution.rate_per_hour  # type: ignore[union-attr]
            for s in standbys
        )
    )
    notes: list[str] = []
    if cold and identical_exponential:
        rate = primary.distribution.rate_per_hour  # type: ignore[union-attr]
        reliability = standby_reliability_cold_identical(
            rate, duration, len(standbys), item.switch_reliability
        )
        notes.append(
            f"Erlang closed form for {len(standbys)} identical cold standby unit(s) "
            "at a constant rate; no quadrature."
        )
        error = 0.0
    elif len(standbys) == 1:
        reliability, error = standby_reliability_general(
            primary.distribution,
            standbys[0].distribution,  # type: ignore[union-attr]
            duration,
            item.switch_reliability,
            item.standby_hazard_fraction,
        )
        notes.append(
            f"convolution integral, quad absolute error bound {error:.3e} on the "
            "reliability"
        )
    else:
        raise ValueError(
            f"standby node {item.label or 'standby'!r} declares "
            f"{len(standbys)} standby units with a non-zero hazard fraction or "
            "non-identical distributions. The closed form implemented here covers "
            "any number of identical cold standby units at a constant rate, and "
            "one standby unit of any distribution at any hazard fraction. A warm "
            "multi-unit group of non-identical units needs a nested convolution "
            "that this package does not have, and approximating it silently would "
            "be worse than refusing."
        )
    reliability = min(max(reliability, 0.0), 1.0)
    unreliability = 1.0 - reliability
    notes.append(
        "Q is 1 - R for a standby group: the standby reliability is computed in "
        "R-space, so Q below about 1e-16 is not resolved. Reported rather than "
        "silently returned as a small number that is mostly rounding."
    )
    return Quantity(
        reliability=reliability,
        unreliability=unreliability,
        hazard=hazard_from_unreliability(unreliability),
        label=item.label or "standby",
        kind="standby",
        parts=parts,
        notes=tuple(notes),
        detail={
            "switch_reliability": float(item.switch_reliability),
            "standby_hazard_fraction": float(item.standby_hazard_fraction),
            "n_standby": float(len(standbys)),
            "quadrature_abs_error": float(error),
        },
    )


# --------------------------------------------------------------- shared load


def shared_load_reliability(
    rate_1_per_hour: float,
    rate_2_per_hour: float,
    shared_rate_factor: float,
    t_hours: float,
) -> float:
    """Four-state Markov closed form for two load-sharing units.

    Derivation in the module docstring. Units: rates in 1/hour, ``t_hours`` in
    hours, the reliability dimensionless and in ``[0, 1]``.

    The two difference-of-exponentials terms are evaluated through
    :func:`_expm1_over_x`, which is exact when a denominator vanishes, so this
    function has no degeneracy branch and no tuning constant.
    """
    for name, value in (("rate_1_per_hour", rate_1_per_hour), ("rate_2_per_hour", rate_2_per_hour)):
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    if not 0.0 < shared_rate_factor <= 1.0:
        raise ValueError(f"shared_rate_factor must lie in (0, 1], got {shared_rate_factor}")
    if t_hours < 0.0:
        raise ValueError(f"t_hours must be >= 0, got {t_hours}")
    s = shared_rate_factor
    load = s * (rate_1_per_hour + rate_2_per_hour)
    both = reliability_from_hazard(load * t_hours)
    total = both
    for rate_failed, rate_survivor in (
        (rate_1_per_hour, rate_2_per_hour),
        (rate_2_per_hour, rate_1_per_hour),
    ):
        # (exp(-a t) - exp(-L t)) / (L - a) == t exp(-L t) expm1((L - a) t)
        #                                       / ((L - a) t)
        gap_t = (load - rate_survivor) * t_hours
        total += s * rate_failed * t_hours * both * _expm1_over_x(gap_t)
    return min(max(total, 0.0), 1.0)


def _quantify_shared_load(
    item: SharedLoad, ages: dict[str, float], parts: tuple[Quantity, ...]
) -> Quantity:
    """Quantify a shared-load node."""
    first, second = item.units
    duration = ages[first.name]
    if ages[second.name] != duration:
        raise ValueError(
            f"shared_load node {item.label or 'shared_load'!r}: its two units have "
            f"different effective ages ({duration} and {ages[second.name]} hours). "
            "A shared-load pair carries one load and therefore one clock; a mission "
            "profile that gives them different duty cycles is not a shared-load pair."
        )
    reliability = shared_load_reliability(
        first.distribution.rate_per_hour,  # type: ignore[union-attr]
        second.distribution.rate_per_hour,  # type: ignore[union-attr]
        item.shared_rate_factor,
        duration,
    )
    unreliability = 1.0 - reliability
    notes = [
        "Q is 1 - R for a shared-load pair: the reliability is a sum of "
        "exponentials computed in R-space.",
        "The Boolean structure function of this node is parallel, so its minimal "
        "cut set is both units; the quantification is NOT the independent-event "
        "product, because the survivor's hazard rate depends on the other unit.",
    ]
    notes.append(
        "The difference-of-exponentials terms are evaluated as "
        "t exp(-L t) expm1(g t) / (g t), which is exact when a denominator "
        "vanishes, so there is no degeneracy threshold and no tuning constant."
    )
    return Quantity(
        reliability=reliability,
        unreliability=unreliability,
        hazard=hazard_from_unreliability(unreliability),
        label=item.label or "shared_load",
        kind="shared_load",
        parts=parts,
        notes=tuple(notes),
        detail={"shared_rate_factor": float(item.shared_rate_factor)},
    )


# ------------------------------------------------------------------ network


def _quantify_network(item: Network, parts: tuple[Quantity, ...]) -> Quantity:
    """Quantify a two-terminal network by pivotal decomposition.

    Pivotal (Shannon) decomposition on one edge at a time: ``Q_sys = q_e
    Q(e failed) + (1 - q_e) Q(e working)``, recursing until the residual
    network is a single edge or is decided. The recursion is in unreliability
    space throughout, so a network of high-reliability edges does not lose its
    answer, and the terms are non-negative so there is no cancellation.

    Barlow, R. E. and Proschan, F. (1975), chapter 1, for the decomposition;
    the derivation of the five-edge bridge's closed form from it is in
    :mod:`reliablock.bridge`.
    """
    unreliabilities = {
        block.name: part.unreliability
        for (_, _, block), part in zip(item.edges, parts, strict=True)
    }
    order = [block.name for _, _, block in item.edges]
    unreliability = _network_unreliability(item, order, 0, set(), set(), unreliabilities)
    return Quantity(
        reliability=1.0 - unreliability,
        unreliability=unreliability,
        hazard=hazard_from_unreliability(unreliability),
        label=item.label or "network",
        kind="network",
        parts=parts,
        notes=(
            "Pivotal decomposition in unreliability space over "
            f"{len(item.edges)} edges; terms are non-negative, so no cancellation.",
        ),
        detail={"n_edges": float(len(item.edges))},
    )


def _network_unreliability(
    item: Network,
    order: list[str],
    index: int,
    failed: set[str],
    working: set[str],
    unreliabilities: dict[str, float],
) -> float:
    """Exact ``Q`` of a network by pivotal decomposition, in unreliability space."""
    # Decided already? If the sink is unreachable using every undecided edge as
    # working, the system has failed whatever the rest do; if it is reachable
    # using only edges already declared working, it has succeeded.
    optimistic = set(order) - failed
    if not _connected(item, optimistic):
        return 1.0
    if _connected(item, working):
        return 0.0
    if index >= len(order):  # pragma: no cover - both branches above decide
        return 0.0
    name = order[index]
    if name in failed or name in working:
        return _network_unreliability(item, order, index + 1, failed, working, unreliabilities)
    q = unreliabilities[name]
    down = _network_unreliability(
        item, order, index + 1, failed | {name}, working, unreliabilities
    )
    up = _network_unreliability(item, order, index + 1, failed, working | {name}, unreliabilities)
    return q * down + (1.0 - q) * up


def _connected(item: Network, working: set[str]) -> bool:
    """True if ``working`` edges connect source to sink."""
    adjacency: dict[str, list[str]] = {}
    for a, b, block in item.edges:
        if block.name in working:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
    seen = {item.source}
    stack = [item.source]
    while stack:
        current = stack.pop()
        if current == item.sink:
            return True
        for neighbour in adjacency.get(current, ()):
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return item.sink in seen
