"""The rules of thumb this package is competing with, implemented honestly.

Each function here is a rule a link budget actually uses. They are
implemented so that the gap between them and the optimum can be *measured*
rather than asserted: see ``validation/validate_heuristic_gap.py``. Where a
heuristic matches the optimum, that is reported too, because a rule of thumb
that is usually right is a reasonable thing to keep using.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problem import InfeasibleProblem, RateProblem
from .select import select_rate

__all__ = [
    "HeuristicComparison",
    "compare_to_optimum",
    "highest_feasible_rate",
    "highest_rate_within_reserve",
]


def highest_feasible_rate(problem: RateProblem) -> int:
    """Highest-rate MODCOD whose own availability meets the target.

    The standard rule: pick the fastest thing that closes. It ignores that
    goodput is rate times availability, so a MODCOD that barely meets the
    target can lose to a slower one that meets it comfortably -- and it
    ignores time-sharing entirely.

    Returns the MODCOD index. Raises :class:`InfeasibleProblem` if none
    qualifies.
    """
    avail = problem.availabilities()
    rates = problem.modcods.rates
    ok = np.flatnonzero(avail >= problem.availability_target)
    if ok.size == 0:
        best = int(np.argmax(avail))
        raise InfeasibleProblem(
            f"no MODCOD meets availability target {problem.availability_target:.6f}; best is "
            f"{problem.modcods.names[best]!r} at {float(avail[best]):.6f}",
            best_availability=float(avail[best]),
            best_modcod=problem.modcods.names[best],
            target=problem.availability_target,
            mode=problem.mode,
        )
    # Highest rate, tie-broken by lower threshold then canonical index.
    order = sorted(ok, key=lambda i: (-rates[i], problem.modcods.thresholds_db[i], i))
    return int(order[0])


def highest_rate_within_reserve(problem: RateProblem, reserve_db: float) -> int:
    """Highest-rate MODCOD whose threshold fits ``margin - reserve_db``.

    The fixed-margin rule: reserve a fade allowance in dB, then take the
    fastest MODCOD that still closes on what is left. It never looks at the
    fade distribution at all, which is why the reserve has to be guessed.

    Raises :class:`InfeasibleProblem` if nothing fits.
    """
    budget = problem.margin_db - float(reserve_db)
    thresholds = problem.modcods.thresholds_db
    rates = problem.modcods.rates
    ok = np.flatnonzero(thresholds <= budget + 1e-12)
    if ok.size == 0:
        avail = problem.availabilities()
        best = int(np.argmin(thresholds))
        raise InfeasibleProblem(
            f"no MODCOD threshold fits a {budget:.2f} dB budget "
            f"({problem.margin_db:.2f} dB margin less {float(reserve_db):.2f} dB reserve); "
            f"the lowest threshold in the table is {float(thresholds[best]):.2f} dB",
            best_availability=float(avail[best]),
            best_modcod=problem.modcods.names[best],
            target=problem.availability_target,
            mode=problem.mode,
        )
    order = sorted(ok, key=lambda i: (-rates[i], thresholds[i], i))
    return int(order[0])


@dataclass(frozen=True)
class HeuristicComparison:
    """One heuristic measured against the optimum on one instance.

    Attributes
    ----------
    name
        Heuristic label.
    heuristic_goodput
        Expected goodput of the heuristic's choice, bits/symbol. NaN when the
        heuristic refused to choose.
    optimal_goodput
        Expected goodput of the optimum, bits/symbol.
    relative_loss
        ``1 - heuristic / optimal``, dimensionless. NaN when the heuristic
        refused to choose. **A negative value is not the heuristic winning.**
        It means the heuristic chose a MODCOD that violates the availability
        constraint, so its goodput is being compared against a different,
        unconstrained problem; ``heuristic_meets_target`` is False in that
        case and the comparison should be read as a constraint violation, not
        as a gain.
    heuristic_meets_target
        Whether the heuristic's choice actually satisfies the availability
        constraint. A heuristic can fail this while the instance is feasible.
    matches_optimum
        True when the heuristic picked an optimal support.
    """

    name: str
    heuristic_goodput: float
    optimal_goodput: float
    relative_loss: float
    heuristic_meets_target: bool
    matches_optimum: bool


def compare_to_optimum(
    problem: RateProblem, *, reserve_db: float = 3.0
) -> tuple[HeuristicComparison, ...]:
    """Measure both heuristics against the optimum on ``problem``.

    ``reserve_db`` is the fade allowance given to
    :func:`highest_rate_within_reserve`.
    """
    optimum = select_rate(problem)
    avail = problem.availabilities()
    goodput = problem.modcods.rates * avail
    out = []
    for name, picker in (
        ("highest_feasible_rate", lambda p: highest_feasible_rate(p)),
        ("highest_rate_within_reserve", lambda p: highest_rate_within_reserve(p, reserve_db)),
    ):
        try:
            idx = picker(problem)
        except InfeasibleProblem:
            out.append(
                HeuristicComparison(
                    name=name,
                    heuristic_goodput=float("nan"),
                    optimal_goodput=optimum.expected_goodput,
                    relative_loss=float("nan"),
                    heuristic_meets_target=False,
                    matches_optimum=False,
                )
            )
            continue
        g = float(goodput[idx])
        out.append(
            HeuristicComparison(
                name=name,
                heuristic_goodput=g,
                optimal_goodput=optimum.expected_goodput,
                relative_loss=float(1.0 - g / optimum.expected_goodput),
                heuristic_meets_target=bool(avail[idx] >= problem.availability_target),
                matches_optimum=bool(
                    abs(g - optimum.expected_goodput) <= optimum.tolerance
                ),
            )
        )
    return tuple(out)
