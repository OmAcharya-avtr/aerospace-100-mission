"""Held resource draw over a sequence, with the ceiling and the breaches.

The accumulated-draw defect is the one a per-command check cannot see, because
no single command in this dictionary exceeds a ceiling on its own. The figure
shows why: the breach is an area under a step function built from overlapping
holds, not a property of any one command.
"""

from __future__ import annotations

import _common
from _common import FORWARD, REFUSE, SHADE, save, style

_common.bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from cmdguard.loader import load_case  # noqa: E402
from cmdguard.trace import ViolationKind, forward_execute  # noqa: E402


def profile(case, key: str) -> tuple[list[float], list[float]]:
    """Piecewise-constant held draw of ``key``, sampled on hold boundaries.

    The profile is exact between boundaries because a hold contributes a
    constant draw over ``[t_start, t_start + hold_s)``; the sample points are
    every start and every end, which is where it can change.
    """
    events: list[tuple[float, float]] = []
    for step in case.sequence.steps:
        spec = case.dictionary.get(step.command)
        if spec is None or key not in spec.resource_draw:
            continue
        amount = float(spec.resource_draw[key])
        events.append((step.t_s, amount))
        events.append((step.t_s + float(spec.hold_s), -amount))
    events.sort()
    times: list[float] = [0.0]
    levels: list[float] = [0.0]
    running = 0.0
    for when, delta in events:
        times.append(when)
        levels.append(running)
        running += delta
        times.append(when)
        levels.append(running)
    times.append(max(times) + 20.0)
    levels.append(running)
    return times, levels


def main() -> None:
    style()
    case = load_case("examples/cases/resource_and_exclusion.yaml")
    report = forward_execute(case)
    keys = sorted(case.dictionary.resource_limits.limits)

    figure = plt.figure(figsize=(10.5, 6.4))
    grid = figure.add_gridspec(len(keys), 1, hspace=0.35)
    for row, key in enumerate(keys):
        axis = figure.add_subplot(grid[row])
        times, levels = profile(case, key)
        ceiling = case.dictionary.resource_limits.limit_for(key) or 0.0
        axis.step(times, levels, where="post", color=FORWARD, linewidth=1.6,
                  label="held draw")
        axis.axhline(ceiling, color=REFUSE, linewidth=1.2, linestyle="--",
                     label=f"declared ceiling {ceiling:g}")
        axis.fill_between(
            times,
            [ceiling] * len(times),
            levels,
            where=[level > ceiling for level in levels],
            step="post",
            color=SHADE,
            label="over the ceiling",
        )
        for violation in report.violations:
            if violation.kind.value != "resource_limit" or key not in violation.detail:
                continue
            axis.plot([violation.t_s], [ceiling], marker="v", markersize=8, color=REFUSE)
            axis.annotate(
                f"step {violation.step} {violation.command}\nover by "
                f"{violation.shortfall:.6g}",
                (violation.t_s, ceiling),
                textcoords="offset points",
                xytext=(8, 14),
                fontsize=7,
                color=REFUSE,
            )
        # Two stagger rows, for the same reason as in violation_timeline.py:
        # commands close in time overlap illegibly on one row.
        for index, step in enumerate(case.sequence.steps):
            axis.annotate(
                step.command,
                (step.t_s, 0.0),
                textcoords="offset points",
                xytext=(0, -14 if index % 2 == 0 else -24),
                ha="center",
                fontsize=6,
                color="#555555",
            )
        axis.set_ylabel(key)
        axis.set_xlabel("time after sequence epoch (s)")
        axis.set_ylim(-0.24 * max(levels + [ceiling]), 1.35 * max(levels + [ceiling]))
        axis.legend(loc="upper right", fontsize=7, ncol=3)

    figure.suptitle(
        "resource_and_exclusion: no single command exceeds a ceiling, and the sequence "
        "crosses both\n"
        "the breach is an accumulation over overlapping holds, which a per-command check "
        "cannot see",
        fontsize=10,
    )
    print(f"wrote {save(figure, 'resource_profile')}")
    for key in keys:
        times, levels = profile(case, key)
        print(f"{key:<18s} peak held {max(levels):.6g} against a ceiling of "
              f"{case.dictionary.resource_limits.limit_for(key):.6g}")
    for violation in report.of_kind(ViolationKind.RESOURCE_LIMIT):
        print(f"  step {violation.step} {violation.command}: over by {violation.shortfall:.6g}")


if __name__ == "__main__":
    main()
