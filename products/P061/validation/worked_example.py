"""The README's worked example, run so that its printed output cannot drift.

Everything the README shows under "A worked example" is produced here and
written to ``worked_example_output.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from _bootstrap import Tee, add_src_to_path

add_src_to_path()

from activityplan import (  # noqa: E402
    DurationEnergyPredictor,
    evaluate,
    feature_matrix,
    named_instance,
    plan_greedy,
    realise,
    render_feasibility,
    sample_requests,
    solve_exact,
)
from activityplan.timeline import RasterisedProblem  # noqa: E402

OUTPUT = Path(__file__).with_name("worked_example_output.txt")


def main() -> int:
    say = Tee(OUTPUT)

    # Declare the problem and project it onto a 150 s grid.
    problem = named_instance("leo-imaging")
    rasterised = RasterisedProblem.build(problem, 150.0)

    # Baseline 1: priority-ordered earliest-feasible-start insertion.
    greedy = plan_greedy(rasterised)

    # Baseline 2: the exact optimum of the discretised problem.
    exact = solve_exact(rasterised, time_limit_s=150.0)

    say(f"instance            : {problem.name}")
    say(
        f"activities          : {problem.n_activities}, total priority "
        f"{problem.total_priority:.1f}"
    )
    say(f"grid                : {rasterised.n_steps} steps of {rasterised.grid.step_s:.0f} s")
    say(f"greedy objective    : {greedy.value:.1f} in {greedy.simulations} simulations")
    say(
        f"exact objective     : {exact.value:.1f}, {exact.node_count} nodes, "
        f"{exact.solve_s:.3f} s, proven optimal {exact.proven_optimal}"
    )
    say(f"greedy shortfall    : {100.0 * (1.0 - greedy.value / exact.value):.2f} %")
    say("")

    # The deliverable: the binding constraint and the slack on everything else.
    report = evaluate(rasterised, exact.plan)
    say(render_feasibility(report, rasterised))
    say("")

    # The learned predictor, with a calibrated interval on each activity.
    rng = np.random.default_rng(61)
    requests = sample_requests(1000, rng)
    realised = [realise(request, rng) for request in requests]
    predictor = DurationEnergyPredictor(
        alpha=0.10, mode="normalised", n_estimators=160, random_state=61
    ).fit(
        feature_matrix(list(requests)),
        np.array([item.duration_s for item in realised], dtype=float),
        np.array([item.energy_wh for item in realised], dtype=float),
    )
    held_out = sample_requests(5, np.random.default_rng(99), name_prefix="new")
    duration_interval, energy_interval = predictor.predict_requests(held_out)
    say("learned duration and energy predictions with 90 % conformal intervals")
    say(
        f"{'activity':10s} {'type':12s} {'duration/s':>11s} {'lower':>9s} {'upper':>9s} "
        f"{'energy/Wh':>10s} {'lower':>8s} {'upper':>8s}"
    )
    for index, request in enumerate(held_out):
        say(
            f"{request.name:10s} {request.activity_type:12s} "
            f"{duration_interval.point[index]:11.2f} {duration_interval.lower[index]:9.2f} "
            f"{duration_interval.upper[index]:9.2f} {energy_interval.point[index]:10.4f} "
            f"{energy_interval.lower[index]:8.4f} {energy_interval.upper[index]:8.4f}"
        )
    say("")
    say(
        f"calibration split   : {predictor.n_train_} proper-train, "
        f"{predictor.n_calibration_} calibration"
    )
    say(
        f"duration quantile   : {predictor.duration.quantile_:.6f} "
        f"(normalised nonconformity, dimensionless)"
    )
    say(f"level attainable    : {predictor.duration.level_attainable}")
    say.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
