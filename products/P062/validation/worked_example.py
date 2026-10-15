"""The worked example quoted in README.md, run as a script.

Run from the repository root: ``python validation/worked_example.py``
Writes ``validation/worked_example_output.txt``.

The README quotes this file's code and this file's committed output, so the two
cannot drift apart.
"""

from __future__ import annotations

import time

import numpy as np

from _bootstrap import Tee, add_src_to_path

ROOT = add_src_to_path()

from twincalib import (  # noqa: E402
    CalibrationProblem,
    analyse,
    calibration_model,
    decompose_residual,
    design,
    fit,
    log_sensitivity,
    make_dataset,
    profile_likelihood,
    run_benchmark,
)


def main() -> None:
    started = time.perf_counter()
    out = Tee(ROOT / "validation" / "worked_example_output.txt")

    # 1. Ask what the design can determine, BEFORE fitting anything.
    d = design("full-sun")
    model = calibration_model()
    data = make_dataset(d, seed=11, model=model)
    sens = log_sensitivity(
        model, d.theta_true, data.inputs, d.initial_state, d.channels, data.sigma
    )
    report = analyse(sens)
    out(report.verdict())
    for combo in report.null_combinations:
        out(f"unconstrained: {combo.as_text()}")

    # 2. Fit the physics anyway, and see what the covariance claims.
    problem = CalibrationProblem(
        model=model,
        inputs=data.inputs,
        initial_state=d.initial_state,
        channels=d.channels,
        sample_stride=d.sample_stride,
        y=data.y,
        sigma=data.sigma,
    )
    result = fit(problem, np.asarray(d.theta_true, dtype=float))
    out("")
    out(f"q_sun estimate {result.theta[4]:.4f} W, relative sd {result.relative_sd[4]}")
    out(f"q_sun + q_par  {result.theta[4] + result.theta[5]:.4f} W "
        f"(true {d.theta_true[4] + d.theta_true[5]:.4f} W)")

    # 3. Profile it, which is the answer the covariance could not give.
    curve = profile_likelihood(problem, result, 4, n_points=2, refine_steps=8)
    out("")
    out(f"q_sun profile interval: lower {curve.lower}, upper {curve.upper:.4f} W")
    out(f"searched [{curve.search_lower:.4f}, {curve.search_upper:.4f}] W")

    # 4. Only now, the learned residual, and the absorption measurement with it.
    bench = run_benchmark(design("baseline-eclipse"), n_train=3, n_test=2)
    out("")
    out(f"held-out RMSE physics {bench.rmse_physics_pooled:.4f} K, "
        f"with residual {bench.rmse_combined_pooled:.4f} K")
    out(f"improvement {bench.improvement_pooled:+.4f}, winner {bench.winner}")
    split = bench.splits[1e-2]
    out(f"absorption (weak) fraction {split.weak_fraction:.5f}, "
        f"genuine model-form fraction {split.orthogonal_fraction:.5f}")
    out(f"physics parameters got worse: {bench.drift.parameters_got_worse}")

    # 5. The same decomposition done by hand, to show the API is not a black box.
    test = make_dataset(design("baseline-eclipse"), seed=99, phase=0.8)
    sens_test = log_sensitivity(
        model,
        bench.physics_fit.theta,
        test.inputs,
        test.design.initial_state,
        test.design.channels,
        test.sigma,
    )
    field = test.y - problem.model.observe(
        bench.physics_fit.theta,
        test.inputs,
        test.design.initial_state,
        test.design.channels,
    )
    by_hand = decompose_residual(sens_test, field)
    out("")
    out(f"raw held-out residual field: strong {by_hand.strong_fraction:.5f}, "
        f"weak {by_hand.weak_fraction:.5f}, orthogonal {by_hand.orthogonal_fraction:.5f}")

    out("")
    out(f"elapsed {time.perf_counter() - started:.1f} s")
    out.save()


if __name__ == "__main__":
    main()
