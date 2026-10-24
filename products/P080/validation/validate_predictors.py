"""Learned predictor against classical predictors, on rate, at identical distortion.

The classical predictors were implemented first and the result is published as
measured. Nothing here is retuned to make the learned component win, and a
baseline win is research evidence.

Why the comparison is clean
---------------------------

Every predictor in this script drives the same quantiser at the same declared
bound, and everything downstream of the quantiser is lossless on the integer
symbol stream. The reconstruction is therefore bit-identical across every
predictor and the distortion is *identical*, not matched by search. Comparing
rates is comparing one number, and the number is ``len(bytes)`` of a container
that round-trips.

What is charged to the learned predictor
----------------------------------------

Its parameters. A learned predictor has to be on the spacecraft and on the
ground, and the break-even array count -- model bytes divided by mean bytes
saved per array -- is reported whether or not it flatters the model. Also
reported: the classical predictors are integer-exact and portable bit-for-bit
to any machine with int64, and the learned ones are not, because their forward
pass calls ``tanh``.
"""

from __future__ import annotations

from _bootstrap import Tee, Timer, bootstrap, environment_lines, fit_models

ROOT = bootstrap()

import numpy as np  # noqa: E402

from boundcompress import benchmark, dataset  # noqa: E402
from boundcompress.codec import encode  # noqa: E402
from boundcompress.predictors import CLASSICAL_NAMES, noise_gain  # noqa: E402
from boundcompress.ratedistortion import geometric_bounds  # noqa: E402
from boundcompress.report import format_table  # noqa: E402

SAMPLES = 4096
POINTS = 5


def main() -> int:
    out = Tee("validate_predictors_output.txt")
    out.rule("boundcompress -- learned against classical, on rate, identical distortion")
    for line in environment_lines():
        out(line)

    signals = {
        name: signal.values
        for name, signal in dataset.generate_all(samples=SAMPLES).items()
    }
    bounds = {
        name: geometric_bounds(values, count=POINTS) for name, values in signals.items()
    }

    fit_timer = Timer()
    with fit_timer:
        models = fit_models()
    out("")
    out("learned models fitted on the disjoint training realisations (seed offset 1000)")
    for label, model in models.items():
        out(
            f"  learned_{label:<7s}: members={model.members} rows={model.train_rows} "
            f"params={model.parameter_count} bytes={model.model_bytes} "
            f"converged={model.converged}"
        )

    out.rule("classical predictor noise gains")
    out("Sum of the absolute finite-difference coefficients. A first-order account of")
    out("why a higher-order extrapolator loses on a noisy source: independent errors")
    out("of scale sigma in the history give a prediction error scaling with this.")
    out(
        format_table(
            ["predictor", "coefficients", "noise_gain"],
            [
                [
                    name,
                    str(list(__import__(
                        "boundcompress.predictors", fromlist=["x"]
                    ).CLASSICAL_COEFFICIENTS[name])),
                    str(noise_gain(name)),
                ]
                for name in CLASSICAL_NAMES
            ],
        )
    )

    out.rule("full benchmark: every signal, five bounds, every predictor")
    cells = benchmark.run(signals, bounds, dict(models))
    out.check("every cell holds its bound", all(cell.bound_holds for cell in cells))
    out.check(
        "distortion is identical within every cell group",
        all(
            len({
                cell.max_error
                for cell in cells
                if cell.signal == signal and cell.bound == bound
            }) == 1
            for signal in signals
            for bound in bounds[signal]
        ),
    )

    predictor_names = [*CLASSICAL_NAMES, "learned_linear", "learned_mlp"]
    rows: list[list[str]] = []
    for signal in signals:
        for bound in bounds[signal]:
            group = {
                cell.predictor: cell
                for cell in cells
                if cell.signal == signal and cell.bound == bound
            }
            best = min(group.values(), key=lambda cell: cell.encoded_bytes)
            row = [signal, f"{bound:.3e}"]
            row += [str(group[name].encoded_bytes) for name in predictor_names]
            row.append(best.predictor)
            rows.append(row)
    out("")
    out(format_table(["signal", "bound", *predictor_names, "winner"], rows))

    out.rule("verdict")
    verdict = benchmark.summarise(cells)
    totals: dict[str, int] = {}
    for cell in cells:
        totals[cell.predictor] = totals.get(cell.predictor, 0) + cell.encoded_bytes
    out(
        format_table(
            ["item", "value"],
            [
                ["cells compared", str(len(bounds) * POINTS)],
                ["cells won by a classical predictor", str(verdict.classical_wins)],
                ["cells won by a learned predictor", str(verdict.learned_wins)],
                ["median learned gain over best classical, %",
                 f"{verdict.median_learned_gain_pct:.4f}"],
                ["max learned gain over best classical, %",
                 f"{verdict.max_learned_gain_pct:.4f}"],
                ["worst learned loss against best classical, %",
                 f"{verdict.worst_learned_loss_pct:.4f}"],
                ["best classical predictor overall", verdict.best_classical_overall],
                ["best learned predictor overall", verdict.best_learned_overall],
            ],
        )
    )
    out("")
    out("total bytes over every cell, by predictor:")
    out(
        format_table(
            ["predictor", "total_bytes", "ratio_to_best"],
            [
                [name, str(totals[name]), f"{totals[name] / min(totals.values()):.4f}"]
                for name in sorted(totals, key=lambda key: totals[key])
            ],
        )
    )

    out.rule("the cost of the learned predictor's parameters")
    out("Bytes the decoder must already hold, against the mean bytes saved per array.")
    out("An infinite break-even means the model saves nothing on average, which is")
    out("the honest answer and not an error.")
    cost_rows = []
    for label, model in models.items():
        break_even = benchmark.break_even_arrays(cells, label, model.model_bytes)
        savings = []
        for signal in signals:
            for bound in bounds[signal]:
                group = {
                    cell.predictor: cell
                    for cell in cells
                    if cell.signal == signal and cell.bound == bound
                }
                best_classical = min(
                    (group[name] for name in CLASSICAL_NAMES),
                    key=lambda cell: cell.encoded_bytes,
                )
                savings.append(
                    best_classical.encoded_bytes - group[f"learned_{label}"].encoded_bytes
                )
        cost_rows.append(
            [
                f"learned_{label}",
                str(model.parameter_count),
                str(model.model_bytes),
                f"{float(np.mean(savings)):.2f}",
                f"{float(np.median(savings)):.2f}",
                f"{float(np.min(savings)):.0f}",
                f"{float(np.max(savings)):.0f}",
                "inf" if break_even == float("inf") else f"{break_even:.1f}",
            ]
        )
    out("")
    out(
        format_table(
            ["predictor", "params", "model_bytes", "mean_saving_B", "median_saving_B",
             "worst_saving_B", "best_saving_B", "break_even_arrays"],
            cost_rows,
        )
    )

    out.rule("portability: which predictors are bit-exact on any machine")
    out(
        format_table(
            ["predictor", "arithmetic", "cross-machine bit-exact"],
            [
                *[[name, "integer only", "yes"] for name in CLASSICAL_NAMES],
                ["learned_linear", "float64 dot product", "no (libm-free, but float)"],
                ["learned_mlp", "float64 with tanh", "no (tanh is a libm call)"],
            ],
        )
    )
    out("")
    out("A coder whose decoder runs on different silicon from its encoder needs the")
    out("prediction to be reproducible bit-for-bit, because the residual stream is")
    out("decoded against it. The classical predictors give that for free. The learned")
    out("ones do not, and no amount of rate improvement substitutes for it.")

    out.rule("timings -- UNSTABLE, for order of magnitude only")
    out("Wall clock on a shared two-core container; no conclusion derives from these.")
    timing_rows = [["model fit (both kinds)", f"{fit_timer.elapsed_s * 1000.0:.0f}", "1"]]
    for name in (*CLASSICAL_NAMES, "learned"):
        timer = Timer()
        model = models["mlp"] if name == "learned" else None
        for _ in range(3):
            with timer:
                encode(
                    signals["battery_voltage"], "abs", 1e-3,
                    predictor=name, model=model,
                )
        timing_rows.append([f"encode {name}", f"{timer.mean_ms:.2f}", str(timer.calls)])
    out(format_table(["operation", "mean_ms", "calls"], timing_rows))

    path = out.save()
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
