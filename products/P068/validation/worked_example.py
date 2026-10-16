"""The worked example quoted in README.md, run so the output cannot drift.

Everything between the two markers below is the code block in the README. Its
printed output is the block beneath it there, copied from
``validation/worked_example_output.txt``.
"""

from __future__ import annotations

from _bootstrap import Tee, add_src_to_path, environment_lines

ROOT = add_src_to_path()
OUT = ROOT / "validation" / "worked_example_output.txt"


def main() -> int:
    say = Tee(OUT)
    say("worked_example.py -- the README example, executed")
    for line in environment_lines():
        say(line)
    say("")

    # --- README EXAMPLE BEGINS ---
    import numpy as np

    from degradepredict import (
        DEFAULT_LIMIT,
        LearnedCrossingForecaster,
        coverage_by_censoring,
        exponential_baseline,
        generate_population,
        linear_baseline,
        powerlaw_fade_baseline,
    )

    train = generate_population(1200, seed=20261016)
    test = generate_population(400, seed=20261017)
    unit = test[0]

    lines = [
        f"unit {unit.traj_id}: {len(unit.times_days)} samples to day "
        f"{unit.origin_days:.0f}, true crossing {unit.crossing_time_days:.0f} d "
        f"({'complete' if unit.event else 'right-censored'})",
        f"{'model':<22}{'median d':>10}{'90% interval d':>22}",
    ]
    for fitter in (linear_baseline, exponential_baseline, powerlaw_fade_baseline):
        fit = fitter(unit.times_days, unit.values_percent, DEFAULT_LIMIT)
        lo, hi = fit.forecast.interval(0.90)
        lines.append(f"{fit.model:<22}{fit.forecast.median():>10.1f}"
                     f"{f'[{lo:.1f}, {hi:.1f}]':>22}")

    model = LearnedCrossingForecaster(limit=DEFAULT_LIMIT).fit(train)
    lo, hi = model.predict(unit).interval(0.90)
    lines.append(f"{'learned-ipcw':<22}{model.predict(unit).median():>10.1f}"
                 f"{f'[{lo:.1f}, {hi:.1f}]':>22}")
    lines.append(f"{'truth':<22}{unit.remaining_true_days:>10.1f}{'':>22}")

    truth = np.array([tr.remaining_true_days for tr in test])
    event = np.array([tr.event for tr in test])
    cov = coverage_by_censoring(model.predict_many(test), truth, event, level=0.90)
    lines.append("")
    lines.append(f"90% interval coverage over {int(cov['n'])} held-out units:")
    lines.append(f"  complete ({int(cov['n_complete'])})  : {cov['coverage_complete']:.4f}")
    lines.append(f"  censored ({int(cov['n_censored'])})  : {cov['coverage_censored']:.4f}")
    lines.append(f"  pooled                : {cov['coverage_pooled']:.4f}  "
                 f"<- matches neither")
    print("\n".join(lines))
    # --- README EXAMPLE ENDS ---

    for line in lines:
        say.lines.append(line)
    path = say.save()
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
