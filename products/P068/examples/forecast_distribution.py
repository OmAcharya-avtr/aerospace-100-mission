"""Predictive distributions over the crossing time for one complete and one censored unit.

What to notice: every model produces a distribution, not a date. The curves
are predictive CDFs of the remaining life; where a curve has already reached 1
to the left of the dotted truth line, that model has placed essentially no
probability on the outcome that happened. On the complete unit the learned
forecaster is sharp and close to the truth. On the censored unit -- a
long-lived one -- every model is confidently short, and the learned forecaster
is the most confident and the most wrong.

The CDF is plotted rather than a density because the learned forecaster carries
only seven fitted quantiles, so its density is a histogram of seven cells and
its tails are flat by construction; the CDF shows that honestly.
"""

from __future__ import annotations

import numpy as np

from _common import MODEL_COLOURS, TRUTH, add_src_to_path, save, style

add_src_to_path()

import matplotlib.pyplot as plt  # noqa: E402

from degradepredict.baselines import exponential_baseline, linear_baseline  # noqa: E402
from degradepredict.generate import DEFAULT_LIMIT, generate_population  # noqa: E402
from degradepredict.learned import LearnedCrossingForecaster  # noqa: E402
from degradepredict.physics import powerlaw_fade_baseline  # noqa: E402

LEVELS = np.linspace(0.01, 0.99, 199)


def main() -> None:
    style()
    train = generate_population(1200, seed=20261016)
    test = generate_population(400, seed=20261017)
    model = LearnedCrossingForecaster(limit=DEFAULT_LIMIT).fit(train)

    complete = next(tr for tr in test if tr.event and tr.remaining_true_days > 200.0)
    censored = next(tr for tr in test if not tr.event and tr.remaining_true_days > 1200.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, tr, label in ((axes[0], complete, "complete"),
                          (axes[1], censored, "right-censored")):
        forecasts = {}
        for name, fitter in (("linear", linear_baseline),
                             ("exponential", exponential_baseline),
                             ("powerlaw-fade", powerlaw_fade_baseline)):
            forecasts[name] = fitter(tr.times_days, tr.values_percent,
                                     DEFAULT_LIMIT).forecast
        forecasts["learned-ipcw"] = model.predict(tr)
        upper = max(tr.remaining_true_days * 1.4,
                    max(float(f.quantile(0.97)) for f in forecasts.values()) * 1.05)
        for name, f in forecasts.items():
            q = np.asarray(f.quantile(LEVELS), dtype=float)
            ax.plot(q, LEVELS, color=MODEL_COLOURS[name], lw=1.3, label=name)
            lo, hi = f.interval(0.90)
            ax.plot([lo, hi], [-0.06 - 0.05 * list(forecasts).index(name)] * 2,
                    color=MODEL_COLOURS[name], lw=2.5, solid_capstyle="butt")
        ax.axvline(tr.remaining_true_days, color=TRUTH, lw=1.2, ls=":",
                   label="true remaining life")
        if not tr.event:
            ax.axvline(tr.followup_days, color="#d9700b", lw=1.0, ls="-.",
                       label="end of follow-up (all that is known in the field)")
        ax.set_xlim(0.0, upper)
        ax.set_ylim(-0.30, 1.05)
        ax.set_xlabel("remaining life from the forecast origin [d]")
        ax.set_ylabel("predictive CDF")
        ax.set_title(
            f"unit {tr.traj_id} ({label}, true shape {tr.family}): truth "
            f"{tr.remaining_true_days:.0f} d",
            fontsize=10,
        )
        ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle(
        "Predictive CDFs of remaining life, with each model's 90 percent interval "
        "drawn beneath",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    print(f"wrote {save(fig, 'forecast_distribution.png')}")


if __name__ == "__main__":
    main()
