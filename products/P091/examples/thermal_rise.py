"""Die temperature against time, with the RC fit and the equilibrium window.

What to notice: the fitted relaxation sits on the measured rise, the shaded band
on the right is the window the equilibrium test is applied to, and the annotated
slope is the number the verdict turns on. The lower panel is the fit residual,
which is where a single-exponential model shows whether the system really has one
time constant.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import _common  # noqa: E402

ROOT = _common.bootstrap()

import numpy as np  # noqa: E402

from thermalbudget.equilibrium import equilibrium_test  # noqa: E402
from thermalbudget.rc_model import fit_exponential_relaxation  # noqa: E402


def main() -> int:
    plt = _common.pyplot()
    campaign, is_measured = _common.load_for_example("no_throttle")
    trace = campaign["soak_1.00_r1"]
    fit = fit_exponential_relaxation(trace.t_rel, trace.die_temperature_c)
    verdict = equilibrium_test(
        trace, window_s=_common.WINDOW_S, threshold_c_per_s=_common.THRESHOLD_C_PER_S
    )
    colour = _common.MEASURED_COLOUR if is_measured else _common.SYNTHETIC_COLOUR

    figure, (upper, lower) = plt.subplots(
        2, 1, figsize=(8.4, 5.6), sharex=True, height_ratios=[2.4, 1.0]
    )
    time_s = trace.t_rel
    temperature = trace.die_temperature_c
    upper.plot(time_s, temperature, lw=0.9, color=colour, label="tj-thermal, measured channel")
    model = fit.predict(time_s - float(np.nanmin(time_s)))
    upper.plot(
        time_s,
        model,
        lw=1.6,
        ls="--",
        color=_common.ACCENT,
        label=(
            f"RC fit: Tinf {fit.asymptote_c:.2f} degC, tau {fit.tau_s:.1f} s, "
            f"rms {fit.rms_residual_c:.3f} degC"
        ),
    )
    upper.axhline(
        fit.asymptote_c, lw=0.8, ls=":", color=_common.ACCENT,
        label=f"fitted asymptote {fit.asymptote_c:.2f} degC",
    )
    end = float(time_s[-1])
    upper.axvspan(
        end - _common.WINDOW_S,
        end,
        color="#8fb8d8",
        alpha=0.25,
        label=f"equilibrium window, final {_common.WINDOW_S:g} s",
    )
    upper.annotate(
        f"{verdict.status}\nslope {verdict.slope_c_per_s:+.6f} degC/s\n"
        f"criterion {verdict.threshold_c_per_s:g} degC/s\n"
        f"n = {verdict.n_samples} finite samples",
        xy=(end - 0.5 * _common.WINDOW_S, float(np.nanmean(temperature[-30:]))),
        xytext=(0.30, 0.30),
        textcoords="axes fraction",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.4", "fc": "#ffffff", "ec": "#888888", "lw": 0.6},
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "#666666"},
    )
    upper.set_ylabel("die temperature, degC")
    upper.set_title(
        f"Die-temperature rise, run {trace.label}, duty {trace.duty:g}  "
        f"[{'MEASURED' if is_measured else 'SYNTHETIC'}]"
    )
    upper.legend(loc="lower right", fontsize=7.4, ncols=2)

    residual = temperature - model
    lower.plot(time_s, residual, lw=0.8, color="#555555")
    lower.axhline(0.0, lw=0.8, color=_common.ACCENT)
    lower.set_ylabel("residual, degC")
    lower.set_xlabel("time since run start, s")
    lower.set_title(
        f"Fit residual. Lag-1 autocorrelation {fit.residual_autocorrelation:+.3f}: a value "
        "near zero means one time constant is enough, a large positive value means it is not",
        fontsize=8,
    )

    _common.stamp(
        figure,
        [upper, lower],
        is_measured,
        detail=f"window spans {fit.span_in_tau:.2f} tau.",
    )
    figure.subplots_adjust(bottom=0.14, hspace=0.28)
    path = _common.save(figure, "thermal_rise.png")
    plt.close(figure)
    print(f"wrote {path}")
    print(
        f"  {trace.label}: tau {fit.tau_s:.2f} s, Tinf {fit.asymptote_c:.3f} degC, "
        f"rms residual {fit.rms_residual_c:.4f} degC, rho "
        f"{fit.residual_autocorrelation:+.3f}"
    )
    print(f"  equilibrium: {verdict.status}, slope {verdict.slope_c_per_s:+.6f} degC/s")
    print(f"  origin: {'MEASURED' if is_measured else 'SYNTHETIC'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
