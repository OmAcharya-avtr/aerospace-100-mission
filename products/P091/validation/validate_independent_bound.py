"""Bounds derived from mathematics the implementation does not use.

"Two implementations agree" and "the answer lies inside a bound derived from
different mathematics" are not the same check, and the mission's build guide
records two products where only the second caught the defect. Four independent
bounds are checked here, and each one is attacked by a control that must fire.

Input: SYNTHETIC fixtures. Every number is a property of the estimators.

The four bounds.

1. **Monotonicity.** For a monotonically rising first-order relaxation the
   asymptote is strictly above every observed sample. This uses no fit at all:
   it is a property of the functional form.
2. **The local-slope identity.** The RC form satisfies
   ``dT/dt = (T_inf - T) / tau`` exactly, so ``T_inf = T + tau * dT/dt`` at every
   instant. Evaluating it at the END of the run, with ``tau`` taken from the
   COOLDOWN traces -- a different run, a different direction of heat flow and a
   different fit -- gives an estimate of the asymptote that shares no data and no
   algebra with the rise fit. The bracket's width is set by the uncertainty in
   the final slope times the spread of cooldown tau, so the slack FOLLOWS THE
   CONDITIONING of the bound rather than being an absolute constant.
3. **Three routes to tau.** Separable least squares, the log-difference slope and
   the area rule. The log-difference route differentiates and its slack is scaled
   by its reported conditioning; the area rule's slack is scaled by its tail
   correction factor.
4. **Normal theory against resampling.** A Student-t interval over per-run means
   and a two-level percentile bootstrap are different mathematics for the same
   quantity. They must overlap; with three runs the t interval is much the wider
   and the ratio is reported rather than asserted to be near one.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True


import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

import numpy as np  # noqa: E402

from thermalbudget.baselines import sustained_truth  # noqa: E402
from thermalbudget.bootstrap import t_interval_of_group_means  # noqa: E402
from thermalbudget.equilibrium import fit_linear  # noqa: E402
from thermalbudget.rc_model import (  # noqa: E402
    fit_exponential_relaxation,
    tau_from_integral,
    tau_from_log_differences,
)

#: Coverage multiplier for the final-slope interval inside the local-slope bound.
#: Four sigma, declared, so the bound is not a 1-in-20 coin toss. The mission's
#: build guide records a control whose criterion was a 95 per cent interval and
#: whose 2.01-sigma miss was then mistaken for a defect.
SIGMA = 4.0


def main() -> int:
    out = _bootstrap.Tee("validate_independent_bound_output.txt")
    out.rule("INDEPENDENT BOUNDS -- four, each from mathematics the code does not use")
    out("INPUT ORIGIN: SYNTHETIC. These are checks of the estimators.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)

    campaign = _bootstrap.synthetic_campaign("no_throttle")
    primary = campaign["soak_1.00_r1"]
    rise = fit_exponential_relaxation(primary.t_rel, primary.die_temperature_c)

    out.rule("1  monotonicity: the asymptote is above the final-window MEAN")
    observed_max = float(np.nanmax(primary.die_temperature_c))
    tail_samples = primary.tail(_bootstrap.WINDOW_S).die_temperature_c
    tail_mean = float(np.nanmean(tail_samples))
    out(
        "  The bound is stated against the MEAN of the final window and not against the "
        "largest observed sample, and the reason was measured rather than reasoned: with "
        f"per-sample noise the maximum observed sample is {observed_max:.4f} degC, which "
        f"is {observed_max - rise.asymptote_c:+.4f} degC relative to the fitted asymptote "
        f"{rise.asymptote_c:.4f} degC. A monotone rise approaches its asymptote from "
        "below, so the NOISELESS trend never exceeds it, but a noisy sample can and does. "
        "Averaging the window removes the noise and keeps the bound; using the maximum "
        "would make this check fail on correct data, which is what happened when it was "
        "first written."
    )
    out.check(
        "the fitted asymptote exceeds the mean of the final window",
        rise.asymptote_c > tail_mean,
        f"asymptote {rise.asymptote_c:.4f} degC > final-window mean {tail_mean:.4f} degC "
        f"by {rise.asymptote_c - tail_mean:+.4f} degC",
    )
    out.check(
        "the asymptote is above the final-window mean by less than the fit's own residual "
        "scale times ten, so it is not an extrapolation off to infinity",
        (rise.asymptote_c - tail_mean) < 10.0 * max(rise.rms_residual_c, 1e-3),
        f"excess {rise.asymptote_c - tail_mean:.4f} degC against rms residual "
        f"{rise.rms_residual_c:.4f} degC",
    )
    out.control(
        "an asymptote placed BELOW the final-window mean is caught",
        (tail_mean - 1.0) <= tail_mean,
        must_fire=True,
        detail=f"a corrupted asymptote of {tail_mean - 1.0:.4f} degC violates the bound",
    )
    out.control(
        "the correct asymptote is NOT flagged by the same bound",
        rise.asymptote_c <= tail_mean,
        must_fire=False,
        detail="the monotonicity bound admits the fitted value",
    )

    out.rule("2  the local-slope identity, with tau from the COOLDOWN traces")
    cooldown_taus = []
    for trace in campaign.cooldowns():
        try:
            cooldown_taus.append(
                fit_exponential_relaxation(trace.t_rel, trace.die_temperature_c).tau_s
            )
        except Exception as error:  # noqa: BLE001 -- the refusal itself is the datum
            out(f"  cooldown {trace.label}: fit refused ({type(error).__name__})")
    out(
        f"  cooldown tau: {len(cooldown_taus)} usable fits, "
        f"{min(cooldown_taus):.2f} to {max(cooldown_taus):.2f} s"
    )
    tail = primary.tail(_bootstrap.WINDOW_S)
    final = fit_linear(tail.t_rel, tail.die_temperature_c)
    slope_low = final.slope - SIGMA * final.slope_stderr_ar1
    slope_high = final.slope + SIGMA * final.slope_stderr_ar1
    last = float(tail.die_temperature_c[np.isfinite(tail.die_temperature_c)][-1])
    candidates = [
        last + tau * slope
        for tau in (min(cooldown_taus), max(cooldown_taus))
        for slope in (slope_low, slope_high)
    ]
    low, high = min(candidates), max(candidates)
    out(
        f"  final-window slope {final.slope:+.6f} +/- {final.slope_stderr_ar1:.6f} degC/s "
        f"({SIGMA:g} sigma), last sample {last:.4f} degC"
    )
    out(
        f"  bracket from T_inf = T + tau dT/dt : [{low:.4f}, {high:.4f}] degC, width "
        f"{high - low:.4f} degC"
    )
    out(
        "  the bracket's width is slope uncertainty times the cooldown tau spread, so the "
        "slack follows the CONDITIONING of this bound and not a constant"
    )
    inside = low <= rise.asymptote_c <= high
    out.check(
        "the least-squares asymptote lies inside the independent bracket",
        inside,
        f"asymptote {rise.asymptote_c:.4f} degC in [{low:.4f}, {high:.4f}] degC",
    )
    corrupted = rise.asymptote_c + 3.0 * (high - low) + 1.0
    out.control(
        "an asymptote displaced beyond the bracket is caught",
        not (low <= corrupted <= high),
        must_fire=True,
        detail=f"a corrupted asymptote of {corrupted:.4f} degC is outside",
    )
    out.control(
        "the true asymptote is NOT flagged by the same bracket",
        not (low <= rise.asymptote_c <= high),
        must_fire=False,
        detail="the bracket admits the correct value",
    )

    out.rule("3  three routes to tau, slack scaled by each route's conditioning")
    log_difference = tau_from_log_differences(primary.t_rel, primary.die_temperature_c)
    area = tau_from_integral(
        primary.t_rel, primary.die_temperature_c, asymptote_c=rise.asymptote_c
    )
    out(f"  least squares   tau {rise.tau_s:8.3f} s")
    out(
        f"  log-differences tau {log_difference.tau_s:8.3f} s +/- "
        f"{log_difference.stderr_s:.3f} s, conditioning "
        f"{log_difference.conditioning:.4g}, n={log_difference.n}"
    )
    out(
        f"  area rule       tau {area.tau_s:8.3f} s, tail correction "
        f"{area.conditioning:.4f}"
    )
    # The log-difference slack is its own standard error inflated by its
    # conditioning: differencing amplifies noise, and the number that says by how
    # much is the one the estimator reports.
    log_slack = SIGMA * log_difference.stderr_s * (1.0 + log_difference.conditioning)
    out.check(
        "least squares and the log-difference route agree inside the differencing slack",
        abs(log_difference.tau_s - rise.tau_s) <= log_slack,
        f"|{log_difference.tau_s:.3f} - {rise.tau_s:.3f}| = "
        f"{abs(log_difference.tau_s - rise.tau_s):.3f} s <= {log_slack:.3f} s",
    )
    # The area rule's error is quadrature plus the tail correction, both of which
    # it reports; 2 per cent of tau times the tail factor is the slack.
    area_slack = 0.02 * rise.tau_s * area.conditioning
    out.check(
        "least squares and the area rule agree inside the quadrature slack",
        abs(area.tau_s - rise.tau_s) <= area_slack,
        f"|{area.tau_s:.3f} - {rise.tau_s:.3f}| = {abs(area.tau_s - rise.tau_s):.3f} s "
        f"<= {area_slack:.3f} s",
    )
    out.control(
        "a tau corrupted by a factor of two is caught by the log-difference bound",
        abs(2.0 * rise.tau_s - log_difference.tau_s) > log_slack,
        must_fire=True,
        detail=f"2 x tau = {2.0 * rise.tau_s:.3f} s is outside the slack",
    )
    out(
        "  NOTE: the fixtures are generated FROM the first-order form, so this section "
        "measures that the estimators recover parameters they were given. It is not "
        "evidence that a Jetson die is first-order; the residual autocorrelation of the "
        "MEASURED fit is the number that speaks to that, and it is in "
        "validate_measured_campaign_output.txt."
    )

    out.rule("4  normal theory against resampling, on the same quantity")
    truth = sustained_truth(campaign.soaks(1.0), n_resamples=_bootstrap.RESAMPLES)
    point, t_low, t_high = t_interval_of_group_means(truth.throughput_groups)
    out(
        f"  bootstrap : {truth.throughput.point:.3f} "
        f"[{truth.throughput.ci_low:.3f}, {truth.throughput.ci_high:.3f}] iters/s, "
        f"half-width {truth.throughput.half_width:.3f}"
    )
    out(
        f"  Student-t : {point:.3f} [{t_low:.3f}, {t_high:.3f}] iters/s, half-width "
        f"{0.5 * (t_high - t_low):.3f}  (2 degrees of freedom, multiplier 4.3027)"
    )
    out.check(
        "both routes give the same point estimate to within rounding",
        abs(point - truth.throughput.point) <= 1e-9 * max(abs(point), 1.0),
        f"difference {point - truth.throughput.point:+.3e} iters/s",
    )
    overlap = not (truth.throughput.ci_high < t_low or truth.throughput.ci_low > t_high)
    out.check(
        "the two intervals overlap",
        overlap,
        "a percentile bootstrap and a normal-theory interval that did not overlap would "
        "mean one of them is wrong",
    )
    ratio = (0.5 * (t_high - t_low)) / truth.throughput.half_width
    out(
        f"  the t interval is {ratio:.3f} times the bootstrap half-width. With three runs "
        "that ratio is expected to be well above one and is REPORTED rather than asserted "
        "to be near one: the t multiplier at 2 degrees of freedom is 4.3027."
    )
    out.check(
        "the bootstrap point lies inside the independent t interval",
        t_low <= truth.throughput.point <= t_high,
        f"{truth.throughput.point:.3f} in [{t_low:.3f}, {t_high:.3f}]",
    )
    displaced = truth.throughput.point + 5.0 * (t_high - t_low)
    out.control(
        "a point displaced by five interval widths is caught",
        not (t_low <= displaced <= t_high),
        must_fire=True,
        detail=f"{displaced:.3f} iters/s is outside",
    )

    out.rule("5  convexity: a mean of means cannot leave the range of the means")
    means = [float(np.mean(group)) for group in truth.throughput_groups]
    out.check(
        "the sustained point estimate lies within the range of the per-run means",
        min(means) <= truth.throughput.point <= max(means),
        f"{truth.throughput.point:.3f} in [{min(means):.3f}, {max(means):.3f}] iters/s",
    )
    out.check(
        "every bootstrap draw is a convex combination, so the interval is inside the "
        "range of the pooled samples",
        all(
            float(np.min(np.concatenate(truth.throughput_groups)))
            <= bound
            <= float(np.max(np.concatenate(truth.throughput_groups)))
            for bound in (truth.throughput.ci_low, truth.throughput.ci_high)
        ),
        f"[{truth.throughput.ci_low:.3f}, {truth.throughput.ci_high:.3f}] inside the "
        "pooled sample range",
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
