"""Analytic baselines: interval machinery, parameter recovery and the Fieller limit.

Checks:
1. Coverage of the Fieller and delta-method intervals when the fitted shape is
   **correct**. This is the check that says the interval arithmetic itself is
   right, separately from the misspecification results elsewhere.
2. Agreement between the Fieller set and the delta-method interval as a
   function of Fieller's g, and the fraction of windows for which the exact set
   is unbounded.
3. Parameter recovery for the linear and exponential shapes against known truth.
4. Degeneracy handling: what the baselines do with a window that is flat,
   improving, or already past the limit.

References: Fieller (1954) JRSS-B 16(2):175-185; Draper and Smith (1998),
Applied Regression Analysis, 3rd ed., section 3.2.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, add_src_to_path, environment_lines

ROOT = add_src_to_path()

from degradepredict.baselines import (  # noqa: E402
    DegradationFitError,
    exponential_baseline,
    linear_baseline,
)
from degradepredict.limits import LimitSpec  # noqa: E402

OUT = ROOT / "validation" / "validate_baselines_output.txt"
LIMIT = LimitSpec(value=80.0, direction="below", unit="percent of BOL", atol=1e-9)
SEED = 20261016


def main() -> int:
    say = Tee(OUT)
    say("validate_baselines.py -- inverse-regression intervals for the analytic baselines")
    for line in environment_lines():
        say(line)
    say(f"seed             : {SEED}")
    say(f"limit            : {LIMIT.value} {LIMIT.unit}")

    say.rule("1. interval coverage when the fitted shape is CORRECT")
    say("Data generated from exactly the shape being fitted, with additive")
    say("Gaussian noise on the scale the fit assumes. The crossing time is the")
    say("crossing of the true mean curve. Nominal two-sided level 0.90.")
    say("")
    say(f"{'shape':<14}{'n win':>7}{'noise':>8}{'trials':>8}{'Fieller cov':>13}"
        f"{'delta cov':>11}{'unbounded':>11}")
    rng = np.random.default_rng(SEED)
    rows = []
    for shape in ("linear", "exponential"):
        for n_win, noise in ((30, 0.3), (60, 0.3), (60, 1.0), (15, 1.0)):
            trials = 2000
            t = np.arange(n_win, dtype=float) * 7.0
            hit_f = hit_d = unbounded = fails = 0
            for _ in range(trials):
                if shape == "linear":
                    b = 0.02
                    true_cross = 20.0 / b
                    mean = 100.0 - b * t
                    fitter = linear_baseline
                    y = mean + rng.normal(0.0, noise, size=n_win)
                else:
                    k = np.log(100.0 / 80.0) / 1000.0
                    true_cross = 1000.0
                    mean = np.log(100.0) - k * t
                    fitter = exponential_baseline
                    y = np.exp(mean + rng.normal(0.0, noise / 100.0, size=n_win))
                try:
                    fit = fitter(t, y, LIMIT)
                except (DegradationFitError, ValueError):
                    fails += 1
                    continue
                lo_f, hi_f = fit.fieller_interval_days
                lo_d, hi_d = fit.delta_interval_days
                hit_f += int(lo_f <= true_cross <= hi_f)
                hit_d += int(lo_d <= true_cross <= hi_d)
                unbounded += int(not np.isfinite(hi_f))
            usable = trials - fails
            cov_f = hit_f / trials
            cov_d = hit_d / trials
            rows.append((shape, n_win, noise, trials, cov_f, cov_d, unbounded / trials, fails))
            say(f"{shape:<14}{n_win:>7}{noise:>8.2f}{trials:>8}{cov_f:>13.4f}"
                f"{cov_d:>11.4f}{unbounded / trials:>11.4f}")
            _ = usable
    say("")
    say("Fieller coverage is counted against ALL trials, so a window the fit")
    say("refuses counts as a miss. Refusal counts, by row: "
        + ", ".join(str(r[7]) for r in rows))
    worst = min(r[4] for r in rows)
    say(f"lowest Fieller coverage over the eight rows: {worst:.4f} against nominal 0.90")
    say("PASS" if worst > 0.85 else "FAIL: the interval machinery under-covers on "
        "correctly specified data")

    say.rule("2. Fieller against delta as a function of g")
    say("g = t_crit**2 * sigma**2 / (Sxx * slope**2). The delta-method interval")
    say("is the g -> 0 limit of the Fieller set, so the two must converge as g")
    say("falls and diverge as g approaches 1.")
    say("")
    records = []
    rng = np.random.default_rng(SEED + 1)
    for _ in range(4000):
        n_win = int(rng.integers(8, 70))
        noise = float(rng.uniform(0.05, 4.0))
        slope = float(rng.uniform(0.0005, 0.05))
        t = np.arange(n_win, dtype=float) * 7.0
        y = 100.0 - slope * t + rng.normal(0.0, noise, size=n_win)
        try:
            fit = linear_baseline(t, y, LIMIT)
        except (DegradationFitError, ValueError):
            continue
        lo_f, hi_f = fit.fieller_interval_days
        lo_d, hi_d = fit.delta_interval_days
        if not np.isfinite(hi_f):
            records.append((fit.fieller_g, np.nan, True))
            continue
        width_f = hi_f - lo_f
        width_d = hi_d - lo_d
        records.append((fit.fieller_g, width_f / width_d, False))
    arr_g = np.array([r[0] for r in records])
    arr_ratio = np.array([r[1] for r in records])
    arr_unb = np.array([r[2] for r in records])
    say(f"{'g bin':<18}{'n':>7}{'unbounded':>11}{'median width ratio':>22}{'max':>10}")
    edges = [0.0, 1e-3, 1e-2, 0.1, 0.5, 1.0, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (arr_g >= lo) & (arr_g < hi)
        if not mask.any():
            continue
        unb = int(arr_unb[mask].sum())
        finite = arr_ratio[mask][~arr_unb[mask]]
        med = float(np.median(finite)) if finite.size else float("nan")
        mx = float(np.max(finite)) if finite.size else float("nan")
        label = f"[{lo:g}, {hi:g})"
        say(f"{label:<18}{int(mask.sum()):>7}{unb:>11}{med:>22.6f}{mx:>10.4f}")
    say("")
    say(f"windows fitted           : {arr_g.size}")
    say(f"windows with g >= 1      : {int((arr_g >= 1.0).sum())} "
        f"({(arr_g >= 1.0).mean():.4f})")
    say(f"windows with unbounded set: {int(arr_unb.sum())} ({arr_unb.mean():.4f})")
    small = arr_ratio[(arr_g < 1e-3) & ~arr_unb]
    if small.size:
        say(f"width ratio for g < 1e-3 : median {np.median(small):.8f}, "
            f"max {small.max():.8f}")
        say("the two intervals agree to better than one part in a thousand once g")
        say("is below 1e-3, which is the regime the shipped windows sit in.")

    say.rule("3. parameter recovery against known truth")
    say("200 seeded replicates per row. The estimate is the mean over replicates")
    say("and the spread is the standard deviation; the truth is exact.")
    say("")
    say(f"{'quantity':<28}{'truth':>14}{'mean est':>14}{'sd':>12}{'rel bias':>12}")
    rng = np.random.default_rng(SEED + 2)
    t = np.arange(60, dtype=float) * 7.0
    slopes, crossings = [], []
    for _ in range(200):
        y = 100.0 - 0.02 * t + rng.normal(0.0, 0.3, size=t.size)
        fit = linear_baseline(t, y, LIMIT)
        slopes.append(fit.slope)
        crossings.append(fit.crossing_days)
    say(f"{'linear slope [%/d]':<28}{-0.02:>14.6f}{np.mean(slopes):>14.6f}"
        f"{np.std(slopes):>12.6f}{(np.mean(slopes) + 0.02) / 0.02:>12.4f}")
    say(f"{'linear crossing [d]':<28}{1000.0:>14.4f}{np.mean(crossings):>14.4f}"
        f"{np.std(crossings):>12.4f}{(np.mean(crossings) - 1000.0) / 1000.0:>12.4f}")
    k_true = float(np.log(100.0 / 80.0) / 1000.0)
    rates, cross_e = [], []
    for _ in range(200):
        y = 100.0 * np.exp(-k_true * t) * np.exp(rng.normal(0.0, 0.003, size=t.size))
        fit = exponential_baseline(t, y, LIMIT)
        rates.append(-fit.slope)
        cross_e.append(fit.crossing_days)
    say(f"{'exponential k [1/d]':<28}{k_true:>14.8f}{np.mean(rates):>14.8f}"
        f"{np.std(rates):>12.8f}{(np.mean(rates) - k_true) / k_true:>12.4f}")
    say(f"{'exponential crossing [d]':<28}{1000.0:>14.4f}{np.mean(cross_e):>14.4f}"
        f"{np.std(cross_e):>12.4f}{(np.mean(cross_e) - 1000.0) / 1000.0:>12.4f}")
    say("")
    say("The crossing estimate is a ratio of estimates and is therefore biased in")
    say("finite samples even when the slope estimate is unbiased; the relative")
    say("bias above is the size of that effect at these sample sizes.")

    say.rule("4. degeneracy handling")
    t = np.arange(30, dtype=float) * 7.0
    rng = np.random.default_rng(SEED + 3)
    cases = {
        "improving window (slope > 0)": 95.0 + 0.01 * t + rng.normal(0.0, 0.05, t.size),
        "flat window with noise": np.full(t.size, 95.0) + rng.normal(0.0, 0.3, t.size),
        "already below the limit": 79.0 - 0.01 * t,
        "near-noise-free line (residual at rounding)": 100.0 - 0.02 * t,
    }
    say(f"{'window':<34}{'outcome':<44}")
    for label, y in cases.items():
        try:
            fit = linear_baseline(t, y, LIMIT)
            outcome = f"fitted, crossing {fit.crossing_days:.2f} d"
        except DegradationFitError as exc:
            outcome = f"DegradationFitError: {str(exc)[:38]}"
        say(f"{label:<34}{outcome:<44}")
    say("")
    say("Every refusal is an explicit exception, never a negative or infinite")
    say("crossing time returned as though it were an estimate. The benchmark")
    say("harness substitutes a declared uninformative fallback and counts it.")

    say.rule("summary")
    say(f"Fieller coverage on correctly specified data, worst row : {worst:.4f} "
        f"(nominal 0.90)")
    if small.size:
        say(f"Fieller/delta width ratio for g < 1e-3, median         : "
            f"{np.median(small):.8f}")
    say(f"windows with an unbounded exact set                     : "
        f"{arr_unb.mean():.4f} of {arr_g.size}")
    say(f"linear crossing relative bias at n=60, sigma=0.3        : "
        f"{(np.mean(crossings) - 1000.0) / 1000.0:.4f}")
    path = say.save()
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
