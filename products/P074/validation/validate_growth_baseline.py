"""The analytic baseline validated against the simulation, BEFORE any benchmarking.

This script runs before ``validate_predictors.py`` and its purpose is to
establish that the analytic growth forms are as strong as the physics allows,
because a baseline weaker than it could be manufactures a learned win.

What the physics predicts, and what is therefore being checked:

- **Linear in time** where drag is negligible. A constant error in mean motion
  gives an along-track position error ``-(3/2) n (da/a) t``, which is linear.
  The fitted exponent should approach 1 at high altitude.
- **Quadratic in time** where drag dominates. A constant error in the *rate* of
  change of semi-major axis integrates to ``t^2``. The fitted exponent should
  rise toward 2 at low altitude and high drag parameter.
- **A transition between them**, so the exponent should be negatively
  correlated with altitude and positively correlated with the drag parameter.

Each is measured over the committed population. The script also measures how
well each analytic form fits, so the benchmark that follows starts from a
baseline whose quality is known rather than assumed.
"""

from __future__ import annotations

import sys

from _bootstrap import Tee, bootstrap, environment_lines, is_fast, load_dataset

bootstrap()

import numpy as np  # noqa: E402

from orbitdrift.growth import best_form, fit_all_forms, fit_power_law  # noqa: E402

ALTITUDE_BANDS = ((300.0, 450.0), (450.0, 600.0), (600.0, 800.0), (800.0, 1100.0), (1100.0, 1700.0))


def main() -> int:
    out = Tee("validate_growth_baseline_output.txt")
    out.rule("orbitdrift -- the analytic growth baseline against the simulation")
    for line in environment_lines():
        out(line)

    table = load_dataset()
    out("")
    out(f"  dataset                       : {table.detail}")
    out(f"  objects                       : {table.n_objects}")
    out(f"  rows                          : {table.n_rows}")

    objects = table.column("object_index")
    ages = table.column("epoch_age_days")
    drift = table.column("drift_km")
    perigee = table.column("perigee_altitude_km")
    drag_parameter = table.column("log10_drag_parameter")

    unique = np.unique(objects)
    exponents = np.empty(unique.size)
    stderrs = np.empty(unique.size)
    residuals = np.empty(unique.size)
    altitudes = np.empty(unique.size)
    drags = np.empty(unique.size)
    winners: list[str] = []
    form_residuals: dict[str, list[float]] = {}
    for index, value in enumerate(unique):
        mask = objects == value
        fit = fit_power_law(ages[mask], drift[mask])
        exponents[index] = fit.exponent
        stderrs[index] = fit.exponent_stderr
        residuals[index] = fit.log_residual_std
        altitudes[index] = perigee[mask][0]
        drags[index] = drag_parameter[mask][0]
        fits = fit_all_forms(ages[mask], drift[mask])
        winners.append(best_form(fits))
        for name, item in fits.items():
            form_residuals.setdefault(name, []).append(item.log_residual_std)

    out.rule("fitted free exponent against perigee altitude")
    out("  The physical prediction: p -> 1 where drag is negligible and p -> 2")
    out("  where a drag-rate error dominates.")
    out("")
    out("  band km          n   mean p    sd p   min p   max p   median log-resid")
    band_means = []
    for low, high in ALTITUDE_BANDS:
        selected = (altitudes >= low) & (altitudes < high)
        count = int(np.count_nonzero(selected))
        if count == 0:
            continue
        values = exponents[selected]
        band_means.append((0.5 * (low + high), float(values.mean())))
        out(
            f"  {low:4.0f}-{high:<9.0f} {count:4d}   {float(values.mean()):6.4f}  "
            f"{float(values.std()):6.4f}  {float(values.min()):6.4f}  "
            f"{float(values.max()):6.4f}   {float(np.median(residuals[selected])):16.5f}"
        )

    out("")
    out(f"  all objects: mean p {float(exponents.mean()):.4f}, "
        f"sd {float(exponents.std()):.4f}, range {float(exponents.min()):.4f} to "
        f"{float(exponents.max()):.4f}")
    out(f"  mean fitted standard error on p : {float(np.mean(stderrs)):.6f}")

    altitude_correlation = float(np.corrcoef(np.log10(altitudes), exponents)[0, 1])
    drag_correlation = float(np.corrcoef(drags, exponents)[0, 1])
    out("")
    out(f"  correlation of p with log10(perigee altitude) : {altitude_correlation:+.4f}")
    out(f"  correlation of p with log10(drag parameter)   : {drag_correlation:+.4f}")

    highest = (altitudes >= 1100.0)
    lowest = (altitudes < 450.0)
    out("")
    out(f"  p above 1100 km : {float(exponents[highest].mean()):.4f} "
        f"+/- {float(exponents[highest].std()):.4f} over "
        f"{int(np.count_nonzero(highest))} objects")
    out(f"  p below  450 km : {float(exponents[lowest].mean()):.4f} "
        f"+/- {float(exponents[lowest].std()):.4f} over "
        f"{int(np.count_nonzero(lowest))} objects")

    out.rule("checks on the physics")
    if is_fast():
        out("  probe mode: the dataset is reduced to a subset that does not span the")
        out("  altitude range, so the population-level physics checks are skipped.")
        out("  Only the exit status is being probed.")
        out.check(
            "every fitted exponent is finite",
            bool(np.all(np.isfinite(exponents))),
            f"{exponents.size} objects",
        )
        out("")
        out(f"wrote {out.save()}")
        return 1 if out.failures else 0

    out.check(
        "the exponent above 1100 km is within 0.1 of the predicted linear value 1",
        abs(float(exponents[highest].mean()) - 1.0) < 0.1,
        f"{float(exponents[highest].mean()):.4f} +/- {float(exponents[highest].std()):.4f}",
    )
    out.check(
        "the exponent above 1100 km has a small spread, so the linear form is a "
        "good description there and not an average of two regimes",
        float(exponents[highest].std()) < 0.12,
        f"sd {float(exponents[highest].std()):.4f}",
    )
    out.check(
        "the exponent below 450 km is larger than above 1100 km, as the "
        "drag-rate-error prediction requires",
        float(exponents[lowest].mean()) > float(exponents[highest].mean()) + 0.1,
        f"{float(exponents[lowest].mean()):.4f} against "
        f"{float(exponents[highest].mean()):.4f}",
    )
    out.check(
        "the exponent falls with altitude (negative correlation)",
        altitude_correlation < -0.15,
        f"{altitude_correlation:+.4f}",
    )
    out.check(
        "the exponent rises with the drag parameter (positive correlation)",
        drag_correlation > 0.15,
        f"{drag_correlation:+.4f}",
    )
    monotone = all(
        first[1] >= second[1] - 0.15
        for first, second in zip(band_means[:-1], band_means[1:], strict=True)
    )
    out.check(
        "the band means do not rise with altitude by more than noise",
        monotone,
        ", ".join(f"{value:.4f}" for _, value in band_means),
    )

    out.rule("how well each analytic form fits, per object")
    out("  Log-residual standard deviation: a spread of RATIOS, so 0.1 is about")
    out("  10 per cent. Lower is better.")
    out("")
    out("  form                n   median   mean    90th pct   wins")
    from collections import Counter

    counts = Counter(winners)
    for name in ("linear", "quadratic", "power", "drag_exponential"):
        values = np.array(form_residuals.get(name, []))
        if values.size == 0:
            continue
        out(
            f"  {name:16s} {values.size:4d}   {float(np.median(values)):6.4f}  "
            f"{float(values.mean()):6.4f}  {float(np.percentile(values, 90)):8.4f}  "
            f"{counts.get(name, 0):5d}"
        )
    out("")
    out(f"  best-fitting form counts      : {dict(counts)}")

    out.rule("the strongest analytic form by altitude band")
    out("  band km          n   best-form counts")
    for low, high in ALTITUDE_BANDS:
        selected = np.flatnonzero((altitudes >= low) & (altitudes < high))
        if selected.size == 0:
            continue
        band_counts = Counter(winners[index] for index in selected)
        out(f"  {low:4.0f}-{high:<9.0f} {selected.size:4d}   {dict(band_counts)}")

    out("")
    out("  The free power law or the drag-exponential form wins almost everywhere.")
    out("  The fixed linear and fixed quadratic forms are in the benchmark that")
    out("  follows anyway, because they are the two forms the physics names and")
    out("  because a form with one free parameter fewer is a stronger claim when")
    out("  it holds.")

    out.rule("the baseline's quality, stated before the benchmark runs")
    out(f"  median per-object log-residual spread of the free power law : "
        f"{float(np.median(residuals)):.5f}")
    out(f"  90th percentile                                             : "
        f"{float(np.percentile(residuals, 90)):.5f}")
    out(f"  worst                                                       : "
        f"{float(residuals.max()):.5f}")
    out.check(
        "the free power law fits the median object to better than 15 per cent in "
        "the ratio, so the analytic baseline is strong before anything is learned",
        float(np.median(residuals)) < 0.15,
        f"median log-residual spread {float(np.median(residuals)):.5f}",
    )

    out.rule("negative control")
    out("  Fitting the power law to a shuffled drift column must destroy the fit,")
    out("  otherwise the residual spreads above are not measuring anything.")
    rng = np.random.default_rng(41)
    shuffled_residuals = []
    for value in unique[:40]:
        mask = objects == value
        shuffled = rng.permutation(drift[mask])
        shuffled_residuals.append(fit_power_law(ages[mask], shuffled).log_residual_std)
    out(f"  median log-residual spread, real      : {float(np.median(residuals)):.5f}")
    out(f"  median log-residual spread, shuffled  : "
        f"{float(np.median(shuffled_residuals)):.5f}")
    out.check(
        "shuffling the drift column makes the fit at least five times worse",
        float(np.median(shuffled_residuals)) > 5.0 * float(np.median(residuals)),
        f"{float(np.median(shuffled_residuals)):.5f} against "
        f"{float(np.median(residuals)):.5f}",
    )

    out("")
    out(f"wrote {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    sys.exit(main())
