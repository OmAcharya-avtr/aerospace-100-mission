"""The README worked example, run so that the printed output in the README is real.

Nineteen lines of public API: generate (or supply) an amplitude series, get the
fade statistics with their definitions, test the memoryless assumption, fit the
channel-state model, and forecast an outage with an uncertainty.

Runtime: about 10 s on one core.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from linkoutage import (
    AnalyticLcrPredictor,
    ConstantRatePredictor,
    RandomForestOutageClassifier,
    build_outage_dataset,
    compare_fade_duration_models,
    dwell_time_goodness_of_fit,
    evaluate_forecast,
    fade_durations,
    fade_statistics,
    fit_markov,
    lognormal_amplitude_series,
    state_sequence,
)

print("=" * 78)
print("worked_example.py -- the README example, actually run")
print("=" * 78)

# 1. A supplied amplitude series would go here instead; this one is generated.
series = lognormal_amplitude_series(
    1_000_000, fs_hz=1.0e6, tau_s=2.0e-4, si=0.6, seed=2026
)
amplitude = series.amplitude  # or np.load("my_receiver_log.npy")

# 2. Fade statistics at a 0.6 amplitude threshold, with the definitions stated.
stats = fade_statistics(amplitude, threshold=0.6, fs_hz=1.0e6)
print()
print("# 1. Fade statistics, and the definitions that produced them")
print(f"level-crossing rate : {stats.level_crossing_rate_hz:.3f} Hz")
print(f"mean fade duration  : {stats.mean_fade_duration_s:.4e} s")
print(f"outage fraction     : {stats.outage_fraction:.6f}")
print(f"availability        : {stats.availability:.6f}")
print(f"complete fades      : {stats.n_complete_fades}, censored "
      f"{stats.n_left_censored + stats.n_right_censored}")
print(f"single-sample fades : {stats.n_single_sample_fades}")

# 3. Is the fade-duration distribution exponential? Test it.
complete, censored = fade_durations(amplitude, 0.6, 1.0e6)
fits = compare_fade_duration_models(complete, fs_hz=1.0e6, censored_s=censored)
print()
print("# 2. Is the fade-duration distribution memoryless? Test, do not assume.")
print(f"geometric chi-square : {fits.geometric.statistic:.1f} on "
      f"{fits.geometric.dof} dof, p = {fits.geometric.p_value:.3g}")
print(f"memoryless rejected  : {fits.exponential_rejected}")
print(f"coefficient of variation {fits.extra['coefficient_of_variation']:.3f} "
      "against 1.0 for an exponential")
print(f"best fit by AIC      : {fits.best_by_aic.name}")

# 4. Two-state channel model, with the goodness of fit, not without it.
states = state_sequence(amplitude, [0.6])
markov = fit_markov(states)
gof = dwell_time_goodness_of_fit(states, markov)
print()
print("# 3. Two-state Markov channel, with its goodness of fit")
print(f"P[fade -> good]   : {markov.transition_matrix[0, 1]:.6f}")
print(f"P[good -> fade]   : {markov.transition_matrix[1, 0]:.6f}")
print(f"mean dwell, model : {markov.mean_dwell_samples.round(3).tolist()} samples")
print(f"mean dwell, data  : {markov.empirical_mean_dwell_samples.round(3).tolist()} samples")
for g in gof:
    print(f"state {g.state}: dwell chi2 {g.statistic:.1f} on {g.dof} dof, "
          f"cv measured {g.cv_observed:.3f} against geometric {g.cv_geometric:.3f}")

# 5. Short-horizon outage forecast: baselines first, then the forest.
data = build_outage_dataset(
    amplitude,
    threshold=0.35,
    window_samples=400,
    horizon_samples=200,
    stride_samples=50,
)
train, test = data.split.train, data.split.test
baseline = ConstantRatePredictor().fit(data.y[train])
physics = AnalyticLcrPredictor(threshold=0.35, horizon_samples=200).fit(
    amplitude[: int(data.index[train[-1]]) + 1]
)
forest = RandomForestOutageClassifier(n_estimators=100, min_samples_leaf=5).fit(
    data.x[train], data.y[train]
)
print()
print("# 4. Short-horizon outage forecast, baselines first")
print(f"base rate on the test split : {float(data.y[test].mean()):.5f}")
for model in (baseline, physics, forest):
    report = evaluate_forecast(
        data.y[test],
        model.predict_proba_onset(data.x[test]),
        name=model.name,
        reference_rate=baseline.rate,
    )
    print(
        f"{report.name:<22s} Brier {report.brier:.6f}  skill {report.brier_skill_score:+.4f}  "
        f"ECE {report.ece:.5f}  AUC {report.roc_auc:.4f}"
    )

# 6. The forecast comes with an uncertainty, not just a point estimate.
probability, spread = forest.predict_with_uncertainty(data.x[test][:5])
print()
print("# 5. The forest's forecast carries an ensemble spread")
for p, s in zip(probability, spread, strict=True):
    print(f"P(outage within 0.2 ms) = {p:.4f} +/- {s:.4f} (spread across 100 trees)")
_, all_spread = forest.predict_with_uncertainty(data.x[test])
print()
print(f"max spread over the test split: {float(all_spread.max()):.4f}")
print()
print("Research-grade. Not flight-qualified, not certified, not approved for")
print("operational aerospace use.")
print()
print("done")
