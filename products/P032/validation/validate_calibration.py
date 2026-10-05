"""Link-availability predictors: calibration comparison on identical splits.

Headline quantity
-----------------
CALIBRATION, not accuracy.  A link-availability probability is an input to a
scheduling decision, so a probability that is systematically wrong corrupts
the decision even when the thresholded classification happens to be right.
The reported headline is the reliability term of the Murphy (1973)
decomposition of the Brier score, with the Brier score itself and the expected
calibration error alongside, and accuracy shown last and explicitly demoted.

Predictors, in the order they were implemented
----------------------------------------------
1. climatology baseline -- training base rate of the row's stratum;
2. logistic-regression baseline -- L2 logistic regression on standardised
   features;
3. learned model -- bagged, Platt-calibrated histogram gradient boosting.

Both baselines were built and measured before the learned model, and the
result is reported whichever way it falls.

Protocol
--------
* one synthetic dataset (see ``DATASET_CARD.md``), fixed seed;
* GROUPED splits by link, so no link appears in both train and test; five
  split seeds, every predictor on identical splits;
* the row-wise split is additionally run on one seed to MEASURE the leakage
  the grouped split avoids -- not to report results on;
* bootstrap percentile intervals on the Brier score of each predictor;
* the Brier decomposition both over 10 equal-width bins (where the residual
  is the extra within-bin component of Stephenson, Coelho & Jolliffe 2008)
  and over distinct forecast values (where the three-term identity is exact);
* a reliability diagram with Wilson intervals written to
  ``validation/reliability_diagram.png``.

What this does NOT establish
----------------------------
The labels come from the capacity models this package ships, driven by a
latent weather and scintillation state.  Agreement here is self-consistency,
not agreement with measured link outages.  No claim about real link
availability follows from these numbers.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from constellink.availability import (  # noqa: E402
    ClimatologyBaseline,
    LinkAvailabilityModel,
    LogisticBaseline,
    PredictorScores,
    grouped_split,
    row_split,
    score_predictor,
)
from constellink.metrics import (  # noqa: E402
    bootstrap_ci,
    brier_decomposition,
    reliability_curve,
)
from constellink.synthdata import DatasetConfig, generate_dataset  # noqa: E402

N_BINS = 10
SPLIT_SEEDS = (0, 1, 2, 3, 4)
MODEL_SEED = 1
OUT_PNG = os.path.join(os.path.dirname(__file__), "reliability_diagram.png")


def fit_predict(data, train, test, method="sigmoid"):
    """Fit all three predictors on ``train`` and predict on ``test``."""
    clim = ClimatologyBaseline().fit(data.x[train], data.y[train],
                                     data.stratum[train])
    p_clim = clim.predict_proba(data.x[test], data.stratum[test])
    logi = LogisticBaseline().fit(data.x[train], data.y[train])
    p_logi = logi.predict_proba(data.x[test])
    model = LinkAvailabilityModel(seed=MODEL_SEED, method=method).fit(
        data.x[train], data.y[train])
    p_model, s_model = model.predict_with_uncertainty(data.x[test])
    return {
        "climatology": p_clim,
        "logistic": p_logi,
        "learned": p_model,
    }, s_model, clim.n_fallback_last_call


def main() -> int:
    cfg = DatasetConfig()
    data = generate_dataset(cfg)
    print("Link-availability predictor calibration")
    print("=" * 94)
    print(f"dataset       : {len(data)} contacts from a "
          f"{cfg.n_total}/{cfg.n_planes}/{cfg.phasing_f} Walker shell at "
          f"{cfg.altitude_km:g} km over {cfg.horizon_hours:g} h")
    print(f"dataset seed  : {cfg.seed}")
    print(f"base rate     : {data.base_rate:.6f}")
    print(f"ground rows   : {int(data.x[:, 8].sum())} of {len(data)} "
          f"(the rest are inter-satellite links, which carry no weather term)")
    print(f"distinct links: {len(np.unique(data.group))}")
    print(f"reliability bins: {N_BINS}")
    print("")
    print("HEADLINE IS CALIBRATION: the reliability term REL (lower is better).")
    print("Accuracy is reported last and is NOT the figure of merit.")
    print("")

    per_seed: dict[str, list[PredictorScores]] = {}
    print("Part 1 -- grouped split by link, five split seeds")
    for seed in SPLIT_SEEDS:
        train, test = grouped_split(data, test_fraction=0.3, seed=seed)
        preds, ens_std, n_fallback = fit_predict(data, train, test)
        print(f"  split seed {seed}: {train.size} train / {test.size} test, "
              f"test base rate {data.y[test].mean():.4f}, "
              f"unseen-stratum rows {n_fallback}")
        print("  " + PredictorScores.header())
        print("  " + "-" * len(PredictorScores.header()))
        for name, p in preds.items():
            s = score_predictor(name, p, data.y[test], n_bins=N_BINS)
            per_seed.setdefault(name, []).append(s)
            print("  " + s.format_row())
        print(f"  learned-model ensemble std: mean {ens_std.mean():.5f}, "
              f"max {ens_std.max():.5f}")
        print("")

    print("Part 2 -- across-seed summary (mean +- sample std over 5 seeds)")
    head = (f"  {'predictor':<14}{'Brier':>18}{'REL':>18}{'RES':>18}{'ECE':>18}"
            f"{'acc@0.5':>12}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    summary = {}
    for name, rows in per_seed.items():
        b = np.array([r.brier for r in rows])
        rel = np.array([r.reliability for r in rows])
        res = np.array([r.resolution for r in rows])
        ece = np.array([r.ece for r in rows])
        acc = np.array([r.accuracy_at_half for r in rows])
        summary[name] = (b.mean(), rel.mean(), res.mean(), ece.mean(), acc.mean())
        print(f"  {name:<14}{f'{b.mean():.5f}+-{b.std(ddof=1):.5f}':>18}"
              f"{f'{rel.mean():.5f}+-{rel.std(ddof=1):.5f}':>18}"
              f"{f'{res.mean():.5f}+-{res.std(ddof=1):.5f}':>18}"
              f"{f'{ece.mean():.5f}+-{ece.std(ddof=1):.5f}':>18}"
              f"{acc.mean():>12.4f}")
    print("")
    best_rel = min(summary, key=lambda k: summary[k][1])
    best_brier = min(summary, key=lambda k: summary[k][0])
    print(f"  BEST CALIBRATED (lowest mean REL) : {best_rel}")
    print(f"  BEST PROPER SCORE (lowest Brier)  : {best_brier}")
    if best_rel != "learned":
        print(f"  The learned model is NOT the best calibrated predictor here. "
              f"'{best_rel}' is.")
        print("  That result is kept and reported as measured.")
    print("")

    print("Part 3 -- Brier decomposition identity and bootstrap intervals "
          "(split seed 0)")
    train, test = grouped_split(data, test_fraction=0.3, seed=0)
    preds, _, _ = fit_predict(data, train, test)
    head = (f"  {'predictor':<14}{'bins':<10}{'Brier':>10}{'REL':>10}{'RES':>10}"
            f"{'UNC':>10}{'residual':>13}{'occupied':>10}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for name, p in preds.items():
        d = brier_decomposition(p, data.y[test], n_bins=N_BINS)
        print(f"  {name:<14}{'equal-10':<10}{d.brier:>10.5f}{d.reliability:>10.5f}"
              f"{d.resolution:>10.5f}{d.uncertainty:>10.5f}"
              f"{d.identity_residual:>13.2e}{d.n_occupied_bins:>10d}")
        e = brier_decomposition(p, data.y[test], by_distinct_value=True)
        print(f"  {'':<14}{'distinct':<10}{e.brier:>10.5f}{e.reliability:>10.5f}"
              f"{e.resolution:>10.5f}{e.uncertainty:>10.5f}"
              f"{e.identity_residual:>13.2e}{e.n_occupied_bins:>10d}")
    print("  equal-width bins: the residual is the within-bin component of")
    print("    Stephenson, Coelho & Jolliffe 2008 -- real, not rounding error.")
    print("  distinct-value bins: the three-term identity is exact and the")
    print("    residual is floating-point only. This is the exactness check.")
    print("  Note the degeneracy it exposes: for a continuous predictor every")
    print("    distinct-value bin holds one row, so REL collapses onto the")
    print("    Brier score and RES onto UNC. The distinct-value decomposition")
    print("    is an identity check, NOT a calibration measurement; the")
    print("    equal-width rows are the ones to read for calibration.")
    print("")
    print(f"  {'predictor':<14}{'Brier':>10}{'95% CI low':>13}{'95% CI high':>13}"
          f"{'n_boot':>8}")
    print("  " + "-" * 58)
    for name, p in preds.items():
        pt, lo, hi = bootstrap_ci(p, data.y[test], n_boot=2000, seed=7)
        print(f"  {name:<14}{pt:>10.5f}{lo:>13.5f}{hi:>13.5f}{2000:>8d}")
    print("  (percentile bootstrap over forecast/outcome pairs; the pairs are "
          "not")
    print("   independent within a link, so these intervals are optimistic)")
    print("")

    print("Part 4 -- leakage measurement: row-wise split vs grouped split")
    tr_r, te_r = row_split(data, test_fraction=0.3, seed=0)
    preds_row, _, _ = fit_predict(data, tr_r, te_r)
    head = (f"  {'predictor':<14}{'Brier grouped':>16}{'Brier row-wise':>17}"
            f"{'optimism':>12}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for name in preds:
        s_g = score_predictor(name, preds[name], data.y[test], n_bins=N_BINS)
        s_r = score_predictor(name, preds_row[name], data.y[te_r], n_bins=N_BINS)
        print(f"  {name:<14}{s_g.brier:>16.5f}{s_r.brier:>17.5f}"
              f"{s_g.brier - s_r.brier:>12.5f}")
    print("  (a positive optimism column means the row-wise split flatters the")
    print("   predictor; the grouped numbers are the ones reported everywhere "
          "else)")
    print("")

    print("Part 5 -- reliability diagram")
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 8.4),
                                  gridspec_kw={"height_ratios": [3, 1]},
                                  constrained_layout=True)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    colours = {"climatology": "tab:blue", "logistic": "tab:orange",
               "learned": "tab:green"}
    for name, p in preds.items():
        rc = reliability_curve(p, data.y[test], n_bins=N_BINS)
        m = rc.count > 0
        ax.errorbar(rc.mean_forecast[m], rc.observed_frequency[m],
                    yerr=[rc.observed_frequency[m] - rc.ci_low[m],
                          rc.ci_high[m] - rc.observed_frequency[m]],
                    marker="o", ms=4, lw=1.2, capsize=2,
                    color=colours[name], label=name)
        ax2.step(rc.bin_centre, rc.count, where="mid", color=colours[name],
                 label=name)
    ax.set_xlabel("mean forecast probability")
    ax.set_ylabel("observed frequency of a closed contact")
    ax.set_title("Reliability diagram, grouped split (seed 0), "
                 f"{N_BINS} equal-width bins\nerror bars: Wilson 95 % on the "
                 "observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax2.set_xlabel("forecast probability bin")
    ax2.set_ylabel("test rows per bin")
    ax2.set_yscale("symlog", linthresh=1)
    ax2.set_xlim(0, 1)
    ax2.grid(alpha=0.3)
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"  written: validation/{os.path.basename(OUT_PNG)}")
    print("")

    # Pass criteria: the decomposition identity must hold, every predictor must
    # produce probabilities in range, and every split must be usable. The
    # RANKING is a measurement, not a pass criterion -- a baseline winning is
    # a valid outcome, not a failure.
    ok = True
    for name, p in preds.items():
        if not (np.all(p >= 0.0) and np.all(p <= 1.0)):
            print(f"FAIL: {name} produced out-of-range probabilities")
            ok = False
        e = brier_decomposition(p, data.y[test], by_distinct_value=True)
        if abs(e.identity_residual) > 1e-12:
            print(f"FAIL: {name} distinct-value Brier identity residual "
                  f"{e.identity_residual:.3e} exceeds 1e-12")
            ok = False
    print("=" * 94)
    print("Pass criteria: the distinct-value Brier identity is exact to 1e-12 "
          "and every")
    print("probability is in [0, 1]. The RANKING of the predictors is a "
          "measurement,")
    print("not a pass criterion: a baseline winning is a valid outcome.")
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
