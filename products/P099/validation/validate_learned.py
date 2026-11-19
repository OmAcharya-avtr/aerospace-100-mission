"""The learned predictor against the three baselines, on two held-out splits.

THE ORDER THIS SCRIPT RUNS IN IS THE ORDER THE BUILD GUIDE REQUIRES
-------------------------------------------------------------------
Baselines 1, 2 and 3 are implemented and validated before the learned
component is benchmarked, and they are validated in
``validate_decomposition.py``, ``validate_convolution.py`` and
``validate_bounds.py``, which run before this script. An under-strength
baseline inflates an apparent learned win -- a previous product of this
mission measured exactly that, at about fifty per cent -- so the strength of
the baselines is a correctness requirement and not a courtesy.

WHAT IS COMPARED, AND ON WHAT AXIS
----------------------------------
Everything is scored on the **shortfall of a pass in Mbit**, which is the
product's headline quantity, and the primary score is the **CRPS**, which is
strictly proper over a whole predictive distribution. A point forecast is the
one-member ensemble, for which the CRPS is the absolute error exactly
(Gneiting & Raftery 2007, section 4.2), so baseline 1 sits on the same axis
as a distribution without special pleading.

Because the product's value is a distribution a scheduler books against,
**calibration is reported beside the error and is the figure that decides
whether a forecast is usable**: the PIT Kolmogorov distance from uniform, and
the coverage of the nominal 80 per cent interval.

Five predictors:

- **baseline 1, the booked figure.** Predicts no shortfall. The thing being
  refuted.
- **baseline 2, the analytic convolution**, as the specification names it:
  measured acquisition and re-sync laws convolved over the pass geometry,
  with an unconditional fitted idle law.
- **baseline 2b, the same convolution with a backlog-aware idle term.** The
  backlog-limited dry time is knowable before the pass, so an analytic model
  is entitled to use it. Reported beside the specified form and never
  instead of it.
- **the constant-fraction null.** Every pass gets the training set's
  empirical law of the realised fraction. A learned model that cannot beat
  this has learned nothing from the features, and an analytic baseline that
  cannot beat it is weaker than knowing nothing but the mean.
- **the learned forest**, in both uncertainty modes: the across-tree spread
  and the out-of-bag residual law.

Baseline 3 is ground truth and is what everything is scored against, so it is
not a row in the table; its campaign figures are printed first.

THE TWO SPLITS ARE SCORED SEPARATELY AND THE GAP IS REPORTED
------------------------------------------------------------
Held-out **stations** (whole stations, so the model must transfer to a figure
of merit, a mask and a latitude it never trained on) and held-out **time**
(the last quarter of the campaign, chronologically, never a random subset).
The station split is the harder one and the gap between the two is itself a
reported quantity.

COMPUTE
-------
One forest fit per split, 300 trees, one core. The whole script runs in well
under the 60 s per-fit budget and well under the 600 s the release gate allows
per script. No wall-clock measurement in this script is compared against
another one.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from passyield.baselines import (  # noqa: E402
    accounted_yield,
    analytic_shortfall_distribution,
    backlog_dry_time_s,
    booked_shortfall_mbit,
    fit_event_model,
)
from passyield.dataset import (  # noqa: E402
    DEFAULT_HELDOUT_STATIONS,
    DEFAULT_TIME_HOLDOUT_FRACTION,
    events_of,
    feature_matrix,
    load_passes,
    parse_intervals,
    station_by_name,
    station_split,
    target_vector,
    time_split,
    window_of,
)
from passyield.decompose import decompose_pass  # noqa: E402
from passyield.geometry import satellite  # noqa: E402
from passyield.predictor import (  # noqa: E402
    DECLARED_UNCERTAINTY_MODE,
    UNCERTAINTY_MODES,
    PassYieldPredictor,
    PredictorSettings,
)
from passyield.report import format_accounted, format_comparison  # noqa: E402
from passyield.scoring import (  # noqa: E402
    crps_mean,
    interval_coverage,
    mean_absolute_error,
    pit_calibration_error,
    pit_values,
    root_mean_square_error,
)

#: Members in every predictive ensemble, so the CRPS estimator's own bias is
#: the same for every row of the table.
MEMBERS = 256


def _fit_event_laws(rows: list) -> object:
    """Fit the event laws on a TRAINING subset only. No leakage."""
    return fit_event_model(
        acquisition_s=[row.acquisition_s for row in rows],
        outage_counts=[row.outage_count for row in rows],
        window_durations_s=[row.duration_s for row in rows],
        outage_durations_s=[
            duration for row in rows
            for _start, duration in parse_intervals(row.resync_intervals)
        ],
        idle_totals_s=[row.idle_total_s for row in rows],
    )


def _score(ensemble: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    """CRPS, MAE, RMSE, PIT calibration error and 80 per cent coverage."""
    low = np.quantile(ensemble, 0.10, axis=1)
    high = np.quantile(ensemble, 0.90, axis=1)
    return {
        "crps": crps_mean(ensemble, observed),
        "mae": mean_absolute_error(ensemble.mean(axis=1), observed),
        "rmse": root_mean_square_error(ensemble.mean(axis=1), observed),
        "pit_calibration_error": pit_calibration_error(pit_values(ensemble, observed)),
        "coverage80": interval_coverage(low, high, observed),
    }


def main() -> int:
    """Benchmark the learned component against the baselines on both splits."""
    out = _bootstrap.Tee("validate_learned_output.txt")
    out("Learned predictor against the baselines -- P099 PassYield")
    out("")
    out(f"fast path (PASSYIELD_FAST=1): {_bootstrap.FAST}")
    out(f"seed                        : {_bootstrap.SEED}")
    out(f"ensemble members            : {MEMBERS}")
    out(f"nominal interval coverage   : {_bootstrap.NOMINAL_COVERAGE!r}")
    out("")
    out("EVERYTHING IS SCORED ON THE SHORTFALL OF A PASS IN Mbit, which is this")
    out("product's headline quantity. The primary score is the CRPS, which is")
    out("strictly proper over a whole predictive distribution; a point forecast is")
    out("the one-member ensemble, for which the CRPS is the absolute error exactly")
    out("(Gneiting & Raftery 2007, JASA 102(477), section 4.2).")
    out("")
    out("CALIBRATION IS REPORTED BESIDE THE ERROR and is the figure that decides")
    out("whether a forecast is usable, because the product's value is a")
    out("distribution a scheduler books against. PIT cal err is the Kolmogorov")
    out("distance of the empirical probability integral transform from the uniform")
    out("(Dawid 1984; Gneiting, Balabdaoui & Raftery 2007). cov80 is the coverage")
    out("of the nominal 80 per cent interval: the target is 0.800 and EITHER")
    out("direction away from it is a defect.")

    satrec = satellite()
    record = load_passes()
    if _bootstrap.FAST:
        # The fast path exists for the working-directory probe, whose question
        # is whether this script RUNS from a given directory. No number in
        # VALIDATION.md, README.md or MODEL_CARD.md comes from a fast-path run.
        record = record[::12]
        out("")
        out(f"FAST PATH: the record is subsampled to {len(record)} passes and the")
        out("forest to 40 trees. No reported number comes from a fast-path run.")
    settings = (
        PredictorSettings(n_estimators=40) if _bootstrap.FAST else PredictorSettings()
    )
    out("")
    out(f"forest: {settings.n_estimators} trees, max_depth {settings.max_depth}, "
        f"min_samples_leaf {settings.min_samples_leaf}, random_state "
        f"{settings.random_state}, n_jobs 1")

    out.rule("BASELINE 3 -- GROUND TRUTH OVER THE WHOLE CAMPAIGN")
    out("directly accounted from the per-pass record, no model of anything. The")
    out("resampling unit is the PASS, carried on the returned object and asserted")
    out("by tests/test_baselines.py.")
    out("")
    decompositions = [
        decompose_pass(satrec, station_by_name(row.station), window_of(row), events_of(row))
        for row in record
    ]
    accounted = accounted_yield(
        decompositions,
        seed=_bootstrap.SEED,
        resamples=200 if _bootstrap.FAST else _bootstrap.BOOTSTRAP_RESAMPLES,
        confidence=_bootstrap.CONFIDENCE,
    )
    out(format_accounted(accounted))
    out("")
    out.check(
        "baseline 3 declares the pass as its resampling unit",
        accounted.resampling_unit == "pass",
        f"{accounted.resampling_unit!r}",
    )
    out.check(
        "the bootstrap interval brackets the point estimate",
        accounted.mean_shortfall_ci_mbit[0]
        <= accounted.mean_shortfall_mbit
        <= accounted.mean_shortfall_ci_mbit[1],
    )

    out.rule("BASELINE 1 -- THE BOOKED FIGURE, AT FULL STRENGTH")
    out("nominal window duration times nominal rate. The duration is the true")
    out("geometric mask-to-mask window from the committed SGP4 element set, not a")
    out("padded or rounded block; the rate is the link budget's own rate at a")
    out("declared 30 degree reference elevation with the station's own figure of")
    out("merit, which is exactly what a scheduler with an ephemeris and a station")
    out("catalogue can compute before the pass. Weakening either would inflate")
    out("every comparison below.")
    out("")
    nominal_total = accounted.nominal_mbit
    realised_total = accounted.realised_mbit
    out(f"campaign booked volume, Mbit    : {nominal_total:.1f}")
    out(f"campaign realised volume, Mbit  : {realised_total:.1f}")
    out(f"realised / booked               : {realised_total / nominal_total:.6f}")
    out(f"baseline 1's predicted shortfall: {booked_shortfall_mbit()!r} Mbit")
    out(f"the campaign's actual shortfall : {accounted.shortfall_mbit:.1f} Mbit")
    out("")
    out("THE HEADLINE REFUTATION: a schedule built on baseline 1 books "
        f"{nominal_total / realised_total:.2f} times")
    out("the volume the record accounts for.")

    matrix = feature_matrix(record)
    target = target_vector(record)
    nominal = np.asarray([row.nominal_mbit for row in record])
    capacity = np.asarray([row.capacity_mbit for row in record])
    observed = np.asarray([row.shortfall_mbit for row in record])

    tables: dict[str, dict[str, dict[str, float]]] = {}
    for split in (station_split(record), time_split(record)):
        out.rule(f"SPLIT: {split.name.upper()}")
        out(split.description)
        train = list(split.train)
        test = list(split.test)
        out("")
        out(f"training passes : {len(train)}")
        out(f"held-out passes : {len(test)}")
        out(f"training stations: {sorted({record[index].station for index in train})}")
        out(f"held-out stations: {sorted({record[index].station for index in test})}")

        laws = _fit_event_laws([record[index] for index in train])
        out("")
        out("event laws fitted on the TRAINING side only:")
        out(f"  acquisition Gamma shape, scale : {laws.acquisition_shape:.6f}, "
            f"{laws.acquisition_scale:.6f} s")
        out(f"  outage intensity, 1/s          : {laws.outage_intensity_per_s:.8f}")
        out(f"  mean outage duration, s        : {laws.outage_mean_s:.6f}")
        out(f"  idle Gamma shape, scale        : {laws.idle_shape:.6f}, "
            f"{laws.idle_scale:.6f} s")

        predictor = PassYieldPredictor(settings).fit(matrix[train], target[train])
        residuals = predictor.out_of_bag_residuals
        out("")
        out(f"out-of-bag residuals of the realised fraction: {residuals.size}")
        out(f"  mean    : {float(np.mean(residuals)):+.6f}")
        out(f"  sd      : {float(np.std(residuals, ddof=1)):.6f}")

        ensembles: dict[str, np.ndarray] = {}
        ensembles["baseline 1 booked"] = np.full((len(test), 1), booked_shortfall_mbit())

        levels = (np.arange(MEMBERS) + 0.5) / MEMBERS
        fraction_quantiles = np.quantile(target[train], levels)
        ensembles["null constant fraction"] = (
            nominal[test][:, None] - fraction_quantiles[None, :] * capacity[test][:, None]
        )

        for label, backlog_aware in (
            ("baseline 2 convolution", False),
            ("baseline 2b backlog-aware", True),
        ):
            members: list[np.ndarray] = []
            for index in test:
                row = record[index]
                station = station_by_name(row.station)
                window = window_of(row)
                dry = (
                    backlog_dry_time_s(satrec, station, window, row.backlog_mbit)
                    if backlog_aware
                    else None
                )
                law = analytic_shortfall_distribution(
                    satrec, station, window, laws, nominal[index], capacity[index],
                    deterministic_idle_s=dry,
                )
                drawn = law.samples_for_scoring(MEMBERS)
                members.append(drawn)
            width = min(item.size for item in members)
            ensembles[label] = np.asarray([item[:width] for item in members])

        for mode in UNCERTAINTY_MODES:
            name = f"learned {mode}"
            if mode == DECLARED_UNCERTAINTY_MODE:
                name += " (declared)"
            ensembles[name] = predictor.predict_shortfall_ensemble(
                matrix[test], nominal[test], capacity[test], mode
            )

        scores = {
            name: _score(ensemble, observed[test]) for name, ensemble in ensembles.items()
        }
        tables[split.name] = scores
        out("")
        out(format_comparison(scores, f"scored on the shortfall in Mbit -- {split.name}"))
        out("")
        out(f"{'predictor':<28}{'RMSE, Mbit':>13}{'ensemble members':>19}")
        for name, ensemble in ensembles.items():
            out(f"  {name:<26}{scores[name]['rmse']:>13.2f}{ensemble.shape[1]:>19}")

        best = min(scores, key=lambda name: scores[name]["crps"])
        best_calibrated = min(scores, key=lambda name: scores[name]["pit_calibration_error"])
        out("")
        out(f"lowest CRPS on this split          : {best} "
            f"({scores[best]['crps']:.2f} Mbit)")
        out(f"best calibrated on this split      : {best_calibrated} "
            f"(PIT cal err {scores[best_calibrated]['pit_calibration_error']:.4f})")
        out("")
        out.check(
            "baseline 1 is refuted: every other predictor has a lower CRPS",
            all(
                scores[name]["crps"] < scores["baseline 1 booked"]["crps"]
                for name in scores
                if name != "baseline 1 booked"
            ),
            f"baseline 1 CRPS {scores['baseline 1 booked']['crps']:.2f} Mbit",
        )
        out.check(
            "baseline 1's 80 per cent interval covers nothing, as a point must",
            scores["baseline 1 booked"]["coverage80"] == 0.0,
            f"coverage {scores['baseline 1 booked']['coverage80']:.4f}",
        )

    out.rule("THE GAP BETWEEN THE TWO SPLITS")
    out("the station split is the harder one, and the gap is itself a reported")
    out("quantity: it is the measure of how much of the learned model's advantage")
    out("depends on having trained on the station it is forecasting.")
    out("")
    station_scores = tables["held-out stations"]
    time_scores = tables["held-out time"]
    out(f"{'predictor':<28}{'CRPS stations':>15}{'CRPS time':>12}{'ratio':>8}"
        f"{'PIT stations':>14}{'PIT time':>10}")
    for name in station_scores:
        first = station_scores[name]["crps"]
        second = time_scores[name]["crps"]
        out(
            f"  {name:<26}{first:>15.2f}{second:>12.2f}{first / second:>8.2f}"
            f"{station_scores[name]['pit_calibration_error']:>14.4f}"
            f"{time_scores[name]['pit_calibration_error']:>10.4f}"
        )
    out("")
    declared = f"learned {DECLARED_UNCERTAINTY_MODE} (declared)"
    out(f"the declared learned configuration is {declared!r}.")
    out(f"  CRPS on held-out stations : {station_scores[declared]['crps']:.2f} Mbit")
    out(f"  CRPS on held-out time     : {time_scores[declared]['crps']:.2f} Mbit")
    out(f"  station/time CRPS ratio   : "
        f"{station_scores[declared]['crps'] / time_scores[declared]['crps']:.2f}")
    out(f"  PIT cal err, stations     : "
        f"{station_scores[declared]['pit_calibration_error']:.4f}")
    out(f"  PIT cal err, time         : "
        f"{time_scores[declared]['pit_calibration_error']:.4f}")
    out(f"  cov80, stations           : {station_scores[declared]['coverage80']:.4f}")
    out(f"  cov80, time               : {time_scores[declared]['coverage80']:.4f}")
    out("")
    out.check(
        "the station split is harder than the time split for the learned model",
        station_scores[declared]["crps"] > time_scores[declared]["crps"],
        f"{station_scores[declared]['crps']:.2f} against "
        f"{time_scores[declared]['crps']:.2f} Mbit",
    )

    out.rule("WHICH PREDICTOR WON, AND WHAT THAT MEANS")
    for name, scores in (("held-out stations", station_scores), ("held-out time", time_scores)):
        ranking = sorted(scores, key=lambda key: scores[key]["crps"])
        out(f"{name}, by CRPS (lower is better):")
        for position, key in enumerate(ranking, start=1):
            out(f"  {position}. {key:<28}{scores[key]['crps']:>12.2f} Mbit")
        calibration = sorted(scores, key=lambda key: scores[key]["pit_calibration_error"])
        out(f"{name}, by PIT calibration error (lower is better):")
        for position, key in enumerate(calibration, start=1):
            out(f"  {position}. {key:<28}"
                f"{scores[key]['pit_calibration_error']:>12.4f}")
        out("")
    out("THE FINDINGS, AS MEASURED, WITH NOTHING RETUNED AWAY:")
    out("")
    declared_station = station_scores[declared]
    declared_time = time_scores[declared]
    null_station = station_scores["null constant fraction"]
    null_time = time_scores["null constant fraction"]
    booked_station = station_scores["baseline 1 booked"]
    booked_time = time_scores["baseline 1 booked"]
    out("1. Baseline 1 is refuted on both splits. Its CRPS is "
        f"{booked_station['crps'] / declared_station['crps']:.1f} times the")
    out(f"   learned model's on held-out stations and "
        f"{booked_time['crps'] / declared_time['crps']:.1f} times on held-out")
    out("   time, and its 80 per cent interval covers nothing, because it is a")
    out("   point. That is the product's purpose and it is the expected result.")
    out("")
    # Finding 2 is DERIVED from the measurement rather than written in, because
    # an earlier draft of this script asserted in prose that the null beat both
    # analytic baselines on both splits, which the full record then falsified
    # for baseline 2b on the station split. A narrative that can disagree with
    # the table above it is worse than no narrative.
    for name, scores in (("held-out stations", station_scores), ("held-out time", time_scores)):
        analytic = ("baseline 2 convolution", "baseline 2b backlog-aware")
        beaten = [
            key for key in analytic
            if scores["null constant fraction"]["crps"] < scores[key]["crps"]
        ]
        out(f"2. On {name}, the constant-fraction null beats "
            f"{len(beaten)} of the 2 analytic")
        out(f"   baselines on CRPS: {beaten if beaten else 'none'}.")
    out("   The specification names baseline 2 as the analytic model, and in its")
    out("   specified form it is beaten by the null on BOTH splits. That is")
    out("   reported here rather than removed, and the reason is diagnosable:")
    out("   baseline 2's idle law is UNCONDITIONAL, while the idle time in the")
    out("   record is driven by the onboard backlog, which is a pre-pass")
    out("   quantity. Baseline 2b uses it and is the better analytic form on")
    out("   both splits, so it is reported BESIDE the specified one; it does not")
    out("   replace it, because the specification names the unconditional form")
    out("   and the measurement of that form is the finding.")
    out("")
    ranking_station = sorted(station_scores, key=lambda key: station_scores[key]["crps"])
    ranking_time = sorted(time_scores, key=lambda key: time_scores[key]["crps"])
    out(f"3. Lowest CRPS on held-out stations: {ranking_station[0]}.")
    out(f"   Lowest CRPS on held-out time    : {ranking_time[0]}.")
    out("")
    calibration_station = sorted(
        station_scores, key=lambda key: station_scores[key]["pit_calibration_error"]
    )
    calibration_time = sorted(
        time_scores, key=lambda key: time_scores[key]["pit_calibration_error"]
    )
    out(f"4. Best CALIBRATED on held-out stations: {calibration_station[0]}.")
    out(f"   Best CALIBRATED on held-out time    : {calibration_time[0]}.")
    out("   Calibration is what a scheduler needs in order to book a quantile, so")
    out("   where a baseline wins on it that is a baseline result on the")
    out("   dimension that matters most, and it is published as one.")
    out("")
    out("the measured form of finding 4 on the harder split:")
    out(f"  learned PIT cal err, stations : {declared_station['pit_calibration_error']:.4f}")
    out(f"  null    PIT cal err, stations : {null_station['pit_calibration_error']:.4f}")
    out(f"  learned cov80, stations       : {declared_station['coverage80']:.4f}")
    out(f"  null    cov80, stations       : {null_station['coverage80']:.4f}")
    out("")
    out("and on held-out time, where the learned model's own uncertainty output is")
    out("close to calibrated:")
    out(f"  learned PIT cal err, time     : {declared_time['pit_calibration_error']:.4f}")
    out(f"  null    PIT cal err, time     : {null_time['pit_calibration_error']:.4f}")
    out(f"  learned cov80, time           : {declared_time['coverage80']:.4f} against a "
        f"nominal {_bootstrap.NOMINAL_COVERAGE!r}")
    out("")
    out.check(
        "a baseline result is recorded where one exists: on the station split the "
        "null is better calibrated than the learned model",
        null_station["pit_calibration_error"]
        < declared_station["pit_calibration_error"],
        f"null {null_station['pit_calibration_error']:.4f} against learned "
        f"{declared_station['pit_calibration_error']:.4f}",
    )
    out.check(
        "the learned model's declared uncertainty output is better calibrated than "
        "its tree spread on both splits",
        station_scores[declared]["pit_calibration_error"]
        < station_scores["learned tree-spread"]["pit_calibration_error"]
        and time_scores[declared]["pit_calibration_error"]
        < time_scores["learned tree-spread"]["pit_calibration_error"],
        f"stations {station_scores[declared]['pit_calibration_error']:.4f} against "
        f"{station_scores['learned tree-spread']['pit_calibration_error']:.4f}, time "
        f"{time_scores[declared]['pit_calibration_error']:.4f} against "
        f"{time_scores['learned tree-spread']['pit_calibration_error']:.4f}",
    )
    out("")
    out(f"held-out stations were {sorted(DEFAULT_HELDOUT_STATIONS)}, declared in")
    out("passyield.dataset before any measurement and not changed afterwards.")
    out(f"the time holdout fraction was {DEFAULT_TIME_HOLDOUT_FRACTION!r}, likewise.")
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
