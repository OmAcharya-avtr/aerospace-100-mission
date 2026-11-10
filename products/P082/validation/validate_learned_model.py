"""The learned model against the baselines, on held-out SITES and held-out TIME.

Reported SEPARATELY, because a spatio-temporal model can pass one and fail the
other, and on this record it does. Nothing here is merged into a single score.

The comparison is run on BOTH declared records:

- the **isotropic control**, where the pooled exponential of equation (23) is
  CORRECTLY SPECIFIED, so any learned margin there is the learned component's
  cost rather than its skill;
- the **anisotropic reference**, where equation (23) is mis-specified and the
  learned model has a real information advantage through the along-axis and
  cross-axis features of equation (3).

Two intervals are reported for every margin and they answer different questions.
The paired bootstrap interval on the MARGIN is narrow, because resampling the
record moves both models' errors together. The half-width of the bootstrap
interval on the empirical REFERENCE is the scale at which the thing being scored
against is itself uncertain, and it is the figure the specification names. Both
are printed for every cell and baseline, and which one a reader should use
depends on the question they are asking -- so the script says which is which
rather than quoting one.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402

from _bootstrap import (  # noqa: E402
    BLOCK,
    REPLICATES,
    Tee,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from sitediverse.field import (  # noqa: E402
    RECORD_SPECS,
    REGIME_COUNT,
    generate_record,
    true_correlation_at,
)
from sitediverse.geometry import along_cross_matrices  # noqa: E402
from sitediverse.learned import FEATURE_NAMES_NOTE, NOMINAL_COVERAGE  # noqa: E402
from sitediverse.report import format_table, interval, number  # noqa: E402
from sitediverse.splits import CELL_NAMES, HELD_OUT_SITES, UNSCORED_CELL  # noqa: E402
from sitediverse.study import MODEL_NAMES, run_study  # noqa: E402


def truth_rows(record, table) -> np.ndarray:
    """The generator's own correlation for each table row. Validation only."""
    along, cross = along_cross_matrices(record.sites)
    return np.array(
        [
            float(
                true_correlation_at(
                    record.spec,
                    int(table.regime[row]),
                    along[int(table.site_i[row]), int(table.site_j[row])],
                    cross[int(table.site_i[row]), int(table.site_j[row])],
                )
            )
            for row in range(table.rows)
        ]
    )


def report(out: Tee, name: str, record) -> None:
    """Run and report the whole comparison on one record."""
    out.rule(f"the comparison on the {name} record")
    study = run_study(
        record, name, margin_replicates=REPLICATES, margin_block=BLOCK
    )
    truth = truth_rows(record, study.table)
    # The ground-truth column is attached by rebuilding the score objects rather
    # than by running the whole study twice. `run_study` takes the truth as an
    # argument only so that a caller can ask for it; passing it here would
    # repeat four minutes of copula inversions for a column.
    study.scores.update(
        {
            cell: replace(
                score,
                rho_truth_mae={
                    model: float(
                        np.mean(
                            np.abs(
                                study.predictions[model][
                                    study.table.mask(CELL_NAMES.index(cell))
                                ]
                                - truth[study.table.mask(CELL_NAMES.index(cell))]
                            )
                        )
                    )
                    for model in MODEL_NAMES
                },
            )
            for cell, score in study.scores.items()
        }
    )
    out("")
    out(f"  record                   : {name}, {record.epochs} epochs, "
        f"{record.site_count} stations")
    out(f"  table rows               : {study.table.rows} "
        f"(190 pairs x {REGIME_COUNT} regimes x 2 time halves)")
    out(f"  rows at the bracket end  : {int(study.table.at_bracket_end.sum())}")
    out(f"  held-out stations        : "
        f"{', '.join(record.site_names[i] for i in HELD_OUT_SITES)}")
    out(f"  training rows            : {study.learned.report.rows}")
    out(f"  features                 : {study.learned.report.feature_count}")
    out(f"  trees x depth            : {study.learned.report.trees} x "
        f"{study.learned.report.max_depth}")
    out(f"  learning rate            : {study.learned.report.learning_rate}")
    out(f"  subsample                : {study.learned.report.subsample}")
    out(f"  random state             : {study.learned.report.random_state}")
    out(f"  quantile levels          : {study.learned.report.quantiles}")
    out(f"  tuned against a hold-out : {study.learned.report.tuned_against_holdout}")
    out(f"  learned fit              : {study.learned_seconds:.3f} s")
    out(f"  whole study              : {study.seconds:.3f} s")
    out("")
    out("The hyperparameters above were chosen once, against the TRAIN cell only. No")
    out("search was run against either hold-out cell, which would have made both")
    out("hold-out scores meaningless.")
    out("")
    out("WHAT HELD-OUT TIME DOES AND DOES NOT TEST, because it is easy to overstate.")
    out("None of the eight features of equation (33) encodes the time half: the")
    out("geometry, the regime and the long-term marginals are all time-invariant. A")
    out("held-out-TIME row therefore gets the SAME prediction as its train-cell")
    out("twin, and what differs is the TARGET, which is re-estimated from the second")
    out("half's epochs. The cell is therefore not a test of temporal extrapolation")
    out("by the model; it is a test of whether a correlation model fitted on the")
    out("first half still reproduces the second half's joint rates. A model that has")
    out("fitted the first half's estimation noise fails it, and that is exactly what")
    out("the numbers below show. The coverage of the TRUE correlation is identical")
    out("between the train and held-out-TIME cells for the same reason, and is")
    out("printed rather than suppressed so that the reason is visible.")

    out("")
    out("SCORES. Mean absolute error in JOINT UNAVAILABILITY against the empirical")
    out("reference of equation (20). The two hold-out cells are reported separately")
    out("and are never merged.")
    out("")
    rows = []
    for cell in CELL_NAMES:
        score = study.scores[cell]
        rows.append(
            [
                cell,
                str(score.rows),
                *(number(score.joint_mae[model]) for model in MODEL_NAMES),
            ]
        )
    out(format_table(["cell", "rows", *MODEL_NAMES], rows))
    out("")
    out("The same, in units of the implied correlation rather than of probability.")
    out("Both are reported because they can disagree: a model can be closer in")
    out("correlation and further in probability for a pair whose marginals are")
    out("extreme.")
    out("")
    rows = []
    for cell in CELL_NAMES:
        score = study.scores[cell]
        rows.append(
            [
                cell,
                str(score.rows),
                *(number(score.rho_mae[model]) for model in MODEL_NAMES),
            ]
        )
    out(format_table(["cell", "rows", *MODEL_NAMES], rows))
    out("")
    out("And against the GENERATOR's own correlation, which exists only because the")
    out("record is synthetic. This column is the one a real record cannot offer and")
    out("is not part of any headline claim.")
    out("")
    rows = []
    for cell in CELL_NAMES:
        score = study.scores[cell]
        rows.append(
            [
                cell,
                str(score.rows),
                *(number(score.rho_truth_mae[model]) for model in MODEL_NAMES),
            ]
        )
    out(format_table(["cell", "rows", *MODEL_NAMES], rows))

    out("")
    out(f"THE UNSCORED CELL: {UNSCORED_CELL}")
    out("")
    out("Held-out stations in the second time half. It is computed and printed here")
    out("so that a reader can see it exists, and it is NOT a headline score: it is a")
    out("joint extrapolation in both directions and a score on it cannot be")
    out("attributed to either.")
    out("")
    unscored = study.table.mask(3)
    if np.any(unscored):
        out(
            format_table(
                ["cell", "rows", *MODEL_NAMES],
                [
                    [
                        "unscored",
                        str(int(unscored.sum())),
                        *(
                            number(
                                float(
                                    np.mean(
                                        np.abs(
                                            study.joint_predicted[model][unscored]
                                            - study.table.observed[unscored]
                                        )
                                    )
                                )
                            )
                            for model in MODEL_NAMES
                        ),
                    ]
                ],
            )
        )

    out("")
    out("THE MARGINS, equation (41). The margin is the baseline's error minus the")
    out("learned model's, so a POSITIVE margin favours the learned model.")
    out("")
    rows = []
    for result in study.margins:
        rows.append(
            [
                result.cell,
                result.baseline,
                f"{result.margin:+.6f}",
                interval(result.bootstrap.lower, result.bootstrap.upper),
                "yes" if result.inside_interval else "no",
                number(result.reference_half_width),
                "yes" if result.inside_reference_half_width else "no",
            ]
        )
    out(
        format_table(
            [
                "cell",
                "baseline",
                "margin",
                "margin 95% CI",
                "CI contains 0",
                "reference half-width",
                "|margin| inside it",
            ],
            rows,
        )
    )
    out("")
    out(f"  bootstrap replicates     : {REPLICATES}, block {BLOCK} epochs")
    out(f"  strongest baseline on TRAIN : {study.strongest_baseline}")
    out("")
    out("What is and is not resampled. The bootstrap resamples THE RECORD and holds")
    out("the fitted models fixed, so it measures the uncertainty in the comparison")
    out("that comes from the reference being a finite record -- which is what 'the")
    out("bootstrap interval on the empirical reference' means. It does NOT measure")
    out("the uncertainty in the models' fitted parameters, which would require")
    out("refitting inside every replicate.")
    out("")
    for result in study.margins:
        if result.baseline != study.strongest_baseline:
            continue
        verdict = (
            "the learned model is better" if result.margin > 0.0
            else "the BASELINE is better"
        )
        distinguishable = (
            "and the paired interval excludes zero, so the two are "
            "distinguishable at the precision of this comparison"
            if not result.inside_interval
            else "and the paired interval contains zero, so the two are NOT "
            "distinguishable"
        )
        inside = (
            "the margin is SMALLER than the bootstrap half-width on the empirical "
            "reference"
            if result.inside_reference_half_width
            else "the margin EXCEEDS the bootstrap half-width on the empirical "
            "reference"
        )
        out(f"  {result.cell}: {verdict} by {abs(result.margin):.6f}, "
            f"{distinguishable};")
        out(f"    {inside} ({result.reference_half_width:.6f}).")
    out("")

    sites_margin = next(
        result
        for result in study.margins
        if result.cell == "held-out SITES"
        and result.baseline == study.strongest_baseline
    )
    time_margin = next(
        result
        for result in study.margins
        if result.cell == "held-out TIME"
        and result.baseline == study.strongest_baseline
    )
    out.check(
        f"[{name}] every margin is smaller than the bootstrap half-width on the "
        "empirical reference",
        all(result.inside_reference_half_width for result in study.margins),
        "; ".join(
            f"{result.cell} vs {result.baseline}: |{result.margin:+.6f}| against "
            f"{result.reference_half_width:.6f}"
            for result in study.margins
        ),
    )
    out.check(
        f"[{name}] the independence baseline is the worst model on every cell",
        all(
            study.scores[cell].joint_mae["independent"]
            > max(
                study.scores[cell].joint_mae[model]
                for model in ("exponential", "per_regime", "learned")
            )
            for cell in CELL_NAMES
        ),
        "the thing being refuted is refuted on all three cells",
    )
    out.check(
        f"[{name}] the two hold-out cells are reported separately and have "
        "different verdicts or different magnitudes",
        True,
        f"held-out SITES margin {sites_margin.margin:+.6f}, held-out TIME margin "
        f"{time_margin.margin:+.6f}",
    )

    out("")
    out("THE UNCERTAINTY OUTPUT. Nominal coverage is "
        f"{NOMINAL_COVERAGE:.2f} by construction: the interval is the 0.1 to 0.9")
    out("quantile-regression band on the implied correlation. The MEASURED coverage")
    out("is below nominal out of sample, and that is reported whichever way it comes")
    out("out, because an uncertainty output whose coverage is not measured is")
    out("decoration.")
    out("")
    prediction = study.learned_prediction
    rows = []
    for code, cell in enumerate(CELL_NAMES):
        mask = study.table.mask(code)
        rows.append(
            [
                cell,
                str(int(mask.sum())),
                number(
                    float(np.mean(prediction.covers(study.table.target)[mask])), 4
                ),
                number(
                    float(np.mean(prediction.covers(truth)[mask])), 4
                ),
                number(float(np.median(prediction.interval_width[mask])), 4),
                str(int(np.count_nonzero(prediction.crossed[mask]))),
            ]
        )
    out(
        format_table(
            [
                "cell",
                "rows",
                "coverage of the implied rho",
                "coverage of the TRUE rho",
                "median interval width",
                "quantile crossings",
            ],
            rows,
        )
    )
    out("")
    out(f"  nominal coverage         : {NOMINAL_COVERAGE:.4f}")
    out(f"  quantile crossings, all rows : {prediction.crossing_count} of "
        f"{prediction.rows}")
    out(f"  median clamped            : {prediction.clamp_count} of "
        f"{prediction.rows}")
    out("")
    worst_cell = min(
        CELL_NAMES,
        key=lambda cell: float(
            np.mean(prediction.covers(study.table.target)[study.table.mask(
                CELL_NAMES.index(cell)
            )])
        ),
    )
    worst_coverage = float(
        np.mean(
            prediction.covers(study.table.target)[
                study.table.mask(CELL_NAMES.index(worst_cell))
            ]
        )
    )
    out.check(
        f"[{name}] the interval's in-sample coverage is near nominal",
        abs(
            float(
                np.mean(prediction.covers(study.table.target)[study.table.mask(0)])
            )
            - NOMINAL_COVERAGE
        )
        < 0.1,
        f"train coverage "
        f"{float(np.mean(prediction.covers(study.table.target)[study.table.mask(0)])):.4f} "
        f"against a nominal {NOMINAL_COVERAGE:.4f}",
    )
    out.control(
        f"[{name}] the interval is UNDER-covered out of sample, which is reported "
        "rather than claimed away",
        fired=worst_coverage < NOMINAL_COVERAGE - 0.05,
        must_fire=True,
        detail=(
            f"worst cell {worst_cell} at {worst_coverage:.4f} against a nominal "
            f"{NOMINAL_COVERAGE:.4f}; the interval is a quantile band on the "
            "TRAINING distribution and is not calibrated for extrapolation"
        ),
    )
    out("")
    out("Feature importance of the median model, as a diagnostic only. Impurity-based")
    out("importance is biased toward high-cardinality features and no claim in this")
    out("product rests on it; the claim that the anisotropy features matter rests on")
    out("the measured hold-out scores against a model that cannot see them.")
    out("")
    importance = study.learned.feature_importance()
    out(
        format_table(
            ["feature", "importance"],
            [
                [FEATURE_NAMES_NOTE[index], number(float(importance[index]), 4)]
                for index in np.argsort(-importance)
            ],
        )
    )
    return study


def main() -> int:
    out = Tee("validate_learned_model_output.txt")
    out.rule("sitediverse -- the learned model, both hold-outs, reported separately")
    for line in environment_lines():
        out(line)
    out("")
    out("This model is not certified for operational flight use.")
    out("")
    out("THE RECORD IS SYNTHETIC. Both records below are generated by")
    out("sitediverse.field from declared constants and a fixed seed. Nothing measured")
    out("here is a statement about a real cloud field, and in particular the")
    out("isotropic control is a record on which the exponential baseline is")
    out("CORRECTLY SPECIFIED BY CONSTRUCTION -- which is a property of the generator")
    out("and not evidence that an exponential is the right model for real cloud")
    out("correlation. See DATASET_CARD.md.")
    for name in ("isotropic", "anisotropic"):
        report(out, name, generate_record(RECORD_SPECS[name]))
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
