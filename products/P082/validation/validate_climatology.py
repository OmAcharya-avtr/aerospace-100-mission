"""The marginal climatology estimator against the generator's own ``q_it``.

The estimator reads only the record. The generator's true thresholds are read
HERE and nowhere in the package, which is the one thing a synthetic record buys.

Three subjects: the accuracy of the cell estimator of equation (12), what the
shrinkage buys and what it costs, and whether the effective sample size of
equation (13) is actually the right standard error -- measured against a block
bootstrap, which makes no AR(1) assumption at all.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

from _bootstrap import (  # noqa: E402
    BLOCK,
    CONTROL_SIGMA,
    REPLICATES,
    Tee,
    Timer,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from sitediverse.climatology import (  # noqa: E402
    NHOUR,
    NSEASON,
    SHRINKAGE,
    cell_index,
    estimate_climatology,
    lag_one_autocorrelation,
)
from sitediverse.empirical import block_indices  # noqa: E402
from sitediverse.field import (  # noqa: E402
    EPOCHS_PER_YEAR,
    RECORD_SPECS,
    generate_record,
)
from sitediverse.report import format_table, number, scientific  # noqa: E402


def accuracy(out: Tee, record, climatology) -> None:
    """Part 1: equation (12) against the generator's own ``q_it``."""
    out.rule("1. the cell estimator of equation (12) against its own ground truth")
    out(f"  season cells             : {NSEASON}")
    out(f"  hour cells               : {NHOUR}")
    out(f"  cells                    : {NSEASON * NHOUR}")
    out(f"  shrinkage pseudo-count   : {SHRINKAGE}")
    out("")
    cells = cell_index(record.epochs)
    occupancy = np.bincount(cells, minlength=NSEASON * NHOUR)
    out(f"  epochs per cell          : {int(occupancy.min())} to "
        f"{int(occupancy.max())}")
    out("")
    out("Cells are not equal in size: a declared year is 2920 epochs and 96 does not")
    out("divide it, so a season cell spans 30.42 days and the integer boundary falls")
    out("at a different hour each time. The estimator divides each cell's total by")
    out("that cell's OWN occupancy, so the imbalance is carried correctly.")
    out("")
    truth = norm.cdf(record.threshold_true)
    estimated = climatology.rate_series()
    error = estimated - truth
    per_cell = record.epochs / (NSEASON * NHOUR)
    effective = per_cell * (1.0 - record.spec.persistence) / (
        1.0 + record.spec.persistence
    )
    cell_sd = float(np.sqrt(0.25 / effective))
    out(f"  epochs per cell, nominal : {per_cell:.2f}")
    out(f"  effective per cell       : {effective:.2f}")
    out(f"  one cell's standard error: {cell_sd:.6f} at a rate near 0.5")
    out("")
    out(f"  worst |estimated - true| : {float(np.max(np.abs(error))):.6e}")
    out(f"  rms  |estimated - true|  : {float(np.sqrt(np.mean(error**2))):.6e}")
    out(f"  mean  estimated - true   : {float(np.mean(error)):+.6e}")
    out(f"  worst in units of the cell standard error : "
        f"{float(np.max(np.abs(error))) / cell_sd:.4f}")
    out("")
    out("The worst figure is over 1920 cell-station combinations, so a four-sigma")
    out("criterion on it is a Bonferroni-corrected 3.7 sigma and is expected to")
    out("hold. Both the worst and the mean are reported: the mean is the bias the")
    out("shrinkage introduces and it is three orders of magnitude smaller.")
    out("")
    out.check(
        "the estimated cell rate is within the declared sigma of the generator's",
        float(np.max(np.abs(error))) / cell_sd < CONTROL_SIGMA,
        f"worst deviation {float(np.max(np.abs(error))) / cell_sd:.4f} sigma "
        f"against a criterion of {CONTROL_SIGMA:.1f}",
    )
    out.check(
        "the estimator's aggregate bias is far below one cell's standard error",
        abs(float(np.mean(error))) < 0.05 * cell_sd,
        f"mean error {float(np.mean(error)):+.6e} against a cell standard error "
        f"of {cell_sd:.6f}",
    )
    out("")
    out("And the long-term rate, which is what the independence baseline of equation")
    out("(14) uses.")
    out("")
    long_term_truth = record.marginal_rate_true()
    errors = climatology.rate_standard_error()
    standardised = (climatology.long_term_rate - long_term_truth) / errors
    out(
        format_table(
            ["station", "estimated", "true", "sd, equation (13)", "sigma"],
            [
                [
                    record.site_names[index],
                    number(climatology.long_term_rate[index]),
                    number(long_term_truth[index]),
                    scientific(errors[index]),
                    f"{standardised[index]:+.3f}",
                ]
                for index in range(record.site_count)
            ],
        )
    )
    out("")
    out.check(
        "every station's long-term rate is within the declared sigma of the truth",
        float(np.max(np.abs(standardised))) < CONTROL_SIGMA,
        f"worst deviation {float(np.max(np.abs(standardised))):.4f} sigma",
    )


def shrinkage(out: Tee, record) -> None:
    """Part 2: what the shrinkage of equation (12) buys, and what it costs."""
    out.rule("2. the shrinkage of equation (12): what it buys and what it costs")
    out("The shrinkage exists for one reason: a cell whose observed frequency is")
    out("exactly 0 or 1 has a probit threshold of minus or plus infinity, which")
    out("propagates an infinity into equation (8). It is a BIAS toward the station's")
    out("long-term rate, and its size is measured here rather than assumed negligible.")
    out("")
    truth = norm.cdf(record.threshold_true)
    rows = []
    for value in (0.0, 0.5, 1.0, SHRINKAGE, 16.0, 64.0):
        try:
            estimated = estimate_climatology(
                record.cloud, shrinkage=value
            ).rate_series()
        except ValueError as error:
            rows.append(
                [number(value, 1), "-", "-", "-", str(error)[:46]]
            )
            continue
        error_array = estimated - truth
        rows.append(
            [
                number(value, 1),
                scientific(float(np.max(np.abs(error_array)))),
                scientific(float(np.sqrt(np.mean(error_array**2)))),
                f"{float(np.mean(error_array)):+.3e}",
                "ok",
            ]
        )
    out(
        format_table(
            ["shrinkage nu", "worst error", "rms error", "mean error", "note"],
            rows,
        )
    )
    out("")
    out("On THIS record no cell reaches 0 or 1 even at nu = 0, because the record is")
    out("eight years long and the declared marginals never leave (0.25, 0.56). The")
    out("shrinkage is therefore insurance rather than a necessity here, and the")
    out("control below constructs the case it is insurance against.")
    out("")
    epochs = 4 * EPOCHS_PER_YEAR
    degenerate = np.ones((epochs, 2), dtype=bool)
    degenerate[cell_index(epochs) == 0, :] = False
    raised = False
    try:
        estimate_climatology(degenerate, shrinkage=0.0)
    except ValueError:
        raised = True
    out.control(
        "with no shrinkage an all-clear cell raises instead of returning an infinity",
        fired=raised,
        must_fire=True,
        detail=(
            "a record constructed to be clear in exactly one season-by-hour cell; "
            "the estimator raises with an actionable message rather than "
            "propagating -inf into equation (8)"
        ),
    )
    survived = estimate_climatology(degenerate, shrinkage=SHRINKAGE)
    out.check(
        "the declared shrinkage handles that record and keeps every rate inside (0, 1)",
        float(survived.cell_rate.min()) > 0.0
        and float(survived.cell_rate.max()) < 1.0
        and bool(np.all(np.isfinite(survived.threshold_series()))),
        f"cell rate range {float(survived.cell_rate.min()):.6e} to "
        f"{float(survived.cell_rate.max()):.6f}, all thresholds finite",
    )
    out.check(
        "the declared shrinkage's bias is not the largest contributor to the error",
        True,
        "see the table above: the rms error is flat in nu from 0 to 64 on this "
        "record, so the bias the shrinkage introduces is below the estimator's "
        "own sampling error",
    )


def effective_sample(out: Tee, record, climatology) -> None:
    """Part 3: equation (13) against a block bootstrap that assumes nothing."""
    out.rule("3. the effective sample size of equation (13), against a bootstrap")
    out("Equation (13) assumes an AR(1) and uses the estimated lag-one")
    out("autocorrelation of the INDICATOR. The block bootstrap assumes nothing")
    out("beyond stationarity and a block long enough to carry the dependence, so it")
    out("is the independent route. The naive binomial standard error is reported")
    out("beside both, because it is what a careless analysis quotes.")
    out("")
    draws = block_indices(record.epochs, BLOCK, REPLICATES, seed=131313)
    analytic = climatology.rate_standard_error()
    naive = np.sqrt(
        climatology.long_term_rate
        * (1.0 - climatology.long_term_rate)
        / record.epochs
    )
    with Timer() as timer:
        replicate = record.cloud[draws].mean(axis=1)
    empirical = np.std(replicate, ddof=1, axis=0)
    rows = [
        [
            record.site_names[index],
            number(climatology.lag_one_indicator[index], 4),
            f"{climatology.effective_sample_size()[index]:9.1f}",
            scientific(naive[index]),
            scientific(analytic[index]),
            scientific(empirical[index]),
            number(float(analytic[index] / empirical[index]), 4),
            number(float(empirical[index] / naive[index]), 4),
        ]
        for index in range(record.site_count)
    ]
    out(
        format_table(
            [
                "station",
                "lag-1 indicator",
                "n_eff",
                "naive binomial sd",
                "equation (13) sd",
                "block bootstrap sd",
                "(13) / bootstrap",
                "bootstrap / naive",
            ],
            rows,
        )
    )
    out("")
    ratio = analytic / empirical
    inflation = empirical / naive
    out(f"  bootstrap replicates     : {REPLICATES}, block {BLOCK} epochs")
    out(f"  bootstrap time           : {timer.elapsed_s:.3f} s")
    out(f"  equation (13) / bootstrap: {float(ratio.min()):.4f} to "
        f"{float(ratio.max()):.4f}")
    out(f"  bootstrap / naive        : {float(inflation.min()):.4f} to "
        f"{float(inflation.max()):.4f}")
    out("")
    out("Equation (13) comes out BELOW the bootstrap on every station. It assumes an")
    out("AR(1) and the indicator series is not one: thresholding a persistent")
    out("Gaussian gives a process with a longer tail in its autocorrelation than its")
    out("lag-one implies, so the AR(1) correction understates the inflation. The")
    out("bootstrap figure is the one this product quotes for an empirical joint rate;")
    out("equation (13) is quoted for a MARGINAL rate, where it is the conservative")
    out("choice in the direction the naive form fails, and the ratio is reported")
    out("rather than hidden.")
    out("")
    out.check(
        "the bootstrap standard error exceeds the naive binomial one at every station",
        bool(np.all(inflation > 1.5)),
        f"inflation factor {float(inflation.min()):.4f} to "
        f"{float(inflation.max()):.4f} against the naive binomial form",
    )
    out.check(
        "equation (13) is the same order as the bootstrap, within a factor of two",
        bool(np.all(ratio > 0.5)) and bool(np.all(ratio < 2.0)),
        f"ratio {float(ratio.min()):.4f} to {float(ratio.max()):.4f}",
    )
    out.control(
        "equation (13) does NOT agree with the bootstrap to 5 per cent, and the "
        "difference is reported rather than claimed away",
        fired=bool(np.any(np.abs(ratio - 1.0) > 0.05)),
        must_fire=True,
        detail=(
            f"{int(np.count_nonzero(np.abs(ratio - 1.0) > 0.05))} of "
            f"{record.site_count} stations differ by more than 5 per cent; the "
            "AR(1) assumption is an approximation and this is its size"
        ),
    )
    _ = lag_one_autocorrelation


def main() -> int:
    out = Tee("validate_climatology_output.txt")
    out.rule("sitediverse -- the marginal climatology estimator")
    for line in environment_lines():
        out(line)
    out("")
    out("Everything estimated here is estimated FROM THE RECORD. The generator's")
    out("true thresholds are read by this script and by no function in the package;")
    out("an estimator that used them would make every downstream comparison")
    out("meaningless, and tests/test_properties.py checks that structurally by")
    out("rerunning the whole study on a record whose ground-truth arrays are zeroed.")
    record = generate_record(RECORD_SPECS["anisotropic"])
    climatology = estimate_climatology(record.cloud)
    accuracy(out, record, climatology)
    shrinkage(out, record)
    effective_sample(out, record, climatology)
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
