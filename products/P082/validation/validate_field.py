"""The generator against its own ground truth, and what a thresholded Gaussian is.

This is the check a synthetic record exists for: the true correlation length,
the true persistence and the true marginal climatology are all known, so every
property of the field can be scored against the number that produced it. A real
record offers none of that, which is the trade ``DATASET_CARD.md`` states.

The second subject here is the limit of what equation (6) promises. The cloud
indicator is a STRICT comparison of two rounded quantities. The script measures
how many indicators flip under a perturbation of the latent field, shows that
every flipped one had a margin inside the perturbation, and shows that the cloud
RATE barely moves -- which is the positive form of the rule that a boolean
produced by a strict threshold comparison must never be asserted equal.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

from _bootstrap import (  # noqa: E402
    CONTROL_SIGMA,
    Tee,
    Timer,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from sitediverse.climatology import lag_one_autocorrelation  # noqa: E402
from sitediverse.field import (  # noqa: E402
    EPOCHS_PER_YEAR,
    RECORD_SPECS,
    REGIME_COUNT,
    generate_record,
    regime_labels,
    true_correlation,
)
from sitediverse.geometry import SITES  # noqa: E402
from sitediverse.report import format_table, number, scientific  # noqa: E402


def declared(out: Tee, record) -> None:
    """Part 1: the declared constants, printed for the record."""
    out.rule("1. the declared generator constants: the GROUND TRUTH")
    spec = record.spec
    out(f"  seed                     : {spec.seed}")
    out(f"  years                    : {spec.years}")
    out(f"  epochs                   : {record.epochs} at 3 h")
    out(f"  stations                 : {record.site_count}")
    out(f"  nugget rho_0             : {spec.nugget}")
    out(f"  persistence phi          : {spec.persistence}")
    out(f"  integral time scale      : {spec.integral_time_scale_epochs:.4f} epochs "
        f"= {3.0 * spec.integral_time_scale_epochs:.2f} h")
    out(f"  diurnal amplitude        : {spec.diurnal_amplitude} probit units")
    out("")
    out(
        format_table(
            ["regime", "L_along km", "L_cross km", "anisotropy ratio",
             "geometric mean km"],
            [
                [
                    str(index),
                    number(spec.along_km[index], 2),
                    number(spec.cross_km[index], 2),
                    number(spec.along_km[index] / spec.cross_km[index], 4),
                    number(
                        float(np.sqrt(spec.along_km[index] * spec.cross_km[index])), 2
                    ),
                ]
                for index in range(REGIME_COUNT)
            ],
        )
    )
    out("")
    out("The isotropic CONTROL record replaces each regime's two lengths by their")
    out("geometric mean, so the correlation ellipse keeps its area and the field")
    out("becomes isotropic. Against that record an isotropic two-parameter model is")
    out("CORRECTLY SPECIFIED, which is what makes it the control.")
    out("")
    counts = np.bincount(regime_labels(record.epochs), minlength=REGIME_COUNT)
    out(f"  epochs per regime        : {list(map(int, counts))}")
    out.check(
        "the four declared regimes hold a quarter of the record each",
        int(counts.max() - counts.min()) <= REGIME_COUNT,
        f"counts {list(map(int, counts))}",
    )


def marginals(out: Tee, record) -> None:
    """Part 2: the observed cloud rate against the declared marginal."""
    out.rule("2. the observed cloud rate against the declared marginal of equation (6)")
    out("The marginal cloud probability of equation (6) is EXACTLY Phi of the")
    out("threshold, by construction. The criterion below is the declared four")
    out("standard errors at the record's EFFECTIVE sample size, which is")
    out("epochs (1 - phi) / (1 + phi) and not epochs.")
    out("")
    observed = record.cloud.mean(axis=0)
    truth = record.marginal_rate_true()
    effective = record.epochs * (1.0 - record.spec.persistence) / (
        1.0 + record.spec.persistence
    )
    errors = np.sqrt(truth * (1.0 - truth) / effective)
    standardised = (observed - truth) / errors
    rows = [
        [
            record.site_names[index],
            number(observed[index]),
            number(truth[index]),
            f"{observed[index] - truth[index]:+.6f}",
            scientific(errors[index]),
            f"{standardised[index]:+.3f}",
        ]
        for index in range(record.site_count)
    ]
    out(
        format_table(
            ["station", "observed", "declared", "difference", "sd", "sigma"], rows
        )
    )
    out("")
    out(f"  latent persistence phi   : {record.spec.persistence}")
    out(f"  effective sample size    : {effective:.1f} of {record.epochs} epochs")
    out(f"  worst deviation          : {float(np.max(np.abs(standardised))):.4f} sigma")
    out(f"  mean deviation           : {float(np.mean(standardised)):+.4f} sigma")
    out.check(
        "every station's observed cloud rate is within the declared sigma of its "
        "declared marginal",
        float(np.max(np.abs(standardised))) < CONTROL_SIGMA,
        f"worst deviation {float(np.max(np.abs(standardised))):.4f} sigma against "
        f"a criterion of {CONTROL_SIGMA:.1f}",
    )


def spatial(out: Tee, record) -> None:
    """Part 3: the latent spatial correlation against the declared matrix."""
    out.rule("3. the latent spatial correlation against equation (4)")
    out("Measured within each regime, because the correlation matrix differs between")
    out("them. The criterion is four standard errors of a correlation estimated from")
    out("the EFFECTIVE sample size of one regime, which is 1 / sqrt(n_eff).")
    out("")
    rows = []
    worst_overall = 0.0
    for regime in range(REGIME_COUNT):
        mask = record.regime == regime
        measured = np.corrcoef(record.latent[mask], rowvar=False)
        matrix = true_correlation(record.spec, regime, record.sites)
        upper = np.triu_indices(record.site_count, 1)
        deviation = np.abs(measured[upper] - matrix[upper])
        effective = mask.sum() * (1.0 - record.spec.persistence) / (
            1.0 + record.spec.persistence
        )
        error = 1.0 / float(np.sqrt(effective))
        sigma = float(deviation.max()) / error
        worst_overall = max(worst_overall, sigma)
        rows.append(
            [
                str(regime),
                str(int(mask.sum())),
                number(effective, 1),
                scientific(float(deviation.max())),
                scientific(error),
                f"{sigma:.4f}",
            ]
        )
    out(
        format_table(
            ["regime", "epochs", "n_eff", "worst |measured - true|", "sd", "sigma"],
            rows,
        )
    )
    out("")
    out(f"  worst deviation over all regimes : {worst_overall:.4f} sigma")
    out.check(
        "the latent field's spatial correlation matches equation (4) in every regime",
        worst_overall < CONTROL_SIGMA,
        f"worst deviation {worst_overall:.4f} sigma against a criterion of "
        f"{CONTROL_SIGMA:.1f}",
    )


def temporal(out: Tee, record) -> None:
    """Part 4: the latent persistence, and what thresholding does to it."""
    out.rule("4. the latent persistence of equation (5), and the indicator's")
    latent = lag_one_autocorrelation(record.latent)
    indicator = lag_one_autocorrelation(record.cloud)
    out(f"  declared phi             : {record.spec.persistence}")
    out(
        f"  latent lag-1, range      : {float(latent.min()):.6f} to "
        f"{float(latent.max()):.6f}"
    )
    out(f"  latent lag-1, mean       : {float(latent.mean()):.6f}")
    out(f"  indicator lag-1, range   : {float(indicator.min()):.6f} to "
        f"{float(indicator.max()):.6f}")
    out(f"  indicator lag-1, mean    : {float(indicator.mean()):.6f}")
    out("")
    out("Thresholding destroys correlation: the binary indicator's lag-one is below")
    out("the latent field's at every station. Equation (13) uses the INDICATOR's")
    out("value, which inflates fewer standard errors than the latent value would, so")
    out("the choice is conservative in the direction that matters. The gap is")
    out("reported rather than asserted away.")
    out("")
    out.check(
        "the latent lag-one autocorrelation recovers the declared persistence",
        abs(float(latent.mean()) - record.spec.persistence) < 0.01,
        f"mean {float(latent.mean()):.6f} against a declared "
        f"{record.spec.persistence}",
    )
    out.check(
        "the indicator's lag-one is below the latent field's at every station",
        bool(np.all(indicator < latent)),
        f"worst gap {float(np.max(latent - indicator)):.6f}",
    )
    out("")
    out("And the latent field is marginally standard normal, by construction.")
    out("")
    worst_mean = float(np.max(np.abs(record.latent.mean(axis=0))))
    worst_sd = float(np.max(np.abs(record.latent.std(axis=0) - 1.0)))
    out(f"  worst |mean|             : {worst_mean:.6f}")
    out(f"  worst |sd - 1|           : {worst_sd:.6f}")
    out.check(
        "the latent field is marginally standard normal to the record's own "
        "sampling precision",
        worst_mean < 0.05 and worst_sd < 0.05,
        "mean within 0.05 of zero and sd within 0.05 of one at every station",
    )


def threshold_sensitivity(out: Tee, record) -> None:
    """Part 5: what the strict comparison of equation (6) can and cannot promise."""
    out.rule("5. the strict threshold comparison of equation (6)")
    out("The cloud indicator is a strict comparison of two rounded quantities. Two")
    out("builds that agree on the latent field and the threshold to within one unit")
    out("in the last place can still disagree on the indicator wherever the two sit")
    out("within an ulp of each other, and the number of such epochs grows with the")
    out("record. No test or check in this product asserts bit-exact equality of a")
    out("cloud indicator.")
    out("")
    out("Measured below: a perturbation delta applied to the latent field flips")
    out("exactly those indicators whose margin |z - threshold| is inside delta, and")
    out("the cloud RATE moves by far less than the indicator count suggests.")
    out("")
    margin = record.margin_to_threshold()
    base_rate = float(record.cloud.mean())
    rows = []
    worst_outside = 0.0
    for delta in (1e-16, 1e-12, 1e-8, 1e-4):
        perturbed = (record.latent + delta) < record.threshold_true
        flipped = perturbed != record.cloud
        count = int(np.count_nonzero(flipped))
        outside = (
            float(np.max(np.abs(margin[flipped]))) if count else 0.0
        )
        worst_outside = max(worst_outside, outside - delta)
        rows.append(
            [
                scientific(delta, 0),
                str(count),
                scientific(count / margin.size),
                scientific(abs(float(perturbed.mean()) - base_rate)),
                scientific(outside),
            ]
        )
    out(
        format_table(
            [
                "delta",
                "indicators flipped",
                "fraction",
                "rate change",
                "worst flipped margin",
            ],
            rows,
        )
    )
    out("")
    out(f"  record size              : {margin.size} indicators")
    out(f"  base cloud rate          : {base_rate:.6f}")
    out(f"  smallest |margin|        : {float(np.min(np.abs(margin))):.6e}")
    out.check(
        "every flipped indicator had a margin inside the perturbation that flipped it",
        worst_outside <= 0.0,
        f"worst excess {worst_outside:.6e}; an indicator outside delta cannot flip",
    )
    flipped_count = int(
        np.count_nonzero(
            ((record.latent + 1e-4) < record.threshold_true) != record.cloud
        )
    )
    out.control(
        "a perturbation of 1e-04 does flip indicators, so the sensitivity is real",
        fired=flipped_count > 0,
        must_fire=True,
        detail=f"{flipped_count} indicators flipped at delta = 1e-04",
    )


def determinism(out: Tee) -> None:
    """Part 6: the record is a deterministic function of its declared seed."""
    out.rule("6. determinism")
    out("Two calls with the same spec return the same record within one build of")
    out("numpy, which is what makes every figure in this repository reproducible by")
    out("a reader. The claim that does NOT follow, and is not made anywhere, is that")
    out("the INDICATOR is reproducible across builds: see part 5.")
    out("")
    with Timer() as timer:
        first = generate_record(RECORD_SPECS["short"])
    second = generate_record(RECORD_SPECS["short"])
    out(f"  generation time          : {timer.elapsed_s:.3f} s for "
        f"{first.epochs} epochs")
    identical = bool(np.array_equal(first.latent, second.latent))
    out(f"  latent arrays identical  : {identical}")
    out(f"  indicators identical     : {bool(np.array_equal(first.cloud, second.cloud))}")
    out.check(
        "the record is a deterministic function of its declared seed, in this build",
        bool(np.array_equal(first.cloud, second.cloud))
        and bool(np.array_equal(first.latent, second.latent)),
        "both arrays bit-identical across two calls in one process",
    )
    other = generate_record(RECORD_SPECS["isotropic"])
    length = min(first.epochs, other.epochs)
    disagreement = float(np.mean(first.cloud[:length] != other.cloud[:length]))
    out(f"  isotropic record differs on {disagreement:.4f} of indicators")
    out.control(
        "a different declared spec gives a substantially different record",
        fired=disagreement > 0.2,
        must_fire=True,
        detail=f"{disagreement:.4f} of indicators differ",
    )
    _ = norm, SITES, EPOCHS_PER_YEAR


def main() -> int:
    out = Tee("validate_field_output.txt")
    out.rule("sitediverse -- the synthetic generator against its own ground truth")
    for line in environment_lines():
        out(line)
    out("")
    out("THE RECORD IS SYNTHETIC. The point of this script is that the generator's")
    out("own constants are available to be scored against, which is the one thing a")
    out("synthetic record buys and a real one cannot. A generator that reproduces its")
    out("own constants has demonstrated CONSISTENCY and not skill, and nothing here")
    out("transfers to a real cloud field. See DATASET_CARD.md.")
    record = generate_record(RECORD_SPECS["anisotropic"])
    declared(out, record)
    marginals(out, record)
    spatial(out, record)
    temporal(out, record)
    threshold_sensitivity(out, record)
    determinism(out)
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
