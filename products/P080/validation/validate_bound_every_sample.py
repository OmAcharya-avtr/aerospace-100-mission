"""Assert the declared error bound on every sample of every test array.

The central claim of this product, measured rather than argued. For every
built-in signal, every declared bound mode, every predictor and every bound in
the sweep, the per-sample error of the decoded array is compared to the
declared bound with a non-strict inequality and no tolerance on either side,
and the number of individual comparisons is counted and reported.

Reported, not summarised: the maximum error, the index where it occurred, the
margin to the bound, and the number of samples whose nearest integer had to be
moved one step because floating-point rounding put it outside the bound.
"""

from __future__ import annotations

from _bootstrap import Tee, bootstrap, environment_lines

ROOT = bootstrap()

import numpy as np  # noqa: E402

from boundcompress import dataset  # noqa: E402
from boundcompress.codec import decode, encode  # noqa: E402
from boundcompress.predictors import CLASSICAL_NAMES  # noqa: E402
from boundcompress.ratedistortion import geometric_bounds  # noqa: E402
from boundcompress.report import format_table  # noqa: E402

SAMPLES = 4096
RELATIVE_BOUNDS = (1e-2, 1e-4, 1e-6)


def main() -> int:
    out = Tee("validate_bound_every_sample_output.txt")
    out.rule("boundcompress -- the declared bound, asserted on every sample")
    for line in environment_lines():
        out(line)

    signals = dataset.generate_all(samples=SAMPLES)
    models = None
    comparisons = 0
    worst_margin = float("inf")
    worst_row: list[str] = []
    rows: list[list[str]] = []
    corrections_total = 0

    out.rule("mode='abs' -- every signal, every classical predictor, eight bounds")
    for name, signal in signals.items():
        bounds = geometric_bounds(signal.values, count=8)
        for bound in bounds:
            for predictor in CLASSICAL_NAMES:
                result = encode(signal.values, "abs", bound, predictor=predictor)
                decoded = decode(result.blob)
                error = np.abs(signal.values - decoded)
                inside = error <= bound
                comparisons += int(error.size)
                certificate = result.certificate
                corrections_total += certificate.corrections
                ok = bool(np.all(inside)) and certificate.bound_holds
                if not ok:
                    bad = int(np.flatnonzero(~inside)[0])
                    out.check(
                        f"abs {name} {predictor} bound={bound:.6e}",
                        False,
                        f"sample {bad} error {float(error[bad]):.6e}",
                    )
                if certificate.margin < worst_margin:
                    worst_margin = certificate.margin
                    worst_row = [
                        name,
                        predictor,
                        f"{bound:.6e}",
                        f"{certificate.max_absolute_error:.6e}",
                        str(certificate.argmax_index),
                        f"{certificate.margin:.6e}",
                    ]
            certificate = encode(signal.values, "abs", bound, predictor="linear").certificate
            rows.append(
                [
                    name,
                    f"{bound:.6e}",
                    f"{certificate.max_absolute_error:.6e}",
                    str(certificate.argmax_index),
                    f"{certificate.margin:.6e}",
                    str(certificate.corrections),
                    "HOLDS" if certificate.bound_holds else "FAILED",
                ]
            )
    out.check("every abs cell holds the bound on every sample", not out.failures)
    out("")
    out("per-signal worst case at each bound, predictor='linear':")
    out(
        format_table(
            ["signal", "bound", "max_abs_error", "argmax", "margin", "corrections",
             "verdict"],
            rows,
        )
    )

    out.rule("mode='rel' -- pointwise relative bound on the strictly positive signals")
    relative_rows: list[list[str]] = []
    rel_failures = 0
    for name in dataset.POSITIVE_SIGNALS:
        values = signals[name].values
        for bound in RELATIVE_BOUNDS:
            for predictor in CLASSICAL_NAMES:
                result = encode(values, "rel", bound, predictor=predictor)
                decoded = decode(result.blob)
                error = np.abs(values - decoded)
                inside = error <= bound * np.abs(values)
                comparisons += int(error.size)
                if not bool(np.all(inside)) or not result.certificate.bound_holds:
                    rel_failures += 1
                    bad = int(np.flatnonzero(~inside)[0])
                    out.check(
                        f"rel {name} {predictor} bound={bound:.6e}",
                        False,
                        f"sample {bad} relative error "
                        f"{float(error[bad] / abs(values[bad])):.6e}",
                    )
            certificate = encode(values, "rel", bound, predictor="linear").certificate
            relative_rows.append(
                [
                    name,
                    f"{bound:.6e}",
                    f"{certificate.max_relative_error:.6e}",
                    str(certificate.argmax_index),
                    f"{certificate.margin:.6e}",
                    str(certificate.corrections),
                    "HOLDS" if certificate.bound_holds else "FAILED",
                ]
            )
            corrections_total += certificate.corrections
    out.check("every rel cell holds the pointwise bound on every sample", rel_failures == 0)
    out("")
    out(
        format_table(
            ["signal", "rel_bound", "max_rel_error", "argmax", "margin", "corrections",
             "verdict"],
            relative_rows,
        )
    )

    out.rule("mode='range_rel' -- derived absolute bound, every signal")
    range_rows: list[list[str]] = []
    range_failures = 0
    for name, signal in signals.items():
        span = float(signal.values.max() - signal.values.min())
        for fraction in (1e-2, 1e-4, 1e-6):
            result = encode(signal.values, "range_rel", fraction, predictor="linear")
            decoded = decode(result.blob)
            error = np.abs(signal.values - decoded)
            limit = fraction * span
            inside = error <= limit
            comparisons += int(error.size)
            if not bool(np.all(inside)):
                range_failures += 1
            range_rows.append(
                [
                    name,
                    f"{fraction:.1e}",
                    f"{limit:.6e}",
                    f"{float(error.max()):.6e}",
                    "HOLDS" if result.certificate.bound_holds else "FAILED",
                ]
            )
    out.check("every range_rel cell holds the derived bound", range_failures == 0)
    out("")
    out(
        format_table(
            ["signal", "fraction", "derived_abs_bound", "max_abs_error", "verdict"],
            range_rows,
        )
    )

    out.rule("the learned predictor cannot widen the bound")
    from _bootstrap import fit_models

    models = fit_models()
    learned_failures = 0
    learned_rows: list[list[str]] = []
    for label, model in models.items():
        for name, signal in signals.items():
            span = float(signal.values.max() - signal.values.min())
            bound = 1e-4 * span
            result = encode(
                signal.values, "abs", bound, predictor="learned", model=model
            )
            decoded = decode(result.blob, model=model)
            error = np.abs(signal.values - decoded)
            comparisons += int(error.size)
            ok = bool(np.all(error <= bound)) and result.certificate.bound_holds
            if not ok:
                learned_failures += 1
            learned_rows.append(
                [
                    f"learned_{label}",
                    name,
                    f"{bound:.6e}",
                    f"{float(error.max()):.6e}",
                    str(result.certificate.encoded_bytes),
                    "HOLDS" if ok else "FAILED",
                ]
            )
    out.check("every learned cell holds the bound on every sample", learned_failures == 0)
    out("")
    out(
        format_table(
            ["predictor", "signal", "bound", "max_abs_error", "bytes", "verdict"],
            learned_rows,
        )
    )

    out.rule("totals")
    out(f"individual sample-versus-bound comparisons : {comparisons}")
    out(f"samples moved one step by rounding         : {corrections_total}")
    out(f"tightest margin to the bound observed      : {worst_margin:.6e}")
    out(f"tightest-margin cell                       : {worst_row}")
    out.check("the tightest observed margin is non-negative", worst_margin >= 0.0)
    out.check("at least one million comparisons were made", comparisons >= 1_000_000,
              f"{comparisons}")

    path = out.save()
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
