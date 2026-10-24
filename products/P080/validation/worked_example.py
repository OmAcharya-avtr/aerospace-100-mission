"""The worked example quoted in README.md, run so the quoted output cannot drift.

One array, one declared bound, one certificate, one rate-distortion comparison
at matched distortion. Everything printed here appears verbatim in the
README's "A worked example" section, and this script is re-run as the last
step before hand-off so the two cannot disagree.
"""

from __future__ import annotations

from _bootstrap import Tee, bootstrap, environment_lines

ROOT = bootstrap()

import numpy as np  # noqa: E402

from boundcompress import baselines, dataset  # noqa: E402
from boundcompress.certificate import certify  # noqa: E402
from boundcompress.codec import decode, encode  # noqa: E402
from boundcompress.predictors import CLASSICAL_NAMES, get_predictor  # noqa: E402
from boundcompress.quantise import make_spec, quantise  # noqa: E402
from boundcompress.report import format_certificate, format_table  # noqa: E402


def main() -> int:
    out = Tee("worked_example_output.txt")
    out.rule("boundcompress -- worked example")
    for line in environment_lines():
        out(line)

    signal = dataset.generate("battery_voltage", samples=4096, seed=0)
    bound = 5e-4

    out.rule("the array and the declared bound")
    out(f"signal          : {signal.name}")
    out(f"samples         : {signal.samples}")
    out(f"unit            : {signal.unit}")
    out(f"range           : {signal.values.min():.6f} to {signal.values.max():.6f}")
    out(f"declared bound  : abs {bound:.6e} {signal.unit}")

    out.rule("encode, decode, certify")
    result = encode(signal.values, "abs", bound, predictor="linear")
    decoded = decode(result.blob)
    out(format_certificate(result.certificate))
    out("")
    out(f"decoded equals the encoder's reconstruction : "
        f"{bool(np.array_equal(decoded, result.reconstruction))}")
    out(f"maximum error of the DECODED array          : "
        f"{float(np.max(np.abs(signal.values - decoded))):.6e}")
    out(f"declared bound                              : {bound:.6e}")
    out(f"samples outside the bound                   : "
        f"{int(np.count_nonzero(np.abs(signal.values - decoded) > bound))}")

    out.rule("the same bound with every predictor, at identical distortion")
    rows = []
    spec = make_spec(signal.values, "abs", bound)
    symbols, reconstruction, corrections = quantise(signal.values, spec)
    for predictor in CLASSICAL_NAMES:
        certificate = encode(signal.values, "abs", bound, predictor=predictor).certificate
        rows.append(
            [
                predictor,
                str(certificate.encoded_bytes),
                f"{certificate.bits_per_sample:.4f}",
                f"{certificate.compression_ratio:.3f}",
                f"{certificate.max_absolute_error:.6e}",
                str(certificate.escape_count),
            ]
        )
    out(
        format_table(
            ["predictor", "bytes", "bits/sample", "ratio", "max_abs_error", "escapes"],
            rows,
        )
    )
    out("")
    out("The max_abs_error column is one number repeated, not four close numbers.")
    out("Everything after the quantiser is lossless on the integer symbols, so the")
    out("predictor moves the rate and cannot move the error.")

    out.rule("against the standard-library coders on the same symbol stream")
    out("numcodecs and zfpy are the specification's baseline 2 and are not installed")
    out("in this environment; no number here comes from either.")
    residuals = get_predictor("linear").residuals(symbols)
    measurements = baselines.measure_all(
        values=signal.values, symbols=symbols, residuals=residuals
    )
    baseline_rows = [
        [
            "boundcompress (rANS, linear)",
            str(result.certificate.encoded_bytes),
            f"{result.certificate.bits_per_sample:.4f}",
            f"{result.certificate.max_absolute_error:.6e}",
            "yes",
        ]
    ]
    for method in baselines.LOSSY_METHODS:
        measurement = measurements[method]
        baseline_rows.append(
            [
                method,
                str(measurement.total_bytes),
                f"{measurement.bits_per_sample:.4f}",
                f"{result.certificate.max_absolute_error:.6e}",
                "yes" if measurement.roundtrip_ok else "NO",
            ]
        )
    for method in baselines.LOSSLESS_METHODS:
        measurement = measurements[method]
        baseline_rows.append(
            [
                method + " (lossless)",
                str(measurement.total_bytes),
                f"{measurement.bits_per_sample:.4f}",
                "0.000000e+00",
                "yes" if measurement.roundtrip_ok else "NO",
            ]
        )
    out(
        format_table(
            ["coder", "bytes", "bits/sample", "max_abs_error", "roundtrip"],
            baseline_rows,
        )
    )

    out.rule("the negative control: a deliberate violation the certificate must catch")
    corrupted = result.reconstruction.copy()
    index = 2048
    corrupted[index] = signal.values[index] + 3.0 * bound
    failed = certify(
        signal.values,
        corrupted,
        "abs",
        bound,
        predictor="injected-violation",
        notes=(f"sample {index} set to the true value plus 3 x the declared bound",),
    )
    out(format_certificate(failed))
    out("")
    out(f"certificate caught the injected violation : {failed.bound_holds is False}")
    out(f"and named the right sample                : "
        f"{index in failed.violation_indices}")

    out.rule("what the guarantee cost")
    lossless_best = min(
        measurements[method].total_bytes for method in baselines.LOSSLESS_METHODS
    )
    out(f"raw float64 bytes                          : {8 * signal.samples}")
    out(f"best lossless, zero distortion             : {lossless_best}")
    out(f"boundcompress at {bound:.0e} absolute            : "
        f"{result.certificate.encoded_bytes}")
    out(f"rounding corrections made by the encoder   : {corrections}")
    out(f"residual entropy reference, bits/symbol    : "
        f"{result.certificate.residual_entropy_bits:.6f}")
    out(f"realised rate, bits/sample                 : "
        f"{result.certificate.bits_per_sample:.6f}")
    out("")
    out("The entropy figure is a memoryless reference and not a floor; see")
    out("validation/validate_rate_distortion_output.txt, where a coder with memory")
    out("goes below it on three of eight signals.")

    path = out.save()
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
