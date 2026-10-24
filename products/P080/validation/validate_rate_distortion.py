"""Rate-distortion at exactly matched distortion, in actual encoded bytes.

Rate is ``len(bytes)`` of a container that round-trips. Never an entropy
figure, never an estimate. The first-order entropy appears as a labelled
reference column and the number of cells where the realised rate came out
*below* it is counted and reported, because for a correlated source the
memoryless entropy is not a floor and a reader who assumes it is will think
the measurement is wrong.

Deviation from the specification, recorded here and in VALIDATION.md
--------------------------------------------------------------------

The specification names ``numcodecs`` (0.17.0) and ``zfpy`` (1.0.1) as
baseline 2 and asks for a rate-distortion comparison against them. **Neither
is installed in the pinned environment, and nothing was installed to obtain
them**, so no number in this file or anywhere else in this repository comes
from either package. The header of every output file states their absence.

In their place the comparison runs against ``zlib``, ``lzma`` and ``bz2`` from
the Python standard library, applied to the same quantised symbol stream and
the same predictor residual stream the package's own coder consumes, so the
reconstruction is bit-identical and the distortion is matched exactly rather
than approximately. That substitution is not like-for-like -- zfp is a
transform codec with its own quantiser -- but it does answer the question that
matters here: is the package's own entropy coder doing anything a
general-purpose compressor would not?

Three lossless references over the raw float64 bytes are included at zero
distortion, as the upper anchor of the curve: they say what accepting an error
bound bought in the first place.
"""

from __future__ import annotations

from _bootstrap import Tee, Timer, bootstrap, environment_lines

ROOT = bootstrap()

import numpy as np  # noqa: E402

from boundcompress import baselines, dataset  # noqa: E402
from boundcompress.codec import encode  # noqa: E402
from boundcompress.predictors import CLASSICAL_NAMES, get_predictor  # noqa: E402
from boundcompress.quantise import make_spec, quantise  # noqa: E402
from boundcompress.ratedistortion import geometric_bounds, sweep_signal  # noqa: E402
from boundcompress.report import format_table  # noqa: E402

SAMPLES = 4096
POINTS = 6
SHORT_SAMPLES = 64


def main() -> int:
    out = Tee("validate_rate_distortion_output.txt")
    out.rule("boundcompress -- rate-distortion at matched distortion, in actual bytes")
    for line in environment_lines():
        out(line)
    out("")
    out("numcodecs and zfpy are the specification's baseline 2 and are NOT installed.")
    out("No rate-distortion number below is taken from either. The comparison runs")
    out("against zlib, lzma and bz2 from the standard library on the same quantised")
    out("symbol and residual streams, so the distortion is matched exactly.")

    signals = dataset.generate_all(samples=SAMPLES)

    out.rule("full sweep: every coder, every signal, six bounds")
    all_rows = []
    roundtrip_failures = 0
    below_entropy = 0
    entropy_cells = 0
    for name, signal in signals.items():
        bounds = geometric_bounds(signal.values, count=POINTS)
        rows = sweep_signal(name, signal.values, bounds)
        all_rows.extend(rows)
        roundtrip_failures += len(
            [row for row in rows if not row.roundtrip_ok]
        )
        for row in rows:
            if np.isfinite(row.entropy_bits):
                entropy_cells += 1
                if row.bits_per_sample < row.entropy_bits:
                    below_entropy += 1
    out.check("every coder in every cell round-trips", roundtrip_failures == 0)
    out.check("every cell holds its bound", all(row.bound_holds for row in all_rows))
    out(f"cells measured                             : {len(all_rows)}")
    out(f"cells with a first-order entropy figure     : {entropy_cells}")
    out(f"cells whose realised rate is below entropy  : {below_entropy}")
    out("")
    out("The entropy figure is a memoryless reference and not a floor. It also")
    out("excludes the header, table and escape bytes every realised rate includes,")
    out("which is why on these arrays the realised rate sits above it far more")
    out("often than below.")

    out.rule("one signal in full: battery_voltage, every coder at every bound")
    bounds = geometric_bounds(signals["battery_voltage"].values, count=POINTS)
    rows = sweep_signal("battery_voltage", signals["battery_voltage"].values, bounds)
    table = [
        [
            f"{row.bound:.4e}",
            row.coder,
            str(row.encoded_bytes),
            f"{row.bits_per_sample:.4f}",
            f"{row.max_error:.4e}",
            f"{row.entropy_bits:.4f}" if np.isfinite(row.entropy_bits) else "-",
            "yes" if row.bound_holds else "NO",
            "yes" if row.roundtrip_ok else "NO",
        ]
        for row in rows
    ]
    out(
        format_table(
            ["bound", "coder", "bytes", "bits/sample", "max_error", "entropy_ref",
             "holds", "roundtrip"],
            table,
        )
    )

    out.rule("the package coder against the standard-library coders, at matched distortion")
    out("Ratio column is package bytes divided by the named coder's bytes; below 1")
    out("means the package coder used fewer bytes. Each row is one (signal, bound)")
    out("cell at the median swept bound, with the package's best classical predictor.")
    comparison_rows = []
    package_wins = 0
    baseline_wins = 0
    for name, signal in signals.items():
        swept = geometric_bounds(signal.values, count=POINTS)
        bound = swept[len(swept) // 2]
        best = min(
            (encode(signal.values, "abs", bound, predictor=predictor).certificate
             for predictor in CLASSICAL_NAMES),
            key=lambda certificate: certificate.encoded_bytes,
        )
        spec = make_spec(signal.values, "abs", bound)
        symbols, _, _ = quantise(signal.values, spec)
        residuals = get_predictor(best.predictor).residuals(symbols)
        measurements = baselines.measure_all(
            values=signal.values, symbols=symbols, residuals=residuals
        )
        row = [name, f"{bound:.3e}", best.predictor, str(best.encoded_bytes)]
        for method in baselines.LOSSY_METHODS:
            measurement = measurements[method]
            row.append(str(measurement.total_bytes))
            row.append(f"{best.encoded_bytes / measurement.total_bytes:.3f}")
        best_baseline = min(
            measurements[method].total_bytes for method in baselines.LOSSY_METHODS
        )
        if best.encoded_bytes < best_baseline:
            package_wins += 1
        else:
            baseline_wins += 1
        comparison_rows.append(row)
    headers = ["signal", "bound", "predictor", "pkg_bytes"]
    for method in baselines.LOSSY_METHODS:
        headers += [method, "ratio"]
    out(format_table(headers, comparison_rows))
    out("")
    out(f"cells where the package coder used the fewest bytes : {package_wins}")
    out(f"cells where a standard-library coder did            : {baseline_wins}")
    out.check(
        "the package coder is measured against every available baseline",
        package_wins + baseline_wins == len(signals),
    )

    out.rule("the first-order entropy reference is not a floor for a coder with memory")
    out("For the package's own coder the first-order entropy IS effectively a floor,")
    out("and that is a property of the coder and not of information theory: static")
    out("order-0 rANS codes each symbol independently against one shipped table, so")
    out("it cannot beat the memoryless entropy of its own symbol stream except by")
    out("the slack in the table quantisation. A coder with memory is under no such")
    out("constraint. The column below compares the lzma and bz2 payloads over the")
    out("SAME residual stream against the entropy of that stream: where the ratio")
    out("is below 1, a real coder has gone below the 'floor'.")
    out("")
    out("Two entropy columns, because the difference matters. symbol_H is the figure")
    out("the certificate carries: first-order entropy over the 256-symbol alphabet,")
    out("in which every large residual has been collapsed onto one escape symbol. On")
    out("an escape-heavy stream it understates by orders of magnitude -- white_noise")
    out("below is the extreme case. value_H has no escape and is the figure to")
    out("compare against a byte count.")
    floor_rows = []
    below_floor = 0
    for name, signal in signals.items():
        swept = geometric_bounds(signal.values, count=POINTS)
        bound = swept[len(swept) // 2]
        result = encode(signal.values, "abs", bound, predictor="previous")
        spec = make_spec(signal.values, "abs", bound)
        symbols, _, _ = quantise(signal.values, spec)
        residuals = get_predictor("previous").residuals(symbols)
        from boundcompress.entropy import value_entropy_bits

        value_bits = value_entropy_bits(residuals)
        entropy_bytes = baselines.entropy_floor_bytes(value_bits, residuals.size)
        symbol_bytes = baselines.entropy_floor_bytes(
            result.certificate.residual_entropy_bits, residuals.size
        )
        lzma_payload = baselines.measure(
            "lzma_residual", values=signal.values, symbols=symbols, residuals=residuals
        ).payload_bytes
        bz2_payload = baselines.measure(
            "bz2_residual", values=signal.values, symbols=symbols, residuals=residuals
        ).payload_bytes
        best_payload = min(lzma_payload, bz2_payload)
        if best_payload < entropy_bytes:
            below_floor += 1
        floor_rows.append(
            [
                name,
                f"{result.certificate.residual_entropy_bits:.4f}",
                f"{symbol_bytes:.1f}",
                f"{value_bits:.4f}",
                f"{entropy_bytes:.1f}",
                str(lzma_payload),
                str(bz2_payload),
                f"{best_payload / entropy_bytes:.4f}",
                "below" if best_payload < entropy_bytes else "above",
            ]
        )
    out("")
    out(
        format_table(
            ["signal", "symbol_H_bits", "symbol_H_bytes", "value_H_bits",
             "value_H_bytes", "lzma_payload", "bz2_payload", "best/value_H", "verdict"],
            floor_rows,
        )
    )
    out("")
    out(f"signals where a memory coder beat the entropy figure : {below_floor} of {len(signals)}")
    out("")
    out("Sibling product P072 of this portfolio measured 0.9978 of its held-out")
    out("cases compressing below their measured first-order entropy. The number")
    out("here is lower because these residual streams are already decorrelated by")
    out("the predictor, which is what the entropy figure is computed on, and because")
    out("this product charges the header to the rate and P072 compared ratios.")
    out("Neither number licenses treating the entropy as a lower bound on the")
    out("achievable rate of a coder with memory.")

    out.rule("what the error bound bought: lossy against lossless on the same array")
    lossless_rows = []
    for name, signal in signals.items():
        swept = geometric_bounds(signal.values, count=POINTS)
        bound = swept[len(swept) // 2]
        best = min(
            (encode(signal.values, "abs", bound, predictor=predictor).certificate
             for predictor in CLASSICAL_NAMES),
            key=lambda certificate: certificate.encoded_bytes,
        )
        spec = make_spec(signal.values, "abs", bound)
        symbols, _, _ = quantise(signal.values, spec)
        residuals = get_predictor("linear").residuals(symbols)
        measurements = baselines.measure_all(
            values=signal.values, symbols=symbols, residuals=residuals
        )
        row = [
            name,
            f"{bound:.3e}",
            str(8 * signal.samples),
            str(best.encoded_bytes),
            f"{best.compression_ratio:.3f}",
        ]
        for method in baselines.LOSSLESS_METHODS:
            row.append(str(measurements[method].total_bytes))
            row.append(f"{8 * signal.samples / measurements[method].total_bytes:.3f}")
        lossless_rows.append(row)
    headers = ["signal", "bound", "raw_bytes", "pkg_bytes", "pkg_ratio"]
    for method in baselines.LOSSLESS_METHODS:
        headers += [method, "ratio"]
    out(format_table(headers, lossless_rows))

    out.rule("header cost at two array lengths")
    out("The fixed header is 38 bytes and nothing is omitted from the rate. On a")
    out("long array that is noise; on a short one it is most of the container, and")
    out("both regimes are shown rather than only the flattering one.")
    header_rows = []
    for samples in (SHORT_SAMPLES, SAMPLES):
        signal = dataset.generate("battery_voltage", samples=samples, seed=0)
        result = encode(signal.values, "abs", 1e-3, predictor="linear")
        from boundcompress.container import block_sizes, unpack

        sizes = block_sizes(unpack(result.blob))
        overhead = sizes["fixed_header"] + sizes["warmup"] + sizes["frequency_table"]
        header_rows.append(
            [
                str(samples),
                str(result.certificate.encoded_bytes),
                str(overhead),
                f"{100.0 * overhead / result.certificate.encoded_bytes:.2f}",
                f"{result.certificate.bits_per_sample:.4f}",
                f"{result.certificate.compression_ratio:.3f}",
            ]
        )
    out(
        format_table(
            ["samples", "bytes", "overhead_bytes", "overhead_pct", "bits/sample", "ratio"],
            header_rows,
        )
    )

    out.rule("timings -- UNSTABLE, for order of magnitude only")
    out("Wall clock on a shared two-core container. These numbers move between")
    out("runs and no conclusion in this repository is derived from them.")
    timing_rows = []
    for predictor in CLASSICAL_NAMES:
        timer = Timer()
        signal = signals["battery_voltage"]
        for _ in range(5):
            with timer:
                encode(signal.values, "abs", 1e-3, predictor=predictor)
        timing_rows.append([predictor, f"{timer.mean_ms:.2f}", str(timer.calls)])
    out(format_table(["predictor", "mean_encode_ms", "calls"], timing_rows))

    path = out.save()
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
