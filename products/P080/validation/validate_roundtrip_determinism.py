"""Round-trip reconstruction and determinism, measured on committed bytes.

Two properties, both asserted rather than described:

**Round-trip.** For every signal, mode and predictor, the decoded array equals
the encoder's own reconstruction bit-for-bit, and the integer symbol stream
recovered from the container equals the stream the encoder produced. The
equality is exact and that is the honest assertion: the decoder recomputes one
multiplication, ``symbols * step``, from the same int64 symbols and the same
float64 step carried in the header. What is guaranteed is the equality of that
single multiplication, not equality through a pipeline.

**Determinism.** Three independent encodes of the same array with the same
settings must produce the same container, asserted on the SHA-256 of the
bytes. The learned predictors are included, because they are the part of the
pipeline where determinism is arranged rather than free: the encoder evaluates
the ensemble forward pass over the whole array in one batch and the decoder
evaluates it one sample at a time, and the forward pass is hand-written in
elementwise numpy so the two cannot disagree in the last ulp and flip a
rounding.
"""

from __future__ import annotations

from _bootstrap import Tee, bootstrap, environment_lines, fit_models

ROOT = bootstrap()

import numpy as np  # noqa: E402

from boundcompress import dataset  # noqa: E402
from boundcompress.certificate import digest_bytes  # noqa: E402
from boundcompress.codec import decode, encode  # noqa: E402
from boundcompress.container import block_sizes, pack, unpack  # noqa: E402
from boundcompress.predictors import CLASSICAL_NAMES  # noqa: E402
from boundcompress.quantise import make_spec, quantise  # noqa: E402
from boundcompress.ratedistortion import residuals_from_container  # noqa: E402
from boundcompress.report import format_table  # noqa: E402

SAMPLES = 4096
REPEATS = 3


def main() -> int:
    out = Tee("validate_roundtrip_determinism_output.txt")
    out.rule("boundcompress -- round-trip reconstruction and determinism")
    for line in environment_lines():
        out(line)

    signals = dataset.generate_all(samples=SAMPLES)
    models = fit_models()

    out.rule("round-trip: decoded array equals the encoder's reconstruction, bit-for-bit")
    rows: list[list[str]] = []
    mismatches = 0
    symbol_mismatches = 0
    for name, signal in signals.items():
        span = float(signal.values.max() - signal.values.min())
        bound = 1e-4 * span
        spec = make_spec(signal.values, "abs", bound)
        symbols, _, _ = quantise(signal.values, spec)
        for predictor in CLASSICAL_NAMES:
            result = encode(signal.values, "abs", bound, predictor=predictor)
            decoded = decode(result.blob)
            identical = bool(np.array_equal(decoded, result.reconstruction))
            if not identical:
                mismatches += 1
            residuals = residuals_from_container(result.blob)
            from boundcompress.predictors import get_predictor

            engine = get_predictor(predictor)
            rebuilt = engine.reconstruct(
                symbols[: engine.order].tolist(), residuals
            )
            symbols_ok = bool(np.array_equal(rebuilt, symbols))
            if not symbols_ok:
                symbol_mismatches += 1
            rows.append(
                [
                    name,
                    predictor,
                    str(result.certificate.encoded_bytes),
                    "exact" if identical else "DIFFERS",
                    "exact" if symbols_ok else "DIFFERS",
                    f"{float(np.max(np.abs(signal.values - decoded))):.6e}",
                ]
            )
    out.check("every decoded array is bit-identical to its reconstruction", mismatches == 0)
    out.check(
        "every symbol stream recovered from the container matches the encoder",
        symbol_mismatches == 0,
    )
    out("")
    out(
        format_table(
            ["signal", "predictor", "bytes", "array", "symbols", "max_abs_error"], rows
        )
    )

    out.rule("round-trip in every declared bound mode")
    mode_rows: list[list[str]] = []
    mode_failures = 0
    for name, signal in signals.items():
        span = float(signal.values.max() - signal.values.min())
        for mode, bound in (("abs", 1e-4 * span), ("range_rel", 1e-4)):
            result = encode(signal.values, mode, bound, predictor="linear")
            ok = bool(np.array_equal(decode(result.blob), result.reconstruction))
            mode_failures += 0 if ok else 1
            mode_rows.append(
                [name, mode, f"{bound:.6e}", str(result.certificate.encoded_bytes),
                 "exact" if ok else "DIFFERS"]
            )
    for name in dataset.POSITIVE_SIGNALS:
        result = encode(signals[name].values, "rel", 1e-5, predictor="linear")
        ok = bool(np.array_equal(decode(result.blob), result.reconstruction))
        mode_failures += 0 if ok else 1
        mode_rows.append(
            [name, "rel", "1.000000e-05", str(result.certificate.encoded_bytes),
             "exact" if ok else "DIFFERS"]
        )
    out.check("every mode round-trips exactly", mode_failures == 0)
    out("")
    out(format_table(["signal", "mode", "bound", "bytes", "array"], mode_rows))

    out.rule("determinism: three encodes, one digest")
    digest_rows: list[list[str]] = []
    digest_failures = 0
    for name, signal in signals.items():
        span = float(signal.values.max() - signal.values.min())
        bound = 1e-4 * span
        for predictor in ("previous", "linear"):
            digests = {
                digest_bytes(
                    encode(signal.values, "abs", bound, predictor=predictor).blob
                )
                for _ in range(REPEATS)
            }
            stable = len(digests) == 1
            digest_failures += 0 if stable else 1
            digest_rows.append(
                [name, predictor, next(iter(digests))[:16], "stable" if stable else "VARIES"]
            )
    for label, model in models.items():
        for name in ("gyro_rate", "wheel_rate", "battery_voltage"):
            signal = signals[name]
            span = float(signal.values.max() - signal.values.min())
            digests = {
                digest_bytes(
                    encode(
                        signal.values, "abs", 1e-4 * span, predictor="learned", model=model
                    ).blob
                )
                for _ in range(REPEATS)
            }
            stable = len(digests) == 1
            digest_failures += 0 if stable else 1
            digest_rows.append(
                [name, f"learned_{label}", next(iter(digests))[:16],
                 "stable" if stable else "VARIES"]
            )
    out.check("every digest is stable across three encodes", digest_failures == 0)
    out("")
    out(format_table(["signal", "predictor", "sha256[:16]", "verdict"], digest_rows))

    out.rule("the encoder's batch forward pass equals the decoder's sequential one")
    from boundcompress.learned import WINDOW

    agreement_rows: list[list[str]] = []
    agreement_failures = 0
    for label, model in models.items():
        signal = signals["gyro_rate"]
        spec = make_spec(signal.values, "abs", 1e-6)
        symbols, _, _ = quantise(signal.values, spec)
        batch, _ = model.predict_batch(symbols)
        history = symbols[:WINDOW].tolist()
        sequential: list[int] = []
        for index in range(WINDOW, symbols.size):
            sequential.append(model.predict_one(history))
            history.append(int(symbols[index]))
        identical = batch.tolist() == sequential
        agreement_failures += 0 if identical else 1
        differences = int(np.count_nonzero(batch - np.asarray(sequential, dtype=np.int64)))
        agreement_rows.append(
            [f"learned_{label}", str(batch.size), str(differences),
             "identical" if identical else "DIFFERS"]
        )
    out.check(
        "the batch and sequential learned forward passes are bit-identical",
        agreement_failures == 0,
    )
    out("")
    out(format_table(["model", "predictions", "differences", "verdict"], agreement_rows))

    out.rule("container accounting: the blocks sum to the rate")
    accounting_rows: list[list[str]] = []
    accounting_failures = 0
    for name in ("battery_voltage", "white_noise", "packet_counter"):
        signal = signals[name]
        span = float(signal.values.max() - signal.values.min())
        result = encode(signal.values, "abs", 1e-4 * span, predictor="linear")
        container = unpack(result.blob)
        sizes = block_sizes(container)
        total = sum(sizes.values())
        ok = total == len(result.blob) == result.certificate.encoded_bytes
        ok = ok and pack(container) == result.blob
        accounting_failures += 0 if ok else 1
        accounting_rows.append(
            [
                name,
                str(sizes["fixed_header"]),
                str(sizes["warmup"]),
                str(sizes["frequency_table"]),
                str(sizes["escapes"]),
                str(sizes["payload"]),
                str(total),
                str(result.certificate.encoded_bytes),
                "ok" if ok else "MISMATCH",
            ]
        )
    out.check(
        "every container's blocks sum to its length and repack identically",
        accounting_failures == 0,
    )
    out("")
    out(
        format_table(
            ["signal", "header", "warmup", "table", "escapes", "payload", "sum",
             "certificate", "verdict"],
            accounting_rows,
        )
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
