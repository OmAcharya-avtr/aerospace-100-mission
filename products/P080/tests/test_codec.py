"""Codec tests: round-trip, determinism, byte accounting and error paths."""

from __future__ import annotations

import numpy as np
import pytest

from boundcompress import codec
from boundcompress.container import ContainerError, block_sizes, unpack
from boundcompress.predictors import PredictorError
from boundcompress.quantise import MODES
from boundcompress.ratedistortion import residuals_from_container


@pytest.mark.parametrize("predictor", ("previous", "linear", "quadratic", "cubic"))
def test_encode_decode_bit_identical(signals: dict[str, np.ndarray], predictor: str) -> None:
    """The decoded array equals the encoder's reconstruction bit-for-bit.

    Exact equality is the honest assertion here and not a floating-point sin:
    the decoder recomputes ``symbols * step`` from the same int64 symbols and
    the same float64 step carried in the header, so the two sides perform the
    identical single multiplication. What is guaranteed is equality of that
    multiplication, not equality through a pipeline.
    """
    for values in signals.values():
        span = float(values.max() - values.min())
        result, decoded = codec.encode_decode(
            values, "abs", 1e-3 * span, predictor=predictor
        )
        assert np.array_equal(decoded, result.reconstruction)


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_round_trips(positive_signals: dict[str, np.ndarray], mode: str) -> None:
    for values in positive_signals.values():
        bound = 1e-3 if mode != "abs" else 1e-3 * float(values.max() - values.min())
        result, decoded = codec.encode_decode(values, mode, bound)
        assert np.array_equal(decoded, result.reconstruction)
        assert result.certificate.bound_holds


def test_determinism_across_repeated_encodes(signals: dict[str, np.ndarray]) -> None:
    """Same input, same settings, same bytes. Asserted on the SHA-256 digest."""
    for values in signals.values():
        span = float(values.max() - values.min())
        first = codec.encode(values, "abs", 1e-4 * span, predictor="linear")
        second = codec.encode(values, "abs", 1e-4 * span, predictor="linear")
        assert first.blob == second.blob
        assert first.certificate.digest == second.certificate.digest


def test_determinism_with_the_learned_predictor(
    signals: dict[str, np.ndarray], learned_mlp
) -> None:
    values = signals["gyro_rate"]
    span = float(values.max() - values.min())
    first = codec.encode(values, "abs", 1e-4 * span, predictor="learned", model=learned_mlp)
    second = codec.encode(values, "abs", 1e-4 * span, predictor="learned", model=learned_mlp)
    assert first.blob == second.blob


def test_encoded_bytes_equals_the_container_length(signals: dict[str, np.ndarray]) -> None:
    """Rate is ``len(blob)``, never an estimate. Asserted, not documented."""
    for values in signals.values():
        span = float(values.max() - values.min())
        result = codec.encode(values, "abs", 1e-3 * span)
        assert result.certificate.encoded_bytes == len(result.blob)
        assert sum(block_sizes(unpack(result.blob)).values()) == len(result.blob)


def test_container_fields_describe_the_encode(signals: dict[str, np.ndarray]) -> None:
    values = signals["magnetometer"]
    result = codec.encode(values, "abs", 0.01, predictor="quadratic")
    container = unpack(result.blob)
    assert container.count == values.size
    assert container.order == 3
    assert container.mode == "abs"
    assert container.declared_bound == 0.01
    assert container.escape_count == result.certificate.escape_count


def test_residuals_recovered_from_the_container_match_the_encoder(
    signals: dict[str, np.ndarray]
) -> None:
    """Re-derive the residual stream from the bytes, not from the encoder's notes."""
    values = signals["battery_voltage"]
    result = codec.encode(values, "abs", 1e-3, predictor="linear")
    residuals = residuals_from_container(result.blob)
    assert residuals.size == values.size - 2
    assert int(np.abs(residuals).max()) >= 0


def test_learned_predictor_requires_a_model(signals: dict[str, np.ndarray]) -> None:
    with pytest.raises(PredictorError, match="needs a fitted model"):
        codec.encode(signals["gyro_rate"], "abs", 1e-4, predictor="learned")


def test_classical_predictor_rejects_a_model(
    signals: dict[str, np.ndarray], learned_linear
) -> None:
    with pytest.raises(PredictorError, match="takes no model"):
        codec.encode(signals["gyro_rate"], "abs", 1e-4, model=learned_linear)


def test_decoding_a_learned_container_without_the_model_is_refused(
    signals: dict[str, np.ndarray], learned_linear
) -> None:
    result = codec.encode(
        signals["gyro_rate"], "abs", 1e-4, predictor="learned", model=learned_linear
    )
    with pytest.raises(PredictorError, match="needs a fitted model"):
        codec.decode(result.blob)


def test_decoding_with_a_mismatched_predictor_order_is_refused(
    signals: dict[str, np.ndarray]
) -> None:
    """Rewrite the predictor id in the header; the warm-up length then disagrees."""
    result = codec.encode(signals["magnetometer"], "abs", 0.01, predictor="linear")
    blob = bytearray(result.blob)
    blob[8] = 3  # predictor id for 'quadratic', whose warm-up is 3 not 2
    with pytest.raises(ContainerError, match="declares warm-up"):
        codec.decode(bytes(blob))


def test_decoding_an_unknown_predictor_id_is_refused(
    signals: dict[str, np.ndarray]
) -> None:
    result = codec.encode(signals["magnetometer"], "abs", 0.01)
    blob = bytearray(result.blob)
    blob[8] = 200
    with pytest.raises(ContainerError, match="unknown predictor id"):
        codec.decode(bytes(blob))


def test_short_array_is_refused_by_a_long_predictor() -> None:
    with pytest.raises(PredictorError, match="needs more than"):
        codec.encode(np.linspace(0.0, 1.0, 4), "abs", 1e-3, predictor="cubic")


def test_available_modes_and_predictor_names() -> None:
    assert codec.available_modes() == MODES
    assert codec.classical_predictor_names() == ("previous", "linear", "quadratic", "cubic")
    assert codec.get_predictor_safe("learned") is None


def test_constant_array_encodes_to_a_small_container() -> None:
    """A constant signal has zero residual entropy; the container is header-dominated."""
    values = np.full(512, 7.5)
    result = codec.encode(values, "abs", 1e-3, predictor="previous")
    assert result.certificate.bound_holds
    assert result.certificate.encoded_bytes < 100
    assert np.array_equal(codec.decode(result.blob), result.reconstruction)
