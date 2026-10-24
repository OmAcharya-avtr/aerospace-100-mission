"""The bound-certified codec: encode, decode, and the certificate of the encode.

Pipeline
--------

.. code-block:: text

    values -> quantise.make_spec -> quantise.quantise -> predictors.residuals
           -> entropy.to_symbols -> rans.encode -> container.pack -> bytes

and back the same way. Everything after the quantiser is **lossless on the
integer symbol stream**, verified by test, so:

* the reconstruction is a function of the quantiser alone;
* changing the predictor changes the rate and cannot change any reconstructed
  sample, so two predictors are compared at *identical* distortion rather than
  at matched-by-measurement distortion;
* the certificate's bound claim rests on the encode-time verification in
  :func:`boundcompress.quantise.quantise` and on nothing else.

What this is not
----------------

Not a codec you should fly. ``numcodecs`` and ``zfpy`` are the right tools for
compressing an array, and CCSDS 121.0-B-3 and 122.0-B-2 are the standards a
flight implementation should use. The contribution here is the certificate and
the measured cost of the guarantee, not the entropy coder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import entropy, rans
from .certificate import Certificate, certify, digest_bytes
from .container import Container, ContainerError, pack, unpack
from .predictors import (
    PREDICTOR_IDS,
    PREDICTOR_NAMES,
    ClassicalPredictor,
    PredictorError,
    get_predictor,
)
from .quantise import MODES, QuantiserSpec, check_array, dequantise, make_spec, quantise

__all__ = ["EncodeResult", "encode", "decode", "encode_decode"]


@dataclass(frozen=True)
class EncodeResult:
    """The bytes to transmit, the reconstruction they decode to, and the certificate."""

    blob: bytes
    reconstruction: np.ndarray
    certificate: Certificate
    spec: QuantiserSpec


def _predictor_for(name: str, model: object | None) -> object:
    if name == "learned":
        if model is None:
            raise PredictorError(
                "predictor='learned' needs a fitted model; build one with "
                "boundcompress.learned.LearnedPredictor and pass it as model="
            )
        return model
    if model is not None:
        raise PredictorError(
            f"predictor={name!r} is a classical predictor and takes no model; got "
            f"{type(model).__name__}"
        )
    return get_predictor(name)


def encode(
    values: np.ndarray,
    mode: str,
    bound: float,
    *,
    predictor: str = "linear",
    model: object | None = None,
) -> EncodeResult:
    """Encode ``values`` under a declared error bound and certify the result.

    Parameters
    ----------
    values:
        1-D finite array of samples, any unit.
    mode:
        ``"abs"``, ``"rel"`` or ``"range_rel"``; see
        :mod:`boundcompress.quantise`.
    bound:
        Declared bound. In the array's unit for ``"abs"``, dimensionless for
        the other two.
    predictor:
        ``"previous"``, ``"linear"``, ``"quadratic"``, ``"cubic"`` or
        ``"learned"``. Affects the rate only, never the reconstruction.
    model:
        A fitted :class:`boundcompress.learned.LearnedPredictor`, required when
        ``predictor="learned"`` and rejected otherwise.

    Raises
    ------
    ValueError
        For an invalid bound, mode or array; see
        :func:`boundcompress.quantise.make_spec`.
    boundcompress.quantise.BoundNotAchievable
        If some sample has no admissible integer, in which case nothing is
        returned rather than a certificate that cannot be supported.
    boundcompress.predictors.PredictorError
        For an unknown predictor name, a missing model, or an array no longer
        than the predictor's warm-up.
    """
    original = check_array(values)
    spec = make_spec(original, mode, bound)
    symbols, reconstruction, corrections = quantise(original, spec)

    engine = _predictor_for(predictor, model)
    residuals = engine.residuals(symbols)
    stream = entropy.to_symbols(residuals)
    frequencies = rans.normalise_frequencies(entropy.symbol_counts(stream.symbols))
    payload, state = rans.encode(stream.symbols, frequencies)

    container = Container(
        mode=spec.mode,
        transform=spec.transform,
        sign=spec.sign,
        predictor=PREDICTOR_IDS[predictor],
        order=int(engine.order),
        declared_bound=float(spec.declared_bound),
        step=float(spec.step),
        count=int(original.size),
        escape_count=int(stream.escape_count),
        state=int(state),
        warmup=[int(value) for value in symbols[: engine.order]],
        frequencies=frequencies,
        escapes=stream.escapes,
        payload=payload,
    )
    blob = pack(container)

    certificate = certify(
        original,
        reconstruction,
        spec.mode,
        spec.declared_bound,
        encoded_bytes=len(blob),
        residual_entropy_bits=entropy.shannon_entropy_bits(stream.symbols),
        escape_count=stream.escape_count,
        corrections=corrections,
        predictor=predictor,
        digest=digest_bytes(blob),
    )
    return EncodeResult(blob, reconstruction, certificate, spec)


def decode(blob: bytes, *, model: object | None = None) -> np.ndarray:
    """Decode a container to a float64 reconstruction.

    Parameters
    ----------
    blob:
        Bytes produced by :func:`encode`.
    model:
        The same fitted learned predictor used to encode, when the container
        names ``learned``. The model is **not** carried in the container: its
        parameters are a shared decoder asset, and the cost of shipping it is
        reported separately in ``MODEL_CARD.md`` rather than folded into the
        rate.

    Raises
    ------
    boundcompress.container.ContainerError
        If the buffer is not a readable container, or if its declared sample
        count disagrees with what the coder produced.
    boundcompress.predictors.PredictorError
        If the container names ``learned`` and no model was supplied.
    """
    container = unpack(blob)
    name = PREDICTOR_NAMES.get(container.predictor)
    if name is None:
        raise ContainerError(f"container names unknown predictor id {container.predictor}")
    engine = _predictor_for(name, model)
    if int(engine.order) != container.order:
        raise ContainerError(
            f"container declares warm-up {container.order} but predictor {name!r} needs "
            f"{int(engine.order)}"
        )
    residual_count = container.count - container.order
    if residual_count < 0:
        raise ContainerError(
            f"container declares {container.count} sample(s) and a warm-up of "
            f"{container.order}"
        )
    symbol_list = rans.decode(container.payload, container.state, container.frequencies,
                              residual_count)
    escaped = sum(1 for symbol in symbol_list if symbol == entropy.ESCAPE)
    if escaped != container.escape_count:
        raise ContainerError(
            f"container declares {container.escape_count} escape(s), the payload holds "
            f"{escaped}"
        )
    residuals = entropy.from_symbols(symbol_list, container.escapes)
    symbols = engine.reconstruct(container.warmup, residuals)
    if symbols.size != container.count:
        raise ContainerError(
            f"reconstructed {symbols.size} symbol(s), container declares {container.count}"
        )
    spec = QuantiserSpec(
        mode=container.mode,
        declared_bound=container.declared_bound,
        step=container.step,
        transform=container.transform,
        sign=container.sign,
    )
    return dequantise(symbols, spec)


def encode_decode(
    values: np.ndarray,
    mode: str,
    bound: float,
    *,
    predictor: str = "linear",
    model: object | None = None,
) -> tuple[EncodeResult, np.ndarray]:
    """Encode then decode, returning both. The two reconstructions must agree exactly.

    The agreement is exact -- bit-for-bit, not to a tolerance -- because the
    decoder recomputes ``symbols * step`` from the same integers and the same
    float64 step. That is an integer identity followed by one multiplication,
    not a floating-point pipeline, so equality is the honest assertion here;
    it is checked in ``tests/test_codec.py::test_encode_decode_bit_identical``.
    """
    result = encode(values, mode, bound, predictor=predictor, model=model)
    return result, decode(result.blob, model=model)


def available_modes() -> tuple[str, ...]:
    """The declared bound modes, for the CLI and for tests to iterate."""
    return MODES


def classical_predictor_names() -> tuple[str, ...]:
    """Classical predictor names, in increasing order."""
    return tuple(
        name for name in PREDICTOR_IDS if isinstance(get_predictor_safe(name), ClassicalPredictor)
    )


def get_predictor_safe(name: str) -> object | None:
    """``get_predictor`` that returns ``None`` instead of raising. Used by the CLI."""
    try:
        return get_predictor(name)
    except PredictorError:
        return None
