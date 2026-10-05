"""A small explicit MLP whose storage is a flat, bit-addressable parameter block.

Why not inject directly into a ``sklearn`` estimator? Because
``MLPClassifier.coefs_`` is a list of float64 arrays with no defined byte
layout, and an upset model needs a definite one: a contiguous block of float32
or int8 words at known offsets, so that "parameter 37, bit 30" and "byte 148"
name the same physical thing.

:class:`MlpParameters` is therefore a flat float32 vector plus a
:class:`ParameterLayout` describing which slice holds which tensor.
:func:`from_sklearn` packs a trained ``MLPClassifier`` into it, and
``validation/validate_reference_forward.py`` checks that the forward pass here
reproduces ``MLPClassifier.predict_proba`` to float64 round-off, so that the
injection target is the same function the library computes.

Forward pass
------------
Two affine layers with a ReLU and a softmax, matching
``sklearn.neural_network.MLPClassifier`` with one hidden layer,
``activation="relu"`` and more than two classes:

    z1 = x W1 + b1
    a1 = max(z1, 0)
    z2 = a1 W2 + b2                                     ("logits")
    p  = softmax(z2)

Shapes: ``x`` is ``(n_samples, n_in)``, ``W1`` is ``(n_in, n_hidden)``, ``W2``
is ``(n_hidden, n_out)``. Units are dimensionless throughout; inputs are
standardised features.

Non-finite logits
-----------------
An exponent-bit upset can send a weight to ``+-inf`` or NaN, and the softmax of
such a logit vector is not defined by the formula above. This module uses one
stated convention, implemented in :func:`softmax`:

* a logit vector containing any NaN yields an all-NaN probability vector;
* otherwise, if any logit is ``+inf``, the probability mass is spread uniformly
  over the ``+inf`` entries and the rest are zero (the limit of the softmax
  along any path where those logits grow together);
* ``-inf`` logits receive zero probability.

Downstream, :func:`bitflipsim.criticality.total_variation` maps an all-NaN
probability vector to the maximum distance 1.0. Both conventions are stated
here and in the model card rather than left to whatever ``numpy`` does with
``exp(inf)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParameterLayout:
    """Byte and element offsets of each tensor inside the flat parameter block.

    Attributes
    ----------
    n_in, n_hidden, n_out:
        Layer widths.
    slices:
        ``{"W1": slice, "b1": slice, "W2": slice, "b2": slice}`` into the flat
        vector, in packing order.
    itemsize_bytes:
        Storage width of one parameter, bytes.
    """

    n_in: int
    n_hidden: int
    n_out: int
    slices: dict[str, slice]
    itemsize_bytes: int

    @property
    def size(self) -> int:
        """Total number of parameters, dimensionless."""
        return self.n_in * self.n_hidden + self.n_hidden + self.n_hidden * self.n_out + self.n_out

    @property
    def total_bytes(self) -> int:
        """Storage of the whole parameter block in bytes."""
        return self.size * self.itemsize_bytes

    @property
    def bits_per_parameter(self) -> int:
        return self.itemsize_bytes * 8

    def tensor_of(self, index: int) -> str:
        """Name of the tensor containing flat parameter ``index``."""
        if not 0 <= index < self.size:
            raise ValueError(f"parameter index {index} outside [0, {self.size})")
        for name, sl in self.slices.items():
            if sl.start <= index < sl.stop:
                return name
        raise AssertionError("parameter layout does not cover its own range")

    def byte_offset_of(self, index: int) -> int:
        """Byte offset of flat parameter ``index`` within the block."""
        if not 0 <= index < self.size:
            raise ValueError(f"parameter index {index} outside [0, {self.size})")
        return index * self.itemsize_bytes

    def is_bias(self, index: int) -> bool:
        return self.tensor_of(index) in ("b1", "b2")

    def layer_of(self, index: int) -> int:
        """0 for the first affine layer, 1 for the second."""
        return 0 if self.tensor_of(index) in ("W1", "b1") else 1


def make_layout(n_in: int, n_hidden: int, n_out: int, itemsize_bytes: int = 4) -> ParameterLayout:
    """Build the packing layout ``[W1 | b1 | W2 | b2]``, row-major within each."""
    for name, value in (("n_in", n_in), ("n_hidden", n_hidden), ("n_out", n_out)):
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
    if itemsize_bytes not in (1, 2, 4, 8):
        raise ValueError(f"itemsize_bytes must be 1, 2, 4 or 8, got {itemsize_bytes}")
    cursor = 0
    slices: dict[str, slice] = {}
    for name, count in (
        ("W1", n_in * n_hidden),
        ("b1", n_hidden),
        ("W2", n_hidden * n_out),
        ("b2", n_out),
    ):
        slices[name] = slice(cursor, cursor + count)
        cursor += count
    return ParameterLayout(
        n_in=n_in, n_hidden=n_hidden, n_out=n_out, slices=slices, itemsize_bytes=itemsize_bytes
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax with the stated non-finite conventions.

    Parameters
    ----------
    logits:
        Shape ``(n_samples, n_out)``, dimensionless.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_samples, n_out)``, rows summing to 1 where finite, all-NaN
        for rows containing a NaN logit, and uniform over the ``+inf`` entries
        for rows containing one or more ``+inf``.
    """
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"logits must be 2-D, got shape {z.shape}")
    out = np.empty_like(z)
    has_nan = np.isnan(z).any(axis=1)
    pos_inf = np.isposinf(z)
    has_pos_inf = pos_inf.any(axis=1)

    out[has_nan, :] = np.nan

    inf_rows = has_pos_inf & ~has_nan
    if inf_rows.any():
        counts = pos_inf[inf_rows].sum(axis=1, keepdims=True)
        out[inf_rows] = pos_inf[inf_rows] / counts

    plain = ~has_nan & ~has_pos_inf
    if plain.any():
        zp = z[plain]
        # -inf rows are handled by the shift: all-(-inf) cannot occur unless the
        # whole row is -inf, in which case the shift makes it 0 - 0 = NaN, so
        # guard it explicitly.
        shift = zp.max(axis=1, keepdims=True)
        all_neg_inf = np.isneginf(shift).ravel()
        with np.errstate(invalid="ignore"):
            exp = np.exp(zp - shift)
        exp[np.isnan(exp)] = 0.0
        total = exp.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            probs = exp / total
        if all_neg_inf.any():
            probs[all_neg_inf] = 1.0 / zp.shape[1]
        out[plain] = probs
    return out


@dataclass(frozen=True)
class MlpParameters:
    """A flat parameter block plus its layout.

    ``values`` is contiguous and of the dtype that is actually exposed to
    upsets (float32 by default, int8 for the quantized case via
    :func:`quantize_int8`).
    """

    values: np.ndarray
    layout: ParameterLayout

    def __post_init__(self) -> None:
        arr = np.ascontiguousarray(self.values)
        if arr.ndim != 1:
            raise ValueError(f"values must be 1-D, got shape {arr.shape}")
        if arr.size != self.layout.size:
            raise ValueError(
                f"values has {arr.size} entries but the layout describes {self.layout.size}"
            )
        if arr.dtype.itemsize != self.layout.itemsize_bytes:
            raise ValueError(
                f"values dtype {arr.dtype.name} is {arr.dtype.itemsize} bytes "
                f"but the layout says {self.layout.itemsize_bytes}"
            )
        object.__setattr__(self, "values", arr)

    def tensors(self) -> dict[str, np.ndarray]:
        """Unpack to ``{"W1", "b1", "W2", "b2"}`` views with their 2-D shapes."""
        lay = self.layout
        v = self.values
        return {
            "W1": v[lay.slices["W1"]].reshape(lay.n_in, lay.n_hidden),
            "b1": v[lay.slices["b1"]],
            "W2": v[lay.slices["W2"]].reshape(lay.n_hidden, lay.n_out),
            "b2": v[lay.slices["b2"]],
        }

    def with_values(self, values: np.ndarray) -> MlpParameters:
        return MlpParameters(values=values, layout=self.layout)

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Pre-softmax outputs, shape ``(n_samples, n_out)``, dimensionless.

        Computed in float64 from the float32 (or int8-dequantized) stored
        values, so the arithmetic is not itself a source of degradation.
        """
        t = self.tensors()
        xs = np.asarray(x, dtype=np.float64)
        if xs.ndim != 2 or xs.shape[1] != self.layout.n_in:
            raise ValueError(
                f"x must have shape (n_samples, {self.layout.n_in}), got {xs.shape}"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            z1 = xs @ t["W1"].astype(np.float64) + t["b1"].astype(np.float64)
            a1 = np.maximum(z1, 0.0)
            z2 = a1 @ t["W2"].astype(np.float64) + t["b2"].astype(np.float64)
        return z2

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        """Softmax of :meth:`logits`, shape ``(n_samples, n_out)``."""
        return softmax(self.logits(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Argmax class index, shape ``(n_samples,)``, dtype int64.

        Rows whose probabilities are all NaN are reported as class ``-1`` so
        that an accuracy computation cannot silently credit them.
        """
        p = self.probabilities(x)
        bad = np.isnan(p).any(axis=1)
        out = np.full(p.shape[0], -1, dtype=np.int64)
        if (~bad).any():
            out[~bad] = np.argmax(p[~bad], axis=1)
        return out

    def hidden_activations(self, x: np.ndarray) -> np.ndarray:
        """Post-ReLU hidden layer, shape ``(n_samples, n_hidden)``."""
        t = self.tensors()
        xs = np.asarray(x, dtype=np.float64)
        return np.maximum(xs @ t["W1"].astype(np.float64) + t["b1"].astype(np.float64), 0.0)


def from_sklearn(estimator: object, itemsize_bytes: int = 4) -> MlpParameters:
    """Pack a fitted one-hidden-layer ``MLPClassifier`` into an :class:`MlpParameters`.

    Raises
    ------
    ValueError
        If the estimator is not fitted, has more than one hidden layer, or does
        not use the ReLU activation this module implements.
    """
    coefs = getattr(estimator, "coefs_", None)
    intercepts = getattr(estimator, "intercepts_", None)
    if coefs is None or intercepts is None:
        raise ValueError("estimator has no coefs_/intercepts_; fit it first")
    if len(coefs) != 2:
        raise ValueError(f"expected exactly one hidden layer, got {len(coefs) - 1}")
    activation = getattr(estimator, "activation", "relu")
    if activation != "relu":
        raise ValueError(f"only activation='relu' is implemented, got {activation!r}")
    dtype = {1: np.int8, 2: np.float16, 4: np.float32, 8: np.float64}[itemsize_bytes]
    if itemsize_bytes == 1:
        raise ValueError("use quantize_int8 for the int8 case, not from_sklearn")
    w1, w2 = coefs
    b1, b2 = intercepts
    layout = make_layout(w1.shape[0], w1.shape[1], w2.shape[1], itemsize_bytes)
    flat = np.concatenate(
        [w1.ravel(), np.asarray(b1).ravel(), w2.ravel(), np.asarray(b2).ravel()]
    ).astype(dtype)
    return MlpParameters(values=flat, layout=layout)


@dataclass(frozen=True)
class Int8Quantization:
    """Symmetric per-tensor int8 quantization of a float parameter block.

    ``q = round(clip(w / scale, -127, 127))`` with ``scale = max|w| / 127``, so
    that the int8 range is used symmetrically and -128 is never produced by
    quantization (it is still reachable by an upset of the sign bit, which is
    exactly the asymmetry the quantized criticality sweep measures).

    Attributes
    ----------
    codes:
        int8 parameter block, same layout as the float one.
    scales:
        One scale per tensor, dimensionless (parameter units per code unit).
    """

    codes: np.ndarray
    scales: dict[str, float]
    layout: ParameterLayout

    def dequantized(self) -> np.ndarray:
        """Reconstruct a float32 parameter vector from the codes, elementwise."""
        out = np.empty(self.layout.size, dtype=np.float32)
        for name, sl in self.layout.slices.items():
            out[sl] = self.codes[sl].astype(np.float32) * np.float32(self.scales[name])
        return out

    def as_parameters(self) -> MlpParameters:
        """Dequantize into an :class:`MlpParameters` with a float32 layout."""
        float_layout_ = make_layout(
            self.layout.n_in, self.layout.n_hidden, self.layout.n_out, itemsize_bytes=4
        )
        return MlpParameters(values=self.dequantized(), layout=float_layout_)


def quantize_int8(params: MlpParameters) -> Int8Quantization:
    """Symmetric per-tensor int8 quantization of ``params``."""
    lay = params.layout
    if lay.itemsize_bytes == 1:
        raise ValueError("parameters are already int8")
    codes = np.zeros(lay.size, dtype=np.int8)
    scales: dict[str, float] = {}
    for name, sl in lay.slices.items():
        block = np.asarray(params.values[sl], dtype=np.float64)
        peak = float(np.max(np.abs(block))) if block.size else 0.0
        scale = peak / 127.0 if peak > 0.0 else 1.0
        scales[name] = scale
        codes[sl] = np.clip(np.rint(block / scale), -127, 127).astype(np.int8)
    int_layout_ = make_layout(lay.n_in, lay.n_hidden, lay.n_out, itemsize_bytes=1)
    return Int8Quantization(codes=codes, scales=scales, layout=int_layout_)
