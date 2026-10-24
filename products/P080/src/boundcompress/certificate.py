"""The certificate: what the bound was, whether it held, and where it did not.

This module is deliberately independent of the encoder. :func:`certify` takes
an original array, *any* reconstruction of it, a declared mode and a declared
bound, and reports whether that reconstruction meets that bound. It does not
know or care how the reconstruction was produced. That independence is what
makes the negative control in
``validation/validate_negative_control_output.txt`` meaningful: a
reconstruction with a deliberately injected violation is handed to the same
function the codec's own output goes through, and it has to come back
``bound_holds=False`` with the right index.

What is asserted, and how
-------------------------

The check is a non-strict inequality, per sample, with **no tolerance added on
either side**:

* ``abs`` and ``range_rel``: :math:`|x_i - \\hat x_i| \\le b_{\\mathrm{eff}}`
* ``rel``: :math:`|x_i - \\hat x_i| \\le r\\,|x_i|`

A tolerance would make the claim unfalsifiable. The codec can afford a
tolerance-free comparison because :func:`boundcompress.quantise.quantise`
verifies and corrects every sample at encode time and refuses to encode when
it cannot, so the inequality is established by construction rather than
assumed to follow from the algebra. The separate question of floating-point
rounding is handled there and recorded as ``corrections`` here.

The certificate reports a maximum and a location, never a mean. A mean
per-sample error is the statistic this product exists to not report: it is
satisfied by a reconstruction that is wildly wrong on one sample, and one
sample is what breaks a downstream flight-software limit check.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .quantise import QuantiserSpec, check_array, make_spec

#: Number of violating indices listed in full before the list is truncated.
MAX_LISTED_VIOLATIONS = 16


@dataclass(frozen=True)
class Certificate:
    """Measured evidence about one encode of one array.

    Error fields
    ------------
    declared_mode, declared_bound:
        What the caller asked for.
    effective_absolute_bound:
        The absolute bound the mode implies, in the array's unit; ``nan`` for
        ``rel``, which has no single absolute bound by construction.
    max_absolute_error, max_relative_error:
        Maxima over every sample, in the array's unit and dimensionless. The
        relative error is ``nan`` for an array containing a zero sample, where
        it is undefined.
    argmax_index, argmax_value, argmax_reconstruction:
        Where the largest error in the *declared* metric occurred, the sample
        there, and what it decoded to. The declared metric is relative error in
        ``rel`` mode and absolute error otherwise, so this index is the one
        that nearly broke the promise.
    margin:
        Declared bound minus the realised maximum in the declared metric. Zero
        or positive when the bound holds. Reported so a reader can see how
        close the quantiser ran, rather than only that it got away with it.
    violations, violation_indices, bound_holds:
        Number of samples outside the bound, the first
        :data:`MAX_LISTED_VIOLATIONS` of them, and the verdict.

    Rate fields
    -----------
    samples, original_bytes, encoded_bytes, bits_per_sample, compression_ratio:
        ``original_bytes`` is ``8 * samples``, the float64 array as stored.
        ``encoded_bytes`` is ``len(blob)`` of the container and includes every
        header, table and escape byte. ``bits_per_sample`` is
        ``8 * encoded_bytes / samples``.
    residual_entropy_bits:
        First-order empirical entropy of the coded **symbol** stream, in bits
        per symbol. A **reference figure for a memoryless source, not a lower
        bound on the achievable rate**: a coder with memory goes below it,
        which ``validation/validate_rate_distortion.py`` measures on three of
        eight signals, and the sibling product P072 of this portfolio measured
        0.9978 of its held-out cases below their own entropy figure. Two
        further caveats, both load-bearing: it excludes every header, table and
        escape byte the realised rate includes, and it is computed over the
        256-symbol alphabet in which every large residual has been collapsed
        onto one escape symbol, so on an escape-heavy stream it understates
        badly. Compare a byte count against
        :func:`boundcompress.entropy.value_entropy_bits` instead.
    escape_count, corrections, predictor:
        Residuals too large for a direct symbol; samples whose integer was
        moved one step because floating-point rounding put the nearest integer
        outside the bound; and the predictor name.
    digest:
        SHA-256 of the container, hex. Two encodes of the same array with the
        same settings must produce the same digest, which is what
        ``validation/validate_determinism.py`` asserts.
    """

    declared_mode: str
    declared_bound: float
    effective_absolute_bound: float
    max_absolute_error: float
    max_relative_error: float
    argmax_index: int
    argmax_value: float
    argmax_reconstruction: float
    margin: float
    violations: int
    violation_indices: tuple[int, ...]
    bound_holds: bool
    samples: int
    original_bytes: int
    encoded_bytes: int
    bits_per_sample: float
    compression_ratio: float
    residual_entropy_bits: float
    escape_count: int
    corrections: int
    predictor: str
    digest: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def rate_below_entropy_reference(self) -> bool:
        """Whether the realised rate came out below the first-order entropy figure.

        Not a defect and not a paradox: the symbol stream is correlated, so the
        memoryless entropy is not a floor. Recorded because a reader who takes
        the entropy for a floor will otherwise think the measurement is wrong.
        """
        return self.bits_per_sample < self.residual_entropy_bits


def error_metrics(
    values: np.ndarray, reconstruction: np.ndarray, spec: QuantiserSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample absolute error, relative error, and the declared-metric error.

    The declared-metric error is the relative error in ``rel`` mode and the
    absolute error otherwise -- the quantity the declared bound bounds.
    Relative error is ``nan`` where the sample is zero.
    """
    original = check_array(values)
    decoded = np.ascontiguousarray(reconstruction, dtype=np.float64)
    if decoded.shape != original.shape:
        raise ValueError(
            f"reconstruction has shape {decoded.shape}, original has {original.shape}"
        )
    absolute = np.abs(original - decoded)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        relative = np.where(original != 0.0, absolute / np.abs(original), np.nan)
    declared = relative if spec.mode == "rel" else absolute
    return absolute, relative, declared


def certify(
    values: np.ndarray,
    reconstruction: np.ndarray,
    mode: str,
    bound: float,
    *,
    encoded_bytes: int = 0,
    residual_entropy_bits: float = 0.0,
    escape_count: int = 0,
    corrections: int = 0,
    predictor: str = "unknown",
    digest: str = "",
    notes: tuple[str, ...] = (),
) -> Certificate:
    """Certify ``reconstruction`` against ``values`` at a declared bound.

    Knows nothing about how ``reconstruction`` was produced. The codec calls
    this on its own output; the negative control calls it on a reconstruction
    with an injected violation; both get the same arithmetic.

    Parameters
    ----------
    values, reconstruction:
        1-D float64 arrays of equal length.
    mode, bound:
        The declared bound, as in :func:`boundcompress.quantise.make_spec`.
    encoded_bytes:
        ``len(blob)`` of the container, if there is one. Zero when certifying a
        bare reconstruction, in which case the rate fields are zero and the
        certificate is about the error only.

    Raises
    ------
    ValueError
        On a shape mismatch, an invalid mode or bound, or a non-finite sample
        in ``values``. A non-finite *reconstruction* is not refused: it is
        exactly the case the certificate must be able to fail on, and it does,
        because ``nan <= bound`` is false.
    """
    spec = make_spec(values, mode, bound)
    original = check_array(values)
    absolute, relative, declared = error_metrics(original, reconstruction, spec)

    if spec.mode == "rel":
        limit = spec.declared_bound
    else:
        limit = spec.absolute_bound

    with np.errstate(invalid="ignore"):
        inside = declared <= limit
    # A nan error is not inside the bound. ``nan <= limit`` is already False,
    # but the comparison is written out so the intent is not mistaken for an
    # accident of IEEE-754 semantics.
    inside = inside & ~np.isnan(declared)
    bad = np.flatnonzero(~inside)

    finite_declared = declared[np.isfinite(declared)]
    if finite_declared.size and bad.size == 0:
        argmax = int(np.nanargmax(declared))
    elif bad.size:
        argmax = int(bad[0])
    else:  # pragma: no cover - requires an all-nan declared metric with no violation
        argmax = 0
    realised = float(declared[argmax])

    samples = int(original.size)
    return Certificate(
        declared_mode=spec.mode,
        declared_bound=float(spec.declared_bound),
        effective_absolute_bound=float(spec.absolute_bound),
        max_absolute_error=float(np.nanmax(absolute)) if samples else 0.0,
        max_relative_error=(
            float(np.nanmax(relative)) if np.any(np.isfinite(relative)) else float("nan")
        ),
        argmax_index=argmax,
        argmax_value=float(original[argmax]),
        argmax_reconstruction=float(np.asarray(reconstruction, dtype=np.float64)[argmax]),
        margin=float(limit - realised),
        violations=int(bad.size),
        violation_indices=tuple(int(index) for index in bad[:MAX_LISTED_VIOLATIONS]),
        bound_holds=bool(bad.size == 0),
        samples=samples,
        original_bytes=8 * samples,
        encoded_bytes=int(encoded_bytes),
        bits_per_sample=(8.0 * encoded_bytes / samples) if samples else 0.0,
        compression_ratio=(8.0 * samples / encoded_bytes) if encoded_bytes else 0.0,
        residual_entropy_bits=float(residual_entropy_bits),
        escape_count=int(escape_count),
        corrections=int(corrections),
        predictor=predictor,
        digest=digest,
        notes=tuple(notes),
    )


def digest_bytes(blob: bytes) -> str:
    """SHA-256 of ``blob``, hex. Used for the determinism check."""
    return hashlib.sha256(blob).hexdigest()
