"""Rate-distortion sweeps at exactly matched distortion.

A rate-distortion curve is only a comparison if the distortion really is
matched. Here it is matched by construction and not by search: every coder
compared in this module consumes the same integer symbol stream from the same
quantiser at the same declared bound, so the reconstruction -- and therefore
every error statistic -- is bit-identical across all of them. The only axis
that moves is bytes.

Rate is ``len(bytes)``. Never an entropy figure, never an estimate. The
first-order entropy appears in the tables as a labelled reference column and
is explicitly not a floor; see
:func:`boundcompress.entropy.shannon_entropy_bits`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import baselines
from .codec import encode
from .container import unpack
from .entropy import from_symbols
from .predictors import CLASSICAL_NAMES, get_predictor
from .quantise import make_spec, quantise


@dataclass(frozen=True)
class SweepRow:
    """One (signal, bound, coder) measurement.

    Attributes
    ----------
    signal, mode, bound:
        What was encoded and under what declared bound.
    coder:
        Predictor name for the package's own coder, or a baseline method name.
    encoded_bytes, bits_per_sample:
        Actual bytes and ``8 * bytes / samples``.
    max_error:
        Realised maximum per-sample error in the declared metric. Identical
        across every coder in a row group, which is the point.
    bound_holds:
        Whether the certificate passed.
    entropy_bits:
        First-order entropy of the coded symbol stream for the package's own
        coder; ``nan`` for a baseline, which codes a different byte
        serialisation. A reference figure, not a floor.
    roundtrip_ok:
        Whether the coder reproduced its input exactly.
    """

    signal: str
    mode: str
    bound: float
    coder: str
    encoded_bytes: int
    bits_per_sample: float
    max_error: float
    bound_holds: bool
    entropy_bits: float
    roundtrip_ok: bool


def sweep_signal(
    name: str,
    values: np.ndarray,
    bounds: list[float],
    *,
    mode: str = "abs",
    predictors: tuple[str, ...] = CLASSICAL_NAMES,
    model: object | None = None,
    include_baselines: bool = True,
    baseline_predictor: str = "linear",
) -> list[SweepRow]:
    """Measure every coder at every bound for one array.

    Parameters
    ----------
    name:
        Label for the array, carried into the rows.
    values:
        1-D float64 samples.
    bounds:
        Declared bounds to sweep, in the mode's own units.
    mode:
        Declared bound mode.
    predictors:
        Package predictors to measure. ``"learned"`` requires ``model``.
    model:
        Fitted :class:`boundcompress.learned.LearnedPredictor`.
    include_baselines:
        Whether to add the standard-library baselines of
        :mod:`boundcompress.baselines`.
    baseline_predictor:
        Which predictor's residual stream the ``*_residual`` baselines
        compress, so that the comparison is coder-against-coder rather than
        predictor-against-coder.

    Raises
    ------
    ValueError
        Propagated from the quantiser for an inadmissible bound or array.
    """
    rows: list[SweepRow] = []
    for bound in bounds:
        spec = make_spec(values, mode, bound)
        symbols, reconstruction, _ = quantise(values, spec)
        error = np.abs(np.asarray(values, dtype=np.float64) - reconstruction)
        if mode == "rel":
            metric = float(np.max(error / np.abs(values)))
        else:
            metric = float(np.max(error))

        for predictor in predictors:
            result = encode(
                values,
                mode,
                bound,
                predictor=predictor,
                model=model if predictor == "learned" else None,
            )
            certificate = result.certificate
            container = unpack(result.blob)
            ok = bool(container.count == np.asarray(values).size)
            rows.append(
                SweepRow(
                    signal=name,
                    mode=mode,
                    bound=float(bound),
                    coder=predictor,
                    encoded_bytes=certificate.encoded_bytes,
                    bits_per_sample=certificate.bits_per_sample,
                    max_error=metric,
                    bound_holds=certificate.bound_holds,
                    entropy_bits=certificate.residual_entropy_bits,
                    roundtrip_ok=ok,
                )
            )

        if include_baselines:
            engine = get_predictor(baseline_predictor)
            residuals = engine.residuals(symbols)
            for method, measurement in baselines.measure_all(
                values=values, symbols=symbols, residuals=residuals
            ).items():
                lossless = method in baselines.LOSSLESS_METHODS
                rows.append(
                    SweepRow(
                        signal=name,
                        mode=mode,
                        bound=float(bound),
                        coder=method,
                        encoded_bytes=measurement.total_bytes,
                        bits_per_sample=measurement.bits_per_sample,
                        max_error=0.0 if lossless else metric,
                        bound_holds=True,
                        entropy_bits=float("nan"),
                        roundtrip_ok=measurement.roundtrip_ok,
                    )
                )
    return rows


def geometric_bounds(values: np.ndarray, count: int = 6, *, coarsest: float = 1e-2,
                     finest: float = 1e-6) -> list[float]:
    """Bounds spanning ``coarsest`` to ``finest`` as fractions of the dynamic range.

    Expressed as absolute bounds in the array's own unit, so they can be used
    with ``mode="abs"``. Geometric spacing, because rate is roughly linear in
    ``log(1/bound)`` for a smooth source -- the classical
    :math:`\\tfrac{1}{2}\\log_2` high-rate result (Gersho & Gray 1992,
    chapter 4) -- and a linear sweep would spend every point at one end.
    """
    if count < 2:
        raise ValueError(f"need at least two bounds, got {count}")
    span = float(np.max(values) - np.min(values))
    if span <= 0.0:
        raise ValueError("cannot derive bounds from a constant array; pass them explicitly")
    fractions = np.geomspace(coarsest, finest, count)
    return [float(fraction * span) for fraction in fractions]


def best_rows(rows: list[SweepRow]) -> dict[tuple[str, float], SweepRow]:
    """Lowest-rate coder for each ``(signal, bound)`` pair, lossless coders excluded.

    A lossless baseline is at zero distortion and therefore not competing at
    the same point on the curve, so including it would be a category error.
    """
    best: dict[tuple[str, float], SweepRow] = {}
    for row in rows:
        if row.coder in baselines.LOSSLESS_METHODS:
            continue
        key = (row.signal, row.bound)
        if key not in best or row.encoded_bytes < best[key].encoded_bytes:
            best[key] = row
    return best


def verify_roundtrip(rows: list[SweepRow]) -> list[SweepRow]:
    """Rows whose coder failed to reproduce its input. Empty is the expected result."""
    return [row for row in rows if not row.roundtrip_ok]


def residuals_from_container(blob: bytes) -> np.ndarray:
    """Decode a container's residual stream without reconstructing the signal.

    Used by the validation scripts to re-derive the symbol statistics from the
    committed bytes rather than from the encoder's own bookkeeping, so the two
    cannot silently diverge.
    """
    from . import rans

    container = unpack(blob)
    symbol_list = rans.decode(
        container.payload, container.state, container.frequencies,
        container.count - container.order,
    )
    return from_symbols(symbol_list, container.escapes)
