"""Pinned counterexamples. Literal inputs, named, with the defect they found.

``.hypothesis/`` is gitignored, so a falsifying example found by one run is
lost on a cold clone. Every defect found during the build of this package is
pinned here with its literal inputs so that a regression is caught by the
suite and not rediscovered by chance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from boundcompress import entropy as E
from boundcompress.codec import decode, encode
from boundcompress.container import ContainerError, unpack
from boundcompress.quantise import (
    MAX_SYMBOL_MAGNITUDE,
    BoundNotAchievable,
    make_spec,
    quantise,
)


def test_regression_zigzag_int64_wrap_at_large_span_and_fine_bound() -> None:
    """Hypothesis, 2026-10-24. Negative symbol index from a wrapped zigzag fold.

    Literal counterexample: six samples, four zeros followed by 826207132638.0
    and -273304495592.0, with mode='abs' and bound=1.192092896e-07. The
    quantiser step is 2.384185792e-07, so the symbols reach 3.47e18 and the
    first difference reaches 4.61e18; ``2 * residual`` then wrapped int64 to a
    negative value, which indexed the frequency table out of range and raised
    ``IndexError: list index out of range`` three frames away in
    ``entropy.symbol_counts``.

    The fix is the derived guard ``quantise.MAX_SYMBOL_MAGNITUDE = 2**57``,
    which refuses the encode with a message naming the symbol limit. The
    refusal, not an encode, is the correct behaviour: the pipeline cannot carry
    these symbols, and a wrapped reconstruction is worse than no
    reconstruction.
    """
    values = np.array(
        [0.0, 0.0, 0.0, 0.0, 826207132638.0, -273304495592.0], dtype=np.float64
    )
    bound = 1.192092896e-07
    with pytest.raises(BoundNotAchievable, match="symbol limit"):
        encode(values, "abs", bound, predictor="linear")
    with pytest.raises(BoundNotAchievable, match="symbol limit"):
        encode(values, "abs", bound, predictor="previous")


def test_regression_overflow_error_escaped_the_correction_loop() -> None:
    """Hand-written, 2026-10-24. ``OverflowError`` instead of ``BoundNotAchievable``.

    At x=1e300 with an absolute bound of 1e-300 the needed integer is near
    5e599. The first version of ``quantise`` cast it to int64 anyway, and the
    correction loop then raised ``OverflowError: Python int too large to
    convert to C long`` from ``dequantise``. The guard now refuses before the
    cast.
    """
    values = np.array([1e300, 1e300], dtype=np.float64)
    spec = make_spec(values, "abs", 1e-300)
    with pytest.raises(BoundNotAchievable, match="symbol limit"):
        quantise(values, spec)


def test_regression_relative_step_literal_was_wrong_in_the_seventh_figure() -> None:
    """Self-caught, 2026-10-24. A known-answer test asserted the wrong constant.

    The first version of ``tests/test_quantise.py`` asserted that the step for
    ``mode='rel'`` at r=0.01 is 0.01990049875623903. The correct float64 value
    of ``2 * math.log1p(0.01)`` is 0.019900661706336167 -- a disagreement in
    the seventh significant figure, from a hand-rendered logarithm. Pinned here
    against the library function rather than against a literal, so the two can
    never drift apart again.
    """
    spec = make_spec(np.array([1.0, 2.0]), "rel", 0.01)
    assert spec.step == 2.0 * math.log1p(0.01)
    assert spec.step == pytest.approx(0.019900661706336167, rel=1e-15)


def test_regression_container_body_errors_are_container_errors() -> None:
    """Hand-written, 2026-10-24. ``SymbolError`` escaped ``unpack``.

    A container truncated inside its payload-length varint raised
    ``entropy.SymbolError``, so a caller handling a malformed buffer needed
    two exception types. ``unpack`` now wraps every body parse failure in
    ``ContainerError``.
    """
    values = np.linspace(1.0, 2.0, 64)
    blob = encode(values, "abs", 1e-3).blob
    with pytest.raises(ContainerError):
        unpack(blob[:-3])


def test_regression_header_predictor_id_is_byte_eight() -> None:
    """Self-caught, 2026-10-24. A test patched the wrong header byte.

    The container's fixed header is ``4s`` magic, then version, mode,
    transform, sign, predictor, order as single bytes. The predictor id is
    therefore at offset 8 and the warm-up length at offset 9. A test that
    patched offset 9 corrupted the warm-up length instead and produced a
    frequency-table error, which looked like a parser bug and was not one.
    """
    values = np.linspace(0.0, 1.0, 64)
    blob = bytearray(encode(values, "abs", 1e-3, predictor="linear").blob)
    assert blob[8] == 2  # 'linear'
    assert blob[9] == 2  # warm-up of 2
    blob[8] = 3  # 'quadratic', whose warm-up is 3
    with pytest.raises(ContainerError, match="declares warm-up"):
        decode(bytes(blob))


def test_regression_submodule_shadowed_by_a_same_named_function() -> None:
    """Self-caught, 2026-10-24. ``from boundcompress import quantise`` gave a function.

    ``boundcompress/__init__.py`` re-exported the function
    ``quantise.quantise``, which bound the package attribute ``quantise`` to
    the function and shadowed the submodule. ``quantise.MODES`` then raised
    ``AttributeError: 'function' object has no attribute 'MODES'`` from the
    CLI. The function is no longer re-exported at package level.
    """
    import boundcompress
    from boundcompress import quantise as quantise_module

    assert hasattr(quantise_module, "MODES")
    assert not hasattr(boundcompress, "quantise") or hasattr(
        boundcompress.quantise, "MODES"
    )


def test_regression_symbol_limit_is_the_derived_value() -> None:
    """The guard is a derivation and not a round number; pinned so it stays one.

    ``cubic`` has absolute coefficient sum 15, so the residual is at most 16S
    and its zigzag fold at most 32S. ``32 * 2**57 == 2**62``, one bit below the
    int64 ceiling.
    """
    assert MAX_SYMBOL_MAGNITUDE == 2**57
    assert 32 * MAX_SYMBOL_MAGNITUDE == 2**62
    assert 32 * MAX_SYMBOL_MAGNITUDE < 2**63


def test_regression_zigzag_refuses_rather_than_wrapping() -> None:
    """The public fold now refuses an unfoldable value instead of wrapping."""
    with pytest.raises(E.SymbolError, match="zigzag folded"):
        E.zigzag(np.array([E.MAX_FOLDABLE], dtype=np.int64))
    folded = E.zigzag(np.array([E.MAX_FOLDABLE - 1], dtype=np.int64))
    assert int(folded[0]) > 0
