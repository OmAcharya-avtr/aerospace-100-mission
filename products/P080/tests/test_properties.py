"""Hypothesis property tests over the algebraic identities this package relies on.

Hand-written cases check what the author thought of. On P071 of this mission,
Hypothesis found six equation defects and the hand-written tests found none,
which is why these exist rather than a longer list of fixed inputs. Every
falsifying example Hypothesis finds here is pinned as a named regression test
with literal inputs in ``tests/test_regressions.py``, because ``.hypothesis/``
is gitignored and a counterexample found on one machine is lost on a cold
clone.

The central property is the one the product is about: **for every array and
every declared bound, either every sample of the reconstruction is inside the
bound, or the encoder refuses to produce one.** The second branch is not a
hedge: :class:`boundcompress.quantise.BoundNotAchievable` is raised when the
integer a sample needs exceeds the pipeline's symbol limit, and a refusal is
the only honest alternative to a certificate the package cannot support.
Stated as a property over generated arrays, not only over the eight built-in
signals.

Three defects in this package were found here and nowhere else:

* an ``int64`` wrap in the zigzag fold at values spanning 8.26e11 with an
  absolute bound of 1.192092896e-07, which produced a negative symbol index
  and an ``IndexError`` three frames away in the frequency table;
* an ``OverflowError`` escaping the quantiser's correction loop at 1e300 with
  an absolute bound of 1e-300;
* a known-answer test of its own that asserted the wrong value for
  ``2 * log1p(0.01)``.

All three are pinned in ``tests/test_regressions.py``.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from boundcompress import entropy as E
from boundcompress import rans
from boundcompress.certificate import certify
from boundcompress.codec import decode, encode
from boundcompress.container import pack, unpack
from boundcompress.predictors import CLASSICAL_NAMES, get_predictor
from boundcompress.quantise import (
    BoundNotAchievable,
    admissible,
    dequantise,
    make_spec,
    quantise,
)

PROFILE = settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)

finite_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, width=64
)
positive_floats = st.floats(
    min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False, width=64
)
small_integers = st.integers(min_value=-(10**6), max_value=10**6)


@PROFILE
@given(st.integers(min_value=-(10**15), max_value=10**15))
def test_zigzag_is_a_bijection_on_the_integers(value: int) -> None:
    folded = E.zigzag(np.array([value]))
    assert int(folded[0]) >= 0
    assert int(E.unzigzag(folded)[0]) == value


@PROFILE
@given(st.integers(min_value=0, max_value=2**40))
def test_varint_roundtrip(value: int) -> None:
    out = bytearray()
    E.put_varint(value, out)
    assert E.get_varint(bytes(out), 0) == (value, len(out))


@PROFILE
@given(st.lists(small_integers, min_size=1, max_size=200))
def test_symbol_stream_roundtrip(residuals: list[int]) -> None:
    array = np.asarray(residuals, dtype=np.int64)
    stream = E.to_symbols(array)
    assert np.array_equal(E.from_symbols(stream.symbols, stream.escapes), array)


@PROFILE
@given(st.lists(st.integers(min_value=0, max_value=40), min_size=1, max_size=300))
def test_rans_roundtrip(symbols: list[int]) -> None:
    counts = [0] * 41
    for symbol in symbols:
        counts[symbol] += 1
    frequencies = rans.normalise_frequencies(counts)
    assert sum(frequencies) == rans.SCALE_TOTAL
    payload, state = rans.encode(symbols, frequencies)
    assert state < 1 << 32
    assert rans.decode(payload, state, frequencies, len(symbols)) == symbols


@PROFILE
@given(
    st.lists(st.integers(min_value=1, max_value=10**6), min_size=1, max_size=120)
)
def test_frequency_table_always_sums_to_the_scale_total(counts: list[int]) -> None:
    frequencies = rans.normalise_frequencies(counts)
    assert sum(frequencies) == rans.SCALE_TOTAL
    assert all(frequency >= 1 for frequency in frequencies)


@PROFILE
@given(
    values=st.lists(finite_floats, min_size=6, max_size=200),
    bound=st.floats(min_value=1e-4, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_absolute_bound_holds_on_every_generated_sample(
    values: list[float], bound: float
) -> None:
    """The central claim, over generated arrays rather than chosen ones."""
    array = np.asarray(values, dtype=np.float64)
    try:
        result = encode(array, "abs", bound, predictor="linear")
    except BoundNotAchievable as error:
        # A refusal is the documented alternative to an unsupportable
        # certificate, and it must say why it refused.
        assert "symbol limit" in str(error) or "no integer within one step" in str(error)
        return
    error = np.abs(array - result.reconstruction)
    assert bool(np.all(error <= bound))
    assert result.certificate.bound_holds
    assert result.certificate.violations == 0


@PROFILE
@given(
    values=st.lists(positive_floats, min_size=6, max_size=200),
    bound=st.floats(min_value=1e-6, max_value=0.2, allow_nan=False, allow_infinity=False),
)
def test_pointwise_relative_bound_holds_on_every_generated_sample(
    values: list[float], bound: float
) -> None:
    array = np.asarray(values, dtype=np.float64)
    try:
        result = encode(array, "rel", bound, predictor="previous")
    except BoundNotAchievable:
        return
    error = np.abs(array - result.reconstruction)
    assert bool(np.all(error <= bound * np.abs(array)))
    assert result.certificate.bound_holds


@PROFILE
@given(
    values=st.lists(finite_floats, min_size=8, max_size=150),
    bound=st.floats(min_value=1e-3, max_value=5.0, allow_nan=False, allow_infinity=False),
    predictor=st.sampled_from(CLASSICAL_NAMES),
)
def test_encode_decode_is_an_identity_on_the_reconstruction(
    values: list[float], bound: float, predictor: str
) -> None:
    array = np.asarray(values, dtype=np.float64)
    try:
        result = encode(array, "abs", bound, predictor=predictor)
    except BoundNotAchievable:
        return
    assert np.array_equal(decode(result.blob), result.reconstruction)
    assert result.certificate.encoded_bytes == len(result.blob)


@PROFILE
@given(
    values=st.lists(finite_floats, min_size=8, max_size=150),
    bound=st.floats(min_value=1e-3, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_container_pack_unpack_is_an_identity(values: list[float], bound: float) -> None:
    array = np.asarray(values, dtype=np.float64)
    try:
        blob = encode(array, "abs", bound).blob
    except BoundNotAchievable:
        return
    assert pack(unpack(blob)) == blob


@PROFILE
@given(
    symbols=st.lists(small_integers, min_size=6, max_size=200),
    predictor=st.sampled_from(CLASSICAL_NAMES),
)
def test_predictor_residuals_and_reconstruct_are_inverse(
    symbols: list[int], predictor: str
) -> None:
    engine = get_predictor(predictor)
    array = np.asarray(symbols, dtype=np.int64)
    assume(array.size > engine.order)
    residuals = engine.residuals(array)
    rebuilt = engine.reconstruct(array[: engine.order].tolist(), residuals)
    assert np.array_equal(rebuilt, array)


@PROFILE
@given(
    values=st.lists(finite_floats, min_size=4, max_size=120),
    bound=st.floats(min_value=1e-3, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_dequantise_inverts_quantise_to_within_the_bound(
    values: list[float], bound: float
) -> None:
    array = np.asarray(values, dtype=np.float64)
    spec = make_spec(array, "abs", bound)
    try:
        symbols, reconstruction, _ = quantise(array, spec)
    except BoundNotAchievable:
        return
    assert np.array_equal(dequantise(symbols, spec), reconstruction)
    assert bool(np.all(admissible(array, reconstruction, spec)))


@PROFILE
@given(
    values=st.lists(finite_floats, min_size=4, max_size=120),
    bound=st.floats(min_value=1e-3, max_value=5.0, allow_nan=False, allow_infinity=False),
    offset=st.floats(min_value=1.5, max_value=50.0, allow_nan=False, allow_infinity=False),
    index=st.integers(min_value=0, max_value=119),
)
def test_any_injected_violation_is_caught(
    values: list[float], bound: float, offset: float, index: int
) -> None:
    """A property-level negative control: no injected violation may slip past.

    Generated arrays, generated bounds, a generated violation magnitude
    strictly greater than the bound and a generated index. The certificate
    must fail and must name that index among the violations.
    """
    array = np.asarray(values, dtype=np.float64)
    position = index % array.size
    spec = make_spec(array, "abs", bound)
    try:
        _, reconstruction, _ = quantise(array, spec)
    except BoundNotAchievable:
        return
    corrupted = reconstruction.copy()
    corrupted[position] = array[position] + offset * bound
    certificate = certify(array, corrupted, "abs", bound)
    assert certificate.bound_holds is False
    assert position in certificate.violation_indices or certificate.violations > len(
        certificate.violation_indices
    )


@PROFILE
@given(
    values=st.lists(finite_floats, min_size=8, max_size=150),
    bound=st.floats(min_value=1e-3, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_rate_is_the_container_length_and_never_an_estimate(
    values: list[float], bound: float
) -> None:
    array = np.asarray(values, dtype=np.float64)
    try:
        result = encode(array, "abs", bound)
    except BoundNotAchievable:
        return
    assert result.certificate.encoded_bytes == len(result.blob)
    assert result.certificate.bits_per_sample == 8.0 * len(result.blob) / array.size


@PROFILE
@given(st.lists(st.integers(min_value=0, max_value=255), min_size=1, max_size=400))
def test_entropy_is_bounded_by_log2_of_the_alphabet(symbols: list[int]) -> None:
    import math

    value = E.shannon_entropy_bits(symbols)
    assert 0.0 <= value <= math.log2(E.ALPHABET) + 1e-12
