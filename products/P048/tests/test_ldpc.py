"""LDPC construction and sum-product decoder tests.

The known-answer test for the decoder is exactness on a cycle-free Tanner
graph, checked against brute-force bit a-posteriori probabilities over all
codewords. A regular bipartite graph with variable degree above one always
contains a cycle, so the cycle-free case here is two disjoint single-parity
checks, which still exercises the leave-one-out product over three edges.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from softdecode.ldpc import LdpcCode, make_regular_ldpc, sum_product_decode


def _cycle_free_code() -> LdpcCode:
    h = np.array(
        [[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]],
        dtype=np.int8,
    )
    check_vars = np.array([[0, 1, 2], [3, 4, 5]])
    var_edges = np.arange(6).reshape(6, 1)
    generator = np.array(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 1],
        ],
        dtype=np.int8,
    )
    return LdpcCode(h, generator, np.arange(6), check_vars, var_edges, 0)


def _brute_force_posterior(h: np.ndarray, llr: np.ndarray) -> np.ndarray:
    n = h.shape[1]
    words = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int8)
    words = words[~np.any((words @ h.T) % 2, axis=1)]
    out = np.empty_like(llr)
    for b in range(llr.shape[0]):
        score = -(words.astype(float) @ llr[b])
        for i in range(n):
            zero = words[:, i] == 0
            lo = np.max(score[zero]) + np.log(np.sum(np.exp(score[zero] - np.max(score[zero]))))
            hi = np.max(score[~zero]) + np.log(
                np.sum(np.exp(score[~zero] - np.max(score[~zero])))
            )
            out[b, i] = lo - hi
    return out


def test_sum_product_is_exact_on_cycle_free_graph():
    code = _cycle_free_code()
    rng = np.random.default_rng(11)
    llr = rng.standard_normal((20, 6)) * 2.5
    _, posterior, _ = sum_product_decode(code, llr, iterations=6, early_stop=False)
    exact = _brute_force_posterior(code.parity_check, llr)
    assert np.max(np.abs(posterior - exact)) < 1e-9


def test_default_code_is_regular(code):
    assert np.all(code.parity_check.sum(axis=1) == 6)
    assert np.all(code.parity_check.sum(axis=0) == 3)
    assert code.length == 96
    assert code.n_checks == 48


def test_default_code_rate_is_reported_not_assumed(code):
    # Gallager's construction with three row groups leaves two dependent rows,
    # so the realised dimension is 50, not 48, and the rate is 50/96.
    assert code.dimension == 50
    assert code.rate == pytest.approx(50.0 / 96.0)


def test_encoding_produces_codewords(code, rng):
    messages = rng.integers(0, 2, size=(64, code.dimension)).astype(np.int8)
    words = code.encode(messages)
    assert not np.any(code.syndrome(words))
    assert np.array_equal(words[:, code.message_positions()], messages)


def test_generator_rows_are_codewords(code):
    words = code.encode(np.eye(code.dimension, dtype=np.int8))
    assert not np.any(code.syndrome(words))


def test_decoder_recovers_noiseless_codewords(code, rng):
    messages = rng.integers(0, 2, size=(32, code.dimension)).astype(np.int8)
    words = code.encode(messages)
    llr = (1 - 2 * words.astype(float)) * 20.0
    bits, _, used = sum_product_decode(code, llr, iterations=20)
    assert np.array_equal(bits, words)
    assert int(used) == 1


def test_decoder_corrects_errors(code, rng):
    messages = rng.integers(0, 2, size=(200, code.dimension)).astype(np.int8)
    words = code.encode(messages)
    llr = (1 - 2 * words.astype(float)) * 4.0 + rng.standard_normal(words.shape) * 2.0
    raw = float(np.mean((llr < 0) != words))
    bits, _, _ = sum_product_decode(code, llr, iterations=25)
    decoded = float(np.mean(bits != words))
    assert raw > 0.01
    assert decoded < raw / 5.0


def test_sum_product_is_not_scale_invariant(code, rng):
    # The complement of test_codes.test_soft_decoding_is_scale_invariant, and
    # the reason this decoder is used for the mismatch study.
    messages = rng.integers(0, 2, size=(300, code.dimension)).astype(np.int8)
    words = code.encode(messages)
    llr = (1 - 2 * words.astype(float)) * 1.2 + rng.standard_normal(words.shape) * 1.5
    base, _, _ = sum_product_decode(code, llr, iterations=20, early_stop=False)
    scaled, _, _ = sum_product_decode(code, llr * 12.0, iterations=20, early_stop=False)
    assert not np.array_equal(base, scaled)


def test_posterior_sign_of_single_bit_follows_channel(code):
    llr = np.zeros((1, code.length))
    llr[0, 0] = 6.0
    _, posterior, _ = sum_product_decode(code, llr, iterations=1, early_stop=False)
    assert posterior[0, 0] > 0.0


def test_cycle_count_is_reported():
    built = make_regular_ldpc(seed=20261006, attempts=40)
    assert built.cycles4 == 26


def test_construction_validation():
    with pytest.raises(ValueError, match="multiple of dc"):
        make_regular_ldpc(n=50, dv=3, dc=6)
    with pytest.raises(ValueError, match="at least 2"):
        make_regular_ldpc(n=96, dv=1, dc=6)


def test_decode_validation(code):
    with pytest.raises(ValueError, match="last dimension"):
        sum_product_decode(code, np.zeros((2, 95)))
    with pytest.raises(ValueError, match="iterations"):
        sum_product_decode(code, np.zeros((2, 96)), iterations=0)
    with pytest.raises(ValueError, match="message_clip"):
        sum_product_decode(code, np.zeros((2, 96)), message_clip=0.0)
    with pytest.raises(ValueError, match="last dimension"):
        code.encode(np.zeros((2, 3), dtype=np.int8))


def test_other_shapes_build(code):
    other = make_regular_ldpc(n=60, dv=3, dc=5, seed=3, attempts=8)
    assert other.length == 60
    assert np.all(other.parity_check.sum(axis=1) == 5)
    assert np.all(other.parity_check.sum(axis=0) == 3)
