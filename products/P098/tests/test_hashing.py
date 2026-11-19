"""Content hashing, closure digests and the published collision bound."""

from __future__ import annotations

import math

import pytest

from evidencestale.hashing import (
    DIGEST_BITS,
    closure_digest,
    collision_bound_exponent,
    collision_probability_bound,
    content_hash,
    distinct_contents,
    snapshot_hashes,
)


def test_known_answer_sha256_of_the_empty_string() -> None:
    # FIPS 180-4 test vector for the empty message, as published by NIST.
    assert content_hash("") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_known_answer_sha256_of_a_corpus_content_token() -> None:
    # Computed once with hashlib and pinned, so a change to the encoding or the
    # digest would be caught rather than silently changing every verdict.
    assert content_hash("bottom:v0") == (
        "0b1ed7504d0fa7412e9cbe1114b0953e4893e5a282e62c55cca0c612d2be89b6"
    )
    assert content_hash("bottom:v1") == (
        "30a248148484e2c0d9854f6416531008f07605d1a47950e4020cbe31d6b1ed43"
    )


def test_digest_is_64_lowercase_hex_characters() -> None:
    digest = content_hash("anything")
    assert len(digest) == 64
    assert digest == digest.lower()
    assert set(digest) <= set("0123456789abcdef")


def test_bytes_are_rejected_with_an_actionable_message() -> None:
    with pytest.raises(TypeError, match="Decode bytes"):
        content_hash(b"x")  # type: ignore[arg-type]


def test_snapshot_hashes_covers_every_entry() -> None:
    snapshot = snapshot_hashes({"a": "bottom:v0", "b": "bottom:v1"})
    assert snapshot["a"] == content_hash("bottom:v0")
    assert snapshot["b"] == content_hash("bottom:v1")


def test_closure_digest_is_independent_of_mapping_order() -> None:
    first = closure_digest({"a": content_hash("x"), "b": content_hash("y")})
    second = closure_digest({"b": content_hash("y"), "a": content_hash("x")})
    assert first == second


def test_closure_digest_changes_when_a_member_changes() -> None:
    before = closure_digest({"a": content_hash("x"), "b": content_hash("y")})
    after = closure_digest({"a": content_hash("x"), "b": content_hash("z")})
    assert before != after


def test_collision_bound_is_the_pairwise_union_bound() -> None:
    # Hand arithmetic: for n = 4 there are 4*3/2 = 6 pairs, so the bound is
    # 6 * 2**-256. The ratio against 2**-256 is therefore exactly 6.
    bound = collision_probability_bound(4)
    assert bound / 2.0**-256 == pytest.approx(6.0, rel=1e-15, abs=0.0)


def test_collision_bound_is_zero_for_fewer_than_two_contents() -> None:
    assert collision_probability_bound(0) == 0.0
    assert collision_probability_bound(1) == 0.0
    assert collision_bound_exponent(1) == float("-inf")


def test_collision_bound_at_corpus_scale_is_small_but_representable() -> None:
    # 5000 distinct contents: 5000*4999/2 = 12 497 500 pairs, so the bound is
    # 12 497 500 * 2**-256. That is of order 1e-70 -- small, and well above the
    # smallest normal double, so it does NOT underflow and a reported 0.0 would
    # mean "fewer than two contents" rather than "too small to represent".
    bound = collision_probability_bound(5000)
    assert 0.0 < bound < 1e-60
    assert bound / 2.0**-256 == pytest.approx(12_497_500.0, rel=1e-12, abs=0.0)
    exponent = collision_bound_exponent(5000)
    expected = math.log10(12_497_500) - DIGEST_BITS * math.log10(2.0)
    assert exponent == pytest.approx(expected, rel=1e-12, abs=0.0)
    assert exponent < -60.0


def test_collision_bound_rejects_a_negative_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        collision_probability_bound(-1)


def test_distinct_contents_counts_unique_values() -> None:
    assert distinct_contents(["a", "a", "b"]) == 2
