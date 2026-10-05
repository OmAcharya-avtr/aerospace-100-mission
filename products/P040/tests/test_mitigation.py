"""Clamping bound, the two voters, and the mitigation cost accounting."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from bitflipsim.injection import BitUpset, apply_upsets, sample_upsets
from bitflipsim.mitigation import (
    bit_majority_vote,
    clamp_logit_bound,
    clamp_parameters,
    clamping_cost,
    expected_live_upsets,
    measure_clamp_latency,
    reload_cost,
    triplication_cost,
    word_majority_vote,
)


def test_clamping_at_max_absolute_weight_leaves_the_golden_model_unchanged(params):
    limit = float(np.abs(params.values).max())
    clamped = clamp_parameters(params, limit)
    assert np.array_equal(clamped.values, params.values)


def test_clamp_bound_is_hand_computable():
    # B = 2C max(C(n_in Xmax + 1), 1); C = 2, n_in = 8, Xmax = 3
    # inner = 2 * (8*3 + 1) = 50, so B = 2*2*50 = 200
    assert clamp_logit_bound(2.0, 8, 3.0) == pytest.approx(200.0)
    # the max(..., 1) branch: C = 0.01, n_in = 1, Xmax = 0 -> inner = 0.01 < 1
    assert clamp_logit_bound(0.01, 1, 0.0) == pytest.approx(0.02)


def test_clamp_bound_holds_for_every_single_upset_on_a_sample(params, problem):
    limit = float(np.abs(params.values).max())
    max_input = float(np.abs(problem.evaluation.x).max())
    bound = clamp_logit_bound(limit, params.layout.n_in, max_input)
    golden = clamp_parameters(params, limit).logits(problem.evaluation.x)
    rng = np.random.default_rng(31)
    worst = 0.0
    for upset in sample_upsets(params.layout.size, 32, 600, rng):
        faulty = clamp_parameters(
            params.with_values(apply_upsets(params.values, [upset])), limit
        )
        deviation = float(np.abs(faulty.logits(problem.evaluation.x) - golden).max())
        worst = max(worst, deviation)
    assert worst <= bound, (worst, bound)
    assert np.isfinite(worst)


def test_clamping_maps_nan_to_zero_and_infinity_to_the_limit(params):
    broken = params.values.copy()
    broken[0] = np.float32(np.nan)
    broken[1] = np.float32(np.inf)
    broken[2] = np.float32(-np.inf)
    clamped = clamp_parameters(params.with_values(broken), 1.0)
    assert clamped.values[0] == 0.0
    assert clamped.values[1] == pytest.approx(1.0)
    assert clamped.values[2] == pytest.approx(-1.0)


def test_clamp_validation(params):
    with pytest.raises(ValueError, match="finite and > 0"):
        clamp_parameters(params, 0.0)
    with pytest.raises(ValueError, match="finite and > 0"):
        clamp_parameters(params, float("inf"))
    with pytest.raises(ValueError, match="limit must be"):
        clamp_logit_bound(0.0, 8, 1.0)
    with pytest.raises(ValueError, match="n_in"):
        clamp_logit_bound(1.0, 0, 1.0)
    with pytest.raises(ValueError, match="max_abs_input"):
        clamp_logit_bound(1.0, 8, -1.0)


def test_voters_pass_three_identical_copies_through():
    block = np.array([1.5, -2.5, 3.0], dtype=np.float32)
    voted, uncorrectable = word_majority_vote(block, block, block)
    assert np.array_equal(voted, block)
    assert not uncorrectable.any()
    assert np.array_equal(bit_majority_vote(block, block, block), block)


def test_voters_validation():
    a = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError, match="share shape and dtype"):
        word_majority_vote(a, a, np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError, match="share shape and dtype"):
        bit_majority_vote(a, a, np.zeros(3, dtype=np.float64))


def test_exhaustive_single_upset_recovery_float16():
    """Every single upset in any of three copies, exhaustively, is corrected."""
    golden = np.array([1.5, -0.25, 3.0], dtype=np.float16)
    n, bits, copies = golden.size, 16, 3
    failures_word = 0
    failures_bit = 0
    for copy_index, element, bit in itertools.product(range(copies), range(n), range(bits)):
        trio = [golden.copy() for _ in range(copies)]
        trio[copy_index] = apply_upsets(trio[copy_index], [BitUpset(element, bit)])
        voted, uncorrectable = word_majority_vote(*trio)
        if uncorrectable.any() or not np.array_equal(voted, golden):
            failures_word += 1
        if not np.array_equal(bit_majority_vote(*trio), golden):
            failures_bit += 1
    assert copies * n * bits == 144
    assert failures_word == 0
    assert failures_bit == 0


def test_exhaustive_double_upset_classification_float16():
    """All double-upset pairs, classified by where the two upsets land.

    The word-level voter is correct whenever the two upsets land in the same
    copy, or in different copies but different elements. It is silently wrong
    when both hit the same element and the same bit in two different copies,
    and it flags the element uncorrectable when both hit the same element at
    different bits. The bitwise voter never flags anything and is wrong only in
    the same-element-same-bit case.
    """
    golden = np.array([1.5, -0.25, 3.0], dtype=np.float16)
    n, bits, copies = golden.size, 16, 3
    sites = [(c, e, b) for c in range(copies) for e in range(n) for b in range(bits)]
    counts = {
        "same_copy": [0, 0, 0],
        "cross_copy_same_element_same_bit": [0, 0, 0],
        "cross_copy_same_element_other_bit": [0, 0, 0],
        "cross_copy_other_element": [0, 0, 0],
    }
    bit_voter_failures = 0
    for first, second in itertools.combinations(sites, 2):
        trio = [golden.copy() for _ in range(copies)]
        for copy_index, element, bit in (first, second):
            trio[copy_index] = apply_upsets(trio[copy_index], [BitUpset(element, bit)])
        voted, uncorrectable = word_majority_vote(*trio)
        correct = (not uncorrectable.any()) and np.array_equal(voted, golden)
        detected = bool(uncorrectable.any())
        if first[0] == second[0]:
            key = "same_copy"
        elif first[1] != second[1]:
            key = "cross_copy_other_element"
        elif first[2] == second[2]:
            key = "cross_copy_same_element_same_bit"
        else:
            key = "cross_copy_same_element_other_bit"
        counts[key][0] += 1
        counts[key][1] += int(correct)
        counts[key][2] += int(detected)
        if not np.array_equal(bit_majority_vote(*trio), golden):
            bit_voter_failures += 1

    total = sum(entry[0] for entry in counts.values())
    assert total == 144 * 143 // 2 == 10_296

    # same copy: the other two copies still agree, so the vote is correct.
    assert counts["same_copy"][1] == counts["same_copy"][0]
    assert counts["same_copy"][2] == 0
    # cross copy, different elements: each element still has a 2-of-3 majority.
    assert counts["cross_copy_other_element"][1] == counts["cross_copy_other_element"][0]
    assert counts["cross_copy_other_element"][2] == 0
    # cross copy, same element and same bit: two copies agree on the WRONG
    # value, so the word voter returns it silently. Never correct, never flagged.
    same_bit = counts["cross_copy_same_element_same_bit"]
    assert same_bit[1] == 0
    assert same_bit[2] == 0
    # cross copy, same element, different bits: three distinct words, so the
    # word voter cannot correct but does flag every one of them.
    other_bit = counts["cross_copy_same_element_other_bit"]
    assert other_bit[1] == 0
    assert other_bit[2] == other_bit[0]
    # The bitwise voter is wrong on exactly the same-element-same-bit cases.
    assert bit_voter_failures == same_bit[0]


def test_expected_live_upsets_is_half_rate_times_interval():
    assert expected_live_upsets(1e-3, 100.0) == pytest.approx(0.05)
    assert expected_live_upsets(0.0, 100.0) == 0.0
    with pytest.raises(ValueError, match=">= 0"):
        expected_live_upsets(-1.0, 1.0)


def test_cost_accounting():
    triple = triplication_cost(588, inference_time_s=1e-3, vote_time_s=5e-5)
    assert triple.extra_memory_bytes == 1176
    assert triple.memory_factor == 3.0
    assert triple.latency_overhead_fraction == pytest.approx(0.05)

    cost, live = reload_cost(588, scrub_interval_s=60.0, reload_time_s=0.01,
                             upset_rate_per_s=1e-3)
    assert cost.extra_memory_bytes == 588
    assert cost.memory_factor == 2.0
    assert cost.latency_overhead_fraction == pytest.approx(0.01 / 60.0)
    assert live == pytest.approx(0.03)

    latency = measure_clamp_latency(147, repeats=20)
    assert latency["median_s"] > 0.0
    assert latency["per_parameter_ns"] > 0.0
    clamp = clamping_cost(588, latency, inference_time_s=1e-3)
    assert clamp.extra_memory_bytes == 0
    assert clamp.memory_factor == 1.0


def test_cost_validation():
    with pytest.raises(ValueError, match="protected_bytes"):
        triplication_cost(-1, 1e-3, 1e-5)
    with pytest.raises(ValueError, match="inference_time_s"):
        triplication_cost(10, 0.0, 1e-5)
    with pytest.raises(ValueError, match="vote_time_s"):
        triplication_cost(10, 1e-3, -1.0)
    with pytest.raises(ValueError, match="scrub_interval_s"):
        reload_cost(10, 0.0, 0.0, 1e-3)
    with pytest.raises(ValueError, match="cannot exceed"):
        reload_cost(10, 1.0, 2.0, 1e-3)
    with pytest.raises(ValueError, match="must be > 0"):
        measure_clamp_latency(0)
