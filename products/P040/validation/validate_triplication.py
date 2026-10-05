"""Validation: triplication with a majority vote, enumerated exhaustively.

The specification's expectation is that triplication "recovers exactly the
single-upset cases and fails on the double-upset cases". The first half holds
unconditionally. The second half is true of the case that matters - two upsets
landing on the same stored element in two different copies - and this script
enumerates every double-upset pair so that the statement is precise rather than
approximate.

Two voters
----------
* word-level  : compare whole storage words; take the value held by at least
  two copies; flag the element uncorrectable when all three differ. This is
  what a TMR register file or a voted memory read does.
* bitwise     : ``(a & b) | (a & c) | (b & c)`` per storage bit. Always returns
  a value; never flags anything.

Enumeration
-----------
Protected set: 3 float16 parameters in 3 copies, so 3 x 3 x 16 = 144 upset
sites. All 144 single-upset cases and all C(144, 2) = 10296 double-upset pairs
are enumerated. float16 is used so that the double-upset enumeration is
exhaustive rather than sampled; a float32 enumeration of the same shape would
be C(288, 2) = 41328 pairs and is run as a second, single-parameter case below
to show the result does not depend on the width.

Runtime: about 20 s on one core.
"""

from __future__ import annotations

import itertools
import sys

import numpy as np

from bitflipsim.injection import BitUpset, apply_upsets
from bitflipsim.mitigation import bit_majority_vote, triplication_cost, word_majority_vote

failures: list[str] = []


def report(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


def classify(first: tuple[int, int, int], second: tuple[int, int, int]) -> str:
    if first[0] == second[0]:
        return "same copy"
    if first[1] != second[1]:
        return "different copies, different elements"
    if first[2] == second[2]:
        return "different copies, same element, same bit"
    return "different copies, same element, different bit"


def enumerate_case(golden: np.ndarray, width: int, label: str) -> None:
    n = golden.size
    copies = 3
    sites = [(c, e, b) for c in range(copies) for e in range(n) for b in range(width)]
    print()
    print("=" * 78)
    print(f"{label}: {n} {golden.dtype.name} parameters x {width} bits x {copies} copies "
          f"= {len(sites)} sites")
    print("=" * 78)

    single_word_wrong = 0
    single_word_flagged = 0
    single_bit_wrong = 0
    for copy_index, element, bit in sites:
        trio = [golden.copy() for _ in range(copies)]
        trio[copy_index] = apply_upsets(trio[copy_index], [BitUpset(element, bit)])
        voted, uncorrectable = word_majority_vote(*trio)
        if uncorrectable.any():
            single_word_flagged += 1
        if not np.array_equal(voted, golden):
            single_word_wrong += 1
        if not np.array_equal(bit_majority_vote(*trio), golden):
            single_bit_wrong += 1
    print(f"single upsets enumerated          {len(sites)}")
    print(f"word voter wrong                  {single_word_wrong}")
    print(f"word voter flagged uncorrectable  {single_word_flagged}")
    print(f"bitwise voter wrong               {single_bit_wrong}")
    report(f"{label} single-upset recovery",
           single_word_wrong == 0 and single_word_flagged == 0 and single_bit_wrong == 0,
           f"all {len(sites)} single-upset cases recovered exactly by both voters, "
           f"0 flagged, 0 wrong")

    buckets: dict[str, dict[str, int]] = {}
    bit_voter_wrong_by_bucket: dict[str, int] = {}
    pairs = 0
    for first, second in itertools.combinations(sites, 2):
        trio = [golden.copy() for _ in range(copies)]
        for copy_index, element, bit in (first, second):
            trio[copy_index] = apply_upsets(trio[copy_index], [BitUpset(element, bit)])
        voted, uncorrectable = word_majority_vote(*trio)
        correct = (not uncorrectable.any()) and np.array_equal(voted, golden)
        flagged = bool(uncorrectable.any())
        bit_correct = np.array_equal(bit_majority_vote(*trio), golden)
        key = classify(first, second)
        bucket = buckets.setdefault(key, {"cases": 0, "word_correct": 0, "word_flagged": 0,
                                          "bit_correct": 0})
        bucket["cases"] += 1
        bucket["word_correct"] += int(correct)
        bucket["word_flagged"] += int(flagged)
        bucket["bit_correct"] += int(bit_correct)
        bit_voter_wrong_by_bucket[key] = bit_voter_wrong_by_bucket.get(key, 0) + int(
            not bit_correct
        )
        pairs += 1
    expected_pairs = len(sites) * (len(sites) - 1) // 2
    print()
    print(f"double-upset pairs enumerated     {pairs} (= C({len(sites)}, 2) = {expected_pairs})")
    print(f"{'case':<46} {'cases':>7} {'word ok':>8} {'flagged':>8} {'bit ok':>7}")
    for key in (
        "same copy",
        "different copies, different elements",
        "different copies, same element, same bit",
        "different copies, same element, different bit",
    ):
        bucket = buckets.get(key, {"cases": 0, "word_correct": 0, "word_flagged": 0,
                                   "bit_correct": 0})
        print(f"{key:<46} {bucket['cases']:>7} {bucket['word_correct']:>8} "
              f"{bucket['word_flagged']:>8} {bucket['bit_correct']:>7}")

    zero = {"cases": 0, "word_correct": 0, "word_flagged": 0, "bit_correct": 0}
    same_copy = buckets.get("same copy", dict(zero))
    other_element = buckets.get("different copies, different elements", dict(zero))
    same_bit = buckets.get("different copies, same element, same bit", dict(zero))
    other_bit = buckets.get("different copies, same element, different bit", dict(zero))

    report(f"{label} double upsets in one copy are recovered",
           same_copy["word_correct"] == same_copy["cases"]
           and same_copy["word_flagged"] == 0
           and same_copy["bit_correct"] == same_copy["cases"],
           f"{same_copy['cases']} cases, all recovered by both voters")
    if other_element["cases"] == 0:
        print("[n/a ] double upsets in different copies but different elements: "
              "this case has a single element, so the bucket is empty by construction")
    else:
        report(f"{label} double upsets in different copies but different elements",
               other_element["word_correct"] == other_element["cases"]
               and other_element["word_flagged"] == 0,
               f"{other_element['cases']} cases, all recovered: each element still has a "
               f"2-of-3 majority")
    report(f"{label} double upsets on the same element and same bit FAIL silently",
           same_bit["word_correct"] == 0 and same_bit["word_flagged"] == 0
           and same_bit["bit_correct"] == 0,
           f"{same_bit['cases']} cases, 0 recovered, 0 flagged by either voter: two "
           f"copies agree on the wrong value, which is the classic TMR double-upset "
           f"failure")
    report(f"{label} double upsets on the same element at different bits are detected",
           other_bit["word_correct"] == 0
           and other_bit["word_flagged"] == other_bit["cases"]
           and other_bit["bit_correct"] == other_bit["cases"],
           f"{other_bit['cases']} cases: the word voter corrects none and flags all "
           f"{other_bit['cases']}; the bitwise voter silently corrects all of them")

    total_word_recovered = sum(b["word_correct"] for b in buckets.values())
    total_word_flagged = sum(b["word_flagged"] for b in buckets.values())
    print()
    print(f"word voter over all {pairs} double-upset pairs: "
          f"{total_word_recovered} recovered "
          f"({100.0 * total_word_recovered / pairs:.3f} %), "
          f"{total_word_flagged} flagged "
          f"({100.0 * total_word_flagged / pairs:.3f} %), "
          f"{pairs - total_word_recovered - total_word_flagged} silently wrong "
          f"({100.0 * (pairs - total_word_recovered - total_word_flagged) / pairs:.3f} %)")


print("=" * 78)
print("Triplication with a majority vote, enumerated exhaustively")
print("=" * 78)
print("Protected words are stored three times and voted at use. Two voters are")
print("compared; the single-upset claim is unconditional, and the double-upset")
print("behaviour is enumerated by case rather than summarised as pass or fail.")

enumerate_case(np.array([1.5, -0.25, 3.0], dtype=np.float16), 16, "Case A, float16")
enumerate_case(np.array([-2.75], dtype=np.float32), 32, "Case B, float32 single parameter")

print()
print("=" * 78)
print("Cost of triplication")
print("=" * 78)
cost = triplication_cost(588, inference_time_s=1.0e-3, vote_time_s=3.0e-6)
print(f"scheme                    {cost.scheme}")
print(f"protected                 {cost.protected_bytes} bytes")
print(f"extra memory              {cost.extra_memory_bytes} bytes")
print(f"memory factor             {cost.memory_factor:.3f}x")
print(f"latency overhead          {cost.latency_overhead_fraction:.6f} of inference time")
print("(the inference and vote times above are the illustrative 1 ms / 3 us pair;")
print("the measured figures for this host are in examples/mitigation_tradeoff.py)")
print()
print("What triplication does NOT buy, stated plainly:")
print("  - it does not reduce the upset rate, only the probability that an upset")
print("    changes the result;")
print("  - 3x memory on the protected region means 3x the exposed bits in that")
print("    region, so the rate of double upsets within a voted triple grows with")
print("    the protection, which is why the scrub interval matters;")
print("  - a word voter that flags an uncorrectable element has still lost that")
print("    element; detection is not correction.")

print()
print("=" * 78)
if failures:
    print(f"RESULT: {len(failures)} FAILED check(s): {failures}")
    sys.exit(1)
print("RESULT: all checks PASSED")
