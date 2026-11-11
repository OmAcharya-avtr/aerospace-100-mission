"""Three checks whose mathematics this package's implementation does not use.

"Two implementations agree" and "the answer lies inside a bound derived from
different mathematics" are different checks, and the second catches what the
first cannot. A product of this mission had 777 tests, property tests
included, and none of them could catch catastrophic cancellation in a closed
form, because every test compared the series against expressions sharing the
same cancellation.

``boardprobe.provenance`` computes ``meet`` as ``min`` over an integer rank.
Everything below avoids that encoding.

1. **Order-theoretic.** The chain is re-derived from its HASSE DIAGRAM -- the
   two covering edges ``simulated -> recorded`` and ``recorded -> live`` --
   by Warshall's transitive-closure algorithm over a boolean reachability
   matrix (Warshall 1962, *JACM* 9(1), 11-12). The meet of a subset is then
   defined order-theoretically, as the unique element of the subset that is
   below every element of it under the CLOSURE relation, with no ranks
   anywhere. That is compared against ``meet_all`` over every assignment up to
   length 6. If the rank encoding were not an order embedding of the chain,
   this disagrees.

2. **Combinatorial.** For ``n`` readings drawn from the three provenances, the
   number of assignments whose meet is each element follows from inclusion and
   exclusion: an assignment meets at ``simulated`` exactly when at least one
   reading is simulated, so there are ``3^n - 2^n`` of them; it meets at
   ``recorded`` when none is simulated and at least one is recorded, so
   ``2^n - 1``; and at ``live`` only when all are live, so ``1``. The three
   sum to ``3^n``, which is the identity the check also verifies. This counts
   the OUTPUTS of ``meet_all`` by exhaustive enumeration and compares against
   the closed form. It is exact integer arithmetic with no tolerance, and it
   is sensitive to any error in propagation that a single-case test would
   miss: get the dilution rule wrong in one direction and ``3^n - 2^n``
   immediately disagrees.

3. **A timing sandwich from independently measured quantities.** The harness
   reports its in-bracket overhead as the difference of two mean timings.
   Independently: a sealed ``Reading`` is constructed inside each timed
   bracket, and at most one clock-reading PAIR surrounds it, so

       cost(one Reading)  <=  in-bracket overhead  <=  cost(one clock pair)

   where both bounds are measured by a different arrangement -- one bracket
   around a block of N operations, rather than N brackets differenced. Neither
   side is computed from the quantity it bounds. This is wall-clock timing on
   a shared two-core host, so it is run over several trials and the HIT RATE
   of each inequality is reported along with the median, rather than one noisy
   trial being asserted. The slack is zero: the inequalities are structural,
   and if they fail often that is information about the host, reported as such.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import itertools  # noqa: E402

import numpy as np  # noqa: E402

from _bootstrap import (  # noqa: E402
    OVERHEAD_REPEATS,
    OVERHEAD_TRIALS,
    Tee,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from boardprobe import (  # noqa: E402
    ORDER,
    ORIN_NANO,
    LiveSource,
    Provenance,
    characterise_overhead,
    format_table,
    meet_all,
)

#: The Hasse diagram of the chain: the COVERING relation only, with no
#: transitive edges and no ranks. Everything in check 1 is derived from this.
COVERS: tuple[tuple[Provenance, Provenance], ...] = (
    (Provenance.SIMULATED, Provenance.RECORDED),
    (Provenance.RECORDED, Provenance.LIVE),
)


def warshall_closure() -> dict[tuple[Provenance, Provenance], bool]:
    """Reflexive-transitive closure of :data:`COVERS` by Warshall's algorithm."""
    reach = {(a, b): a is b for a in ORDER for b in ORDER}
    for lower, upper in COVERS:
        reach[(lower, upper)] = True
    for middle in ORDER:
        for left in ORDER:
            for right in ORDER:
                if reach[(left, middle)] and reach[(middle, right)]:
                    reach[(left, right)] = True
    return reach


def order_theoretic_meet(
    assignment: tuple[Provenance, ...],
    reach: dict[tuple[Provenance, Provenance], bool],
) -> Provenance:
    """The unique element of ``assignment`` below every element of it.

    No ranks, no ``min``: the definition of a greatest lower bound that happens
    to lie inside the set, which in a chain it always does.
    """
    candidates = [
        candidate
        for candidate in set(assignment)
        if all(reach[(candidate, other)] for other in assignment)
    ]
    if len(candidates) != 1:
        raise AssertionError(
            f"the closure relation gives {len(candidates)} lower bounds inside "
            f"{[p.value for p in assignment]}; a chain must give exactly one"
        )
    return candidates[0]


def main() -> int:
    out = Tee("validate_independent_bound_output.txt")
    out.rule("boardprobe -- three checks against mathematics the code does not use")
    for line in environment_lines():
        out(line)

    out.rule("1. the chain re-derived from its Hasse diagram by Warshall's algorithm")
    out("Warshall 1962, JACM 9(1), 11-12. The input is the COVERING relation")
    out("only -- two edges -- and the meet is then defined order-theoretically.")
    out("No integer rank appears anywhere in this check.")
    out("")
    reach = warshall_closure()
    out(f"covering edges supplied : {[(a.value, b.value) for a, b in COVERS]}")
    out("")
    out(format_table(
        ["a <= b ?", *[p.value for p in ORDER]],
        [[a.value, *["yes" if reach[(a, b)] else "no" for b in ORDER]] for a in ORDER],
    ))
    out("")
    out.check(
        "the closure is reflexive",
        all(reach[(a, a)] for a in ORDER),
        "3 elements",
    )
    out.check(
        "the closure is transitive",
        all(
            reach[(a, c)]
            for a, b, c in itertools.product(ORDER, repeat=3)
            if reach[(a, b)] and reach[(b, c)]
        ),
        "27 triples",
    )
    out.check(
        "the closure is antisymmetric, so it is a partial order",
        all(
            a is b
            for a, b in itertools.product(ORDER, repeat=2)
            if reach[(a, b)] and reach[(b, a)]
        ),
        "9 pairs",
    )
    out.check(
        "the closure is a TOTAL order, so the structure is a chain",
        all(
            reach[(a, b)] or reach[(b, a)]
            for a, b in itertools.product(ORDER, repeat=2)
        ),
        "9 pairs comparable",
    )
    rows = []
    total = 0
    disagreements = 0
    for length in range(1, 7):
        count = 0
        local = 0
        for assignment in itertools.product(ORDER, repeat=length):
            count += 1
            total += 1
            if order_theoretic_meet(assignment, reach) is not meet_all(assignment):
                local += 1
                disagreements += 1
        rows.append([str(length), str(count), str(local)])
    out("")
    out(format_table(["length", "assignments", "disagreements"], rows))
    out("")
    out.check(
        "the order-theoretic meet agrees with meet_all on every assignment",
        disagreements == 0,
        f"{total} assignments, {disagreements} disagreements",
    )

    out.rule("2. the propagation counted by inclusion and exclusion")
    out("For n readings over three provenances, the closed forms are")
    out("")
    out("    #{meet = simulated} = 3^n - 2^n     (at least one simulated)")
    out("    #{meet = recorded}  = 2^n - 1       (none simulated, one recorded)")
    out("    #{meet = live}      = 1             (all live)")
    out("")
    out("summing to 3^n. Exact integer arithmetic; no tolerance.")
    out("")
    rows = []
    mismatches = 0
    enumerated = 0
    for length in range(1, 9):
        counts = dict.fromkeys(ORDER, 0)
        for assignment in itertools.product(ORDER, repeat=length):
            counts[meet_all(assignment)] += 1
            enumerated += 1
        expected = {
            Provenance.SIMULATED: 3**length - 2**length,
            Provenance.RECORDED: 2**length - 1,
            Provenance.LIVE: 1,
        }
        agree = counts == expected and sum(counts.values()) == 3**length
        mismatches += 0 if agree else 1
        rows.append([
            str(length),
            f"{counts[Provenance.SIMULATED]} / {expected[Provenance.SIMULATED]}",
            f"{counts[Provenance.RECORDED]} / {expected[Provenance.RECORDED]}",
            f"{counts[Provenance.LIVE]} / {expected[Provenance.LIVE]}",
            str(3**length),
            "ok" if agree else "MISMATCH",
        ])
    out(format_table(
        ["n", "simulated obs/exp", "recorded obs/exp", "live obs/exp", "3^n", "verdict"],
        rows,
    ))
    out("")
    out(f"assignments enumerated : {enumerated}")
    out.check(
        "the enumerated propagation matches the inclusion-exclusion closed form "
        "for n = 1..8",
        mismatches == 0,
        f"{enumerated} assignments, {mismatches} mismatched rows",
    )

    out.rule("3. the harness overhead sandwich, from independently measured bounds")
    out("    cost(one sealed Reading)  <=  in-bracket overhead  <=  cost(one clock pair)")
    out("")
    out("The left bound holds because exactly one Reading is constructed inside")
    out("each timed bracket. The right bound holds because the overhead inside a")
    out("bracket cannot exceed the cost of the pair of reads that delimits it.")
    out("Both bounds are measured by a DIFFERENT arrangement from the quantity")
    out("they bound: one bracket around a block of N operations, rather than N")
    out("brackets differenced. Wall-clock on a shared two-core host, so the hit")
    out("rate over several trials is reported rather than one trial asserted.")
    out("")
    out("EVERY NUMBER IN THIS SECTION IS A PROPERTY OF THE BUILD HOST, NOT OF AN")
    out("NVIDIA JETSON ORIN NANO. The declared target was not reachable.")
    out("")
    clock = LiveSource(ORIN_NANO)
    rows = []
    lower_hits = 0
    upper_hits = 0
    in_bracket = []
    reading_cost = []
    pair_cost = []
    for trial in range(OVERHEAD_TRIALS):
        report = characterise_overhead(clock, repeats=OVERHEAD_REPEATS)
        lower_ok = report.reading_cost_s <= report.in_bracket_overhead_s
        upper_ok = report.in_bracket_overhead_s <= report.clock_pair_cost_s
        lower_hits += int(lower_ok)
        upper_hits += int(upper_ok)
        in_bracket.append(report.in_bracket_overhead_s)
        reading_cost.append(report.reading_cost_s)
        pair_cost.append(report.clock_pair_cost_s)
        rows.append([
            str(trial),
            f"{report.reading_cost_s * 1e6:.3f}",
            f"{report.in_bracket_overhead_s * 1e6:.3f}",
            f"{report.clock_pair_cost_s * 1e6:.3f}",
            "ok" if lower_ok else "MISS",
            "ok" if upper_ok else "MISS",
            f"{report.perturbation_ratio:.4f}",
        ])
    out(format_table(
        ["trial", "Reading us", "in-bracket us", "clock pair us", "lower", "upper",
         "perturbation"],
        rows,
    ))
    out("")
    median_reading = float(np.median(reading_cost))
    median_bracket = float(np.median(in_bracket))
    median_pair = float(np.median(pair_cost))
    out(f"trials                        : {OVERHEAD_TRIALS} at "
        f"{OVERHEAD_REPEATS} repeats each")
    out(f"median cost of one Reading    : {median_reading * 1e6:.3f} us")
    out(f"median in-bracket overhead    : {median_bracket * 1e6:.3f} us")
    out(f"median cost of one clock pair : {median_pair * 1e6:.3f} us")
    out(f"lower bound hit rate          : {lower_hits}/{OVERHEAD_TRIALS}")
    out(f"upper bound hit rate          : {upper_hits}/{OVERHEAD_TRIALS}")
    out("")
    out.check(
        "the median in-bracket overhead lies inside the independently measured "
        "sandwich",
        median_reading <= median_bracket <= median_pair,
        f"{median_reading * 1e6:.3f} <= {median_bracket * 1e6:.3f} <= "
        f"{median_pair * 1e6:.3f} us",
    )
    out.check(
        "the upper bound holds on a majority of trials",
        upper_hits * 2 > OVERHEAD_TRIALS,
        f"{upper_hits}/{OVERHEAD_TRIALS}",
    )
    out.check(
        "the lower bound holds on a majority of trials",
        lower_hits * 2 > OVERHEAD_TRIALS,
        f"{lower_hits}/{OVERHEAD_TRIALS}",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
