"""The lattice laws, checked exhaustively rather than sampled.

``tests/test_provenance_properties.py`` checks these with Hypothesis. This
script checks them by enumerating every case, which for a three-element chain
is cheap and complete: 9 pairs, 27 triples, and every non-empty subset. An
exhaustive check and a property test are not the same evidence -- the first
leaves nothing to a sampler -- and both are reported because the property test
is what will catch a change to the lattice's SIZE.

It also prints the meet table and the label rules, which are what a reader of
``README.md`` is being shown, so the figures in the README trace to a script
that was run.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import itertools  # noqa: E402

from _bootstrap import Tee, bootstrap, environment_lines  # noqa: E402

ROOT = bootstrap()

from boardprobe import (  # noqa: E402
    ABSORBING,
    IDENTITY,
    ORDER,
    Label,
    Provenance,
    ProvenanceError,
    format_table,
    join,
    label_for,
    leq,
    meet,
    meet_all,
    rank,
    render_label_rules,
    render_lattice,
)


def main() -> int:
    out = Tee("validate_provenance_lattice_output.txt")
    out.rule("boardprobe -- the provenance lattice, every case enumerated")
    for line in environment_lines():
        out(line)
    out("")
    out("Reference for the axioms: B. A. Davey and H. A. Priestley, Introduction")
    out("to Lattices and Order, 2nd edition, Cambridge University Press 2002,")
    out("chapter 2, definitions 2.1-2.8. Nothing here is novel mathematics; what")
    out("is engineering is that this package refuses to let a figure escape")
    out("without it.")

    out.rule("1. the lattice, as the README prints it")
    out(render_lattice())
    out("")
    out(render_label_rules())
    out("")
    out(f"chain, weakest first : {[p.value for p in ORDER]}")
    out(f"ranks                : {[rank(p) for p in ORDER]}")
    out(f"identity of meet     : {IDENTITY.value}")
    out(f"absorbing element    : {ABSORBING.value}")

    pairs = list(itertools.product(ORDER, repeat=2))
    triples = list(itertools.product(ORDER, repeat=3))
    out.rule("2. the semilattice laws, over all 9 pairs and all 27 triples")
    out(f"pairs enumerated   : {len(pairs)}")
    out(f"triples enumerated : {len(triples)}")
    out("")
    out.check(
        "meet is commutative over all 9 pairs",
        all(meet(a, b) is meet(b, a) for a, b in pairs),
        f"{len(pairs)} pairs",
    )
    out.check(
        "meet is associative over all 27 triples",
        all(
            meet(meet(a, b), c) is meet(a, meet(b, c)) for a, b, c in triples
        ),
        f"{len(triples)} triples",
    )
    out.check(
        "meet is idempotent over all 3 elements",
        all(meet(a, a) is a for a in ORDER),
        "3 elements",
    )
    out.check(
        f"{IDENTITY.value} is the identity of meet",
        all(meet(IDENTITY, a) is a and meet(a, IDENTITY) is a for a in ORDER),
        "3 elements, both sides",
    )
    out.check(
        f"{ABSORBING.value} absorbs",
        all(meet(ABSORBING, a) is ABSORBING for a in ORDER),
        "3 elements",
    )
    out.check(
        "meet is monotone in each argument over all 27 triples",
        all(
            leq(meet(a, c), meet(b, c))
            for a, b, c in triples
            if leq(a, b)
        ),
        "every triple with a <= b",
    )
    out.check(
        "meet is the greatest lower bound over all 9 pairs",
        all(
            leq(meet(a, b), a)
            and leq(meet(a, b), b)
            and all(
                leq(d, meet(a, b))
                for d in ORDER
                if leq(d, a) and leq(d, b)
            )
            for a, b in pairs
        ),
        "lower bound, and greatest among them",
    )
    out.check(
        "the absorption laws relate meet and join over all 9 pairs",
        all(
            meet(a, join(a, b)) is a and join(a, meet(a, b)) is a for a, b in pairs
        ),
        "both laws",
    )
    out.check(
        "the lattice is distributive over all 27 triples",
        all(
            meet(a, join(b, c)) is join(meet(a, b), meet(a, c))
            and join(a, meet(b, c)) is meet(join(a, b), join(a, c))
            for a, b, c in triples
        ),
        "both distributive laws",
    )

    out.rule("3. meet_all over every multiset up to length 6")
    rows = []
    total = 0
    disagreements = 0
    for length in range(1, 7):
        count = 0
        for assignment in itertools.product(ORDER, repeat=length):
            count += 1
            total += 1
            folded = meet_all(assignment)
            weakest = min(assignment, key=rank)
            reversed_fold = meet_all(list(reversed(assignment)))
            doubled = meet_all([*assignment, *assignment])
            if not (folded is weakest and reversed_fold is folded and doubled is folded):
                disagreements += 1
        rows.append([str(length), str(count), f"{3 ** length}"])
    out(format_table(["length", "assignments checked", "3^length"], rows))
    out("")
    out.check(
        "meet_all is the weakest element present, order-independent and "
        "duplication-invariant",
        disagreements == 0,
        f"{total} assignments, {disagreements} disagreements",
    )

    out.rule("4. the empty fold is refused, not taken as the identity")
    out("The meet of an empty collection is the monoid identity, which here is")
    out("the STRONGEST element in the lattice. Returning it would let a figure")
    out("with no inputs at all be labelled a measurement, so it raises.")
    out("")
    refused = False
    try:
        meet_all([])
    except ProvenanceError as error:
        refused = True
        out(f"  {type(error).__name__}: {str(error).splitlines()[0][:110]}")
    out.check("meet_all([]) raises rather than returning the identity", refused)

    out.rule("5. only one provenance admits the MEASURED label")
    rows = [
        [p.value, label_for(p).value, "yes" if label_for(p) is Label.MEASURED else "no"]
        for p in reversed(ORDER)
    ]
    out(format_table(["provenance", "label", "may be called MEASURED"], rows))
    out("")
    measured = [p for p in ORDER if label_for(p) is Label.MEASURED]
    out.check(
        "exactly one lattice element maps to MEASURED, and it is live",
        measured == [Provenance.LIVE],
        f"{[p.value for p in measured]}",
    )
    out.check(
        "label_for is a total function on the lattice",
        all(label_for(p) in set(Label) for p in ORDER),
        f"{len(ORDER)} elements, {len(set(Label))} labels",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
