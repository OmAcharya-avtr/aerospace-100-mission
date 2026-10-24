"""The irreversible-command gate must refuse, and must refuse for the right reason.

Four populations:

1. The hand-constructed refusal case and its corrected counterpart.
2. Every ordering of one irreversible command against its own verification
   step, enumerated, so the refusal boundary is measured rather than asserted
   at one point.
3. Seeded generated sequences with the ``irreversible_early`` defect injected.
4. The clean generated population, on which the gate must refuse nothing.

The gate's clean verdict is ``NO_REFUSAL_FOUND`` and this script prints that
string rather than any synonym for safe, because there is no measurement here
that would support one.
"""

from __future__ import annotations

import _bootstrap
from _bootstrap import Tee, header, relative

ROOT = _bootstrap.bootstrap()

from cmdguard.dataset import generate_clean_case  # noqa: E402
from cmdguard.defects import inject  # noqa: E402
from cmdguard.gate import Verdict, gate_case  # noqa: E402
from cmdguard.loader import load_case  # noqa: E402
from cmdguard.model import (  # noqa: E402
    Case,
    CommandDictionary,
    CommandSequence,
    CommandSpec,
    EffectPolicy,
    InitialState,
    ModeMachine,
    ResourceLimits,
    SequenceStep,
    Verity,
)
from cmdguard.trace import forward_execute  # noqa: E402

CASES = ROOT / "examples" / "cases"
N_SEEDS = 80


def main() -> int:
    tee = Tee("validate_irreversible_gate_output.txt")
    header(tee, "cmdguard -- the irreversible-command gate must REFUSE")

    tee.rule("1. the hand-constructed pair")
    for name, expected in (
        ("irreversible_before_verify", Verdict.REFUSE),
        ("irreversible_after_verify", Verdict.NO_REFUSAL_FOUND),
    ):
        case = load_case(CASES / f"{name}.yaml")
        result = gate_case(case)
        tee("")
        tee(f"  {relative(CASES / f'{name}.yaml')}")
        for line in result.render().splitlines():
            tee(f"    {line}")
        tee.check(f"{name} verdict is {expected.value}", result.verdict is expected,
                  result.verdict.value)
        tee.check(f"{name} exit code is {2 if expected is Verdict.REFUSE else 0}",
                  result.exit_code == (2 if expected is Verdict.REFUSE else 0))
    tee("")
    refused = gate_case(load_case(CASES / "irreversible_before_verify.yaml"))
    tee.check("the refusal names the irreversible command",
              refused.blocking[0].command == "ANT_DEPLOY")
    tee.check("the refusal's trace names the unverified declaration",
              "attitude_verified=unknown" in refused.blocking[0].trace[0].note)
    for policy in EffectPolicy:
        under = gate_case(load_case(CASES / "irreversible_before_verify.yaml"), policy)
        tee.check(f"refusal holds under effect policy {policy.value}", under.refused)

    tee.rule("2. every ordering of one irreversible command against its verification")
    tee("  FIRE is irreversible and requires p=true. VERIFY_P sets p=true. p is")
    tee("  unverified in the declared initial state. FILLER does nothing. All six")
    tee("  orderings of the three commands are enumerated; the gate must refuse")
    tee("  exactly those in which FIRE precedes VERIFY_P.")
    commands = [
        CommandSpec(identifier="FIRE", preconditions={"p": Verity.TRUE}, irreversible=True),
        CommandSpec(identifier="VERIFY_P", effects={"p": Verity.TRUE}),
        CommandSpec(identifier="FILLER"),
    ]
    machine = ModeMachine(modes=("NOMINAL",), transitions={"NOMINAL": ()})
    from itertools import permutations

    tee("")
    tee(f"  {'ordering':<34s} {'fire before verify':>19s} {'verdict':>18s}")
    mismatches = 0
    for ordering in permutations(["FIRE", "VERIFY_P", "FILLER"]):
        case = Case(
            name="gate_ordering",
            dictionary=CommandDictionary(
                commands={spec.identifier: spec for spec in commands},
                resource_limits=ResourceLimits({}),
            ),
            machine=machine,
            initial=InitialState(mode="NOMINAL", predicates={"p": Verity.UNKNOWN}),
            sequence=CommandSequence(
                tuple(
                    SequenceStep(t_s=float(10 * index), command=command)
                    for index, command in enumerate(ordering)
                )
            ),
        )
        result = gate_case(case)
        fire_first = ordering.index("FIRE") < ordering.index("VERIFY_P")
        tee(f"  {' -> '.join(ordering):<34s} {str(fire_first):>19s} "
            f"{result.verdict.value:>18s}")
        if result.refused != fire_first:
            mismatches += 1
    tee.check("the gate refuses exactly the orderings that put FIRE first",
              mismatches == 0, f"{mismatches} mismatch(es) of 6")

    tee.rule("3. a false precondition is not escalated to the gate")
    tee("  The gate exists for the UNVERIFIED case. A precondition that is verified")
    tee("  FALSE is caught by any per-command check, so it stays an ordinary finding.")
    case = Case(
        name="gate_false_precondition",
        dictionary=CommandDictionary(
            commands={"FIRE": commands[0]}, resource_limits=ResourceLimits({})
        ),
        machine=machine,
        initial=InitialState(mode="NOMINAL", predicates={"p": Verity.FALSE}),
        sequence=CommandSequence((SequenceStep(t_s=0.0, command="FIRE"),)),
    )
    report = forward_execute(case)
    tee(f"  kinds: {report.kind_counts()}")
    tee.check("a false precondition is precondition_false",
              report.kind_counts() == {"precondition_false": 1})
    tee.check("and the gate does not refuse on it", not gate_case(case).refused)

    tee.rule("4. generated sequences with the irreversible_early defect injected")
    applied = 0
    noop = 0
    refusals = 0
    gate_kind_present = 0
    for seed in range(N_SEEDS):
        clean = generate_clean_case(seed)
        defective, record = inject(clean, "irreversible_early", seed)
        if not record.applied:
            noop += 1
            continue
        applied += 1
        report = forward_execute(defective)
        if report.gate_blocking:
            gate_kind_present += 1
        if gate_case(defective).refused:
            refusals += 1
    tee(f"  seeds attempted                      : {N_SEEDS}")
    tee(f"  injections applied                   : {applied}")
    tee(f"  injections that were no-ops          : {noop}")
    tee(f"  applied cases with a gate-blocking kind: {gate_kind_present}")
    tee(f"  applied cases the gate refused       : {refusals}")
    tee.check("the gate refuses every case that carries a gate-blocking kind",
              refusals == gate_kind_present, f"{refusals} vs {gate_kind_present}")
    tee("")
    tee("  Not every applied injection produces a gate-blocking violation, and the")
    tee("  count above says how many did. Moving an irreversible command ahead of a")
    tee("  step that re-asserts an already-true predicate changes nothing, which is")
    tee("  why the no-op and the applied-but-not-blocking counts are both printed")
    tee("  rather than the injection being retried until it bites.")

    tee.rule("5. the clean population: the gate must refuse nothing")
    clean_refusals = 0
    for seed in range(N_SEEDS):
        if gate_case(generate_clean_case(seed)).refused:
            clean_refusals += 1
    tee(f"  clean cases checked : {N_SEEDS}")
    tee(f"  refusals            : {clean_refusals}")
    tee.check("the gate refuses no clean generated sequence", clean_refusals == 0,
              str(clean_refusals))

    tee.rule("6. what the clean verdict is called, and why")
    tee("  The gate's two values are REFUSE and NO_REFUSAL_FOUND. The second is named")
    tee("  for what was measured: no gate-blocking violation was found against the")
    tee("  declared model. Nothing in this repository measures whether a sequence is")
    tee("  safe to uplink, so nothing in this repository says so.")

    code = tee.summarise()
    tee(f"transcript written to {tee.save()}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
