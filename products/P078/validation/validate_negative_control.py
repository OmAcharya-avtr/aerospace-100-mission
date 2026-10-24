"""The negative control: sequences with no violations must report none.

A checker that reports a violation on a clean sequence is worse than useless,
because the next clean sequence it flags is the one an operator stops reading.
This script measures the false-alarm rate of the forward walk on two
populations:

1. The two hand-constructed clean cases in ``examples/cases/``.
2. Seeded generated sequences, built by rejection against forward execution,
   over a range of generator shapes.

The second population is clean *by construction*, which is a weaker claim than
it sounds: the generator uses the same executor that is being tested, so a
false alarm the executor produces consistently would be invisible. The
independent part is the structure -- the generator never looks at the
expectations in this script, and the hand-constructed cases were written before
the generator existed and are checked against hand arithmetic in
``validate_forward_known.py``. Both are reported; neither on its own is proof.
"""

from __future__ import annotations

import _bootstrap
from _bootstrap import Tee, Timer, header, relative

ROOT = _bootstrap.bootstrap()

from cmdguard.dataset import GeneratorConfig, generate_clean_case  # noqa: E402
from cmdguard.gate import gate_case  # noqa: E402
from cmdguard.loader import load_case  # noqa: E402
from cmdguard.model import EffectPolicy  # noqa: E402
from cmdguard.trace import forward_execute  # noqa: E402

CASES = ROOT / "examples" / "cases"
HAND_CLEAN = ("nominal_science_pass", "irreversible_after_verify")

SHAPES = (
    ("default", GeneratorConfig()),
    ("short", GeneratorConfig(n_steps=12)),
    ("long", GeneratorConfig(n_steps=64)),
    ("mostly unverified", GeneratorConfig(unverified_fraction=0.6)),
    ("fully verified", GeneratorConfig(unverified_fraction=0.0)),
    ("tight spacing", GeneratorConfig(dt_min_s=1.0, dt_max_s=12.0)),
    ("wide jitter", GeneratorConfig(draw_jitter=0.4, hold_jitter=0.5, limit_jitter=0.3)),
)

N_SEEDS = 60


def main() -> int:
    tee = Tee("validate_negative_control_output.txt")
    header(tee, "cmdguard -- negative control: no violation must be reported")

    tee.rule("1. hand-constructed clean cases")
    for name in HAND_CLEAN:
        case = load_case(CASES / f"{name}.yaml")
        report = forward_execute(case)
        gate = gate_case(case)
        tee(f"  {relative(CASES / f'{name}.yaml')}")
        tee(f"    steps {len(case.sequence)}, span {case.sequence.duration_s:.1f} s")
        tee(f"    violations {report.n_violations}, gate {gate.verdict.value}")
        tee.check(f"{name} reports no violation", report.n_violations == 0,
                  str(report.kind_counts()))
        tee.check(f"{name} is not refused by the gate", not gate.refused)
        blocked = forward_execute(case, EffectPolicy.BLOCK)
        tee.check(f"{name} is clean under both effect policies", blocked.clean)

    tee.rule("2. seeded generated sequences, per generator shape")
    tee(f"  seeds per shape: {N_SEEDS}")
    tee("")
    tee(f"  {'shape':<20s} {'cases':>6s} {'steps':>7s} {'false alarms':>13s} "
        f"{'rate':>10s} {'mean gen ms':>12s}")
    total_cases = 0
    total_alarms = 0
    total_steps = 0
    rows: list[tuple[str, int, int, int]] = []
    for label, config in SHAPES:
        timer = Timer()
        alarms = 0
        steps = 0
        for seed in range(N_SEEDS):
            with timer:
                case = generate_clean_case(seed, config)
            report = forward_execute(case)
            steps += len(case.sequence)
            if not report.clean:
                alarms += 1
                tee(f"    FALSE ALARM on {label} seed {seed}: {report.kind_counts()}")
        rows.append((label, N_SEEDS, steps, alarms))
        total_cases += N_SEEDS
        total_alarms += alarms
        total_steps += steps
        tee(f"  {label:<20s} {N_SEEDS:>6d} {steps:>7d} {alarms:>13d} "
            f"{alarms / N_SEEDS:>10.6f} {timer.mean_ms:>12.3f}")
    tee("")
    tee(f"  totals: {total_cases} cases, {total_steps} command occurrences, "
        f"{total_alarms} false alarm(s)")
    tee(f"  false-alarm rate per case : {total_alarms / total_cases:.6f}")
    tee(f"  false-alarm rate per step : {total_alarms / total_steps:.9f}")
    tee.check("no generated clean sequence reports a violation", total_alarms == 0,
              f"{total_alarms} of {total_cases}")
    tee("")
    tee("  the generation times above are wall clock on a shared two-core container.")
    tee("  they are reported in this table only and no conclusion is drawn from them.")

    tee.rule("3. the gate on the clean population")
    refusals = 0
    for _label, config in SHAPES:
        for seed in range(N_SEEDS):
            if gate_case(generate_clean_case(seed, config)).refused:
                refusals += 1
    tee(f"  gate refusals over {total_cases} clean cases: {refusals}")
    tee.check("the gate refuses no clean sequence", refusals == 0, str(refusals))

    tee.rule("4. what this measurement does not establish")
    tee("  The generated population is clean by construction and the construction uses")
    tee("  the executor under test, so a false alarm the executor raises consistently")
    tee("  would not appear here. The hand-constructed cases are the independent part,")
    tee("  and there are two of them. A zero false-alarm rate over 420 generated cases")
    tee("  is evidence about consistency, not about correctness.")

    code = tee.summarise()
    tee(f"transcript written to {tee.save()}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
