"""The synthetic dataset: determinism, balance, split hygiene and compute budget.

Nothing in this repository is trained on anything but the output of
:func:`contactsched.dataset.generate_instance`. That makes three things worth
measuring before any model score is quoted:

* **determinism.** The same seed must give the same instance and the same label,
  byte for byte in the dataframe, or no reported score is reproducible.
* **balance and split hygiene.** The positive rate, and a check that no
  ``(seed, request_id)`` key appears on both sides of the split.
* **compute budget.** Wall clock for the whole build, so a reader knows what
  reproducing the model costs and can see it is inside the three-minute budget
  this mission works to.

The distributional choices of the generator are also printed, because the
learned model's skill is skill on this generator and a reader has to be able to
judge how far it is from their own traffic.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, Timer, bootstrap, environment_lines

bootstrap()

from contactsched.dataset import (  # noqa: E402
    INCLINATION_POOL,
    MISSION_POOL,
    STATION_POOL,
    build_dataset,
    generate_instance,
    split_by_instance,
)
from contactsched.features import FEATURE_NAMES  # noqa: E402

FAILURES: list[str] = []
TRAIN_SEEDS = range(0, 240)
TEST_SEEDS = range(240, 320)
HORIZON_S = 43200.0


def main() -> None:
    out = Tee("validate_dataset_output.txt")
    out.rule("contactsched validate_dataset")
    for line in environment_lines():
        out(line)

    out.rule("1. The generator's declared distributions")
    out("Every one of these is a choice written down in contactsched/dataset.py, and")
    out("every reported model score is a score on this generator and not on any real")
    out("network's request mix.")
    out("")
    out(f"station pool            : {len(STATION_POOL)} sites")
    for name, lat, lon, alt, floor, turnaround in STATION_POOL:
        out(
            f"  {name:<14} lat {lat:>7.2f}  lon {lon:>8.2f}  alt {alt:>5.2f} km  "
            f"mask floor {floor:>4.1f} deg  turnaround {turnaround:>5.0f} s"
        )
    out(f"inclination pool        : {INCLINATION_POOL} deg")
    out(f"mission pool            : {MISSION_POOL}")
    out("antennas per instance   : 3 to 4, drawn without replacement")
    out("spacecraft per instance : 4 to 6")
    out("requests per instance   : 7 to 11")
    out("priority distribution   : 1,2,3,4 with probabilities 0.2, 0.3, 0.3, 0.2")
    out("contact duration        : one of 180, 300, 420, 600 s")
    out("minimum contacts        : round(U(1, 2 + 2*load)) clipped to [1, 6]")
    out("maximum contacts        : minimum plus 0 or 1")
    out("allowed antennas        : unrestricted with probability 0.55, else a subset")
    out("load knob               : U(0.7, 2.3), scaling the declared minimum contacts")
    out("mean motion             : U(13.9, 15.5) rev/day")
    out("eccentricity            : U(0.0002, 0.004)")
    out("scan step / slot grid   : 45 s / 240 s")

    out.rule("2. Determinism of instance generation")
    mismatches = 0
    for seed in list(TRAIN_SEEDS)[:40]:
        first = generate_instance(seed, horizon_s=HORIZON_S)
        second = generate_instance(seed, horizon_s=HORIZON_S)
        same = (
            [s.name for s in first.stations] == [s.name for s in second.stations]
            and [s.name for s in first.fleet] == [s.name for s in second.fleet]
            and len(first.problem.candidates) == len(second.problem.candidates)
            and first.load == second.load
            and [w.rise_s for w in first.problem.windows]
            == [w.rise_s for w in second.problem.windows]
        )
        mismatches += int(not same)
    out("seeds re-generated : 40")
    out(f"mismatches         : {mismatches}")
    if mismatches:
        FAILURES.append(f"{mismatches} instances are not reproducible from their seed")

    out.rule("3. Building the dataset")
    timer = Timer()
    with timer:
        frame = build_dataset(
            list(TRAIN_SEEDS) + list(TEST_SEEDS), horizon_s=HORIZON_S
        )
    out(f"train seeds        : {TRAIN_SEEDS.start}..{TRAIN_SEEDS.stop - 1}")
    out(f"test seeds         : {TEST_SEEDS.start}..{TEST_SEEDS.stop - 1}")
    out(f"instances          : {len(TRAIN_SEEDS) + len(TEST_SEEDS)}")
    out(f"rows               : {len(frame)}")
    out(f"columns            : {len(frame.columns)}")
    out(f"features           : {len(FEATURE_NAMES)}")
    out(f"positive rate      : {frame['unsatisfiable'].mean():.6f}")
    out(f"positives          : {int(frame['unsatisfiable'].sum())}")
    out("")
    out("Compute budget, wall clock on a shared two-core machine. Unstable, table only:")
    out(f"  dataset build    : {timer.elapsed_s:.2f} s")
    instance_count = len(TRAIN_SEEDS) + len(TEST_SEEDS)
    out(f"  per instance     : {1000.0 * timer.elapsed_s / instance_count:.1f} ms")
    out(f"  cpu cores        : {len(__import__('os').sched_getaffinity(0))}")
    if timer.elapsed_s > 180.0:
        FAILURES.append(
            f"the dataset build took {timer.elapsed_s:.1f} s, above the 180 s budget"
        )
    if frame["unsatisfiable"].nunique() != 2:
        FAILURES.append("the dataset has only one class, so no benchmark is possible")

    out.rule("4. Determinism of the dataframe")
    with Timer():
        repeat = build_dataset(list(TRAIN_SEEDS)[:20], horizon_s=HORIZON_S)
    original = frame[frame["seed"].isin(list(TRAIN_SEEDS)[:20])].reset_index(drop=True)
    identical = original.equals(repeat)
    out(f"20-seed rebuild identical to the slice of the full build : {identical}")
    if not identical:
        FAILURES.append("rebuilding the dataset does not reproduce the same dataframe")

    out.rule("5. Split hygiene")
    train, test = split_by_instance(frame, TRAIN_SEEDS, TEST_SEEDS)
    train_keys = set(zip(train["seed"], train["request_id"], strict=True))
    test_keys = set(zip(test["seed"], test["request_id"], strict=True))
    out(f"train rows                       : {len(train)}")
    out(f"test rows                        : {len(test)}")
    out(f"train positive rate              : {train['unsatisfiable'].mean():.6f}")
    out(f"test positive rate               : {test['unsatisfiable'].mean():.6f}")
    out(f"shared (seed, request_id) keys   : {len(train_keys & test_keys)}")
    out(f"shared seeds                     : {len(set(train['seed']) & set(test['seed']))}")
    if train_keys & test_keys:
        FAILURES.append("the split leaks requests between train and test")
    out("")
    out("The split is by INSTANCE. Requests inside one instance share its network,")
    out("its horizon and its contention, so splitting by request would leak and the")
    out("reported scores would be optimistic. An overlapping seed range raises in")
    out("split_by_instance rather than being caught here.")
    try:
        split_by_instance(frame, range(0, 10), range(5, 15))
    except ValueError as error:
        out(f"overlapping ranges raise: {str(error)[:70]}")
    else:
        FAILURES.append("an overlapping split did not raise")

    out.rule("6. Feature sanity")
    out(f"{'feature':<36} {'min':>12} {'median':>12} {'max':>12} {'non-finite':>11}")
    out("-" * 88)
    for name in FEATURE_NAMES:
        column = frame[name].to_numpy(dtype=float)
        finite = np.isfinite(column)
        if np.isnan(column).any():
            FAILURES.append(f"feature {name} contains NaN")
        shown = column[finite]
        out(
            f"{name:<36} {shown.min():>12.4f} {np.median(shown):>12.4f} "
            f"{shown.max():>12.4f} {int((~finite).sum()):>11d}"
        )
    out("")
    out("Non-finite values are legitimate and are not errors: demand_over_usable is")
    out("infinite for a request with declared demand and no usable window, which is a")
    out("real and highly predictive state. contactsched.features.design_matrix replaces")
    out("them with a finite sentinel of 1.0e6 before any model sees them.")

    out.rule("7. Label composition")
    deficit = frame["capacity_deficit"].to_numpy(dtype=float)
    label = frame["unsatisfiable"].to_numpy(dtype=int)
    easy = deficit > 0
    out(f"{'subset':<52} {'rows':>7} {'positives':>10} {'rate':>8}")
    out("-" * 80)
    for label_text, mask in (
        ("all requests", np.ones_like(easy, dtype=bool)),
        ("capacity_deficit > 0 (asked for more than its passes allow)", easy),
        ("capacity_deficit <= 0 (lost the contention, or won)", ~easy),
    ):
        out(
            f"{label_text:<52} {int(mask.sum()):>7d} {int(label[mask].sum()):>10d} "
            f"{float(label[mask].mean()):>8.4f}"
        )
    out("")
    out("The first subset is trivially predictable: a request asking for more contacts")
    out("than its own passes can carry is unsatisfiable whatever the competition does,")
    out("and the feature capacity_deficit says so directly. Those rows are kept because")
    out("they are a real part of the label, and every model score in")
    out("validate_predictors_output.txt is reported BOTH over all requests and over the")
    out("hard subset, because reporting only the first would flatter every model.")

    out.rule("Result")
    if FAILURES:
        out(f"FAILED {len(FAILURES)} check(s):")
        for failure in FAILURES[:20]:
            out(f"  - {failure}")
    else:
        out("All checks in this script passed.")
    out("")
    out(f"written to {out.save()}")
    if FAILURES:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
