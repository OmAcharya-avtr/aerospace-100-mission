"""Seeded reproducibility: identical seeds must give identical output.

Three levels are checked, because they fail in different ways:

1. one simulated run -- every field of the result, compared field by field;
2. one Monte Carlo ensemble -- the whole delivery-time array and the quantile
   table;
3. one generated scenario table, including the simulated target column, which
   is what the learned component trains on.

A fourth check runs the ensemble *interleaved* with other ensembles on other
scenarios, to confirm that the result does not depend on how many random draws
some other part of the process has made. A simulator that shares one global
generator passes checks 1 to 3 and fails this one.
"""

from __future__ import annotations

from dataclasses import fields

from _bootstrap import Tee, bootstrap, environment_lines, relative

bootstrap()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from relfiledelivery.dataset import generate_scenario, generate_table  # noqa: E402
from relfiledelivery.link import LinkConditions, OutageModel, uniform_windows  # noqa: E402
from relfiledelivery.modes import ALL_MODES  # noqa: E402
from relfiledelivery.montecarlo import run_ensemble  # noqa: E402
from relfiledelivery.payload import FileTransferSpec  # noqa: E402
from relfiledelivery.simulate import TransferResult, simulate_transfer  # noqa: E402

out = Tee("validate_reproducibility_output.txt")

SPEC = FileTransferSpec(77 * 512 - 1, 512, 12)
LINKS = {
    "clean continuous": LinkConditions(1.0e5, 1.0e4, 0.5, 0.0),
    "lossy continuous": LinkConditions(1.0e5, 1.0e4, 0.5, 0.20),
    "windowed bursty": LinkConditions(
        1.0e5,
        1.0e4,
        0.25,
        0.05,
        outage=OutageModel(mean_up_s=10.0, mean_down_s=4.0),
        windows=uniform_windows(8, 20.0, 40.0),
    ),
}

out.rule("environment")
for line in environment_lines():
    out(line)

out.rule("check 1: every field of a single run")
out(f"fields compared per run        : {len(fields(TransferResult))}")
out("")
header = f"{'link':>19}{'mode':>16}{'seeds':>7}{'identical':>11}{'differing fields':>18}"
out(header)
out("-" * len(header))
field_failures: list[str] = []
for link_name, link in LINKS.items():
    for mode in ALL_MODES:
        identical = 0
        differing: set[str] = set()
        for seed in range(25):
            first = simulate_transfer(SPEC, link, mode, seed)
            second = simulate_transfer(SPEC, link, mode, seed)
            names = [
                f.name
                for f in fields(TransferResult)
                if getattr(first, f.name) != getattr(second, f.name)
            ]
            if names:
                differing.update(names)
                field_failures.append(f"{link_name}/{mode.value}/seed {seed}: {names}")
            else:
                identical += 1
        label = ",".join(sorted(differing)) if differing else "none"
        out(f"{link_name:>19}{mode.value:>16}{25:>7}{identical:>11}{label:>18}")
out("")
out(f"runs compared                  : {len(LINKS) * len(ALL_MODES) * 25}")
out(f"field mismatches               : {len(field_failures)}")

out.rule("check 2: a whole Monte Carlo ensemble")
out("")
header = f"{'link':>19}{'mode':>16}{'runs':>7}{'array equal':>13}{'quantiles equal':>17}"
out(header)
out("-" * len(header))
ensemble_failures: list[str] = []
for link_name, link in LINKS.items():
    for mode in ALL_MODES:
        first = run_ensemble(SPEC, link, mode, 150, 4242)
        second = run_ensemble(SPEC, link, mode, 150, 4242)
        array_equal = bool(np.array_equal(first.delivery_times_s, second.delivery_times_s))
        quantiles_equal = first.quantiles == second.quantiles
        if not (array_equal and quantiles_equal):  # pragma: no cover
            ensemble_failures.append(f"{link_name}/{mode.value}")
        out(
            f"{link_name:>19}{mode.value:>16}{150:>7}"
            f"{str(array_equal):>13}{str(quantiles_equal):>17}"
        )
out("")
out(f"ensemble mismatches            : {len(ensemble_failures)}")

out.rule("check 3: the generated scenario table, target column included")
first_table = generate_table(60, 515151)
second_table = generate_table(60, 515151)
try:
    pd.testing.assert_frame_equal(first_table, second_table)
    table_equal = True
    table_note = "every column identical, including the simulated delivery time"
except AssertionError as error:  # pragma: no cover
    table_equal = False
    table_note = str(error).splitlines()[0]
out(f"rows                           : {len(first_table)}")
out(f"columns                        : {len(first_table.columns)}")
out(f"identical                      : {table_equal}")
out(f"note                           : {table_note}")
out("")
out("Scenario construction is checked separately from the simulated target, so a")
out("generator that is reproducible and a simulation that is not can be told")
out("apart.")
scenario_stable = all(
    generate_scenario(seed, regime).link == generate_scenario(seed, regime).link
    and generate_scenario(seed, regime).spec == generate_scenario(seed, regime).spec
    for seed in range(40)
    for regime in ("no_outage", "correlated_outage")
)
out(f"scenario construction stable   : {scenario_stable} over 80 scenarios")

out.rule("check 4: independence from the surrounding call order")
out("Each run derives its own generators from its own seed through a numpy")
out("SeedSequence, so a run must give the same answer whether it is made first,")
out("last, or interleaved with unrelated work. A simulator sharing one global")
out("generator passes checks 1 to 3 and fails this one.")
out("")
reference = {}
for link_name, link in LINKS.items():
    for mode in ALL_MODES:
        reference[(link_name, mode)] = simulate_transfer(SPEC, link, mode, 31337)

# Now interleave: make unrelated draws and unrelated simulations between the
# repeats, in a different order.
noise_rng = np.random.default_rng(1)
interleaved_failures: list[str] = []
pairs = [(name, mode) for name in LINKS for mode in ALL_MODES]
for link_name, mode in reversed(pairs):
    noise_rng.random(1000)
    simulate_transfer(FileTransferSpec(2048, 256, 12), LINKS["lossy continuous"], mode, 11)
    generate_scenario(5, "correlated_outage")
    repeat = simulate_transfer(SPEC, LINKS[link_name], mode, 31337)
    if repeat != reference[(link_name, mode)]:  # pragma: no cover
        interleaved_failures.append(f"{link_name}/{mode.value}")
out(f"pairs compared                 : {len(pairs)}")
out(f"order-dependent results        : {len(interleaved_failures)}")
for item in interleaved_failures:
    out(f"  FAILED {item}")

out.rule("different seeds do produce different runs")
out("Reproducibility is only interesting if the seed does something. If two")
out("different seeds gave the same run, every check above would pass trivially.")
out("")
out("Unacknowledged mode sends every unit exactly once whatever the seed, so its")
out("transmission count is deterministic by construction and the quantity that")
out("has to vary there is how many units ARRIVED. The first run of this check used")
out("the transmission count for all three modes and reported a false failure on")
out("all three unacknowledged rows. The criterion column below names which")
out("quantity each row is judged on.")
out("")
header = (
    f"{'link':>19}{'mode':>16}{'distinct tx':>13}{'distinct recv':>15}"
    f"{'distinct times':>16}{'criterion':>12}{'varies':>8}"
)
out(header)
out("-" * len(header))
seed_sensitive = True
for link_name, link in LINKS.items():
    for mode in ALL_MODES:
        results = [simulate_transfer(SPEC, link, mode, seed) for seed in range(60)]
        counts = len({r.forward_transmissions for r in results})
        received = len({r.received_count for r in results})
        times = len({r.delivery_time_s for r in results if r.delivered})
        stochastic = link.pdu_loss_probability > 0.0 or not link.outage.is_outage_free
        if mode.retransmits:
            criterion, value = "tx", counts
        else:
            criterion, value = "received", received
        varies = value > 1
        if stochastic:
            seed_sensitive = seed_sensitive and varies
        out(
            f"{link_name:>19}{mode.value:>16}{counts:>13}{received:>15}"
            f"{times:>16}{criterion:>12}{str(varies):>8}"
        )
out("")
out("The clean continuous link is deterministic by construction -- no loss and no")
out("outages -- so one distinct value there is the correct answer, not a defect,")
out("and those three rows are excluded from the verdict.")
out(f"seed sensitivity on the stochastic links: {seed_sensitive}")

out.rule("summary")
checks = {
    "every field of a repeated run is identical": not field_failures,
    "a repeated ensemble is identical": not ensemble_failures,
    "a repeated scenario table is identical": table_equal,
    "scenario construction is stable": scenario_stable,
    "results do not depend on the surrounding call order": not interleaved_failures,
    "different seeds give different runs on stochastic links": seed_sensitive,
}
for name, passed in checks.items():
    out(f"{'PASS' if passed else 'FAIL'}  {name}")
out("")
out(f"output written to {relative(out.path)}")
out.save()
if not all(checks.values()):
    raise SystemExit("validate_reproducibility.py: at least one check FAILED")
