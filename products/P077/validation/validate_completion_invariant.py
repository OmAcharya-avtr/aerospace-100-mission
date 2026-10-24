"""The completion invariant over a wide run array, plus its negative control.

The invariant is that the bytes the receiver holds equal the bytes the sender
started with, verified by SHA-256. It is checked inside the simulator on every
run that completes, so this script does not re-implement the check -- it
*exercises* it over a wide array and then proves it can fail, which is the part
that makes an always-passing check into evidence.

A check that has never been observed to fail is indistinguishable from a check
that does nothing. So the negative control here corrupts one bit of the
reassembly, at the first byte, somewhere in the middle, and at the last byte, in
every mode, and requires the harness to raise.
"""

from __future__ import annotations

import itertools

from _bootstrap import Tee, bootstrap, environment_lines, relative

bootstrap()

from relfiledelivery.link import LinkConditions, OutageModel, uniform_windows  # noqa: E402
from relfiledelivery.modes import ALL_MODES  # noqa: E402
from relfiledelivery.payload import (  # noqa: E402
    CompletionInvariantError,
    FileTransferSpec,
    digest,
    source_payload,
)
from relfiledelivery.simulate import simulate_transfer  # noqa: E402

out = Tee("validate_completion_invariant_output.txt")

SPECS = {
    "950B/100B": FileTransferSpec(950, 100, 12),
    "10kB/256B": FileTransferSpec(40 * 256, 256, 12),
    "39kB/512B": FileTransferSpec(77 * 512 - 1, 512, 12),
    "205kB/1024B": FileTransferSpec(200 * 1024 + 37, 1024, 12),
}

LINKS = {
    "clean continuous": LinkConditions(1.0e4, 1.0e3, 0.5, 0.0),
    "lossy continuous": LinkConditions(5.0e4, 5.0e3, 0.5, 0.20),
    "bursty continuous": LinkConditions(
        5.0e4, 5.0e3, 0.25, 0.05, outage=OutageModel(mean_up_s=8.0, mean_down_s=3.0)
    ),
    "windowed bursty": LinkConditions(
        5.0e4,
        5.0e3,
        0.25,
        0.05,
        outage=OutageModel(mean_up_s=15.0, mean_down_s=5.0),
        windows=uniform_windows(10, 60.0, 120.0),
    ),
    "growing NAK": LinkConditions(5.0e4, 5.0e3, 0.5, 0.15, nak_bytes_per_segment=4.0),
}

SEEDS = tuple(range(40))

out.rule("environment")
for line in environment_lines():
    out(line)

out.rule("run array")
out(f"files                          : {len(SPECS)}")
out(f"links                          : {len(LINKS)}")
out(f"modes                          : {len(ALL_MODES)}")
out(f"seeds per cell                 : {len(SEEDS)}")
total_runs = len(SPECS) * len(LINKS) * len(ALL_MODES) * len(SEEDS)
out(f"total runs                     : {total_runs}")

out.rule("digest comparison on every delivered run")
header = (
    f"{'file':>13}{'link':>19}{'mode':>16}{'delivered':>11}"
    f"{'digest match':>14}{'mismatch':>10}"
)
out(header)
out("-" * len(header))

delivered_total = 0
undelivered_total = 0
mismatches: list[str] = []
no_digest_violations: list[str] = []

for spec_name, link_name, mode in itertools.product(SPECS, LINKS, ALL_MODES):
    spec = SPECS[spec_name]
    link = LINKS[link_name]
    delivered = 0
    matched = 0
    local_mismatch = 0
    for seed in SEEDS:
        result = simulate_transfer(spec, link, mode, seed)
        if result.delivered:
            delivered += 1
            delivered_total += 1
            if result.delivered_digest == result.source_digest:
                matched += 1
            else:  # pragma: no cover - the simulator raises before reaching here
                local_mismatch += 1
                mismatches.append(f"{spec_name}/{link_name}/{mode.value}/seed {seed}")
        else:
            undelivered_total += 1
            if result.delivered_digest is not None:
                no_digest_violations.append(
                    f"{spec_name}/{link_name}/{mode.value}/seed {seed}"
                )
    out(
        f"{spec_name:>13}{link_name:>19}{mode.value:>16}"
        f"{delivered:>11}{matched:>14}{local_mismatch:>10}"
    )

out("")
out(f"runs delivered                 : {delivered_total} of {total_runs}")
out(f"runs not delivered             : {undelivered_total}")
out(f"digest mismatches              : {len(mismatches)}")
out(f"undelivered runs with a digest : {len(no_digest_violations)}")
out("")
out("An undelivered run has no reassembly and must report no digest at all;")
out("reporting one would mean the harness had hashed a partial file.")

out.rule("the array exercises both branches")
out("A test that only ever sees delivered runs cannot show that the undelivered")
out("branch is handled, and the reverse. Both counts above must be positive.")
both_branches = delivered_total > 0 and undelivered_total > 0
out(f"verdict                        : {'PASS' if both_branches else 'FAIL'}")

out.rule("negative control: one flipped bit must be caught")
out("inject_corruption_at flips one bit of the reassembly after a successful")
out("transfer and before hashing. Every cell below must raise.")
out("")
header = f"{'mode':>16}{'byte offset':>14}{'raised':>9}{'exception':>28}"
out(header)
out("-" * len(header))
control_spec = SPECS["950B/100B"]
control_link = LINKS["clean continuous"]
control_caught = 0
control_total = 0
control_failures: list[str] = []
for mode in ALL_MODES:
    for offset in (0, 1, 474, 949, 100_000):
        control_total += 1
        try:
            simulate_transfer(
                control_spec, control_link, mode, 7, inject_corruption_at=offset
            )
        except CompletionInvariantError as error:
            control_caught += 1
            label = type(error).__name__
            out(f"{mode.value:>16}{offset:>14}{'yes':>9}{label:>28}")
        else:  # pragma: no cover - a silent pass is the failure being guarded
            control_failures.append(f"{mode.value}/offset {offset}")
            out(f"{mode.value:>16}{offset:>14}{'NO':>9}{'none':>28}")
out("")
out(f"corruptions caught             : {control_caught} of {control_total}")
out("The 100000 offset is past the end of a 950-byte file and wraps, so the hook")
out("cannot silently no-op on an out-of-range offset.")

out.rule("the source payload itself is reproducible")
out("The invariant compares the reassembly against the source digest, so the")
out("source has to be a function of the seed alone or the comparison is circular.")
out("")
header = f"{'file':>13}{'seed':>6}{'digest (first 24 hex)':>26}{'stable':>8}"
out(header)
out("-" * len(header))
stable = True
for spec_name, spec in SPECS.items():
    for seed in (1, 2):
        first = digest(source_payload(spec, seed))
        second = digest(source_payload(spec, seed))
        ok = first == second
        stable = stable and ok
        out(f"{spec_name:>13}{seed:>6}{first[:24]:>26}{str(ok):>8}")

out.rule("summary")
checks = {
    "no digest mismatch on any delivered run": not mismatches,
    "no digest reported on an undelivered run": not no_digest_violations,
    "the array exercises delivered and undelivered runs": both_branches,
    "every injected corruption was caught": not control_failures,
    "source payload is reproducible from the seed": stable,
}
for name, passed in checks.items():
    out(f"{'PASS' if passed else 'FAIL'}  {name}")
out("")
out(f"output written to {relative(out.path)}")
out.save()
if not all(checks.values()):
    raise SystemExit("validate_completion_invariant.py: at least one check FAILED")
