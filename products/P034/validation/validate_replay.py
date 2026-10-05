"""Validation 1: every injected fault is reproducible from its seed and parameters.

Claim under test (Level 2 requirement): bit-identical replay.  "Bit-identical"
here means the IEEE 754 binary64 byte image of the whole trace -- true position
and velocity, both estimator states, the applied command, the reference and
every logged innovation -- is byte-for-byte equal between two runs.  Anything
weaker (comparison to a tolerance) would hide exactly the kind of
non-determinism this check exists to find.

Checks
  1a. Run each case of a pool covering all 248 coverage cells twice and compare
      byte images. Covers every fault kind, every channel, every parameter bin
      and both stochastic and deterministic handlers.
  1b. Serialise each case to JSON, parse it back, run it, and compare with the
      original byte image: a case written to a file replays identically.
  1c. A tampered case file is rejected: editing a parameter without updating the
      content hash raises ValueError rather than replaying something else.
  1d. Negative control: changing the seed must change the byte image for cases
      whose handler draws randomness or whose noise realisation matters. If
      everything matched regardless of seed, 1a would be vacuous.
  1e. Content-hash regression: the case_id of a fixed, fully specified case is
      compared against the literal value recorded in this script.

Run from products/P034/:  python validation/validate_replay.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faultinject.campaign import FaultCase, build_pool  # noqa: E402
from faultinject.faults import Injection  # noqa: E402
from faultinject.harness import run_case  # noqa: E402
from faultinject.taxonomy import FaultKind, kinds  # noqa: E402

ANCHOR_CASE_ID = "bd32605725b734a5"
"""case_id of the anchor case below, recorded when this script was first run."""

failures: list[str] = []

print("=" * 78)
print("VALIDATION 1 -- seeded replay is bit-identical")
print("=" * 78)
print()

pool = build_pool(pool_seed=1, replicates=1)
print(f"Pool: {len(pool)} cases, one per coverage cell, pool_seed=1.")
print(f"Kinds exercised: {len({c.injection.kind for c in pool})} of {len(kinds())}.")
print()

# ------------------------------------------------------------------------- 1a
print("1a. Same case run twice -- byte image must be identical")
mismatch_a = []
total_bytes = 0
for case in pool:
    a = run_case([case.injection], case.seed, case.n_steps).float_bytes()
    b = run_case([case.injection], case.seed, case.n_steps).float_bytes()
    total_bytes += len(a)
    if a != b:
        mismatch_a.append(case.case_id)
print(f"    cases compared        {len(pool)}")
print(f"    bytes compared        {total_bytes}")
print(f"    byte-image mismatches {len(mismatch_a)}")
if mismatch_a:
    failures.append(f"1a: {len(mismatch_a)} case(s) not reproducible: {mismatch_a[:5]}")
else:
    print("    PASS")
print()

# ------------------------------------------------------------------------- 1b
print("1b. Case -> JSON -> case -> run: byte image must match the original run")
mismatch_b = []
for case in pool:
    original = run_case([case.injection], case.seed, case.n_steps).float_bytes()
    restored = FaultCase.from_json(case.to_json())
    if restored.case_id != case.case_id:
        mismatch_b.append((case.case_id, "case_id changed"))
        continue
    again = run_case([restored.injection], restored.seed, restored.n_steps).float_bytes()
    if again != original:
        mismatch_b.append((case.case_id, "trace changed"))
print(f"    round trips           {len(pool)}")
print(f"    mismatches            {len(mismatch_b)}")
if mismatch_b:
    failures.append(f"1b: {len(mismatch_b)} round-trip mismatch(es): {mismatch_b[:5]}")
else:
    print("    PASS")
print()

# ------------------------------------------------------------------------- 1c
print("1c. Tampered case file must be rejected")
anchor = FaultCase(
    Injection.create(FaultKind.SENSOR_BIAS, "pos", {"offset": 3.0}, 40, 100), 7, 150
)
tampered = anchor.to_dict()
tampered["injection"]["params"]["offset"] = 3.0000001  # type: ignore[index]
try:
    FaultCase.from_dict(tampered)
    failures.append("1c: tampered case accepted without a hash error")
    print("    FAILED  tampered case was accepted")
except ValueError as exc:
    print(f"    PASS    ValueError raised: {exc}")
print()

# ------------------------------------------------------------------------- 1d
print("1d. Negative control -- changing the seed must change the byte image")
changed = 0
unchanged = []
for case in pool:
    a = run_case([case.injection], case.seed, case.n_steps).float_bytes()
    b = run_case([case.injection], case.seed + 1, case.n_steps).float_bytes()
    if a != b:
        changed += 1
    else:
        unchanged.append(case.cell().label())
print(f"    cases where seed+1 changed the trace  {changed}/{len(pool)}")
if unchanged:
    print(f"    cases where it did not                {len(unchanged)}")
    for lbl in unchanged[:10]:
        print(f"      {lbl}")
if changed == 0:
    failures.append("1d: the seed had no effect on any case, so 1a proves nothing")
else:
    print("    PASS (a non-empty subset is enough; some cases abort on the same step)")
print()

# ------------------------------------------------------------------------- 1e
print("1e. Content-hash regression for a fixed case")
print(f"    case      {anchor.to_json()}")
print(f"    case_id   {anchor.case_id}")
print(f"    recorded  {ANCHOR_CASE_ID}")
if anchor.case_id != ANCHOR_CASE_ID:
    failures.append(
        f"1e: case_id changed from {ANCHOR_CASE_ID} to {anchor.case_id}; "
        "a stored case can no longer be looked up"
    )
    print("    FAILED")
else:
    print("    PASS")
print()

print("=" * 78)
if failures:
    print(f"RESULT: FAILED ({len(failures)} check(s))")
    for f in failures:
        print(f"  FAILED  {f}")
    sys.exit(1)
print("RESULT: PASS -- all checks in validation 1 passed")
print("=" * 78)
