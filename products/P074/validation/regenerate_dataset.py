"""Regenerate the committed synthetic population and divergence dataset.

Run this to rebuild ``fixtures/population.tle`` and
``fixtures/divergence_dataset.npz`` from nothing. It is the only script here
that writes into ``fixtures/``, and it is deterministic: the population comes
from a seeded generator and the divergence table is pure computation on it, so
a rebuild on the same numpy and sgp4 versions reproduces the committed files
byte for byte. ``validate_reproducibility.py`` checks that claim.

It takes about three minutes on two cores, which is why every other script
loads the committed file instead of regenerating it. In fast probe mode it
regenerates a small population into a temporary directory and leaves the
committed files alone, so that a probe run can never replace a seven-day
dataset with a one-day one.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

from _bootstrap import (
    DATASET_PATH,
    POPULATION_PATH,
    Tee,
    Timer,
    bootstrap,
    environment_lines,
    is_fast,
    relative,
)

ROOT = bootstrap()

from orbitdrift import build_divergence_table, synthetic_population_tle  # noqa: E402
from orbitdrift.dataset import (  # noqa: E402
    DEFAULT_REPORT_EVERY_HOURS,
    DEFAULT_SAMPLE_EVERY,
    DEFAULT_SPAN_DAYS,
    DEFAULT_STEPS_PER_REV,
    POPULATION_SPEC,
)


def main() -> int:
    out = Tee("regenerate_dataset_output.txt")
    out.rule("orbitdrift -- regenerate the synthetic population and divergence dataset")
    for line in environment_lines():
        out(line)

    out.rule("population specification")
    for key, value in POPULATION_SPEC.items():
        out(f"  {key:28s}: {value}")

    span_days = 0.1 if is_fast() else DEFAULT_SPAN_DAYS
    out.rule("generation")
    timer_population = Timer()
    with timer_population:
        tle_text = synthetic_population_tle()
    if is_fast():
        # Keep three objects, one per epoch group, so a probe is quick.
        lines = tle_text.splitlines()
        header = [line for line in lines[:2]]
        body: list[str] = []
        kept = 0
        index = 2
        while index < len(lines) and kept < 3:
            body.extend(lines[index : index + 3])
            kept += 1
            index += 3
        tle_text = "\n".join(header + body) + "\n"

    out(f"  population lines              : {len(tle_text.splitlines())}")
    out(f"  population generation         : {timer_population.elapsed_s:.3f} s (table only)")

    timer_table = Timer()
    with timer_table:
        table = build_divergence_table(
            tle_text,
            span_days=span_days,
            steps_per_rev=DEFAULT_STEPS_PER_REV,
            report_every_hours=DEFAULT_REPORT_EVERY_HOURS,
            sample_every=DEFAULT_SAMPLE_EVERY,
            progress=lambda index, total, label: out(
                f"  batch {index + 1}/{total}: {label}"
            ),
        )

    out("")
    out(f"  objects                       : {table.n_objects}")
    out(f"  rows                          : {table.n_rows}")
    out(f"  columns                       : {len(table.columns)}")
    out(f"  detail                        : {table.detail}")

    drift = table.column("drift_km")
    ages = table.column("epoch_age_days")
    out(f"  drift range, km               : {drift.min():.6f} to {drift.max():.3f}")
    out(f"  epoch age range, days         : {ages.min():.4f} to {ages.max():.4f}")

    if is_fast():
        with tempfile.TemporaryDirectory() as scratch:
            target = Path(scratch) / "divergence_dataset.npz"
            table.save(target)
            out("")
            out("  fast probe: wrote a reduced dataset to a temporary directory and")
            out("  left the committed fixtures untouched.")
        out("")
        out(f"wrote {out.save()}")
        return 0

    (ROOT / "fixtures").mkdir(exist_ok=True)
    (ROOT / POPULATION_PATH).write_text(tle_text)
    table.save(ROOT / DATASET_PATH)
    out("")
    for path in (POPULATION_PATH, DATASET_PATH):
        data = (ROOT / path).read_bytes()
        out(f"  {path:36s} {len(data):9d} bytes  sha256 {hashlib.sha256(data).hexdigest()}")
    out("")
    out("timings (UNSTABLE: wall clock on a shared two-core container)")
    out(f"  divergence table              : {timer_table.elapsed_s:.1f} s")
    out("")
    out(f"wrote {out.save()}")
    out(f"wrote {relative(ROOT / POPULATION_PATH)} and {relative(ROOT / DATASET_PATH)}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    sys.exit(main())
