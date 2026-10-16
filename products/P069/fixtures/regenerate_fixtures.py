"""Regenerate every committed fixture deterministically at a fixed seed.

    python fixtures/regenerate_fixtures.py

Writes, all under ``fixtures/``:

* ``space.json``                 the declared scenario space (7 parameters);
* ``classes.json``               declared equivalence classes and boundaries;
* ``campaign_executed.json``     the finished campaign, with five planted gaps;
* ``candidate_pool.json``        scenarios that could be executed next;
* ``generation_parameters.json`` the seed, the sampling model and the planted
  gaps, so that every number measured on these fixtures can be traced to the
  decision that produced it.

The campaign is SYNTHETIC. It is not a record of any real rendezvous
verification campaign, and nothing measured on it is evidence about any real
system. It is shaped like one: a nominal-heavy block of scenarios followed by a
coverage-driven fill, with deliberate holes.

The five planted gaps, and which measure is supposed to find each:

    G1  lighting = eclipse AND closing_speed in [0.20, 0.50]   -> 2-way tuple
    G2  thruster_health = one_failed AND sensor_mode = camera_only -> 2-way tuple
    G3  closing_speed in [0.20, 0.50] AND range in [200, 400]  -> 2-way tuple
                                                                  and 9 empty
                                                                  grid cells
    G4  nav_position_error >= 1.5 m never executed             -> equivalence
                                                                  class
                                                                  'excessive'
    G5  no scenario within 2.5 m of range = 200 m              -> boundary
                                                                  'handover_200m'

G4 and G5 are invisible to the combinatorial measures by construction: both sit
inside a single declared bin. That is the point of carrying three different
measures.

Scenario parameters and their units are those of a spacecraft close-range
rendezvous verification campaign. The ranges are plausible for a small
servicing vehicle and are declared, not derived: no source is claimed for them,
because they are fixture values, not engineering data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from scenariocov.classes import BoundarySpec, EquivalenceClass  # noqa: E402
from scenariocov.combinatorial import tway_coverage  # noqa: E402
from scenariocov.io import save_campaign, save_space  # noqa: E402
from scenariocov.occupancy import grid_occupancy  # noqa: E402
from scenariocov.space import Parameter, ScenarioSpace  # noqa: E402

SEED = 20261016
N_NOMINAL = 60
MAX_FILL = 4000
TARGET_TOTAL = 240

RANGE_BOUNDARY = 200.0
RANGE_EXCLUSION = 2.5
NAV_CAP = 1.5


def build_space() -> ScenarioSpace:
    """The declared scenario space: 4 continuous and 3 categorical parameters."""
    return ScenarioSpace(
        [
            Parameter.continuous("closing_speed", [0.01, 0.05, 0.10, 0.20, 0.50], units="m/s"),
            Parameter.continuous("range_at_handover", [20.0, 50.0, 100.0, 200.0, 400.0], units="m"),
            Parameter.continuous("target_tumble_rate", [0.0, 0.5, 2.0, 5.0], units="deg/s"),
            Parameter.continuous("nav_position_error", [0.0, 0.1, 0.5, 2.0], units="m"),
            Parameter.categorical("lighting", ["full_sun", "terminator", "eclipse"]),
            Parameter.categorical("sensor_mode", ["lidar_only", "camera_only", "fused"]),
            Parameter.categorical("thruster_health", ["nominal", "one_degraded", "one_failed"]),
        ]
    )


def build_classes() -> tuple[list[EquivalenceClass], list[BoundarySpec]]:
    """Declared equivalence classes and boundaries, as a verification plan would."""
    classes = [
        EquivalenceClass("closing_speed", "creep", lower=0.01, upper=0.05),
        EquivalenceClass("closing_speed", "slow", lower=0.05, upper=0.10),
        EquivalenceClass("closing_speed", "nominal", lower=0.10, upper=0.20),
        EquivalenceClass(
            "closing_speed", "fast", lower=0.20, upper=0.50, closed_upper=True, expected=False
        ),
        EquivalenceClass("nav_position_error", "tight", lower=0.0, upper=0.1),
        EquivalenceClass("nav_position_error", "moderate", lower=0.1, upper=0.5),
        EquivalenceClass("nav_position_error", "degraded", lower=0.5, upper=1.5),
        EquivalenceClass(
            "nav_position_error", "excessive", lower=1.5, upper=2.0, closed_upper=True,
            expected=False,
        ),
        EquivalenceClass("target_tumble_rate", "stable", lower=0.0, upper=0.5),
        EquivalenceClass("target_tumble_rate", "slow_tumble", lower=0.5, upper=2.0),
        EquivalenceClass(
            "target_tumble_rate", "fast_tumble", lower=2.0, upper=5.0, closed_upper=True,
            expected=False,
        ),
        EquivalenceClass("lighting", "lit", labels=("full_sun",)),
        EquivalenceClass("lighting", "degraded_light", labels=("terminator", "eclipse"),
                         expected=False),
        EquivalenceClass("thruster_health", "healthy", labels=("nominal",)),
        EquivalenceClass("thruster_health", "faulted", labels=("one_degraded", "one_failed"),
                         expected=False),
    ]
    boundaries = [
        BoundarySpec("closing_speed", "speed_limit_0p20", value=0.20, tolerance=0.005),
        BoundarySpec("nav_position_error", "nav_limit_0p50", value=0.50, tolerance=0.010),
        BoundarySpec("target_tumble_rate", "tumble_limit_2p0", value=2.0, tolerance=0.020),
        BoundarySpec("range_at_handover", "handover_200m", value=RANGE_BOUNDARY, tolerance=2.0),
    ]
    return classes, boundaries


def in_planted_gap(record: dict[str, object]) -> bool:
    """True when a record falls in any of the five planted gaps."""
    speed = float(record["closing_speed"])
    rng_m = float(record["range_at_handover"])
    nav = float(record["nav_position_error"])
    fast = speed >= 0.20
    if fast and record["lighting"] == "eclipse":
        return True  # G1
    if record["thruster_health"] == "one_failed" and record["sensor_mode"] == "camera_only":
        return True  # G2
    if fast and rng_m >= 200.0:
        return True  # G3
    if nav >= NAV_CAP:
        return True  # G4
    return abs(rng_m - RANGE_BOUNDARY) <= RANGE_EXCLUSION  # G5


def draw_nominal(rng: np.random.Generator) -> dict[str, object]:
    """A nominal-weighted scenario: slow, close, well-lit, healthy."""
    return {
        "closing_speed": round(float(rng.uniform(0.02, 0.12)), 4),
        "range_at_handover": round(float(rng.uniform(25.0, 150.0)), 2),
        "target_tumble_rate": round(float(rng.uniform(0.0, 1.2)), 3),
        "nav_position_error": round(float(rng.uniform(0.0, 0.45)), 4),
        "lighting": str(rng.choice(["full_sun", "full_sun", "terminator"])),
        "sensor_mode": str(rng.choice(["fused", "fused", "lidar_only"])),
        "thruster_health": str(rng.choice(["nominal", "nominal", "one_degraded"])),
    }


def draw_uniform(rng: np.random.Generator, space: ScenarioSpace) -> dict[str, object]:
    """A scenario drawn uniformly over the declared intervals and level sets."""
    record: dict[str, object] = {}
    for p in space.parameters:
        if p.kind == "continuous":
            value = float(rng.uniform(p.lower, p.upper))
            record[p.name] = round(value, 4 if p.upper <= 1.0 else 3)
        else:
            record[p.name] = str(rng.choice(list(p.levels)))
    return record


def draw_stratified(rng: np.random.Generator, space: ScenarioSpace) -> dict[str, object]:
    """A scenario drawn uniformly over the DISCRETISED cells, not over values.

    A level is chosen uniformly for every parameter and a value is then drawn
    uniformly inside that bin. This makes the grid cells equiprobable, which is
    what phase 3 of the campaign wants and what equation (3) in
    :mod:`scenariocov.occupancy` assumes. Drawing uniformly over the raw
    intervals instead, as :func:`draw_uniform` does, makes wide bins far more
    likely than narrow ones; the two sampling rules are kept separate here so
    the difference is visible rather than accidental.
    """
    record: dict[str, object] = {}
    for p in space.parameters:
        if p.kind == "continuous":
            level = int(rng.integers(0, p.n_levels))
            value = float(rng.uniform(p.edges[level], p.edges[level + 1]))
            record[p.name] = round(value, 4 if p.upper <= 1.0 else 3)
        else:
            record[p.name] = str(rng.choice(list(p.levels)))
    return record


def generate_campaign(space: ScenarioSpace, rng: np.random.Generator) -> list[dict[str, object]]:
    """Three gap-rejecting phases: a nominal block, a coverage fill, a bulk tail.

    Phase 1 is ``N_NOMINAL`` nominal-weighted scenarios, which is what a real
    campaign accumulates first. Phase 2 accepts uniform draws only while they
    add a 2-way tuple, so that every tuple except the planted ones is covered.
    Phase 3 adds cell-stratified draws (see :func:`draw_stratified`) up to
    ``TARGET_TOTAL`` so the campaign has the bulk a finished campaign has, and
    so that grid occupancy is measuring something other than phase 2's
    deliberate spread.
    """
    records: list[dict[str, object]] = []
    while len(records) < N_NOMINAL:
        candidate = draw_nominal(rng)
        if not in_planted_gap(candidate):
            records.append(candidate)
    for _ in range(MAX_FILL):
        result = tway_coverage(space, space.encode_many(records), t=2)
        if result.n_uncovered <= 3:
            break
        candidate = draw_uniform(rng, space)
        if in_planted_gap(candidate):
            continue
        before = result.covered
        after = tway_coverage(space, space.encode_many([*records, candidate]), t=2).covered
        if after > before:
            records.append(candidate)
    while len(records) < TARGET_TOTAL:
        candidate = draw_stratified(rng, space)
        if not in_planted_gap(candidate):
            records.append(candidate)
    return records


def main() -> int:
    """Write every fixture and print the checks a reader can verify."""
    rng = np.random.default_rng(SEED)
    space = build_space()
    classes, boundaries = build_classes()

    campaign = generate_campaign(space, rng)
    pool = [draw_uniform(rng, space) for _ in range(400)]

    save_space(space, HERE / "space.json")
    (HERE / "classes.json").write_text(
        json.dumps(
            {
                "classes": [c.to_dict() for c in classes],
                "boundaries": [b.to_dict() for b in boundaries],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    save_campaign(
        campaign,
        HERE / "campaign_executed.json",
        metadata={
            "description": "synthetic close-range rendezvous verification campaign",
            "seed": SEED,
            "planted_gaps": ["G1", "G2", "G3", "G4", "G5"],
        },
    )
    save_campaign(
        pool,
        HERE / "candidate_pool.json",
        metadata={"description": "uniform candidate pool, planted regions INCLUDED", "seed": SEED},
    )
    (HERE / "generation_parameters.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "n_nominal_block": N_NOMINAL,
                "fill_rule": "uniform draws accepted only when they add a 2-way tuple",
                "tail_rule": "cell-stratified draws up to the target size, gaps rejected",
                "target_total": TARGET_TOTAL,
                "pool_size": len(pool),
                "pool_rule": "uniform over declared intervals and level sets, gaps INCLUDED",
                "planted_gaps": {
                    "G1": "lighting == eclipse AND closing_speed >= 0.20 m/s",
                    "G2": "thruster_health == one_failed AND sensor_mode == camera_only",
                    "G3": "closing_speed >= 0.20 m/s AND range_at_handover >= 200 m",
                    "G4": f"nav_position_error >= {NAV_CAP} m",
                    "G5": f"|range_at_handover - {RANGE_BOUNDARY}| <= {RANGE_EXCLUSION} m",
                },
                "rounding": "continuous values rounded to 3 or 4 decimals before writing",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    encoded = space.encode_many(campaign)
    two_way = tway_coverage(space, encoded, t=2)
    occ = grid_occupancy(space, encoded)
    print(f"space: {len(space)} parameters, level counts {space.level_counts}, "
          f"{space.n_combinations} combinations")
    print(f"campaign: {len(campaign)} scenarios, pool: {len(pool)} scenarios")
    print(f"2-way coverage: {two_way.covered}/{two_way.total} = {two_way.fraction:.6f}")
    print(f"uncovered 2-way tuples ({two_way.n_uncovered}):")
    for u in two_way.uncovered:
        print(f"  {u.as_text()}")
    print(f"grid occupancy over {occ.parameters}: {occ.occupied}/{occ.n_cells} "
          f"= {occ.fraction:.6f}, largest empty block {occ.largest_empty_block()} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
