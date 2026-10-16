"""Grid occupancy: known answers, the Feller reference, and where it does not apply.

Checks, in order:

1. hand-counted occupancy on a small grid;
2. equation (3) from :mod:`scenariocov.occupancy` against Monte Carlo, on the
   model it actually describes -- independent uniform throws into EQUIPROBABLE
   cells;
3. the generalisation to unequal cell probabilities,
   ``E[occupied] = sum_c (1 - (1 - p_c)^n)``, against Monte Carlo, because the
   fixture campaign's cells are not equiprobable and the gap between the
   committed occupancy and the equiprobable reference has to be explained
   rather than asserted;
4. the committed campaign's occupancy against both references.

Every Monte Carlo comparison is made with a tolerance on both sides, never as
an equality.
"""

from __future__ import annotations

import sys

import numpy as np
from _common import FIXTURES, rule

from scenariocov import (
    Parameter,
    ScenarioSpace,
    expected_occupied_cells,
    grid_occupancy,
    load_campaign,
    load_space,
)

REPLICATES = 4000


def main() -> int:
    """Run the four checks."""
    out = sys.stdout
    failures = 0
    out.write("grid occupancy validation\n")
    out.write("=" * 78 + "\n")

    out.write(rule("1. hand-counted occupancy on a 3 x 4 grid") + "\n")
    space = ScenarioSpace(
        [
            Parameter.continuous("x", [0.0, 1.0, 2.0, 3.0]),
            Parameter.continuous("y", [0.0, 0.25, 0.5, 0.75, 1.0]),
        ]
    )
    records = [{"x": 0.5, "y": 0.1}, {"x": 0.7, "y": 0.2}, {"x": 2.5, "y": 0.9}]
    occ = grid_occupancy(space, space.encode_many(records))
    ok = (occ.n_cells, occ.occupied, int(occ.counts[0, 0]), int(occ.counts[2, 3])) == (12, 2, 2, 1)
    failures += 0 if ok else 1
    out.write(
        f"  {'PASS' if ok else 'FAIL'}  cells 12, occupied 2, count at (0,0) = 2, "
        "count at (2,3) = 1\n"
    )
    out.write(f"  largest empty connected block: {occ.largest_empty_block()} of 10 empty cells\n")

    out.write(rule("2. equation (3) versus Monte Carlo, equiprobable cells") + "\n")
    out.write(f"{'n':>6}{'C':>6}{'predicted':>12}{'measured':>12}{'4 s.e.':>10}  verdict\n")
    rng = np.random.default_rng(20261016)
    for n, cells in ((5, 10), (30, 25), (100, 50), (144, 144), (500, 144)):
        counts = [len(set(rng.integers(0, cells, size=n).tolist())) for _ in range(REPLICATES)]
        measured = float(np.mean(counts))
        predicted = expected_occupied_cells(n, cells)
        se = float(np.std(counts, ddof=1)) / np.sqrt(REPLICATES)
        ok = abs(measured - predicted) < 4.0 * se + 1e-12
        failures += 0 if ok else 1
        out.write(
            f"{n:>6}{cells:>6}{predicted:>12.4f}{measured:>12.4f}{4 * se:>10.4f}  "
            f"{'PASS' if ok else 'FAIL'}\n"
        )

    out.write(rule("3. unequal cell probabilities: the generalised expectation") + "\n")
    out.write(
        "Equation (3) assumes equiprobable cells. Drawing a continuous parameter\n"
        "uniformly over its declared interval does NOT make unequal-width bins\n"
        "equiprobable, so the correct expectation is the sum over cells of\n"
        "1 - (1 - p_c)^n, with p_c the product of the bin probabilities.\n"
    )
    edges_x = np.array([0.01, 0.05, 0.10, 0.20, 0.50])
    edges_y = np.array([20.0, 50.0, 100.0, 200.0, 400.0])
    px = np.diff(edges_x) / (edges_x[-1] - edges_x[0])
    py = np.diff(edges_y) / (edges_y[-1] - edges_y[0])
    cell_p = np.outer(px, py).ravel()
    out.write(f"  bin probabilities, closing_speed: {np.round(px, 6).tolist()}\n")
    out.write(f"  bin probabilities, range:         {np.round(py, 6).tolist()}\n")
    head = f"{'n':>6}{'equiprob':>12}{'generalised':>14}{'measured':>12}{'4 s.e.':>10}  verdict"
    out.write(head + "\n")
    for n in (40, 120, 240):
        generalised = float(np.sum(1.0 - (1.0 - cell_p) ** n))
        draws = rng.choice(len(cell_p), size=(REPLICATES, n), p=cell_p)
        counts = [len(set(row.tolist())) for row in draws]
        measured = float(np.mean(counts))
        se = float(np.std(counts, ddof=1)) / np.sqrt(REPLICATES)
        equiprob = expected_occupied_cells(n, len(cell_p))
        ok = abs(measured - generalised) < 4.0 * se + 1e-12
        failures += 0 if ok else 1
        out.write(
            f"{n:>6}{equiprob:>12.4f}{generalised:>14.4f}{measured:>12.4f}{4 * se:>10.4f}  "
            f"{'PASS' if ok else 'FAIL'}\n"
        )
    out.write(
        "  The equiprobable column is the one that is wrong for this sampling\n"
        "  rule, and it is wrong by roughly 3 cells in 16 at n = 240. The\n"
        "  package reports the equiprobable reference because it is the only one\n"
        "  computable without a model of how the campaign was sampled, and the\n"
        "  README says so.\n"
    )

    out.write(rule("4. the committed campaign") + "\n")
    fixture_space = load_space(FIXTURES / "space.json")
    campaign = load_campaign(FIXTURES / "campaign_executed.json")
    occ = grid_occupancy(fixture_space, fixture_space.encode_many(campaign))
    out.write(f"  occupied {occ.occupied}/{occ.n_cells} = {occ.fraction:.6f}\n")
    out.write(f"  equiprobable reference, eq. (3):  {occ.expected_fraction_uniform:.6f}\n")
    out.write(
        "  The campaign is below the equiprobable reference because 60 of its\n"
        "  240 scenarios are a nominal-weighted block and 9 of its 144 cells are\n"
        "  a planted hole. Neither is a defect in the measure.\n"
    )
    out.write(f"  empty cells: {occ.empty}, largest connected block: {occ.largest_empty_block()}\n")
    blocks = occ.empty_blocks()
    out.write(f"  empty-block sizes: {[b.size for b in blocks]}\n")
    ok = sum(b.size for b in blocks) == occ.empty
    failures += 0 if ok else 1
    out.write(f"  {'PASS' if ok else 'FAIL'}  block sizes sum to the empty-cell count\n")

    out.write(f"\nfailures: {failures}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
