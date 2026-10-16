"""Draw the grid occupancy and name the largest untested region.

    python examples/occupancy_map.py

Writes ../screenshots/occupancy_map.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scenariocov import grid_occupancy, load_campaign, load_space  # noqa: E402
from scenariocov.plotting import plot_occupancy_slices  # noqa: E402


def main() -> int:
    """Measure occupancy, print the empty blocks and save the figure."""
    space = load_space(ROOT / "fixtures" / "space.json")
    campaign = load_campaign(ROOT / "fixtures" / "campaign_executed.json")
    occ = grid_occupancy(space, space.encode_many(campaign))
    print(f"grid over {occ.parameters}")
    print(f"discretisation {occ.shape}, {occ.n_cells} cells, {occ.n_scenarios} scenarios")
    print(f"occupied {occ.occupied}/{occ.n_cells} = {occ.fraction:.6f}")
    print(f"uniform-random reference (equiprobable cells): {occ.expected_fraction_uniform:.6f}")
    blocks = occ.empty_blocks()
    print(f"empty cells {occ.empty} in {len(blocks)} connected blocks")
    for block in blocks[:3]:
        print(f"  block of {block.size} cells, first {block.cells[0]}, last {block.cells[-1]}")
    biggest = blocks[0]
    corner = biggest.cells[0]
    names = [
        space.parameter(n).levels[level]
        for n, level in zip(occ.parameters, corner, strict=True)
    ]
    print("  largest block starts at " + ", ".join(
        f"{n}={level}" for n, level in zip(occ.parameters, names, strict=True)
    ))
    out = plot_occupancy_slices(occ, ROOT / "screenshots" / "occupancy_map.png")
    print(f"wrote screenshots/{out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
