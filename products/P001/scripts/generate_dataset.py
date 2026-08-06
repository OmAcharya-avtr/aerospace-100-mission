"""Generate the surrogate training dataset from seeded Monte Carlo runs.

Deterministic: fixed master seed 42 spawns per-scenario MC seeds.
Output: data/surrogate_dataset.npz (X: (n, 5) features, y: log10 P_fade).
Runtime: ~20 s for 4000 scenarios x 50000 MC samples on 2 CPU cores.

Usage: python scripts/generate_dataset.py [n_scenarios]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beamtwin.surrogate import FEATURE_NAMES, generate_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    t0 = time.perf_counter()
    x, y = generate_dataset(n_scenarios=n, seed=SEED)
    dt = time.perf_counter() - t0
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    np.savez(out / "surrogate_dataset.npz", X=x, y=y, feature_names=list(FEATURE_NAMES))
    print(f"wrote {out / 'surrogate_dataset.npz'}: X{x.shape}, y{y.shape} in {dt:.1f} s")
    print(f"y (log10 P_fade) range: [{y.min():.2f}, {y.max():.2f}], floor fraction "
          f"{float(np.mean(y <= -4 + 1e-9)):.2%}")


if __name__ == "__main__":
    main()
