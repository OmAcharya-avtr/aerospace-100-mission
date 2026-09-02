"""Regenerate the committed training dataset. Run: ``python3 generate_dataset.py``

Deterministic: every episode is drawn by ``momentummgr.episodes.sample_episode(seed)``
and every label comes from ``momentummgr.learned.search_best_mask(episode, seed=0)``.
Deleting ``training_features.csv`` and rerunning this script reproduces it exactly.

Nothing in the dataset is measured. See ``DATASET_CARD.md``.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from momentummgr import FEATURE_NAMES, sample_episode  # noqa: E402
from momentummgr.learned import harvest_training_rows, search_best_mask  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIT_SEEDS = list(range(1000, 1060))
TUNE_SEEDS = list(range(2000, 2025))
HELDOUT_SEEDS = list(range(5000, 5080))


def main() -> int:
    """Build the episodes, run the label search, and write the CSV and the manifest."""
    t0 = time.time()
    print(f"Building {len(FIT_SEEDS)} fitting episodes and searching their schedules...")
    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    costs: list[float] = []
    attempts_note = []
    for seed in FIT_SEEDS:
        episode = sample_episode(seed)
        result = search_best_mask(episode, seed=0)
        x, y = harvest_training_rows(episode, result.mask)
        rows.append(x)
        labels.append(y)
        costs.append(result.metrics.cost)
        attempts_note.append(episode.n_windows)
    features = np.vstack(rows)
    target = np.concatenate(labels)
    table = np.column_stack([features, target.astype(float)])
    header = ",".join([*FEATURE_NAMES, "label"])
    csv_path = HERE / "training_features.csv"
    np.savetxt(csv_path, table, delimiter=",", header=header, comments="", fmt="%.10g")
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    manifest = HERE / "dataset_manifest.txt"
    manifest.write_text(
        "momentummgr synthetic desaturation-scheduling dataset\n"
        "=====================================================\n\n"
        "Every row is simulated. No flight telemetry is used anywhere in this package.\n\n"
        f"generator            momentummgr.episodes.sample_episode(seed)\n"
        f"labels               momentummgr.learned.search_best_mask(episode, seed=0)\n"
        f"fitting seeds        {FIT_SEEDS[0]}..{FIT_SEEDS[-1]} ({len(FIT_SEEDS)} episodes)\n"
        f"knob-tuning seeds    {TUNE_SEEDS[0]}..{TUNE_SEEDS[-1]} ({len(TUNE_SEEDS)} episodes)\n"
        f"held-out seeds       {HELDOUT_SEEDS[0]}..{HELDOUT_SEEDS[-1]} "
        f"({len(HELDOUT_SEEDS)} episodes)\n"
        f"windows per episode  {min(attempts_note)}..{max(attempts_note)}\n"
        f"feature rows         {features.shape[0]} x {features.shape[1]}\n"
        f"positive label rate  {target.mean():.6f}\n"
        f"searched mean cost   {float(np.mean(costs)):.6f}\n"
        f"columns              {header}\n"
        f"training_features.csv sha256 {digest}\n"
        f"bytes                {csv_path.stat().st_size}\n"
    )
    print(manifest.read_text())
    print(f"wall time {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
