"""Train the learned policy deterministically and export it as a lookup table.

No model binary is committed.  The learned policy's two continuous observables
are quantised (``arqlonghaul.policy.QUANTISATION_LEVELS``), so the greedy policy
is a finite map from observation to action, and that map plus the per-action
predicted cost and across-tree standard deviation is the whole deployable
artifact.  It is written to ``policy_table_<env>.npz``, a few tens of kilobytes,
instead of a megabyte of pickled forest.

Determinism.  Everything that could vary is fixed: the fit seeds come from
``arqlonghaul.datasets.SEED_SPLIT.fit``, the behaviour policy's action draws
from ``BEHAVIOUR_SEED``, the forest from ``FOREST_SEED``, and the states
enumerated are those visited on a fixed list of enumeration seeds.  Re-running
this script reproduces the same ``.npz`` byte for byte on the same numpy and
scikit-learn versions; the SHA-256 of each array is printed so that can be
checked without a diff.  The versions are printed too, because a forest is not
guaranteed to be bit-identical across scikit-learn releases and pretending
otherwise would be dishonest.

Runtime: about 60 s on one core.
"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import sklearn

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
)

from arqlonghaul.datasets import SEED_SPLIT, long_burst_env, short_burst_env  # noqa: E402
from arqlonghaul.policy import (  # noqa: E402
    FEATURE_NAMES,
    QUANTISATION_LEVELS,
    LearnedRedundancyPolicy,
    collect_transitions,
    run_episode,
)

HERE = os.path.dirname(os.path.abspath(__file__))

FIT_SEEDS = SEED_SPLIT.fit[:120]
FIT_FRAMES = 150
ENUM_SEEDS = SEED_SPLIT.fit[:20]
ENUM_FRAMES = 600
BEHAVIOUR_SEED = 7
FOREST_SEED = 0
N_ESTIMATORS = 140
MAX_DEPTH = 10
MIN_LEAF = 20


def sha256_array(a: np.ndarray) -> str:
    """SHA-256 of an array's bytes, with its dtype and shape folded in."""
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def main() -> None:
    print(f"numpy {np.__version__}   scikit-learn {sklearn.__version__}")
    print(f"quantisation levels {QUANTISATION_LEVELS}")
    print(f"forest: {N_ESTIMATORS} trees, max_depth {MAX_DEPTH}, "
          f"min_samples_leaf {MIN_LEAF}, random_state {FOREST_SEED}")
    print(f"fit seeds {FIT_SEEDS[0]}..{FIT_SEEDS[-1]} x {FIT_FRAMES} frames, "
          f"behaviour seed {BEHAVIOUR_SEED}")
    print()
    for env in (long_burst_env(), short_burst_env()):
        tag = "long_burst" if env.mean_burst_rounds > 5 else "short_burst"
        tr = collect_transitions(env, FIT_SEEDS, FIT_FRAMES, rng_seed=BEHAVIOUR_SEED)
        policy = LearnedRedundancyPolicy(
            len(env.actions),
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_LEAF,
            random_state=FOREST_SEED,
        )
        policy.fit(tr["features"], tr["actions"], tr["cost"])

        seen: list[np.ndarray] = []
        for seed in ENUM_SEEDS:
            _, rec = run_episode(env, policy, ENUM_FRAMES, int(seed), record=True)
            seen.append(rec["features"])
        states = np.unique(np.concatenate(seen, axis=0), axis=0)
        order = np.lexsort(tuple(states[:, i] for i in range(states.shape[1] - 1, -1, -1)))
        states = states[order]

        cost = np.empty((states.shape[0], len(env.actions)))
        sd = np.empty_like(cost)
        for i, row in enumerate(states):
            cost[i], sd[i] = policy.predict_with_uncertainty(row)
        action = np.argmin(cost, axis=1).astype(np.int16)

        out = os.path.join(HERE, f"policy_table_{tag}.npz")
        np.savez_compressed(
            out,
            states=states.astype(np.float32),
            action_index=action,
            predicted_cost=cost.astype(np.float32),
            predicted_sd=sd.astype(np.float32),
            actions=np.asarray(env.actions, dtype=np.int32),
            feature_names=np.asarray(FEATURE_NAMES),
        )
        print(f"{env.name}")
        print(f"  states enumerated      {states.shape[0]}")
        print(f"  file                   policy_table_{tag}.npz "
              f"({os.path.getsize(out)} bytes)")
        print(f"  sha256 states          {sha256_array(states.astype(np.float32))}")
        print(f"  sha256 action_index    {sha256_array(action)}")
        print(f"  sha256 predicted_cost  {sha256_array(cost.astype(np.float32))}")
        counts = np.bincount(action, minlength=len(env.actions))
        print("  action index histogram "
              + " ".join(f"{a}:{c}" for a, c in zip(env.actions, counts, strict=True)))
        print()
    print("To regenerate: python validation/export_policy_table.py")
    print("The hashes above are over float32 arrays and are reproducible on the")
    print("numpy and scikit-learn versions printed at the top of this output.")


if __name__ == "__main__":
    main()
