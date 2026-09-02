"""Labelled candidate generation for the learned ranker.

Nothing is committed as data. This module is the dataset: it is deterministic
in an integer seed, and rerunning it reproduces the training and test matrices
exactly. ``DATASET_CARD.md`` documents what a row is, how it is drawn, and
what the generative model leaves out.

A **row is one candidate**, not one frame: the geometric search of
:func:`skymatch.identify.gather_candidates` proposes catalogue triangles for
observed triples, and each proposal becomes a row with

* ``X`` -- the 13 features of :data:`skymatch.identify.FEATURE_NAMES`,
* ``y`` -- 1 if all three of its correspondences are the truth, else 0,
* ``group`` -- the frame the candidate came from.

The class balance follows from the search, not from a choice: a frame yields
at most one correct candidate and however many wrong ones the tolerance
admits, so ``y = 1`` is a minority class whose fraction is itself a property of
the operating point. It is reported rather than rebalanced, because rebalancing
would destroy the base rate the ranker's probability output is supposed to
express.

Frames are drawn from a grid of operating points (magnitude limit, centroid
noise, false-star count) so that the ranker sees the whole regime it will be
scored on, including the regime where nothing works. Splitting is **by frame**,
not by row: candidates from one frame are correlated through their shared
spots, and splitting rows at random would leak.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import CameraModel
from .catalogue import StarCatalogue, generate_catalogue
from .identify import FEATURE_NAMES, SearchConfig, gather_candidates
from .pairtable import PairTable
from .scene import SceneConfig, simulate_scene
from .triangle import separation_tolerance

__all__ = [
    "DEFAULT_GRID",
    "CandidateDataset",
    "OperatingPoint",
    "build_catalogue_tables",
    "generate_candidate_dataset",
]


@dataclass(frozen=True)
class OperatingPoint:
    """One cell of the training grid.

    Attributes
    ----------
    magnitude_limit
        Catalogue limiting magnitude [mag].
    centroid_sigma_arcsec
        Per-axis centroid noise [arcsec].
    n_false_stars
        False detections injected into the frame.
    weight
        Relative share of frames drawn from this cell.
    """

    magnitude_limit: float
    centroid_sigma_arcsec: float
    n_false_stars: int
    weight: float = 1.0


#: The default training grid. Deliberately includes cells where identification
#: is hopeless (12 and 16 false stars), so the ranker learns what *no* evidence
#: looks like instead of only ever seeing solvable frames.
DEFAULT_GRID: tuple[OperatingPoint, ...] = (
    OperatingPoint(5.5, 2.0, 0),
    OperatingPoint(5.5, 10.0, 2),
    OperatingPoint(6.0, 1.0, 0),
    OperatingPoint(6.0, 5.0, 0),
    OperatingPoint(6.0, 5.0, 4),
    OperatingPoint(6.0, 20.0, 2),
    OperatingPoint(6.0, 40.0, 0),
    OperatingPoint(6.0, 5.0, 8),
    OperatingPoint(6.0, 5.0, 12),
    OperatingPoint(6.0, 10.0, 16),
    OperatingPoint(6.5, 5.0, 0),
    OperatingPoint(6.5, 20.0, 4),
)


@dataclass(frozen=True)
class CandidateDataset:
    """Feature matrix, labels, and the frame each row came from.

    Attributes
    ----------
    features
        ``(n_rows, 13)`` in the order of
        :data:`skymatch.identify.FEATURE_NAMES`.
    labels
        ``(n_rows,)`` of 0/1.
    groups
        ``(n_rows,)`` frame index.
    frame_solvable
        ``(n_frames,)`` bool: whether the frame produced a correct candidate
        at all. A ranker cannot recover a frame where the search never
        proposed the truth, and the fraction of such frames is the ceiling on
        every decision rule.
    feature_names
        Column names.
    metadata
        Counters recorded during generation.
    """

    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    frame_solvable: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES
    metadata: dict[str, float] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        """Number of candidate rows."""
        return int(self.features.shape[0])

    @property
    def n_frames(self) -> int:
        """Number of frames the rows came from."""
        return int(self.frame_solvable.shape[0])

    @property
    def positive_fraction(self) -> float:
        """Fraction of rows labelled correct."""
        return float(self.labels.mean()) if self.n_rows else 0.0


def build_catalogue_tables(
    magnitude_limits: tuple[float, ...],
    camera: CameraModel,
    seed: int,
    min_separation_rad: float = 0.0,
) -> dict[float, tuple[StarCatalogue, PairTable]]:
    """Generate one catalogue and pair table per magnitude limit.

    The pair table is quadratic in catalogue size (Eq. P1), so this is the
    expensive part of any sweep and is built once and reused.
    """
    tables: dict[float, tuple[StarCatalogue, PairTable]] = {}
    for limit in sorted(set(magnitude_limits)):
        cat = generate_catalogue(limit, seed=seed, min_separation_rad=min_separation_rad)
        tables[float(limit)] = (cat, PairTable(cat, camera.max_separation_rad))
    return tables


def generate_candidate_dataset(
    n_frames: int,
    seed: int,
    camera: CameraModel | None = None,
    grid: tuple[OperatingPoint, ...] = DEFAULT_GRID,
    catalogue_seed: int = 20260902,
    search: SearchConfig | None = None,
    max_stars: int = 10,
    tables: dict[float, tuple[StarCatalogue, PairTable]] | None = None,
) -> CandidateDataset:
    """Draw ``n_frames`` frames from ``grid`` and label every candidate they produce.

    Parameters
    ----------
    n_frames
        Frames to simulate. Rows come out at roughly 20-30 per frame.
    seed
        Seed for the frames (attitudes, noise, false stars). Distinct from
        ``catalogue_seed`` so that train and test can share a catalogue and
        differ only in the observations, which is the split a star tracker
        actually faces.
    camera
        Camera model; the default 12 deg / 1024 px model if omitted.
    grid
        Operating points to draw frames from, in proportion to their weights.
    catalogue_seed
        Seed for the catalogues.
    search
        Search limits.
    max_stars
        Spots handed to the matcher.
    tables
        Prebuilt ``{magnitude_limit: (catalogue, pair_table)}``, to avoid
        rebuilding pair tables between the train and test calls.

    Raises ``ValueError`` for a non-positive ``n_frames`` or an empty grid.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    if not grid:
        raise ValueError("grid must contain at least one OperatingPoint")
    cam = camera or CameraModel()
    cfg = search or SearchConfig()
    if tables is None:
        tables = build_catalogue_tables(
            tuple(p.magnitude_limit for p in grid), cam, catalogue_seed
        )

    weights = np.array([p.weight for p in grid], dtype=float)
    if np.any(weights <= 0.0):
        raise ValueError("OperatingPoint weights must be > 0")
    counts = np.maximum(1, np.round(n_frames * weights / weights.sum()).astype(int))

    rng = np.random.default_rng(int(seed))
    feats: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    solvable: list[bool] = []
    n_candidates_total = 0
    frame = 0
    for point, count in zip(grid, counts, strict=True):
        cat, table = tables[float(point.magnitude_limit)]
        scene_cfg = SceneConfig(
            camera=cam,
            centroid_sigma_arcsec=point.centroid_sigma_arcsec,
            n_false_stars=point.n_false_stars,
            max_stars=max_stars,
        )
        tol = separation_tolerance(max(point.centroid_sigma_arcsec, 0.5))
        for _ in range(int(count)):
            scene = simulate_scene(cat, scene_cfg, rng)
            candidates, _ = gather_candidates(
                scene.vectors, scene.magnitudes, table, tol, cam, cfg
            )
            n_candidates_total += len(candidates)
            any_correct = False
            for cand in candidates:
                correct = cand.is_correct(scene.truth_index)
                any_correct = any_correct or correct
                feats.append(cand.features)
                labels.append(int(correct))
                groups.append(frame)
            solvable.append(any_correct)
            frame += 1

    x = np.array(feats, dtype=float) if feats else np.zeros((0, len(FEATURE_NAMES)))
    y = np.array(labels, dtype=int)
    g = np.array(groups, dtype=int)
    s = np.array(solvable, dtype=bool)
    return CandidateDataset(
        features=x,
        labels=y,
        groups=g,
        frame_solvable=s,
        metadata={
            "n_frames": float(frame),
            "n_rows": float(x.shape[0]),
            "rows_per_frame": float(x.shape[0] / frame) if frame else 0.0,
            "positive_fraction": float(y.mean()) if y.size else 0.0,
            "solvable_frame_fraction": float(s.mean()) if s.size else 0.0,
            "n_candidates_total": float(n_candidates_total),
        },
    )
