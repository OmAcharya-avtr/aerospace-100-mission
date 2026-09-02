"""SkyMatch: lost-in-space star identification, measured on both kinds of error.

Public API, grouped by what it does:

Catalogue
    :func:`~skymatch.catalogue.generate_catalogue`,
    :func:`~skymatch.catalogue.remove_close_pairs`,
    :class:`~skymatch.catalogue.StarCatalogue`,
    :class:`~skymatch.pairtable.PairTable`

Observation
    :class:`~skymatch.camera.CameraModel`,
    :class:`~skymatch.scene.SceneConfig`,
    :func:`~skymatch.scene.simulate_scene`

Matching
    :func:`~skymatch.triangle.triangle_candidates`,
    :func:`~skymatch.pyramid.confirm_with_fourth_star`,
    :func:`~skymatch.identify.gather_candidates`,
    :func:`~skymatch.identify.pyramid_decision`,
    :func:`~skymatch.identify.triangle_decision`,
    :func:`~skymatch.identify.resolve`

Learned ranking
    :class:`~skymatch.ranker.LearnedRanker`,
    :func:`~skymatch.dataset.generate_candidate_dataset`

Scoring
    :func:`~skymatch.benchmark.run_trials`,
    :func:`~skymatch.benchmark.wilson_interval`

Nothing in this package is flight software. See ``README.md`` for the safety
statement and ``MODEL_CARD.md`` for what the learned ranker is and is not.
"""

from .benchmark import MethodResult, SweepPoint, run_trials, wilson_interval
from .camera import CameraModel
from .catalogue import (
    StarCatalogue,
    expected_close_pairs,
    generate_catalogue,
    predicted_count,
    remove_close_pairs,
)
from .dataset import (
    DEFAULT_GRID,
    CandidateDataset,
    OperatingPoint,
    build_catalogue_tables,
    generate_candidate_dataset,
)
from .geometry import (
    ARCSEC,
    angle_between_dcm,
    angular_separation,
    davenport_attitude,
    dcm_from_quat,
    quat_from_dcm,
    radec_from_unit_vectors,
    random_rotation,
    unit_vectors_from_radec,
)
from .identify import (
    FEATURE_NAMES,
    Candidate,
    Identification,
    SearchConfig,
    gather_candidates,
    observed_separations,
    pyramid_decision,
    resolve,
    triangle_decision,
)
from .pairtable import PairTable, expected_pair_count
from .pyramid import confirm_with_fourth_star, pyramid_triple_order
from .ranker import (
    LearnedRanker,
    brier_score,
    expected_calibration_error,
    reliability_table,
)
from .scene import Scene, SceneConfig, simulate_scene
from .triangle import separation_tolerance, triangle_candidates, triangle_edge_angles

__version__ = "0.1.0"

__all__ = [
    "ARCSEC",
    "DEFAULT_GRID",
    "FEATURE_NAMES",
    "CameraModel",
    "Candidate",
    "CandidateDataset",
    "Identification",
    "LearnedRanker",
    "MethodResult",
    "OperatingPoint",
    "PairTable",
    "Scene",
    "SceneConfig",
    "SearchConfig",
    "StarCatalogue",
    "SweepPoint",
    "__version__",
    "angle_between_dcm",
    "angular_separation",
    "brier_score",
    "build_catalogue_tables",
    "confirm_with_fourth_star",
    "davenport_attitude",
    "dcm_from_quat",
    "expected_calibration_error",
    "expected_close_pairs",
    "expected_pair_count",
    "gather_candidates",
    "generate_candidate_dataset",
    "generate_catalogue",
    "observed_separations",
    "predicted_count",
    "pyramid_decision",
    "pyramid_triple_order",
    "quat_from_dcm",
    "radec_from_unit_vectors",
    "random_rotation",
    "reliability_table",
    "remove_close_pairs",
    "resolve",
    "run_trials",
    "separation_tolerance",
    "simulate_scene",
    "triangle_candidates",
    "triangle_decision",
    "triangle_edge_angles",
    "unit_vectors_from_radec",
    "wilson_interval",
]
