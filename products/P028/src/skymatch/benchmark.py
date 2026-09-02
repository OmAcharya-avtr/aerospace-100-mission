"""Scoring harness: identification rate, false-identification rate, and their error bars.

Three outcomes, exhaustive and mutually exclusive, so the three rates sum to
one for every method:

* **identified correctly** -- the accepted candidate's three core
  correspondences are all the truth;
* **false identification** -- a candidate was accepted and at least one core
  correspondence is wrong. This is the number most star-identification work
  omits, and it is the reason a decision rule with a high identification rate
  can still be the wrong choice;
* **no solution** -- nothing was accepted.

A false identification is far worse than no solution: a star tracker that
reports no attitude is a known-unknown that the spacecraft can wait out, while
one that reports a confidently wrong attitude will be believed. The two are
therefore never summed into a single "accuracy".

Scoring is on the **core** correspondence -- the triangle the decision rule
accepted -- not on the extended match list of
:func:`skymatch.identify.resolve`. The extension can attach a spurious spot to
a correct identification without the identification being wrong, and folding
that in would confuse two different quantities. The attitude error of accepted
solutions is reported separately, so an identification that is "correct" but
badly conditioned still shows up.

Error bars are Wilson score intervals (Wilson 1927), not the normal
approximation, because the interesting counts here are 0 and the normal
interval is degenerate there. A measured zero false identifications out of
``n`` trials is reported with its Wilson upper bound, which is roughly
``3.8/n``: **zero observed is not zero, and the table says so.**
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .camera import CameraModel
from .catalogue import StarCatalogue
from .geometry import angle_between_dcm
from .identify import (
    SearchConfig,
    gather_candidates,
    pyramid_decision,
    resolve,
    triangle_decision,
)
from .pairtable import PairTable
from .ranker import LearnedRanker
from .scene import SceneConfig, simulate_scene
from .triangle import separation_tolerance

__all__ = ["MethodResult", "SweepPoint", "run_trials", "wilson_interval"]


def wilson_interval(successes: int, trials: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    ``z = 1.96`` gives the two-sided 95% interval. Returns ``(lo, hi)``.
    Well behaved at ``successes = 0`` and ``successes = trials``, which the
    normal approximation is not.
    """
    if trials <= 0:
        raise ValueError(f"trials must be > 0, got {trials}")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes must be in [0, {trials}], got {successes}")
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denom
    half = (z / denom) * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


@dataclass
class MethodResult:
    """Counts and rates for one decision rule over one sweep point."""

    name: str
    n_trials: int = 0
    n_correct: int = 0
    n_false: int = 0
    n_none: int = 0
    attitude_errors_arcsec: list[float] = field(default_factory=list)
    confidences_correct: list[float] = field(default_factory=list)
    confidences_false: list[float] = field(default_factory=list)

    @property
    def identification_rate(self) -> float:
        """Fraction of trials identified correctly."""
        return self.n_correct / self.n_trials if self.n_trials else 0.0

    @property
    def false_identification_rate(self) -> float:
        """Fraction of trials that accepted a wrong candidate."""
        return self.n_false / self.n_trials if self.n_trials else 0.0

    @property
    def no_solution_rate(self) -> float:
        """Fraction of trials with nothing accepted."""
        return self.n_none / self.n_trials if self.n_trials else 0.0

    @property
    def identification_ci(self) -> tuple[float, float]:
        """95% Wilson interval on the identification rate."""
        return wilson_interval(self.n_correct, self.n_trials)

    @property
    def false_identification_ci(self) -> tuple[float, float]:
        """95% Wilson interval on the false-identification rate."""
        return wilson_interval(self.n_false, self.n_trials)

    @property
    def median_attitude_error_arcsec(self) -> float:
        """Median attitude error over correctly identified trials, or ``nan``."""
        if not self.attitude_errors_arcsec:
            return float("nan")
        return float(np.median(self.attitude_errors_arcsec))

    @property
    def p95_attitude_error_arcsec(self) -> float:
        """95th-percentile attitude error over correctly identified trials, or ``nan``."""
        if not self.attitude_errors_arcsec:
            return float("nan")
        return float(np.percentile(self.attitude_errors_arcsec, 95))


@dataclass
class SweepPoint:
    """Every method's result at one operating point, plus the point's own statistics."""

    label: str
    centroid_sigma_arcsec: float
    n_false_stars: int
    magnitude_limit: float
    n_trials: int
    methods: dict[str, MethodResult]
    mean_spots: float = 0.0
    mean_true_spots: float = 0.0
    frames_below_four_spots: int = 0
    solvable_fraction: float = 0.0
    mean_candidates: float = 0.0
    mean_seconds_per_frame: float = 0.0

    @property
    def ceiling(self) -> float:
        """Fraction of frames whose candidate list contained the truth at all.

        No decision rule can exceed this. It separates a search failure from a
        decision failure, which is the distinction that says whether a better
        ranker could possibly help.
        """
        return self.solvable_fraction


def run_trials(
    catalogue: StarCatalogue,
    table: PairTable,
    camera: CameraModel,
    scene_config: SceneConfig,
    n_trials: int,
    seed: int,
    ranker: LearnedRanker | None = None,
    thresholds: tuple[float, ...] = (0.5,),
    search: SearchConfig | None = None,
    tolerance_sigma_arcsec: float | None = None,
    label: str = "",
    with_attitude: bool = True,
) -> SweepPoint:
    """Run ``n_trials`` frames at one operating point and score every decision rule.

    All rules see the **same** candidate list from one geometric search, so
    differences between them are differences of decision, not of search. The
    classical early exit is not used here; it is timed separately in
    ``validation/validate_ml_vs_classical.py``.

    Parameters
    ----------
    catalogue, table, camera
        The prepared catalogue, its pair table, and the camera.
    scene_config
        Observation conditions.
    n_trials
        Frames to simulate.
    seed
        Seed for the frames.
    ranker
        A fitted :class:`skymatch.ranker.LearnedRanker`, or ``None`` to score
        only the classical rules.
    thresholds
        Acceptance thresholds to score the ranker at; each becomes its own
        method row named ``ranker@<threshold>``.
    search
        Search limits.
    tolerance_sigma_arcsec
        Noise the tolerance is sized for. Defaults to the true centroid noise;
        set it to something else to measure a mismatched tolerance.
    label
        Row label for reporting.
    with_attitude
        Resolve the attitude of accepted candidates. Costs about 0.3 ms per
        accepted frame.

    Returns a :class:`SweepPoint`.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if ranker is not None and not ranker.fitted:
        raise ValueError("ranker must be fitted before it can be scored")
    cfg = search or SearchConfig()
    sigma = (
        scene_config.centroid_sigma_arcsec
        if tolerance_sigma_arcsec is None
        else tolerance_sigma_arcsec
    )
    tol = separation_tolerance(max(sigma, 0.5))
    rng = np.random.default_rng(int(seed))

    names = ["triangle", "pyramid"]
    if ranker is not None:
        names += [f"ranker@{t:g}" for t in thresholds]
    methods = {n: MethodResult(name=n) for n in names}

    spots: list[int] = []
    true_spots: list[int] = []
    n_short = 0
    n_solvable = 0
    n_candidates = 0
    t_start = time.perf_counter()
    for _ in range(n_trials):
        scene = simulate_scene(catalogue, scene_config, rng)
        spots.append(scene.n_spots)
        true_spots.append(scene.n_true_stars)
        if scene.n_spots < 4:
            n_short += 1
        candidates, _ = gather_candidates(
            scene.vectors, scene.magnitudes, table, tol, camera, cfg
        )
        n_candidates += len(candidates)
        if any(c.is_correct(scene.truth_index) for c in candidates):
            n_solvable += 1

        decisions: list[tuple[str, object, float]] = [
            ("triangle", triangle_decision(candidates), 1.0),
            ("pyramid", pyramid_decision(candidates), 1.0),
        ]
        if ranker is not None:
            scores = ranker.score_candidates(candidates)
            if scores.size:
                best = int(np.argmax(scores))
                best_score = float(scores[best])
                for t in thresholds:
                    chosen = candidates[best] if best_score >= t else None
                    decisions.append((f"ranker@{t:g}", chosen, best_score))
            else:
                for t in thresholds:
                    decisions.append((f"ranker@{t:g}", None, 0.0))

        for name, cand, conf in decisions:
            result = methods[name]
            result.n_trials += 1
            if cand is None:
                result.n_none += 1
                continue
            correct = cand.is_correct(scene.truth_index)
            if correct:
                result.n_correct += 1
                result.confidences_correct.append(conf)
            else:
                result.n_false += 1
                result.confidences_false.append(conf)
            if with_attitude and correct:
                ident = resolve(cand, scene.vectors, catalogue, camera, tol)
                if ident.attitude is not None:
                    err = angle_between_dcm(ident.attitude, scene.attitude)
                    result.attitude_errors_arcsec.append(float(np.degrees(err) * 3600.0))
    elapsed = time.perf_counter() - t_start

    return SweepPoint(
        label=label or f"sigma={scene_config.centroid_sigma_arcsec}",
        centroid_sigma_arcsec=scene_config.centroid_sigma_arcsec,
        n_false_stars=scene_config.n_false_stars,
        magnitude_limit=catalogue.magnitude_limit,
        n_trials=n_trials,
        methods=methods,
        mean_spots=float(np.mean(spots)),
        mean_true_spots=float(np.mean(true_spots)),
        frames_below_four_spots=n_short,
        solvable_fraction=n_solvable / n_trials,
        mean_candidates=n_candidates / n_trials,
        mean_seconds_per_frame=elapsed / n_trials,
    )
