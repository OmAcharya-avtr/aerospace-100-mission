"""Candidate generation, the classical decision rules, and the resolved answer.

The identification is split into two stages on purpose, because that split is
what makes the comparison in ``MODEL_CARD.md`` a fair one:

**Stage 1, geometric search** (:func:`gather_candidates`). Walk observed
triples in the Pyramid scan order, find the catalogue triangles that match
(Eq. T1), and test each against up to ``max_confirm_stars`` further observed
spots (Eq. Y1). This produces a list of :class:`Candidate` objects, each
carrying the evidence for and against it. This stage is entirely classical and
is shared by every decision rule below.

**Stage 2, decision.** Choose one candidate, or none:

* :func:`triangle_decision` -- the first triple with exactly one matching
  catalogue triangle. No fourth star. This is the weak baseline, and its
  false-identification rate is the number it exists to expose.
* :func:`pyramid_decision` -- the first triple with exactly one candidate that
  a fourth spot confirms. This is the classical Pyramid rule (Mortari et al.
  2004) and the baseline the learned ranker is measured against.
* :class:`skymatch.ranker.LearnedRanker` -- score every candidate and take the
  best if it clears a threshold. Same candidates, different rule.

Both classical rules are *first-match* rules: they stop at the first triple
that satisfies them, which is what an onboard implementation does. Passing
``stop_when_confirmed=True`` to :func:`gather_candidates` reproduces that
early exit, and is how the classical runtime is measured; the learned ranker
needs the full list and therefore pays for the whole scan. That cost is
reported, not hidden.

An accepted candidate is turned into an attitude and a full correspondence
list by :func:`resolve`, which solves Wahba's problem (Eq. G3) on the
confirmed correspondences and then matches the remaining spots against the
catalogue stars the resulting attitude puts in the field.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import CameraModel
from .catalogue import StarCatalogue
from .geometry import ARCSEC, davenport_attitude
from .pairtable import PairTable
from .pyramid import confirm_with_fourth_star, pyramid_triple_order
from .triangle import triangle_candidates

__all__ = [
    "FEATURE_NAMES",
    "MAGNITUDE_FEATURE_INDICES",
    "Candidate",
    "Identification",
    "SearchConfig",
    "gather_candidates",
    "observed_separations",
    "pyramid_decision",
    "resolve",
    "triangle_decision",
    "triple_scan_order",
]

#: Order of the columns produced by :meth:`Candidate.features`.
FEATURE_NAMES: tuple[str, ...] = (
    "resid_max_norm",
    "resid_rms_norm",
    "n_confirm",
    "confirm_fraction",
    "confirm_resid_norm",
    "log1p_rivals",
    "min_edge_rad",
    "mean_edge_rad",
    "magnitude_resid_rms",
    "magnitude_rank_agreement",
    "log10_tolerance_arcsec",
    "expected_stars_in_field",
    "n_spots",
)

#: Columns of :data:`FEATURE_NAMES` that use photometry, for the ablation in
#: ``MODEL_CARD.md``. The simulator gives the instrument the catalogue's own
#: magnitude scale plus Gaussian noise, which is optimistic; the ablation
#: measures how much the ranker leans on it.
MAGNITUDE_FEATURE_INDICES: tuple[int, ...] = (8, 9)

_NO_CONFIRMATION_RESIDUAL = 2.0

_TRIPLE_ORDER_CACHE: dict[int, list[tuple[int, int, int]]] = {}


def triple_scan_order(n_spots: int) -> list[tuple[int, int, int]]:
    """Memoised :func:`skymatch.pyramid.pyramid_triple_order`.

    The scan order depends only on the spot count, and a frame is processed
    thousands of times in a benchmark sweep, so it is built once per count.
    The returned list is shared and must not be mutated.
    """
    order = _TRIPLE_ORDER_CACHE.get(n_spots)
    if order is None:
        order = pyramid_triple_order(n_spots)
        _TRIPLE_ORDER_CACHE[n_spots] = order
    return order


def observed_separations(vectors: np.ndarray) -> np.ndarray:
    """All pairwise angles ``(n, n)`` [rad] between observed directions (Eq. G2).

    Computed once per frame. Every triple and every confirmation reads this
    matrix instead of recomputing a cross product, which is most of the
    difference between a 90 ms frame and a 10 ms one.
    """
    v = np.asarray(vectors, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vectors must have shape (n, 3), got {v.shape}")
    dot = np.clip(v @ v.T, -1.0, 1.0)
    cross = np.linalg.norm(np.cross(v[:, None, :], v[None, :, :]), axis=2)
    return np.arctan2(cross, dot)


@dataclass(frozen=True)
class SearchConfig:
    """Limits on the geometric search.

    Parameters
    ----------
    max_triples
        Observed triples tried, in Pyramid scan order. 25 covers every triple
        of the 6 brightest spots and part of the 7th.
    max_candidates_per_triple
        A triple producing more matching catalogue triangles than this is
        abandoned: it carries no information and the confirmation cost is
        linear in the count. The count is still recorded in the diagnostics.
    max_confirm_stars
        Fourth spots tried against each candidate triangle.
    max_candidates
        Hard cap on the returned candidate list.
    """

    max_triples: int = 25
    max_candidates_per_triple: int = 24
    max_confirm_stars: int = 5
    max_candidates: int = 120

    def __post_init__(self) -> None:
        for name in (
            "max_triples",
            "max_candidates_per_triple",
            "max_confirm_stars",
            "max_candidates",
        ):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value}")


@dataclass(frozen=True, eq=False)
class Candidate:
    """One catalogue triangle proposed for one observed triple.

    ``eq=False``: the generated ``__eq__`` would compare NumPy arrays and
    raise "truth value of an array is ambiguous". Compare
    ``(observed, catalogue)`` instead, which is what identity of a candidate
    actually means.

    Attributes
    ----------
    observed
        Indices of the three observed spots, in scan order.
    catalogue
        The proposed catalogue indices, aligned with ``observed``.
    scan_position
        Position of the observed triple in the Pyramid scan.
    n_rivals
        Number of *other* catalogue triangles matching the same observed
        triple. 0 means the triangle match was already unique.
    edge_residuals
        ``(3,)`` signed ``theta_catalogue - theta_observed`` [rad].
    confirm_observed, confirm_catalogue
        Spots that confirmed this candidate under Eq. Y1, and the catalogue
        stars they matched.
    confirm_residual_rms
        Mean over confirming spots of the RMS of their three Eq. Y1
        residuals [rad]; ``nan`` if nothing confirmed.
    features
        ``(13,)`` feature vector in the order of :data:`FEATURE_NAMES`.
    """

    observed: tuple[int, int, int]
    catalogue: tuple[int, int, int]
    scan_position: int
    n_rivals: int
    edge_residuals: np.ndarray
    confirm_observed: tuple[int, ...] = ()
    confirm_catalogue: tuple[int, ...] = ()
    confirm_residual_rms: float = float("nan")
    features: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURE_NAMES)))

    @property
    def n_confirm(self) -> int:
        """Number of spots that confirmed this candidate."""
        return len(self.confirm_observed)

    @property
    def all_observed(self) -> tuple[int, ...]:
        """Triangle plus confirming spots."""
        return tuple(self.observed) + tuple(self.confirm_observed)

    @property
    def all_catalogue(self) -> tuple[int, ...]:
        """Triangle plus confirming catalogue stars."""
        return tuple(self.catalogue) + tuple(self.confirm_catalogue)

    def is_correct(self, truth_index: np.ndarray) -> bool:
        """True if every core correspondence matches ``truth_index``.

        ``truth_index[o] == -1`` marks a false detection, which can never be
        correctly matched to any catalogue star.
        """
        t = np.asarray(truth_index)
        return all(t[o] == s for o, s in zip(self.observed, self.catalogue, strict=True))


@dataclass(frozen=True, eq=False)
class Identification:
    """The resolved answer for one frame.

    ``eq=False`` for the same reason as :class:`Candidate`.

    Attributes
    ----------
    status
        ``"identified"`` or ``"no_solution"``.
    candidate
        The accepted :class:`Candidate`, or ``None``.
    attitude
        ``(3, 3)`` estimated DCM (catalogue -> camera), or ``None``.
    observed_indices, catalogue_indices
        The full correspondence after extension, aligned.
    confidence
        The decision rule's confidence in ``[0, 1]``. The classical rules
        return 1.0 when they accept: they are hard rules with no graded
        output, and that is one of the things the learned ranker changes.
    n_candidates
        Length of the candidate list the decision saw.
    diagnostics
        Search counters.
    """

    status: str
    candidate: Candidate | None
    attitude: np.ndarray | None
    observed_indices: np.ndarray
    catalogue_indices: np.ndarray
    confidence: float
    n_candidates: int
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def identified(self) -> bool:
        """True if a candidate was accepted."""
        return self.status == "identified"


def _magnitude_features(
    observed_mag: np.ndarray, catalogue_mag: np.ndarray
) -> tuple[float, float]:
    """RMS magnitude residual after removing the zero point, and rank agreement.

    The zero point of an instrument magnitude is not known a priori, so the
    mean offset is removed before the residual is taken. Rank agreement is the
    fraction of the three pairwise brightness comparisons that agree.
    """
    d = observed_mag - catalogue_mag
    resid = float(np.sqrt(np.mean((d - d.mean()) ** 2)))
    agree = 0
    for p in range(3):
        for q in range(p + 1, 3):
            if (observed_mag[p] < observed_mag[q]) == (catalogue_mag[p] < catalogue_mag[q]):
                agree += 1
    return resid, agree / 3.0


def gather_candidates(
    vectors: np.ndarray,
    magnitudes: np.ndarray,
    table: PairTable,
    tolerance_rad: float,
    camera: CameraModel,
    config: SearchConfig | None = None,
    stop_when_confirmed: bool = False,
) -> tuple[list[Candidate], dict[str, float]]:
    """Stage 1: every candidate triangle, with its supporting evidence.

    Parameters
    ----------
    vectors
        ``(n, 3)`` measured unit directions in the camera frame, brightest
        first. Nothing else about the frame may be used: this is the
        lost-in-space problem.
    magnitudes
        ``(n,)`` measured instrument magnitudes, used only as features.
    table
        Catalogue pair table.
    tolerance_rad
        Angular match tolerance [rad], from
        :func:`skymatch.triangle.separation_tolerance`.
    camera
        Camera model, for the expected-star-count feature.
    config
        Search limits; defaults to :class:`SearchConfig`.
    stop_when_confirmed
        Return as soon as a triple satisfies the Pyramid rule. This reproduces
        the classical algorithm's early exit and is how its runtime is
        measured. The learned ranker needs ``False``.

    Returns ``(candidates, diagnostics)``.
    """
    cfg = config or SearchConfig()
    v = np.asarray(vectors, dtype=float)
    mags = np.asarray(magnitudes, dtype=float).reshape(-1)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vectors must have shape (n, 3), got {v.shape}")
    if mags.shape[0] != v.shape[0]:
        raise ValueError(
            f"magnitudes has length {mags.shape[0]}, expected {v.shape[0]}"
        )
    if tolerance_rad <= 0.0:
        raise ValueError(f"tolerance_rad must be > 0, got {tolerance_rad}")

    n = v.shape[0]
    diagnostics = {
        "n_spots": float(n),
        "triples_tried": 0.0,
        "triangles_found": 0.0,
        "triples_abandoned": 0.0,
        "confirmations_run": 0.0,
    }
    candidates: list[Candidate] = []
    if n < 4:
        return candidates, diagnostics

    cat_mag = table.catalogue.magnitude
    log_tol = float(np.log10(max(tolerance_rad / ARCSEC, 1e-6)))
    expected_in_field = table.catalogue.expected_in_solid_angle(camera.solid_angle_sr)

    sep = observed_separations(v)

    for position, (i, j, k) in enumerate(triple_scan_order(n)):
        if position >= cfg.max_triples or len(candidates) >= cfg.max_candidates:
            break
        diagnostics["triples_tried"] += 1.0
        t_ij, t_ik, t_jk = sep[i, j], sep[i, k], sep[j, k]
        a, b, c, res = triangle_candidates(
            table, t_ij, t_ik, t_jk, tolerance_rad, cfg.max_candidates_per_triple + 1
        )
        if a.size == 0:
            continue
        if a.size > cfg.max_candidates_per_triple:
            diagnostics["triples_abandoned"] += 1.0
            continue
        diagnostics["triangles_found"] += float(a.size)

        others = [r for r in range(n) if r not in (i, j, k)][: cfg.max_confirm_stars]
        confirm_obs: list[list[int]] = [[] for _ in range(a.size)]
        confirm_cat: list[list[int]] = [[] for _ in range(a.size)]
        confirm_res: list[list[float]] = [[] for _ in range(a.size)]
        if others:
            # One join for every (candidate triangle, fourth spot) pair, not one
            # per fourth spot: the numpy call overhead dominates at these sizes.
            n_r = len(others)
            big_a = np.tile(a, n_r)
            big_b = np.tile(b, n_r)
            big_c = np.tile(c, n_r)
            t_ar = np.repeat(sep[i, others], a.size)
            t_br = np.repeat(sep[j, others], a.size)
            t_cr = np.repeat(sep[k, others], a.size)
            diagnostics["confirmations_run"] += float(n_r)
            n_match, star_d, rms = confirm_with_fourth_star(
                table, big_a, big_b, big_c, t_ar, t_br, t_cr, tolerance_rad
            )
            for flat in np.flatnonzero(n_match == 1):
                h = int(flat % a.size)
                confirm_obs[h].append(others[int(flat // a.size)])
                confirm_cat[h].append(int(star_d[flat]))
                confirm_res[h].append(float(rms[flat]))

        n_rivals = int(a.size) - 1
        obs_mag = mags[[i, j, k]]
        edges = np.array([t_ij, t_ik, t_jk])
        for m_idx in range(a.size):
            trio = (int(a[m_idx]), int(b[m_idx]), int(c[m_idx]))
            mag_resid, mag_rank = _magnitude_features(obs_mag, cat_mag[list(trio)])
            n_conf = len(confirm_obs[m_idx])
            conf_norm = (
                float(np.mean(confirm_res[m_idx])) / tolerance_rad
                if n_conf
                else _NO_CONFIRMATION_RESIDUAL
            )
            abs_res = np.abs(res[m_idx])
            feats = np.array(
                [
                    float(abs_res.max()) / tolerance_rad,
                    float(np.sqrt(np.mean(abs_res**2))) / tolerance_rad,
                    float(n_conf),
                    float(n_conf) / max(len(others), 1),
                    conf_norm,
                    float(np.log1p(n_rivals)),
                    float(edges.min()),
                    float(edges.mean()),
                    mag_resid,
                    mag_rank,
                    log_tol,
                    float(expected_in_field),
                    float(n),
                ]
            )
            candidates.append(
                Candidate(
                    observed=(i, j, k),
                    catalogue=trio,
                    scan_position=position,
                    n_rivals=n_rivals,
                    edge_residuals=res[m_idx].copy(),
                    confirm_observed=tuple(confirm_obs[m_idx]),
                    confirm_catalogue=tuple(confirm_cat[m_idx]),
                    confirm_residual_rms=(
                        float(np.mean(confirm_res[m_idx])) if n_conf else float("nan")
                    ),
                    features=feats,
                )
            )
        if stop_when_confirmed:
            confirmed = [c for c in candidates if c.scan_position == position and c.n_confirm]
            if len(confirmed) == 1:
                break
    return candidates, diagnostics


def triangle_decision(candidates: list[Candidate]) -> Candidate | None:
    """Accept the first observed triple whose catalogue triangle is unique.

    No fourth-star check. This is the rule the Pyramid algorithm replaces, and
    it is here so that the improvement can be measured rather than asserted.
    """
    for position in sorted({c.scan_position for c in candidates}):
        group = [c for c in candidates if c.scan_position == position]
        if len(group) == 1:
            return group[0]
    return None


def pyramid_decision(candidates: list[Candidate]) -> Candidate | None:
    """The classical Pyramid rule: first triple with exactly one confirmed candidate.

    "Confirmed" means at least one fourth spot satisfied Eq. Y1 against a
    single catalogue star. If two candidates for the same triple are both
    confirmed the triple is ambiguous and is skipped, which is the behaviour
    that keeps the false-identification rate low.
    """
    for position in sorted({c.scan_position for c in candidates}):
        confirmed = [c for c in candidates if c.scan_position == position and c.n_confirm]
        if len(confirmed) == 1:
            return confirmed[0]
    return None


def resolve(
    candidate: Candidate | None,
    vectors: np.ndarray,
    catalogue: StarCatalogue,
    camera: CameraModel,
    tolerance_rad: float,
    confidence: float = 1.0,
    n_candidates: int = 0,
    diagnostics: dict[str, float] | None = None,
) -> Identification:
    """Turn an accepted candidate into an attitude and a full correspondence.

    Solves Eq. G3 on the triangle plus confirming spots, then rotates the
    catalogue stars the estimate puts inside the field into the camera frame
    and matches each remaining spot to the nearest one within
    ``tolerance_rad``. A spot with two catalogue stars inside the window is
    left unmatched rather than guessed.

    Returns a ``"no_solution"`` :class:`Identification` when ``candidate`` is
    ``None``.
    """
    diag = dict(diagnostics or {})
    if candidate is None:
        return Identification(
            status="no_solution",
            candidate=None,
            attitude=None,
            observed_indices=np.empty(0, dtype=np.int64),
            catalogue_indices=np.empty(0, dtype=np.int64),
            confidence=0.0,
            n_candidates=n_candidates,
            diagnostics=diag,
        )
    v = np.asarray(vectors, dtype=float)
    obs = np.array(candidate.all_observed, dtype=np.int64)
    cat = np.array(candidate.all_catalogue, dtype=np.int64)
    attitude = davenport_attitude(v[obs], catalogue.vectors[cat])

    boresight = attitude.T @ np.array([0.0, 0.0, 1.0])
    near = catalogue.stars_within(boresight, camera.half_diagonal_rad)
    matched_obs = list(obs)
    matched_cat = list(cat)
    if near.size:
        projected = catalogue.vectors[near] @ attitude.T
        for spot in range(v.shape[0]):
            if spot in obs:
                continue
            cosang = projected @ v[spot]
            within = np.flatnonzero(cosang >= np.cos(tolerance_rad))
            if within.size == 1:
                matched_obs.append(spot)
                matched_cat.append(int(near[within[0]]))
    diag["n_extended"] = float(len(matched_obs) - len(obs))
    return Identification(
        status="identified",
        candidate=candidate,
        attitude=attitude,
        observed_indices=np.array(matched_obs, dtype=np.int64),
        catalogue_indices=np.array(matched_cat, dtype=np.int64),
        confidence=float(confidence),
        n_candidates=n_candidates,
        diagnostics=diag,
    )
