"""Attainable-moment-set tests, including the closed-form known answers."""

import numpy as np
import pytest

from alloclab.ams import (
    MAX_BRUTEFORCE_EFFECTORS,
    attainable_moment_set,
    expected_vertex_count,
    zonotope_volume,
)
from alloclab.dataset import reference_thruster_cluster
from alloclab.effectors import (
    general_effector_set,
    orthogonal_effectors,
    pyramid_reaction_wheels,
)


# --------------------------------------------------------------------------
# Known-answer: the orthogonal triad's AMS is exactly the cube [-1, 1]^3
# --------------------------------------------------------------------------


def test_triad_ams_is_the_unit_cube():
    e = orthogonal_effectors(1.0)
    ams = attainable_moment_set(e)
    assert ams.n_vertices == 8
    assert ams.volume == pytest.approx(8.0, abs=1e-12)
    assert ams.area == pytest.approx(24.0, abs=1e-12)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    got = np.sort(np.round(ams.vertices, 12), axis=0)
    want = np.sort(corners.astype(float), axis=0)
    assert np.allclose(got, want)


def test_triad_ams_scales_with_the_bound():
    e = orthogonal_effectors(2.0)
    ams = attainable_moment_set(e)
    assert ams.volume == pytest.approx(64.0, abs=1e-10)


def test_triad_closed_form_volume_matches_hull():
    e = orthogonal_effectors(1.5)
    assert zonotope_volume(e) == pytest.approx(attainable_moment_set(e).volume, rel=1e-12)


def test_triad_boundary_scale_along_x_is_the_bound():
    ams = attainable_moment_set(orthogonal_effectors(1.0))
    assert ams.boundary_scale([1.0, 0.0, 0.0]) == pytest.approx(1.0, abs=1e-12)
    # Along the body diagonal the cube's boundary is at sqrt(3).
    assert ams.boundary_scale([1.0, 1.0, 1.0]) == pytest.approx(np.sqrt(3.0), abs=1e-9)


# --------------------------------------------------------------------------
# Known-answer: vertex count of a zonotope in general position
# --------------------------------------------------------------------------


def test_pyramid_vertex_count_matches_the_closed_form():
    e = pyramid_reaction_wheels(0.1)
    ams = attainable_moment_set(e)
    # 4 distinct generator lines in general position: 4*3 + 2 = 14.
    assert expected_vertex_count(e) == 14
    assert ams.n_vertices == 14


def test_reference_cluster_vertex_count_matches_the_closed_form():
    e = reference_thruster_cluster(1.0, 0.5)
    ams = attainable_moment_set(e)
    # 8 thrusters but only 4 distinct generator lines (three antiparallel
    # couples plus one skew couple), so 4*3 + 2 = 14.
    assert expected_vertex_count(e) == 14
    assert ams.n_vertices == 14


def test_parallel_generators_are_merged_in_the_vertex_formula():
    # Four columns but only three distinct lines: +x, -x (parallel), +y, +z.
    e = general_effector_set(
        np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        np.zeros(4),
        np.ones(4),
    )
    assert expected_vertex_count(e) == 8


def test_coplanar_generators_do_not_lose_facet_vertices():
    """Regression: three coplanar columns make a facet that is not a parallelogram.

    Columns 1, 2 and 3 all have a +y component and columns 1 and 3 are almost
    parallel, so the plane spanned by columns 0 and 1 contains three
    generators. Enumerating only the four corners of the (0, 1) parallelogram
    lost a vertex 0.5 (N*m) away and under-reported the volume; found by
    Hypothesis against the brute-force construction.
    """
    e = general_effector_set(
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 3.0, 1e-12]]),
        np.zeros(4),
        np.array([1.0, 1.0, 3.0, 0.5]),
    )
    pair = attainable_moment_set(e, method="pairwise")
    brute = attainable_moment_set(e, method="bruteforce")
    assert pair.volume == pytest.approx(13.5, rel=1e-9)
    assert brute.volume == pytest.approx(13.5, rel=1e-9)
    assert zonotope_volume(e) == pytest.approx(13.5, rel=1e-9)


def test_expected_vertex_count_is_zero_below_three_generators():
    e = general_effector_set(
        np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]), np.zeros(2), np.ones(2)
    )
    assert expected_vertex_count(e) == 0


# --------------------------------------------------------------------------
# The two constructions agree
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "eset",
    [
        orthogonal_effectors(1.0),
        pyramid_reaction_wheels(0.1),
        reference_thruster_cluster(1.0, 0.5),
        pyramid_reaction_wheels(0.05, half_angle_deg=35.0, n_wheels=5),
    ],
    ids=["triad", "pyramid4", "thrusters8", "pyramid5"],
)
def test_pairwise_and_bruteforce_agree(eset):
    a = attainable_moment_set(eset, method="pairwise")
    b = attainable_moment_set(eset, method="bruteforce")
    assert a.n_vertices == b.n_vertices
    assert a.volume == pytest.approx(b.volume, rel=1e-10)
    assert a.area == pytest.approx(b.area, rel=1e-9)


@pytest.mark.parametrize(
    "eset",
    [
        orthogonal_effectors(1.0),
        pyramid_reaction_wheels(0.1),
        reference_thruster_cluster(1.0, 0.5),
    ],
    ids=["triad", "pyramid4", "thrusters8"],
)
def test_hull_volume_matches_the_closed_form_zonotope_volume(eset):
    assert zonotope_volume(eset) == pytest.approx(
        attainable_moment_set(eset).volume, rel=1e-10
    )


# --------------------------------------------------------------------------
# Membership
# --------------------------------------------------------------------------


def test_contains_scalar_and_batch():
    ams = attainable_moment_set(orthogonal_effectors(1.0))
    assert bool(ams.contains([0.5, 0.5, 0.5]))
    assert not bool(ams.contains([1.5, 0.0, 0.0]))
    got = ams.contains(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    assert got.tolist() == [True, False]


def test_contains_rejects_wrong_width():
    ams = attainable_moment_set(orthogonal_effectors(1.0))
    with pytest.raises(ValueError, match="last dimension 3"):
        ams.contains(np.zeros((2, 2)))


# --------------------------------------------------------------------------
# Degenerate sets
# --------------------------------------------------------------------------


def test_rank_deficient_set_is_reported_degenerate():
    e = orthogonal_effectors(1.0).with_failures([2])
    ams = attainable_moment_set(e)
    assert ams.degenerate
    assert ams.volume == 0.0
    assert ams.hull is None
    assert ams.n_facets == 0


def test_degenerate_membership_and_scale_raise():
    ams = attainable_moment_set(orthogonal_effectors(1.0).with_failures([0, 1, 2]))
    with pytest.raises(RuntimeError, match="degenerate"):
        ams.contains([0.0, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="degenerate"):
        ams.boundary_scale([1.0, 0.0, 0.0])


def test_failed_effector_shrinks_but_does_not_empty_the_ams():
    e = reference_thruster_cluster(1.0, 0.5)
    full = attainable_moment_set(e).volume
    part = attainable_moment_set(e.with_failures([0])).volume
    assert 0.0 < part < full


def test_stuck_open_thruster_translates_the_ams():
    e = reference_thruster_cluster(1.0, 0.5)
    nominal = attainable_moment_set(e)
    stuck = attainable_moment_set(e.with_failures([0], stuck_at=1.0))
    # The set shrinks (t1 is no longer free) and its centroid moves along +x
    # by t1's contribution.
    assert stuck.volume < nominal.volume
    assert stuck.vertices[:, 0].mean() > nominal.vertices[:, 0].mean()


# --------------------------------------------------------------------------
# Input validation and limits
# --------------------------------------------------------------------------


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        attainable_moment_set(orthogonal_effectors(1.0), method="montecarlo")


def test_bruteforce_is_capped():
    m = MAX_BRUTEFORCE_EFFECTORS + 1
    e = general_effector_set(np.random.default_rng(0).normal(size=(3, m)), np.zeros(m), np.ones(m))
    with pytest.raises(ValueError, match="cap is"):
        attainable_moment_set(e, method="bruteforce")


def test_boundary_scale_rejects_a_zero_direction():
    ams = attainable_moment_set(orthogonal_effectors(1.0))
    with pytest.raises(ValueError, match="non-zero vector"):
        ams.boundary_scale([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match=r"direction must have shape \(3,\)"):
        ams.boundary_scale([1.0, 0.0])
