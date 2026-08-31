"""End-to-end integration and benchmark/regression tests.

The regression tests pin numbers produced by this repository at version 0.1.0.
They exist so that a refactor that quietly changes an allocation is caught;
they are not independent evidence, which lives in ``validation/``.
"""

import time

import numpy as np
import pytest

from alloclab.allocation import METHODS, allocate, lp_allocate, qp_allocate
from alloclab.ams import attainable_moment_set, zonotope_volume
from alloclab.dataset import generate_dataset, reference_thruster_cluster
from alloclab.effectors import orthogonal_effectors, pyramid_reaction_wheels
from alloclab.failure import reallocate_after_failure
from alloclab.ml import LearnedAllocator


@pytest.fixture(scope="module")
def cluster():
    return reference_thruster_cluster(1.0, 0.5)


# --------------------------------------------------------------------------
# Integration: configuration -> AMS -> allocation -> failure -> reallocation
# --------------------------------------------------------------------------


def test_full_pipeline_thruster_cluster(cluster):
    ams = attainable_moment_set(cluster)
    assert not ams.degenerate

    direction = np.array([0.3, -0.6, 0.74])
    direction /= np.linalg.norm(direction)
    tau = 0.5 * ams.boundary_scale(direction) * direction
    assert ams.contains(tau)

    nominal = qp_allocate(cluster, tau)
    assert nominal.feasible
    assert nominal.bound_violation == pytest.approx(0.0, abs=1e-9)

    report = reallocate_after_failure(cluster, tau, failed=[6], method="qp")
    assert report.remaining_rank == 3
    if report.attainable:
        assert report.degraded.feasible
        assert report.degraded.residual_norm <= 1e-8
    else:  # pragma: no cover - configuration-dependent
        assert report.degraded.status == "infeasible"

    # Push the same direction past the degraded boundary: must be reported.
    degraded_ams = attainable_moment_set(cluster.with_failures([6]))
    far = 1.3 * degraded_ams.boundary_scale(direction) * direction
    over = reallocate_after_failure(cluster, far, failed=[6], method="qp")
    assert not over.attainable
    assert over.degraded.status == "infeasible"
    assert over.degraded.bound_violation == pytest.approx(0.0, abs=1e-9)


def test_full_pipeline_reaction_wheels():
    e = pyramid_reaction_wheels(0.1)
    ams = attainable_moment_set(e)
    assert ams.n_vertices == 14
    assert zonotope_volume(e) == pytest.approx(ams.volume, rel=1e-10)

    tau = np.array([0.02, -0.01, 0.03])
    results = {m: allocate(e, tau, method=m) for m in METHODS}
    for method, res in results.items():
        assert res.residual_norm <= 1e-7, method
    # Only the bounds-aware methods are guaranteed to be inside the box.
    for method in ("lp", "qp", "rpi"):
        assert results[method].bound_violation == pytest.approx(0.0, abs=1e-9), method


def test_ml_pipeline_end_to_end(cluster):
    train = generate_dataset(cluster, 300, seed=41)
    model = LearnedAllocator(
        cluster, n_estimators=2, hidden_layer_sizes=(32, 16), max_iter=120, random_state=0
    ).fit(train.torques, train.health, train.commands)
    test = generate_dataset(cluster, 100, seed=42)
    out = model.predict(test.torques, test.health)
    achieved = out.commands @ cluster.matrix.T
    err = np.linalg.norm(test.torques - achieved, axis=1)
    assert err.shape == (100,)
    assert np.all(np.isfinite(err))
    # The QP labels are at least as good as the learned imitation of them.
    assert np.mean(test.residual_norm) <= np.mean(err)


# --------------------------------------------------------------------------
# Benchmark / regression
# --------------------------------------------------------------------------


def test_regression_triad_ams_volume_and_vertices():
    e = orthogonal_effectors(1.0)
    ams = attainable_moment_set(e)
    assert ams.n_vertices == 8
    assert ams.volume == pytest.approx(8.0, abs=1e-12)


def test_regression_pyramid_ams_volume():
    # Pinned at 0.1 N*m per wheel, 54.7356 deg half angle. Cross-checked
    # against the closed-form zonotope volume in the same call.
    e = pyramid_reaction_wheels(0.1)
    ams = attainable_moment_set(e)
    assert ams.volume == pytest.approx(0.024633611485424, rel=1e-9)
    assert zonotope_volume(e) == pytest.approx(0.024633611485424, rel=1e-9)


def test_regression_reference_cluster_ams_volume(cluster):
    # arm = 0.5 m, max_thrust = 1 N.
    ams = attainable_moment_set(cluster)
    assert ams.volume == pytest.approx(3.828427124746190, rel=1e-9)
    assert ams.n_vertices == 14


def test_regression_qp_allocation_on_the_cluster(cluster):
    res = qp_allocate(cluster, [0.4, -0.2, 0.1])
    assert res.feasible
    assert np.allclose(
        res.commands, [0.9, 0.1, 0.3, 0.7, 0.6, 0.4, 0.5, 0.5], atol=1e-6
    )


def test_regression_dataset_attainable_fraction(cluster):
    ds = generate_dataset(cluster, 500, seed=1)
    assert ds.attainable_fraction == pytest.approx(0.948, abs=1e-9)


def test_benchmark_qp_is_faster_than_the_lp_on_this_problem(cluster):
    """Runtime benchmark. The absolute numbers are in ``validation/``.

    The assertion is deliberately loose (a factor of 3 margin) because CI
    machines vary; it fails only if the QP path picks up an order-of-magnitude
    regression relative to the LP.
    """
    taus = generate_dataset(cluster, 60, seed=2).torques

    t0 = time.perf_counter()
    for tau in taus:
        qp_allocate(cluster, tau)
    qp_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for tau in taus:
        lp_allocate(cluster, tau)
    lp_time = time.perf_counter() - t0

    assert qp_time < 3.0 * lp_time, f"qp={qp_time:.4f}s lp={lp_time:.4f}s"


def test_benchmark_allocation_throughput(cluster):
    """The QP must sustain at least 200 allocations per second on 2 cores."""
    taus = generate_dataset(cluster, 100, seed=3).torques
    t0 = time.perf_counter()
    for tau in taus:
        qp_allocate(cluster, tau)
    elapsed = time.perf_counter() - t0
    rate = len(taus) / elapsed
    assert rate > 200.0, f"{rate:.1f} allocations/s"
