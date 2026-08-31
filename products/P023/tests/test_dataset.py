"""Dataset generation: determinism, shapes, labelling, input validation."""

import numpy as np
import pytest

from alloclab.ams import attainable_moment_set
from alloclab.dataset import generate_dataset, reference_thruster_cluster, torque_scale
from alloclab.effectors import orthogonal_effectors


@pytest.fixture(scope="module")
def cluster():
    return reference_thruster_cluster(1.0, 0.5)


def test_reference_cluster_columns_match_the_documented_table(cluster):
    arm = 0.5
    s = 1.0 / np.sqrt(2.0)
    want = np.array(
        [
            [arm, -arm, 0.0, 0.0, 0.0, 0.0, arm * s, -arm * s],
            [0.0, 0.0, arm, -arm, 0.0, 0.0, arm * s, -arm * s],
            [0.0, 0.0, 0.0, 0.0, arm, -arm, -2.0 * arm * s, 2.0 * arm * s],
        ]
    )
    assert np.allclose(cluster.matrix, want, atol=1e-12)
    assert np.allclose(cluster.lower, 0.0)
    assert np.allclose(cluster.upper, 1.0)


def test_reference_cluster_arm_must_be_positive():
    with pytest.raises(ValueError, match="arm must be > 0"):
        reference_thruster_cluster(1.0, 0.0)


def test_torque_scale_of_the_triad_is_the_cube_diagonal():
    # The AMS of the unit triad is [-1,1]^3; its farthest vertex is at sqrt(3).
    assert torque_scale(orthogonal_effectors(1.0)) == pytest.approx(np.sqrt(3.0), abs=1e-9)


def test_generation_is_deterministic_in_the_seed(cluster):
    a = generate_dataset(cluster, 120, seed=7)
    b = generate_dataset(cluster, 120, seed=7)
    assert np.array_equal(a.torques, b.torques)
    assert np.array_equal(a.health, b.health)
    assert np.allclose(a.commands, b.commands, atol=0.0, rtol=0.0)


def test_different_seeds_give_different_data(cluster):
    a = generate_dataset(cluster, 120, seed=7)
    b = generate_dataset(cluster, 120, seed=8)
    assert not np.allclose(a.torques, b.torques)


def test_shapes_and_length(cluster):
    ds = generate_dataset(cluster, 64, seed=3)
    assert len(ds) == 64
    assert ds.torques.shape == (64, 3)
    assert ds.health.shape == (64, 8)
    assert ds.commands.shape == (64, 8)
    assert ds.residual_norm.shape == (64,)
    assert ds.attainable.shape == (64,)
    assert ds.seed == 3


def test_every_label_is_inside_the_command_box(cluster):
    ds = generate_dataset(cluster, 250, seed=11)
    assert np.max(cluster.bound_violation(ds.commands)) <= 1e-9


def test_labels_of_failed_effectors_are_pinned_to_zero(cluster):
    ds = generate_dataset(cluster, 250, seed=12, failure_prob=1.0, max_failures=2)
    dead = ds.health == 0.0
    assert dead.any()
    assert np.max(np.abs(ds.commands[dead])) <= 1e-9


def test_attainable_flag_agrees_with_the_residual(cluster):
    ds = generate_dataset(cluster, 200, seed=13)
    assert np.all(ds.residual_norm[ds.attainable] <= 1e-8)
    assert np.all(ds.residual_norm[~ds.attainable] > 1e-8)


def test_attainable_fraction_tracks_the_magnitude_range(cluster):
    # rho ~ U(0, 1.05) puts 1/1.05 = 0.952 of the samples inside the AMS.
    ds = generate_dataset(cluster, 800, seed=5)
    assert ds.attainable_fraction == pytest.approx(1.0 / 1.05, abs=0.05)


def test_all_samples_attainable_when_magnitudes_stay_inside(cluster):
    ds = generate_dataset(cluster, 150, seed=21, magnitude_range=(0.0, 0.9))
    assert ds.attainable_fraction == 1.0


def test_no_failures_when_disabled(cluster):
    ds = generate_dataset(cluster, 100, seed=4, max_failures=0)
    assert np.all(ds.health == 1.0)
    ds = generate_dataset(cluster, 100, seed=4, failure_prob=0.0)
    assert np.all(ds.health == 1.0)


def test_commanded_torques_stay_near_the_attainable_set(cluster):
    ds = generate_dataset(cluster, 200, seed=17, max_failures=0)
    ams = attainable_moment_set(cluster)
    radius = float(np.max(np.linalg.norm(ams.vertices, axis=1)))
    assert np.max(np.linalg.norm(ds.torques, axis=1)) <= 1.05 * radius + 1e-9


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_dataset_input_validation(cluster):
    with pytest.raises(ValueError, match="n_samples must be > 0"):
        generate_dataset(cluster, 0, seed=1)
    with pytest.raises(ValueError, match=r"max_failures must be in \[0, 8\]"):
        generate_dataset(cluster, 10, seed=1, max_failures=9)
    with pytest.raises(ValueError, match=r"failure_prob must be in \[0, 1\]"):
        generate_dataset(cluster, 10, seed=1, failure_prob=1.5)
    with pytest.raises(ValueError, match="magnitude_range"):
        generate_dataset(cluster, 10, seed=1, magnitude_range=(1.0, 0.5))
    with pytest.raises(ValueError, match="magnitude_range"):
        generate_dataset(cluster, 10, seed=1, magnitude_range=(-0.1, 0.5))
