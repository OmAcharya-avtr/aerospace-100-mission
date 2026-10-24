"""Seeded reproducibility, asserted by test at every level that has a seed."""

from __future__ import annotations

import numpy as np
import pytest

from timetransfer.acquire import acquire
from timetransfer.asym import (
    GeneratorConfig,
    LearnedAsymmetryCorrector,
    RidgeAsymmetryCorrector,
    TrainingConfig,
    build_dataset,
)
from timetransfer.backends import SimulatedBackend, make_backend_pair
from timetransfer.clockmodel import ClockModel, ClockSpec
from timetransfer.linkmodel import LinkModel, LinkSpec
from timetransfer.noise import synthetic_phase
from timetransfer.skew import fit_least_squares, kalman_skew
from timetransfer.twoway import two_way_estimate

from conftest import make_spec


def _run(seed: int, n: int = 150) -> tuple[bytes, float, float]:
    backend = SimulatedBackend(make_spec(seed=seed, q1=1e-20, q2=1e-26, loss_probability=0.1))
    backend.open()
    exchanges = acquire(backend, n, min_exchanges=8).exchanges
    backend.close()
    estimate = two_way_estimate(exchanges)
    fit = fit_least_squares(estimate.midpoint_s, estimate.offset_s, order=2)
    kal = kalman_skew(estimate.midpoint_s, estimate.offset_s)
    return exchanges.timestamp_bytes(), fit.offset0_s, kal.final_offset_s


def test_an_acquisition_is_reproducible_from_its_seed() -> None:
    first = _run(31337)
    second = _run(31337)
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert first[2] == second[2]


def test_a_different_seed_gives_a_different_realisation() -> None:
    assert _run(1)[0] != _run(2)[0]


def test_the_digest_is_stable_across_repeated_runs() -> None:
    digests = set()
    for _ in range(3):
        backend = SimulatedBackend(make_spec(seed=4242, q1=1e-20))
        backend.open()
        digests.add(acquire(backend, 80, min_exchanges=8).exchanges.digest())
        backend.close()
    assert len(digests) == 1


def test_clock_and_link_models_are_reproducible() -> None:
    def clock(seed: int) -> list[float]:
        model = ClockModel(ClockSpec(q1_s2_per_s=1e-18, q2_per_s=1e-24), seed=seed)
        return [model.offset_at(float(t)) for t in range(50)]

    def link(seed: int) -> list[tuple[float, float, bool]]:
        model = LinkModel(LinkSpec(loss_probability=0.2), seed=seed)
        return [model.draw(float(t)) for t in range(50)]

    assert clock(5) == clock(5)
    assert clock(5) != clock(6)
    assert link(5) == link(5)
    assert link(5) != link(6)


def test_synthetic_noise_is_reproducible() -> None:
    a = synthetic_phase("random_walk_fm", 300, tau0_s=1.0, level=1e-24, seed=77)
    b = synthetic_phase("random_walk_fm", 300, tau0_s=1.0, level=1e-24, seed=77)
    assert a.tobytes() == b.tobytes()


def test_the_backend_pair_is_reproducible_together_and_separately() -> None:
    spec = make_spec(seed=909, q1=1e-20, loss_probability=0.07)
    sim_a, dev_a = make_backend_pair(spec)
    sim_b, dev_b = make_backend_pair(spec)
    for backend in (sim_a, dev_a, sim_b, dev_b):
        backend.open()
    a1 = acquire(sim_a, 100, min_exchanges=8).exchanges.timestamp_bytes()
    a2 = acquire(dev_a, 100, min_exchanges=8).exchanges.timestamp_bytes()
    b1 = acquire(sim_b, 100, min_exchanges=8).exchanges.timestamp_bytes()
    b2 = acquire(dev_b, 100, min_exchanges=8).exchanges.timestamp_bytes()
    assert a1 == a2 == b1 == b2


def test_the_dataset_is_reproducible_from_its_base_seed() -> None:
    config = GeneratorConfig(window=40)
    a = build_dataset(40, seed=5150, config=config)
    b = build_dataset(40, seed=5150, config=config)
    assert a.features.tobytes() == b.features.tobytes()
    assert a.target_s.tobytes() == b.target_s.tobytes()
    assert a.seeds.tolist() == b.seeds.tolist()


def test_scenario_i_depends_only_on_base_seed_plus_i() -> None:
    config = GeneratorConfig(window=40)
    whole = build_dataset(20, seed=700, config=config)
    shifted = build_dataset(20, seed=705, config=config)
    # scenario 5 of the first dataset and scenario 0 of the second share a seed.
    assert whole.features[5].tobytes() == shifted.features[0].tobytes()
    assert whole.target_s[5] == shifted.target_s[0]


def test_the_learned_models_are_reproducible_on_the_same_split() -> None:
    dataset = build_dataset(140, seed=818, config=GeneratorConfig(window=40))
    train, test = dataset.split(train_fraction=0.7)
    config = TrainingConfig(n_estimators=40, min_samples_leaf=4)

    def predictions() -> tuple[bytes, bytes]:
        gbr = LearnedAsymmetryCorrector(config).fit(
            dataset.features[train], dataset.target_s[train]
        )
        ridge = RidgeAsymmetryCorrector().fit(dataset.features[train], dataset.target_s[train])
        return (
            gbr.predict(dataset.features[test]).tobytes(),
            ridge.predict(dataset.features[test]).tobytes(),
        )

    first = predictions()
    second = predictions()
    assert first == second


def test_the_learned_uncertainty_is_reproducible() -> None:
    dataset = build_dataset(140, seed=818, config=GeneratorConfig(window=40))
    train, test = dataset.split(train_fraction=0.7)
    config = TrainingConfig(n_estimators=40, min_samples_leaf=4)
    sigmas = []
    for _ in range(2):
        model = LearnedAsymmetryCorrector(config).fit(
            dataset.features[train], dataset.target_s[train]
        )
        sigmas.append(model.predict_with_uncertainty(dataset.features[test])[1])
    assert np.asarray(sigmas[0]).tobytes() == np.asarray(sigmas[1]).tobytes()


def test_snapshot_restore_reproduces_an_acquisition_exactly() -> None:
    backend = SimulatedBackend(make_spec(seed=606, q1=1e-20, loss_probability=0.1))
    backend.open()
    snapshot = backend.snapshot()
    first = acquire(backend, 90, min_exchanges=8).exchanges
    backend.restore(snapshot)
    second = acquire(backend, 90, min_exchanges=8).exchanges
    backend.close()
    assert first.timestamp_bytes() == second.timestamp_bytes()


def test_reported_uncertainty_is_reproducible() -> None:
    from timetransfer.uncertainty import AsymmetrySource, offset_uncertainty

    values = []
    for _ in range(2):
        backend = SimulatedBackend(make_spec(seed=424, q1=1e-20, timestamp_noise_s=2e-8))
        backend.open()
        exchanges = acquire(backend, 200, min_exchanges=8).exchanges
        backend.close()
        estimate = two_way_estimate(exchanges)
        fit = fit_least_squares(estimate.midpoint_s, estimate.offset_s, order=1)
        unc = offset_uncertainty(
            fit,
            asymmetry_source=AsymmetrySource.DECLARED_BOUND,
            asymmetry_bound_s=1e-3,
        )
        values.append((unc.offset_s, unc.u_combined_s, unc.n_effective))
    assert values[0] == values[1]
    assert values[0][1] == pytest.approx(values[1][1], rel=0.0, abs=0.0)
