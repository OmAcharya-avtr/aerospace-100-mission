"""Performance benchmarks (Level 3 requirement).

These assert *budgets*, not competitive timings: the whole point is that a
change which makes the simulation an order of magnitude slower fails the suite
rather than quietly costing every downstream user.  The limits are generous
(roughly 5x the times measured on the 2-core build machine) so that a slower
CI host does not produce spurious failures.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest

from waveforge.atmosphere import phase_screen
from waveforge.loop import AOConfig, AOSystem
from waveforge.predictor import LinearSlopePredictor, build_lagged_dataset

BENCH_CONFIG = AOConfig(n_pix=32, n_sub=4, n_act=5, screen_pixels=256, n_subharmonics=3, seed=7)


def _time(callable_, repeats: int = 1) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        callable_()
    return (time.perf_counter() - start) / repeats


class TestThroughput:
    def test_phase_screen_generation(self):
        # 512 x 512 with three subharmonic levels: measured ~0.06 s
        elapsed = _time(lambda: phase_screen(512, 0.01, 0.1, n_subharmonics=3, rng=0), repeats=3)
        assert elapsed < 1.0

    def test_system_assembly(self):
        # includes the screen, the slope operator and the SVD reconstructor
        assert _time(lambda: AOSystem(BENCH_CONFIG)) < 5.0

    def test_closed_loop_frame_rate(self):
        system = AOSystem(BENCH_CONFIG)
        n_frames = 200
        elapsed = _time(lambda: system.run(n_frames, warmup_frames=50))
        per_frame_ms = 1e3 * elapsed / n_frames
        # measured ~0.6 ms/frame for this geometry
        assert per_frame_ms < 10.0

    def test_predictor_training(self):
        system = AOSystem(BENCH_CONFIG)
        sequences = [system.open_loop_slopes(150, start_frame=offset) for offset in (0, 180)]
        elapsed = _time(
            lambda: LinearSlopePredictor(n_history=4, horizon=2, n_members=4).fit(sequences)
        )
        assert elapsed < 20.0

    def test_batch_prediction_is_vectorised(self):
        system = AOSystem(BENCH_CONFIG)
        sequences = [system.open_loop_slopes(150)]
        model = LinearSlopePredictor(n_history=4, horizon=2, n_members=4).fit(sequences)
        x, _ = build_lagged_dataset(sequences, 4, 2)
        batch = _time(lambda: model.predict_batch(x))
        single = _time(lambda: model.predict(sequences[0][:4]), repeats=5)
        # a batch of ~145 samples must cost far less than 145 single calls
        assert batch < 20.0 * single


class TestScaling:
    def test_loop_cost_is_linear_in_frames(self):
        system = AOSystem(BENCH_CONFIG)
        short = _time(lambda: system.run(60, warmup_frames=10))
        long = _time(lambda: system.run(240, warmup_frames=10))
        assert long < 8.0 * short

    def test_memory_footprint_of_the_operators(self):
        # The interaction, reconstruction and influence matrices must stay well
        # under 1 MB for this geometry so nothing needs to be cached on disk.
        system = AOSystem(BENCH_CONFIG)
        total = sum(
            array.nbytes
            for array in (
                system.interaction_matrix,
                system.reconstructor,
                system.mirror.influence_matrix,
                system.sensor.operator,
            )
        )
        assert total < 1_000_000

    def test_flagship_geometry_operators_stay_under_the_commit_limit(self):
        # The default 64-pixel, 8x8, 9x9 configuration is the one documented in
        # the README; its operators must also stay below the 1 MB threshold
        # that would force them to be committed as data.
        config = replace(AOConfig(), screen_pixels=256, n_subharmonics=0)
        system = AOSystem(config)
        assert system.interaction_matrix.nbytes < 1_000_000
        assert system.reconstructor.nbytes < 1_000_000

    def test_no_nan_under_repeated_runs(self):
        system = AOSystem(BENCH_CONFIG)
        for offset in (0, 50, 100):
            result = system.run(60, warmup_frames=10, start_frame=offset)
            assert np.all(np.isfinite(result.residual_variance))


@pytest.mark.parametrize("n_frames", [50, 100])
def test_run_completes_within_the_compute_budget(n_frames):
    """No single simulation in this package may take minutes."""
    system = AOSystem(BENCH_CONFIG)
    assert _time(lambda: system.run(n_frames, warmup_frames=10)) < 10.0
