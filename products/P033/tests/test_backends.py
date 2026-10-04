"""Backend contract, the injected cost model, and dry-run behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from edgeinfer.backends import (
    InferenceBackend,
    OnnxRuntimeBackend,
    PipelineStage,
    SimulatedBackend,
    SklearnBackend,
)


class TestPipelineStage:
    def test_constant_stage_returns_its_mean(self, rng) -> None:
        stage = PipelineStage("s", 1e-4, 0.0, "constant")
        assert stage.sample(rng) == pytest.approx(1e-4)

    def test_zero_sigma_returns_the_mean_whatever_the_distribution(self, rng) -> None:
        for dist in ("normal", "lognormal"):
            assert PipelineStage("s", 2e-4, 0.0, dist).sample(rng) == pytest.approx(2e-4)

    def test_lognormal_draws_reproduce_the_declared_mean_and_sd(self) -> None:
        """Johnson, Kotz & Balakrishnan 1994 ch. 14 moment relations:
        sigma^2 = ln(1 + (sd/mean)^2), mu = ln(mean) - sigma^2/2 gives a
        lognormal whose mean and sd are the declared ones."""
        stage = PipelineStage("s", 100e-6, 30e-6, "lognormal")
        rng = np.random.default_rng(11)
        draws = np.array([stage.sample(rng) for _ in range(40_000)])
        assert float(draws.mean()) == pytest.approx(100e-6, rel=0.02)
        assert float(draws.std(ddof=1)) == pytest.approx(30e-6, rel=0.05)

    def test_lognormal_is_never_negative(self) -> None:
        stage = PipelineStage("s", 10e-6, 50e-6, "lognormal")
        rng = np.random.default_rng(12)
        assert all(stage.sample(rng) > 0 for _ in range(2000))

    def test_normal_is_clipped_at_zero(self) -> None:
        stage = PipelineStage("s", 1e-6, 10e-6, "normal")
        rng = np.random.default_rng(13)
        draws = np.array([stage.sample(rng) for _ in range(2000)])
        assert draws.min() >= 0.0
        assert (draws == 0.0).any()  # clipping did occur, so the mean is distorted

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"mean_s": 0.0}, "mean_s"),
            ({"mean_s": -1e-6}, "mean_s"),
            ({"std_s": -1e-6}, "std_s"),
            ({"dist": "weibull"}, "dist must be"),
        ],
    )
    def test_invalid_stages_are_rejected(self, kwargs: dict, match: str) -> None:
        base = {"name": "s", "mean_s": 1e-4, "std_s": 0.0, "dist": "lognormal"}
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            PipelineStage(**base)

    def test_as_dict_has_the_crosscheck_keys(self) -> None:
        payload = PipelineStage("s", 1e-4, 2e-5, "lognormal").as_dict()
        assert set(payload) == {"name", "mean_s", "std_s", "dist"}


class TestSimulatedBackend:
    def test_injected_mean_is_the_sum_of_the_stage_means(self) -> None:
        stages = (
            PipelineStage("a", 10e-6, 0.0, "constant"),
            PipelineStage("b", 30e-6, 0.0, "constant"),
        )
        assert SimulatedBackend(stages).injected_mean_s == pytest.approx(40e-6)

    def test_injected_sd_adds_variances(self) -> None:
        """Independent stages: var(sum) = sum(var), so sd = sqrt(3^2 + 4^2) = 5."""
        stages = (
            PipelineStage("a", 100e-6, 3e-6),
            PipelineStage("b", 100e-6, 4e-6),
        )
        assert SimulatedBackend(stages).injected_std_s == pytest.approx(5e-6)

    def test_sampling_without_consuming_time_recovers_the_cost_model(self) -> None:
        stages = (
            PipelineStage("a", 100e-6, 20e-6),
            PipelineStage("b", 50e-6, 10e-6),
        )
        backend = SimulatedBackend(stages, seed=21, consume_time=False)
        backend.prepare()
        draws = np.array([backend.infer() for _ in range(20_000)])
        assert float(draws.mean()) == pytest.approx(backend.injected_mean_s, rel=0.02)
        assert float(draws.std(ddof=1)) == pytest.approx(backend.injected_std_s, rel=0.08)

    def test_prepare_resets_the_stream_so_a_run_is_reproducible(self) -> None:
        stages = (PipelineStage("a", 100e-6, 20e-6),)
        backend = SimulatedBackend(stages, seed=5, consume_time=False)
        backend.prepare()
        first = [backend.infer() for _ in range(10)]
        backend.prepare()
        second = [backend.infer() for _ in range(10)]
        assert first == second

    def test_a_different_seed_gives_a_different_stream(self) -> None:
        stages = (PipelineStage("a", 100e-6, 20e-6),)
        a = SimulatedBackend(stages, seed=1, consume_time=False)
        b = SimulatedBackend(stages, seed=2, consume_time=False)
        a.prepare()
        b.prepare()
        assert a.infer() != b.infer()

    def test_last_sampled_records_every_stage(self) -> None:
        stages = (PipelineStage("a", 1e-5, 0.0, "constant"),
                  PipelineStage("b", 2e-5, 0.0, "constant"))
        backend = SimulatedBackend(stages, consume_time=False)
        backend.prepare()
        total = backend.infer()
        assert len(backend.last_sampled_s) == 2
        assert sum(backend.last_sampled_s) == pytest.approx(total)

    def test_infer_before_prepare_is_an_error(self) -> None:
        backend = SimulatedBackend((PipelineStage("a", 1e-5),), consume_time=False)
        with pytest.raises(RuntimeError, match="prepare"):
            backend.infer()

    def test_no_stages_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one stage"):
            SimulatedBackend(())

    def test_negative_work_bytes_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="work_bytes"):
            SimulatedBackend((PipelineStage("a", 1e-5),), work_bytes=-1)

    def test_work_bytes_shows_up_in_a_memory_measurement(self) -> None:
        from edgeinfer.memtrace import measure_peak_python_bytes

        backend = SimulatedBackend(
            (PipelineStage("a", 1e-6, 0.0, "constant"),),
            consume_time=False,
            work_bytes=2_000_000,
        )
        backend.prepare()
        _result, peak = measure_peak_python_bytes(backend.infer)
        assert peak.peak_above_baseline_bytes >= 2_000_000

    def test_pipeline_dict_is_the_crosscheck_definition(self) -> None:
        stages = (PipelineStage("a", 1e-4, 1e-5, "lognormal"),)
        payload = SimulatedBackend(stages, seed=9).pipeline_dict(n_samples=500)
        assert payload["seed"] == 9
        assert payload["n_samples"] == 500
        assert payload["stages"][0]["name"] == "a"

    def test_the_backend_declares_its_kind(self) -> None:
        assert SimulatedBackend((PipelineStage("a", 1e-5),)).kind == "simulated"


class TestOnnxRuntimeBackend:
    def test_runs_and_matches_a_numpy_recomputation(self, hand_mlp) -> None:
        feed = hand_mlp.input_feed(3)
        with OnnxRuntimeBackend(hand_mlp.model_bytes, feed) as backend:
            got = backend.infer()[0]
        arrays = hand_mlp.initializer_arrays
        hidden = np.maximum(feed["X"] @ arrays["W0"] + arrays["B0"], 0.0)
        assert np.allclose(got, hidden @ arrays["W1"] + arrays["B1"], atol=1e-5)

    def test_infer_before_prepare_is_an_error(self, hand_mlp) -> None:
        backend = OnnxRuntimeBackend(hand_mlp.model_bytes, hand_mlp.input_feed(0))
        with pytest.raises(RuntimeError, match="prepare"):
            backend.infer()

    def test_close_is_idempotent(self, hand_mlp) -> None:
        backend = OnnxRuntimeBackend(hand_mlp.model_bytes, hand_mlp.input_feed(0))
        backend.prepare()
        backend.close()
        backend.close()
        assert backend.session is None

    def test_zero_threads_is_rejected(self, hand_mlp) -> None:
        with pytest.raises(ValueError, match="intra_op_num_threads"):
            OnnxRuntimeBackend(
                hand_mlp.model_bytes, hand_mlp.input_feed(0), intra_op_num_threads=0
            )

    def test_dry_run_discards_the_output(self, hand_mlp) -> None:
        from edgeinfer.backends import DRY_RUN_SINK

        with OnnxRuntimeBackend(hand_mlp.model_bytes, hand_mlp.input_feed(0)) as backend:
            backend.dry_run()
        assert DRY_RUN_SINK == []

    def test_the_backend_declares_its_kind(self, hand_mlp) -> None:
        assert OnnxRuntimeBackend(hand_mlp.model_bytes, {}).kind == "onnxruntime"


class TestSklearnBackend:
    def test_predicts_through_the_contract(self) -> None:
        from sklearn.linear_model import Ridge

        rng = np.random.default_rng(1)
        x = rng.standard_normal((40, 4))
        y = x @ np.array([1.0, -2.0, 0.5, 3.0])
        model = Ridge(alpha=1e-8).fit(x, y)
        with SklearnBackend(model, x[:5]) as backend:
            out = backend.infer()
        assert out.shape == (5,)
        assert np.allclose(out, y[:5], atol=1e-4)

    def test_an_object_without_predict_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="no predict"):
            SklearnBackend(object(), np.zeros((2, 2)))

    def test_the_backend_declares_its_kind(self) -> None:
        from sklearn.dummy import DummyRegressor

        model = DummyRegressor().fit(np.zeros((3, 1)), np.zeros(3))
        assert SklearnBackend(model, np.zeros((1, 1))).kind == "sklearn"


class TestContract:
    def test_every_backend_is_an_inference_backend(self, hand_mlp) -> None:
        from sklearn.dummy import DummyRegressor

        model = DummyRegressor().fit(np.zeros((3, 1)), np.zeros(3))
        for backend in (
            SimulatedBackend((PipelineStage("a", 1e-5),)),
            OnnxRuntimeBackend(hand_mlp.model_bytes, {}),
            SklearnBackend(model, np.zeros((1, 1))),
        ):
            assert isinstance(backend, InferenceBackend)

    def test_the_abstract_base_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            InferenceBackend()  # type: ignore[abstract]

    def test_context_manager_prepares_and_closes(self, hand_mlp) -> None:
        backend = OnnxRuntimeBackend(hand_mlp.model_bytes, hand_mlp.input_feed(0))
        with backend as entered:
            assert entered is backend
            assert backend.session is not None
        assert backend.session is None
