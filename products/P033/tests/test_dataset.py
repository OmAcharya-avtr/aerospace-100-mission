"""Synthetic population: reproducibility, runnability, and the declared ranges."""

from __future__ import annotations

import functools

import numpy as np
import pytest

from edgeinfer.analytic import analytic_estimate
from edgeinfer.dataset import (
    generate_population,
    hand_counted_cnn,
    hand_counted_mlp,
    random_cnn,
    random_mlp,
)
from edgeinfer.ops import SUPPORTED_OPS


class TestReproducibility:
    def test_the_same_seed_gives_byte_identical_models(self) -> None:
        a, _ = generate_population(8, seed=99)
        b, _ = generate_population(8, seed=99)
        assert [m.model_bytes for m in a] == [m.model_bytes for m in b]

    def test_a_different_seed_gives_different_models(self) -> None:
        a, _ = generate_population(8, seed=99)
        b, _ = generate_population(8, seed=100)
        assert [m.model_bytes for m in a] != [m.model_bytes for m in b]

    def test_the_hand_counted_networks_are_fixed(self) -> None:
        assert hand_counted_mlp().model_bytes == hand_counted_mlp().model_bytes
        assert hand_counted_cnn().model_bytes == hand_counted_cnn().model_bytes

    def test_population_size_is_honoured(self) -> None:
        models, summaries = generate_population(11, seed=1)
        assert len(models) == 11
        assert len(summaries) == 11

    def test_too_small_a_population_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_models must be >= 2"):
            generate_population(1)


@functools.lru_cache(maxsize=1)
def _population():
    """Generated once per test session: 24 models cost about 0.02 s to build,
    but every test in the class would otherwise rebuild them."""
    return generate_population(24, seed=7)


class TestPopulationProperties:
    @pytest.fixture
    def population(self):
        return _population()

    def test_both_families_are_present(self, population) -> None:
        _models, summaries = population
        families = {s.family for s in summaries}
        assert families == {"mlp", "cnn"}

    def test_every_operator_used_is_supported(self, population) -> None:
        models, _ = population
        for model in models:
            for op in model.graph.op_types:
                assert op in SUPPORTED_OPS

    def test_every_model_is_loadable_by_onnxruntime(self, population) -> None:
        import onnxruntime as ort

        models, _ = population
        for model in models:
            options = ort.SessionOptions()
            options.log_severity_level = 3
            session = ort.InferenceSession(
                model.model_bytes, sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            out = session.run(None, model.input_feed(0))[0]
            assert np.all(np.isfinite(out))

    def test_every_model_has_a_costable_graph(self, population) -> None:
        from edgeinfer.roofline import DeviceModel

        models, _ = population
        device = DeviceModel("d", 1e9, 1e9)
        for model in models:
            estimate = analytic_estimate(model.graph, device)
            assert estimate.total_flops > 0
            assert estimate.peak_memory_bytes > 0

    def test_latencies_span_at_least_two_decades_of_operation_count(self) -> None:
        """A narrow population would make any predictor comparison vacuous."""
        from edgeinfer.roofline import DeviceModel

        models, _ = generate_population(60, seed=11)
        device = DeviceModel("d", 1e9, 1e9)
        flops = np.array([analytic_estimate(m.graph, device).total_flops for m in models])
        assert flops.max() / flops.min() > 100.0

    def test_summaries_describe_each_model(self, population) -> None:
        models, summaries = population
        for model, summary in zip(models, summaries, strict=True):
            assert summary.name == model.graph.name
            assert summary.n_nodes == len(model.graph.nodes)
            assert summary.weight_bytes == model.graph.weight_bytes
            assert summary.detail


class TestGenerators:
    def test_mlp_without_hidden_layers_is_a_single_gemm(self, rng) -> None:
        model = random_mlp(rng, "flat", n_in=4, widths=(), n_out=2)
        assert model.graph.op_types == ("Gemm",)

    def test_mlp_activation_choice_is_honoured(self, rng) -> None:
        model = random_mlp(rng, "t", n_in=4, widths=(3,), n_out=2, activation="Tanh")
        assert "Tanh" in model.graph.op_types

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_in": 0}, ">= 1"),
            ({"widths": (0,)}, ">= 1"),
            ({"activation": "Swish"}, "activation must be"),
        ],
    )
    def test_invalid_mlp_arguments_are_rejected(self, rng, kwargs: dict, match: str) -> None:
        base = {"name": "m", "n_in": 4, "widths": (3,), "n_out": 2}
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            random_mlp(rng, **base)

    def test_cnn_without_pooling_keeps_the_spatial_size(self, rng) -> None:
        model = random_cnn(
            rng, "nopool", spatial=8, channels=(2,), kernel=3, pool_every=False, n_out=2
        )
        model.graph.infer_shapes()
        assert "MaxPool" not in model.graph.op_types
        assert model.graph.spec("C0").shape[2:] == (8, 8)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"channels": ()}, "at least one entry"),
            ({"kernel": 2}, "odd"),
            ({"spatial": 2, "kernel": 5}, "must be >= kernel"),
        ],
    )
    def test_invalid_cnn_arguments_are_rejected(self, rng, kwargs: dict, match: str) -> None:
        base = {"name": "c", "spatial": 8, "channels": (2,), "kernel": 3, "n_out": 2}
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            random_cnn(rng, **base)

    def test_generated_weights_stay_finite_through_the_runtime(self, rng) -> None:
        import onnxruntime as ort

        model = random_mlp(rng, "deep", n_in=64, widths=(128, 128, 128), n_out=8)
        session = ort.InferenceSession(
            model.model_bytes, providers=["CPUExecutionProvider"]
        )
        assert np.all(np.isfinite(session.run(None, model.input_feed(0))[0]))
