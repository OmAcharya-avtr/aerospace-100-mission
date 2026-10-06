"""End-to-end integration tests and repository hygiene checks.

These exercise the whole chain -- channel, measured thresholds, feedback delay,
policy, accounting -- and assert the properties the README claims, so that a
change which breaks a published claim fails here rather than in review.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from acmpilot import __version__
from acmpilot.benchmark import BenchmarkConfig, BenchmarkSplit, benchmark_at_delay
from acmpilot.channel import ChannelConfig, fade_statistics, snr_db_path
from acmpilot.policy import ClairvoyantUpperBound, FixedMargin, ThresholdHysteresis
from acmpilot.simulate import run_policy, sweep_delay

REPO = Path(__file__).resolve().parent.parent
ABSOLUTE_PATH = re.compile(r"(/home/[A-Za-z0-9._-]+/|/Users/[A-Za-z0-9._-]+/)")
TRACKED_SUFFIXES = (
    ".py", ".md", ".toml", ".json", ".txt", ".yml", ".yaml", ".cff", ".gitignore",
)
FORBIDDEN_PHRASES = (
    "flight-safe",
    "flight safe",
    "certified",
    "mission-ready",
    "mission ready",
    "production-ready",
    "production ready",
    "flight-qualified for",
)


class TestEndToEnd:
    def test_full_chain_at_three_delays(self, table, config):
        snr = snr_db_path(config, 20_000, 77)
        previous = None
        for tau_ms in (0.0, 5.0, 20.0):
            d = config.delay_slots(tau_ms * 1e-3)
            acc = run_policy(
                table, snr, FixedMargin(margin_db=2.0), delay_slots=d, warmup=100
            ).accounting
            assert acc.n_slots == 19_900
            assert 0.0 < acc.goodput_bit_per_symbol < float(
                table.spectral_efficiencies[-1]
            )
            if previous is not None:
                assert acc.goodput_bit_per_symbol < previous
            previous = acc.goodput_bit_per_symbol

    def test_clairvoyant_bound_holds_across_the_whole_sweep(self, table, config):
        rows = sweep_delay(
            table, config,
            (FixedMargin(2.0), ThresholdHysteresis(), ClairvoyantUpperBound()),
            tau_list_s=(0.0, 2e-3, 10e-3, 40e-3), n_slots=8000, seeds=(1, 2),
        )
        bound = {
            float(r["tau_s"]): float(r["goodput_bit_per_symbol"])
            for r in rows if not r["causal"]
        }
        assert len(set(bound.values())) == 1
        for row in rows:
            if row["causal"]:
                assert (
                    float(row["goodput_bit_per_symbol"])
                    <= bound[float(row["tau_s"])] + 1e-12
                )

    def test_tau_over_tau_c_is_what_matters(self, table):
        """Doubling both the delay and the correlation time leaves goodput close.

        Both runs have tau/tau_c = 1. They are not identical, because the slot
        rate is fixed in both so the second channel is sampled relatively
        slower, but they are much closer to each other than either is to a run
        at tau/tau_c = 4.
        """
        def goodput(tau_ms: float, tau_c_ms: float) -> float:
            cfg = ChannelConfig(tau_c_s=tau_c_ms * 1e-3, sigma_i2=0.5, mean_snr_db=14.0)
            return float(
                sweep_delay(
                    table, cfg, (FixedMargin(2.0),), tau_list_s=(tau_ms * 1e-3,),
                    n_slots=12_000, seeds=(1, 2, 3),
                )[0]["goodput_bit_per_symbol"]
            )

        same_a = goodput(5.0, 5.0)
        same_b = goodput(20.0, 20.0)
        different = goodput(20.0, 5.0)
        assert abs(same_a - same_b) < abs(same_a - different)

    def test_predictive_policy_beats_tuned_baselines_at_large_delay(self, table, config):
        """The headline claim of the README, on a small but honest benchmark.

        Baselines are tuned on the tuning seeds; everything is scored on the test
        seeds. The assertion is deliberately one-sided and loose: it checks that
        the predictive policies are not WORSE than the tuned baselines at a large
        delay, which is the direction the published numbers claim.
        """
        bench = BenchmarkConfig(
            n_slots=6000, train_slots=3000, n_lags=3, n_estimators=20,
            split=BenchmarkSplit(
                train_seeds=(101, 102), tune_seeds=(201, 202), test_seeds=(301, 302, 303)
            ),
        )
        result = benchmark_at_delay(table, config, tau_s=20e-3, bench=bench)
        tuned = max(
            float(r["goodput_bit_per_symbol"])
            for r in result["scores"] if r["variant"] == "tuned"
        )
        predictive = max(
            float(r["goodput_bit_per_symbol"])
            for r in result["scores"] if r["variant"] == "predictive"
        )
        assert predictive >= tuned

    def test_learned_and_analytic_predictors_are_close(self, table, config):
        """The published result: the learned model does not pull away.

        A 5% band is used rather than a statistical test, because the point is a
        magnitude claim, not a hypothesis test; the statistical comparison with
        standard errors is in ``validation/validate_predictor.txt``.
        """
        bench = BenchmarkConfig(
            n_slots=6000, train_slots=3000, n_lags=3, n_estimators=20,
            split=BenchmarkSplit(
                train_seeds=(101, 102), tune_seeds=(201, 202), test_seeds=(301, 302, 303)
            ),
        )
        result = benchmark_at_delay(table, config, tau_s=10e-3, bench=bench)
        learned = next(float(r["goodput_bit_per_symbol"]) for r in result["scores"] if r["learned"])
        analytic = next(
            float(r["goodput_bit_per_symbol"])
            for r in result["scores"]
            if r["variant"] == "predictive" and not r["learned"]
        )
        assert abs(learned - analytic) / analytic < 0.05

    def test_fade_statistics_of_the_reference_channel(self, config, table):
        """The reference channel's emergent fade statistics at MODCOD 0."""
        snr = snr_db_path(config, 200_000, 55)
        stats = fade_statistics(
            snr, float(table.thresholds_db[0]), dt_s=config.slot_s
        )
        assert 0.0 < stats["outage_fraction"] < 0.05
        assert stats["mean_fade_duration_s"] > 0.0
        assert stats["max_fade_duration_s"] >= stats["mean_fade_duration_s"]

    def test_gamma_gamma_channel_runs_end_to_end(self, table):
        cfg = ChannelConfig(marginal="gamma-gamma", sigma_i2=0.5, mean_snr_db=14.0)
        snr = snr_db_path(cfg, 12_000, 61)
        acc = run_policy(
            table, snr, FixedMargin(2.0), delay_slots=10, warmup=100
        ).accounting
        assert acc.goodput_bit_per_symbol > 0.0
        assert 0.0 <= acc.outage_fraction <= 1.0


class TestRepositoryHygiene:
    @staticmethod
    def _tracked_files() -> list[Path]:
        files = []
        for path in REPO.rglob("*"):
            if not path.is_file():
                continue
            parts = set(path.relative_to(REPO).parts)
            if parts & {"__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis", "artifacts"}:
                continue
            if path.suffix in TRACKED_SUFFIXES or path.name == ".gitignore":
                files.append(path)
        return files

    def test_no_absolute_paths_in_tracked_text(self):
        offenders = []
        for path in self._tracked_files():
            text = path.read_text(errors="ignore")
            if ABSOLUTE_PATH.search(text):
                offenders.append(str(path.relative_to(REPO)))
        assert offenders == []

    def test_no_model_binaries_present(self):
        bad = [
            str(p.relative_to(REPO))
            for p in REPO.rglob("*")
            if p.is_file() and p.suffix in {".pt", ".pth", ".ckpt", ".onnx"}
        ]
        assert bad == []

    def test_no_certification_claims_in_documentation(self):
        """No unqualified certification or flight-safety claim in any document.

        The required disclaimers contain these very words -- "not certified",
        "not flight-qualified" -- and so does the requirement that forbids them,
        so a line is accepted when it carries a negation or the word
        "unqualified" anywhere on it. A line containing one of these phrases with
        no negation at all is the thing being caught.
        """
        allowed = ("not ", "never ", "no ", "unqualified", "nothing ")
        offenders = []
        for path in REPO.rglob("*.md"):
            for number, line in enumerate(
                path.read_text(errors="ignore").lower().splitlines(), start=1
            ):
                for phrase in FORBIDDEN_PHRASES:
                    if phrase not in line:
                        continue
                    if any(marker in line for marker in allowed):
                        continue
                    offenders.append(
                        f"{path.relative_to(REPO)}:{number}: {phrase}"
                    )
        assert offenders == []

    def test_every_example_writes_to_screenshots(self):
        for path in sorted((REPO / "examples").glob("*.py")):
            text = path.read_text()
            assert "screenshots" in text, path.name
            assert 'matplotlib.use("Agg")' in text, path.name
            assert ".show()" not in text, path.name

    def test_every_screenshot_exists_for_every_example(self):
        examples = {p.stem for p in (REPO / "examples").glob("*.py")}
        shots = {p.stem for p in (REPO / "screenshots").glob("*.png")}
        assert examples <= shots

    def test_required_files_present(self):
        for name in (
            "README.md", "LICENSE", "CHANGELOG.md", "MODEL_CARD.md",
            "DATASET_CARD.md", "CITATION.cff", "pyproject.toml", ".gitignore",
            "docs/REQUIREMENTS.md", "validation/VALIDATION.md",
        ):
            assert (REPO / name).exists(), name

    def test_version_consistent_with_pyproject(self):
        text = (REPO / "pyproject.toml").read_text()
        assert f'version = "{__version__}"' in text

    def test_licence_is_agpl(self):
        text = (REPO / "LICENSE").read_text()
        assert "GNU AFFERO GENERAL PUBLIC LICENSE" in text.upper()
        assert "2026 OPTIMA Organisation" in text

    def test_credits_line_appears_exactly_once_in_readme(self):
        text = (REPO / "README.md").read_text()
        line = "This is under reserved rights obtained by OPTIMA Organisation."
        assert text.count(line) == 1

    def test_readme_names_p015_linkswitch(self):
        text = (REPO / "README.md").read_text()
        assert "P015" in text
        assert "LinkSwitch" in text

    def test_readme_labels_the_clairvoyant_bound(self):
        from acmpilot.policy import CLAIRVOYANT_LABEL

        text = (REPO / "README.md").read_text()
        assert "upper bound" in text.lower()
        assert "no causal policy" in text.lower()
        assert CLAIRVOYANT_LABEL.startswith("clairvoyant")

    def test_every_validation_script_has_saved_output(self):
        scripts = sorted((REPO / "validation").glob("validate_*.py"))
        assert len(scripts) >= 4
        for script in scripts:
            assert (script.with_suffix(".txt")).exists(), script.name

    def test_requirements_doc_numbers_and_names_tests(self):
        text = (REPO / "docs" / "REQUIREMENTS.md").read_text()
        ids = re.findall(r"\bR-\d{3}\b", text)
        assert len(set(ids)) >= 20
        assert "test_" in text

    @pytest.mark.parametrize("name", ["MODEL_CARD.md", "DATASET_CARD.md"])
    def test_cards_carry_the_required_statement(self, name):
        text = (REPO / name).read_text()
        assert "This model is not certified for operational flight use." in text or (
            "not certified for operational flight use" in text
        )

    def test_pyproject_has_pytest_pythonpath(self):
        text = (REPO / "pyproject.toml").read_text()
        assert "[tool.pytest.ini_options]" in text
        assert 'pythonpath = ["src"]' in text

    def test_ruff_configuration_as_specified(self):
        text = (REPO / "pyproject.toml").read_text()
        assert "line-length = 100" in text
        assert 'select = ["E", "F", "W", "I", "UP", "B"]' in text


class TestNumpyInteroperability:
    def test_no_nan_in_any_accounting_field_on_the_reference_run(self, table, config):
        snr = snr_db_path(config, 10_000, 71)
        acc = run_policy(table, snr, FixedMargin(2.0), delay_slots=10).accounting
        for key, value in acc.as_dict().items():
            assert np.isfinite(value), key
