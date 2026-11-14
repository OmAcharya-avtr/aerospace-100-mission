"""The measured scripts' reporting paths, exercised on SYNTHETIC input.

The three ``validation/validate_measured_*.py`` scripts refuse and exit non-zero
until the hardware campaign's traces are committed, which is the behaviour
``tests/test_refusal.py`` checks. That leaves everything AFTER the refusal
unexercised, and a formatting or arithmetic mistake in a script that cannot run
is a mistake that surfaces for the first time on the night the data lands.

So each of those scripts is split into ``main``, which refuses and loads, and
``report``, which takes an already-loaded campaign. The tests below call
``report`` with a SYNTHETIC campaign and a :class:`_bootstrap.Tee` pointed at a
temporary directory.

Two things this does NOT do, deliberately:

* it does not weaken the refusal. ``main`` is untouched and still requires a
  complete, non-synthetic campaign;
* it does not write into ``validation/``, so no committed output file can be
  overwritten with synthetic figures.

A check inside ``report`` may legitimately FAIL on synthetic input -- the
between-run interval of four byte-identical runs has zero width, for instance --
so these tests assert that the report RUNS and emits its sections, and print any
failed checks rather than requiring none.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from fixtures import synthetic_campaign

ROOT = Path(__file__).resolve().parent.parent
VALIDATION = ROOT / "validation"
if str(VALIDATION) not in sys.path:
    sys.path.insert(0, str(VALIDATION))

MODULES = (
    "validate_measured_headline",
    "validate_measured_baselines",
    "validate_measured_learned",
)


@pytest.fixture(scope="module")
def jittered(tmp_path_factory):
    """A synthetic campaign with jitter, so the between-run spread is positive."""
    return synthetic_campaign.campaign(
        tmp_path_factory.mktemp("measured-report"),
        cycles=10,
        current_jitter_lsb=2,
        time_jitter_s=0.004,
        seed=4711,
    )


@pytest.fixture(scope="module")
def bootstrap_module():
    return importlib.import_module("_bootstrap")


@pytest.mark.parametrize("module_name", MODULES)
def test_report_runs_on_a_synthetic_campaign(module_name, jittered, bootstrap_module, tmp_path):
    module = importlib.import_module(module_name)
    out = bootstrap_module.Tee(f"{module_name}_test_output.txt", directory=tmp_path)
    module.report(out, jittered)
    text = "\n".join(out.lines)
    assert out.checks > 0, "the report ran no checks at all"
    assert "campaign identity" in text
    assert len(text) > 1000
    if out.failures:
        print(f"{module_name}: checks that failed on synthetic input: {out.failures}")


def test_headline_report_emits_the_decomposition(jittered, bootstrap_module, tmp_path):
    module = importlib.import_module("validate_measured_headline")
    out = bootstrap_module.Tee("headline_test_output.txt", directory=tmp_path)
    module.report(out, jittered)
    text = "\n".join(out.lines)
    for fragment in (
        "HEADLINE understatement factor F",
        "rejection factor N/A",
        "duty-cycle factor",
        "rail-voltage factor",
        "closure error of the three",
        "PHASE OCCUPANCY, a diagnostic",
        "sampler span",
        "Riemann bracket",
        "quantiser sd",
        "cpu_busy_fraction over the run",
        "the permissive-gate control",
        "the reject-reason census",
    ):
        assert fragment in text, fragment


def test_baselines_report_emits_all_three_baselines(jittered, bootstrap_module, tmp_path):
    module = importlib.import_module("validate_measured_baselines")
    out = bootstrap_module.Tee("baselines_test_output.txt", directory=tmp_path)
    module.report(out, jittered)
    text = "\n".join(out.lines)
    for fragment in (
        "the measured background floor",
        "baseline 1 across the grid",
        "baseline 2, with every component measured",
        "baseline 2 against baseline 3",
        "the measured rail-accounting ratio against load",
        "the between-run interval",
        "measurement back-action",
        "E_RAMP IS NEGATIVE",
        "the run order was NOT randomised",
        "floor sensitivity",
        "DRIFT-ADJUSTED",
        "the reject-reason census",
    ):
        assert fragment in text, fragment


def test_learned_report_evaluates_the_pre_registration(jittered, bootstrap_module, tmp_path):
    module = importlib.import_module("validate_measured_learned")
    out = bootstrap_module.Tee("learned_test_output.txt", directory=tmp_path)
    module.report(out, jittered)
    text = "\n".join(out.lines)
    for fragment in (
        "where foreign occupancy was separable",
        "held-out duty",
        "held-out frame_dim",
        "the substitution: energy per useful decision",
        "descriptor importances",
        "THE PRE-REGISTERED EXPECTATION, evaluated",
        "VERDICT:",
    ):
        assert fragment in text, fragment
    assert ("is REFUTED" in text) or ("is NOT REFUTED" in text)


@pytest.mark.parametrize("module_name", MODULES)
def test_report_writes_nothing_into_the_validation_directory(
    module_name, jittered, bootstrap_module, tmp_path
):
    before = {path.name for path in VALIDATION.iterdir()}
    module = importlib.import_module(module_name)
    out = bootstrap_module.Tee(f"{module_name}_isolated.txt", directory=tmp_path)
    module.report(out, jittered)
    out.save()
    after = {path.name for path in VALIDATION.iterdir()}
    assert before == after
    assert (tmp_path / f"{module_name}_isolated.txt").exists()
