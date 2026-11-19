"""Nothing in this product is intermittent, and this is where that is checked.

The product is about intermittency. The mission it was built in lost a whole
product to checks that compared two wall-clock readings on a shared two-core
runner, and the rule that came out of that is binding: never assert a
relationship between two live wall-clock measurements. So:

* no module in the package imports ``time``, and this file checks that by
  parsing the sources rather than by inspection;
* every stochastic step goes through ``numpy.random.default_rng`` at a declared
  seed, and two runs at the same seed agree;
* no test asserts on a duration that was not read from the committed history.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from flakerank.baselines import accumulated_wasted_time
from flakerank.history import RunHistory, history_sha256
from flakerank.ranking import rank_tests

from _support import HISTORY_PATH, HISTORY_SHA256, ROOT

#: Modules that must not appear in any import in the package. ``timeit`` and
#: ``datetime`` are included because a duration computed from either is a live
#: measurement just as much as one from ``perf_counter``.
FORBIDDEN_IMPORTS = frozenset({"time", "timeit", "datetime", "resource"})

#: Attribute names that read a clock, checked as a second, independent screen
#: in case a clock were reached through an already-imported module.
FORBIDDEN_ATTRIBUTES = frozenset(
    {"perf_counter", "process_time", "monotonic", "time_ns", "perf_counter_ns", "clock"}
)


def _package_sources() -> list[Path]:
    """Every Python file in the package."""
    return sorted((ROOT / "src" / "flakerank").rglob("*.py"))


def test_the_package_has_sources_to_scan() -> None:
    """A scan over zero files would pass vacuously."""
    assert len(_package_sources()) >= 10


def test_no_package_module_imports_a_clock() -> None:
    """Parsed from the syntax tree, not grepped, so a comment cannot trip it."""
    offenders: list[str] = []
    for path in _package_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in FORBIDDEN_IMPORTS:
                        offenders.append(f"{path.name}:{node.lineno} import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in FORBIDDEN_IMPORTS:
                    offenders.append(f"{path.name}:{node.lineno} from {node.module}")
    assert not offenders, f"the package must read no clock: {offenders}"


def test_no_package_module_reads_a_clock_attribute() -> None:
    """A second screen: no attribute access that names a clock function."""
    offenders: list[str] = []
    for path in _package_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
                offenders.append(f"{path.name}:{node.lineno} .{node.attr}")
            elif isinstance(node, ast.Name) and node.id in FORBIDDEN_ATTRIBUTES:
                offenders.append(f"{path.name}:{node.lineno} {node.id}")
    assert not offenders, f"the package must read no clock: {offenders}"


def test_no_test_module_imports_a_clock() -> None:
    """The suite is held to the same rule as the package."""
    offenders: list[str] = []
    for path in sorted((ROOT / "tests").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in FORBIDDEN_IMPORTS:
                        offenders.append(f"{path.name}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in FORBIDDEN_IMPORTS:
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"no test may read a clock: {offenders}"


def test_committed_history_digest(history: RunHistory) -> None:
    """The digest pinned in ``tests/_support.py`` is the committed file's.

    Every number in this suite and in the documentation is a function of this
    file, so a change to it must break the suite rather than silently move the
    numbers.
    """
    assert history_sha256(HISTORY_PATH) == HISTORY_SHA256
    assert history.n_attempts == 23836


def test_two_ground_truth_runs_at_one_seed_agree(history: RunHistory) -> None:
    """Identical interval ends from two runs at the same seed.

    PCG64's draw order is fixed by the seed, so the two runs perform the same
    arithmetic on the same draws and the ends agree to the last bit. That is
    the one situation in which a zero tolerance is a correct claim, and the
    tolerance is written as ``abs=0.0`` rather than ``==`` so the intent is
    visible in the source.
    """
    first = accumulated_wasted_time(history, resamples=200, seed=97044)
    second = accumulated_wasted_time(history, resamples=200, seed=97044)
    assert first.total_interval.low == pytest.approx(second.total_interval.low, abs=0.0)
    assert first.total_interval.high == pytest.approx(second.total_interval.high, abs=0.0)


def test_two_reports_at_one_seed_agree(history: RunHistory) -> None:
    """The whole top-level report is reproducible at a declared seed."""
    first = rank_tests(history, resamples=120, seed=4242)
    second = rank_tests(history, resamples=120, seed=4242)
    assert first.truth_vs_rate.tau_b_interval.low == pytest.approx(
        second.truth_vs_rate.tau_b_interval.low, abs=0.0
    )
    assert first.truth_vs_rate.tau_b_interval.high == pytest.approx(
        second.truth_vs_rate.tau_b_interval.high, abs=0.0
    )
    assert first.capture_by_rate == pytest.approx(second.capture_by_rate, abs=0.0)


def test_point_statistics_do_not_depend_on_the_seed(history: RunHistory) -> None:
    """A point estimate is a function of the data alone; only intervals resample.

    If a point statistic moved with the seed it would be a property of the
    resampler and not of the history, which is the defect class an adjacent
    product of this mission exists to detect.
    """
    first = rank_tests(history, resamples=60, seed=1)
    second = rank_tests(history, resamples=60, seed=999)
    assert first.truth_vs_rate.tau_b == pytest.approx(second.truth_vs_rate.tau_b, abs=0.0)
    assert first.truth_vs_rate.rho == pytest.approx(second.truth_vs_rate.rho, abs=0.0)
    assert first.truth.total_seconds == pytest.approx(second.truth.total_seconds, abs=0.0)
    assert first.capture_by_rate == pytest.approx(second.capture_by_rate, abs=0.0)


def test_intervals_do_depend_on_the_seed(history: RunHistory) -> None:
    """A resampler that ignored its seed would pass every test above."""
    first = rank_tests(history, resamples=60, seed=1)
    second = rank_tests(history, resamples=60, seed=999)
    assert (
        first.truth.total_interval.low,
        first.truth.total_interval.high,
    ) != (
        second.truth.total_interval.low,
        second.truth.total_interval.high,
    )


def test_the_generator_declares_one_seed() -> None:
    """``data/generate_history.py`` declares the seed the suite expects."""
    source = (ROOT / "data" / "generate_history.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    seeds = {
        target.id: node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == "SEED"
    }
    assert seeds == {"SEED": 97044}


def test_the_history_is_the_only_source_of_durations(history: RunHistory) -> None:
    """Every duration in the product is an integer millisecond from the file.

    Asserted on the loaded arrays rather than by inspection: the dtype is
    integral, so no float duration can have entered from anywhere.
    """
    assert np.issubdtype(history.attempt_duration_ms.dtype, np.integer)
    assert int(history.attempt_duration_ms.min()) >= 1
    assert int(history.attempt_duration_ms.max()) == 1848115
