"""Known-answer tests for the import graph against the bundled sample project.

The sample project's import structure is fixed and small enough to write out
by hand, which is what makes these known-answer tests rather than
self-consistency tests:

    sampleproj.vecops     imports nothing local
    sampleproj.stats      imports nothing local
    sampleproj.integrate  imports nothing local
    sampleproj.orbit      imports nothing local
    sampleproj.quat       imports sampleproj, sampleproj.vecops
    sampleproj.filters    imports sampleproj, sampleproj.stats
    sampleproj.kinematics imports sampleproj, sampleproj.quat, sampleproj.vecops
"""

from __future__ import annotations

from pathlib import Path

import pytest

from regressselect.importgraph import build_import_graph, module_name_for_path

EXPECTED_SOURCE_EDGES = {
    "sampleproj": set(),
    "sampleproj.vecops": set(),
    "sampleproj.stats": set(),
    "sampleproj.integrate": set(),
    "sampleproj.orbit": set(),
    "sampleproj.quat": {"sampleproj", "sampleproj.vecops"},
    "sampleproj.filters": {"sampleproj", "sampleproj.stats"},
    "sampleproj.kinematics": {"sampleproj", "sampleproj.quat", "sampleproj.vecops"},
}


def test_source_edges_are_exactly_the_hand_written_structure(graph):
    for module, expected in EXPECTED_SOURCE_EDGES.items():
        assert graph.imports_of(module) == expected, module


def test_seven_test_modules_are_identified(graph):
    assert graph.test_modules == {
        "tests.test_filters",
        "tests.test_integrate",
        "tests.test_kinematics",
        "tests.test_orbit",
        "tests.test_quat",
        "tests.test_stats",
        "tests.test_vecops",
    }


def test_hand_calculated_distances(graph):
    # tests/test_quat.py imports sampleproj.quat, which imports sampleproj.vecops:
    #   test_quat -> quat -> vecops = 2 hops.
    # tests/test_kinematics.py imports both kinematics and vecops directly, so
    # vecops is 1 hop, while quat is reached only through kinematics, so 2 hops.
    # tests/test_filters.py imports filters, which imports stats: 2 hops.
    assert graph.distance("tests.test_quat", "sampleproj.vecops") == 2
    assert graph.distance("tests.test_kinematics", "sampleproj.vecops") == 1
    assert graph.distance("tests.test_kinematics", "sampleproj.kinematics") == 1
    assert graph.distance("tests.test_kinematics", "sampleproj.quat") == 2
    assert graph.distance("tests.test_vecops", "sampleproj.vecops") == 1
    assert graph.distance("tests.test_filters", "sampleproj.filters") == 1
    assert graph.distance("tests.test_filters", "sampleproj.stats") == 2


def test_unreachable_pairs_return_none(graph):
    assert graph.distance("tests.test_filters", "sampleproj.vecops") is None
    assert graph.distance("tests.test_orbit", "sampleproj.quat") is None
    assert graph.distance("tests.test_stats", "sampleproj.integrate") is None


def test_distance_to_self_is_zero(graph):
    assert graph.distance("sampleproj.quat", "sampleproj.quat") == 0


def test_selection_known_answers(graph):
    assert graph.select_tests(["sampleproj.vecops"]) == {
        "tests.test_vecops",
        "tests.test_quat",
        "tests.test_kinematics",
    }
    assert graph.select_tests(["sampleproj.stats"]) == {
        "tests.test_stats",
        "tests.test_filters",
    }
    assert graph.select_tests(["sampleproj.orbit"]) == {"tests.test_orbit"}


def test_changing_the_package_init_selects_the_whole_suite(graph):
    # Every test module imports the package, so a change to sampleproj/__init__.py
    # selects everything. This is the "a dependency-based selector usually
    # selects almost everything" behaviour, measured rather than asserted.
    assert graph.select_tests(["sampleproj"]) == graph.test_modules


def test_selection_is_monotone_in_the_change_set(graph):
    one = graph.select_tests(["sampleproj.stats"])
    two = graph.select_tests(["sampleproj.stats", "sampleproj.orbit"])
    assert one <= two


def test_unknown_changed_module_is_an_error_not_an_empty_selection(graph):
    with pytest.raises(ValueError, match="not present in the graph"):
        graph.select_tests(["sampleproj.does_not_exist"])


def test_unknown_source_module_is_an_error(graph):
    with pytest.raises(ValueError, match="unknown source module"):
        graph.distance("nope", "sampleproj.vecops")


def test_min_distance_to_change_picks_the_nearest(graph):
    got = graph.min_distance_to_change(
        "tests.test_kinematics", ["sampleproj.vecops", "sampleproj.kinematics"]
    )
    assert got == 1
    assert graph.min_distance_to_change("tests.test_orbit", ["sampleproj.quat"]) is None


def test_module_name_for_path_handles_package_init(sample_project: Path):
    root = sample_project
    assert module_name_for_path(root / "sampleproj" / "__init__.py", root) == "sampleproj"
    assert module_name_for_path(root / "sampleproj" / "quat.py", root) == "sampleproj.quat"


def test_missing_root_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="not a directory"):
        build_import_graph(tmp_path / "nope")


def test_relative_imports_resolve(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "low.py").write_text("X = 1\n", encoding="utf-8")
    (pkg / "high.py").write_text("from .low import X\n", encoding="utf-8")
    sub = pkg / "deep"
    sub.mkdir()
    (sub / "__init__.py").write_text("", encoding="utf-8")
    (sub / "far.py").write_text("from ..low import X\n", encoding="utf-8")
    graph = build_import_graph(tmp_path)
    assert "pkg.low" in graph.imports_of("pkg.high")
    assert "pkg.low" in graph.imports_of("pkg.deep.far")


def test_pycache_is_not_parsed(tmp_path: Path):
    pkg = tmp_path / "pkg"
    (pkg / "__pycache__").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__pycache__" / "broken.py").write_text("def (:\n", encoding="utf-8")
    graph = build_import_graph(tmp_path)
    assert graph.modules == {"pkg"}


def test_conftest_is_not_a_selectable_test_module(tmp_path: Path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tmp_path / "mod.py").write_text("V = 1\n", encoding="utf-8")
    (tests_dir / "conftest.py").write_text("import mod\n", encoding="utf-8")
    (tests_dir / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")
    graph = build_import_graph(tmp_path)
    assert "tests.conftest" in graph.modules
    assert graph.test_modules == {"tests.test_a"}


def test_conftest_coupling_adds_the_edge_pytest_actually_imposes(tmp_path: Path):
    # tests/test_a.py imports nothing, so the import graph selects nothing for a
    # change to mod.py. pytest nevertheless imports tests/conftest.py before
    # collecting test_a, so test_a does depend on mod.
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tmp_path / "mod.py").write_text("V = 1\n", encoding="utf-8")
    (tests_dir / "conftest.py").write_text("import mod\n", encoding="utf-8")
    (tests_dir / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")
    graph = build_import_graph(tmp_path)
    assert graph.select_tests(["mod"]) == set()
    coupled = graph.with_conftest_coupling()
    assert coupled.select_tests(["mod"]) == {"tests.test_a"}
    assert coupled.distance("tests.test_a", "mod") == 2


def test_conftest_coupling_is_scoped_to_the_package_and_its_ancestors(tmp_path: Path):
    tests_dir = tmp_path / "tests"
    (tests_dir / "sub").mkdir(parents=True)
    (tmp_path / "shared.py").write_text("V = 1\n", encoding="utf-8")
    (tmp_path / "deep.py").write_text("V = 2\n", encoding="utf-8")
    (tests_dir / "conftest.py").write_text("import shared\n", encoding="utf-8")
    (tests_dir / "sub" / "conftest.py").write_text("import deep\n", encoding="utf-8")
    (tests_dir / "test_top.py").write_text("def test_t():\n    assert True\n", encoding="utf-8")
    (tests_dir / "sub" / "test_deep.py").write_text(
        "def test_d():\n    assert True\n", encoding="utf-8"
    )
    coupled = build_import_graph(tmp_path).with_conftest_coupling()
    # the top-level test sees only the top-level conftest
    assert coupled.select_tests(["deep"]) == {"tests.sub.test_deep"}
    # the nested test sees both conftests
    assert coupled.select_tests(["shared"]) == {"tests.test_top", "tests.sub.test_deep"}


def test_conftest_coupling_does_not_shrink_any_selection(graph):
    coupled = graph.with_conftest_coupling()
    for module in sorted(graph.modules - graph.test_modules):
        assert graph.select_tests([module]) <= coupled.select_tests([module]), module


def test_conftest_coupling_on_a_project_without_conftest_is_a_no_op(graph):
    coupled = graph.with_conftest_coupling()
    assert coupled.edges == graph.edges
    assert coupled.test_modules == graph.test_modules
