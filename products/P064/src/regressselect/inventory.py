"""Test inventory: the set of selectable test node identifiers in a project.

The unit of selection in this package is the pytest *node*, written
``<path>::<function>``, because that is the unit a per-test outcome record
carries. Import reachability is resolved at *module* granularity, so every
node in one test file inherits that file's graph distance. Collection is done
by parsing with :mod:`ast` rather than by running pytest, so an inventory is
reproducible without executing the project under test; the cost is that
parametrised and dynamically generated tests are reported as their single
defining function, not as one node per parameter.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

__all__ = ["TestInventory", "collect_tests", "module_of_node"]


def module_of_node(node_id: str) -> str:
    """Dotted module name owning a pytest node id.

    ``tests/test_vecops.py::test_norm`` becomes ``tests.test_vecops``.
    """
    if "::" not in node_id:
        raise ValueError(f"not a pytest node id: {node_id!r}")
    path = node_id.split("::", 1)[0]
    if not path.endswith(".py"):
        raise ValueError(f"node id path is not a python file: {node_id!r}")
    return path[: -len(".py")].replace("/", ".").replace("\\", ".")


@dataclass(frozen=True)
class TestInventory:
    """Immutable list of pytest node ids and the test modules that own them."""

    node_ids: tuple[str, ...]
    modules: tuple[str, ...]

    def nodes_in(self, module: str) -> tuple[str, ...]:
        """Node ids belonging to one test module."""
        return tuple(n for n in self.node_ids if module_of_node(n) == module)

    def __len__(self) -> int:
        return len(self.node_ids)


def collect_tests(root: str | Path, test_dir: str = "tests") -> TestInventory:
    """Collect ``test_*`` functions under ``root/test_dir`` by parsing the files.

    Node ids are returned with POSIX separators relative to ``root``, sorted,
    so the inventory is stable across platforms and across runs.
    """
    root_path = Path(root).resolve()
    test_root = root_path / test_dir
    if not test_root.is_dir():
        raise ValueError(f"test directory not found: {test_root}")
    nodes: list[str] = []
    modules: set[str] = set()
    for path in sorted(test_root.rglob("test_*.py")):
        rel = path.relative_to(root_path).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for item in tree.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and item.name.startswith(
                "test_"
            ):
                nodes.append(f"{rel}::{item.name}")
                modules.add(module_of_node(f"{rel}::{item.name}"))
    if not nodes:
        raise ValueError(f"no test functions found under {test_root}")
    return TestInventory(node_ids=tuple(sorted(nodes)), modules=tuple(sorted(modules)))
