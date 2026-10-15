"""Import-graph reachability: Baseline 1, the safe dependency-based selector.

The graph is built by parsing every ``.py`` file under a project root with
:mod:`ast` and recording the module-level and function-level ``import`` and
``from ... import`` statements that resolve to a module inside the same root.
Selection is then "every test module that can reach a changed module through
one or more import edges".

What the word *safe* means here, precisely: the selection contains every test
whose failure can be explained by the *declared import dependencies* of the
project. It does not contain tests that depend on a changed module through a
channel the import graph cannot see -- a data file, a ``conftest.py`` fixture
that constructs state from a changed module by string name, a subprocess, a
monkeypatched attribute, or a plugin loaded by entry point. Those channels
exist in real repositories and the measured safety loss in
``validation/validate_benchmark_output.txt`` is not zero for that reason.

Reference for the safe-selection notion: G. Rothermel and M. J. Harrold,
"Analyzing Regression Test Selection Techniques", IEEE Transactions on
Software Engineering, 22(8), August 1996.
"""

from __future__ import annotations

import ast
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ImportGraph", "build_import_graph", "module_name_for_path"]

CONFTEST_BASENAME = "conftest"


def module_name_for_path(path: Path, root: Path) -> str:
    """Dotted module name of ``path`` relative to ``root``.

    Units: dimensionless identifier. ``root/pkg/mod.py`` becomes ``pkg.mod``
    and ``root/pkg/__init__.py`` becomes ``pkg``.
    """
    rel = path.resolve().relative_to(root.resolve())
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts)


def _resolve_relative(module: str | None, level: int, importer: str) -> str:
    """Resolve a ``from ... import x`` target to an absolute module name.

    ``level`` is the ``ast.ImportFrom.level`` dot count: 0 is an absolute
    import and the module name is returned unchanged; 1 means "the package
    containing the importer", so one component is dropped from the importer's
    dotted name, and so on.
    """
    if level == 0:
        return module or ""
    base = importer.split(".")
    drop = min(level, len(base))
    prefix = base[: len(base) - drop]
    tail = module.split(".") if module else []
    return ".".join([*prefix, *tail])


@dataclass
class ImportGraph:
    """Directed import graph over the modules of one project root.

    ``edges[a]`` is the set of modules that ``a`` imports. ``modules`` is every
    module discovered under the root; ``test_modules`` is the subset whose path
    lives under a directory named ``tests`` or whose basename starts with
    ``test_``, excluding ``conftest`` modules, which hold fixtures rather than
    tests and are therefore not selectable units.

    A plain import graph does not see pytest's ``conftest.py`` mechanism: every
    test module in a directory implicitly depends on the ``conftest`` modules in
    that directory and its ancestors, without importing them. Selecting from the
    import graph alone therefore misses tests that a changed module reaches only
    through a fixture, which makes a nominally safe selector unsafe. Use
    :meth:`with_conftest_coupling` to add those edges explicitly;
    ``validation/validate_self_selection_output.txt`` measures how much the
    selection grows when they are added.
    """

    root: Path
    edges: dict[str, set[str]] = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)
    test_modules: set[str] = field(default_factory=set)

    @property
    def modules(self) -> set[str]:
        """Every module name in the graph (dimensionless)."""
        return set(self.edges)

    def imports_of(self, module: str) -> set[str]:
        """Direct in-project imports of ``module``."""
        return set(self.edges.get(module, set()))

    def distance(self, source: str, target: str) -> int | None:
        """Shortest import-edge distance from ``source`` to ``target``.

        Returns 0 when ``source == target``, the hop count when a path exists,
        and ``None`` when ``target`` is not reachable. Unit: hops (integer).
        """
        if source not in self.edges:
            raise ValueError(f"unknown source module: {source!r}")
        if source == target:
            return 0
        seen = {source}
        queue: deque[tuple[str, int]] = deque([(source, 0)])
        while queue:
            node, dist = queue.popleft()
            for nxt in self.edges.get(node, ()):
                if nxt in seen:
                    continue
                if nxt == target:
                    return dist + 1
                seen.add(nxt)
                queue.append((nxt, dist + 1))
        return None

    def reachable_targets(self, source: str) -> dict[str, int]:
        """Every module reachable from ``source``, mapped to its hop distance."""
        if source not in self.edges:
            raise ValueError(f"unknown source module: {source!r}")
        out = {source: 0}
        queue: deque[str] = deque([source])
        while queue:
            node = queue.popleft()
            for nxt in self.edges.get(node, ()):
                if nxt not in out:
                    out[nxt] = out[node] + 1
                    queue.append(nxt)
        return out

    def select_tests(self, changed_modules: list[str] | set[str]) -> set[str]:
        """Test modules that reach at least one changed module.

        This is the Baseline 1 selection. A changed module that is not in the
        graph is treated as reaching nothing, which is the conservative-free
        direction, so :meth:`select_tests` raises instead of silently ignoring
        it.
        """
        changed = set(changed_modules)
        unknown = sorted(c for c in changed if c not in self.edges)
        if unknown:
            raise ValueError(f"changed modules not present in the graph: {unknown}")
        selected = set()
        for test in self.test_modules:
            targets = self.reachable_targets(test)
            if changed & set(targets):
                selected.add(test)
        return selected

    def with_conftest_coupling(self) -> ImportGraph:
        """Copy of this graph with pytest's implicit conftest edges added.

        Every test module gains an edge to each ``conftest`` module in its own
        package or an ancestor package, because pytest collects fixtures from
        exactly those files. The added edges are dependency edges in the same
        sense as an import, so distances and selections are computed the same
        way afterwards; they are simply not visible in the source text.
        """
        conftests = [m for m in self.edges if m.rsplit(".", 1)[-1] == CONFTEST_BASENAME]
        edges = {k: set(v) for k, v in self.edges.items()}
        for test in self.test_modules:
            package = test.rsplit(".", 1)[0] if "." in test else ""
            for conftest in conftests:
                owner = conftest.rsplit(".", 1)[0] if "." in conftest else ""
                same_or_ancestor = (
                    owner == "" or package == owner or package.startswith(owner + ".")
                )
                if same_or_ancestor and conftest != test:
                    edges[test].add(conftest)
        return ImportGraph(
            root=self.root,
            edges=edges,
            paths=dict(self.paths),
            test_modules=set(self.test_modules),
        )

    def min_distance_to_change(
        self, test: str, changed_modules: list[str] | set[str]
    ) -> int | None:
        """Minimum hop distance from ``test`` to any changed module, or ``None``."""
        targets = self.reachable_targets(test)
        hits = [targets[c] for c in changed_modules if c in targets]
        return min(hits) if hits else None


def _collect_imports(tree: ast.Module, importer: str) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_relative(node.module, node.level, importer)
            if resolved:
                out.add(resolved)
            for alias in node.names:
                # "from pkg import mod" where mod is itself a module.
                out.add(f"{resolved}.{alias.name}" if resolved else alias.name)
    return out


def build_import_graph(
    root: str | Path, test_dir_names: tuple[str, ...] = ("tests",)
) -> ImportGraph:
    """Parse every ``.py`` file under ``root`` and build the import graph.

    ``root`` must be the directory that the project's top-level packages sit
    in, because module names are computed relative to it. Files that fail to
    parse raise :class:`SyntaxError`; this is deliberate, because silently
    dropping a file would silently shrink the selection and a selector that
    under-selects without saying so is the failure mode this product exists to
    measure.
    """
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ValueError(f"project root is not a directory: {root_path}")
    graph = ImportGraph(root=root_path)
    raw: dict[str, set[str]] = {}
    for path in sorted(root_path.rglob("*.py")):
        if any(part in {"__pycache__", ".venv", "venv", "build", "dist"} for part in path.parts):
            continue
        name = module_name_for_path(path, root_path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        raw[name] = _collect_imports(tree, name)
        graph.paths[name] = path
        rel_parts = path.resolve().relative_to(root_path).parts
        is_test_path = (
            any(p in test_dir_names for p in rel_parts[:-1])
            or rel_parts[-1].startswith("test_")
        )
        if is_test_path and name.rsplit(".", 1)[-1] != CONFTEST_BASENAME:
            graph.test_modules.add(name)
    known = set(raw)
    for name, imported in raw.items():
        graph.edges[name] = {m for m in imported if m in known and m != name}
    return graph
