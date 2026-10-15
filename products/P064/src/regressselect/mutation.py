"""Seeded-fault (mutation-style) evaluation against a bundled sample project.

A seeded fault is a declared textual substitution in one source file of the
sample project, written as a classical mutation operator -- arithmetic
operator replacement, relational operator replacement, constant replacement,
statement deletion. Each fault is applied to a throwaway copy of the project,
the full suite is run with pytest, and the set of test nodes that failed is
recorded. That set is the ground truth against which a selection strategy is
scored, and it is obtained by execution, not by assertion.

Test counts and per-node outcomes are read from the junit XML that pytest
writes, never from its standard output: a suite that collects zero tests
prints success and exits 0.

Mutation analysis as a measure of test-suite fault detection follows
R. A. DeMillo, R. J. Lipton and F. G. Sayward, "Hints on Test Data Selection:
Help for the Practicing Programmer", Computer, 11(4), 34-41, April 1978. The
faults here are *seeded by hand* rather than generated exhaustively, so the
detection rates below are properties of this fault set and of this sample
project, and are not an estimate of a mutation score for any other project.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "SeededFault",
    "SuiteResult",
    "FaultMatrix",
    "load_faults",
    "materialise_fault",
    "run_suite",
    "run_mutation_evaluation",
]


@dataclass(frozen=True)
class SeededFault:
    """One declared textual mutation of one sample-project source file."""

    fault_id: str
    module: str
    relative_path: str
    old_text: str
    new_text: str
    operator: str
    description: str


@dataclass(frozen=True)
class SuiteResult:
    """Counts and failing nodes parsed from one junit XML document.

    ``tests``, ``failures``, ``errors`` and ``skipped`` are counts read from
    the ``testsuite`` element. ``testcase_count`` is an independent count of
    ``testcase`` elements and must agree with ``tests``; disagreement is
    reported by :meth:`cross_check`, not silently reconciled.
    """

    tests: int
    failures: int
    errors: int
    skipped: int
    testcase_count: int
    failing_nodes: tuple[str, ...]
    all_nodes: tuple[str, ...]
    returncode: int

    def cross_check(self) -> bool:
        """True when the testsuite ``tests`` attribute equals the element count."""
        return self.tests == self.testcase_count


@dataclass(frozen=True)
class FaultMatrix:
    """Which test nodes detect which seeded faults.

    ``detecting[fault_id]`` is the tuple of node ids that failed when that
    fault was applied. A fault with an empty tuple was detected by no test in
    the suite; those faults are kept in the matrix and counted in the
    full-suite upper bound, because dropping them is how a mutation score
    comes to read 100 per cent.
    """

    fault_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    detecting: dict[str, tuple[str, ...]]
    changed_modules: dict[str, tuple[str, ...]]

    @property
    def detected_fault_ids(self) -> tuple[str, ...]:
        """Faults that at least one test in the suite detects."""
        return tuple(f for f in self.fault_ids if self.detecting.get(f))

    def full_suite_detection(self) -> float:
        """Fraction of seeded faults the whole suite detects (dimensionless)."""
        if not self.fault_ids:
            raise ValueError("fault matrix is empty")
        return len(self.detected_fault_ids) / len(self.fault_ids)

    def to_json(self) -> str:
        """Serialise to the committed fixture form (stable key order)."""
        return (
            json.dumps(
                {
                    "schema": "regressselect-faultmatrix/1",
                    "fault_ids": list(self.fault_ids),
                    "node_ids": list(self.node_ids),
                    "detecting": {f: list(self.detecting.get(f, ())) for f in self.fault_ids},
                    "changed_modules": {
                        f: list(self.changed_modules.get(f, ())) for f in self.fault_ids
                    },
                },
                indent=1,
            )
            + "\n"
        )


def load_faults(path: str | Path) -> tuple[SeededFault, ...]:
    """Load seeded faults from the committed JSON fixture."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema") != "regressselect-faults/1":
        raise ValueError(f"unsupported fault schema {raw.get('schema')!r}")
    return tuple(
        SeededFault(
            fault_id=f["fault_id"],
            module=f["module"],
            relative_path=f["relative_path"],
            old_text=f["old_text"],
            new_text=f["new_text"],
            operator=f["operator"],
            description=f["description"],
        )
        for f in raw["faults"]
    )


def materialise_fault(project_root: str | Path, fault: SeededFault, dest: str | Path) -> Path:
    """Copy the project to ``dest`` with ``fault`` applied. Returns ``dest``.

    Raises :class:`ValueError` when ``fault.old_text`` does not occur exactly
    once in the target file, so a fault that silently stopped applying after a
    source edit is a hard error rather than a quietly undetected mutation.
    """
    src = Path(project_root).resolve()
    out = Path(dest)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(
        src, out, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    )
    target = out / fault.relative_path
    if not target.is_file():
        raise ValueError(f"fault {fault.fault_id} targets a missing file: {fault.relative_path}")
    text = target.read_text(encoding="utf-8")
    occurrences = text.count(fault.old_text)
    if occurrences != 1:
        raise ValueError(
            f"fault {fault.fault_id}: old_text occurs {occurrences} times in "
            f"{fault.relative_path}, expected exactly 1"
        )
    target.write_text(text.replace(fault.old_text, fault.new_text), encoding="utf-8")
    return out


def _parse_junit(path: Path) -> tuple[int, int, int, int, int, tuple[str, ...], tuple[str, ...]]:
    tree = ET.parse(path)
    root = tree.getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tests = failures = errors = skipped = 0
    for suite in suites:
        tests += int(suite.get("tests", 0))
        failures += int(suite.get("failures", 0))
        errors += int(suite.get("errors", 0))
        skipped += int(suite.get("skipped", 0))
    cases = list(root.iter("testcase"))
    failing: list[str] = []
    everything: list[str] = []
    for case in cases:
        file_attr = case.get("file")
        name = case.get("name", "")
        if file_attr:
            node = f"{Path(file_attr).as_posix()}::{name}"
        else:
            classname = case.get("classname", "").replace(".", "/")
            node = f"{classname}.py::{name}"
        everything.append(node)
        if case.find("failure") is not None or case.find("error") is not None:
            failing.append(node)
    return (
        tests,
        failures,
        errors,
        skipped,
        len(cases),
        tuple(sorted(failing)),
        tuple(sorted(everything)),
    )


def run_suite(
    project_root: str | Path,
    node_ids: Sequence[str] | None = None,
    junit_path: str | Path | None = None,
    timeout_s: float = 120.0,
) -> SuiteResult:
    """Run pytest in ``project_root`` and parse the junit XML it writes.

    ``node_ids`` restricts the run to those pytest node ids; ``None`` runs the
    whole ``tests`` directory. ``timeout_s`` is a wall-clock limit in seconds.
    """
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise ValueError(f"project root is not a directory: {root}")
    cleanup = junit_path is None
    xml_path = Path(junit_path) if junit_path else root / "_regressselect_junit.xml"
    targets = list(node_ids) if node_ids else ["tests"]
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        "-q",
        "-p",
        "no:cacheprovider",
        f"--junitxml={xml_path}",
    ]
    proc = subprocess.run(
        cmd, cwd=root, capture_output=True, text=True, timeout=timeout_s, check=False
    )
    if not xml_path.is_file():
        raise RuntimeError(
            f"pytest wrote no junit XML in {root}; returncode {proc.returncode}\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    tests, failures, errors, skipped, case_count, failing, everything = _parse_junit(xml_path)
    if cleanup:
        xml_path.unlink()
    return SuiteResult(
        tests=tests,
        failures=failures,
        errors=errors,
        skipped=skipped,
        testcase_count=case_count,
        failing_nodes=failing,
        all_nodes=everything,
        returncode=proc.returncode,
    )


def run_mutation_evaluation(
    project_root: str | Path,
    faults: Sequence[SeededFault],
    node_ids: Sequence[str] | None = None,
    workdir: str | Path | None = None,
    timeout_s: float = 120.0,
) -> FaultMatrix:
    """Apply each fault in turn, run the full suite, and record the failures.

    Compute cost: one pytest process per fault. On the two-core build machine
    the bundled sample project costs roughly one second per fault; a project
    whose suite takes minutes costs that per fault and this harness is not
    the right shape for it.
    """
    faults = tuple(faults)
    if not faults:
        raise ValueError("no faults to evaluate")
    root = Path(project_root).resolve()
    detecting: dict[str, tuple[str, ...]] = {}
    changed: dict[str, tuple[str, ...]] = {}
    base = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="regressselect_mut_"))
    base.mkdir(parents=True, exist_ok=True)
    try:
        for fault in faults:
            work = base / fault.fault_id
            materialise_fault(root, fault, work)
            result = run_suite(work, node_ids=node_ids, timeout_s=timeout_s)
            if not result.cross_check():
                raise RuntimeError(
                    f"junit cross-check failed for {fault.fault_id}: "
                    f"tests={result.tests} testcase elements={result.testcase_count}"
                )
            detecting[fault.fault_id] = result.failing_nodes
            changed[fault.fault_id] = (fault.module,)
            shutil.rmtree(work, ignore_errors=True)
    finally:
        if workdir is None:
            shutil.rmtree(base, ignore_errors=True)
    universe = tuple(node_ids) if node_ids else ()
    if not universe:
        clean = run_suite(root, timeout_s=timeout_s)
        if clean.failing_nodes:
            raise RuntimeError(
                "the unmutated sample project does not pass; these nodes failed: "
                f"{list(clean.failing_nodes)}"
            )
        universe = clean.all_nodes
    return FaultMatrix(
        fault_ids=tuple(f.fault_id for f in faults),
        node_ids=tuple(universe),
        detecting=detecting,
        changed_modules=changed,
    )
