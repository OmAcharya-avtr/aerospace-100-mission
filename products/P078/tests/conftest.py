"""Shared fixtures. Every path is anchored on ``__file__``, never on the cwd.

The suite must pass when pytest is invoked from the product root, from
``tests/`` or from anywhere else, so the case directory is derived from this
file's location. Four products in Batch 07 of this mission failed a release
gate on exactly this, and the fix is one expression.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cmdguard.loader import load_case
from cmdguard.model import (
    Case,
    CommandDictionary,
    CommandSequence,
    CommandSpec,
    InitialState,
    ModeMachine,
    ResourceLimits,
    SequenceStep,
    Verity,
)

ROOT = Path(__file__).resolve().parent.parent
CASE_DIR = ROOT / "examples" / "cases"

#: Case files that must load. ``missing_declarations`` is excluded on purpose.
LOADABLE_CASES = (
    "nominal_science_pass",
    "late_payload_enable",
    "irreversible_before_verify",
    "irreversible_after_verify",
    "resource_and_exclusion",
)

#: Cases that must report no violation at all. The negative control.
CLEAN_CASES = ("nominal_science_pass", "irreversible_after_verify")


@pytest.fixture(scope="session")
def case_dir() -> Path:
    """Directory holding the committed case files."""
    return CASE_DIR


@pytest.fixture(scope="session")
def cases() -> dict[str, Case]:
    """Every loadable committed case, keyed by name."""
    return {name: load_case(CASE_DIR / f"{name}.yaml") for name in LOADABLE_CASES}


def tiny_case(
    steps: list[tuple[float, str]],
    commands: list[CommandSpec] | None = None,
    predicates: dict[str, Verity] | None = None,
    mode: str = "NOMINAL",
    limits: dict[str, float] | None = None,
    modes: tuple[str, ...] = ("SAFE", "NOMINAL", "SCIENCE"),
    transitions: dict[str, tuple[str, ...]] | None = None,
    name: str = "tiny",
) -> Case:
    """Build a minimal case inline, for hand-calculated unit tests.

    Keeping the builder here rather than in the package means a test can
    declare exactly the two or three commands its arithmetic needs, and the
    expected answer in the test body is checkable by hand.
    """
    commands = commands or [
        CommandSpec(identifier="SET_P", effects={"p": Verity.TRUE}),
        CommandSpec(identifier="NEED_P", preconditions={"p": Verity.TRUE}),
    ]
    return Case(
        name=name,
        dictionary=CommandDictionary(
            commands={spec.identifier: spec for spec in commands},
            resource_limits=ResourceLimits(limits or {}),
        ),
        machine=ModeMachine(
            modes=modes,
            transitions=transitions
            or {"SAFE": ("NOMINAL",), "NOMINAL": ("SAFE", "SCIENCE"), "SCIENCE": ("NOMINAL",)},
        ),
        initial=InitialState(mode=mode, predicates=predicates or {}),
        sequence=CommandSequence(
            tuple(SequenceStep(t_s=float(t), command=c) for t, c in steps)
        ),
    )
