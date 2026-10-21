"""Shared fixtures. Paths are anchored on ``__file__``, never on the caller's cwd."""

from __future__ import annotations

from pathlib import Path

import pytest

from downlinkpick.cases import Case, load_case
from downlinkpick.contacts import ContactPlan, ContactWindow, constant_rate_profile
from downlinkpick.dataset import GeneratedInstance, generate_instance, generate_training_table
from downlinkpick.inventory import DataProduct, Inventory
from downlinkpick.selection import SelectionProblem, sizes_from_ratios

ROOT = Path(__file__).resolve().parent.parent
CASE_DIR = ROOT / "examples" / "cases"


@pytest.fixture
def tiny_inventory() -> Inventory:
    """Three products, hand-written, with round numbers for hand calculation.

    ``cal`` is 1 MB at ratio 0.5, so 4e6 bits; ``img`` is 2 MB at ratio 0.25, so
    4e6 bits, and depends on ``cal``; ``bulk`` is 3 MB at ratio 0.9, so
    2.16e7 bits.
    """
    return Inventory(
        [
            DataProduct("cal", 1.0e6, 0.0, 1000.0, declared_value=5.0, compressibility=0.5),
            DataProduct(
                "img",
                2.0e6,
                0.0,
                500.0,
                declared_value=8.0,
                compressibility=0.25,
                depends_on=("cal",),
            ),
            DataProduct("bulk", 3.0e6, 0.0, 10000.0, declared_value=20.0, compressibility=0.9),
        ]
    )


@pytest.fixture
def tiny_plan() -> ContactPlan:
    """Two flat windows: 2e7 bits each, reliabilities 0.8 and 1.0."""
    return ContactPlan(
        [
            ContactWindow(
                "w1",
                100.0,
                200.0,
                constant_rate_profile(100.0, 200.0, 2.0e5),
                reliability=0.8,
            ),
            ContactWindow(
                "w2",
                600.0,
                700.0,
                constant_rate_profile(600.0, 700.0, 2.0e5),
                reliability=1.0,
            ),
        ]
    )


@pytest.fixture
def tiny_problem(tiny_inventory: Inventory, tiny_plan: ContactPlan) -> SelectionProblem:
    """The hand-calculable problem used by the known-answer tests."""
    return SelectionProblem.from_declared(tiny_inventory, tiny_plan)


@pytest.fixture
def instance() -> GeneratedInstance:
    """One generated instance, seed 20261021, 24 products over 3 windows."""
    return generate_instance(20261021, n_products=24, n_windows=3)


@pytest.fixture
def instance_problem(instance: GeneratedInstance) -> SelectionProblem:
    """The generated instance as a problem on declared values and true ratios."""
    values = {
        name: float(instance.inventory[name].declared_value or 0.0)
        for name in instance.inventory.names
    }
    return SelectionProblem(
        inventory=instance.inventory,
        plan=instance.plan,
        values=values,
        sizes_bits=sizes_from_ratios(instance.inventory, instance.true_ratios),
    )


@pytest.fixture
def reference_case() -> Case:
    """The shipped ``three_pass_declared`` case."""
    return load_case(CASE_DIR / "three_pass_declared.yaml")


@pytest.fixture
def small_case() -> Case:
    """The shipped ``single_short_pass`` case, small enough to enumerate."""
    return load_case(CASE_DIR / "single_short_pass.yaml")


@pytest.fixture
def missing_case() -> Case:
    """The shipped ``missing_declarations`` case."""
    return load_case(CASE_DIR / "missing_declarations.yaml")


@pytest.fixture(scope="session")
def training_table():
    """A 1200-row training table, seed 4242. Session-scoped: it is not cheap."""
    return generate_training_table(1200, 4242)
