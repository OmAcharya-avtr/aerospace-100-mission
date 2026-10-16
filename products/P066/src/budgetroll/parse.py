"""Strict YAML reader for a budget declaration.

Strict means: an unknown key is rejected rather than ignored, a duplicate key
in one mapping is rejected rather than silently collapsed, a dangling
correlation reference is rejected at load time, and every rejection names the
offending key path. A declaration that silently drops the line you meant to
add is worse than one that will not load.

Schema
------
::

    budget:
      id: <str>
      name: <str>
      unit: <str>              # kg, W, Mbit, dB - not interpreted
      quantity: <str>          # mass, power, data, link
      policy: <str>            # key into 'policies'
      k_sigma: <float>         # optional, default 1.0
      root: <node>

    node:
      id: <str>
      name: <str>
      combination: linear | rss | correlated
      allocation: <float>                  # optional
      independence_rationale: <str>        # optional
      children: [<node> | <item>, ...]

    item:
      id: <str>
      name: <str>
      value: <float>                       # budget unit
      maturity: <str>                      # key into the policy, optional if
      contingency: <float>                 # given; dimensionless, optional
      uncertainty: <float>                 # 1 sigma, budget unit, default 0
      common_driver: <str>                 # optional tag

    policies:
      <name>:
        description: <str>
        source: <str>
        levels: {<maturity>: <float>, ...}

    correlations:
      - items: [<leaf id>, <leaf id>]
        rho: <float>
        rationale: <str>
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from .errors import DeclarationError
from .model import Budget, Correlation, Item, MarginPolicy, Node

BUDGET_KEYS = {"id", "name", "unit", "quantity", "policy", "root", "k_sigma"}
NODE_KEYS = {"id", "name", "combination", "children", "allocation", "independence_rationale"}
ITEM_KEYS = {"id", "name", "value", "maturity", "contingency", "uncertainty", "common_driver"}
POLICY_KEYS = {"description", "source", "levels"}
CORRELATION_KEYS = {"items", "rho", "rationale"}
TOP_KEYS = {"budget", "policies", "correlations"}


class _StrictLoader(yaml.SafeLoader):
    """A SafeLoader that rejects duplicate keys instead of keeping the last."""


def _no_duplicates(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise DeclarationError(
                f"duplicate key {key!r} in one mapping at line "
                f"{key_node.start_mark.line + 1}; a duplicate silently discards the "
                "first value and is rejected here"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


def _require_mapping(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        raise DeclarationError(f"{path}: expected a mapping, got {type(value).__name__}")
    return value


def _reject_unknown(mapping: dict, allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise DeclarationError(
            f"{path}: unknown key(s) {', '.join(repr(k) for k in unknown)}; allowed keys "
            f"are {', '.join(sorted(allowed))}"
        )


def _require(mapping: dict, key: str, path: str) -> Any:
    if key not in mapping:
        raise DeclarationError(f"{path}: required key {key!r} is missing")
    return mapping[key]


def _as_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeclarationError(f"{path}: expected a non-empty string, got {value!r}")
    return value


def _as_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeclarationError(f"{path}: expected a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise DeclarationError(f"{path}: expected a finite number, got {value!r}")
    return result


def _parse_policy(name: str, raw: Any) -> MarginPolicy:
    path = f"policies.{name}"
    mapping = _require_mapping(raw, path)
    _reject_unknown(mapping, POLICY_KEYS, path)
    levels_raw = _require_mapping(_require(mapping, "levels", path), f"{path}.levels")
    levels = {
        _as_str(level, f"{path}.levels"): _as_float(frac, f"{path}.levels.{level}")
        for level, frac in levels_raw.items()
    }
    return MarginPolicy(
        name=name,
        description=_as_str(_require(mapping, "description", path), f"{path}.description"),
        source=_as_str(_require(mapping, "source", path), f"{path}.source"),
        levels=levels,
    )


def _parse_item(mapping: dict, path: str) -> Item:
    _reject_unknown(mapping, ITEM_KEYS, path)
    ident = _as_str(_require(mapping, "id", path), f"{path}.id")
    maturity = mapping.get("maturity")
    if maturity is not None:
        maturity = _as_str(maturity, f"{path}.maturity")
    contingency = mapping.get("contingency")
    if contingency is not None:
        contingency = _as_float(contingency, f"{path}.contingency")
    driver = mapping.get("common_driver")
    if driver is not None:
        driver = _as_str(driver, f"{path}.common_driver")
    return Item(
        id=ident,
        name=_as_str(_require(mapping, "name", path), f"{path}.name"),
        value=_as_float(_require(mapping, "value", path), f"{path}.value"),
        maturity=maturity,
        contingency=contingency,
        uncertainty=_as_float(mapping.get("uncertainty", 0.0), f"{path}.uncertainty"),
        common_driver=driver,
    )


def _parse_node(mapping: dict, path: str) -> Item | Node:
    if "children" not in mapping:
        return _parse_item(mapping, path)
    _reject_unknown(mapping, NODE_KEYS, path)
    ident = _as_str(_require(mapping, "id", path), f"{path}.id")
    children_raw = _require(mapping, "children", path)
    if not isinstance(children_raw, list) or not children_raw:
        raise DeclarationError(f"{path}.children: expected a non-empty list")
    children = tuple(
        _parse_node(_require_mapping(child, f"{path}.children[{i}]"), f"{path}.children[{i}]")
        for i, child in enumerate(children_raw)
    )
    allocation = mapping.get("allocation")
    if allocation is not None:
        allocation = _as_float(allocation, f"{path}.allocation")
    return Node(
        id=ident,
        name=_as_str(_require(mapping, "name", path), f"{path}.name"),
        combination=_as_str(_require(mapping, "combination", path), f"{path}.combination"),
        children=children,
        allocation=allocation,
        independence_rationale=str(mapping.get("independence_rationale", "")),
    )


def _parse_correlation(raw: Any, index: int) -> Correlation:
    path = f"correlations[{index}]"
    mapping = _require_mapping(raw, path)
    _reject_unknown(mapping, CORRELATION_KEYS, path)
    items = _require(mapping, "items", path)
    if not isinstance(items, list) or len(items) != 2:
        raise DeclarationError(
            f"{path}.items: expected a list of exactly two leaf identifiers, got {items!r}"
        )
    return Correlation(
        a=_as_str(items[0], f"{path}.items[0]"),
        b=_as_str(items[1], f"{path}.items[1]"),
        rho=_as_float(_require(mapping, "rho", path), f"{path}.rho"),
        rationale=_as_str(_require(mapping, "rationale", path), f"{path}.rationale"),
    )


def loads(text: str) -> Budget:
    """Parse a budget declaration from YAML text.

    Raises
    ------
    DeclarationError
        On any malformed, unknown, duplicate or dangling construct. The
        message names the YAML key path.
    """
    try:
        document = yaml.load(text, Loader=_StrictLoader)
    except DeclarationError:
        raise
    except yaml.YAMLError as exc:
        raise DeclarationError(f"the document is not valid YAML: {exc}") from exc
    if document is None:
        raise DeclarationError("the document is empty")
    top = _require_mapping(document, "<document>")
    _reject_unknown(top, TOP_KEYS, "<document>")

    policies_raw = _require_mapping(_require(top, "policies", "<document>"), "policies")
    if not policies_raw:
        raise DeclarationError("policies: at least one margin policy must be declared")
    policies = {name: _parse_policy(str(name), raw) for name, raw in policies_raw.items()}

    budget_raw = _require_mapping(_require(top, "budget", "<document>"), "budget")
    _reject_unknown(budget_raw, BUDGET_KEYS, "budget")
    policy_name = _as_str(_require(budget_raw, "policy", "budget"), "budget.policy")
    if policy_name not in policies:
        known = ", ".join(sorted(policies))
        raise DeclarationError(
            f"budget.policy references {policy_name!r}, which is not declared; "
            f"declared policies are: {known}"
        )
    root = _parse_node(
        _require_mapping(_require(budget_raw, "root", "budget"), "budget.root"), "budget.root"
    )
    if not isinstance(root, Node):
        raise DeclarationError("budget.root: the root must be a node with children, not an item")

    correlations = tuple(
        _parse_correlation(raw, i) for i, raw in enumerate(top.get("correlations") or [])
    )

    budget = Budget(
        id=_as_str(_require(budget_raw, "id", "budget"), "budget.id"),
        name=_as_str(_require(budget_raw, "name", "budget"), "budget.name"),
        unit=_as_str(_require(budget_raw, "unit", "budget"), "budget.unit"),
        quantity=_as_str(_require(budget_raw, "quantity", "budget"), "budget.quantity"),
        policy=policies[policy_name],
        root=root,
        correlations=correlations,
        k_sigma=_as_float(budget_raw.get("k_sigma", 1.0), "budget.k_sigma"),
    )
    for item in _all_items(budget.root):
        if item.maturity is not None:
            budget.policy.contingency_for(item.maturity)
    return budget


def _all_items(node: Item | Node) -> list[Item]:
    if isinstance(node, Item):
        return [node]
    out: list[Item] = []
    for child in node.children:
        out.extend(_all_items(child))
    return out


def load(path: str | Path) -> Budget:
    """Parse a budget declaration from a file.

    Raises
    ------
    DeclarationError
        If the file does not exist, is a directory, or is not valid UTF-8,
        as well as for every schema rejection raised by :func:`loads`.
    """
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DeclarationError(f"no such budget declaration: {file_path}") from exc
    except IsADirectoryError as exc:
        raise DeclarationError(f"{file_path} is a directory, not a budget declaration") from exc
    except UnicodeDecodeError as exc:
        raise DeclarationError(f"{file_path} is not valid UTF-8: {exc}") from exc
    return loads(text)
