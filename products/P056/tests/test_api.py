"""The public surface: everything in ``__all__`` imports and is documented."""

from __future__ import annotations

import inspect

import calibaudit


def test_version_is_the_release_under_test():
    assert calibaudit.__version__ == "0.1.0"


def test_every_name_in_all_is_importable():
    missing = [name for name in calibaudit.__all__ if not hasattr(calibaudit, name)]
    assert missing == []


def test_all_is_sorted():
    assert list(calibaudit.__all__) == sorted(calibaudit.__all__)


def test_every_public_callable_has_a_docstring():
    undocumented = []
    for name in calibaudit.__all__:
        obj = getattr(calibaudit, name)
        if inspect.isfunction(obj) or inspect.isclass(obj):
            if not (obj.__doc__ or "").strip():
                undocumented.append(name)
    assert undocumented == []


def test_package_docstring_states_the_scope():
    doc = calibaudit.__doc__ or ""
    lowered = doc.lower()
    assert "not flight-qualified" in lowered
    assert "not certified" in lowered


def test_submodules_import_cleanly():
    import importlib

    for mod in (
        "binning",
        "decomposition",
        "ece",
        "plotting",
        "recalibration",
        "reliability",
        "scores",
        "synthetic",
    ):
        importlib.import_module(f"calibaudit.{mod}")


def test_main_module_builds_its_parser():
    from calibaudit.__main__ import build_parser

    parser = build_parser()
    assert parser.prog == "python -m calibaudit"


def test_no_print_in_library_modules():
    from pathlib import Path

    src = Path(calibaudit.__file__).parent
    offenders = []
    for path in sorted(src.glob("*.py")):
        if path.name == "__main__.py":
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("print(") or " print(" in stripped:
                offenders.append(f"{path.name}:{lineno}")
    assert offenders == []
