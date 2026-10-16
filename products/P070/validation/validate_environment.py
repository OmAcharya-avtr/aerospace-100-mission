"""Record the exact environment the other validation scripts ran in.

Dependency ranges in pyproject.toml are declared only from versions that were
installed and had the whole suite run against them. This script records one
end of that measurement; VALIDATION.md section 8 records both.
"""

from __future__ import annotations

import platform
import sys

from _support import Emitter


def main() -> int:
    """Print and record the environment. Returns a process exit code."""
    emit = Emitter(__file__)
    emit("watchdogspec validation: environment")
    emit()
    emit(f"  python            : {sys.version.split()[0]}")
    emit(f"  implementation    : {platform.python_implementation()}")
    emit(f"  platform          : {platform.platform()}")
    emit(f"  machine           : {platform.machine()}")
    emit()

    import watchdogspec

    emit(f"  watchdogspec      : {watchdogspec.__version__}")

    for name in ("numpy", "yaml", "matplotlib", "pytest", "hypothesis"):
        try:
            module = __import__(name)
        except ImportError:  # pragma: no cover - every one is installed here
            emit(f"  {name:<18}: not installed")
            continue
        emit(f"  {name:<18}: {getattr(module, '__version__', 'unknown')}")

    emit()
    emit("  declared ranges (pyproject.toml):")
    emit("    numpy       >=2.1.0,<2.6")
    emit("    pyyaml      >=6.0.1,<6.1")
    emit("    matplotlib  >=3.10.7,<3.12  (plot extra, figures only)")
    emit("    pytest      >=8.3.5,<9.2    (test extra)")
    emit("    hypothesis  >=6.100.0,<6.171 (test extra)")
    emit("    ruff        >=0.16.0,<0.18  (test extra)")
    emit()
    emit("  Two floors are set by measurement, not by preference:")
    emit("   - matplotlib 3.10.0 fails tests/test_diagram.py here: it calls")
    emit("     pyparsing's deprecated oneOf, pyparsing 3.3.3 warns, and")
    emit("     [tool.pytest.ini_options] filterwarnings turns a")
    emit("     DeprecationWarning into an error. 3.10.7 does not; 3.10.7 is")
    emit("     therefore the lowest matplotlib this suite was measured on.")
    emit("   - PyYAML 6.0 has no cp313 wheel and its sdist fails to build")
    emit("     against Cython 3 on Python 3.13 (build_ext has no attribute")
    emit("     cython_sources), so 6.0.1 is the lowest version measurable here.")
    emit()
    emit("  simpy is NOT a dependency of this package. The discrete-event")
    emit("  simulation in watchdogspec.simulate is written here, in about 150")
    emit("  lines, because a cross-check of a closed form should not share a")
    emit("  scheduler with anything. simpy is named in the README alternatives")
    emit("  table as the thing to use for a simulation of any size.")

    emit()
    emit.check("watchdogspec version is 0.1.0", watchdogspec.__version__ == "0.1.0")
    emit.check(
        "python is 3.11 or newer, as pyproject declares",
        sys.version_info >= (3, 11),
        f"found {sys.version.split()[0]}",
    )
    return emit.finish()


if __name__ == "__main__":
    raise SystemExit(main())
