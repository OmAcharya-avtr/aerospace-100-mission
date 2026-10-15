"""Record the exact environment the other validation scripts ran in.

Dependency ranges in pyproject.toml are declared only from versions that were
installed and had the whole suite run against them. This script records one end
of that measurement; VALIDATION.md section 8 records both.
"""

from __future__ import annotations

import platform
import sys

from _support import Emitter


def main() -> int:
    """Print and record the environment. Returns a process exit code."""
    emit = Emitter(__file__)
    emit("hazardtrace validation: environment")
    emit()
    emit(f"  python            : {sys.version.split()[0]}")
    emit(f"  implementation    : {platform.python_implementation()}")
    emit(f"  platform          : {platform.platform()}")
    emit(f"  machine           : {platform.machine()}")
    emit()

    import hazardtrace

    emit(f"  hazardtrace       : {hazardtrace.__version__}")

    for name in ("yaml", "matplotlib", "pytest", "hypothesis"):
        try:
            module = __import__(name)
        except ImportError:  # pragma: no cover - every one is installed here
            emit(f"  {name:<18}: not installed")
            continue
        version = getattr(module, "__version__", "unknown")
        emit(f"  {name:<18}: {version}")

    emit()
    emit("  declared ranges (pyproject.toml):")
    emit("    pyyaml      >=6.0.1,<6.1")
    emit("    matplotlib  >=3.11.0,<3.12  (optional, figures only)")
    emit("    pytest      >=8.3.5,<9.2    (test extra)")
    emit("    hypothesis  >=6.100.0,<6.170 (test extra)")
    emit("    ruff        >=0.16.8,<0.17  (test extra)")
    emit()
    emit("  Two floors are set by measurement, not by preference:")
    emit("   - PyYAML 6.0 has no cp313 wheel and fails to build against Cython 3")
    emit("     on Python 3.13, so 6.0.1 is the lowest version measurable here.")
    emit("   - matplotlib 3.9.2 and 3.10.0 both fail this suite at COLLECTION")
    emit("     time: they call pyparsing's deprecated oneOf, pyparsing 3.3.3")
    emit("     warns, and [tool.pytest.ini_options] filterwarnings turns a")
    emit("     DeprecationWarning into an error. matplotlib 3.11.0 does not.")

    emit.check("hazardtrace version is 0.1.0", hazardtrace.__version__ == "0.1.0")
    emit.check(
        "python is 3.11 or newer, as pyproject declares",
        sys.version_info >= (3, 11),
        f"found {sys.version.split()[0]}",
    )
    return emit.finish()


if __name__ == "__main__":
    raise SystemExit(main())
