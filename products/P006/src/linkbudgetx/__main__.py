"""Module entry point: ``python -m linkbudgetx --config example.yaml``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
