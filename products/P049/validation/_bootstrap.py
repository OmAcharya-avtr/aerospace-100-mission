"""Put the sibling ``src`` directory on ``sys.path``.

Importing this module makes the validation and example scripts runnable from
their own directory with no environment variables set, while remaining correct
when the runner has already exported ``PYTHONPATH=src``. No absolute path is
written into any file; the path is derived from ``__file__`` at run time.
"""

from __future__ import annotations

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
