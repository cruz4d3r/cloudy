# Autor: Sergio Martinez de Unlockers Cloud
# URL: https://1lockers.net
"""Runtime detection for Mac bot vs Cloudy Edge."""

from __future__ import annotations

import os


def is_edge_runtime() -> bool:
    """True when code runs on Cloudy Edge (VPS), not the Mac webhook."""
    return os.environ.get("CLOUDY_EDGE_RUNTIME", "").strip() == "1"
