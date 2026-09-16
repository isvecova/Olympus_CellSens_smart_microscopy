"""No command-line interface is exposed for this package.

Use the Python workflow API or one of the repository UI/script entry points.
"""

from __future__ import annotations


def main() -> int:
    raise RuntimeError(
        "smart_acquisition no longer parses CLI input. "
        "Call smart_acquisition.workflow from a UI or script instead."
    )
