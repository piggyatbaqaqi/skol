"""
Local copies of files fetched from the web, kept under /data/skol/www.

The tree mirrors URLs the way ``wget -x`` does: ``<root>/<host>/<path>``,
host lower-cased, path segments as in the URL.  Files here are for
SKOL's own use and are not served (``SKOL_WWW_DIR`` names the root;
default /data/skol/www).
"""

from pathlib import Path
from typing import Any, Dict

DEFAULT_ROOT = Path('/data/skol/www')


def mirror_path(url: str, root: Path) -> Path:
    """
    The local path for ``url`` under ``root``.

    Raises ValueError unless ``url`` is http(s), has a host, has no
    query or fragment, and its path names one file (no empty, ``.`` or
    ``..`` segments, no trailing slash).
    """
    raise NotImplementedError


def save_mirror_copy(url: str, data: bytes, root: Path) -> Dict[str, Any]:
    """
    Write ``data`` at ``mirror_path(url, root)`` atomically, creating
    directories as needed (they inherit the parent's setgid group).

    Returns:
        ``{'path', 'sha256', 'bytes'}``.
    """
    raise NotImplementedError
