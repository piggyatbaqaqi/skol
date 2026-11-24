"""
Local copies of files fetched from the web, kept under /data/skol/www.

The tree mirrors URLs the way ``wget -x`` does: ``<root>/<host>/<path>``,
host lower-cased, path segments as in the URL.  Files here are for
SKOL's own use and are not served (``SKOL_WWW_DIR`` names the root;
default /data/skol/www).
"""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlsplit

DEFAULT_ROOT = Path('/data/skol/www')


def mirror_path(url: str, root: Path) -> Path:
    """
    The local path for ``url`` under ``root``.

    Raises ValueError unless ``url`` is http(s), has a host, has no
    query or fragment, and its path names one file (no empty, ``.`` or
    ``..`` segments, no trailing slash).
    """
    parts = urlsplit(url)
    if parts.scheme.lower() not in ('http', 'https'):
        raise ValueError(f"not an http(s) URL: {url!r}")
    host = parts.netloc.lower()
    if not host or '@' in host:
        raise ValueError(f"URL has no plain host: {url!r}")
    if parts.query or parts.fragment or url.rstrip().endswith(('?', '#')):
        raise ValueError(f"URL has a query or fragment: {url!r}")
    segments = parts.path.split('/')[1:]
    if not segments or any(seg in ('', '.', '..') for seg in segments):
        raise ValueError(f"URL path does not name one file: {url!r}")
    return root.joinpath(host, *segments)


def save_mirror_copy(url: str, data: bytes, root: Path) -> Dict[str, Any]:
    """
    Write ``data`` at ``mirror_path(url, root)`` atomically, creating
    directories as needed (they inherit the parent's setgid group).

    Returns:
        ``{'path', 'sha256', 'bytes'}``.
    """
    path = mirror_path(url, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix='.',
                                    suffix='.part')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        # mkstemp makes 0600; use the tree's mode (-rw-rw-r--, group
        # skol via the setgid directories).
        os.chmod(tmp_name, 0o664)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return {
        'path': str(path),
        'sha256': hashlib.sha256(data).hexdigest(),
        'bytes': len(data),
    }
