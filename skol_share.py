"""Where skol's shipped, read-only data lives.

Code that opens a file from the repository's ``data/`` directory finds
it through this module, so the same call works from a source checkout
and from an installed package (Trello EMbuJc0r).

The directory is chosen in CLAUDE.md priority order:

1. an explicit value (a ``--share-dir`` flag, through env_config);
2. the ``SKOL_SHARE`` environment variable;
3. the checkout's ``data/``, when this module runs from a source
   checkout (its directory holds ``pyproject.toml`` and ``data/``);
4. ``/opt/skol/share``, which the deb links to
   ``/opt/skol/versions/<ver>/share/`` and switches with
   ``/opt/skol/bin``.

Only ``os.environ`` is read here.  ``env_config.get_env_config`` also
reads ``~/.skol_env`` and passes the result in as the explicit value.
"""
import os
from pathlib import Path
from typing import Optional, Union

SHARE_ENV = 'SKOL_SHARE'
INSTALLED_SHARE = Path('/opt/skol/share')

PathLike = Union[str, Path]


def checkout_data_dir(module_file: Optional[Path] = None) -> Optional[Path]:
    """The checkout's ``data/`` if ``module_file`` runs from one, else None.

    A checkout is a directory holding both ``pyproject.toml`` and
    ``data/``.  An installed module sits in site-packages, which has
    neither.
    """
    root = (module_file or Path(__file__)).resolve().parent
    data = root / 'data'
    if (root / 'pyproject.toml').is_file() and data.is_dir():
        return data
    return None


def share_dir(cli_value: Optional[PathLike] = None, *,
              module_file: Optional[Path] = None) -> Path:
    """The share directory, in the priority order above."""
    if cli_value:
        return Path(cli_value)
    from_env = os.environ.get(SHARE_ENV)
    if from_env:
        return Path(from_env)
    return checkout_data_dir(module_file) or INSTALLED_SHARE


def share_file(name: PathLike, cli_value: Optional[PathLike] = None, *,
               module_file: Optional[Path] = None) -> Path:
    """``name`` (relative to the share directory) as a full path."""
    return share_dir(cli_value, module_file=module_file) / name
