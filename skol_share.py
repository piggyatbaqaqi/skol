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

Skeleton: the functions are stubs until the tests are approved.
"""
from pathlib import Path
from typing import Optional, Union

SHARE_ENV = 'SKOL_SHARE'
INSTALLED_SHARE = Path('/opt/skol/share')

PathLike = Union[str, Path]


def checkout_data_dir(module_file: Optional[Path] = None) -> Optional[Path]:
    """The checkout's ``data/`` if ``module_file`` runs from one, else None."""
    raise NotImplementedError


def share_dir(cli_value: Optional[PathLike] = None, *,
              module_file: Optional[Path] = None) -> Path:
    """The share directory, in the priority order above."""
    raise NotImplementedError


def share_file(name: PathLike, cli_value: Optional[PathLike] = None, *,
               module_file: Optional[Path] = None) -> Path:
    """``name`` (relative to the share directory) as a full path."""
    raise NotImplementedError
