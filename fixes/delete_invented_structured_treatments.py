#!/usr/bin/env python3
"""Delete structured treatments whose description had no usable text.

Before 8d15386, treatments_to_json sent every description to the
model, including empty ones, ones without letters, and OCR noise.
The model answered each with an invented taxon (a frog, for one
Nomen ignotum).  The pipeline now drops such descriptions before
inference and never saves them; this one-shot fix removes the ones
saved before that, using the same rule
(``treatments_json_translator.is_translatable``).

The target database must be named explicitly, with ``--experiment``
(its treatments_structured database) or ``--db``.

Supports ``--dry-run`` / ``--limit`` / ``--verbosity``.

Usage::

    fixes/delete_invented_structured_treatments.py \\
        --experiment production_v4_1 --dry-run
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))


# ---------------------------------------------------------------------------
# Pure helpers (covered by delete_invented_structured_treatments_test.py)
# ---------------------------------------------------------------------------


def is_invented(doc: Dict[str, Any]) -> bool:
    """True when ``doc``'s description gave the model nothing to read."""
    raise NotImplementedError


def target_db_name(
    db_arg: Optional[str],
    experiment_name: Optional[str],
    config: Dict[str, Any],
) -> str:
    """The database to clean: ``--db``, else the experiment's
    treatments_structured database.

    Raises:
        ValueError: If neither ``--db`` nor ``--experiment`` was given;
            a delete must never fall back to a default database.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# CouchDB iteration / delete (network-touching)
# ---------------------------------------------------------------------------


def delete_invented(
    db: Any,
    dry_run: bool = False,
    limit: Optional[int] = None,
    verbosity: int = 1,
) -> Dict[str, int]:
    """Walk ``db``; delete every doc for which ``is_invented`` is True.

    Returns ``{scanned, invented, deleted}``.  Under ``dry_run``,
    ``deleted`` counts what would have been deleted.
    """
    raise NotImplementedError


def main() -> int:
    """CLI entry point."""
    raise NotImplementedError


if __name__ == '__main__':
    sys.exit(main())
