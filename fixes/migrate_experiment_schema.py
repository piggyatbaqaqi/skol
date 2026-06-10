#!/usr/bin/env python3
"""Rewrite experiment docs to the post-2026-06-10 schema.

Per the rename pass settled 2026-06-09 in
``docs/skol-db-naming-cleanup.md``, each experiment doc's
``databases`` section migrates from:

    {
      "annotations": "skol_exp_<name>_ann",
      "taxa":        "skol_exp_<name>_taxa",
      "taxa_full":   "skol_exp_<name>_taxa_full",
    }

(or the mid-migration ``treatments`` / ``treatments_full``
variants) to:

    {
      "annotations":          "skol_exp_<name>_01_00_ann",
      "annotations_eval":     "skol_exp_<name>_01_00_ann_eval",
      "treatments_prose":     "skol_exp_<name>_02_00_treatments_prose",
      "treatments_prose_eval":"skol_exp_<name>_02_00_treatments_prose_eval",
      "treatments_structured":"skol_exp_<name>_03_00_treatments_structured",
      "treatments_structured_eval":
                              "skol_exp_<name>_03_00_treatments_structured_eval",
    }

The script ONLY rewrites the experiment-doc fields.  Actual DB
renames happen in ``fixes/rename_dbs.py`` (CouchDB
replicate-to-new-name + delete-old).  Run this first to get the
schema in place; then run ``rename_dbs.py`` in the background
to do the bytes-on-disk work.

The OLD field names (``taxa``, ``taxa_full``, ``treatments``,
``treatments_full``) are deliberately PRESERVED in the doc so
in-flight code that still reads them keeps working through the
transition window.  A follow-up commit drops them once every
consumer is on the new names.

Idempotent: re-running on an already-migrated doc is a no-op.

Usage::

    # Dry run first — print every doc that would change.
    python fixes/migrate_experiment_schema.py --dry-run

    # Actually apply.
    python fixes/migrate_experiment_schema.py --execute
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from env_config import get_env_config  # type: ignore[import]  # noqa: E402


def _legacy_db_name(databases: Dict[str, str], *fields: str) -> Optional[str]:
    """Return the first non-empty value among ``fields`` in
    ``databases``.  Walks the legacy fallback chain so a doc
    anywhere on the migration timeline still resolves."""
    for f in fields:
        v = databases.get(f)
        if v:
            return v
    return None


def _stage_tagged(
    legacy_name: Optional[str], experiment_name: str,
    stage_tag: str, role: str,
) -> str:
    """Compute the post-2026-06-10 DB name from a legacy name.

    Strategy: if the legacy name carries a ``_combined`` or other
    model-variant suffix (the ``_combined`` from Step 7's
    cutover), preserve it at the END so the per-doc renamed DB
    keeps its variant identity through Q5.  Otherwise use the
    canonical ``skol_exp_<name>_<stage_tag>_<role>`` form.
    """
    canonical = f"skol_exp_{experiment_name}_{stage_tag}_{role}"
    if not legacy_name:
        return canonical
    # Detect a `_combined` (or similar) suffix on the legacy name.
    # Pattern: `skol_exp_<X>_<role-fragment>_<suffix>`.  Extract
    # the suffix by stripping any `skol_exp_<name>_<role>` prefix.
    # Longest prefixes FIRST so e.g. ``_taxa_full`` matches before
    # ``_taxa`` and we don't misread the `_full` token as a model
    # variant suffix.
    prefixes_to_strip = (
        f"skol_exp_{experiment_name}_treatments_full",
        f"skol_exp_{experiment_name}_treatments",
        f"skol_exp_{experiment_name}_taxa_full",
        f"skol_exp_{experiment_name}_taxa",
        f"skol_exp_{experiment_name}_ann",
    )
    for prefix in prefixes_to_strip:
        if legacy_name.startswith(prefix):
            suffix = legacy_name[len(prefix):].lstrip('_')
            if suffix:
                return f"{canonical}_{suffix}"
            return canonical
    # Legacy name doesn't fit the pattern (operator-overridden
    # custom name).  Preserve as-is at the canonical slot —
    # operator intent overrides automation.
    return legacy_name


def migrate_doc(doc: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Compute the post-migration ``databases`` block for one
    experiment doc.

    Returns ``(changed, new_databases)``.  ``changed`` is False
    when the doc is already on the post-migration schema
    (idempotent).
    """
    name = doc.get('_id', '')
    databases = doc.get('databases', {}) or {}
    new_db = dict(databases)  # start with a copy of existing fields

    # Resolve legacy DB names via the fallback chain (current → mid
    # → legacy field), then compute the post-2026-06-10 names.
    legacy_ann = databases.get('annotations')
    legacy_prose = _legacy_db_name(
        databases, 'treatments_prose', 'treatments', 'taxa',
    )
    legacy_structured = _legacy_db_name(
        databases,
        'treatments_structured', 'treatments_full', 'taxa_full',
    )

    new_ann = _stage_tagged(legacy_ann, name, '01_00', 'ann')
    new_prose = _stage_tagged(
        legacy_prose, name, '02_00', 'treatments_prose',
    )
    new_structured = _stage_tagged(
        legacy_structured, name, '03_00', 'treatments_structured',
    )

    # Build the post-migration block.  Old fields are PRESERVED
    # so in-flight code keeps working through the transition
    # window; the script-side helpers walk the fallback chain.
    new_db.setdefault('annotations', new_ann)
    new_db['annotations'] = new_ann
    new_db['annotations_eval'] = f"{new_ann}_eval"
    new_db['treatments_prose'] = new_prose
    new_db['treatments_prose_eval'] = f"{new_prose}_eval"
    new_db['treatments_structured'] = new_structured
    new_db['treatments_structured_eval'] = f"{new_structured}_eval"
    # spans is co-located with annotations.
    new_db['spans'] = new_ann

    changed = (new_db != databases)
    return changed, new_db


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Migrate experiment docs to post-2026-06-10 schema',
    )
    parser.add_argument(
        '--dry-run', action='store_true', default=True,
        help='Print changes without saving (default).',
    )
    parser.add_argument(
        '--execute', action='store_false', dest='dry_run',
        help='Actually save the migrated docs.',
    )
    parser.add_argument(
        '--experiments-db', default='skol_experiments',
        help='Name of the experiments registry DB.',
    )
    args = parser.parse_args()

    config = get_env_config()
    import couchdb  # type: ignore[import]
    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (
        config['couchdb_username'],
        config['couchdb_password'],
    )
    db = server[args.experiments_db]

    n_examined = 0
    n_changed = 0
    for doc_id in db:
        if doc_id.startswith('_'):
            continue
        doc = db[doc_id]
        n_examined += 1
        changed, new_db = migrate_doc(doc)
        if not changed:
            print(f'  {doc_id:<30} unchanged')
            continue
        n_changed += 1
        print(f'  {doc_id:<30} CHANGED')
        if args.dry_run:
            old_db = doc.get('databases', {}) or {}
            for k in sorted(set(old_db) | set(new_db)):
                old_v = old_db.get(k)
                new_v = new_db.get(k)
                if old_v != new_v:
                    print(f'     {k:<32}  {old_v!r:<50}  →  {new_v!r}')
        else:
            doc['databases'] = new_db
            db.save(doc)
            print(f'     saved {doc_id} (rev now {doc.get("_rev")})')

    print(
        f'\n{n_examined} docs examined, {n_changed} would change '
        f'({"dry-run" if args.dry_run else "applied"}).'
    )
    if args.dry_run and n_changed > 0:
        print('Re-run with --execute to apply.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
