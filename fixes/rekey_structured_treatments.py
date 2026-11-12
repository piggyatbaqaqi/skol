#!/usr/bin/env python3
"""Rename structured treatments to their prose treatment's ``_id``.

treatments_to_json used to name each structured treatment by a hash of
its treatment name and description-plus-diagnosis text, while prose
treatments are named by a hash of all eleven section fields.  The two
never match, so anything that looks a structured treatment up by the
prose ``_id`` it was given (the JSON Features tab) finds nothing.
treatments_to_json now names structured treatments by the prose
``_id``; this one-shot fix renames the existing ones.

For each translatable prose treatment, the old name is
``taxon_doc_id(treatment, build_llm_input_text(doc))``.  The document is
copied to the prose ``_id`` (prose treatments sharing identical text
each get their own copy); old names are deleted only after every copy
succeeded.  Rerunning finds nothing left to do.

Usage::

    fixes/rekey_structured_treatments.py --experiment production_v4_1 --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from treatments_json_translator import (  # noqa: E402
    build_llm_input_text,
    is_translatable,
)
from treatments_to_structured.modal_inference import (  # noqa: E402
    taxon_doc_id,
)


def legacy_structured_id(prose_doc: Dict[str, Any]) -> str:
    """The name treatments_to_json gave this prose treatment's JSON."""
    return taxon_doc_id(prose_doc.get('treatment') or '',
                        build_llm_input_text(prose_doc))


def plan_rekey(
    prose_docs: Iterable[Dict[str, Any]],
    structured_ids: Set[str],
) -> Dict[str, Any]:
    """What renaming would do, without doing it.

    Returns ``{copies: [(old_id, new_id)], already: [new_id],
    missing: [prose_id], retire: {old_id}, untranslatable: int}``:

    - copies: prose treatments whose JSON exists under the old name only;
    - already: prose treatments already named by their own ``_id``;
    - missing: translatable prose treatments with no JSON under either;
    - retire: old names to delete once every copy is made;
    - untranslatable: prose treatments with no usable text (never sent).
    """
    copies: List[Tuple[str, str]] = []
    already: List[str] = []
    missing: List[str] = []
    retire: Set[str] = set()
    untranslatable = 0
    for doc in prose_docs:
        if not is_translatable(build_llm_input_text(doc)):
            untranslatable += 1
            continue
        new_id = doc['_id']
        old_id = legacy_structured_id(doc)
        if new_id in structured_ids:
            already.append(new_id)
            if old_id != new_id and old_id in structured_ids:
                retire.add(old_id)
        elif old_id in structured_ids:
            copies.append((old_id, new_id))
            retire.add(old_id)
        else:
            missing.append(new_id)
    # An old name that is also some prose treatment's own name stays.
    named = {new for _, new in copies} | set(already)
    return {'copies': copies, 'already': already, 'missing': missing,
            'retire': retire - named, 'untranslatable': untranslatable}


def apply_rekey(
    db: Any,
    plan: Dict[str, Any],
    dry_run: bool = False,
    verbosity: int = 1,
) -> Dict[str, int]:
    """Make the copies, then delete the retired names.

    A copy is the old document's content under the new ``_id`` (no
    ``_rev``).  If any copy fails nothing is deleted, so a rerun can
    finish the job.  Returns ``{copied, failed, deleted}``; under
    ``dry_run`` the counts are what would happen and nothing is written.
    """
    copies: List[Tuple[str, str]] = plan['copies']
    retire: Set[str] = plan['retire']
    if dry_run:
        return {'copied': len(copies), 'failed': 0, 'deleted': len(retire)}
    copied = failed = 0
    for old_id, new_id in copies:
        try:
            doc = dict(db[old_id])
            doc.pop('_rev', None)
            doc['_id'] = new_id
            db.save(doc)
            copied += 1
        except Exception as exc:  # noqa: BLE001 - reported, then counted
            failed += 1
            if verbosity >= 1:
                print(f'  copy {old_id} -> {new_id} failed: {exc}')
    deleted = 0
    if failed:
        if verbosity >= 1:
            print(f'  {failed} copies failed; deleting nothing')
    else:
        for old_id in sorted(retire):
            doc = db.get(old_id)
            if doc is not None:
                db.delete(doc)
                deleted += 1
    return {'copied': copied, 'failed': failed, 'deleted': deleted}


def target_dbs(experiment_name: Optional[str],
               config: Dict[str, Any]) -> Dict[str, str]:
    """``{prose, structured}`` database names for the experiment.

    Raises:
        ValueError: Without ``--experiment``; a rename must never fall
            back to a default database.
    """
    if not experiment_name:
        raise ValueError('--experiment is required')
    return {'prose': config['treatments_prose_db_name'],
            'structured': config['treatments_structured_db_name']}


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    from env_config import common_parser, get_env_config

    parser = argparse.ArgumentParser(
        parents=[common_parser()],
        description="Rename structured treatments to their prose "
                    "treatment's _id.",
    )
    args = parser.parse_args(argv)
    config = get_env_config(cli_args=args)
    verbosity = config.get('verbosity', 1)
    try:
        dbs = target_dbs(config.get('experiment_name'), config)
    except ValueError as exc:
        parser.error(str(exc))

    import couchdb
    server = couchdb.Server(config['couchdb_url'])
    user = config.get('couchdb_username') or ''
    password = config.get('couchdb_password') or ''
    if user and password:
        server.resource.credentials = (user, password)
    for name in dbs.values():
        if name not in server:
            print(f'error: database {name!r} not found at '
                  f'{config["couchdb_url"]}', file=sys.stderr)
            return 2

    dry_run = bool(config.get('dry_run'))
    prose_db = server[dbs['prose']]
    structured_db = server[dbs['structured']]
    structured_ids = {i for i in structured_db if not i.startswith('_design')}
    prose_docs = (prose_db[i] for i in prose_db if not i.startswith('_design'))
    plan = plan_rekey(prose_docs, structured_ids)
    if verbosity >= 1:
        print(f'Prose: {dbs["prose"]}')
        print(f'Structured: {dbs["structured"]} '
              f'{"(DRY RUN)" if dry_run else ""}')
        print(f'  {len(plan["copies"])} to copy, '
              f'{len(plan["already"])} already named, '
              f'{len(plan["missing"])} missing, '
              f'{plan["untranslatable"]} untranslatable, '
              f'{len(plan["retire"])} old names to retire')
    counts = apply_rekey(structured_db, plan, dry_run=dry_run,
                         verbosity=verbosity)
    if verbosity >= 1:
        verb = 'would' if dry_run else 'did'
        print(f'  {verb} copy {counts["copied"]}, '
              f'delete {counts["deleted"]}; {counts["failed"]} failed')
    return 1 if counts['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
