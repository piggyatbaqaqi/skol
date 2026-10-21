#!/usr/bin/env python3
"""Record a quality verdict for every treatment in an experiment.

Runs the detector suite that ``tests/pathologies_test.py`` holds to
the fixture ground truth (``triage_signals.treatment_signals`` plus
``merge_metric.treatment_merge_metric``, rolled up by
``predicted_issues``) over the experiment's prose treatments, and
writes one verdict per treatment to the experiment's
``databases.treatment_quality`` database::

    {"_id": "<treatment id>", "clean": true, "flags": [],
     "usable_text": true, "gn_evaluated": true,
     "checks_version": "<hash>", "checked_at": "<ISO time>"}

A treatment is ``clean`` only when no detector fires, its text is
usable (``treatments_json_translator.is_translatable``: an empty
treatment fires no flags), and the gnfinder authored-binomial check
actually ran.  Clean means no known pathology, not perfect.

``checks_version`` hashes the detector sources, so adding or changing
a detector makes every verdict visibly stale; the next run reassesses
those and skips current ones (``--force`` reassesses everything).

Usage::

    bin/assess_treatments --experiment production_v4_1
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent.parent
DETECTOR_SOURCES: Sequence[Path] = (
    REPO / 'treatments_to_structured' / 'triage_signals.py',
    REPO / 'treatments_to_structured' / 'merge_metric.py',
)

GnCheck = Callable[[str], Optional[bool]]


def checks_version(sources: Sequence[Path] = DETECTOR_SOURCES) -> str:
    """Hash of the detector source files, in order."""
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.name.encode('utf-8') + b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def assess(
    treatment: Dict[str, Any],
    authored_binomial: Optional[bool],
) -> Dict[str, Any]:
    """Run the detectors on one treatment.

    ``authored_binomial`` is the gnfinder result, or None when the
    check did not run.

    Returns ``{clean, flags, usable_text, gn_evaluated}``; ``flags`` is
    ``predicted_issues`` split on ``|`` (values kept, e.g.
    ``§6:merge_metric=14``).
    """
    from treatments_json_translator import (
        build_llm_input_text,
        is_translatable,
    )
    from treatments_to_structured.merge_metric import (
        treatment_merge_metric,
    )
    from treatments_to_structured.triage_signals import (
        predicted_issues,
        treatment_signals,
    )

    signals = treatment_signals(
        treatment, authored_binomial_in_desc=authored_binomial)
    issues = predicted_issues(signals, treatment_merge_metric(treatment))
    flags = issues.split('|') if issues else []
    usable_text = is_translatable(build_llm_input_text(treatment))
    gn_evaluated = authored_binomial is not None
    return {
        'clean': not flags and usable_text and gn_evaluated,
        'flags': flags,
        'usable_text': usable_text,
        'gn_evaluated': gn_evaluated,
    }


def verdict_doc(
    treatment_id: str,
    verdict: Dict[str, Any],
    version: str,
    checked_at: str,
) -> Dict[str, Any]:
    """The quality-DB document for one verdict."""
    return {
        '_id': treatment_id,
        'clean': verdict['clean'],
        'flags': list(verdict['flags']),
        'usable_text': verdict['usable_text'],
        'gn_evaluated': verdict['gn_evaluated'],
        'checks_version': version,
        'checked_at': checked_at,
    }


def needs_assessment(
    existing: Optional[Dict[str, Any]],
    version: str,
    force: bool = False,
) -> bool:
    """True when there is no verdict, it is stale, or ``force``."""
    if force or existing is None:
        return True
    return bool(existing.get('checks_version') != version)


def quality_db_name(config: Dict[str, Any]) -> str:
    """The experiment's treatment_quality database.

    Raises:
        ValueError: If the experiment record has none.
    """
    name = config.get('treatment_quality_db_name')
    if not name:
        raise ValueError(
            'the experiment record has no databases.treatment_quality; '
            'run: bin/manage_experiment update <experiment>')
    return str(name)


def assess_all(
    treatments: Iterable[Dict[str, Any]],
    quality_db: Any,
    gn_check: GnCheck,
    version: str,
    checked_at: str,
    force: bool = False,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Assess each treatment that needs it and save its verdict.

    Returns ``{scanned, assessed, clean, skipped}``.
    """
    counts = {'scanned': 0, 'assessed': 0, 'clean': 0, 'skipped': 0}
    for treatment in treatments:
        treatment_id = treatment['_id']
        if treatment_id.startswith('_design/'):
            continue
        counts['scanned'] += 1
        existing = quality_db.get(treatment_id)
        if not needs_assessment(existing, version, force):
            counts['skipped'] += 1
            continue
        verdict = assess(treatment,
                         gn_check(treatment.get('description') or ''))
        doc = verdict_doc(treatment_id, verdict, version, checked_at)
        if existing is not None and existing.get('_rev'):
            doc['_rev'] = existing['_rev']
        if not dry_run:
            quality_db.save(doc)
        counts['assessed'] += 1
        counts['clean'] += int(verdict['clean'])
    return counts


def make_gn_check(config: Dict[str, Any]) -> GnCheck:
    """gnfinder authored-binomial check; None once the services fail.

    The first failure is warned once; every later treatment then gets
    None (not evaluated), so none of them can be clean.
    """
    from treatments_to_structured import gn_client

    gnfinder_url = config.get('gnfinder_url',
                              gn_client.DEFAULT_GNFINDER_URL)
    gnparser_url = config.get('gnparser_url',
                              gn_client.DEFAULT_GNPARSER_URL)
    unavailable = False

    def gn_check(text: str) -> Optional[bool]:
        nonlocal unavailable
        if unavailable:
            return None
        try:
            return bool(gn_client.authored_binomial_in_text(
                text, gnfinder_url, gnparser_url))
        except gn_client.GnServiceUnavailable as exc:
            print(f'warning: gn services unavailable ({exc}); no '
                  f'treatment can be clean for the rest of this run',
                  file=sys.stderr)
            unavailable = True
            return None

    return gn_check


def _iter_treatments(db: Any) -> Iterable[Dict[str, Any]]:
    for doc_id in db:
        if not doc_id.startswith('_design/'):
            yield dict(db[doc_id])


def build_parser() -> argparse.ArgumentParser:
    from env_config import common_parser

    return argparse.ArgumentParser(
        parents=[common_parser()],
        description='Record a detector-suite quality verdict for every '
                    'treatment in an experiment.',
    )


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    from env_config import get_env_config

    parser = build_parser()
    args = parser.parse_args(argv)
    config = get_env_config(cli_args=args)
    verbosity = config.get('verbosity', 1)
    try:
        quality_name = quality_db_name(config)
    except ValueError as exc:
        parser.error(str(exc))
    source_name = (config.get('treatments_prose_db_name')
                   or config.get('treatments_db_name'))

    import couchdb
    server = couchdb.Server(config['couchdb_url'])
    user = config.get('couchdb_username') or ''
    password = config.get('couchdb_password') or ''
    if user and password:
        server.resource.credentials = (user, password)
    if source_name not in server:
        print(f'error: database {source_name!r} not found at '
              f'{config["couchdb_url"]}', file=sys.stderr)
        return 2
    dry_run = bool(config.get('dry_run'))
    if quality_name not in server and not dry_run:
        server.create(quality_name)
    quality_db: Any = (server[quality_name] if quality_name in server
                       else {})

    version = checks_version()
    if verbosity >= 1:
        print(f'Treatments: {source_name}')
        print(f'Verdicts:   {quality_name} '
              f'{"(DRY RUN)" if dry_run else ""}')
        print(f'Checks version: {version}')
    treatments = _iter_treatments(server[source_name])
    limit = config.get('limit')
    if limit:
        treatments = itertools.islice(treatments, limit)
    counts = assess_all(
        treatments, quality_db, make_gn_check(config), version,
        checked_at=datetime.now(timezone.utc).isoformat(),
        force=bool(config.get('force')), dry_run=dry_run,
    )
    if verbosity >= 1:
        print(f'Scanned {counts["scanned"]}, assessed {counts["assessed"]} '
              f'({counts["clean"]} clean), skipped {counts["skipped"]} '
              f'current')
    return 0


if __name__ == '__main__':
    sys.exit(main())
