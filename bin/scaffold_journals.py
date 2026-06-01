#!/usr/bin/env python3
"""Scaffold a draft ``JOURNALS`` dict from the current ``SOURCES``
table, optionally enriching with Crossref ``/journals/{issn}`` lookups.

Operator workflow:

    bin/scaffold_journals.py --output ingestors/JOURNALS_draft.py [--all]

Produces a Python literal that you hand-edit (fill in official
websites, ISO 4 abbreviations, alias lists), then move into
``ingestors/publications.py`` as part of the phase-1 commit.

Re-runnable for a single journal when adding a new one:

    bin/scaffold_journals.py --journal sydowia --output JOURNALS_draft.py

The script never overwrites an existing ``JOURNALS`` row in
``publications.py`` — it only writes a draft file the operator
inspects and merges.  See
``docs/publications_metadata_consolidation.md``.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ``strip_publisher_suffix`` lives in ingestors/publications.py
# (one source of truth for the recognised-publisher list — used
# by ``normalize_journal_name`` there and by this scaffolding
# script).
from ingestors.publications import strip_publisher_suffix  # noqa: E402


# ---------------------------------------------------------------------------
# Pure helpers (covered by scaffold_journals_test.py)
# ---------------------------------------------------------------------------


_SLUG_RE = re.compile(r'[^a-z0-9]+')


def slugify(name: str) -> str:
    """Display name → URL-safe slug.

    Lowercase; runs of non-alphanumeric characters collapse to a
    single hyphen; leading/trailing hyphens stripped.  Idempotent
    on already-slug-shaped input.
    """
    return _SLUG_RE.sub('-', name.lower()).strip('-')


def infer_slug_from_journal_name(name: str) -> str:
    """``"Journal of Fungi (PMC)"`` and ``"Journal of Fungi"`` both
    resolve to ``"journal-of-fungi"`` — that's how multi-source
    journals collapse into one JOURNALS slug."""
    return slugify(strip_publisher_suffix(name))


def collect_unique_journals(
    sources: Dict[str, Dict[str, Any]],
) -> Dict[str, List[str]]:
    """Walk SOURCES → ``{slug: [source_keys_pointing_at_it]}``.

    Entries whose ``journal`` field is missing / None are skipped
    (local-mirror ingestors like mykoweb-caf, where the journal
    isn't a single thing).  Source-key lists are sorted so the
    scaffold's output diffs cleanly across re-runs.
    """
    by_slug: Dict[str, List[str]] = {}
    for source_key, cfg in sources.items():
        journal = cfg.get('journal')
        if not journal:
            continue
        slug = infer_slug_from_journal_name(journal)
        by_slug.setdefault(slug, []).append(source_key)
    return {slug: sorted(keys) for slug, keys in by_slug.items()}


def extract_journal_fields_from_crossref(
    msg: Dict[str, Any],
) -> Dict[str, Any]:
    """Pull JournalEntry-shaped fields out of a Crossref
    ``/journals/{issn}`` reply.

    Returns a dict carrying ``name`` / ``publisher`` / ``issn`` /
    ``eissn`` where Crossref had them.  Crossref doesn't reliably
    return official websites or ISO 4 abbreviations — those stay
    hand-edited in the resulting draft.
    """
    out: Dict[str, Any] = {}
    title = msg.get('title')
    if isinstance(title, str):
        stripped = title.strip()
        if stripped:
            out['name'] = stripped
    publisher = msg.get('publisher')
    if isinstance(publisher, str) and publisher.strip():
        out['publisher'] = publisher.strip()
    issns = msg.get('ISSN')
    if isinstance(issns, list) and issns:
        out['issn'] = issns[0]
        if len(issns) > 1:
            out['eissn'] = issns[1]
    return out


# ---------------------------------------------------------------------------
# Crossref enrichment (network-touching)
# ---------------------------------------------------------------------------


def _lookup_journal_in_crossref(
    issn: str, mailto: str, verbosity: int = 1,
) -> Dict[str, Any]:
    """Hit Crossref ``/journals/{issn}``; return the ``message`` dict
    on success, an empty dict on any failure (network, 404, parse).
    """
    from habanero import Crossref
    cr = Crossref(mailto=mailto)
    try:
        resp = cr.journals(ids=issn)
    except Exception as exc:
        if verbosity >= 1:
            print(f'  warning: Crossref failed for ISSN {issn}: {exc}',
                  file=sys.stderr)
        return {}
    msg = resp.get('message') if isinstance(resp, dict) else None
    if not isinstance(msg, dict):
        return {}
    return msg


def _collect_issns_from_sources(
    source_keys: List[str], sources: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Gather the ISSN values present on any of the SOURCES entries
    pointing at one journal.  De-duped, order preserved."""
    seen: List[str] = []
    for key in source_keys:
        issn = sources[key].get('issn')
        if isinstance(issn, str) and issn and issn not in seen:
            seen.append(issn)
    return seen


def build_journal_draft_record(
    slug: str,
    source_keys: List[str],
    sources: Dict[str, Dict[str, Any]],
    mailto: Optional[str] = None,
    verbosity: int = 1,
) -> Dict[str, Any]:
    """One scaffold record: combines whatever ISSNs the current
    SOURCES rows carry with whatever Crossref returns for those
    ISSNs.  ``name`` falls back to the stripped journal name when
    Crossref doesn't return one.
    """
    issns = _collect_issns_from_sources(source_keys, sources)
    record: Dict[str, Any] = {}
    if mailto and issns:
        if verbosity >= 1:
            print(f'  {slug}: Crossref lookup ISSN={issns[0]}')
        crossref_msg = _lookup_journal_in_crossref(
            issns[0], mailto=mailto, verbosity=verbosity,
        )
        record.update(extract_journal_fields_from_crossref(crossref_msg))
    # Fill in ISSN/eISSN from SOURCES if Crossref didn't (this
    # mainly happens for non-ISSN-bearing entries or Crossref
    # 404s on obscure ISSNs).
    if 'issn' not in record and issns:
        record['issn'] = issns[0]
    if 'eissn' not in record and len(issns) > 1:
        record['eissn'] = issns[1]
    # Fallback name from the first SOURCES entry.
    if 'name' not in record:
        first_name = sources[source_keys[0]].get('journal')
        if isinstance(first_name, str) and first_name.strip():
            record['name'] = strip_publisher_suffix(first_name)
    return record


# ---------------------------------------------------------------------------
# Output formatting + CLI
# ---------------------------------------------------------------------------


_DRAFT_HEADER = '''"""Draft JOURNALS rows produced by bin/scaffold_journals.py.

Hand-edit (fill in official-website URLs, abbreviations, aliases)
then merge into ingestors/publications.py.  This file is intended
to be consumed by a human, not imported by code.
"""

JOURNALS_DRAFT = {
'''

_DRAFT_FOOTER = '}\n'


def _format_record(slug: str, record: Dict[str, Any],
                   source_keys: List[str]) -> str:
    """One JOURNALS entry, formatted for a Python literal that's
    easy to scan and hand-edit.  Includes a comment listing the
    SOURCES entries that pointed at this slug — operator can spot
    typos / mis-attributions at a glance."""
    lines: List[str] = []
    lines.append(f'    # from SOURCES: {", ".join(source_keys)}')
    lines.append(f'    {slug!r}: {{')
    # Always lead with the human display fields.
    if 'name' in record:
        lines.append(f'        {"name"!r:<12s}: {record["name"]!r},')
    for k in ('publisher', 'issn', 'eissn', 'isbn', 'doi', 'abbrev'):
        if k in record:
            lines.append(f'        {k!r:<12s}: {record[k]!r},')
    lines.append(f'        {"address"!r:<12s}: \'\',  # TODO: official website')
    lines.append(f'        {"aliases"!r:<12s}: [],   # TODO: known alias variants')
    lines.append('    },')
    return '\n'.join(lines)


def emit_draft(
    drafts: Dict[str, Dict[str, Any]],
    by_slug_sources: Dict[str, List[str]],
    out_path: Path,
) -> None:
    """Write the scaffolded ``JOURNALS_DRAFT`` literal to ``out_path``."""
    parts = [_DRAFT_HEADER]
    for slug in sorted(drafts):
        parts.append(_format_record(
            slug, drafts[slug], by_slug_sources[slug],
        ))
    parts.append(_DRAFT_FOOTER)
    out_path.write_text('\n'.join(parts), encoding='utf-8')


def main() -> int:
    """CLI entry point — parse args, dispatch to the scaffold."""
    parser = argparse.ArgumentParser(
        description='Scaffold a draft JOURNALS dict from current '
                    'SOURCES, optionally enriched via Crossref.',
    )
    parser.add_argument(
        '--crossref-mailto', type=str, default=None,
        help='Email for the Crossref polite pool.  Required for '
             'Crossref enrichment; without it the draft is built '
             'from SOURCES data only.',
    )
    parser.add_argument(
        '--output', type=Path,
        default=Path('ingestors/JOURNALS_draft.py'),
        help='Output path for the draft Python file.',
    )
    parser.add_argument(
        '--journal', type=str, default=None,
        help='Scaffold a single journal by slug (use with --all '
             'off; e.g. --journal sydowia).  Useful when adding a '
             'new entry without re-scaffolding the whole table.',
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Scaffold every journal found in SOURCES.  Mutually '
             'exclusive with --journal.',
    )
    parser.add_argument(
        '--verbosity', type=int, default=1,
        help='0=quiet, 1=normal, 2=verbose.',
    )
    args = parser.parse_args()

    if not args.all and not args.journal:
        parser.error('one of --all or --journal SLUG is required')

    from ingestors.publications import PublicationRegistry
    by_slug = collect_unique_journals(PublicationRegistry.SOURCES)
    if args.journal:
        by_slug = {args.journal: by_slug.get(args.journal, [])}
        if not by_slug[args.journal]:
            print(f'error: no SOURCES entry maps to slug '
                  f'{args.journal!r}', file=sys.stderr)
            return 2

    if args.verbosity >= 1:
        print(f'Scaffolding {len(by_slug)} journals')

    drafts: Dict[str, Dict[str, Any]] = {}
    for slug, source_keys in by_slug.items():
        drafts[slug] = build_journal_draft_record(
            slug, source_keys, PublicationRegistry.SOURCES,
            mailto=args.crossref_mailto,
            verbosity=args.verbosity,
        )

    out_path = args.output.expanduser().resolve()
    emit_draft(drafts, by_slug, out_path)
    print(f'Wrote {len(drafts)} draft journal entries → {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
