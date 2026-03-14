#!/usr/bin/env python3
"""
Link skol_training documents to their skol_dev counterparts.

Sets a 'skol_dev_id' field on each skol_training document that has a matching
document in skol_dev. The matching logic depends on journal and file_type:

  - Mycotaxon sections (Vol 117-119): section number -> page-order position
  - Mycotaxon issues (Vol 54-58): sole skol_dev doc for that journal+volume
  - Persoonia issues (Vol 16-18): match by journal+volume+issue number
  - Persoonia Vol 19 sections, Mycologia: no skol_dev counterpart

Usage:
    python fixes/link_training_to_dev.py --dry-run
    python fixes/link_training_to_dev.py
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from env_config import get_env_config


def _volume_variants(volume):
    """Return list of volume string variants to match (with/without zero padding)."""
    variants = [volume]
    stripped = volume.lstrip('0') or '0'
    if stripped != volume:
        variants.append(stripped)
    return variants


def build_dev_page_order(db_dev, journal, volume):
    """Get skol_dev articles for a journal+volume, sorted by starting page.

    Returns list of (start_page, doc_id, title).
    """
    results = db_dev.find({
        'selector': {
            'journal': {'$eq': journal},
            'volume': {'$in': _volume_variants(volume)},
        },
        'limit': 200,
        'fields': ['_id', 'pages', 'title'],
    })

    articles = []
    for doc in results:
        pages = doc.get('pages', '') or ''
        start = 0
        if '-' in pages:
            try:
                start = int(pages.split('-')[0])
            except ValueError:
                start = -1
        elif pages:
            try:
                start = int(pages)
            except ValueError:
                start = -1
        articles.append((start, doc['_id'], doc.get('title', '')))

    articles.sort()
    return articles


def build_dev_issue_index(db_dev, journal, volume):
    """Get skol_dev docs for a journal+volume, indexed by issue number.

    Returns dict of {number_str: (doc_id, title)}.
    """
    results = db_dev.find({
        'selector': {
            'journal': {'$eq': journal},
            'volume': {'$in': _volume_variants(volume)},
        },
        'limit': 50,
        'fields': ['_id', 'number', 'title'],
    })

    index = {}
    for doc in results:
        number = doc.get('number')
        if number is not None:
            index[str(number)] = (doc['_id'], doc.get('title', ''))
    return index


def get_section_number(source_file):
    """Extract section number from source_file path.

    'data/annotated/.../s6.txt.ann' -> 6
    """
    basename = Path(source_file).stem.replace('.txt', '')
    m = re.match(r'^s(\d+)$', basename)
    if m:
        return int(m.group(1))
    return None


def get_issue_number(source_file):
    """Extract issue number from source_file path.

    'data/annotated/.../n3.txt.ann' -> '3'
    """
    basename = Path(source_file).stem.replace('.txt', '')
    m = re.match(r'^n(\d+)$', basename)
    if m:
        return m.group(1)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Link skol_training documents to skol_dev counterparts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without modifying documents.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase verbosity.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output.",
    )

    args = parser.parse_args()
    verbosity = 0 if args.quiet else args.verbose

    config = get_env_config()

    import couchdb as couchdb_lib

    server = couchdb_lib.Server(config["couchdb_url"])
    server.resource.credentials = (
        config["couchdb_username"],
        config["couchdb_password"],
    )
    db_train = server['skol_training']
    db_dev = server['skol_dev']

    # Load all training docs
    train_docs = []
    for row in db_train.view('_all_docs', include_docs=True):
        if not row.id.startswith('_design/'):
            train_docs.append(row.doc)

    if verbosity >= 1:
        print(f"Loaded {len(train_docs)} training documents", file=sys.stderr)

    # Group by journal+volume for efficient matching
    groups = {}
    for doc in train_docs:
        key = (doc.get('journal', ''), doc.get('volume', ''))
        groups.setdefault(key, []).append(doc)

    linked = 0
    already_linked = 0
    unmatched = 0

    for (journal, volume), docs in sorted(groups.items()):
        if verbosity >= 1:
            print(
                f"\n{journal} Vol {volume}: {len(docs)} training docs",
                file=sys.stderr,
            )

        file_type = docs[0].get('file_type', 'unknown')

        if journal == 'Mycotaxon' and file_type == 'section':
            # Section-to-page-order matching
            dev_articles = build_dev_page_order(db_dev, journal, volume)
            if verbosity >= 1:
                print(
                    f"  {len(dev_articles)} skol_dev articles (page-order match)",
                    file=sys.stderr,
                )

            for doc in docs:
                sec_num = get_section_number(doc.get('source_file', ''))
                if sec_num is None or sec_num < 1 or sec_num > len(dev_articles):
                    if verbosity >= 1:
                        print(
                            f"  {doc['_id'][:16]}: no match (section {sec_num})",
                            file=sys.stderr,
                        )
                    unmatched += 1
                    continue

                _, dev_id, dev_title = dev_articles[sec_num - 1]
                if _link_doc(db_train, doc, dev_id, dev_title, sec_num,
                             args.dry_run, verbosity):
                    linked += 1
                else:
                    already_linked += 1

        elif file_type == 'issue':
            if journal == 'Mycotaxon':
                # Sole doc matching for whole-volume entries
                dev_articles = build_dev_page_order(db_dev, journal, volume)
                if len(dev_articles) == 1:
                    _, dev_id, dev_title = dev_articles[0]
                    for doc in docs:
                        if _link_doc(db_train, doc, dev_id, dev_title, None,
                                     args.dry_run, verbosity):
                            linked += 1
                        else:
                            already_linked += 1
                elif len(dev_articles) == 0:
                    if verbosity >= 1:
                        print(f"  No skol_dev docs found", file=sys.stderr)
                    unmatched += len(docs)
                else:
                    if verbosity >= 1:
                        print(
                            f"  Ambiguous: {len(dev_articles)} skol_dev docs",
                            file=sys.stderr,
                        )
                    unmatched += len(docs)

            elif journal == 'Persoonia':
                # Match by issue number
                issue_index = build_dev_issue_index(db_dev, journal, volume)
                if verbosity >= 1:
                    print(
                        f"  {len(issue_index)} skol_dev issues (number match)",
                        file=sys.stderr,
                    )

                for doc in docs:
                    issue_num = get_issue_number(doc.get('source_file', ''))
                    # Also check the 'number' field on the doc itself
                    if issue_num is None:
                        issue_num = doc.get('number')
                    if issue_num and str(issue_num) in issue_index:
                        dev_id, dev_title = issue_index[str(issue_num)]
                        if _link_doc(db_train, doc, dev_id, dev_title,
                                     f"n{issue_num}", args.dry_run, verbosity):
                            linked += 1
                        else:
                            already_linked += 1
                    else:
                        if verbosity >= 1:
                            print(
                                f"  {doc['_id'][:16]}: no match "
                                f"(issue {issue_num})",
                                file=sys.stderr,
                            )
                        unmatched += 1
            else:
                if verbosity >= 1:
                    print(f"  Unknown journal for issue matching", file=sys.stderr)
                unmatched += len(docs)

        else:
            # Sections for non-Mycotaxon (Persoonia Vol 19, Mycologia)
            if verbosity >= 1:
                print(
                    f"  No matching strategy for {journal} {file_type}",
                    file=sys.stderr,
                )
            unmatched += len(docs)

    if verbosity >= 1:
        prefix = "Would link" if args.dry_run else "Linked"
        print(f"\n{prefix} {linked} training docs to skol_dev.", file=sys.stderr)
        if already_linked:
            print(f"Already linked: {already_linked}", file=sys.stderr)
        print(f"Unmatched: {unmatched}", file=sys.stderr)


def _link_doc(db_train, doc, dev_id, dev_title, label, dry_run, verbosity):
    """Set skol_dev_id on a training doc. Returns True if newly linked."""
    existing = doc.get('skol_dev_id')
    if existing == dev_id:
        if verbosity >= 3:
            print(
                f"  {doc['_id'][:16]}: already linked to {dev_id[:16]}",
                file=sys.stderr,
            )
        return False

    label_str = f" ({label})" if label else ""

    if dry_run:
        if verbosity >= 1:
            print(
                f"  {doc['_id'][:16]}{label_str} -> {dev_id[:16]} "
                f"\"{dev_title[:40]}\"",
                file=sys.stderr,
            )
        return True

    # Re-read to get latest _rev
    fresh_doc = db_train[doc['_id']]
    fresh_doc['skol_dev_id'] = dev_id
    db_train.save(fresh_doc)

    if verbosity >= 1:
        print(
            f"  {doc['_id'][:16]}{label_str} -> {dev_id[:16]} "
            f"\"{dev_title[:40]}\"",
            file=sys.stderr,
        )
    return True


if __name__ == "__main__":
    main()
