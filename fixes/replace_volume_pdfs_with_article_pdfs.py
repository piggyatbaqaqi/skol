#!/usr/bin/env python3
"""
Replace whole-volume PDFs in skol_training with individual article PDFs from skol_dev.

For Mycotaxon volumes 117-119, the training database has whole-volume PDFs (~33 MB)
shared across all articles. skol_dev has individual article PDFs (~0.3-3 MB each).

This script matches training docs to skol_dev docs by section number (sN.txt.ann)
mapped to page-order position, then replaces the article.pdf attachment.

Usage:
    python fixes/replace_volume_pdfs_with_article_pdfs.py --dry-run
    python fixes/replace_volume_pdfs_with_article_pdfs.py
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from env_config import get_env_config

VOLUMES = ['117', '118', '119']


def build_dev_page_order(db_dev, volume):
    """Get skol_dev articles for a volume, sorted by starting page number.

    Returns list of doc IDs in page order.
    """
    results = db_dev.find({
        'selector': {
            'journal': {'$eq': 'Mycotaxon'},
            'volume': {'$in': [volume]},
        },
        'limit': 200,
        'fields': ['_id', 'pages', 'title'],
    })

    articles = []
    for doc in results:
        pages = doc.get('pages', '') or ''
        start = 0
        if '-' in pages:
            # Handle roman numerals for front matter
            try:
                start = int(pages.split('-')[0])
            except ValueError:
                start = -1  # Front matter sorts first
        elif pages:
            try:
                start = int(pages)
            except ValueError:
                start = -1
        articles.append((start, doc['_id'], doc.get('title', '')))

    articles.sort()
    return articles


def get_section_number(source_file):
    """Extract section number from source_file path.

    'data/annotated/journals/Mycotaxon/Vol118/s6.txt.ann' -> 6
    Returns None if not a section file.
    """
    basename = Path(source_file).stem.replace('.txt', '')  # 's6' from 's6.txt.ann'
    m = re.match(r'^s(\d+)$', basename)
    if m:
        return int(m.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Replace whole-volume PDFs with individual article PDFs.",
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

    replaced = 0
    skipped = 0
    errors = 0

    for volume in VOLUMES:
        if verbosity >= 1:
            print(f"\nProcessing Mycotaxon Vol {volume}...", file=sys.stderr)

        # Build page-order index from skol_dev
        dev_articles = build_dev_page_order(db_dev, volume)
        if verbosity >= 1:
            print(
                f"  Found {len(dev_articles)} articles in skol_dev",
                file=sys.stderr,
            )

        # Get training docs for this volume
        train_docs = []
        for row in db_train.view('_all_docs', include_docs=True):
            if row.id.startswith('_design/'):
                continue
            doc = row.doc
            if doc.get('journal') != 'Mycotaxon' or doc.get('volume') != volume:
                continue
            if doc.get('file_type') != 'section':
                continue
            train_docs.append(doc)

        if verbosity >= 1:
            print(
                f"  Found {len(train_docs)} section docs in skol_training",
                file=sys.stderr,
            )

        if len(train_docs) != len(dev_articles):
            print(
                f"  WARNING: count mismatch! "
                f"{len(train_docs)} training vs {len(dev_articles)} dev",
                file=sys.stderr,
            )

        for doc in train_docs:
            source_file = doc.get('source_file', '')
            sec_num = get_section_number(source_file)
            if sec_num is None:
                if verbosity >= 1:
                    print(
                        f"  {doc['_id']}: can't parse section from {source_file}",
                        file=sys.stderr,
                    )
                skipped += 1
                continue

            if sec_num < 1 or sec_num > len(dev_articles):
                print(
                    f"  {doc['_id']}: section {sec_num} out of range "
                    f"(1-{len(dev_articles)})",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            # Section N maps to index N-1 in the sorted list
            _, dev_id, dev_title = dev_articles[sec_num - 1]

            if args.dry_run:
                if verbosity >= 1:
                    print(
                        f"  s{sec_num} ({doc['_id'][:12]}...) -> "
                        f"\"{dev_title[:50]}\" ({dev_id[:12]}...)",
                        file=sys.stderr,
                    )
                replaced += 1
                continue

            # Get PDF from skol_dev
            try:
                pdf_attachment = db_dev.get_attachment(dev_id, 'article.pdf')
                if pdf_attachment is None:
                    print(
                        f"  {doc['_id']}: no article.pdf in dev doc {dev_id}",
                        file=sys.stderr,
                    )
                    skipped += 1
                    continue

                pdf_content = pdf_attachment.read()

                # Re-read training doc to get latest _rev
                train_doc = db_train[doc['_id']]
                db_train.put_attachment(
                    train_doc,
                    pdf_content,
                    filename='article.pdf',
                    content_type='application/pdf',
                )
                replaced += 1

                if verbosity >= 1:
                    size_mb = len(pdf_content) / (1024 * 1024)
                    print(
                        f"  s{sec_num} ({doc['_id'][:12]}...): "
                        f"replaced with {size_mb:.1f} MB article PDF "
                        f"(\"{dev_title[:40]}\")",
                        file=sys.stderr,
                    )
            except Exception as e:
                errors += 1
                print(
                    f"  {doc['_id']}: ERROR: {e}",
                    file=sys.stderr,
                )

    if verbosity >= 1:
        prefix = "Would replace" if args.dry_run else "Replaced"
        print(f"\n{prefix} {replaced} whole-volume PDFs with article PDFs.", file=sys.stderr)
        if skipped:
            print(f"Skipped: {skipped}", file=sys.stderr)
        if errors:
            print(f"Errors: {errors}", file=sys.stderr)


if __name__ == "__main__":
    main()
