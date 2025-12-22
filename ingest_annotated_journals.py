#!/usr/bin/env python3
"""
Ingest annotated journal data into CouchDB skol_training database.

This script processes .txt.ann files from data/annotated/journals and creates
CouchDB documents with structure matching skol_dev database.
"""

import os
import re
import couchdb
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse


class JournalIngester:
    """Ingest annotated journal files into CouchDB."""

    def __init__(self, couchdb_url: str, username: str, password: str,
                 database_name: str = 'skol_training', verbosity: int = 1,
                 local_pdf_root: Optional[str] = None):
        """
        Initialize the ingester.

        Args:
            couchdb_url: CouchDB server URL
            username: CouchDB username
            password: CouchDB password
            database_name: Target database name
            verbosity: Logging level (0=silent, 1=info, 2=debug)
            local_pdf_root: Local filesystem path to PDF files (e.g., /data/skol/www/mykoweb.com)
        """
        self.verbosity = verbosity
        self.database_name = database_name
        self.local_pdf_root = Path(local_pdf_root) if local_pdf_root else None

        # Build authenticated URL
        auth_url = self._build_auth_url(couchdb_url, username, password)

        # Connect to CouchDB
        self.couch = couchdb.Server(auth_url)

        # Create or get database
        if database_name in self.couch:
            self.db = self.couch[database_name]
            if self.verbosity >= 1:
                print(f"Using existing database: {database_name}")
        else:
            self.db = self.couch.create(database_name)
            if self.verbosity >= 1:
                print(f"Created new database: {database_name}")

    def _build_auth_url(self, url: str, username: str, password: str) -> str:
        """Build authenticated CouchDB URL."""
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(url)
        netloc = f"{username}:{password}@{parsed.hostname}"
        if parsed.port:
            netloc += f":{parsed.port}"

        return urlunparse((
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment
        ))

    def parse_file_path(self, file_path: Path) -> Dict[str, str]:
        """
        Parse journal file path to extract metadata.

        Args:
            file_path: Path to .txt.ann file

        Returns:
            Dictionary with journal, volume, section/number, etc.
        """
        # Path structure: data/annotated/journals/JOURNAL/VolNNN/file.txt.ann
        parts = file_path.parts

        # Find 'journals' in path
        try:
            journals_idx = parts.index('journals')
        except ValueError:
            raise ValueError(f"Path does not contain 'journals': {file_path}")

        # Extract components
        journal = parts[journals_idx + 1]
        volume_dir = parts[journals_idx + 2]
        filename = parts[-1].replace('.txt.ann', '')

        # Parse volume number (e.g., "Vol054" -> "054")
        vol_match = re.match(r'Vol(\d+)', volume_dir)
        if not vol_match:
            raise ValueError(f"Cannot parse volume from: {volume_dir}")
        volume = vol_match.group(1)

        # Parse file type (n1, n2, s1, s2, etc.)
        # n = issue/number, s = section/article
        if filename.startswith('n'):
            file_type = 'issue'
            number = filename[1:]
        elif filename.startswith('s'):
            file_type = 'section'
            number = filename[1:]
        else:
            file_type = 'unknown'
            number = filename

        return {
            'journal': journal,
            'volume': volume,
            'file_type': file_type,
            'number': number,
            'filename': filename,
            'original_path': str(file_path)
        }

    def generate_pdf_url(self, metadata: Dict[str, str]) -> str:
        """
        Generate PDF URL based on journal and metadata.

        URL patterns (from user examples):
        - Mycotaxon/Vol054/n1 -> https://mykoweb.com/.../Mycotaxon/Mycotaxon%20v054.pdf
        - Persoonia/Vol016/n1 -> https://mykoweb.com/.../Persoonia/Persoonia%20v16n1.pdf

        Args:
            metadata: Parsed file metadata

        Returns:
            PDF URL string
        """
        journal = metadata['journal']
        volume = metadata['volume']
        file_type = metadata['file_type']
        number = metadata['number']

        base_url = "https://mykoweb.com/systematics/journals"

        if journal == 'Mycotaxon':
            # Mycotaxon: whole volume PDFs
            # Vol054 -> v054
            pdf_name = f"Mycotaxon%20v{volume}.pdf"
            return f"{base_url}/Mycotaxon/{pdf_name}"

        elif journal == 'Persoonia':
            # Persoonia: issue-level PDFs for "n" files
            # Vol016/n1 -> v16n1
            if file_type == 'issue':
                vol_num = volume.lstrip('0')  # Remove leading zeros
                pdf_name = f"Persoonia%20v{vol_num}n{number}.pdf"
                return f"{base_url}/Persoonia/{pdf_name}"
            else:
                # For section files, use volume PDF
                vol_num = volume.lstrip('0')
                pdf_name = f"Persoonia%20v{vol_num}.pdf"
                return f"{base_url}/Persoonia/{pdf_name}"

        elif journal == 'Mycologia':
            # Mycologia: need more info from user
            # For now, use generic pattern
            pdf_name = f"Mycologia%20v{volume}.pdf"
            return f"{base_url}/Mycologia/{pdf_name}"

        else:
            # Generic pattern
            pdf_name = f"{journal}%20v{volume}.pdf"
            return f"{base_url}/{journal}/{pdf_name}"

    def generate_doc_id(self, metadata: Dict[str, str]) -> str:
        """
        Generate unique document ID.

        Uses hash of journal/volume/file to create consistent IDs.
        """
        key = f"{metadata['journal']}/{metadata['volume']}/{metadata['filename']}"
        return hashlib.md5(key.encode()).hexdigest()

    def create_document(self, file_path: Path) -> Tuple[str, Dict]:
        """
        Create CouchDB document from annotation file.

        Args:
            file_path: Path to .txt.ann file

        Returns:
            Tuple of (doc_id, document_dict)
        """
        # Parse metadata
        metadata = self.parse_file_path(file_path)

        # Generate PDF URL
        pdf_url = self.generate_pdf_url(metadata)

        # Generate document ID
        doc_id = self.generate_doc_id(metadata)

        # Create document matching skol_dev schema
        doc = {
            '_id': doc_id,
            'meta': {},
            'pdf_url': pdf_url,
            'journal': metadata['journal'],
            'volume': metadata['volume'],
            'itemtype': 'article',
            # Fields we don't have - set to None or empty
            'author': None,
            'title': None,
            'number': metadata['number'] if metadata['file_type'] == 'issue' else None,
            'year': None,
            'issn': None,
            'eissn': None,
            'publication date': None,
            'pages': None,
            'url': pdf_url,
            'parent_itemid': None,
            'publishercode': None,
            # Add source metadata
            'source_file': metadata['original_path'],
            'file_type': metadata['file_type'],
        }

        return doc_id, doc

    def pdf_url_to_local_path(self, pdf_url: str) -> Optional[Path]:
        """
        Convert PDF URL to local filesystem path.

        Args:
            pdf_url: PDF URL (e.g., https://mykoweb.com/systematics/journals/Mycotaxon/Mycotaxon%20v054.pdf)

        Returns:
            Local filesystem path, or None if local_pdf_root not configured
        """
        if not self.local_pdf_root:
            return None

        from urllib.parse import urlparse, unquote

        # Parse URL to get path component
        parsed = urlparse(pdf_url)

        # URL path example: /systematics/journals/Mycotaxon/Mycotaxon%20v054.pdf
        # We need to:
        # 1. URL decode it (Mycotaxon%20v054.pdf -> Mycotaxon v054.pdf)
        # 2. Append to local_pdf_root

        # Remove leading slash and URL decode
        url_path = unquote(parsed.path.lstrip('/'))

        # Construct local path
        local_path = self.local_pdf_root / url_path

        return local_path

    def ingest_file(self, file_path: Path, overwrite: bool = False) -> Tuple[bool, bool]:
        """
        Ingest a single annotation file into CouchDB.

        Args:
            file_path: Path to .txt.ann file
            overwrite: If True, overwrite existing documents

        Returns:
            Tuple of (success, pdf_attached)
        """
        try:
            # Create document
            doc_id, doc = self.create_document(file_path)

            # Check if document exists
            if doc_id in self.db:
                if not overwrite:
                    if self.verbosity >= 2:
                        print(f"Skipping existing document: {doc_id} ({file_path.name})")
                    return True, False
                else:
                    # Get existing doc to preserve _rev
                    existing_doc = self.db[doc_id]
                    doc['_rev'] = existing_doc['_rev']
                    if self.verbosity >= 2:
                        print(f"Updating document: {doc_id} ({file_path.name})")

            # Save document
            self.db.save(doc)

            # Attach the .txt.ann file
            with open(file_path, 'rb') as f:
                content = f.read()
                self.db.put_attachment(
                    doc,
                    content,
                    filename='article.txt.ann',
                    content_type='text/plain'
                )

            # Attach PDF if available locally
            pdf_attached = False
            if self.local_pdf_root:
                pdf_path = self.pdf_url_to_local_path(doc['pdf_url'])
                if pdf_path and pdf_path.exists():
                    try:
                        with open(pdf_path, 'rb') as f:
                            pdf_content = f.read()
                            self.db.put_attachment(
                                doc,
                                pdf_content,
                                filename='article.pdf',
                                content_type='application/pdf'
                            )
                        pdf_attached = True
                        if self.verbosity >= 2:
                            print(f"  Attached PDF: {pdf_path.name} ({len(pdf_content):,} bytes)")
                    except Exception as e:
                        if self.verbosity >= 2:
                            print(f"  Warning: Could not attach PDF {pdf_path}: {e}")
                elif self.verbosity >= 2:
                    if pdf_path:
                        print(f"  PDF not found: {pdf_path}")
                    else:
                        print(f"  Could not map URL to local path: {doc['pdf_url']}")

            if self.verbosity >= 1:
                pdf_status = " [+PDF]" if pdf_attached else ""
                print(f"✓ Ingested{pdf_status}: {file_path.relative_to(file_path.parents[3])} -> {doc_id}")
                if self.verbosity >= 2 and not pdf_attached:
                    print(f"  PDF URL: {doc['pdf_url']}")

            return True, pdf_attached

        except Exception as e:
            print(f"✗ Error ingesting {file_path}: {e}")
            if self.verbosity >= 2:
                import traceback
                traceback.print_exc()
            return False, False

    def ingest_directory(self, base_dir: Path, overwrite: bool = False) -> Dict[str, int]:
        """
        Ingest all .txt.ann files from directory.

        Args:
            base_dir: Base directory containing journals/
            overwrite: If True, overwrite existing documents

        Returns:
            Statistics dictionary
        """
        # Find all .txt.ann files
        ann_files = sorted(base_dir.rglob('*.txt.ann'))

        # Filter out .txt.anns files (plural - backup files?)
        ann_files = [f for f in ann_files if not str(f).endswith('.txt.anns')]

        if self.verbosity >= 1:
            print(f"\nFound {len(ann_files)} annotation files")
            print("=" * 70)

        # Ingest each file
        stats = {
            'total': len(ann_files),
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'pdfs_attached': 0
        }

        for file_path in ann_files:
            success, pdf_attached = self.ingest_file(file_path, overwrite=overwrite)
            if success:
                stats['success'] += 1
                if pdf_attached:
                    stats['pdfs_attached'] += 1
            else:
                stats['failed'] += 1

        return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Ingest annotated journal data into CouchDB'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/annotated/journals'),
        help='Base directory containing annotated journals (default: data/annotated/journals)'
    )
    parser.add_argument(
        '--database',
        default='skol_training',
        help='CouchDB database name (default: skol_training)'
    )
    parser.add_argument(
        '--pdf-dir',
        type=Path,
        default=None,
        help='Local directory containing PDF files to attach (e.g., /data/skol/www/mykoweb.com)'
    )
    parser.add_argument(
        '--couchdb-url',
        default=os.environ.get('COUCHDB_URL', 'http://localhost:5984'),
        help='CouchDB server URL (default: from COUCHDB_URL env var)'
    )
    parser.add_argument(
        '--username',
        default=os.environ.get('COUCHDB_USER', 'admin'),
        help='CouchDB username (default: from COUCHDB_USER env var)'
    )
    parser.add_argument(
        '--password',
        default=os.environ.get('COUCHDB_PASSWORD', ''),
        help='CouchDB password (default: from COUCHDB_PASSWORD env var)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing documents'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=1,
        help='Increase verbosity (can be repeated: -v, -vv)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress output'
    )

    args = parser.parse_args()

    # Set verbosity
    verbosity = 0 if args.quiet else args.verbose

    # Create ingester
    ingester = JournalIngester(
        couchdb_url=args.couchdb_url,
        username=args.username,
        password=args.password,
        database_name=args.database,
        verbosity=verbosity,
        local_pdf_root=args.pdf_dir
    )

    # Ingest directory
    stats = ingester.ingest_directory(
        base_dir=args.data_dir,
        overwrite=args.overwrite
    )

    # Print summary
    if verbosity >= 1:
        print("=" * 70)
        print("\nIngestion Summary:")
        print(f"  Total files:           {stats['total']}")
        print(f"  Successfully ingested: {stats['success']}")
        print(f"  Failed:                {stats['failed']}")
        print(f"  Skipped:               {stats['skipped']}")
        if args.pdf_dir:
            print(f"  PDFs attached:         {stats['pdfs_attached']}")
        print(f"\nDatabase: {args.database}")
        print(f"Documents in database: {len(ingester.db)}")


if __name__ == '__main__':
    main()
