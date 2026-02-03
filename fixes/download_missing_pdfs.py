#!/usr/bin/env python3
"""
Download Missing PDF Attachments

This script scans CouchDB for documents that have a pdf_url but are missing
the article.pdf attachment, and attempts to download and attach the PDF.

Features:
- Resolves DOI redirects to actual PDF URLs
- Rate limiting per domain (respects robots.txt crawl-delay)
- Validates PDF magic bytes before attaching
- Records download_error on failures
- Skips documents that already have download_error (use --retry-errors to retry)
- Incremental mode saves after each download (crash-resistant)

Usage:
    python download_missing_pdfs.py --database skol_dev --dry-run
    python download_missing_pdfs.py --database skol_dev --limit 10
    python download_missing_pdfs.py --database skol_dev --retry-errors
    python download_missing_pdfs.py --database skol_dev --domain doi.org

Options:
    --database      CouchDB database name
    --doc-id        Process only this specific document ID
    --dry-run       Preview without downloading
    --limit N       Process at most N documents
    --domain        Only process documents with this domain in pdf_url
    --retry-errors  Retry documents that previously had download_error
    --incremental   Save after each download (default: True)
"""

import argparse
import random
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from env_config import get_env_config


# Rate limit tracking per domain
_domain_last_fetch: Dict[str, float] = {}
_domain_robots: Dict[str, Optional[RobotFileParser]] = {}

# Default rate limits (milliseconds)
DEFAULT_RATE_LIMIT_MIN_MS = 2000
DEFAULT_RATE_LIMIT_MAX_MS = 5000

# User agent for requests
USER_AGENT = "SKOL-PDFDownloader/1.0 (https://github.com/piggyatbaqaqi/skol; scholarly research)"


def get_robots_parser(domain: str) -> Optional[RobotFileParser]:
    """Get or create a robots.txt parser for a domain."""
    if domain not in _domain_robots:
        try:
            rp = RobotFileParser()
            rp.set_url(f"https://{domain}/robots.txt")
            rp.read()
            _domain_robots[domain] = rp
        except Exception:
            _domain_robots[domain] = None
    return _domain_robots[domain]


def get_crawl_delay(domain: str) -> int:
    """Get crawl delay for domain from robots.txt, or use default."""
    rp = get_robots_parser(domain)
    if rp:
        delay = rp.crawl_delay(USER_AGENT)
        if delay:
            return int(delay * 1000)  # Convert to milliseconds
    return random.randint(DEFAULT_RATE_LIMIT_MIN_MS, DEFAULT_RATE_LIMIT_MAX_MS)


def apply_rate_limit(domain: str) -> None:
    """Apply rate limiting for a domain."""
    now = time.time()
    if domain in _domain_last_fetch:
        delay_ms = get_crawl_delay(domain)
        elapsed_ms = (now - _domain_last_fetch[domain]) * 1000
        if elapsed_ms < delay_ms:
            sleep_time = (delay_ms - elapsed_ms) / 1000
            time.sleep(sleep_time)
    _domain_last_fetch[domain] = time.time()


def resolve_doi(doi_url: str, verbosity: int = 1) -> Optional[str]:
    """
    Resolve a DOI URL to the actual PDF URL.

    DOI URLs redirect to the publisher's page. We follow redirects
    to get the final URL, then try to construct the PDF URL.

    Returns:
        The resolved URL, or None if resolution failed
    """
    domain = urlparse(doi_url).netloc
    apply_rate_limit(domain)

    try:
        # Follow redirects to get the landing page
        response = requests.head(
            doi_url,
            headers={'User-Agent': USER_AGENT},
            allow_redirects=True,
            timeout=30
        )
        final_url = response.url

        if verbosity >= 3:
            print(f"    DOI resolved: {doi_url} -> {final_url}")

        return final_url

    except Exception as e:
        if verbosity >= 2:
            print(f"    DOI resolution failed: {e}")
        return None


def download_pdf(
    url: str,
    verbosity: int = 1,
    max_redirects: int = 5
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Download a PDF from a URL.

    Args:
        url: URL to download from
        verbosity: Verbosity level
        max_redirects: Maximum number of redirects to follow

    Returns:
        Tuple of (pdf_content, error_message)
        On success: (bytes, None)
        On failure: (None, error_string)
    """
    domain = urlparse(url).netloc

    # Check robots.txt
    rp = get_robots_parser(domain)
    if rp and not rp.can_fetch(USER_AGENT, url):
        return None, "Blocked by robots.txt"

    apply_rate_limit(domain)

    try:
        response = requests.get(
            url,
            headers={
                'User-Agent': USER_AGENT,
                'Accept': 'application/pdf,*/*',
            },
            allow_redirects=True,
            timeout=60,
            stream=True
        )

        if response.status_code == 403:
            return None, "HTTP 403 Forbidden"
        elif response.status_code == 404:
            return None, "HTTP 404 Not Found"
        elif response.status_code == 429:
            return None, "HTTP 429 Rate Limited"
        elif response.status_code != 200:
            return None, f"HTTP {response.status_code}"

        content = response.content

        # Validate PDF magic bytes
        if not content.startswith(b'%PDF'):
            # Check if we got HTML instead (common for paywalled articles)
            if content.startswith(b'<!DOCTYPE') or content.startswith(b'<html'):
                return None, "Got HTML instead of PDF (likely paywall)"
            preview = content[:20].hex() if len(content) >= 20 else content.hex()
            return None, f"Invalid PDF (starts with: {preview})"

        return content, None

    except requests.exceptions.Timeout:
        return None, "Request timeout"
    except requests.exceptions.ConnectionError as e:
        return None, f"Connection error: {type(e).__name__}"
    except Exception as e:
        return None, f"Download error: {type(e).__name__}: {e}"


def find_pdf_url_for_publisher(landing_url: str, verbosity: int = 1) -> Optional[str]:
    """
    Given a publisher landing page URL, try to find the PDF download URL.

    Different publishers have different URL patterns for PDFs.
    """
    parsed = urlparse(landing_url)
    domain = parsed.netloc
    path = parsed.path

    # Taylor & Francis (tandfonline.com)
    if 'tandfonline.com' in domain:
        # Landing: /doi/full/10.1080/xxx or /doi/abs/10.1080/xxx
        # PDF: /doi/pdf/10.1080/xxx
        if '/doi/full/' in path or '/doi/abs/' in path:
            pdf_path = path.replace('/doi/full/', '/doi/pdf/').replace('/doi/abs/', '/doi/pdf/')
            return f"https://{domain}{pdf_path}"

    # MDPI (mdpi.com)
    if 'mdpi.com' in domain:
        # Landing: /journal/volume/issue/article
        # PDF: /journal/volume/issue/article/pdf
        if not path.endswith('/pdf'):
            return f"{landing_url}/pdf"

    # Pensoft (pensoft.net)
    if 'pensoft.net' in domain:
        # Landing: /article/XXXXX/
        # PDF: /article/XXXXX/download/pdf/
        if '/article/' in path and '/download/pdf' not in path:
            article_path = path.rstrip('/')
            return f"https://{domain}{article_path}/download/pdf/"

    # Ingenta (ingentaconnect.com)
    if 'ingentaconnect.com' in domain:
        # Already has ?crawler=true for PDF download
        if '?crawler=true' not in landing_url:
            return f"{landing_url}?crawler=true"
        return landing_url

    # Mycosphere
    if 'mycosphere.org' in domain:
        # Try adding /pdf to the path
        if not path.endswith('.pdf'):
            return f"{landing_url}.pdf"

    # Default: return as-is
    return landing_url


def process_document(
    db,
    doc_id: str,
    dry_run: bool = False,
    verbosity: int = 1
) -> Tuple[bool, Optional[str]]:
    """
    Process a single document - download and attach PDF if missing.

    Returns:
        Tuple of (success, error_message)
    """
    try:
        doc = db[doc_id]
    except Exception as e:
        return False, f"Failed to fetch document: {e}"

    pdf_url = doc.get('pdf_url')
    if not pdf_url:
        return False, "No pdf_url in document"

    # Check if already has PDF
    attachments = doc.get('_attachments', {})
    if 'article.pdf' in attachments:
        return True, None  # Already has PDF

    if verbosity >= 2:
        print(f"  pdf_url: {pdf_url}")

    # Step 1: Resolve DOI if needed
    download_url = pdf_url
    parsed = urlparse(pdf_url)

    if parsed.netloc == 'doi.org':
        resolved = resolve_doi(pdf_url, verbosity)
        if not resolved:
            error = "Failed to resolve DOI"
            if not dry_run:
                doc['download_error'] = error
                db.save(doc)
            return False, error
        download_url = resolved

    # Step 2: Find the actual PDF URL for this publisher
    pdf_download_url = find_pdf_url_for_publisher(download_url, verbosity)

    if verbosity >= 2 and pdf_download_url != download_url:
        print(f"  PDF URL: {pdf_download_url}")

    if dry_run:
        print(f"  [DRY RUN] Would download from: {pdf_download_url}")
        return True, None

    # Step 3: Download the PDF
    content, error = download_pdf(pdf_download_url, verbosity)

    if error:
        # Save error to document
        try:
            fresh_doc = db[doc_id]
            fresh_doc['download_error'] = error
            db.save(fresh_doc)
        except Exception as save_e:
            if verbosity >= 2:
                print(f"  Warning: Could not save download_error: {save_e}")
        return False, error

    # Step 4: Attach the PDF
    try:
        fresh_doc = db[doc_id]
        db.put_attachment(
            fresh_doc,
            BytesIO(content),
            'article.pdf',
            'application/pdf'
        )

        # Clear any previous download_error
        fresh_doc = db[doc_id]
        if 'download_error' in fresh_doc:
            del fresh_doc['download_error']
            db.save(fresh_doc)

        if verbosity >= 2:
            print(f"  Attached PDF ({len(content)} bytes)")

        return True, None

    except Exception as e:
        return False, f"Failed to attach PDF: {e}"


def main():
    """Main entry point."""
    config = get_env_config()

    parser = argparse.ArgumentParser(
        description='Download missing PDF attachments from CouchDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--database',
        default=None,
        help='CouchDB database name'
    )

    parser.add_argument(
        '--doc-id',
        help='Process only this specific document ID'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview without downloading'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Process at most N documents'
    )

    parser.add_argument(
        '--domain',
        help='Only process documents with this domain in pdf_url'
    )

    parser.add_argument(
        '--retry-errors',
        action='store_true',
        help='Retry documents that previously had download_error'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed progress'
    )

    args, _ = parser.parse_known_args()

    database = args.database or config.get('ingest_database') or config.get('couchdb_database')
    if not database:
        parser.error("--database is required")

    verbosity = 3 if args.verbose else config.get('verbosity', 2)

    print(f"\n{'='*70}")
    print(f"Download Missing PDF Attachments")
    print(f"{'='*70}")
    print(f"Database: {database}")
    print(f"CouchDB: {config['couchdb_url']}")
    if args.domain:
        print(f"Domain filter: {args.domain}")
    if args.doc_id:
        print(f"Document: {args.doc_id}")
    if args.limit:
        print(f"Limit: {args.limit}")
    print(f"Retry errors: {args.retry_errors}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"{'='*70}\n")

    try:
        import couchdb

        server = couchdb.Server(config['couchdb_url'])
        if config['couchdb_username'] and config['couchdb_password']:
            server.resource.credentials = (config['couchdb_username'], config['couchdb_password'])

        if database not in server:
            print(f"Error: Database '{database}' not found")
            return 1

        db = server[database]

        # Find documents to process
        if args.doc_id:
            doc_ids = [args.doc_id]
        else:
            print("Scanning for documents missing article.pdf...")
            doc_ids = []
            for doc_id in db:
                try:
                    doc = db[doc_id]

                    # Must have pdf_url
                    if 'pdf_url' not in doc:
                        continue

                    # Skip if already has PDF
                    attachments = doc.get('_attachments', {})
                    if 'article.pdf' in attachments:
                        continue

                    # Skip if has download_error (unless --retry-errors)
                    if 'download_error' in doc and not args.retry_errors:
                        continue

                    # Apply domain filter
                    if args.domain:
                        pdf_url = doc.get('pdf_url', '')
                        if args.domain not in pdf_url:
                            continue

                    doc_ids.append(doc_id)

                    # Check limit
                    if args.limit and len(doc_ids) >= args.limit:
                        break

                except Exception:
                    continue

        if not doc_ids:
            print("No documents found that need PDF downloads")
            return 0

        print(f"Found {len(doc_ids)} document(s) to process\n")

        total_success = 0
        total_failed = 0

        for idx, doc_id in enumerate(doc_ids, 1):
            print(f"[{idx}/{len(doc_ids)}] {doc_id}")

            success, error = process_document(
                db=db,
                doc_id=doc_id,
                dry_run=args.dry_run,
                verbosity=verbosity
            )

            if success:
                total_success += 1
                if verbosity >= 2:
                    print("  OK")
            else:
                total_failed += 1
                print(f"  FAILED: {error}")

        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")
        print(f"Documents processed: {len(doc_ids)}")
        print(f"Successful: {total_success}")
        print(f"Failed: {total_failed}")
        if args.dry_run:
            print("\nThis was a DRY RUN - no PDFs were downloaded.")
        print()

        return 0 if total_failed == 0 else 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
