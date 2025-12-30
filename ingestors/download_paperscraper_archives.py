#!/usr/bin/env python3
"""
Download all paperscraper archives.

This script downloads pre-built paper archives from paperscraper's data sources.
These archives contain metadata and potentially PDFs for open access papers.

Usage:
    ./download_paperscraper_archives.py [--output-dir DIR] [--list-only]
"""

import sys
import argparse
from pathlib import Path
import time

try:
    from paperscraper import get_dumps
    from paperscraper.get_dumps import QUERY_FN_DICT
except ImportError:
    print("ERROR: paperscraper library not found.", file=sys.stderr)
    print("Install with: pip install paperscraper", file=sys.stderr)
    sys.exit(1)


def list_available_archives():
    """
    List all available paperscraper archives.

    Returns:
        Dictionary of archive names to their query functions
    """
    print("=" * 80)
    print("Available paperscraper archives:")
    print("=" * 80)
    print()

    if not QUERY_FN_DICT:
        print("No archives found in QUERY_FN_DICT")
        return {}

    for idx, (name, fn) in enumerate(QUERY_FN_DICT.items(), 1):
        print(f"{idx}. {name}")
        print(f"   Function: {fn.__name__}")
        if fn.__doc__:
            doc_lines = fn.__doc__.strip().split('\n')
            print(f"   Description: {doc_lines[0]}")
        print()

    return QUERY_FN_DICT


def download_archive(name, output_dir, verbose=True):
    """
    Download a specific archive.

    Args:
        name: Name of the archive to download
        output_dir: Directory to save the archive
        verbose: Print progress information

    Returns:
        True if successful, False otherwise
    """
    if name not in QUERY_FN_DICT:
        print(f"ERROR: Archive '{name}' not found", file=sys.stderr)
        return False

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Downloading archive: {name}")
        print(f"Output directory: {output_path}")
        print()

    try:
        # Get the query function for this archive
        query_fn = QUERY_FN_DICT[name]

        # Download the archive
        start_time = time.time()

        if verbose:
            print(f"Fetching {name}...")

        # Call the query function
        # Different archives may have different interfaces
        try:
            result = query_fn(save_path=str(output_path))
        except TypeError:
            # Some functions might not take save_path
            result = query_fn()

        elapsed = time.time() - start_time

        if verbose:
            print(f"Download completed in {elapsed:.2f} seconds")
            print(f"Result: {result}")
            print()

        return True

    except Exception as e:
        print(f"ERROR downloading {name}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def download_all_archives(output_dir, verbose=True):
    """
    Download all available archives.

    Args:
        output_dir: Directory to save archives
        verbose: Print progress information

    Returns:
        Dictionary mapping archive names to success status
    """
    archives = QUERY_FN_DICT

    if not archives:
        print("No archives available to download")
        return {}

    results = {}
    total = len(archives)

    print("=" * 80)
    print(f"Downloading {total} archives to {output_dir}")
    print("=" * 80)
    print()

    for idx, name in enumerate(archives.keys(), 1):
        print(f"[{idx}/{total}] Processing: {name}")
        print("-" * 80)

        success = download_archive(name, output_dir, verbose=verbose)
        results[name] = success

        if not success:
            print(f"WARNING: Failed to download {name}")

        print()

        # Rate limiting between downloads
        if idx < total:
            time.sleep(1)

    # Print summary
    print("=" * 80)
    print("Download Summary")
    print("=" * 80)
    print()

    successful = sum(1 for s in results.values() if s)
    failed = total - successful

    print(f"Total archives: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()

    if failed > 0:
        print("Failed archives:")
        for name, success in results.items():
            if not success:
                print(f"  - {name}")
        print()

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download paperscraper archives',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available archives
  %(prog)s --list-only

  # Download all archives to current directory
  %(prog)s

  # Download to specific directory
  %(prog)s --output-dir /path/to/archives

  # Download specific archive
  %(prog)s --archive medrxiv --output-dir ./data

  # Quiet mode
  %(prog)s --quiet
        """
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('./paperscraper_archives'),
        help='Output directory for archives (default: ./paperscraper_archives)'
    )
    parser.add_argument(
        '--list-only', '-l',
        action='store_true',
        help='List available archives without downloading'
    )
    parser.add_argument(
        '--archive', '-a',
        type=str,
        help='Download specific archive by name'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    # List archives
    if args.list_only:
        list_available_archives()
        return 0

    # Download specific archive
    if args.archive:
        success = download_archive(
            args.archive,
            args.output_dir,
            verbose=not args.quiet
        )
        return 0 if success else 1

    # Download all archives
    results = download_all_archives(
        args.output_dir,
        verbose=not args.quiet
    )

    # Return non-zero if any failed
    failed = sum(1 for s in results.values() if not s)
    return 1 if failed > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
