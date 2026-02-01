#!/usr/bin/env python3
"""Watch for new deb packages and install them incrementally.

Unlike watch_install which watches specific files for changes, this script
watches glob patterns and installs NEW files that appear. Files present at
startup are not installed automatically.

Usage:
    watch_incremental [--postinstall=CMD] [--interval=SECS] [--install-cmd=CMD] \
                      GLOB_PATTERN... [-- INSTALL_ARGS...]

Example:
    watch_incremental --postinstall="systemctl restart skol-django" \
        './skol_*_all.deb' './skol-django_*_all.deb' './dr-drafts-mycosearch_*_all.deb'

Each glob pattern represents one package. When new files matching a pattern
appear, the latest version (by filename sort) is installed.
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set


def get_matching_files(pattern: str) -> Set[Path]:
    """Get all files matching a glob pattern."""
    return {Path(f) for f in glob.glob(pattern)}


def get_latest_file(files: Set[Path]) -> Optional[Path]:
    """Get the latest file from a set (by name, which works for versioned debs).

    Debian package names like 'skol_0.1.0-5_all.deb' sort correctly by filename
    because version numbers are structured for lexicographic sorting.
    """
    if not files:
        return None
    return max(files, key=lambda f: f.name)


def install_packages(
    files: List[Path],
    install_cmd: str,
    install_args: List[str],
    postinstall: Optional[str],
    verbosity: int = 1,
) -> bool:
    """Install the specified deb files.

    Args:
        files: List of deb file paths to install
        install_cmd: Installation command (e.g., 'dpkg -i')
        install_args: Additional arguments for the install command
        postinstall: Optional shell command to run after installation
        verbosity: Verbosity level

    Returns:
        True if installation succeeded, False otherwise
    """
    # Build the install command
    cmd_parts = install_cmd.split()
    cmd_parts.extend(install_args)
    cmd_parts.extend(str(f) for f in files)

    if verbosity >= 1:
        print(f"[watch_incremental] Installing: {' '.join(str(f) for f in files)}")
        if verbosity >= 2:
            print(f"[watch_incremental] Command: {' '.join(cmd_parts)}")

    try:
        result = subprocess.run(cmd_parts, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[watch_incremental] Installation failed with exit code {e.returncode}",
              file=sys.stderr)
        return False
    except FileNotFoundError as e:
        print(f"[watch_incremental] Command not found: {e}", file=sys.stderr)
        return False

    # Run postinstall command if specified
    if postinstall:
        if verbosity >= 1:
            print(f"[watch_incremental] Running postinstall: {postinstall}")
        try:
            subprocess.run(postinstall, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[watch_incremental] Postinstall failed with exit code {e.returncode}",
                  file=sys.stderr)
            return False
        if verbosity >= 1:
            print(f"[watch_incremental] Postinstall completed successfully.")

    return True


def watch_and_install(
    patterns: List[str],
    install_cmd: str,
    install_args: List[str],
    postinstall: Optional[str],
    interval: float,
    verbosity: int = 1,
) -> None:
    """Watch glob patterns and install new packages as they appear.

    Args:
        patterns: List of glob patterns to watch (each represents one package)
        install_cmd: Installation command
        install_args: Additional arguments for the install command
        postinstall: Optional shell command to run after installation
        interval: Check interval in seconds
        verbosity: Verbosity level
    """
    # Record initial files for each pattern (don't install on first run)
    seen_files: Dict[str, Set[Path]] = {}

    for pattern in patterns:
        current_files = get_matching_files(pattern)
        seen_files[pattern] = current_files

        if verbosity >= 1:
            if current_files:
                latest = get_latest_file(current_files)
                print(f"[watch_incremental] Pattern '{pattern}': "
                      f"{len(current_files)} file(s), latest: {latest.name if latest else 'none'}")
            else:
                print(f"[watch_incremental] Pattern '{pattern}': no files yet")

    print(f"[watch_incremental] Watching {len(patterns)} pattern(s), "
          f"checking every {interval}s. Press Ctrl+C to stop.")
    print(f"[watch_incremental] Files present at startup will NOT be installed.")

    try:
        while True:
            time.sleep(interval)

            new_packages: List[Path] = []

            for pattern in patterns:
                current_files = get_matching_files(pattern)
                old_files = seen_files[pattern]

                # Find new files
                new_files = current_files - old_files

                if new_files:
                    # Get the latest new file for this pattern
                    latest_new = get_latest_file(new_files)
                    if latest_new:
                        if verbosity >= 1:
                            print(f"[watch_incremental] New package detected: {latest_new.name}")
                        new_packages.append(latest_new)

                    # Update seen files
                    seen_files[pattern] = current_files

            # Install new packages
            if new_packages:
                install_packages(
                    new_packages,
                    install_cmd,
                    install_args,
                    postinstall,
                    verbosity,
                )

    except KeyboardInterrupt:
        print("\n[watch_incremental] Stopped.")


def main() -> int:
    """Main entry point."""
    # Parse arguments manually to handle -- delimiter
    args = sys.argv[1:]

    # Find -- delimiter
    try:
        delimiter_idx = args.index('--')
        pre_delimiter = args[:delimiter_idx]
        install_args = args[delimiter_idx + 1:]
    except ValueError:
        pre_delimiter = args
        install_args = []

    # Parse pre-delimiter arguments
    parser = argparse.ArgumentParser(
        description='Watch for new deb packages and install them incrementally.',
        epilog='Use -- to pass additional arguments to the install command.',
    )
    parser.add_argument(
        'patterns',
        nargs='+',
        type=str,
        help='Glob patterns for deb files to watch (e.g., "./skol_*_all.deb")',
    )
    parser.add_argument(
        '--postinstall',
        type=str,
        default=None,
        help='Shell command to run after installing packages',
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=2.0,
        help='Check interval in seconds (default: 2)',
    )
    parser.add_argument(
        '--install-cmd',
        type=str,
        default='dpkg -i',
        help='Installation command (default: "dpkg -i")',
    )
    parser.add_argument(
        '-v', '--verbosity',
        action='count',
        default=1,
        help='Increase verbosity',
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress output',
    )

    parsed = parser.parse_args(pre_delimiter)

    verbosity = 0 if parsed.quiet else parsed.verbosity

    watch_and_install(
        patterns=parsed.patterns,
        install_cmd=parsed.install_cmd,
        install_args=install_args,
        postinstall=parsed.postinstall,
        interval=parsed.interval,
        verbosity=verbosity,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
