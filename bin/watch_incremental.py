#!/usr/bin/env python3
"""Watch for new deb packages and install them incrementally.

Watches glob patterns and installs NEW files that appear.  Files present
at startup are recorded as a baseline and are not installed.

Quote the patterns so your shell does not expand them once at launch:
build-deb.sh increments .build-number on every build, so the package you
are waiting for has a filename that shell expansion could not know.

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
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def get_matching_files(pattern: str) -> Set[Path]:
    """Get all files matching a glob pattern."""
    return {Path(f) for f in glob.glob(pattern)}


def version_key(name: str) -> List[Any]:
    """Sort key that orders embedded numbers numerically.

    Deb filenames do NOT sort correctly as plain strings: '99' > '100'
    lexicographically, so a naive max() installs build 99 over build
    100.  Splitting into digit and non-digit runs and comparing the
    digit runs as ints fixes that, and handles the upstream version
    too ('0.10.0' > '0.9.0').

    Non-digit runs compare as strings, digit runs as (1, int) so a
    number always outranks a letter at the same position without
    comparing str to int.
    """
    parts: List[Any] = []
    for run in re.findall(r'\d+|\D+', name):
        if run.isdigit():
            parts.append((1, int(run)))
        else:
            parts.append((0, run))
    return parts


def get_latest_file(files: Set[Path]) -> Optional[Path]:
    """Get the newest-versioned file from a set.

    "Newest" is by version-aware filename ordering (see version_key),
    not mtime: the build number is the authority on which package is
    later, and copying an old deb back in should not outrank it.
    """
    if not files:
        return None
    return max(files, key=lambda f: version_key(f.name))


def heartbeat_due(now: float, last_emit: float, interval: float) -> bool:
    """Whether a heartbeat line is owed.  interval <= 0 disables it."""
    if interval <= 0:
        return False
    return (now - last_emit) >= interval


def _format_duration(seconds: float) -> str:
    """Compact human duration: 45s, 2m5s, 1h3m."""
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m{total % 60}s"
    return f"{total // 3600}h{(total % 3600) // 60}m"


def format_heartbeat(
    patterns: List[str],
    seen_files: Dict[str, Set[Path]],
    waited: float,
) -> str:
    """One line saying the watcher is alive and what it is holding.

    Printed while nothing is happening, so that a quiet terminal reads
    as "waiting" rather than "wedged".
    """
    parts = []
    for pattern in patterns:
        latest = get_latest_file(seen_files.get(pattern, set()))
        parts.append(f"{Path(pattern).name}="
                     f"{latest.name if latest else 'none yet'}")
    return (f"[watch_incremental] waiting {_format_duration(waited)} -- "
            + ", ".join(parts))


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
    heartbeat: float = 60.0,
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

    started = time.monotonic()
    last_beat = started

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
                # An install is its own proof of life; restart the clock
                # so the next heartbeat measures the new quiet period.
                last_beat = time.monotonic()
                continue

            now = time.monotonic()
            if verbosity >= 1 and heartbeat_due(now, last_beat, heartbeat):
                print(format_heartbeat(patterns, seen_files, now - started))
                last_beat = now

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
        '--heartbeat',
        type=float,
        default=60.0,
        help=(
            'Print a "still waiting" line every N seconds while nothing '
            'is happening, so a quiet watcher is distinguishable from a '
            'wedged one (default: 60; 0 disables)'
        ),
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
        heartbeat=parsed.heartbeat,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
