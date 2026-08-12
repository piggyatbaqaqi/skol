#!/usr/bin/env python3
"""Watch deb files and install them when they change.

Arguments are shell-glob PATTERNS, re-expanded on every poll.  That is
what lets this install a package that does not exist yet: build-deb.sh
increments .build-number on every build, so each build lands under a
brand-new filename.  Quote the patterns so your shell does not expand
them once at launch and freeze the list to the files already present.

Usage:
    watch_install [--postinstall=CMD] [--interval=SECS] [--install-cmd=CMD] \
                  PATTERN... [-- INSTALL_ARGS...]

Example:
    watch_install --postinstall="systemctl restart skol-django" \
        './skol_*_all.deb' './skol-django_*_all.deb' -- --force-all
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Per-pattern memory of the newest match we have already seen:
# pattern -> (path, mtime), or None when the pattern matches nothing.
WatchState = Dict[str, Optional[Tuple[Path, float]]]


def get_mtime(filepath: Path) -> Optional[float]:
    """Get modification time of a file, or None if it doesn't exist."""
    try:
        return filepath.stat().st_mtime
    except FileNotFoundError:
        return None


def newest_match(pattern: str) -> Optional[Path]:
    """Most recently modified file matching `pattern`, or None.

    "Newest" is by mtime, not by version string: the point is to catch
    whatever just landed, and a deliberately copied-in older build
    should still install.  A pattern with no wildcard is simply a
    pattern that matches at most one file, so explicit filenames keep
    working exactly as before.
    """
    candidates = []
    for name in glob.glob(pattern):
        path = Path(name)
        mtime = get_mtime(path)
        # Skip anything that vanished between glob and stat.
        if mtime is not None and path.is_file():
            candidates.append((mtime, path))
    if not candidates:
        return None
    return max(candidates)[1]


def scan_once(
    patterns: List[str],
    state: WatchState,
) -> Tuple[List[Path], WatchState]:
    """One poll: re-expand every pattern and report what to install.

    Returns (files_to_install, new_state).  A pattern contributes a
    file when its newest match is one we have not installed before --
    either a filename we have never seen (a new build arrived) or the
    same filename with a newer mtime (rebuilt in place).

    Patterns absent from `state` are being seen for the first time, so
    their current match is recorded as the baseline and NOT installed;
    starting the watcher must not reinstall what is already there.
    """
    changed: List[Path] = []
    new_state: WatchState = {}

    for pattern in patterns:
        current = newest_match(pattern)
        seen_before = pattern in state
        previous = state.get(pattern)

        if current is None:
            new_state[pattern] = None
            continue

        mtime = get_mtime(current)
        if mtime is None:
            new_state[pattern] = previous
            continue

        new_state[pattern] = (current, mtime)

        if not seen_before:
            continue  # baseline only
        if previous is None:
            changed.append(current)          # first file for this pattern
        elif previous[0] != current:
            changed.append(current)          # a new build arrived
        elif mtime > previous[1]:
            changed.append(current)          # rebuilt in place

    return changed, new_state


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
        print(f"[watch_install] Installing: {' '.join(str(f) for f in files)}")
        if verbosity >= 2:
            print(f"[watch_install] Command: {' '.join(cmd_parts)}")

    try:
        result = subprocess.run(cmd_parts, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[watch_install] Installation failed with exit code {e.returncode}",
              file=sys.stderr)
        return False
    except FileNotFoundError as e:
        print(f"[watch_install] Command not found: {e}", file=sys.stderr)
        return False

    # Run postinstall command if specified
    if postinstall:
        if verbosity >= 1:
            print(f"[watch_install] Running postinstall: {postinstall}")
        try:
            subprocess.run(postinstall, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[watch_install] Postinstall failed with exit code {e.returncode}",
                  file=sys.stderr)
            return False
        if verbosity >= 1:
            print(f"[watch_install] Postinstall completed successfully.")

    return True


def watch_and_install(
    patterns: List[str],
    install_cmd: str,
    install_args: List[str],
    postinstall: Optional[str],
    interval: float,
    verbosity: int = 1,
) -> None:
    """Watch glob patterns and install the newest match as it appears.

    Args:
        patterns: Shell-glob patterns, re-expanded on every poll
        install_cmd: Installation command
        install_args: Additional arguments for the install command
        postinstall: Optional shell command to run after installation
        interval: Check interval in seconds
        verbosity: Verbosity level
    """
    # Baseline pass: record what is already there without installing it.
    state = scan_once(patterns, {})[1]
    if verbosity >= 1:
        for pattern in patterns:
            current = state.get(pattern)
            if current is not None:
                print(f"[watch_install] Watching {pattern} "
                      f"(newest now: {current[0].name})")
            else:
                print(f"[watch_install] Watching {pattern} "
                      f"(no match yet)")

    print(f"[watch_install] Watching {len(patterns)} pattern(s), "
          f"checking every {interval}s. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(interval)

            changed_files, state = scan_once(patterns, state)

            if changed_files:
                if verbosity >= 1:
                    for f in changed_files:
                        print(f"[watch_install] Detected: {f}")
                install_packages(
                    changed_files,
                    install_cmd,
                    install_args,
                    postinstall,
                    verbosity,
                )

    except KeyboardInterrupt:
        print("\n[watch_install] Stopped.")


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
        description='Watch deb files and install them when they change.',
        epilog='Use -- to pass additional arguments to the install command.',
    )
    parser.add_argument(
        'files',
        nargs='+',
        metavar='PATTERN',
        help=(
            'Deb files to watch. Shell-glob patterns are re-expanded on '
            'every poll, so quote them ("./skol_*_all.deb") to keep your '
            'shell from expanding once at launch and missing the build '
            'that has not happened yet.'
        ),
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

    # Warn about patterns matching nothing yet.  This is legitimate --
    # you normally start the watcher before the build finishes -- but a
    # pattern that never matches is usually a typo, so say so once.
    unmatched = [p for p in parsed.files if newest_match(p) is None]
    if unmatched and verbosity >= 1:
        for p in unmatched:
            print(f"[watch_install] Note: no file matches yet: {p}",
                  file=sys.stderr)

    watch_and_install(
        patterns=parsed.files,
        install_cmd=parsed.install_cmd,
        install_args=install_args,
        postinstall=parsed.postinstall,
        interval=parsed.interval,
        verbosity=verbosity,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
