#!/usr/bin/env python3
"""Watch deb files and install them when they change.

Usage:
    watch_install [--postinstall=CMD] [--interval=SECS] [--install-cmd=CMD] \
                  FILE... [-- INSTALL_ARGS...]

Example:
    watch_install --postinstall="systemctl restart skol-django" \
        ./skol_0.1.0_all.deb ./skol-django_0.1.0_all.deb -- --force-all
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


def get_mtime(filepath: Path) -> Optional[float]:
    """Get modification time of a file, or None if it doesn't exist."""
    try:
        return filepath.stat().st_mtime
    except FileNotFoundError:
        return None


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
    files: List[Path],
    install_cmd: str,
    install_args: List[str],
    postinstall: Optional[str],
    interval: float,
    verbosity: int = 1,
) -> None:
    """Watch files and install when they change.

    Args:
        files: List of deb file paths to watch
        install_cmd: Installation command
        install_args: Additional arguments for the install command
        postinstall: Optional shell command to run after installation
        interval: Check interval in seconds
        verbosity: Verbosity level
    """
    # Record initial mtimes (don't install on first run)
    mtimes: Dict[Path, Optional[float]] = {}
    for f in files:
        mtimes[f] = get_mtime(f)
        if verbosity >= 1:
            if mtimes[f] is not None:
                print(f"[watch_install] Watching: {f}")
            else:
                print(f"[watch_install] File not found (will watch): {f}")

    print(f"[watch_install] Watching {len(files)} file(s), "
          f"checking every {interval}s. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(interval)

            changed_files: List[Path] = []
            for f in files:
                current_mtime = get_mtime(f)

                # Skip if file doesn't exist
                if current_mtime is None:
                    if mtimes[f] is not None:
                        print(f"[watch_install] File disappeared: {f}")
                    mtimes[f] = None
                    continue

                # Check if mtime changed
                old_mtime = mtimes[f]
                if old_mtime is None:
                    # File appeared
                    print(f"[watch_install] File appeared: {f}")
                    changed_files.append(f)
                elif current_mtime > old_mtime:
                    # File was modified
                    changed_files.append(f)

                mtimes[f] = current_mtime

            # Install changed files
            if changed_files:
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
        type=Path,
        help='Deb files to watch',
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

    verbosity = 0 if parsed.quiet else parsed.verbose

    # Validate files exist (at least warn)
    missing = [f for f in parsed.files if not f.exists()]
    if missing and verbosity >= 1:
        for f in missing:
            print(f"[watch_install] Warning: File does not exist yet: {f}",
                  file=sys.stderr)

    watch_and_install(
        files=parsed.files,
        install_cmd=parsed.install_cmd,
        install_args=install_args,
        postinstall=parsed.postinstall,
        interval=parsed.interval,
        verbosity=verbosity,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
