#!/usr/bin/env python3
"""
skol-switch-version - Switch between installed versions of SKOL packages

Usage:
    skol-switch-version <package> <version>
    skol-switch-version <package> --list

Examples:
    skol-switch-version skol 0.1.0-5
    skol-switch-version django 0.1.0-3
    skol-switch-version dr-drafts 0.1.0-2
    skol-switch-version skol --list
"""

import argparse
import os
import sys
from pathlib import Path


PACKAGE_CONFIG = {
    'skol': {
        'versions_dir': '/opt/skol/versions',
        'symlinks': [
            ('/opt/skol/bin', 'bin'),
            ('/opt/skol/advanced-databases', 'advanced-databases'),
        ],
    },
    'django': {
        'versions_dir': '/opt/skol/django-versions',
        'symlinks': [
            ('/opt/skol/django', ''),  # Empty string means link to version root
        ],
    },
    'dr-drafts': {
        'versions_dir': '/opt/dr-drafts/versions',
        'symlinks': [
            ('/opt/dr-drafts/mycosearch', 'mycosearch'),
        ],
    },
}


def list_versions(package: str) -> list[str]:
    """List available versions for a package."""
    config = PACKAGE_CONFIG.get(package)
    if not config:
        return []

    versions_dir = Path(config['versions_dir'])
    if not versions_dir.exists():
        return []

    return sorted([d.name for d in versions_dir.iterdir() if d.is_dir()])


def get_current_version(package: str) -> str | None:
    """Get the currently active version for a package."""
    config = PACKAGE_CONFIG.get(package)
    if not config:
        return None

    # Check the first symlink to determine current version
    symlink_path, subdir = config['symlinks'][0]
    symlink = Path(symlink_path)

    if not symlink.is_symlink():
        return None

    target = symlink.resolve()
    versions_dir = Path(config['versions_dir'])

    try:
        # The version is the directory name under versions_dir
        rel_path = target.relative_to(versions_dir)
        return rel_path.parts[0]
    except ValueError:
        return None


def switch_version(package: str, version: str) -> bool:
    """Switch to the specified version."""
    config = PACKAGE_CONFIG.get(package)
    if not config:
        print(f"Error: Unknown package '{package}'", file=sys.stderr)
        print(f"Available packages: {', '.join(PACKAGE_CONFIG.keys())}", file=sys.stderr)
        return False

    versions_dir = Path(config['versions_dir'])
    version_dir = versions_dir / version

    if not version_dir.exists():
        print(f"Error: Version '{version}' not found", file=sys.stderr)
        available = list_versions(package)
        if available:
            print(f"Available versions: {', '.join(available)}", file=sys.stderr)
        else:
            print(f"No versions installed for {package}", file=sys.stderr)
        return False

    # Update all symlinks
    for symlink_path, subdir in config['symlinks']:
        symlink = Path(symlink_path)
        if subdir:
            target = version_dir / subdir
        else:
            target = version_dir

        if not target.exists():
            print(f"Warning: Target {target} does not exist, skipping symlink {symlink_path}",
                  file=sys.stderr)
            continue

        # Remove existing symlink or backup existing directory
        if symlink.is_symlink():
            symlink.unlink()
        elif symlink.exists():
            # Backup existing directory
            backup = symlink.with_suffix(f'.old-{os.getpid()}')
            print(f"Warning: {symlink_path} is not a symlink, moving to {backup}")
            symlink.rename(backup)

        # Create parent directory if needed
        symlink.parent.mkdir(parents=True, exist_ok=True)

        # Create new symlink (use relative path for cleaner output but absolute for reliability)
        symlink.symlink_to(target)
        print(f"  {symlink_path} -> {target}")

    print(f"Switched {package} to version {version}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Switch between installed versions of SKOL packages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('package', choices=list(PACKAGE_CONFIG.keys()),
                        help='Package to switch (skol, django, dr-drafts)')
    parser.add_argument('version', nargs='?',
                        help='Version to switch to, or --list to list versions')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available versions')

    args = parser.parse_args()

    if args.list or args.version == '--list':
        versions = list_versions(args.package)
        current = get_current_version(args.package)

        if not versions:
            print(f"No versions installed for {args.package}")
            return 1

        print(f"Available versions for {args.package}:")
        for v in versions:
            marker = " (current)" if v == current else ""
            print(f"  {v}{marker}")
        return 0

    if not args.version:
        # Show current version
        current = get_current_version(args.package)
        if current:
            print(f"Current {args.package} version: {current}")
        else:
            print(f"No version currently active for {args.package}")
        return 0

    if not switch_version(args.package, args.version):
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
