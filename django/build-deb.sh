#!/bin/bash
# Build Debian package for skol-django using stdeb
#
# Prerequisites:
#   sudo apt install python3-stdeb python3-all debhelper dh-python
#
# Usage:
#   ./build-deb.sh

set -e

cd "$(dirname "$0")"

echo "=== Building source distribution ==="
python3 -m build --sdist

echo "=== Building Debian package with stdeb ==="
python3 setup.py --command-packages=stdeb.command sdist_dsc --with-python3=true bdist_deb

echo "=== Done ==="
echo "Debian packages are in deb_dist/"
ls -la deb_dist/*.deb 2>/dev/null || echo "No .deb files found - check for errors above"
