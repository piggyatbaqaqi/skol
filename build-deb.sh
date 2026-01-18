#!/bin/bash
# Build Debian package for skol using fpm
#
# Prerequisites:
#   sudo apt install ruby ruby-dev build-essential python3-venv python3-pip python3-build
#   sudo gem install fpm
#
# Usage:
#   ./build-deb.sh

set -e

cd "$(dirname "$0")"

VERSION="0.1.0"
PACKAGE="skol"
WHEEL_DIR="/opt/skol/wheels"

echo "=== Building Debian package with fpm ==="

# Clean previous builds
rm -rf dist/ build/ *.egg-info deb_dist/ staging/

# Create output and staging directories
mkdir -p deb_dist
mkdir -p staging${WHEEL_DIR}
mkdir -p staging/opt/skol/bin

# Build the wheel
echo "Building Python wheel..."
python3 -m build --wheel --outdir dist/

# Copy wheel to staging area
cp dist/*.whl staging${WHEEL_DIR}/

# Copy bin/*.py scripts to staging area
echo "Copying bin scripts..."
cp bin/*.py staging/opt/skol/bin/

# Build the deb using fpm from the staging directory
# --no-auto-depends prevents fpm from generating dependencies automatically
fpm -s dir -t deb \
    --name "$PACKAGE" \
    --version "$VERSION" \
    --license "GPL-3.0-or-later" \
    --description "Taxonomic text classification and extraction pipeline for mycological literature" \
    --maintainer "La Monte Henry Piggy Yarroll <piggy@piggy.com>" \
    --url "https://github.com/piggyatbaqaqi/skol" \
    --category "python" \
    --architecture all \
    --no-auto-depends \
    --depends python3.13 \
    --depends python3.13-venv \
    --deb-user root \
    --deb-group root \
    --after-install debian/postinst \
    --before-remove debian/prerm \
    --package "deb_dist/${PACKAGE}_${VERSION}_all.deb" \
    -C staging \
    .

# Clean up staging
rm -rf staging/

echo "=== Done ==="
echo "Debian package created:"
ls -la deb_dist/*.deb
