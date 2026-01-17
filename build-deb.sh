#!/bin/bash
# Build Debian package for skol using fpm
#
# Prerequisites:
#   sudo apt install ruby ruby-dev build-essential python3-venv
#   sudo gem install fpm
#
# Usage:
#   ./build-deb.sh

set -e

cd "$(dirname "$0")"

VERSION="0.1.0"
PACKAGE="skol"

echo "=== Building Debian package with fpm ==="

# Clean previous builds
rm -rf dist/ build/ *.egg-info deb_dist/

# Create output directory
mkdir -p deb_dist

# Build the deb using fpm
fpm -s python -t deb \
    --name "$PACKAGE" \
    --version "$VERSION" \
    --license "GPL-3.0-or-later" \
    --description "Taxonomic text classification and extraction pipeline for mycological literature" \
    --maintainer "La Monte Henry Piggy Yarroll <piggy@piggy.com>" \
    --url "https://github.com/piggyatbaqaqi/skol" \
    --category "python" \
    --python-bin python3 \
    --python-pip pip3 \
    --python-package-name-prefix python3 \
    --depends python3 \
    --depends python3-numpy \
    --depends python3-pandas \
    --depends python3-sklearn \
    --depends python3-redis \
    --depends python3-requests \
    --depends python3-tqdm \
    --deb-user root \
    --deb-group root \
    --after-install debian/postinst \
    --before-remove debian/prerm \
    --package "deb_dist/${PACKAGE}_${VERSION}_all.deb" \
    setup.py

echo "=== Done ==="
echo "Debian package created:"
ls -la deb_dist/*.deb
