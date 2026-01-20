#!/bin/bash
# Build Debian package for skol-django using fpm
#
# Prerequisites:
#   sudo apt install ruby ruby-dev build-essential python3-venv python3-pip python3-build nodejs npm
#   sudo gem install fpm
#
# Usage:
#   ./build-deb.sh

set -e

cd "$(dirname "$0")"

VERSION="0.1.0"
PACKAGE="skol-django"
WHEEL_DIR="/opt/skol/wheels"
SERVICE_DIR="/usr/share/skol-django"

echo "=== Building Debian package with fpm ==="

# Clean previous builds
rm -rf dist/ build/ *.egg-info deb_dist/ staging/

# Create output and staging directories
DJANGO_ROOT="/opt/skol/django"
mkdir -p deb_dist
mkdir -p staging${WHEEL_DIR}
mkdir -p staging${SERVICE_DIR}
mkdir -p staging${DJANGO_ROOT}

# Build the React frontend
echo "Building React PDF viewer..."
if [ -d "frontend" ]; then
    cd frontend
    if [ ! -d "node_modules" ]; then
        echo "Installing npm dependencies..."
        npm install
    fi
    echo "Running webpack build..."
    npm run build
    cd ..
else
    echo "Warning: frontend directory not found, skipping React build"
fi

# Build the wheel
echo "Building Python wheel..."
python3 -m build --wheel --outdir dist/

# Copy wheel to staging area
cp dist/*.whl staging${WHEEL_DIR}/

# Copy service file to staging area
cp debian/skol-django.service staging${SERVICE_DIR}/

# Copy templates and static directories
echo "Copying templates and static files..."
cp -r templates staging${DJANGO_ROOT}/
cp -r static staging${DJANGO_ROOT}/

# Build the deb using fpm from the staging directory
# --no-auto-depends prevents fpm from generating dependencies automatically
fpm -s dir -t deb \
    --name "$PACKAGE" \
    --version "$VERSION" \
    --license "GPL-3.0-or-later" \
    --description "Django web application for SKOL taxonomic search and user management" \
    --maintainer "La Monte Henry Piggy Yarroll <piggy@piggy.com>" \
    --url "https://github.com/piggyatbaqaqi/skol" \
    --category "python" \
    --architecture all \
    --no-auto-depends \
    --depends python3.13 \
    --depends python3.13-venv \
    --depends skol \
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
