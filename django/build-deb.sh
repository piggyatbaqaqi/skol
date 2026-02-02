#!/bin/bash
# Build Debian package for skol-django using fpm
#
# Prerequisites:
#   sudo apt install ruby ruby-dev build-essential python3-venv python3-pip python3-build nodejs npm
#   sudo gem install fpm
#
# Usage:
#   ./build-deb.sh
#
# Note: This build compiles the local react-pdf from source using webpack/babel.
#       The react-pdf source is expected to be at ../../../react-pdf relative to django/frontend.

set -e

cd "$(dirname "$0")"

VERSION="0.1.0"
PACKAGE="skol-django"
WHEEL_DIR="/opt/skol/wheels"
SERVICE_DIR="/usr/share/skol-django"

# Build number management - increments with each build
BUILD_NUMBER_FILE=".build-number"
if [ -f "$BUILD_NUMBER_FILE" ]; then
    BUILD_NUMBER=$(cat "$BUILD_NUMBER_FILE")
else
    BUILD_NUMBER=0
fi
BUILD_NUMBER=$((BUILD_NUMBER + 1))
echo "$BUILD_NUMBER" > "$BUILD_NUMBER_FILE"

FULL_VERSION="${VERSION}-${BUILD_NUMBER}"

# Version-specific installation directory
VERSION_DIR="/opt/skol/django-versions/${FULL_VERSION}"

# Path to local react-pdf (neighbor directory to skol)
# From django, go up 2 levels to piggyatbaqaqi, then into react-pdf
REACT_PDF_DIR="$(cd "$(dirname "$0")/../.." && pwd)/react-pdf"

echo "=== Building Debian package with fpm (${PACKAGE} ${FULL_VERSION}) ==="

# Verify react-pdf source exists
if [ ! -d "$REACT_PDF_DIR/packages/react-pdf/src" ]; then
    echo "Error: react-pdf source directory not found at $REACT_PDF_DIR/packages/react-pdf/src"
    echo "Expected react-pdf to be a neighbor directory to skol"
    exit 1
fi
echo "Found react-pdf source at $REACT_PDF_DIR"

# Clean previous builds
rm -rf dist/ build/ *.egg-info deb_dist/ staging/

# Create output and staging directories
mkdir -p deb_dist
mkdir -p staging${WHEEL_DIR}
mkdir -p staging${SERVICE_DIR}
mkdir -p staging${VERSION_DIR}

# Build the React frontend (compiles react-pdf from source via webpack)
echo "Building React PDF viewer..."
cd "$(dirname "$0")"
if [ -d "frontend" ]; then
    cd frontend
    if [ ! -d "node_modules" ]; then
        echo "Installing npm dependencies (omitting optional deps for smaller install)..."
        npm install --omit=optional
    fi
    echo "Running webpack build (compiles react-pdf from source)..."
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

# Copy Django application files
echo "Copying Django application files..."
cp manage.py staging${VERSION_DIR}/
cp -r skolweb staging${VERSION_DIR}/
cp -r search staging${VERSION_DIR}/
cp -r accounts staging${VERSION_DIR}/
cp -r contact staging${VERSION_DIR}/
cp -r templates staging${VERSION_DIR}/
cp -r static staging${VERSION_DIR}/

# Remove __pycache__ directories and .pyc files
find staging${VERSION_DIR} -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find staging${VERSION_DIR} -type f -name "*.pyc" -delete 2>/dev/null || true

# Inject version into postinst and prerm templates
echo "Injecting version ${FULL_VERSION} into debian scripts..."
mkdir -p staging_scripts
sed "s/__VERSION__/${FULL_VERSION}/g" debian/postinst.template > staging_scripts/postinst
sed "s/__VERSION__/${FULL_VERSION}/g" debian/prerm.template > staging_scripts/prerm
chmod 755 staging_scripts/postinst staging_scripts/prerm

# Build the deb using fpm from the staging directory
# --no-auto-depends prevents fpm from generating dependencies automatically
fpm -s dir -t deb \
    --name "$PACKAGE" \
    --version "$FULL_VERSION" \
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
    --after-install staging_scripts/postinst \
    --before-remove staging_scripts/prerm \
    --package "deb_dist/${PACKAGE}_${FULL_VERSION}_all.deb" \
    -C staging \
    .

# Clean up staging
rm -rf staging/ staging_scripts/

echo "=== Done ==="
echo "Debian package created:"
ls -la deb_dist/*.deb
