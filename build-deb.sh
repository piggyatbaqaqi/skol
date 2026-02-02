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
echo "=== Building Debian package with fpm (${PACKAGE} ${FULL_VERSION}) ==="

# Version-specific installation directory
VERSION_DIR="/opt/skol/versions/${FULL_VERSION}"

# Clean previous builds
rm -rf dist/ build/ *.egg-info deb_dist/ staging/

# Create output and staging directories
mkdir -p deb_dist
mkdir -p staging${WHEEL_DIR}
mkdir -p staging${VERSION_DIR}/bin

# Build the wheel
echo "Building Python wheel..."
python3 -m build --wheel --outdir dist/

# Copy wheel to staging area
cp dist/*.whl staging${WHEEL_DIR}/

# Copy bin/*.py scripts to staging area
echo "Copying bin scripts..."
cp bin/*.py staging${VERSION_DIR}/bin/
chmod 755 staging${VERSION_DIR}/bin/*.py

# Copy cron job to /etc/cron.d/
echo "Copying cron configuration..."
mkdir -p staging/etc/cron.d
cp debian/skol.cron staging/etc/cron.d/skol

# Copy ontology files
echo "Copying ontology files..."
mkdir -p staging/usr/share/skol/ontologies
cp data/ontologies/*.obo staging/usr/share/skol/ontologies/

# Copy advanced-databases directory (docker-compose, redis config, neo4j config)
# This removes the runtime dependency on the git repository
echo "Copying advanced-databases configuration..."
mkdir -p staging${VERSION_DIR}/advanced-databases
cp advanced-databases/docker-compose.yaml staging${VERSION_DIR}/advanced-databases/
cp advanced-databases/redis.conf staging${VERSION_DIR}/advanced-databases/
cp advanced-databases/redis-entrypoint.sh staging${VERSION_DIR}/advanced-databases/
chmod +x staging${VERSION_DIR}/advanced-databases/redis-entrypoint.sh

# Copy Neo4j config (static, read-only)
if [ -d advanced-databases/neo4j ]; then
    cp -a advanced-databases/neo4j staging${VERSION_DIR}/advanced-databases/
fi

# Copy CouchDB config templates (for initialization on first install)
# These go to /usr/share/skol for copying to /data/skol on first install
echo "Copying CouchDB config templates..."
mkdir -p staging/usr/share/skol/couchdb/etc/local.d
mkdir -p staging/usr/share/skol/couchdb/etc/default.d
if [ -d advanced-databases/couchdb/etc ]; then
    # Copy config files but exclude docker.ini (contains hashed passwords)
    cp -a advanced-databases/couchdb/etc/local.ini staging/usr/share/skol/couchdb/etc/ 2>/dev/null || true
    cp -a advanced-databases/couchdb/etc/default.ini staging/usr/share/skol/couchdb/etc/ 2>/dev/null || true
    cp -a advanced-databases/couchdb/etc/vm.args staging/usr/share/skol/couchdb/etc/ 2>/dev/null || true
    # Copy README but not docker.ini (which has runtime secrets)
    cp -a advanced-databases/couchdb/etc/local.d/README staging/usr/share/skol/couchdb/etc/local.d/ 2>/dev/null || true
fi

# Copy environment template file
echo "Copying environment template..."
cp skol_env.example staging${VERSION_DIR}/

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
