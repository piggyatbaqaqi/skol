#!/bin/bash
# Build the Debian package for skol-gnservices using dpkg-buildpackage.
#
# Mirrors the .build-number pattern of ../build-deb.sh (skol) and
# ../../django/build-deb.sh (skol-django): the upstream version stem
# is hardcoded here, and the Debian revision auto-increments on each
# run from .build-number.
#
# The skol-gnservices package is built with debhelper rather than
# fpm, so the version is read from debian/changelog.  We rewrite the
# top entry's version in-place, build, then restore the original so
# the committed changelog stays clean.
#
# Prerequisites:
#   sudo apt install build-essential debhelper dpkg-dev
#
# Usage:
#   ./build-deb.sh

set -euo pipefail

cd "$(dirname "$0")"

PACKAGE="skol-gnservices"
# Upstream version stem: <gnfinder>+<gnparser>.  Bump when
# fetch_binaries.sh's GNFINDER_VERSION / GNPARSER_VERSION change.
BASE_VERSION="1.1.6+1.15.0"

# Build number management - increments with each build.
BUILD_NUMBER_FILE=".build-number"
if [ -f "$BUILD_NUMBER_FILE" ]; then
    BUILD_NUMBER=$(cat "$BUILD_NUMBER_FILE")
else
    BUILD_NUMBER=0
fi
BUILD_NUMBER=$((BUILD_NUMBER + 1))
echo "$BUILD_NUMBER" > "$BUILD_NUMBER_FILE"

FULL_VERSION="${BASE_VERSION}-${BUILD_NUMBER}"
echo "=== Building Debian package with dpkg-buildpackage (${PACKAGE} ${FULL_VERSION}) ==="

CHANGELOG="debian/changelog"
CHANGELOG_BAK="debian/changelog.bak"

# Restore the committed changelog on exit so re-running doesn't
# leave the working tree dirty (and so a failed build doesn't strand
# a half-rewritten changelog).
cleanup() {
    if [ -f "$CHANGELOG_BAK" ]; then
        mv "$CHANGELOG_BAK" "$CHANGELOG"
    fi
}
trap cleanup EXIT

cp "$CHANGELOG" "$CHANGELOG_BAK"

# Rewrite just the version on the first changelog line:
#   skol-gnservices (1.1.6+1.15.0-1) unstable; urgency=medium
# becomes
#   skol-gnservices (1.1.6+1.15.0-N) unstable; urgency=medium
sed -i -E "1s|^(${PACKAGE} \()[^)]+(\).*)|\1${FULL_VERSION}\2|" \
    "$CHANGELOG"

# Build the .deb.  -b = binary-only, -us -uc = unsigned,
# -tc = clean after.  Artefacts land one directory up (in
# packaging/), per dpkg-buildpackage's convention.
dpkg-buildpackage -b -us -uc -tc

echo "=== Done ==="
echo "Debian package created:"
ls -la "../${PACKAGE}_${FULL_VERSION}_"*.deb
