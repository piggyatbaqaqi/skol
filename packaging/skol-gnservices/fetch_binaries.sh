#!/usr/bin/env bash
# Fetch + extract gnfinder + gnparser release binaries from
# github.com/gnames.  Invoked from debian/rules during
# dpkg-buildpackage.  Idempotent: re-uses existing build/ artefacts.

set -euo pipefail

cd "$(dirname "$0")"

# Versions tracked here; bump them when updating to a newer upstream
# release.  Keep ``debian/changelog`` in sync (version stem
# ``<GNFINDER>+<GNPARSER>-<revision>``).
GNFINDER_VERSION="1.1.6"
GNPARSER_VERSION="1.15.0"

GNFINDER_URL="https://github.com/gnames/gnfinder/releases/download/v${GNFINDER_VERSION}/gnfinder-v${GNFINDER_VERSION}-linux.tar.gz"
GNPARSER_URL="https://github.com/gnames/gnparser/releases/download/v${GNPARSER_VERSION}/gnparser-v${GNPARSER_VERSION}-linux-x86.tar.gz"

mkdir -p build
cd build

if [ ! -x gnfinder ]; then
    echo ">>> Fetching gnfinder v${GNFINDER_VERSION}..."
    curl -fsSL "$GNFINDER_URL" -o gnfinder.tar.gz
    tar -xzf gnfinder.tar.gz
    rm -f gnfinder.tar.gz
    chmod 0755 gnfinder
fi

if [ ! -x gnparser ]; then
    echo ">>> Fetching gnparser v${GNPARSER_VERSION}..."
    curl -fsSL "$GNPARSER_URL" -o gnparser.tar.gz
    tar -xzf gnparser.tar.gz
    rm -f gnparser.tar.gz
    chmod 0755 gnparser
fi

echo ">>> Binaries staged in $(pwd):"
ls -la gnfinder gnparser
