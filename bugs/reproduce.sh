#!/usr/bin/env bash
# Replay every Plazi searchByDOI failure listed in failure.csv.
# Usage: ./reproduce.sh [failure.csv]
set -uo pipefail
CSV="${1:-failure.csv}"
UA="skol-plazi-backfill/1.0 (https://synoptickeyof.life)"
python3 - "$CSV" <<'PY' | while IFS=$'\t' read -r doc_id doi url; do
import csv
import sys
with open(sys.argv[1], newline='') as fh:
    for row in csv.DictReader(fh):
        print('\t'.join((row['doc_id'], row['doi'], row['url'])))
PY
    echo "== ${doc_id} (${doi})"
    curl -sS -A "$UA" \
        -w '\nHTTP %{http_code}  %{size_download} bytes\n' "$url" | head -c 600
    echo
    echo '---'
done
