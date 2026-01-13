#!/usr/bin/env python3
"""Analyze page marker patterns in a document."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from env_config import get_env_config
import re

def analyze_markers(doc_id):
    """Analyze page marker patterns in a document."""
    config = get_env_config()

    import couchdb
    couchdb_url = f"http://{config['couchdb_host']}"
    server = couchdb.Server(couchdb_url)
    server.resource.credentials = (config['couchdb_username'], config['couchdb_password'])

    db = server['skol_dev']
    doc = db[doc_id]

    # Get .txt attachment
    txt_content = db.get_attachment(doc_id, 'article.txt').read().decode('utf-8')
    txt_lines = txt_content.split('\n')

    # Find all markers and analyze patterns
    marker_pattern = re.compile(r'^---\s*PDF\s+Page\s+(\d+)\s*---\s*$')

    markers = []
    for i, line in enumerate(txt_lines):
        match = marker_pattern.match(line)
        if match:
            page_num = int(match.group(1))
            markers.append((i, page_num, line))

    print(f"Total page markers: {len(markers)}")
    print(f"Total lines: {len(txt_lines)}")
    print()

    # Analyze adjacent markers (markers with no content between them)
    adjacent_count = 0
    adjacent_groups = []
    current_group = []

    for i, (line_num, page_num, marker) in enumerate(markers):
        if i == 0:
            current_group = [markers[i]]
            continue

        prev_line_num = markers[i-1][0]

        # Check if markers are adjacent (only blank lines between them)
        lines_between = txt_lines[prev_line_num + 1:line_num]
        non_blank_between = [l for l in lines_between if l.strip()]

        if len(non_blank_between) == 0:
            # Adjacent marker
            if len(current_group) == 1:
                adjacent_groups.append(current_group)
            current_group.append(markers[i])
            adjacent_count += 1
        else:
            # Not adjacent
            if len(current_group) > 1:
                adjacent_groups[-1] = current_group
            current_group = [markers[i]]

    if len(current_group) > 1:
        adjacent_groups.append(current_group)

    print(f"Adjacent markers (no content between): {adjacent_count}")
    print(f"Adjacent groups: {len(adjacent_groups)}")
    print()

    # Show first few adjacent groups
    if adjacent_groups:
        print("First 10 adjacent groups:")
        for i, group in enumerate(adjacent_groups[:10]):
            pages = [p[1] for p in group]
            print(f"  Group {i+1}: Pages {pages[0]}-{pages[-1]} ({len(pages)} adjacent markers)")

    # Analyze markers with minimal content
    minimal_content_count = 0
    for i, (line_num, page_num, marker) in enumerate(markers):
        if i == len(markers) - 1:
            # Last marker
            lines_after = txt_lines[line_num + 1:]
        else:
            next_marker_line = markers[i + 1][0]
            lines_after = txt_lines[line_num + 1:next_marker_line]

        non_blank_after = [l for l in lines_after if l.strip()]

        if len(non_blank_after) <= 2:
            minimal_content_count += 1

    print()
    print(f"Markers with <=2 lines of content after: {minimal_content_count}")

if __name__ == '__main__':
    analyze_markers('0107a939a88652a9a196e32e84faa66b')
