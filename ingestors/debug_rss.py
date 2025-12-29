#!/usr/bin/env python3
"""
Debug script to inspect RSS feed structure.

Usage:
    ./debug_rss.py https://www.mdpi.com/rss/journal/jof
"""

import sys
import feedparser
import json


def inspect_feed(url):
    """Inspect and display RSS feed structure."""
    print(f"Fetching and parsing: {url}")
    print("=" * 80)

    feed = feedparser.parse(url)

    print("\nTop-level attributes:")
    print("-" * 80)
    for attr in dir(feed):
        if not attr.startswith('_'):
            print(f"  {attr}")

    print("\n\nfeed object type:", type(feed))
    print("feed object keys:", list(feed.keys()) if hasattr(feed, 'keys') else "N/A")

    # Try to access feed metadata different ways
    print("\n\nTrying different metadata paths:")
    print("-" * 80)

    paths = [
        'feed',
        'channel',
        'header',
        'info',
    ]

    for path in paths:
        try:
            obj = getattr(feed, path)
            print(f"\n✓ feed.{path} exists:")
            print(f"  Type: {type(obj)}")
            if hasattr(obj, 'keys'):
                print(f"  Keys: {list(obj.keys())}")

            # Try to get title
            if hasattr(obj, 'title'):
                print(f"  Title: {obj.title}")
            elif hasattr(obj, 'get'):
                print(f"  Title (via get): {obj.get('title', 'N/A')}")
        except AttributeError as e:
            print(f"\n✗ feed.{path} does not exist: {e}")

    # Check if feed.feed exists and what's in it
    if hasattr(feed, 'feed'):
        print("\n\nfeed.feed contents:")
        print("-" * 80)
        feed_obj = feed.feed
        if hasattr(feed_obj, '__dict__'):
            for key, value in feed_obj.__dict__.items():
                if not key.startswith('_'):
                    val_str = str(value)[:100] if value else 'None'
                    print(f"  {key}: {val_str}")
        elif hasattr(feed_obj, 'keys'):
            for key in feed_obj.keys():
                val_str = str(feed_obj[key])[:100]
                print(f"  {key}: {val_str}")

    # Show first entry if available
    if hasattr(feed, 'entries') and len(feed.entries) > 0:
        print("\n\nFirst entry sample:")
        print("-" * 80)
        entry = feed.entries[0]
        if hasattr(entry, 'title'):
            print(f"  Title: {entry.title}")
        if hasattr(entry, 'link'):
            print(f"  Link: {entry.link}")
        if hasattr(entry, 'keys'):
            print(f"  Keys: {list(entry.keys())[:20]}")  # First 20 keys

    # Raw feed info
    print("\n\nRaw feed structure (feed dict keys):")
    print("-" * 80)
    if isinstance(feed, dict):
        for key in feed.keys():
            print(f"  {key}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: ./debug_rss.py <RSS_URL>")
        print("\nExample:")
        print("  ./debug_rss.py https://www.mdpi.com/rss/journal/jof")
        sys.exit(1)

    url = sys.argv[1]
    inspect_feed(url)
