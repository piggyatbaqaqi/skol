#!/usr/bin/env python3
"""
Test script to verify search tracing works.
Run this to test the search endpoint with diagnostic output.
"""

import sys
import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'skolweb.settings')
django.setup()

# Now we can import Django components
from django.test import RequestFactory
from search.views import SearchView
import json

# Create a test request
factory = RequestFactory()
request = factory.post(
    '/api/search/',
    data=json.dumps({
        'prompt': 'A small mushroom with white gills',
        'embedding_name': 'skol:embedding:v1.1',
        'k': 2
    }),
    content_type='application/json'
)

print("=" * 70)
print("Testing Search API with Diagnostic Tracing")
print("=" * 70)
print("\nRequest:")
print(f"  Prompt: A small mushroom with white gills")
print(f"  Embedding: skol:embedding:v1.1")
print(f"  k: 2")
print("\nRunning search... (check stderr for diagnostics)")
print("-" * 70)

# Execute the view
view = SearchView.as_view()
response = view(request)

print("\nResponse Status:", response.status_code)
print("Response Data:")
if hasattr(response, 'data'):
    if 'error' in response.data:
        print(f"  Error: {response.data['error']}")
        if 'traceback' in response.data:
            print(f"\n  Traceback:\n{response.data['traceback']}")
    elif 'results' in response.data:
        print(f"  Success! Found {response.data['count']} results")
        for i, result in enumerate(response.data['results']):
            print(f"\n  Result {i+1}:")
            print(f"    Title: {result.get('Title', 'N/A')}")
            print(f"    Similarity: {result.get('Similarity', 'N/A'):.3f}")
else:
    print("  (No data)")

print("\n" + "=" * 70)
print("Test complete. Check output above for diagnostic messages.")
print("=" * 70)
