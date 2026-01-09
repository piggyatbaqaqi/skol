#!/bin/bash
# Test script for SKOL Django REST API

BASE_URL="${1:-http://127.0.0.1:8000}"

echo "Testing SKOL Django REST API at $BASE_URL"
echo ""

# Test 1: List embeddings
echo "Test 1: GET /api/embeddings/"
echo "========================================"
curl -s "$BASE_URL/api/embeddings/" | python3 -m json.tool
echo ""
echo ""

# Test 2: Search (requires embedding to exist)
echo "Test 2: POST /api/search/"
echo "========================================"
EMBEDDING=$(curl -s "$BASE_URL/api/embeddings/" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['embeddings'][0] if data['embeddings'] else '')")

if [ -z "$EMBEDDING" ]; then
    echo "No embeddings found in Redis. Please run bin/embed_taxa.py first."
else
    echo "Using embedding: $EMBEDDING"
    echo ""
    curl -s -X POST "$BASE_URL/api/search/" \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"A small mushroom with white gills\",
            \"embedding_name\": \"$EMBEDDING\",
            \"k\": 2
        }" | python3 -m json.tool
fi

echo ""
echo "========================================"
echo "Tests complete!"
