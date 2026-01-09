#!/bin/bash
# Script to recreate SKOL embeddings after data.py structure fix

echo "Recreating SKOL embeddings with correct structure"
echo "================================================="
echo ""

# Delete old embeddings
echo "1. Deleting old embeddings from Redis..."
redis-cli DEL "skol:embedding:v1.1"
echo "   ✓ Old embeddings deleted"
echo ""

# Create new embeddings
echo "2. Creating new embeddings with correct structure..."
echo "   This may take a few minutes..."
cd ../bin || exit 1
python3 embed_taxa.py --force --verbosity 2

echo ""
echo "================================================="
echo "Embeddings recreated successfully!"
echo ""
echo "You can now use the web interface to search."
echo "Restart the Django server if it's running:"
echo "  cd ../django"
echo "  ./run_server.sh"
