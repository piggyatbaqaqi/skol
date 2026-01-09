#!/bin/bash
# Script to fully restart Django server with cleared cache

echo "Stopping Django server..."
pkill -f "python.*manage.py runserver" 2>/dev/null
sleep 2

echo "Clearing Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

echo "Clearing Django cache..."
rm -rf db.sqlite3-journal 2>/dev/null

echo ""
echo "Cache cleared. Ready to start server."
echo ""
echo "To start the server with visible stderr output:"
echo "  python3 manage.py runserver"
echo ""
echo "Or use the convenience script:"
echo "  ./run_server.sh"
echo ""
