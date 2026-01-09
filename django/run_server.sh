#!/bin/bash
# Start the SKOL Django development server

# Set default environment variables if not already set
export REDIS_HOST=${REDIS_HOST:-localhost}
export REDIS_PORT=${REDIS_PORT:-6379}

echo "Starting SKOL Django Web Interface..."
echo "Redis: $REDIS_HOST:$REDIS_PORT"
echo ""
echo "The server will be available at:"
echo "  http://127.0.0.1:8000/"
echo ""
echo "API endpoints:"
echo "  GET  http://127.0.0.1:8000/api/embeddings/"
echo "  POST http://127.0.0.1:8000/api/search/"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the Django development server
python3 manage.py runserver
