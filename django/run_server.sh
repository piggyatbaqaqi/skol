#!/bin/bash
# Start the SKOL Django development server

# Source skol environment if available (for Redis, CouchDB settings)
if [ -r /home/skol/.skol_env ]; then
    source /home/skol/.skol_env
    export REDIS_HOST REDIS_PORT REDIS_USERNAME REDIS_PASSWORD REDIS_TLS
    export COUCHDB_URL COUCHDB_USER COUCHDB_PASSWORD
fi

# Set default environment variables if not already set
export REDIS_HOST=${REDIS_HOST:-localhost}
export REDIS_PORT=${REDIS_PORT:-6380}

echo "Starting SKOL Django Web Interface..."
echo "Redis: $REDIS_HOST:$REDIS_PORT (TLS: ${REDIS_TLS:-false})"
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
