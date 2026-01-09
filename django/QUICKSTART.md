# SKOL Django Web Interface - Quick Start

## Prerequisites

1. **Redis server** running with SKOL embeddings
2. **Python packages** installed (Django, djangorestframework, etc.)

## Start the Server

```bash
cd skol/django
./run_server.sh
```

Or manually:
```bash
python3 manage.py runserver
```

## Access the Interface

Open your browser to: **http://127.0.0.1:8000/**

## Using the Web Interface

1. **Enter a description**: Type or paste a taxonomic description in the text box
2. **Select a model**: Choose an embedding model from the dropdown (populated from Redis)
3. **Set number of results**: Choose how many matches to return (1-20)
4. **Click Search**: Results will appear below with similarity scores
5. **View details**: Click "Show Full JSON" on any result to see complete data

## API Usage

### List Available Embeddings

```bash
curl http://127.0.0.1:8000/api/embeddings/
```

Response:
```json
{
    "embeddings": ["skol:embedding:v1.0", "skol:embedding:v1.1"],
    "count": 2
}
```

### Perform a Search

```bash
curl -X POST http://127.0.0.1:8000/api/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A small mushroom with white gills and a ring",
    "embedding_name": "skol:embedding:v1.1",
    "k": 3
  }'
```

Response:
```json
{
    "results": [
        {
            "Similarity": 0.856,
            "Title": "Agaricus campestris",
            "Description": "...",
            "Feed": "SKOL",
            ...
        }
    ],
    "count": 3,
    "prompt": "...",
    "embedding_name": "skol:embedding:v1.1",
    "k": 3
}
```

## Common Issues

### No embeddings found
Run the embedding tool to create embeddings:
```bash
cd ../bin
./embed_taxa.py
```

### Connection refused
Make sure Redis is running:
```bash
redis-cli ping
```

### Module not found
Install dependencies:
```bash
pip install -r requirements.txt
```

## Environment Variables

- `REDIS_HOST` - Redis server host (default: localhost)
- `REDIS_PORT` - Redis port (default: 6379)

Example:
```bash
export REDIS_HOST=192.168.1.100
export REDIS_PORT=6380
./run_server.sh
```

## Next Steps

- See [README.md](README.md) for full documentation
- Customize the UI by editing `templates/index.html`
- Add new API endpoints in `search/views.py`
- Configure for production deployment
