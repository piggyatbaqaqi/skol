# SKOL Django Web Interface

A Django REST API and web interface for semantic search of taxonomic descriptions using SKOL embeddings stored in Redis.

## Features

- **REST API**: JSON endpoints for semantic search
- **Web Interface**: Clean, modern UI for searching taxa descriptions
- **Redis Integration**: Loads embeddings from Redis keys matching `skol:embedding:*`
- **Flexible Search**: Configurable number of results and embedding model selection
- **PDF Viewer**: React-based viewer for source PDFs (see [docs/PDF_VIEWER.md](docs/PDF_VIEWER.md))
- **CouchDB Integration**: Retrieves PDF attachments from CouchDB documents

## Architecture

The application consists of:

1. **Django REST API** (`search/views.py`):
   - `/api/embeddings/` - List available embedding models from Redis
   - `/api/search/` - Perform semantic search
   - `/api/taxa/<taxa_id>/` - Get taxa document info
   - `/api/taxa/<taxa_id>/pdf/` - Get PDF from taxa source
   - `/api/pdf/<db>/<doc_id>/` - Direct PDF attachment access
   - `/api/classifier/text/` - Text feature classifier (TF-IDF on descriptions)
   - `/api/classifier/json/` - JSON feature classifier (TF-IDF on structured annotations)

2. **Web Interface** (`templates/index.html`):
   - Input text box for description
   - Model selector (populated from Redis)
   - Number of results selector
   - Three result cards showing top matches with JSON export

3. **Integration** (`search/views.py`):
   - Uses `dr-drafts-mycosearch/src/sota_search.py:Experiment` class
   - Loads embeddings from Redis
   - Returns structured JSON results

## Prerequisites

- Python 3.8+ (tested on 3.10, 3.11, 3.12, 3.13)
- Redis server running (with SKOL embeddings)
- Django and dependencies (see requirements.txt)
- `dr-drafts-mycosearch` repository in correct location (see Directory Structure below)

## Directory Structure

The code expects this directory structure:
```
github.com/piggyatbaqaqi/
├── skol/
│   ├── django/              ← This Django project
│   │   ├── manage.py
│   │   ├── search/
│   │   └── skolweb/
│   └── bin/
└── dr-drafts-mycosearch/    ← ML search library
    └── src/
        ├── sota_search.py
        └── data.py
```

The Django app adds `dr-drafts-mycosearch/` to the Python path so it can import `from src.sota_search import Experiment`.

## Installation

1. Install dependencies:
```bash
cd skol/django
pip install -r requirements.txt
```

2. Set environment variables (optional):
```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

3. Run migrations (creates SQLite database for Django admin):
```bash
python manage.py migrate
```

4. Create a superuser (optional, for admin access):
```bash
python manage.py createsuperuser
```

## Running the Server

### Development Server

```bash
python manage.py runserver
```

The server will start at `http://127.0.0.1:8000/`

### Custom Host/Port

```bash
python manage.py runserver 0.0.0.0:8080
```

## API Endpoints

### List Embeddings

**GET** `/api/embeddings/`

Returns all available embedding models from Redis.

**Response:**
```json
{
    "embeddings": [
        "skol:embedding:v1.0",
        "skol:embedding:v1.1"
    ],
    "count": 2
}
```

### Search

**POST** `/api/search/`

Performs semantic search using the specified embedding model.

**Request:**
```json
{
    "prompt": "A small mushroom with white gills",
    "embedding_name": "skol:embedding:v1.1",
    "k": 3
}
```

**Response:**
```json
{
    "results": [
        {
            "Similarity": 0.95,
            "Title": "Agaricus campestris",
            "Description": "...",
            "Feed": "SKOL",
            "FeedID": "12345",
            ...
        },
        ...
    ],
    "count": 3,
    "prompt": "A small mushroom with white gills",
    "embedding_name": "skol:embedding:v1.1",
    "k": 3
}
```

### Text Feature Classifier

**POST** `/api/classifier/text/`

Trains a decision tree classifier on taxa descriptions using TF-IDF encoding and returns ranked feature importances. Uses the `skol_taxa_dev` CouchDB database.

**Request:**
```json
{
    "taxa_ids": ["taxon_001...", "taxon_002...", "taxon_003..."],
    "top_n": 30,
    "max_depth": 10,
    "min_df": 1,
    "max_df": 1.0
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `taxa_ids` | array | (required) | List of taxon document IDs to train on |
| `top_n` | int | 30 | Number of top features to return |
| `max_depth` | int | 10 | Maximum depth for decision tree and JSON export |
| `min_df` | int | 1 | Minimum document frequency for TF-IDF terms |
| `max_df` | float | 1.0 | Maximum document frequency fraction (0.0-1.0) for TF-IDF terms |

**Response:**
```json
{
    "features": [
        {
            "name": "brown",
            "importance": 0.25,
            "display_text": "brown"
        },
        {
            "name": "spores globose",
            "importance": 0.18,
            "display_text": "spores globose"
        }
    ],
    "metadata": {
        "n_classes": 5,
        "n_features": 1200,
        "tree_depth": 8,
        "taxa_count": 6
    },
    "tree_json": {
        "metadata": { "..." : "..." },
        "tree": { "..." : "..." }
    }
}
```

### JSON Feature Classifier

**POST** `/api/classifier/json/`

Trains a decision tree classifier on structured JSON annotations using TF-IDF on flattened key=value tokens. Uses the `skol_taxa_full_dev` CouchDB database.

**Request:** Same format as Text Feature Classifier.

**Response:**
```json
{
    "features": [
        {
            "name": "taxon_name_genus=Aspergillus",
            "importance": 0.30,
            "display_text": "taxon name genus Aspergillus"
        },
        {
            "name": "morphology_spore_shape=globose",
            "importance": 0.20,
            "display_text": "morphology spore shape globose"
        }
    ],
    "metadata": {
        "n_classes": 5,
        "n_features": 800,
        "tree_depth": 6,
        "taxa_count": 6
    },
    "tree_json": {
        "metadata": { "..." : "..." },
        "tree": { "..." : "..." }
    }
}
```

Note: For the JSON classifier, `display_text` converts the raw feature name from `key=value` format to natural language by replacing `=` and `_` with spaces.

**Error responses (both endpoints):**
- `400` - Missing or empty `taxa_ids`, or fewer than 2 valid documents
- `500` - Classifier training failure or CouchDB connection error

## Web Interface

Access the web interface at `http://127.0.0.1:8000/`

### Features:

1. **Description Input**: Large text area for entering taxonomic descriptions
2. **Model Selection**: Dropdown populated with embeddings from Redis
3. **Results Count**: Slider or input to select number of results (1-20)
4. **Results Display**:
   - Shows similarity score as percentage
   - Displays title, description, and metadata
   - Expandable JSON view for full result data

## Configuration

### Environment Variables

- `REDIS_HOST`: Redis server host (default: `localhost`)
- `REDIS_PORT`: Redis server port (default: `6379`)
- `COUCHDB_HOST`: CouchDB server host (default: `localhost`)
- `COUCHDB_PORT`: CouchDB server port (default: `5984`)
- `COUCHDB_USER`: CouchDB username (default: `admin`)
- `COUCHDB_PASSWORD`: CouchDB password
- `COUCHDB_URL`: Full CouchDB URL (overrides host:port)

### Django Settings

Edit `skolweb/settings.py` to customize:

- `ALLOWED_HOSTS`: Add your domain names
- `DEBUG`: Set to `False` in production
- `SECRET_KEY`: Change for production use

## Project Structure

```
django/
├── manage.py                 # Django management script
├── requirements.txt          # Python dependencies
├── build-deb.sh              # Debian package build script
├── README.md                 # This file
├── skolweb/                  # Django project
│   ├── __init__.py
│   ├── settings.py           # Django settings
│   ├── urls.py               # Root URL configuration
│   └── wsgi.py               # WSGI configuration
├── search/                   # Search app
│   ├── __init__.py
│   ├── admin.py              # Django admin (empty)
│   ├── apps.py               # App configuration
│   ├── models.py             # Models (none needed)
│   ├── urls.py               # App URL configuration
│   └── views.py              # REST API views
├── frontend/                 # React PDF viewer
│   ├── package.json          # npm dependencies
│   ├── webpack.config.js     # Webpack build config
│   └── src/                  # React source files
│       ├── index.js
│       ├── PDFViewer.jsx
│       └── styles.css
├── templates/                # HTML templates
│   ├── index.html            # Main search interface
│   └── pdf_viewer.html       # PDF viewer page
├── static/                   # Static files
│   └── js/                   # Built JavaScript (from frontend/)
└── docs/                     # Documentation
    ├── EMAIL_SETUP.md
    ├── HTTPS_SETUP.md
    └── PDF_VIEWER.md
```

## Production Deployment

For production deployment:

1. Set `DEBUG = False` in `settings.py`
2. Configure `ALLOWED_HOSTS` with your domain
3. Change `SECRET_KEY` to a secure random value
4. Use a production-ready web server (gunicorn, uWSGI)
5. Set up a reverse proxy (nginx, Apache)
6. Configure HTTPS/SSL

Example with gunicorn:
```bash
pip install gunicorn
gunicorn skolweb.wsgi:application --bind 0.0.0.0:8000
```

## Troubleshooting

### Python 3.11+ Compatibility
The application uses **two strategies** to support both older and newer Python versions:

1. **Lazy imports**: Heavy ML dependencies (sentence-transformers, transformers, TensorFlow) are only imported when the `/api/search/` endpoint is actually called, not at Django startup.

2. **Compatibility shim**: Provides `inspect.formatargspec()` for libraries that need it. This function was removed in Python 3.11 but is required by `wrapt` (used by TensorFlow).

**Benefits:**
- ✅ Works on Python 3.10 (uses native `formatargspec`)
- ✅ Works on Python 3.11+ (uses compatibility shim)
- ✅ Fast Django startup (doesn't load ML libraries)
- ✅ The `/api/embeddings/` endpoint works immediately
- ⚠️ The first `/api/search/` call may be slightly slower (1-2 seconds) as it loads the ML stack

**Technical details**: The shim monkey-patches `inspect.formatargspec` before importing `sota_search.Experiment`, which transitively imports TensorFlow → wrapt. The shim provides a minimal implementation that formats function signatures, which is all wrapt needs.

### No embeddings found
- Ensure Redis is running
- Check that embeddings exist in Redis: `redis-cli KEYS "skol:embedding:*"`
- Run `bin/embed_taxa.py` to create embeddings

### Module not found errors
- Ensure `dr-drafts-mycosearch` is in the correct location (parallel to `skol/` directory)
- The path setup adds `dr-drafts-mycosearch/` (not `dr-drafts-mycosearch/src/`) to Python path
- This allows `sota_search.py` to correctly import `from src import data`

### Search fails with "'DataFrame' object has no attribute 'filename'"
This error means the embeddings in Redis were created with an old version of `SKOL_TAXA` that didn't include required metadata columns ('source', 'filename', 'row').

**Fix**: Recreate the embeddings with the correct structure:
```bash
cd skol/django
./RECREATE_EMBEDDINGS.sh
```

Or manually:
```bash
redis-cli DEL "skol:embedding:v1.1"
cd ../bin
python3 embed_taxa.py --force --verbosity 2
```

### Other search failures
- Verify the embedding key exists in Redis
- Check Redis connection settings
- Review Django logs for detailed error messages

## Development

To extend the application:

1. **Add new API endpoints**: Edit `search/views.py` and `search/urls.py`
2. **Modify UI**: Edit `templates/index.html`
3. **Add CSS/JS**: Place files in `static/` directory
4. **Configure settings**: Edit `skolweb/settings.py`

## License

Part of the SKOL (Synoptic Key of Life) project.
