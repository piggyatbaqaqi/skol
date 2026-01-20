# PDF Viewer Feature

The SKOL web application includes a React-based PDF viewer that allows users to view source PDFs directly from search results. The viewer uses [react-pdf](https://github.com/wojtekmaj/react-pdf) by wojtekmaj.

## Features

- View PDF attachments from CouchDB documents
- Page navigation (previous/next, jump to page)
- Zoom controls (zoom in/out, reset)
- Download PDF button
- Automatic navigation to the relevant page when viewing from taxa search results

## Architecture

### REST API Endpoints

The following API endpoints are available for PDF access:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/taxa/<taxa_id>/` | GET | Get taxa document info including source PDF details |
| `/api/pdf/<db_name>/<doc_id>/` | GET | Direct PDF attachment retrieval from CouchDB |
| `/api/pdf/<db_name>/<doc_id>/<attachment_name>/` | GET | Retrieve specific attachment by name |
| `/api/taxa/<taxa_id>/pdf/` | GET | Convenience endpoint - looks up taxa's source and returns PDF |

### Query Parameters

- `?download=true` - Force download instead of inline viewing
- `?taxa_db=<db_name>` - Specify taxa database (default: skol_taxa_dev)
- `?page=<n>` - Initial page number for the viewer

### Response Headers

- `X-PDF-Page` - Included when viewing from taxa endpoint, contains the PDF page number

## Frontend Components

The PDF viewer is built with React and bundled using webpack:

- `frontend/src/PDFViewer.jsx` - Main React component
- `frontend/src/index.js` - Entry point that mounts the viewer
- `frontend/src/styles.css` - Viewer styles
- `frontend/webpack.config.js` - Webpack configuration

### Building the Frontend

```bash
cd django/frontend
npm install
npm run build
```

This outputs:
- `static/js/pdf-viewer.bundle.js` - The bundled React app
- `static/js/pdf.worker.min.mjs` - PDF.js worker file

## Configuration

### CouchDB Settings

Add these environment variables to `/opt/skol/django/skol-django.env`:

```bash
# CouchDB configuration for PDF attachment retrieval
COUCHDB_HOST=localhost
COUCHDB_PORT=5984
COUCHDB_USER=admin
COUCHDB_PASSWORD=your_password

# Optional: override host:port with full URL
# COUCHDB_URL=http://localhost:5984
```

### Django Settings

The following settings are configured in `skolweb/settings.py`:

```python
COUCHDB_HOST = os.environ.get('COUCHDB_HOST', 'localhost')
COUCHDB_PORT = int(os.environ.get('COUCHDB_PORT', '5984'))
COUCHDB_USERNAME = os.environ.get('COUCHDB_USER', 'admin')
COUCHDB_PASSWORD = os.environ.get('COUCHDB_PASSWORD', '')
COUCHDB_URL = os.environ.get('COUCHDB_URL', f'http://{COUCHDB_HOST}:{COUCHDB_PORT}')
```

## URL Routes

| URL | View | Description |
|-----|------|-------------|
| `/pdf/` | pdf_viewer | Generic PDF viewer (requires `?db=&doc_id=` params) |
| `/pdf/taxa/<taxa_id>/` | pdf_viewer | View PDF for a taxa document |

## Usage

### From Search Results

When users perform a search, each result card now includes a "View PDF" button that opens the source PDF in a new tab. If the taxa document includes a `pdf_page` field, the viewer automatically navigates to that page.

### Direct Access

You can directly access PDFs using:

```
/pdf/taxa/taxon_abc123/?page=5
```

Or for direct document access:

```
/pdf/?db=skol_dev&doc_id=abc123&attachment=article.pdf
```

## Data Model

Taxa documents in `skol_taxa_dev` or `skol_taxa_full` have a `source` field:

```json
{
  "_id": "taxon_abc123",
  "taxon": "Species name",
  "description": "...",
  "source": {
    "db_name": "skol_dev",
    "doc_id": "source_doc_id",
    "pdf_url": "https://...",
    "human_url": "https://..."
  },
  "pdf_page": 35
}
```

The source document in `skol_dev` has the PDF as an attachment:

```json
{
  "_id": "source_doc_id",
  "_attachments": {
    "article.pdf": {
      "content_type": "application/pdf",
      "length": 3031635
    }
  }
}
```

## Development

### Prerequisites

- Node.js 18+ and npm
- Python 3.10+

### Local Development

1. Build the frontend:
   ```bash
   cd django/frontend
   npm install
   npm run build
   ```

2. Run Django development server:
   ```bash
   cd django
   python manage.py runserver
   ```

3. For frontend development with auto-rebuild:
   ```bash
   cd django/frontend
   npm run watch
   ```

### Building the Debian Package

The `build-deb.sh` script automatically builds the React frontend before creating the package:

```bash
./build-deb.sh
```

This runs `npm install` and `npm run build` in the frontend directory as part of the build process.
