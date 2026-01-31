# PDF Viewer Feature

The SKOL web application includes a React-based PDF viewer that allows users to view source PDFs directly from search results. The viewer uses a locally modified version of [react-pdf](https://github.com/wojtekmaj/react-pdf) by wojtekmaj, with added URL hash navigation support.

## Features

- View PDF attachments from CouchDB documents
- Page navigation (previous/next, jump to page)
- Zoom controls (zoom in/out, reset)
- Download PDF button
- Automatic navigation to the relevant page when viewing from taxa search results
- URL hash support (`#page=<label>`) for bookmarkable page navigation
- Support for PDF page labels (e.g., Roman numerals for front matter)

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

**Important:** The frontend requires the local react-pdf source to be present as a neighbor directory to skol:

```
piggyatbaqaqi/
├── skol/           # This repository
└── react-pdf/      # Local react-pdf fork with URL hash support
```

```bash
cd django/frontend
npm install --omit=optional  # Skip large optional deps
npm run build
```

This compiles react-pdf from TypeScript source and outputs:
- `static/js/pdf-viewer.bundle.js` - The bundled React app with react-pdf compiled in
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

Taxa documents in `skol_taxa_dev` or `skol_taxa_full_dev` have a `source` field:

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
- Local react-pdf fork as neighbor directory to skol (see Building the Frontend)

### Local Development

1. Clone react-pdf fork (if not already present):
   ```bash
   cd /path/to/piggyatbaqaqi
   git clone https://github.com/piggyatbaqaqi/react-pdf.git
   ```

2. Build the frontend:
   ```bash
   cd skol/django/frontend
   npm install --omit=optional
   npm run build
   ```

3. Run Django development server:
   ```bash
   cd django
   python manage.py runserver
   ```

4. For frontend development with auto-rebuild:
   ```bash
   cd django/frontend
   npm run watch
   ```

### Building the Debian Package

The `build-deb.sh` script automatically builds the React frontend before creating the package:

```bash
./build-deb.sh
```

This:
1. Verifies that react-pdf source is present as a neighbor directory
2. Runs `npm install --omit=optional` in the frontend directory
3. Runs `npm run build` to compile react-pdf from source and bundle the viewer
4. Builds the Python wheel and creates the .deb package
