# InternetArchiveIngestor

Ingestor for downloading journal issues from the Internet Archive.

## Overview

The InternetArchiveIngestor downloads journal articles and issues from Internet Archive collections using the `internetarchive` Python library. It extracts both PDF files and OCR text (XML) when available.

## Supported Collections

The ingestor can work with any Internet Archive collection, but is primarily configured for:

- **Sydowia** (pub_sydowia): https://archive.org/details/pub_sydowia
  - 249 items spanning 1903-1968
  - Includes historical Annales Mycologici issues

## Requirements

```bash
pip install internetarchive
```

## Files Downloaded

For each item in the collection:

1. **PDF file**: Main article content (e.g., `sydowia_1903-01_1_1.pdf`)
   - Attached to CouchDB document as `article.pdf`
2. **XML file** (optional): OCR text in DJVU XML format (e.g., `sydowia_1903-01_1_1_djvu.xml`)
   - Attached to CouchDB document as `article.xml`
   - Only downloaded if `download_xml: True` is set in configuration

## Metadata Extracted

- `title`: Item title from Internet Archive metadata
- `volume`: Volume number
- `number`: Issue number (from Internet Archive 'issue' field)
- `publication_date`: Publication date in ISO format (YYYY-MM-DD)
- `year`: Publication year
- `identifier`: Internet Archive item identifier
- `url`: Human-readable URL (archive.org details page)
- `pdf_url`: Direct PDF download URL
- `xml_url`: Direct XML download URL (if available)
- `issn`: Journal ISSN (if configured)
- `publisher`: Publisher name (if available in metadata)

## Date Parsing

The ingestor handles various date formats from Internet Archive:
- `YYYY-MM-DD` → `YYYY-MM-DD`
- `YYYY-MM` → `YYYY-MM-01`
- `YYYY` → `YYYY-01-01`

## Usage

### Via Command Line

```bash
# Download all items from Sydowia collection
./ingestors/main.py --publication sydowia-ia --verbosity 2

# With verbosity levels:
#  0 = Silent (errors only)
#  1 = Basic progress
#  2 = Detailed progress (recommended)
#  3 = Full metadata and debugging
#  4 = Very verbose (includes internetarchive library output)
```

### Configuration in publications.py

```python
'sydowia-ia': {
    'name': 'Sydowia (Internet Archive)',
    'source': 'internet-archive',
    'ingestor_class': 'InternetArchiveIngestor',
    'mode': 'web',
    'collection': 'pub_sydowia',
    'download_xml': True,  # Set to False to skip XML downloads
    'issn': '0082-0598',
    'rate_limit_min_ms': 2000,
    'rate_limit_max_ms': 5000,
},
```

**Configuration Options:**
- `collection`: Internet Archive collection identifier (required)
- `download_xml`: Boolean, whether to download and attach XML files (default: True)
- `issn`: Journal ISSN (optional, added to document metadata)
- `rate_limit_min_ms`, `rate_limit_max_ms`: Rate limiting between requests

### Programmatic Usage

```python
from ingestors.internet_archive import InternetArchiveIngestor
import couchdb

# Connect to CouchDB
server = couchdb.Server('http://admin:password@localhost:5984')
db = server['skol_dev']

# Create ingestor
ingestor = InternetArchiveIngestor(
    collection='pub_sydowia',
    max_items=10,  # Limit for testing
    download_xml=True,
    db=db,
    verbosity=2
)

# Run ingestion
ingestor.ingest()
```

## Implementation Details

### Item Processing

For each item in the collection:

1. Query Internet Archive for item metadata
2. Identify main PDF file (excludes derivative files like `_bw.pdf`, `_text.pdf`)
3. Identify DJVU XML file (OCR text) if available
4. Download files to temporary directory
5. Extract metadata from Internet Archive
6. Ingest into CouchDB with file attachments
7. Clean up temporary files

### File Selection

The ingestor intelligently selects files:
- **PDF**: Main PDF file (e.g., `item_name.pdf`), excluding derivatives
- **XML**: DJVU XML file (e.g., `item_name_djvu.xml`) containing OCR text

### Duplicate Detection and Incremental Downloads

The ingestor intelligently handles existing documents:

1. **Checks existing attachments**: For each document, checks if PDF and/or XML are already attached
2. **Downloads only what's missing**:
   - If document has PDF but no XML (and `download_xml: True`), only downloads XML
   - If document has both attachments, skips entirely
3. **Preserves existing data**: Uses CouchDB's `_rev` to update documents without losing data

**Example behavior:**
- First run with `download_xml: False`: Downloads only PDFs
- Second run with `download_xml: True`: Downloads only XMLs for existing documents
- Third run: Skips all documents (already have everything)

### Error Handling

- Gracefully handles missing PDFs (skips item with warning)
- Continues processing if XML download fails
- Reports errors but continues with remaining items
- Provides detailed error messages at verbosity levels 2+

### Temporary File Management

- Downloads to temporary directory with unique prefix (`ia_ingest_*`)
- Automatically cleaned up after each item (success or failure)
- Uses Python's `tempfile.TemporaryDirectory` for safe cleanup

## Internet Archive Collection Structure

Internet Archive items have:
- **Identifier**: Unique ID (e.g., `sydowia_1903-01_1_1`)
- **Metadata**: Title, volume, issue, date, ISSN, etc.
- **Files**: Multiple formats (PDF, XML, DJVU, etc.)
- **URLs**:
  - Details page: `https://archive.org/details/{identifier}`
  - Download: `https://archive.org/download/{identifier}/{filename}`

## Rate Limiting

Respects configured rate limits:
- Default: 2000-5000ms between requests
- Configurable per publication in `publications.py`
- Applied by the base Ingestor class

## Adding New Collections

To add a new Internet Archive collection:

1. Find the collection identifier (e.g., from URL: `archive.org/details/COLLECTION_ID`)
2. Add entry to `publications.py`:

```python
'my-journal': {
    'name': 'My Journal Name',
    'source': 'internet-archive',
    'ingestor_class': 'InternetArchiveIngestor',
    'mode': 'web',
    'collection': 'COLLECTION_ID',  # Internet Archive collection ID
    'issn': '1234-5678',  # If known
    'rate_limit_min_ms': 2000,
    'rate_limit_max_ms': 5000,
},
```

3. Run ingestor:

```bash
./ingestors/main.py --publication my-journal --verbosity 2
```

## Limitations

- Only downloads items with PDF files (items without PDFs are skipped)
- XML download is optional (controlled by `download_xml` parameter)
- Relies on Internet Archive metadata for volume/issue numbers
- May not work with all collection types (optimized for journal collections)

## See Also

- Internet Archive Python Library: https://archive.org/services/docs/api/internetarchive/
- Internet Archive API: https://archive.org/services/docs/api/
