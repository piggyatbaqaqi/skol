# Ingestors Package

This package provides classes for ingesting web data into CouchDB from various sources.

## Structure

```
ingestors/
├── __init__.py              # Package exports
├── ingestor.py              # Base Ingestor class (abstract)
├── ingenta.py               # IngentaConnect RSS implementation
├── local_ingenta.py         # Local IngentaConnect mirror implementation
├── local_mykoweb.py         # Local Mykoweb journals implementation
├── local_mykoweb_literature.py # Local Mykoweb literature/books implementation
├── mycosphere.py            # Mycosphere web scraper implementation
├── main.py                  # CLI entry point
├── migrate_to_uuid5.py      # Database migration script
├── test_mycosphere.py       # Standalone test for Mycosphere scraper
└── README.md                # This file
```

## Command Line Usage

The easiest way to use the ingestors is through the CLI with publication sources:

```bash
# List all available publication sources
python -m ingestors.main --list-publications

# Ingest from a specific publication
python -m ingestors.main --publication mycotaxon
python -m ingestors.main --publication studies-in-mycology
python -m ingestors.main --publication ingenta-local

# Ingest from ALL predefined sources (from ist769_skol.ipynb)
python -m ingestors.main --all

# Custom sources with explicit parameters
python -m ingestors.main --source ingenta \
  --rss https://api.ingentaconnect.com/content/mtax/mt?format=rss

python -m ingestors.main --source ingenta \
  --local /data/skol/www/www.ingentaconnect.com \
  --verbosity 3

# Using environment variables for credentials
export COUCHDB_URL=http://localhost:5984
export COUCHDB_USER=myuser
export COUCHDB_PASSWORD=mypass
python -m ingestors.main --all

# Silent mode
python -m ingestors.main --publication mycotaxon -v 0
```

### Available publications

The following publication sources are defined:

| Key | Name | Type | URL/Path |
|-----|------|------|----------|
| `mycotaxon` | Mycotaxon | RSS | https://api.ingentaconnect.com/content/mtax/mt?format=rss |
| `studies-in-mycology` | Studies in Mycology | RSS | https://api.ingentaconnect.com/content/wfbi/sim?format=rss |
| `ingenta-local` | Ingenta Local BibTeX Files | Local | /data/skol/www/www.ingentaconnect.com |
| `mykoweb-journals` | Mykoweb Journals (Mycotaxon, Persoonia, Sydowia) | Local | /data/skol/www/mykoweb.com/systematics/journals |
| `mykoweb-literature` | Mykoweb Literature/Books | Local | /data/skol/www/mykoweb.com/systematics/literature |
| `mykoweb-caf` | Mykoweb CAF PDFs | Local | /data/skol/www/mykoweb.com/CAF |
| `mykoweb-crepidotus` | Mykoweb Crepidotus | Local | /data/skol/www/mykoweb.com/Crepidotus |
| `mykoweb-oldbooks` | Mykoweb Old Books | Local | /data/skol/www/mykoweb.com/OldBooks |
| `mykoweb-gsmnp` | Mykoweb GSMNP | Local | /data/skol/www/mykoweb.com/GSMNP |
| `mykoweb-pholiota` | Mykoweb Pholiota | Local | /data/skol/www/mykoweb.com/Pholiota |
| `mykoweb-misc` | Mykoweb Misc | Local | /data/skol/www/mykoweb.com/misc |
| `mycosphere` | Mycosphere | Web | https://mycosphere.org/archives.php |

### Verbosity Levels

- `0` - Silent (no output except errors)
- `1` - Warnings only (robot denials, errors)
- `2` - Normal (default: skip and add messages)
- `3` - Verbose (includes URLs and separators)

### CLI Options

```
# publication/batch options (mutually exclusive with --rss/--local)
--publication KEY              Use predefined source (mycotaxon, studies-in-mycology, ingenta-local)
--all                     Ingest from all predefined sources
--list-publications            List all available publication sources and exit

# Custom source options (require --source)
--source {ingenta}        Data source to ingest from (required with --rss or --local)
--rss URL                 RSS feed URL to ingest from
--local DIR               Local directory containing BibTeX files

# CouchDB connection
--couchdb-url URL         CouchDB server URL (default: $COUCHDB_URL or http://localhost:5984)
--couchdb-username USER   CouchDB username (default: $COUCHDB_USER)
--couchdb-password PASS   CouchDB password (default: $COUCHDB_PASSWORD)
--database NAME           CouchDB database name (default: skol_dev)

# Other options
--user-agent STRING       User agent for HTTP requests (default: synoptickeyof.life)
--robots-url URL          Custom robots.txt URL
--bibtex-pattern PATTERN  Filename pattern for BibTeX files (default: format=bib)
-v, --verbosity {0,1,2,3} Verbosity level (default: 2)
```

### Environment Variables

The following environment variables are supported:

- `COUCHDB_URL` - CouchDB server URL (overridden by `--couchdb-url`)
- `COUCHDB_USER` - CouchDB username (overridden by `--couchdb-username`)
- `COUCHDB_PASSWORD` - CouchDB password (overridden by `--couchdb-password`)

Environment variables provide a convenient way to avoid exposing credentials in command-line arguments.

## Programmatic Usage

### Ingesting from RSS feeds

```python
from urllib.robotparser import RobotFileParser
import couchdb
from ingestors import IngentaIngestor

# Set up dependencies
couch = couchdb.Server('http://localhost:5984')
db = couch['skol_dev']

user_agent = "synoptickeyof.life"
robot_parser = RobotFileParser()
robot_parser.set_url("https://www.ingentaconnect.com/robots.txt")
robot_parser.read()

# Create ingestor with verbosity control
ingestor = IngentaIngestor(
    db=db,
    user_agent=user_agent,
    robot_parser=robot_parser,
    verbosity=2  # 0=silent, 1=warnings, 2=normal, 3=verbose
)

# Ingest from RSS feed
ingestor.ingest_from_rss(
    rss_url='https://api.ingentaconnect.com/content/mtax/mt?format=rss'
)
```

### Ingesting from local mirror

```python
from pathlib import Path
from urllib.robotparser import RobotFileParser
import couchdb
from ingestors import LocalIngentaIngestor

# Set up dependencies
couch = couchdb.Server('http://localhost:5984')
db = couch['skol_dev']

user_agent = "synoptickeyof.life"
robot_parser = RobotFileParser()
robot_parser.set_url("https://www.ingentaconnect.com/robots.txt")
robot_parser.read()

# Create ingestor for local mirror
ingestor = LocalIngentaIngestor(
    db=db,
    user_agent=user_agent,
    robot_parser=robot_parser,
    verbosity=2
)

# Ingest from local BibTeX files
# Converts /data/skol/www/www.ingentaconnect.com/* to
# https://www.ingentaconnect.com/*
ingestor.ingest_from_local_bibtex(
    root=Path("/data/skol/www/www.ingentaconnect.com")
)
```

### Ingesting from local Mykoweb journals

```python
from pathlib import Path
from urllib.robotparser import RobotFileParser
import couchdb
from ingestors import LocalMykowebJournalsIngestor

# Set up dependencies
couch = couchdb.Server('http://localhost:5984')
db = couch['skol_dev']

user_agent = "synoptickeyof.life"
robot_parser = RobotFileParser()
robot_parser.set_url("https://mykoweb.com/robots.txt")
robot_parser.read()

# Configure local PDF mapping to read PDFs from local filesystem
# instead of downloading them
local_pdf_map = {
    'https://mykoweb.com/systematics/journals': '/data/skol/www/mykoweb.com/systematics/journals'
}

# Create ingestor for Mykoweb journals
ingestor = LocalMykowebJournalsIngestor(
    db=db,
    user_agent=user_agent,
    robot_parser=robot_parser,
    verbosity=2,
    local_pdf_map=local_pdf_map
)

# Ingest from local journal PDFs
# Parses filenames like "Mycotaxon v001n1.pdf", "Sydowia Vol20.pdf", etc.
# Converts /data/skol/www/mykoweb.com/systematics/journals/* to
# https://mykoweb.com/systematics/journals/*
# PDFs are read from local files via local_pdf_map instead of being downloaded
ingestor.ingest_from_local_journals(
    root=Path("/data/skol/www/mykoweb.com/systematics/journals")
)
```

### Ingesting from local Mykoweb literature

```python
from pathlib import Path
from urllib.robotparser import RobotFileParser
import couchdb
from ingestors import LocalMykowebLiteratureIngestor

# Set up dependencies
couch = couchdb.Server('http://localhost:5984')
db = couch['skol_dev']

user_agent = "synoptickeyof.life"
robot_parser = RobotFileParser()
robot_parser.set_url("https://mykoweb.com/robots.txt")
robot_parser.read()

# Configure local PDF mapping to read PDFs from local filesystem
local_pdf_map = {
    'https://mykoweb.com/systematics/literature': '/data/skol/www/mykoweb.com/systematics/literature'
}

# Create ingestor for Mykoweb literature
ingestor = LocalMykowebLiteratureIngestor(
    db=db,
    user_agent=user_agent,
    robot_parser=robot_parser,
    verbosity=2,
    local_pdf_map=local_pdf_map
)

# Ingest from local literature PDFs
# Uses filename (without .pdf) as the title
# Example: "Introduction to Mycology.pdf" -> title="Introduction to Mycology"
# itemtype is set to "book"
ingestor.ingest_from_local_literature(
    root=Path("/data/skol/www/mykoweb.com/systematics/literature")
)
```

## Local PDF Mapping

All ingestors support the `local_pdf_map` parameter, which allows PDFs to be read from the local filesystem instead of being downloaded. This is useful when you have a local mirror of web content.

The `local_pdf_map` is a dictionary mapping URL prefixes to local directory paths:

```python
local_pdf_map = {
    'https://example.com/pdfs': '/data/local/mirror/example.com/pdfs',
    'https://other.org/files': '/mnt/archives/other.org/files',
}
```

When ingesting, if a PDF URL starts with any of the configured prefixes:
1. The URL prefix is replaced with the corresponding local directory path
2. The system tries to find the file:
   - First with the URL path as-is (may contain URL encoding like `%20`)
   - If not found, tries with URL-decoded path (e.g., `%20` → space)
3. If the local file exists, it's read from disk instead of being downloaded
4. If the local file doesn't exist (even after URL decoding), the PDF is downloaded normally

**Example:**

```python
# PDF URL: https://example.com/pdfs/journal/vol1/article.pdf
# Local path: /data/local/mirror/example.com/pdfs/journal/vol1/article.pdf

ingestor = SomeIngestor(
    db=db,
    user_agent=user_agent,
    robot_parser=robot_parser,
    local_pdf_map={
        'https://example.com/pdfs': '/data/local/mirror/example.com/pdfs'
    }
)
```

**URL Decoding:**
The local PDF mapping automatically handles URL-encoded characters. For example:
- URL: `https://mykoweb.com/systematics/literature/Introduction%20to%20Mycology.pdf`
- Local file: `/data/local/mykoweb/literature/Introduction to Mycology.pdf` (with space)
- The system will find the file even though the URL has `%20` instead of a space

This feature reduces network traffic and speeds up ingestion when you have local copies of PDFs.

### Ingesting from Mycosphere (web scraping)

```python
from urllib.robotparser import RobotFileParser
import couchdb
from ingestors import MycosphereIngestor

# Set up dependencies
couch = couchdb.Server('http://localhost:5984')
db = couch['skol_dev']

user_agent = "synoptickeyof.life"
robot_parser = RobotFileParser()
robot_parser.set_url("https://mycosphere.org/robots.txt")
robot_parser.read()

# Create ingestor for Mycosphere
ingestor = MycosphereIngestor(
    db=db,
    user_agent=user_agent,
    robot_parser=robot_parser,
    verbosity=2
)

# Ingest from Mycosphere archives
# Starts at archives page, follows volume links, then issue links
# Extracts: title, authors, volume, number, dates, pages, abstract, keywords
ingestor.ingest_from_archives(
    archives_url='https://mycosphere.org/archives.php',
    max_volumes=1  # Optional: limit for testing
)
```

### Testing Mycosphere Scraper

A standalone test program is provided to verify metadata extraction:

```bash
python3 ingestors/test_mycosphere.py
```

This test program:
- Extracts volume links from the archives page
- Tests metadata extraction from both issue pages and volume index pages
- Saves extracted metadata to JSON files for inspection
- Does not actually ingest into the database

## Creating New Ingestors

To create an ingestor for a new data source:

1. Subclass `Ingestor` from `ingestor.py`
2. Implement the two methods (default implementations exist but may need overriding):
   - `format_pdf_url(base_url: str) -> str` - Format PDF URLs for your source
   - `transform_bibtex_content(content: bytes) -> bytes` - Fix any source-specific syntax issues
3. Optionally override `ingest_from_local_bibtex()` to provide source-specific defaults
4. The `verbosity` parameter is automatically inherited from the base class

Example:

```python
from ingestors.ingestor import Ingestor

class ArXivIngestor(Ingestor):
    def format_pdf_url(self, base_url: str) -> str:
        # ArXiv-specific URL formatting
        return base_url.replace('/abs/', '/pdf/') + '.pdf'

    def transform_bibtex_content(self, content: bytes) -> bytes:
        # ArXiv BibTeX is clean, no transformation needed
        return content

# Usage
ingestor = ArXivIngestor(
    db=db,
    user_agent=user_agent,
    robot_parser=robot_parser,
    verbosity=2
)
```

All print statements in your custom methods should check `self.verbosity`:
- `self.verbosity >= 1` - Warnings and errors
- `self.verbosity >= 2` - Normal progress messages
- `self.verbosity >= 3` - Verbose/debug output

## Type Safety

All code follows PEP 484 type hints and PEP 8 formatting conventions.
