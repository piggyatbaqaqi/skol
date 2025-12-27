# Ingestors Package

This package provides classes for ingesting web data into CouchDB from various sources.

## Structure

```
ingestors/
├── __init__.py              # Package exports
├── ingestor.py              # Base Ingestor class (abstract)
├── ingenta.py               # IngentaConnect implementation
├── main.py                  # CLI entry point
└── README.md                # This file
```

## Command Line Usage

The easiest way to use the ingestors is through the CLI:

```bash
# Ingest from Ingenta RSS feed (normal verbosity)
python -m ingestors.main --source ingenta \
  --rss https://api.ingentaconnect.com/content/mtax/mt?format=rss

# Ingest from local BibTeX files (verbose mode)
python -m ingestors.main --source ingenta \
  --local /data/skol/www/www.ingentaconnect.com \
  --verbosity 3

# Silent mode (no output except errors)
python -m ingestors.main --source ingenta \
  --rss https://api.ingentaconnect.com/content/wfbi/sim?format=rss \
  -v 0

# Custom database and CouchDB URL with credentials
python -m ingestors.main --source ingenta \
  --couchdb-url http://localhost:5984 \
  --couchdb-username myuser \
  --couchdb-password mypass \
  --database my_database \
  --rss https://api.ingentaconnect.com/content/mtax/mt?format=rss

# Using environment variables for credentials
export COUCHDB_URL=http://localhost:5984
export COUCHDB_USER=myuser
export COUCHDB_PASSWORD=mypass
python -m ingestors.main --source ingenta \
  --rss https://api.ingentaconnect.com/content/mtax/mt?format=rss
```

### Verbosity Levels

- `0` - Silent (no output except errors)
- `1` - Warnings only (robot denials, errors)
- `2` - Normal (default: skip and add messages)
- `3` - Verbose (includes URLs and separators)

### CLI Options

```
--source {ingenta}        Data source to ingest from (required)
--rss URL                 RSS feed URL to ingest from
--local DIR               Local directory containing BibTeX files
--couchdb-url URL         CouchDB server URL (default: $COUCHDB_URL or http://localhost:5984)
--couchdb-username USER   CouchDB username (default: $COUCHDB_USER)
--couchdb-password PASS   CouchDB password (default: $COUCHDB_PASSWORD)
--database NAME           CouchDB database name (default: skol_dev)
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

```python
from pathlib import Path
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

# Ingest from local BibTeX files
ingestor.ingest_from_local_bibtex(
    root=Path("/data/skol/www/www.ingentaconnect.com")
)
```

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
