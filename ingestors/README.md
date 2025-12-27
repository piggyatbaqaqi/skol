# Ingestors Package

This package provides classes for ingesting web data into CouchDB from various sources.

## Structure

```
ingestors/
├── __init__.py              # Package exports
├── ingestor.py              # Base Ingestor class (abstract)
├── ingenta.py               # IngentaConnect implementation
├── main.py                  # Usage examples
└── README.md                # This file
```

## Usage

```python
from pathlib import Path
from urllib.robotparser import RobotFileParser
import couchdb
from ingestors.ingenta import IngentaIngestor

# Set up dependencies
couch = couchdb.Server('http://localhost:5984')
db = couch['skol_dev']

user_agent = "synoptickeyof.life"
robot_parser = RobotFileParser()
robot_parser.set_url("https://www.ingentaconnect.com/robots.txt")
robot_parser.read()

# Create ingestor
ingestor = IngentaIngestor(
    db=db,
    user_agent=user_agent,
    robot_parser=robot_parser
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
2. Implement the two abstract methods:
   - `format_pdf_url(base_url: str) -> str` - Format PDF URLs for your source
   - `transform_bibtex_content(content: bytes) -> bytes` - Fix any source-specific syntax issues
3. Optionally override `ingest_from_local_bibtex()` to provide source-specific defaults

Example:

```python
from ingesters.ingestor import Ingestor

class ArXivIngestor(Ingestor):
    def format_pdf_url(self, base_url: str) -> str:
        # ArXiv-specific URL formatting
        return base_url.replace('/abs/', '/pdf/') + '.pdf'

    def transform_bibtex_content(self, content: bytes) -> bytes:
        # ArXiv BibTeX is clean, no transformation needed
        return content
```

## Type Safety

All code follows PEP 484 type hints and PEP 8 formatting conventions.
