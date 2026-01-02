# MedwinPublishersIngestor

Ingestor for scraping articles from Medwin Publishers journal websites.

## Overview

The `MedwinPublishersIngestor` class scrapes article metadata and PDFs from Medwin Publishers journals. It navigates through archive pages, issue pages, and individual article pages to collect comprehensive metadata.

## Supported Journals

Currently configured for:
- **OAJMMS**: Open Access Journal of Mycology & Mycological Sciences
  - ISSN: 2689-7822
  - URL: https://www.medwinpublishers.com/OAJMMS/

The ingestor is designed to work with any Medwin Publishers journal that follows the same structure.

## Website Structure

### Navigation Flow

1. **Archive Page** (`archive.php`)
   - Lists all volumes and issues
   - Provides links to issue pages
   - URL pattern: `volume.php?volumeId=X&issueId=Y`

2. **Issue Pages** (`volume.php`)
   - Lists all articles in an issue
   - Provides article titles, authors, DOIs, and article types
   - Links to individual article pages
   - URL pattern: `article-description.php?artId=X`

3. **Article Pages** (`article-description.php`)
   - Contains full metadata: abstract, keywords, publication date
   - Provides "View PDF" button with direct PDF link
   - URL pattern for PDFs: `https://medwinpublishers.com/OAJMMS/[slug].pdf`

## Metadata Extracted

The ingestor extracts the following metadata:

- **title**: Article title
- **authors**: Author names (extracted from issue page)
- **volume**: Volume number
- **number**: Issue number
- **publication_date**: Publication date in ISO format (YYYY-MM-DD)
- **year**: Publication year
- **doi**: Digital Object Identifier
- **abstract**: Full abstract text
- **keywords**: Comma-separated keywords
- **article_type**: Type of article (Research Article, Mini Review, etc.)
- **issn**: Journal ISSN
- **journal**: Journal name
- **url**: URL to article page
- **pdf_url**: Direct URL to PDF file

### Note on Missing Metadata

- **Page numbers**: Not available (online-only format, all articles show "0-0")
- **Submission/Acceptance dates**: Not available on the website
- **Author affiliations**: Available on issue pages but not extracted in current implementation

## Usage

### Command Line

Ingest OAJMMS using the predefined publication:

```bash
./main.py --publication oajmms --verbosity 2
```

### Python API

```python
from ingestors.medwin_publishers import MedwinPublishersIngestor
import couchdb

# Connect to database
couch = couchdb.Server('http://localhost:5984/')
db = couch['skol_dev']

# Create ingestor instance
ingestor = MedwinPublishersIngestor(
    db=db,
    archives_url='https://www.medwinpublishers.com/OAJMMS/archive.php',
    issn='2689-7822',
    journal_name='Open Access Journal of Mycology & Mycological Sciences',
    verbosity=2
)

# Run ingestion
ingestor.ingest()
```

### Testing with Limited Issues

For testing, you can limit the number of issues processed:

```python
ingestor.ingest_from_archives(max_issues=2)
```

## Implementation Details

### Key Methods

#### `_extract_issue_links(soup)`
Extracts issue links from the archive page.

**Returns**: List of dicts with keys:
- `url`: Full URL to issue page
- `volume`: Volume number
- `number`: Issue number
- `year`: Publication year (if available)
- `volumeId`, `issueId`: Query parameters

#### `_extract_article_links_from_issue(soup, volume, number)`
Extracts article information from an issue page.

**Returns**: List of dicts with basic article metadata:
- `article_id`: Article ID from URL
- `url`: URL to article page
- `title`: Article title
- `authors`: Author names (from issue page)
- `doi`: DOI if found
- `article_type`: Article type if found

#### `_extract_article_metadata(soup, base_article)`
Enhances article metadata by scraping the full article page.

**Returns**: Complete article metadata dict with abstract, keywords, dates, and PDF URL.

#### `_parse_date(date_str)`
Parses various date formats to ISO format (YYYY-MM-DD).

Supported formats:
- `July 11, 2024`
- `11 July 2024`
- `Jul 11, 2024`
- `2024-07-11`
- `11/07/2024`
- `07/11/2024`

### Data Flow

1. Fetch archive page
2. Extract issue links
3. For each issue:
   - Fetch issue page
   - Extract basic article information (title, authors, DOI)
4. For each article:
   - Fetch article page
   - Extract detailed metadata (abstract, keywords, dates)
   - Extract PDF URL from "View PDF" button
5. Ingest articles with complete metadata

### Error Handling

- Skips issues that fail to load
- Skips articles that fail to load
- Only ingests articles with valid PDF URLs
- Reports failures at appropriate verbosity levels

## Rate Limiting

The ingestor respects `robots.txt` and applies rate limiting:

- Default rate limit: 1000-5000ms between requests
- Configurable via `rate_limit_min_ms` and `rate_limit_max_ms`
- Uses `Ingestor` base class rate limiting methods

## Verbosity Levels

- **0**: Silent (no output)
- **1**: Warnings and errors only
- **2**: Normal progress messages (default)
  - Shows volume/issue being processed
  - Shows article counts
  - Shows ingestion progress
- **3**: Verbose (includes article titles)
  - Shows each article being fetched
  - Shows metadata extraction status

## Known Limitations

1. **Author Data**: Authors are extracted from issue pages (not article pages), as Medwin Publishers doesn't display author information on article detail pages.

2. **Page Numbers**: Online-only journal format means no traditional page numbers are available (all show "0-0").

3. **Missing Issues**: Some issue links in the archive may return errors ("Article is not exist..!!"). These are skipped automatically.

4. **DOI Patterns**: DOIs follow the pattern `10.23880/oajmms-16000XXX` where XXX is the article number.

## CouchDB Document Structure

Documents are stored with the following structure:

```json
{
  "_id": "uuid-based-on-article-url",
  "url": "https://medwinpublishers.com/article-description.php?artId=12877",
  "title": "Article Title",
  "authors": "Author One, Author Two*",
  "abstract": "Full abstract text...",
  "keywords": "keyword1, keyword2, keyword3",
  "doi": "10.23880/oajmms-16000XXX",
  "volume": "7",
  "number": "2",
  "publication_date": "2024-07-11",
  "year": "2024",
  "article_type": "Research Article",
  "issn": "2689-7822",
  "journal": "Open Access Journal of Mycology & Mycological Sciences",
  "itemtype": "article",
  "_attachments": {
    "article.pdf": {
      "content_type": "application/pdf",
      "length": 123456
    }
  }
}
```

## Example Output

```
Fetching archives from: https://www.medwinpublishers.com/OAJMMS/archive.php
Found 13 issue(s)

============================================================
Processing Volume 7 Issue 2
Year: 2024
============================================================
  Found 8 article(s)
    Fetching metadata for: Morphology and Phylogeny of Lactarius wallichianae sp. nov...
      ✓ Metadata extracted
    Fetching metadata for: Evaluation of Various Extracellular Enzymes...
      ✓ Metadata extracted
  Ingesting 8 article(s)

============================================================
Ingestion complete
============================================================
```

## Related Files

- **Implementation**: [medwin_publishers.py](medwin_publishers.py)
- **Configuration**: [publications.py](publications.py) (search for 'oajmms')
- **Main entry point**: [main.py](main.py)
- **Base class**: [ingestor.py](ingestor.py)

## Comparison with CrossrefIngestor

The commented-out `oajmms-crossref` entry in `publications.py` (lines 161-171) indicates that Crossref API was tried but didn't work for this journal. The `MedwinPublishersIngestor` provides direct web scraping as an alternative approach.

Advantages of direct scraping:
- Guaranteed access to all published articles
- Access to abstracts and keywords (not always in Crossref)
- Direct PDF links

Disadvantages:
- Slower (must fetch multiple pages per article)
- More fragile (breaks if website structure changes)
- Requires more requests and rate limiting
