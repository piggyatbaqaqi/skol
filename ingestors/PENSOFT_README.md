# PensoftIngestor

Ingestor for scraping articles from Pensoft publishing platform journals.

## Overview

The `PensoftIngestor` class scrapes article metadata and PDFs from Pensoft journals. It navigates through issues listing pages (with pagination), individual issue pages (with pagination), and downloads PDFs directly.

## Supported Journals

Currently configured for:
- **MycoKeys**: A peer reviewed open access journal
  - ISSN: 1314-4057
  - eISSN: 1314-4049
  - URL: https://mycokeys.pensoft.net/

The ingestor is designed to work with any Pensoft journal by specifying the journal name and ID.

## Website Structure

### Navigation Flow

1. **Issues Listing Page** (`/issues`)
   - Shows all issues with pagination
   - Sidebar shows "Go to issue" with annotation "from 1 to X" indicating total issues
   - Pagination URL: `browse_journal_issues.php?journal_name=[NAME]&journal_id=[ID]&p=[PAGE]`
   - Pages are 0-indexed (p=0, p=1, p=2, ...)
   - Approximately 10-11 issues per page

2. **Individual Issue Pages** (`/issue/[ISSUE_ID]/`)
   - Lists all articles in the issue
   - May have pagination if many articles
   - Pagination URL: `browse_journal_issue_documents.php?issue_id=[ID]&journal_id=[ID]&p=[PAGE]`
   - Issue IDs are non-sequential database identifiers (e.g., 5378, 5291)

3. **Article Pages** (`/article/[ARTICLE_ID]/`)
   - Full article metadata and content
   - Download links for HTML, XML, and PDF

4. **PDF Downloads** (`/article/[ARTICLE_ID]/download/pdf/[PDF_ID]`)
   - Direct PDF download
   - PDF_ID is different from ARTICLE_ID and must be extracted from the link

## Metadata Extracted

The ingestor extracts the following metadata:

- **title**: Article title
- **authors**: Author names (with ORCID links when available)
- **volume**: Not typically available in Pensoft structure
- **number**: Issue number
- **publication_date**: Publication date in ISO format (YYYY-MM-DD)
- **year**: Publication year
- **doi**: Digital Object Identifier (format: `10.3897/[journal].[issue].[suffix]`)
- **pages**: Page range (e.g., "347-368")
- **article_type**: Type of article (Research Article, Review Article, etc.)
- **issn**: Journal ISSN
- **eissn**: Journal eISSN
- **url**: URL to article page
- **pdf_url**: Direct URL to PDF file
- **article_id**: Article ID from URL

### DOI Pattern

Pensoft DOIs follow a consistent pattern:
```
10.3897/mycokeys.[ISSUE_NUMBER].[DOI_SUFFIX]
```

Example: `10.3897/mycokeys.125.174645`

### Date Formats

Pensoft uses different date formats in different locations:
- **Issue pages**: DD-MM-YYYY (e.g., "27-11-2025")
- **Article pages**: DD Month YYYY (e.g., "27 November 2025")

Both formats are handled and converted to ISO format (YYYY-MM-DD).

## Usage

### Command Line

Ingest MycoKeys using the predefined publication:

```bash
./main.py --publication mycokeys --verbosity 2
```

### Python API

```python
from ingestors.pensoft import PensoftIngestor
import couchdb

# Connect to database
couch = couchdb.Server('http://localhost:5984/')
db = couch['skol_dev']

# Create ingestor instance
ingestor = PensoftIngestor(
    db=db,
    journal_name='mycokeys',
    journal_id='11',
    issn='1314-4057',
    eissn='1314-4049',
    verbosity=2
)

# Run ingestion
ingestor.ingest()
```

### Testing with Limited Issues

For testing, you can limit the number of issues processed:

```python
ingestor.ingest_from_issues(max_issues=2)
```

## Implementation Details

### Key Methods

#### `_extract_total_issues(soup)`
Extracts total number of issues from the sidebar.

**Returns**: Total number of issues, or None if not found

Looks for patterns:
- "from 1 to X" in sidebar
- "X issues matching" in page text

#### `_extract_issue_ids_from_page(soup)`
Extracts issue IDs and metadata from an issues listing page.

**Returns**: List of dicts with keys:
- `issue_id`: Database ID (non-sequential)
- `url`: Full URL to issue page
- `number`: Issue number (extracted from text)
- `text`: Link text

#### `_extract_articles_from_issue_page(soup, issue_number)`
Extracts article metadata from an issue page.

**Returns**: List of dicts with complete article metadata including:
- Article identification (ID, URL, title)
- Authors (extracted from author links)
- DOI
- Publication date
- Page range
- Article type
- PDF URL

#### `_check_for_pagination(soup)`
Checks if there are pagination links and returns total number of pages.

**Returns**: Total number of pages (0-indexed), or None if no pagination

#### `_parse_date(date_str)`
Parses various date formats to ISO format (YYYY-MM-DD).

Supported formats:
- `27-11-2025` (issue pages)
- `27 November 2025` (article pages)
- `27 Nov 2025`
- `November 27, 2025`
- `2025-11-27`

### Data Flow

1. Fetch issues listing page
2. Extract total number of issues from sidebar
3. Check for pagination on issues listing
4. Paginate through all issues listing pages
5. Extract all issue IDs
6. For each issue:
   - Fetch issue page
   - Check for within-issue pagination
   - Extract articles from all pages
7. For each article:
   - Extract metadata directly from issue page
   - Extract PDF URL from download link
8. Ingest articles with complete metadata

### Important Notes

**Issue IDs are non-sequential**: Issue IDs are database identifiers and don't correlate to issue numbers. For example:
- Issue 126 → ID: 5378
- Issue 125 → ID: 5291
- Issue 120 → ID: 5193

**PDF IDs differ from Article IDs**: The PDF download URL contains a separate PDF_ID that must be extracted from the link.

**Pagination is two-level**:
1. Issues listing pagination (10-11 issues per page)
2. Within-issue pagination (when issue has many articles)

### Error Handling

- Skips issues that fail to load
- Only ingests articles with valid PDF URLs
- Reports failures at appropriate verbosity levels
- Continues processing even if some issues fail

## Rate Limiting

The ingestor respects `robots.txt` and applies rate limiting:

- Default rate limit: 1000-5000ms between requests
- Configurable via `rate_limit_min_ms` and `rate_limit_max_ms`
- Uses `Ingestor` base class rate limiting methods

## Verbosity Levels

- **0**: Silent (no output)
- **1**: Warnings and errors only
- **2**: Normal progress messages (default)
  - Shows total issues found
  - Shows issue being processed
  - Shows article counts per issue
  - Shows ingestion progress
- **3**: Verbose (includes pagination details)
  - Shows pagination page numbers
  - Shows within-issue pagination

## URL Patterns Reference

### Issues Listing
```
Base: https://mycokeys.pensoft.net/issues
Pagination: https://mycokeys.pensoft.net/browse_journal_issues.php?journal_name=mycokeys&lang=&journal_id=11&p=[0-N]
```

### Issue Pages
```
Issue: https://mycokeys.pensoft.net/issue/[ISSUE_ID]/
Example: https://mycokeys.pensoft.net/issue/5378/
```

### Issue Pagination
```
https://mycokeys.pensoft.net/browse_journal_issue_documents.php?journal_name=mycokeys&issue_id=[ISSUE_ID]&lang=&journal_id=11&p=[0-N]
```

### Articles
```
Page: https://mycokeys.pensoft.net/article/[ARTICLE_ID]/
PDF: https://mycokeys.pensoft.net/article/[ARTICLE_ID]/download/pdf/[PDF_ID]
XML: https://mycokeys.pensoft.net/article/[ARTICLE_ID]/download/xml/
```

## CouchDB Document Structure

Documents are stored with the following structure:

```json
{
  "_id": "uuid-based-on-article-url",
  "url": "https://mycokeys.pensoft.net/article/174645/",
  "article_id": "174645",
  "title": "Morphological and phylogenetic analyses reveal...",
  "authors": "Gui-Li Zhao, Yong-Zhong Lu, Xing-Juan Xiao, Ying Liu, Qiang Chen, Ning-Guo Liu",
  "doi": "10.3897/mycokeys.125.174645",
  "publication_date": "2025-11-27",
  "year": "2025",
  "number": "125",
  "pages": "347–368",
  "article_type": "Research Article",
  "issn": "1314-4057",
  "eissn": "1314-4049",
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
Fetching issues from: https://mycokeys.pensoft.net/issues
Total issues: 127
Issues listing has 13 page(s)
Found 127 issue(s)

============================================================
Processing Issue 126
============================================================
  Found 8 article(s)
  8 article(s) have PDF URLs
  Ingesting 8 article(s)

============================================================
Processing Issue 125
============================================================
  Issue has 2 page(s) of articles
  Found 15 article(s)
  15 article(s) have PDF URLs
  Ingesting 15 article(s)

============================================================
Ingestion complete
============================================================
```

## Adapting for Other Pensoft Journals

To use this ingestor for other Pensoft journals, you need to determine:

1. **journal_name**: The subdomain name (e.g., `mycokeys` from `mycokeys.pensoft.net`)
2. **journal_id**: The numeric ID used in pagination URLs
3. **issn/eissn**: Journal ISSN and eISSN

You can find these by:
- Visiting the journal's issues page
- Inspecting pagination URLs to extract `journal_name` and `journal_id` parameters
- Looking for ISSN information on the journal website

Example configuration for another journal:

```python
SOURCES: Dict[str, Dict[str, Any]] = {
    'another-journal': {
        'name': 'Another Pensoft Journal',
        'source': 'pensoft',
        'ingestor_class': 'PensoftIngestor',
        'mode': 'web',
        'journal_name': 'anotherjournal',  # From subdomain
        'journal_id': '42',                # From pagination URL
        'issues_url': 'https://anotherjournal.pensoft.net/issues',
        'issn': 'XXXX-XXXX',
        'eissn': 'XXXX-XXXX',
        'rate_limit_min_ms': 1000,
        'rate_limit_max_ms': 5000,
    },
}
```

## Related Files

- **Implementation**: [pensoft.py](pensoft.py)
- **Configuration**: [publications.py](publications.py) (search for 'mycokeys')
- **Main entry point**: [main.py](main.py)
- **Base class**: [ingestor.py](ingestor.py)

## Platform Compatibility

This ingestor should work with any journal on the Pensoft publishing platform, which includes:
- MycoKeys (mycology)
- ZooKeys (zoology)
- PhytoKeys (botany)
- And many others...

The structure and HTML patterns are consistent across the platform.
