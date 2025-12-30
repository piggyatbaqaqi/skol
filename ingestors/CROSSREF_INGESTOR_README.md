# CrossrefIngestor

Ingestor for journals via Crossref API using habanero and pypaperretriever.

## Overview

The CrossrefIngestor retrieves article metadata from Crossref by ISSN/eISSN and downloads PDFs using pypaperretriever. It's an efficient way to ingest entire journals that are indexed in Crossref.

## How It Works

1. **Query Crossref**: Uses habanero to query Crossref API with ISSN filter
2. **Cursor Pagination**: Iterates through all works using cursor-based pagination
3. **Extract Metadata**: Pulls title, authors, DOI, year, journal, volume, issue, pages from Crossref records
4. **Download PDFs**: Uses pypaperretriever to download PDFs via temporary directory
5. **Store in CouchDB**: Saves metadata and PDF attachments, then cleans up temp files

## Features

- **Generator-based**: Streams works one at a time to minimize memory usage
- **Resume capability**: Skips articles already in database with PDFs
- **Automatic cleanup**: Removes temporary PDF files immediately after ingestion
- **Crossref polite pool**: Uses mailto parameter for faster API access
- **BibTeX URLs**: Includes Crossref BibTeX transformation API URLs
- **Error handling**: Continues ingestion even if individual PDFs fail to download

## Installation

Install required packages:

```bash
pip install habanero pypaperretriever
```

## Usage

### Basic Usage

Ingest Journal of Fungi via Crossref:

```bash
./main.py --publication jof-crossref
```

### Configuration

The `jof-crossref` publication is defined in [publications.py](publications.py):

```python
'jof-crossref': {
    'name': 'Journal of Fungi',
    'source': 'crossref',
    'ingestor_class': 'CrossrefIngestor',
    'mode': 'api',
    'issn': '2309-608X',
    'mailto': 'piggy.yarroll+skol@gmail.com',
    'max_articles': None,  # None = all articles
    'allow_scihub': False,
},
```

### Configuration Parameters

- **issn**: ISSN or eISSN of the journal (required)
- **mailto**: Email for Crossref polite pool (default: piggy.yarroll+skol@gmail.com)
- **max_articles**: Maximum articles to ingest (None = all, useful for testing)
- **allow_scihub**: Allow pypaperretriever to use Sci-Hub fallback (default: False)

### Adding New Journals

To add a new journal via Crossref:

1. Find the ISSN/eISSN of the journal
2. Add entry to `publications.py`:

```python
'my-journal-crossref': {
    'name': 'My Journal Name',
    'source': 'crossref',
    'ingestor_class': 'CrossrefIngestor',
    'mode': 'api',
    'issn': '1234-5678',
    'mailto': 'your@email.com',
    'max_articles': None,
    'allow_scihub': False,
},
```

3. Run ingestion:

```bash
./main.py --publication my-journal-crossref
```

## Metadata Structure

Documents in CouchDB include:

```python
{
    '_id': 'uuid-based-on-doi',
    'url': 'https://doi.org/10.3390/jof12010028',
    'doi': '10.3390/jof12010028',
    'title': 'Article Title',
    'author': 'First Author; Second Author; Third Author',
    'year': '2025',
    'journal': 'Journal of Fungi',
    'volume': '12',
    'issue': '1',
    'pages': '28',
    'pdf_url': 'https://doi.org/10.3390/jof12010028',
    'human_url': 'https://www.mdpi.com/2309-608X/12/1/28',  # Publisher's URL
    'bibtex_url': 'https://api.crossref.org/works/10.3390/jof12010028/transform/application/x-bibtex',
    'source': 'crossref',
    '_attachments': {
        'article.pdf': { ... }
    }
}
```

### Human URL Extraction

The `human_url` field is intelligently extracted from Crossref records in order of preference:

1. **`resource.primary.URL`** - Primary publisher resource URL
2. **`link` list entry** - URL with `content-type: "text/html"`
3. **DOI URL fallback** - `https://doi.org/{DOI}`

This ensures that when available, the human URL points directly to the publisher's article page rather than going through DOI resolution.

## BibTeX URLs

Each document includes a `bibtex_url` pointing to Crossref's BibTeX transformation API:

```
https://api.crossref.org/works/{DOI}/transform/application/x-bibtex
```

This can be used to retrieve BibTeX records for citations.

## PDF Download

PDFs are downloaded using pypaperretriever, which:

1. Tries official publisher links first
2. Falls back to open access repositories
3. Optionally uses Sci-Hub if `allow_scihub=True`

Downloaded PDFs are:
- Saved to a temporary directory
- Read into memory
- Attached to CouchDB document
- Deleted from filesystem immediately

**No PDFs are left on the filesystem after ingestion.**

## Performance

### Crossref API

- **Rate limiting**: Built-in 0.1 second delay between batches
- **Polite pool**: Faster access when using mailto parameter
- **Cursor pagination**: Efficient for large journals (1000s of articles)
- **Batch size**: 100 works per API request

### pypaperretriever

- Downloads one PDF at a time
- Uses temporary directory per PDF
- Automatic cleanup after each PDF

### Typical Performance

For a journal with 1000 articles:
- Metadata fetching: ~2-5 minutes
- PDF downloads: ~1-3 hours (depending on publisher response times and network)

## Verbosity Levels

Control output detail with `-v` flag:

```bash
# Level 2 (default): Shows progress for each work
./main.py --publication jof-crossref

# Level 3: Verbose (shows temp directories, PDF sizes, etc)
./main.py --publication jof-crossref -v 3

# Level 1: Warnings/errors only
./main.py --publication jof-crossref -v 1
```

## Testing

Test with a small subset using `max_articles`:

```python
'jof-crossref-test': {
    'name': 'Journal of Fungi (Test)',
    'source': 'crossref',
    'ingestor_class': 'CrossrefIngestor',
    'mode': 'api',
    'issn': '2309-608X',
    'max_articles': 10,  # Only first 10 articles
    'allow_scihub': False,
},
```

## Error Handling

The ingestor continues even if individual works fail:

- **Missing DOI**: Skipped with message
- **PDF download failure**: Metadata saved without PDF attachment
- **pypaperretriever error**: Logged and continues to next article
- **Crossref API error**: Retries or stops batch, logged with details

## Comparison with Other Ingestors

| Feature | CrossrefIngestor | MdpiIngestor | IngentaIngestor |
|---------|------------------|--------------|-----------------|
| Source | Crossref API | MDPI website | IngentaConnect |
| PDF method | pypaperretriever | Direct download | Direct download |
| Metadata | Crossref records | Web scraping | BibTeX + scraping |
| Coverage | Any Crossref journal | MDPI only | IngentaConnect only |
| Authentication | None (open API) | None | None |

## Advantages

✅ **Universal**: Works with any journal indexed in Crossref
✅ **Rich metadata**: Crossref provides comprehensive metadata
✅ **BibTeX support**: Direct API for BibTeX records
✅ **No scraping**: Uses official API, more stable than web scraping
✅ **Open access focus**: pypaperretriever finds open access versions

## Limitations

⚠️ **PDF availability**: Depends on pypaperretriever's ability to find open access PDFs
⚠️ **Slower**: PDF downloads can be slow compared to direct publisher access
⚠️ **Sci-Hub**: Requires enabling `allow_scihub` for paywalled articles
⚠️ **Network dependent**: Requires stable internet connection

## Files

- [crossref.py](crossref.py) - Main CrossrefIngestor implementation (~450 lines)
- [get_journal_dois.py](get_journal_dois.py) - Standalone script to fetch DOIs only
- [publications.py](publications.py) - Publication registry with jof-crossref entry
- [main.py](main.py) - CLI with CrossrefIngestor registered

## See Also

- [Crossref API Documentation](https://www.crossref.org/documentation/retrieve-metadata/)
- [habanero Documentation](https://habanero.readthedocs.io/)
- [pypaperretriever Documentation](https://josephiturner.com/pypaperretriever/)
- [MDPI Ingestor](mdpi.py) - Alternative for MDPI journals
- [Ingenta Ingestor](ingenta.py) - Alternative for IngentaConnect journals

## License

Part of the skol ingestion system.
