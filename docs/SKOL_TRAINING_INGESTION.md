# CouchDB skol_training Database Ingestion

## Overview

Created and populated the `skol_training` CouchDB database with annotated journal data from `data/annotated/journals`.

**Date**: 2025-12-22
**Total Documents**: 190
**Status**: ✅ Complete

## Database Structure

The `skol_training` database matches the schema of `skol_dev` as closely as possible.

### Document Schema

```json
{
  "_id": "unique-hash",
  "_rev": "revision-id",
  "meta": {},
  "pdf_url": "https://mykoweb.com/systematics/journals/...",
  "journal": "Mycotaxon",
  "volume": "054",
  "number": "1",
  "itemtype": "article",
  "author": null,
  "title": null,
  "year": null,
  "issn": null,
  "eissn": null,
  "publication date": null,
  "pages": null,
  "url": "https://mykoweb.com/systematics/journals/...",
  "parent_itemid": null,
  "publishercode": null,
  "source_file": "data/annotated/journals/...",
  "file_type": "issue",
  "_attachments": {
    "article.txt.ann": {
      "content_type": "text/plain",
      "length": 1019341
    }
  }
}
```

### Fields

| Field | Type | Source | Notes |
|-------|------|--------|-------|
| `_id` | String | Generated | MD5 hash of journal/volume/file |
| `meta` | Object | Static | Empty dict (matches skol_dev) |
| `pdf_url` | String | Generated | PDF URL on mykoweb.com |
| `journal` | String | Parsed | From directory structure |
| `volume` | String | Parsed | From directory name (e.g., "054") |
| `number` | String | Parsed | For issue files (n1, n2, etc.) |
| `itemtype` | String | Static | "article" |
| `author` | null | N/A | Not available in source data |
| `title` | null | N/A | Not available in source data |
| `year` | null | N/A | Not available in source data |
| `issn` | null | N/A | Not available in source data |
| `eissn` | null | N/A | Not available in source data |
| `publication date` | null | N/A | Not available in source data |
| `pages` | null | N/A | Not available in source data |
| `url` | String | Generated | Same as pdf_url |
| `parent_itemid` | null | N/A | Not available |
| `publishercode` | null | N/A | Not available |
| `source_file` | String | Original | Path to source .txt.ann file |
| `file_type` | String | Parsed | "issue" or "section" |
| `_attachments` | Object | Attached | article.txt.ann file |

## Source Data

### Directory Structure

```
data/annotated/journals/
├── Mycologia/
│   └── Vol41/
│       └── 2015.28594.Mycologia-Vol-xli-1949_text.txt.ann
├── Mycotaxon/
│   ├── Vol054/
│   │   └── n1.txt.ann
│   ├── Vol117-119/
│   │   ├── s1.txt.ann
│   │   ├── s2.txt.ann
│   │   └── ...
└── Persoonia/
    ├── Vol016-018/
    │   ├── n1.txt.ann (issue 1)
    │   ├── n2.txt.ann (issue 2)
    │   └── ...
    └── Vol019/
        ├── s1.txt.ann (section 1)
        └── ...
```

### File Type Patterns

| Pattern | Type | Example | Description |
|---------|------|---------|-------------|
| `n1`, `n2`, ... | issue | Persoonia Vol016/n1 | Individual issues of a volume |
| `s1`, `s2`, ... | section | Mycotaxon Vol117/s1 | Individual articles/sections |
| Other | unknown | Mycologia/... | Special cases |

## PDF URL Generation

### Mycotaxon

**Pattern**: Whole volume PDFs

```
Mycotaxon/Vol054/n1.txt.ann
→ https://mykoweb.com/systematics/journals/Mycotaxon/Mycotaxon%20v054.pdf
```

All files for a volume point to the same PDF (e.g., Vol117/s1, s2, ... all point to v117.pdf).

### Persoonia

**Pattern**: Issue-level PDFs for "n" files, volume PDFs for "s" files

```
Persoonia/Vol016/n1.txt.ann
→ https://mykoweb.com/systematics/journals/Persoonia/Persoonia%20v16n1.pdf

Persoonia/Vol016/n2.txt.ann
→ https://mykoweb.com/systematics/journals/Persoonia/Persoonia%20v16n2.pdf

Persoonia/Vol019/s1.txt.ann
→ https://mykoweb.com/systematics/journals/Persoonia/Persoonia%20v19.pdf
```

### Mycologia

**Pattern**: Volume-level PDFs

```
Mycologia/Vol41/...
→ https://mykoweb.com/systematics/journals/Mycologia/Mycologia%20v41.pdf
```

## Ingestion Script

**File**: [ingest_annotated_journals.py](../ingest_annotated_journals.py)

### Usage

Basic ingestion (text annotations only):
```bash
python3 ingest_annotated_journals.py
```

With PDF attachments:
```bash
python3 ingest_annotated_journals.py \
  --pdf-dir /data/skol/www/mykoweb.com
```

With all options:
```bash
python3 ingest_annotated_journals.py \
  --data-dir data/annotated/journals \
  --database skol_training \
  --pdf-dir /data/skol/www/mykoweb.com \
  --verbose
```

Re-run to update existing documents with PDFs:
```bash
python3 ingest_annotated_journals.py \
  --pdf-dir /data/skol/www/mykoweb.com \
  --overwrite
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | data/annotated/journals | Source directory |
| `--database` | skol_training | Target database name |
| `--pdf-dir` | None | Local directory with PDFs (e.g., /data/skol/www/mykoweb.com) |
| `--couchdb-url` | $COUCHDB_URL | CouchDB server URL |
| `--username` | $COUCHDB_USER | CouchDB username |
| `--password` | $COUCHDB_PASSWORD | CouchDB password |
| `--overwrite` | False | Overwrite existing documents |
| `-v, --verbose` | 1 | Increase verbosity |
| `-q, --quiet` | False | Suppress output |

### Environment Variables

The script uses these environment variables if options are not provided:
- `COUCHDB_URL`: CouchDB server URL
- `COUCHDB_USER`: Username
- `COUCHDB_PASSWORD`: Password

## Ingestion Results

### Without PDFs (initial run)

```
Total files:           190
Successfully ingested: 190
Failed:                0
Skipped:               0
```

### With PDFs (after --pdf-dir option)

```
Total files:           190
Successfully ingested: 190
Failed:                0
Skipped:               0
PDFs attached:         17
```

**PDFs attached for**:
- Mycotaxon Vol054-058 (5 PDFs)
- Persoonia Vol016-018 (12 PDFs, 4 issues each)

**PDFs not available locally** (only .txt.ann attached):
- Mycotaxon Vol117-119 (156 documents)
- Persoonia Vol019 (8 documents)
- Mycologia Vol041 (1 document)

### Documents by Journal

- **Mycotaxon**: 162 documents
  - Vol054-058: 6 documents (n1 files)
  - Vol117-119: 156 documents (s1-s61 files)
- **Persoonia**: 27 documents
  - Vol016-018: 12 documents (n1-n4 files)
  - Vol019: 8 documents (s1-s8 files)
- **Mycologia**: 1 document
  - Vol41: 1 document

## Attachments

Each document has at least one attachment:

### Text Annotations (all documents)

- **Filename**: `article.txt.ann`
- **Content-Type**: `text/plain`
- **Source**: Original .txt.ann file from data/annotated/journals
- **Format**: YEDDA annotated text with form feed page breaks
- **Count**: 190 documents

### PDF Files (subset of documents)

- **Filename**: `article.pdf`
- **Content-Type**: `application/pdf`
- **Source**: Local copy from /data/skol/www/mykoweb.com/systematics/journals/
- **Count**: 17 documents (only those with PDFs available locally)
- **Size**: 70-90 MB per PDF for Mycotaxon, 20-30 MB per PDF for Persoonia

### YEDDA Annotations

The attached files contain YEDDA format annotations:

```
[@text content
#Label*]
```

These annotations can be parsed using the `PDFSectionExtractor` class which now supports text attachments.

## Integration with PDFSectionExtractor

The attachments can be processed using PDFSectionExtractor:

### Processing Text Annotations

```python
from pyspark.sql import SparkSession
from pdf_section_extractor import PDFSectionExtractor

spark = SparkSession.builder.appName("Training").getOrCreate()
extractor = PDFSectionExtractor(spark=spark)

# Extract sections from text annotation with YEDDA labels
df = extractor.extract_from_document(
    database='skol_training',
    doc_id='b17e337d4823a0d0e4f03c478cde4100',  # Mycotaxon Vol054
    attachment_name='article.txt.ann'
)

# Form feeds are converted to page markers
# YEDDA annotations are parsed into 'label' column
df.select("value", "page_number", "label").show()
```

### Processing PDFs

For documents with PDF attachments:

```python
# Extract sections from PDF (if available)
df = extractor.extract_from_document(
    database='skol_training',
    doc_id='b17e337d4823a0d0e4f03c478cde4100',  # Mycotaxon Vol054
    attachment_name='article.pdf'  # Or auto-detect with no attachment_name
)

# Same DataFrame structure
df.select("value", "page_number", "section_name").show()
```

**Note**: Only 17 documents have PDF attachments. The rest only have .txt.ann files.

## Database Location

- **Server**: http://localhost:5984
- **Database**: `skol_training`
- **Total Size**:
  - Text annotations only: ~200 MB
  - With PDFs: ~1.2 GB (17 PDFs totaling ~1 GB)

## Verification

Check database contents:

```bash
curl -s http://admin:PASSWORD@localhost:5984/skol_training | python3 -m json.tool
```

Query a specific document:

```bash
curl -s http://admin:PASSWORD@localhost:5984/skol_training/b17e337d4823a0d0e4f03c478cde4100 \
  | python3 -m json.tool
```

## Notes

### Missing Fields

The following fields from `skol_dev` are set to `null` because they're not available in the source data:
- author
- title
- year
- issn, eissn
- publication date
- pages
- parent_itemid
- publishercode

These could potentially be filled in later by:
1. Parsing the .txt.ann content (e.g., extract title from first lines)
2. Looking up metadata from journal websites
3. Manual curation

### Additional Fields

The script adds these fields not present in `skol_dev`:
- `source_file`: Original path to .txt.ann file
- `file_type`: "issue" or "section" based on filename

### Re-ingestion

The script is idempotent - running it multiple times will skip existing documents unless `--overwrite` is used. Document IDs are deterministic (MD5 hash of journal/volume/file).

## See Also

- [TXT_ATTACHMENT_IMPLEMENTATION.md](TXT_ATTACHMENT_IMPLEMENTATION.md) - Text file support in PDFSectionExtractor
- [PDF_TXT_ATTACHMENT_SUPPORT.md](PDF_TXT_ATTACHMENT_SUPPORT.md) - User guide for text attachments
- [ingest_annotated_journals.py](../ingest_annotated_journals.py) - Ingestion script source code

---

**Status**: ✅ Complete
**Date**: 2025-12-22
**Documents**: 190
**Database**: skol_training
