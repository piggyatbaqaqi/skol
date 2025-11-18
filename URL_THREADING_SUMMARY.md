# URL Threading Through SKOL Pipeline

## Overview

This document describes how the `url` field flows through the entire SKOL taxon extraction pipeline, from CouchDB source documents to final exported Taxon JSON records.

## Data Flow

```
CouchDB Row (with url field)
  ↓
FileObject (File or CouchDBFile)
  └─ url property
      ↓
Line object
  └─ url property (captured via duck typing)
      ↓
Paragraph object
  └─ url property (from last_line.url)
      ↓
Paragraph.as_dict()
  └─ 'url' key in dictionary
      ↓
Taxon.dictionaries()
  └─ Each paragraph dict includes 'url'
      ↓
Final JSON Document
  ├─ source.url (from first nomenclature line)
  └─ paragraphs[*].url (from each paragraph)
```

## Implementation Details

### 1. FileObject Abstract Class

**Location:** [fileobj.py](fileobj.py:26-29)

```python
@property
@abstractmethod
def url(self) -> Optional[str]:
    return None
```

The abstract `url` property is defined in the FileObject interface.

### 2. File Class Implementation

**Location:** [file.py](file.py:71-73)

```python
@property
def url(self) -> Optional[str]:
    return self._filename
```

For regular files, the URL is the filename.

### 3. CouchDBFile Class Implementation

**Location:** [couchdb_file.py](couchdb_file.py:133-136)

```python
@property
def url(self) -> Optional[str]:
    """URL from the CouchDB row."""
    return self._url
```

For CouchDB files, the URL comes from the `url` field of the row.

**Initialization:** [couchdb_file.py](couchdb_file.py:29-50)

```python
def __init__(
    self,
    content: str,
    doc_id: str,
    attachment_name: str,
    db_name: str,
    url: Optional[str] = None
) -> None:
    # ...
    self._url = url
```

**Population from Row:** [couchdb_file.py](couchdb_file.py:168-179)

```python
for row in partition:
    # Extract url from row if available
    url = getattr(row, 'url', None)

    # Create CouchDBFile object from row data
    file_obj = CouchDBFile(
        content=row.value,
        doc_id=row.doc_id,
        attachment_name=row.attachment_name,
        db_name=db_name,
        url=url
    )
```

### 4. Line Class

**Location:** [line.py](line.py:21)

```python
_url: Optional[str]
```

**Initialization:** [line.py](line.py:39)

```python
self._url = None
```

**Duck Typing Capture:** [line.py](line.py:54-55)

```python
if hasattr(fileobj, 'url'):
    self._url = fileobj.url
```

**Property:** [line.py](line.py:128-131)

```python
@property
def url(self) -> Optional[str]:
    """URL from the source (optional)."""
    return self._url
```

### 5. Paragraph Class

**Location:** [paragraph.py](paragraph.py:328-332)

```python
@property
def url(self) -> Optional[str]:
    if self.last_line is None:
        return None
    return self.last_line.url
```

The paragraph gets its URL from the last line in the paragraph.

**Dictionary Representation:** [paragraph.py](paragraph.py:123-132)

```python
def as_dict(self) -> Dict[str, Optional[str]]:
    return {
        'filename': self.filename,
        'url': self.url,
        'label': str(self.top_label()),
        'paragraph_number': str(self.paragraph_number),
        'page_number': str(self.page_number),
        'empirical_page_number': str(self.empirical_page_number),
        'body': str(self)
    }
```

### 6. Taxon Class

**Location:** [taxon.py](taxon.py:9-13)

```python
FIELDNAMES = [
    'serial_number',
    'filename', 'url', 'label', 'paragraph_number', 'page_number',
    'empirical_page_number', 'body'
]
```

The `url` field is included in the FIELDNAMES list.

**Dictionary Generation:** [taxon.py](taxon.py:54-58)

```python
def dictionaries(self) -> Iterator[Dict[str, str]]:
    for pp in itertools.chain(self._nomenclatures, self._descriptions):
        d = pp.as_dict()
        d['serial_number'] = str(self._serial)
        yield d
```

Each paragraph dictionary includes the `url` field from `pp.as_dict()`.

### 7. JSON Export

**Location:** [extract_taxa_to_couchdb.py](extract_taxa_to_couchdb.py:83-97)

```python
# Extract metadata from first nomenclature paragraph's first line
first_line = first_nomenclature_para.first_line
if not first_line:
    return None

source_doc_id = first_line.doc_id if first_line.doc_id else "unknown"
source_url = first_line.url  # ← URL extracted here
source_db_name = first_line.db_name if first_line.db_name else "unknown"
line_number = first_line.line_number

# Build the document
doc = {
    "type": "taxon",
    "serial_number": paragraphs[0].get('serial_number', '0'),
    "source": {
        "doc_id": source_doc_id,
        "url": source_url,  # ← URL in source metadata
        "db_name": source_db_name,
        "line_number": line_number
    },
    "paragraphs": paragraphs,  # ← Each paragraph includes url field
    # ...
}
```

### 8. Idempotent Document ID

**Location:** [extract_taxa_to_couchdb.py](extract_taxa_to_couchdb.py:33-50)

```python
def generate_taxon_doc_id(doc_id: str, url: Optional[str], line_number: int) -> str:
    """Generate a unique, deterministic document ID for a taxon."""
    # Create composite key
    key_parts = [
        doc_id,
        url if url else "no_url",  # ← URL is part of the key
        str(line_number)
    ]
    composite_key = ":".join(key_parts)

    # Generate deterministic hash
    hash_obj = hashlib.sha256(composite_key.encode('utf-8'))
    doc_hash = hash_obj.hexdigest()

    return f"taxon_{doc_hash}"
```

The URL is a critical component of the idempotent document ID.

## Example Data Flow

### Input: CouchDB Row

```python
Row(
    doc_id="article_2023_001",
    attachment_name="fulltext.txt.ann",
    value="[@Agaricus novus Author 2023#Nomenclature*]\n[@Description text#Description*]",
    url="http://example.com/article"  # ← URL enters here
)
```

### Step 1: CouchDBFile

```python
file_obj = CouchDBFile(
    content="[@Agaricus novus Author 2023#Nomenclature*]...",
    doc_id="article_2023_001",
    attachment_name="fulltext.txt.ann",
    db_name="mycobank",
    url="http://example.com/article"  # ← Stored in file object
)
```

### Step 2: Line Objects

```python
line = Line("Agaricus novus Author 2023", file_obj)
# line.url → "http://example.com/article" (via duck typing)
```

### Step 3: Paragraph Objects

```python
paragraph = Paragraph(lines=[line])
# paragraph.url → "http://example.com/article" (from last_line)
```

### Step 4: Paragraph Dictionary

```python
paragraph.as_dict()
# {
#     'filename': 'mycobank/article_2023_001/fulltext.txt.ann',
#     'url': 'http://example.com/article',  # ← URL preserved
#     'label': 'Nomenclature',
#     'paragraph_number': '1',
#     'page_number': '1',
#     'empirical_page_number': None,
#     'body': 'Agaricus novus Author 2023'
# }
```

### Step 5: Taxon JSON Document

```json
{
  "_id": "taxon_abc123...",
  "type": "taxon",
  "serial_number": "1",
  "source": {
    "doc_id": "article_2023_001",
    "url": "http://example.com/article",
    "db_name": "mycobank",
    "line_number": 1
  },
  "paragraphs": [
    {
      "serial_number": "1",
      "filename": "mycobank/article_2023_001/fulltext.txt.ann",
      "url": "http://example.com/article",
      "label": "Nomenclature",
      "paragraph_number": "1",
      "page_number": "1",
      "empirical_page_number": null,
      "body": "Agaricus novus Author 2023"
    },
    {
      "serial_number": "1",
      "filename": "mycobank/article_2023_001/fulltext.txt.ann",
      "url": "http://example.com/article",
      "label": "Description",
      "paragraph_number": "2",
      "page_number": "1",
      "empirical_page_number": null,
      "body": "Description text"
    }
  ],
  "nomenclature_count": 1,
  "description_count": 1
}
```

## Key Benefits

### 1. Complete Traceability
Every paragraph can be traced back to its original source URL, enabling:
- Citation generation
- Source verification
- Link back to original documents

### 2. Idempotent Operations
The URL is part of the composite key for document IDs:
```
SHA256(doc_id + ":" + url + ":" + line_number)
```

This ensures that taxa from the same source location always get the same ID.

### 3. Distributed Processing
The URL threads through the pipeline seamlessly in distributed PySpark operations:
- Each partition preserves URL metadata
- No special handling required
- Automatic propagation via duck typing

### 4. Flexible Source Support
The abstract `url` property in FileObject allows:
- Regular files: url = filename
- CouchDB files: url = row.url field
- Future sources: implement url property as needed

## Usage in Queries

### Find Taxa by Source URL

```python
import couchdb

server = couchdb.Server("http://localhost:5984")
server.resource.credentials = ("admin", "secret")
db = server["mycobank_taxa"]

# Find all taxa from a specific URL
target_url = "http://example.com/article"

for doc_id in db:
    doc = db[doc_id]
    if doc.get("type") == "taxon":
        if doc.get("source", {}).get("url") == target_url:
            print(f"Found taxon: {doc['serial_number']}")
            for para in doc.get("paragraphs", []):
                print(f"  - {para['label']}: {para['body'][:50]}...")
```

### Group Taxa by URL

```python
from collections import defaultdict

taxa_by_url = defaultdict(list)

for doc_id in db:
    doc = db[doc_id]
    if doc.get("type") == "taxon":
        url = doc.get("source", {}).get("url")
        if url:
            taxa_by_url[url].append(doc)

for url, taxa in taxa_by_url.items():
    print(f"{url}: {len(taxa)} taxa")
```

## Migration Notes

### From Non-URL Pipeline

If you have existing code that doesn't use URLs:

1. **Line objects**: URL will be `None` for non-CouchDB sources
2. **Paragraph.as_dict()**: Will include `'url': None`
3. **Taxon documents**: Will have `"url": null` in source and paragraphs
4. **Idempotent keys**: Will use `"no_url"` in the composite key

### Adding URL to Existing Data

If you need to add URLs to existing taxa:

```python
# Update existing documents with URLs
for doc_id in db:
    doc = db[doc_id]
    if doc.get("type") == "taxon":
        # Extract doc_id to lookup URL
        source_doc_id = doc.get("source", {}).get("doc_id")

        # Lookup URL from source database
        source_db = server["mycobank_annotations"]
        source_doc = source_db.get(source_doc_id)
        url = source_doc.get("url") if source_doc else None

        # Update source metadata
        doc["source"]["url"] = url

        # Update all paragraphs
        for para in doc.get("paragraphs", []):
            para["url"] = url

        db.save(doc)
```

## See Also

- [TAXON_PIPELINE_README.md](TAXON_PIPELINE_README.md) - Complete pipeline documentation
- [COUCHDB_INTEGRATION_SUMMARY.md](COUCHDB_INTEGRATION_SUMMARY.md) - CouchDB integration overview
- [EXTRACTING_TAXON_OBJECTS.md](EXTRACTING_TAXON_OBJECTS.md) - Taxon extraction guide
