# SKOL REST API Reference

This document provides an overview of all REST API endpoints available in the SKOL Django application.

## Base URL

All API endpoints are prefixed with `/api/`.

## Authentication

Most endpoints are public. Collection-related endpoints require session authentication (login via `/accounts/login/`).

---

## Semantic Search

### GET /api/embeddings/

List available embedding models stored in Redis.

**Response:**
```json
{
    "embeddings": ["skol:embedding:v1.0", "skol:embedding:v1.1"],
    "count": 2
}
```

### GET /api/embeddings/build/

Check if the configured embedding model exists.

**Query Parameters:** `embedding_name` (optional)

**Response:**
```json
{
    "status": "exists",
    "embedding_name": "skol:embedding:v1.1",
    "message": "Embedding exists with 1234 entries",
    "embedding_count": 1234
}
```

### POST /api/embeddings/build/

Trigger building the embedding model if it doesn't exist. This loads taxa descriptions from CouchDB, computes sBERT embeddings, and saves them to Redis.

**Request:**
```json
{
    "force": false,
    "embedding_name": "skol:embedding:v1.1"
}
```

**Response:**
```json
{
    "status": "complete",
    "embedding_name": "skol:embedding:v1.1",
    "message": "Embedding built successfully with 1234 entries",
    "embedding_count": 1234
}
```

**Status values:** `exists`, `complete`, `error`

### POST /api/search/

Perform semantic search against taxa descriptions.

**Request:**
```json
{
    "prompt": "red mushroom with white spots",
    "embedding_name": "skol:embedding:v1.1",
    "k": 3
}
```

**Response:**
```json
{
    "results": [
        {
            "Similarity": 0.95,
            "Title": "Amanita muscaria",
            "Description": "...",
            "Feed": "SKOL",
            "URL": "..."
        }
    ],
    "count": 3,
    "prompt": "red mushroom with white spots",
    "embedding_name": "skol:embedding:v1.1",
    "k": 3
}
```

### GET /api/search/nomenclature/

Search taxa by regex pattern on the nomenclature (taxon) field.

**Query Parameters:**
- `pattern` (required): Regex pattern to match against nomenclature
- `embedding_name` (required): Embedding model name (determines which dataset to search)
- `limit` (default: `20`, max: `200`): Maximum results to return

**Response:**
```json
{
    "results": [
        {
            "Similarity": null,
            "Title": "Amanita muscaria",
            "Description": "...",
            "Feed": "SKOL",
            "URL": "...",
            "taxon_id": "taxon_abc123",
            "ResultType": "taxon"
        }
    ],
    "count": 5,
    "pattern": "Amanita",
    "embedding_name": "skol:embedding:v1.1"
}
```

**Notes:**
- Search is case-insensitive
- `Similarity` is always `null` (this is pattern matching, not similarity search)
- Invalid regex patterns return `400` with an error message
- Results use the same structure as `POST /api/search/` for frontend compatibility

---

## Taxa & PDFs

### GET /api/taxa/{taxa_id}/

Get taxa document information including source PDF details.

**Query Parameters:** `taxa_db` (default: `'skol_taxa_dev'`)

**Response:**
```json
{
    "taxon_id": "taxon_abc123",
    "taxa_db": "skol_taxa_dev",
    "Title": "Amanita muscaria",
    "Description": "Pileus convex to plane...",
    "Feed": "CouchDB Taxa",
    "URL": "https://...",
    "PDFDbName": "skol_dev",
    "PDFDocId": "doc_xyz789",
    "PDFPage": 42,
    "PDFLabel": "42",
    "LineNumber": 100,
    "ParagraphNumber": 5,
    "EmpiricalPageNumber": "127"
}
```

### GET /api/pdf/{db_name}/{doc_id}/

Retrieve PDF attachment from CouchDB.

### GET /api/pdf/{db_name}/{doc_id}/{attachment_name}/

Retrieve specific PDF attachment by name.

### GET /api/taxa/{taxa_id}/pdf/

Retrieve PDF for a taxa document's source reference.

### GET /api/taxa/{taxa_id}/context/

Retrieve windowed source text with highlight markers for the Source Context Viewer. Supports scrolling through Nomenclature and Description spans.

**Query Parameters:**
- `field`: `'nomenclature'` or `'description'` (default: `'description'`)
- `span_index`: Which span to show (default: `0`)
- `context_chars`: Characters of context before/after span (default: `500`)
- `taxa_db`: Database name (default: `'skol_taxa_dev'`)

**Response:**
```json
{
    "source_text": "...text with <mark>highlighted</mark> region...",
    "highlight_start": 234,
    "highlight_end": 567,
    "has_gap_before": false,
    "has_gap_after": true,
    "gap_size_before": 0,
    "gap_size_after": 150,
    "prev_span_index": null,
    "next_span_index": 1,
    "pdf_page": 35,
    "pdf_label": "35",
    "empirical_page": "127",
    "total_spans": 2,
    "span_index": 0
}
```

**Navigation:** Use `prev_span_index` and `next_span_index` to scroll through spans. Gap indicators show when there's intervening text between spans.

---

## Vocabulary Tree

Hierarchical vocabulary data for building UI menus.

### GET /api/vocab-tree/

Get the vocabulary tree (full or filtered).

**Query Parameters:** `version`, `path`, `depth`

### GET /api/vocab-tree/versions/

List all available vocabulary tree versions.

### GET /api/vocab-tree/children/

Get children at a specific path (optimized for cascading menus).

**Query Parameters:** `path`, `version`

### GET /api/vocab-tree/build/

Check if the vocabulary tree exists in Redis.

**Response:**
```json
{
    "status": "exists",
    "redis_key": "skol:ui:menus_2026_01_29_12_30",
    "version": "2026_01_29_12_30",
    "message": "Vocabulary tree exists with 5432 nodes",
    "stats": {
        "total_nodes": 5432,
        "max_depth": 5,
        "leaf_count": 3210
    }
}
```

### POST /api/vocab-tree/build/

Trigger building the vocabulary tree if it doesn't exist. This reads JSON representations from CouchDB and builds a hierarchical vocabulary structure for UI menus.

**Request:**
```json
{
    "force": false,
    "db_name": "skol_taxa_full_dev"
}
```

**Response:**
```json
{
    "status": "complete",
    "redis_key": "skol:ui:menus_2026_01_29_12_30",
    "message": "Vocabulary tree built successfully with 5432 nodes",
    "stats": {
        "total_nodes": 5432,
        "max_depth": 5,
        "leaf_count": 3210
    }
}
```

**Status values:** `exists`, `complete`, `error`

See [api-vocab-tree.md](api-vocab-tree.md) for detailed documentation.

---

## Classifiers

Decision tree classifiers for distinguishing taxa based on their features.

### POST /api/classifier/text/

Build a decision tree classifier using TF-IDF features from description text.

**Request:**
```json
{
    "taxa_ids": ["taxon_abc123", "taxon_def456", ...],
    "top_n": 30,
    "max_depth": 10,
    "min_df": 1,
    "max_df": 1.0
}
```

**Response:**
```json
{
    "features": [
        {"name": "pileus", "importance": 0.15, "display_text": "pileus"},
        {"name": "convex", "importance": 0.12, "display_text": "convex"}
    ],
    "metadata": {
        "n_classes": 5,
        "n_features": 150,
        "tree_depth": 8,
        "taxa_count": 5
    },
    "tree_json": { ... }
}
```

### POST /api/classifier/json/

Build a decision tree classifier using structured JSON annotation features (key=value pairs from `json_annotated` field).

**Request:**
```json
{
    "taxa_ids": ["taxon_abc123", "taxon_def456", ...],
    "top_n": 30,
    "max_depth": 10,
    "min_df": 1,
    "max_df": 1.0
}
```

**Response:**
```json
{
    "features": [
        {"name": "pileus_shape=convex", "importance": 0.18, "display_text": "pileus shape convex"},
        {"name": "stipe_color=white", "importance": 0.14, "display_text": "stipe color white"}
    ],
    "metadata": {
        "n_classes": 5,
        "n_features": 85,
        "tree_depth": 6,
        "taxa_count": 5
    },
    "tree_json": { ... }
}
```

**Note:** The JSON classifier uses the `skol_taxa_full_dev` database which contains structured annotations.

---

## Collections (Authenticated)

Research collection management for organizing searches.

### GET /api/identifier-types/

List available external identifier types (iNat, MO, GenBank, etc.).

### GET /api/fungaria/

List fungaria/herbaria from Index Herbariorum registry (for fungarium identifier dropdown).

**Query Parameters:** `search` (filter by code/name), `limit`

**Response:**
```json
{
    "fungaria": [
        {
            "code": "NY",
            "organization": "New York Botanical Garden",
            "num_fungi": 850000,
            "location": "New York, NY, USA",
            "collection_url": "https://...",
            "web_url": "https://..."
        }
    ],
    "count": 50,
    "total_in_registry": 3500
}
```

### Collections CRUD

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/collections/` | GET | List user's collections |
| `/api/collections/` | POST | Create new collection |
| `/api/collections/{id}/` | GET | Get collection details |
| `/api/collections/{id}/` | PUT | Update collection |
| `/api/collections/{id}/` | DELETE | Delete collection |
| `/api/collections/user/{username}/` | GET | List user's collections by username |
| `/api/collections/user-id/{user_id}/` | GET | List user's collections by numeric ID (stable across renames) |
| `/api/collections/{id}/flag/` | POST | Flag collection as inappropriate (any authenticated user) |
| `/api/collections/{id}/flag/` | DELETE | Remove all flags from collection (admin only) |

### Search History

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/collections/{id}/searches/` | GET | List searches in collection |
| `/api/collections/{id}/searches/` | POST | Add search to collection |
| `/api/collections/{id}/searches/{sid}/` | GET | Get search details |
| `/api/collections/{id}/searches/{sid}/` | DELETE | Delete search |

### External Identifiers

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/collections/{id}/identifiers/` | GET | List identifiers |
| `/api/collections/{id}/identifiers/` | POST | Add identifier |
| `/api/collections/{id}/identifiers/{iid}/` | GET | Get identifier |
| `/api/collections/{id}/identifiers/{iid}/` | DELETE | Delete identifier |

---

## Discussion / Comments (Authenticated)

Threaded discussion system for collections, backed by CouchDB.

### GET /api/collections/{id}/comments/

Fetch all comments for a collection in tree order. Deleted comments are sanitized to show `[deleted]` with author info scrubbed.

**Response includes:** `comments` (flat list in sort order), `is_owner`, `is_admin`, `current_user_id`

### POST /api/collections/{id}/comments/

Create a new comment.

**Request:**
```json
{
    "body": "This looks like Geastrum triplex.",
    "nomenclature": "Geastrum triplex",
    "parent_path": "/3/"
}
```

`nomenclature` and `parent_path` are optional. Omit `parent_path` for root-level comments.

### GET /api/collections/{id}/comments/count/

Lightweight count of non-deleted comments. No authentication required.

**Response:**
```json
{
    "count": 12,
    "collection_id": 110439105
}
```

### PUT /api/collections/{id}/comments/{comment_id}/

Edit a comment (author only). Previous version is pushed to `edit_history`.

**Request:**
```json
{
    "body": "Updated text",
    "nomenclature": "Updated nomenclature"
}
```

### DELETE /api/collections/{id}/comments/{comment_id}/

Soft-delete a comment. Author, collection owner, or admin.

### POST /api/collections/{id}/comments/{comment_id}/flag/

Flag a comment as inappropriate (any authenticated user). Idempotent.

### DELETE /api/collections/{id}/comments/{comment_id}/flag/

Remove all flags from a comment (collection owner or admin only).

### POST /api/collections/{id}/comments/{comment_id}/hide/

Hide a flagged comment (collection owner or admin only).

### DELETE /api/collections/{id}/comments/{comment_id}/hide/

Unhide a comment (collection owner or admin only).

### POST /api/collections/{id}/comments/{comment_id}/copy-nomenclature/

Copy a comment's nomenclature to the collection's master nomenclature field (collection owner only). Syncs to CouchDB and records a nomenclature change event.

**Response:**
```json
{
    "status": "ok",
    "nomenclature": "Geastrum triplex",
    "collection_id": 110439105
}
```

---

## Error Responses

All endpoints return errors in this format:

```json
{
    "error": "Description of what went wrong"
}
```

Common HTTP status codes:
- `400` - Bad request (missing/invalid parameters)
- `401` - Authentication required
- `403` - Permission denied
- `404` - Resource not found
- `500` - Internal server error
