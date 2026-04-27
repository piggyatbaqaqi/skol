# SKOL REST API Reference

This document provides an overview of all REST API endpoints available in the SKOL Django application.

## Base URL

All API endpoints are prefixed with `/api/`.

## Authentication

Most endpoints are public. Collection-related endpoints require session authentication (login via `/accounts/login/`).

---

## Experiments

Named experiment configurations that tie together databases, Redis keys, and classifier models. Experiments are stored in the `skol_experiments` CouchDB database.

### GET /api/experiments/

List all available experiments.

**Response:**
```json
{
    "experiments": [
        {
            "name": "production",
            "notes": "Current production pipeline: logistic regression on hand-annotated training data",
            "status": "deployed"
        },
        {
            "name": "jats_v1",
            "notes": "Test JATS-derived training annotations",
            "status": "evaluated"
        }
    ],
    "count": 2
}
```

**Experiment-aware views:** When a user selects an experiment (saved as `default_experiment` in user settings), several views automatically use the experiment's configured databases and Redis keys:

| View | Experiment field used |
|------|----------------------|
| `GET /api/taxa/{id}/` | `databases.taxa` (taxa database) |
| `GET /api/vocab-tree/` | `redis_keys.menus` (menus pointer key) |
| `GET /api/vocab-tree/build/` | `redis_keys.menus`, `databases.taxa_full` |
| `POST /api/vocab-tree/build/` | `redis_keys.menus`, `databases.taxa_full` |
| Sources page (`/sources/`) | `databases.ingest` (ingestion database) |

If no experiment is set or the experiment is not found, these views fall back to the `production` experiment defaults.

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
    "k": 3,
    "nomenclature_pattern": "^Amanita",
    "project_slugs": ["jsmith/field-guide"]
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `prompt` | Yes | Description text to search for |
| `embedding_name` | Yes | Embedding model name |
| `k` | No | Number of results (default: 3, max: 200) |
| `nomenclature_pattern` | No | Regex to pre-filter taxa by nomenclature before similarity ranking. Case-insensitive. |
| `project_slugs` | No | List of namespaced project slugs (`username/slug`). When provided, collection-type results are restricted to collections belonging to any of the listed projects. Taxon results are unaffected. Unknown slugs are silently ignored. An empty list applies no filter. |

When `nomenclature_pattern` is provided, only taxa whose nomenclature matches the
regex are considered for similarity ranking. This is substantially faster than
searching the full embedding space when restricting to a genus or family.

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
    "nomenclature_pattern": "^Amanita",
    "k": 3
}
```

**Notes:**
- `nomenclature_pattern` is only included in the response when it was provided in the request
- Invalid regex patterns return `400` with an error message
- If the pattern matches no taxa, returns `{"results": [], "count": 0, ...}`

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

**Query Parameters:** `taxa_db` (default: user's experiment `databases.taxa`, or `'skol_taxa_dev'`)

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
| `/api/collections/{id}/hide/` | POST | Hide collection (admin only) |
| `/api/collections/{id}/hide/` | DELETE | Unhide collection (admin only) |

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
| `/api/collections/{id}/post-inat-comment/` | POST | Post description as iNaturalist comment |

**POST /api/collections/{id}/post-inat-comment/**: Posts the collection's
description (and nomenclature, if present and not "Unknown") as a comment on the
linked iNaturalist observation. Requires the user to be the collection owner with
a connected iNaturalist account that has `write` scope.

### Measurement Sets

Record specimen measurements (e.g., spore dimensions) for a collection. Each collection can have multiple measurement sets keyed by feature name (e.g., "spores", "basidia", "cystidia"). Raw measurements are stored as JSON; statistics are computed client-side.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/collections/{id}/measurements/` | GET | List measurement sets |
| `/api/collections/{id}/measurements/` | POST | Create measurement set (owner only) |
| `/api/collections/{id}/measurements/{mid}/` | GET | Get measurement set |
| `/api/collections/{id}/measurements/{mid}/` | PUT | Update measurement set (owner only) |
| `/api/collections/{id}/measurements/{mid}/` | DELETE | Delete measurement set (owner only) |

**GET Response:**
```json
{
    "measurement_sets": [
        {
            "id": 1,
            "feature": "spores",
            "is_2d": true,
            "report_q": true,
            "measurements": [
                {"length": 8.5, "width": 6.5},
                {"length": 11.2, "width": 7.8}
            ],
            "created_at": "2026-03-03T12:00:00Z",
            "updated_at": "2026-03-03T12:05:00Z"
        }
    ],
    "count": 1,
    "collection_id": 110439105
}
```

**POST/PUT Request:**
```json
{
    "feature": "spores",
    "is_2d": true,
    "report_q": true,
    "measurements": [
        {"length": 8.5, "width": 6.5},
        {"length": 11.2, "width": 7.8}
    ]
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `feature` | string | `"spores"` | Feature name (unique per collection) |
| `is_2d` | bool | `true` | `true` for length × width, `false` for length only |
| `report_q` | bool | `true` | Whether to include Q (length/width ratio) in formatted output |
| `measurements` | array | `[]` | Raw measurements: `[{"length": N, "width": N}, ...]`. Width is optional for 1D mode. |

**Validation:** Each measurement must have a positive `length`. If `width` is present, it must also be positive. Duplicate feature names within a collection return `409 Conflict`.

**Statistics (computed client-side):** The Metrics UI computes quartile-based statistics from raw measurements in standard mycological notation: `spores (min-) Q1 - Q3 (-max) × (min-) Q1 - Q3 (-max) µm, Q: (min-)Q1-Q3(max)`. The larger of two entered values is assigned as length.

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

## User Settings (Authenticated)

Persistent user preferences for search and display.

### GET /api/user-settings/

Get the current user's settings.

**Response:**
```json
{
    "default_embargo_days": 0,
    "default_embedding": "",
    "default_k": 3,
    "results_per_page": 10,
    "feature_taxa_count": 6,
    "feature_max_tree_depth": 10,
    "default_experiment": "production",
    "receive_admin_summary": false,
    "default_project_slugs": ["jsmith/field-guide"],
    "created_at": "2026-01-15T12:00:00Z",
    "updated_at": "2026-02-17T08:30:00Z"
}
```

### PUT /api/user-settings/

Update the current user's settings (partial updates supported).

**Request:**
```json
{
    "default_k": 20,
    "results_per_page": 10
}
```

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `default_embargo_days` | int | 0 | 0–365 | Default embargo period for new collections |
| `default_embedding` | string | `""` | — | Preferred embedding model |
| `default_experiment` | string | `"production"` | — | Active experiment name (from `skol_experiments`) |
| `default_k` | int | 3 | 1–100 | Default number of search results to fetch |
| `results_per_page` | int | 10 | 5–50 | Number of results to display per page (client-side pagination) |
| `feature_taxa_count` | int | 6 | 2–50 | Number of taxa for feature lists |
| `feature_max_tree_depth` | int | 10 | 1–20 | Maximum depth for feature tree |
| `receive_admin_summary` | bool | false | — | Opt in to daily admin summary email |
| `default_project_slugs` | string[] | `[]` | — | Namespaced slugs of projects to auto-add new collections to |

Settings are created automatically on first access (get-or-create pattern).

When `default_project_slugs` is included in a PUT body, the list replaces the
user's current default projects (unknown slugs are silently ignored). Omitting
the key leaves existing defaults unchanged.

---

## Projects

Projects are public namespaced groupings of collections.  Any authenticated
user may create projects, and any authenticated user may add or remove
collections (democratic governance model).  Only site admins can delete
projects.

Slugs are namespaced per creator: `username/slug` (e.g. `jsmith/field-guide`).

### GET /api/projects/

List all projects site-wide (public).

**Query parameters:**

| Parameter | Description |
|-----------|-------------|
| `q` | Search by project name or creator username (case-insensitive substring) |
| `collection_id` | Return only projects that contain the given collection ID |

**Response:**
```json
{
    "projects": [
        {
            "id": 1,
            "name": "Field Guide",
            "slug": "field-guide",
            "namespaced_slug": "jsmith/field-guide",
            "creator_username": "jsmith",
            "description": "A guide to local fungi",
            "collection_count": 12,
            "created_at": "2026-03-01T10:00:00Z"
        }
    ],
    "count": 1
}
```

### POST /api/projects/

Create a new project. Requires authentication.

**Request:**
```json
{
    "name": "Field Guide",
    "description": "A guide to local fungi"
}
```

**Response:** `201 Created` — the created project object (same schema as list item).

The slug is auto-generated from the name (e.g. `"Field Guide"` → `field-guide`).
Collisions within the same creator namespace append `-2`, `-3`, etc.

### GET /api/projects/{username}/{slug}/

Retrieve a single project with its current collection memberships (public).

**Response:**
```json
{
    "id": 1,
    "name": "Field Guide",
    "slug": "field-guide",
    "namespaced_slug": "jsmith/field-guide",
    "creator_username": "jsmith",
    "description": "A guide to local fungi",
    "notes": "Started at the 2026 NAMA foray.",
    "collection_count": 1,
    "created_at": "2026-03-01T10:00:00Z",
    "memberships": [
        {
            "collection_id": 42,
            "collection_name": "Amanita muscaria",
            "added_by_username": "jsmith",
            "added_at": "2026-03-05T14:00:00Z"
        }
    ],
    "notes_log": [
        {
            "id": 7,
            "changed_by_username": "jsmith",
            "changed_at": "2026-04-10T09:15:00Z",
            "diff": "--- notes (before)\n+++ notes (after)\n@@ -0,0 +1 @@\n+Started at the 2026 NAMA foray."
        }
    ]
}
```

`notes_log` is ordered most-recent-first.  Each entry records the username,
timestamp, and the change as a unified diff (output of Python's
`difflib.unified_diff`).  A log entry is only created when the value actually
changes; a no-op PATCH to `notes` produces no entry.

### PATCH /api/projects/{username}/{slug}/

Update mutable text fields on a project. Requires authentication.

Accepted fields: `notes`, `description`. Unknown fields are silently ignored.
The project's `name`, `slug`, and `creator` cannot be changed via this endpoint.

**Request body:**
```json
{ "notes": "Updated field notes." }
```

**Response:** `200 OK` with the updated project object (same schema as GET above).

### GET /api/projects/{username}/{slug}/export/

Export a project as a ZIP archive.  Requires authentication.

The archive contains:
- `project.json` — project metadata, current memberships, and removal audit log
- `collections/<id>.json` — full collection records for each current member
- `couchdb_collections/<id>.json` — CouchDB treatment docs where available

**Response:** `200 OK` with `Content-Type: application/zip` and
`Content-Disposition: attachment; filename="<slug>.zip"`.

### POST /api/import/

Import a SKOL export ZIP. Requires authentication.

Upload via `multipart/form-data` with field `file` containing the ZIP.

The endpoint auto-detects the archive type:
- `project.json` present → project import
- `user.json` present → 400 (not yet supported)
- Neither → 400 Unknown format

**Request:** `multipart/form-data`, field `file` (ZIP binary).

**Response (project import) `200 OK`:**
```json
{
    "type": "project",
    "project_name": "My Project",
    "namespaced_slug": "alice/my-project",
    "project_url": "/projects/alice/my-project/",
    "collections_imported": 2,
    "collections_linked": 1
}
```

- `collections_imported` — collections that did not exist locally and were created.
- `collections_linked` — collections that already existed and were linked to the new project.
- If the original project creator's username exists on this instance, they become the creator; otherwise the importing user is the creator.
- If the slug already belongs to the resolved creator, a numeric suffix is appended (`-2`, `-3`, …).
- Existing collections are linked but their data is never overwritten.

**Error responses:** `400 Bad Request` (no file, invalid ZIP, unknown format), `403 Forbidden` (unauthenticated).

### POST /api/projects/{username}/{slug}/collections/{collection_id}/

Add a collection to a project. Requires authentication. Idempotent.

**Response:**
- `201 Created` — collection was added
- `200 OK` — collection was already a member

### DELETE /api/projects/{username}/{slug}/collections/{collection_id}/

Remove a collection from a project. Requires authentication.
An audit log record is created automatically.

**Response:** `200 OK`

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
