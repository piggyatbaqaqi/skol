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

---

## Taxa & PDFs

### GET /api/taxa/{taxa_id}/

Get taxa document information.

### GET /api/pdf/{db_name}/{doc_id}/

Retrieve PDF attachment from CouchDB.

### GET /api/pdf/{db_name}/{doc_id}/{attachment_name}/

Retrieve specific PDF attachment by name.

### GET /api/taxa/{taxa_id}/pdf/

Retrieve PDF for a taxa document's source reference.

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

See [api-vocab-tree.md](api-vocab-tree.md) for detailed documentation.

---

## Collections (Authenticated)

Research collection management for organizing searches.

### GET /api/identifier-types/

List available external identifier types (iNat, MO, GenBank, etc.).

### Collections CRUD

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/collections/` | GET | List user's collections |
| `/api/collections/` | POST | Create new collection |
| `/api/collections/{id}/` | GET | Get collection details |
| `/api/collections/{id}/` | PUT | Update collection |
| `/api/collections/{id}/` | DELETE | Delete collection |
| `/api/collections/user/{username}/` | GET | List user's collections (public) |

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
