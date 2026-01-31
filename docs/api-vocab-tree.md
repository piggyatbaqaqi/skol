# Vocabulary Tree API

The Vocabulary Tree API provides access to hierarchical vocabulary data extracted from taxonomic JSON representations stored in CouchDB. The vocabulary tree organizes terms by their position in the JSON structure, enabling UI components like cascading dropdown menus.

## Overview

Vocabulary trees are built by the `build_vocab_tree` command and stored in Redis. Each tree version is timestamped and accessible via the API. A "latest" pointer always references the most recent tree.

## Endpoints

### GET /api/vocab-tree/

Retrieve the vocabulary tree.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `version` | string | latest | Specific version string (e.g., `2026_01_26_10_30`) |
| `path` | string | (root) | Dot-separated path to navigate to a subtree (e.g., `pileus.shape`) |
| `depth` | integer | unlimited | Maximum depth to return; deeper levels show `{"...": "N children"}` |

**Response:**

```json
{
    "version": "2026_01_26_12_38",
    "created_at": "2026-01-26T12:38:01.311273",
    "path": null,
    "tree": {
        "pileus": {
            "shape": {
                "convex": {},
                "flat": {},
                "umbonate": {}
            },
            "color": {
                "brown": {},
                "white": {}
            }
        },
        "stipe": {
            "attachment": {
                "free": {},
                "adnate": {}
            }
        }
    },
    "stats": {
        "total_nodes": 980,
        "max_depth": 5,
        "level_counts": {"1": 67, "2": 248, "3": 399, "4": 166, "5": 100},
        "leaf_count": 525
    }
}
```

**Examples:**

```bash
# Get the latest full tree
curl http://localhost:8000/api/vocab-tree/

# Get a specific version
curl 'http://localhost:8000/api/vocab-tree/?version=2026_01_26_12_38'

# Get subtree at a specific path
curl 'http://localhost:8000/api/vocab-tree/?path=pileus.shape'

# Limit depth to 2 levels
curl 'http://localhost:8000/api/vocab-tree/?depth=2'

# Combine parameters
curl 'http://localhost:8000/api/vocab-tree/?path=conidial%20fungi&depth=3'
```

---

### GET /api/vocab-tree/versions/

List all available vocabulary tree versions with metadata.

**Response:**

```json
{
    "versions": [
        {
            "key": "skol:ui:menus_2026_01_26_12_38",
            "version": "2026_01_26_12_38",
            "created_at": "2026-01-26T12:38:01.311273",
            "stats": {
                "total_nodes": 980,
                "max_depth": 5,
                "level_counts": {"1": 67, "2": 248, "3": 399, "4": 166, "5": 100},
                "leaf_count": 525
            }
        },
        {
            "key": "skol:ui:menus_2026_01_26_12_32",
            "version": "2026_01_26_12_32",
            "created_at": "2026-01-26T12:32:43.674816",
            "stats": {
                "total_nodes": 980,
                "max_depth": 5,
                "level_counts": {"1": 67, "2": 248, "3": 399, "4": 166, "5": 100},
                "leaf_count": 525
            }
        }
    ],
    "count": 2,
    "latest": "skol:ui:menus_2026_01_26_12_38"
}
```

**Example:**

```bash
curl http://localhost:8000/api/vocab-tree/versions/
```

---

### GET /api/vocab-tree/children/

Get the immediate children at a specific path in the vocabulary tree. This endpoint is optimized for building cascading UI menus.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | string | (root) | Dot-separated path (e.g., `pileus.shape`). Omit for top-level terms. |
| `version` | string | latest | Specific version string |

**Response:**

```json
{
    "path": "pileus.shape",
    "children": ["convex", "flat", "umbonate", "campanulate"],
    "count": 4,
    "has_grandchildren": {
        "convex": false,
        "flat": false,
        "umbonate": true,
        "campanulate": false
    },
    "is_leaf": false
}
```

The `has_grandchildren` field indicates whether each child has further descendants, useful for determining if a UI element should show an expansion indicator.

**Examples:**

```bash
# Get top-level vocabulary terms
curl http://localhost:8000/api/vocab-tree/children/

# Get children under "pileus"
curl 'http://localhost:8000/api/vocab-tree/children/?path=pileus'

# Get children under "pileus.shape"
curl 'http://localhost:8000/api/vocab-tree/children/?path=pileus.shape'

# Get children from a specific version
curl 'http://localhost:8000/api/vocab-tree/children/?path=pileus&version=2026_01_26_12_32'
```

---

## Building Vocabulary Trees

Vocabulary trees are built using the `build_vocab_tree` command:

```bash
# Build from default database (skol_taxa_full_dev)
/opt/skol/bin/build_vocab_tree

# Build from a specific database
/opt/skol/bin/build_vocab_tree --db skol_taxa_full_dev

# Build with a custom version string
/opt/skol/bin/build_vocab_tree --version my_custom_version

# Build with TTL (auto-expire after 1 week)
/opt/skol/bin/build_vocab_tree --ttl 604800

# Dry run (build but don't save to Redis)
/opt/skol/bin/build_vocab_tree --dry-run
```

See `build_vocab_tree --help` for all options.

---

## Redis Storage

Vocabulary trees are stored in Redis with the following key structure:

| Key Pattern | Description |
|-------------|-------------|
| `skol:ui:menus_<version>` | The vocabulary tree data for a specific version |
| `skol:ui:menus_latest` | Pointer to the most recent tree key |

Each tree is stored as a JSON object containing:
- `tree`: The hierarchical vocabulary structure
- `stats`: Statistics about the tree (node counts, depth, etc.)
- `version`: The version string
- `created_at`: ISO timestamp of creation

---

## Error Responses

All endpoints return standard HTTP error codes:

| Status | Description |
|--------|-------------|
| 404 | Vocabulary tree or path not found |
| 500 | Internal server error (Redis connection issues, malformed data) |

Error response format:

```json
{
    "error": "Path not found: invalid.path"
}
```

---

## Use Cases

### Building Cascading Dropdown Menus

Use the `/children/` endpoint to lazily load menu options:

```javascript
// Load top-level options
const topLevel = await fetch('/api/vocab-tree/children/').then(r => r.json());
// topLevel.children = ["pileus", "stipe", "lamellae", ...]

// When user selects "pileus", load its children
const pileusChildren = await fetch('/api/vocab-tree/children/?path=pileus').then(r => r.json());
// pileusChildren.children = ["shape", "color", "surface", ...]

// Continue drilling down
const shapeOptions = await fetch('/api/vocab-tree/children/?path=pileus.shape').then(r => r.json());
// shapeOptions.children = ["convex", "flat", "umbonate", ...]
```

### Prefetching Tree Sections

Use the main `/vocab-tree/` endpoint with `depth` to prefetch multiple levels:

```javascript
// Fetch first 3 levels for faster initial rendering
const tree = await fetch('/api/vocab-tree/?depth=3').then(r => r.json());
```

### Displaying Version Information

Show users when the vocabulary was last updated:

```javascript
const versions = await fetch('/api/vocab-tree/versions/').then(r => r.json());
console.log(`Vocabulary last updated: ${versions.versions[0].created_at}`);
console.log(`Contains ${versions.versions[0].stats.total_nodes} terms`);
```
