# Vocabulary Tree API

The Vocabulary Tree API provides access to hierarchical vocabulary data extracted from taxonomic JSON representations stored in CouchDB. The vocabulary tree organizes terms by their position in the JSON structure, enabling UI components like cascading dropdown menus.

## Overview

Vocabulary trees are built by the `build_vocab_tree` command and stored in Redis. The tree a user sees is the one at the key named by their experiment record's `redis_keys.menus` (`build_vocab_tree --experiment <name>` writes it). There is no shared "latest" pointer; see `docs/plans/experiment-record-canonical.md`.

## Endpoints

### GET /api/vocab-tree/

Retrieve the vocabulary tree.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `version` | string | experiment's tree | Specific version string (e.g., `2026_01_26_10_30`) |
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
    "count": 2
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
| `version` | string | experiment's tree | Specific version string |

Children are ordered by descending document support — how many
treatments contain that path — with ties alphabetical.  Children found in
fewer documents than the user's `vocab_min_support` setting (default 2,
the build floor; anonymous users get 2) are left out.

**Response:**

```json
{
    "path": "pileus.shape",
    "children": ["convex", "flat", "umbonate", "campanulate"],
    "counts": {"convex": 812, "flat": 301, "umbonate": 97, "campanulate": 44},
    "also": {},
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

The `has_grandchildren` field indicates whether each child has further descendants that meet the threshold, useful for determining if a UI element should show an expansion indicator.

The `also` field maps a child to the synonyms the synonym map renamed
into it, for the children that have any: at the top level, `{"Pileus":
["Cap"]}`.  The widget shows `Pileus (3,097; also: Cap)`.  Spelling
folds are not listed.  It is `{}` when the tree has no
`{<tree key>}:also` hash.

Since P2a the tree is built from each document's `json_canonical` and
keeps case (`Pileus`, `Conidia`).  Each `path` part matches a key
exactly, else ignoring case.  The build refuses, writing nothing, while
any document's `json_canonical` is missing or stale; run the
experiment's `canonicalize_json` step first.

If the tree is stored without its counts hash (below), the endpoint
returns a 500 naming the missing key.  That is a bug, not an old format:
rebuild the tree.  `GET /api/vocab-tree/build/` reports such a tree as
`not_found`, so the widget's build-and-retry path rebuilds it.

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

### Minimum support

Every node, at every level, found in fewer than `--min-support` documents
(`$VOCAB_MIN_SUPPORT`, default 2) is dropped before the tree is saved,
with its subtree.  A node's support never exceeds its parent's, so nothing
is orphaned.  On `production_v4_1` a floor of 2 keeps 185,689 of 1,886,415
nodes; the rest are mostly citations, OCR damage and one-off labels.
Each user can raise the threshold the menus apply (Settings,
`vocab_min_support`), never lower it below the floor.  This replaced
top-level-only singleton pruning (docs/plans/vocab-tree-narrowing.md,
Proposal 1).

---

## Redis Storage

Vocabulary trees are stored in Redis with the following key structure:

| Key Pattern | Description |
|-------------|-------------|
| `skol:ui:menus_<version>` | The vocabulary tree data for a specific version (a build without `--experiment`) |
| `skol:ui:menus_<experiment>` | An experiment's tree, the key its record's `redis_keys.menus` names |
| `{<tree key>}:counts` | A hash beside each tree, hash-tagged into the tree key's Redis Cluster slot so both are written in one transaction: field = JSON-encoded display path (e.g. `["pileus", "color"]`), value = documents containing it.  Written in the same transaction as the tree. |
| `{<tree key>}:also` | A hash beside each tree, in the same slot and transaction: field = JSON-encoded display path, value = JSON list of the synonyms renamed into it (P2a).  Absent when no synonym applied. |

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
