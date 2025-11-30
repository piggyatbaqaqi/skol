# Taxon Hierarchical Clustering with Neo4j

This guide explains how to use the `TaxonClusterer` class to perform hierarchical clustering on taxa using SBERT embeddings and store the results in Neo4j.

## Overview

The `TaxonClusterer` class:
1. Loads SBERT embeddings from Redis
2. Performs agglomerative hierarchical clustering
3. Stores the resulting tree structure in Neo4j

The resulting graph contains:
- **Taxon nodes** (leaf nodes): Individual taxa from your dataset
- **Pseudoclade nodes** (internal nodes): Hierarchical groupings of taxa
- **PARENT_OF relationships**: Links between parent and child nodes, labeled with cosine similarity distances

## Prerequisites

### Required Services

1. **Redis** running at `localhost:6379` with embeddings stored
2. **Neo4j** running at `localhost:7687`

### Required Python Packages

```bash
pip install redis neo4j scipy numpy
```

## Data Format

### Redis Embeddings

Embeddings should be stored in Redis at key `skol:embedding:v1.1` (configurable) as pickled data in one of these formats:

**Format 1: Structured dictionary**
```python
{
    'embeddings': np.array([[0.1, 0.2, ...], [0.3, 0.4, ...]]),
    'taxon_names': ['Amanita muscaria', 'Boletus edulis', ...]
}
```

**Format 2: Simple mapping**
```python
{
    'Amanita muscaria': np.array([0.1, 0.2, ...]),
    'Boletus edulis': np.array([0.3, 0.4, ...]),
    ...
}
```

**Format 3: Raw numpy array**
```python
np.array([[0.1, 0.2, ...], [0.3, 0.4, ...]])
# Taxon names will be auto-generated as Taxon_0, Taxon_1, etc.
```

## Usage

### Basic Example

```python
from taxon_clusterer import TaxonClusterer

# Initialize the clusterer
clusterer = TaxonClusterer(
    redis_host="localhost",
    redis_port=6379,
    redis_db=0,
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
)

try:
    # Load embeddings from Redis
    clusterer.load_embeddings("skol:embedding:v1.1")

    # Perform clustering
    clusterer.cluster(method="average", metric="cosine")

    # Store in Neo4j with root named "Fungi"
    clusterer.store_in_neo4j(root_name="Fungi", clear_existing=True)

    print("✓ Clustering complete!")

finally:
    clusterer.close()
```

### Using Context Manager

```python
with TaxonClusterer(
    redis_host="localhost",
    redis_port=6379,
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
) as clusterer:
    clusterer.load_embeddings("skol:embedding:v1.1")
    clusterer.cluster(method="average", metric="cosine")
    clusterer.store_in_neo4j(root_name="Fungi")
```

## Clustering Methods

The `cluster()` method supports different linkage methods:

```python
# Average linkage (UPGMA) - recommended for taxonomic data
clusterer.cluster(method="average", metric="cosine")

# Single linkage (minimum distance)
clusterer.cluster(method="single", metric="cosine")

# Complete linkage (maximum distance)
clusterer.cluster(method="complete", metric="cosine")

# Ward's method (minimizes variance)
clusterer.cluster(method="ward", metric="euclidean")
```

### Distance Metrics

- `"cosine"` - Cosine distance (recommended for SBERT embeddings)
- `"euclidean"` - Euclidean distance
- `"cityblock"` - Manhattan distance

## Neo4j Graph Structure

### Node Types

**Taxon (Leaf Nodes)**

Taxon nodes include metadata from the original `Raw_Data_Index.to_dict()` fields:

```cypher
(:Taxon {
    name: "Amanita muscaria",
    node_id: 42,
    source: "doc_123",
    url: "couchdb://localhost:5984/mycobank_taxa/doc_123",
    db_name: "mycobank_taxa",
    line_number: 15,
    paragraph_number: 2,
    page_number: 42,
    empirical_page_number: "40",
    description: "Pileus 5-10 cm broad, convex..."
})
```

The metadata fields may include:
- `name`: Taxon nomenclature from the 'taxon' field (concatenated nomenclature paragraphs)
- `source`: Source document ID
- `url`: Source document URL (e.g., CouchDB URL)
- `db_name`: Source database name
- `filename`: Source file path (for file-based sources)
- `row`: Row index in the original data (for file-based sources)
- `line_number`: Line number of first nomenclature paragraph
- `paragraph_number`: Paragraph number of first nomenclature paragraph
- `page_number`: Page number of first nomenclature paragraph
- `empirical_page_number`: Empirical page number from source document
- `description`: Full taxon description text (concatenated description paragraphs)

**Pseudoclade (Internal Nodes)**
```cypher
(:Pseudoclade {
    name: "Fungi",              // Root clade
    node_id: 150,
    count: 100                  // Number of descendant taxa
})

(:Pseudoclade {
    name: "Pseudoclade_1",      // Generated name
    node_id: 125,
    count: 50
})
```

### Relationship Type

**PARENT_OF**
```cypher
(:Pseudoclade)-[:PARENT_OF {distance: 0.234}]->(:Taxon)
(:Pseudoclade)-[:PARENT_OF {distance: 0.156}]->(:Pseudoclade)
```

The `distance` property represents the cosine similarity distance at which the child node joins the parent clade.

### Example Graph Structure

```
(Fungi:Pseudoclade)
    ├─[:PARENT_OF {distance: 0.234}]→ (Pseudoclade_1:Pseudoclade)
    │   ├─[:PARENT_OF {distance: 0.156}]→ (Amanita muscaria:Taxon {
    │   │                                     source: "doc_123",
    │   │                                     url: "couchdb://localhost:5984/mycobank_taxa/doc_123",
    │   │                                     line_number: 15,
    │   │                                     description: "Pileus 5-10 cm..."
    │   │                                  })
    │   └─[:PARENT_OF {distance: 0.189}]→ (Amanita pantherina:Taxon {...})
    │
    └─[:PARENT_OF {distance: 0.289}]→ (Pseudoclade_2:Pseudoclade)
        ├─[:PARENT_OF {distance: 0.098}]→ (Boletus edulis:Taxon {...})
        └─[:PARENT_OF {distance: 0.123}]→ (Boletus badius:Taxon {...})
```

Each Taxon node contains all metadata fields from the original data source, allowing for rich queries based on source, description content, and other attributes.

## Querying the Neo4j Database

### Get All Taxa

```cypher
MATCH (t:Taxon)
RETURN t.name
ORDER BY t.name
```

### Get All Taxa Under a Specific Clade

```cypher
MATCH (clade:Pseudoclade {name: "Fungi"})-[:PARENT_OF*]->(t:Taxon)
RETURN t.name
ORDER BY t.name
```

Using Python:
```python
taxa = clusterer.get_subtree("Fungi")
print(f"Found {len(taxa)} taxa")
```

### Get Path from Root to Specific Taxon

```cypher
MATCH path = (root:Pseudoclade)-[:PARENT_OF*]->(t:Taxon {name: "Amanita muscaria"})
WHERE NOT (root)<-[:PARENT_OF]-()
RETURN nodes(path)
```

Using Python:
```python
path = clusterer.get_tree_path("Amanita muscaria")
for node_name, distance in path:
    print(f"{node_name} (distance: {distance:.4f})")
```

### Find Closest Relatives

```cypher
// Find siblings (taxa with same parent)
MATCH (parent)-[:PARENT_OF]->(target:Taxon {name: "Amanita muscaria"})
MATCH (parent)-[:PARENT_OF]->(sibling:Taxon)
WHERE sibling <> target
RETURN sibling.name, parent.name
ORDER BY sibling.name
```

### Get Clade Statistics

```cypher
MATCH (p:Pseudoclade)
RETURN p.name, p.count
ORDER BY p.count DESC
```

### Find Most Distant Taxa

```cypher
MATCH (t1:Taxon), (t2:Taxon)
WHERE t1 <> t2
MATCH path = shortestPath((t1)-[:PARENT_OF*]-(t2))
WITH t1, t2, relationships(path) as rels
RETURN t1.name, t2.name, reduce(s = 0, r IN rels | s + r.distance) as total_distance
ORDER BY total_distance DESC
LIMIT 10
```

### Query by Metadata Fields

**Find taxa from a specific database**
```cypher
MATCH (t:Taxon)
WHERE t.db_name = "mycobank_taxa"
RETURN t.name, t.description, t.url
LIMIT 10
```

**Search taxa descriptions**
```cypher
MATCH (t:Taxon)
WHERE t.description CONTAINS "pileus"
RETURN t.name, t.description, t.url
ORDER BY t.name
```

**Find taxa by page number**
```cypher
MATCH (t:Taxon)
WHERE t.page_number >= 10 AND t.page_number <= 20
RETURN t.name, t.page_number, t.url
ORDER BY t.page_number
```

**Find taxa from a specific source document**
```cypher
MATCH (t:Taxon)
WHERE t.url CONTAINS "doc_123"
RETURN t.name, t.description, t.line_number
ORDER BY t.line_number
```

**Get all metadata for a taxon**
```cypher
MATCH (t:Taxon {name: "Amanita muscaria"})
RETURN properties(t) as metadata
```

## API Reference

### TaxonClusterer Class

#### Constructor

```python
TaxonClusterer(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password"
)
```

#### Methods

**load_embeddings(embedding_key: str) -> Tuple[np.ndarray, List[str]]**
- Load embeddings from Redis
- Returns: (embeddings array, taxon names list)

**cluster(method: str = "average", metric: str = "cosine") -> np.ndarray**
- Perform hierarchical clustering
- Returns: linkage matrix from scipy

**store_in_neo4j(root_name: str = "Fungi", clear_existing: bool = True)**
- Store clustering tree in Neo4j
- `root_name`: Name for the root pseudoclade
- `clear_existing`: Clear existing Taxon and Pseudoclade nodes

**get_subtree(clade_name: str) -> List[str]**
- Get all taxa descendant from a given clade
- Returns: List of taxon names

**get_tree_path(taxon_name: str) -> List[Tuple[str, float]]**
- Get path from root to a specific taxon
- Returns: List of (node_name, distance) tuples

**close()**
- Close Redis and Neo4j connections

## Troubleshooting

### Connection Issues

**Redis connection failed**
```python
# Check if Redis is running
redis-cli ping  # Should return PONG
```

**Neo4j connection failed**
```python
# Check Neo4j status
sudo systemctl status neo4j

# Check Neo4j browser at http://localhost:7474
```

### Data Issues

**No embeddings found in Redis**
```python
import redis
r = redis.Redis(host='localhost', port=6379, db=0)
print(r.exists('skol:embedding:v1.1'))  # Should return True
```

**Invalid embedding data format**
- Ensure embeddings are pickled using `pickle.dumps()`
- Verify data structure matches one of the supported formats

### Memory Issues

For large datasets (>10,000 taxa):
- Consider increasing Neo4j heap size in `neo4j.conf`:
  ```
  dbms.memory.heap.initial_size=2g
  dbms.memory.heap.max_size=4g
  ```
- Process embeddings in batches if needed

## Example: Complete Workflow

```python
from taxon_clusterer import TaxonClusterer

# Initialize
clusterer = TaxonClusterer(
    redis_host="localhost",
    redis_port=6379,
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
)

try:
    # 1. Load embeddings
    print("Loading embeddings...")
    embeddings, taxon_names = clusterer.load_embeddings("skol:embedding:v1.1")
    print(f"Loaded {len(taxon_names)} taxa")

    # 2. Perform clustering
    print("\nPerforming clustering...")
    linkage_matrix = clusterer.cluster(method="average", metric="cosine")

    # 3. Store in Neo4j
    print("\nStoring in Neo4j...")
    clusterer.store_in_neo4j(root_name="Fungi", clear_existing=True)

    # 4. Query examples
    print("\n--- Query Examples ---")

    # Get all taxa under Fungi
    all_taxa = clusterer.get_subtree("Fungi")
    print(f"\nTotal taxa under Fungi: {len(all_taxa)}")

    # Get path for first taxon
    if taxon_names:
        example_taxon = taxon_names[0]
        path = clusterer.get_tree_path(example_taxon)
        print(f"\nHierarchical path to {example_taxon}:")
        for i, (node_name, distance) in enumerate(path):
            indent = "  " * i
            print(f"{indent}└─ {node_name} (d={distance:.4f})")

    print("\n✓ Complete!")

finally:
    clusterer.close()
```

## References

- [scipy.cluster.hierarchy documentation](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
- [Neo4j Cypher documentation](https://neo4j.com/docs/cypher-manual/current/)
- [SBERT documentation](https://www.sbert.net/)
