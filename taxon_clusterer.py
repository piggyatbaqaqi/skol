"""
Taxon Hierarchical Clustering with Neo4j Storage

This module provides functionality to:
1. Load SBERT embeddings from Redis
2. Apply agglomerative clustering to create a hierarchical taxonomy
3. Store the resulting tree structure in Neo4j with:
   - Leaf nodes as Taxon objects
   - Interior nodes as Pseudoclade objects
   - Relationships labeled with cosine similarity distances
"""

from typing import List, Dict, Tuple, Optional, Any
import pickle
import numpy as np
import pandas as pd
import redis
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import cosine
from neo4j import GraphDatabase
from dataclasses import dataclass

from taxon import get_ingest_field



@dataclass
class ClusterNode:
    """Represents a node in the clustering tree."""
    node_id: int
    is_leaf: bool
    taxon_name: Optional[str] = None
    left_child: Optional['ClusterNode'] = None
    right_child: Optional['ClusterNode'] = None
    distance: float = 0.0
    count: int = 1
    metadata: Optional[Dict[str, Any]] = None  # Metadata from Raw_Data_Index.to_dict()


class TaxonClusterer:
    """
    Hierarchical clustering of taxa using SBERT embeddings.

    This class loads embeddings from Redis, performs agglomerative clustering,
    and stores the resulting hierarchical tree in Neo4j.

    Example:
        >>> clusterer = TaxonClusterer(
        ...     redis_host="localhost",
        ...     redis_port=6379,
        ...     neo4j_uri="bolt://localhost:7687",
        ...     neo4j_user="neo4j",
        ...     neo4j_password="password"
        ... )
        >>> clusterer.load_embeddings("skol:embedding:v1.1")
        >>> clusterer.cluster(method="average", metric="cosine")
        >>> clusterer.store_in_neo4j(root_name="Fungi")
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password"
    ):
        """
        Initialize the TaxonClusterer.

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db

        # Connect to Redis
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=False
        )

        # Connect to Neo4j
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )

        # Data storage
        self.embeddings: Optional[np.ndarray] = None
        self.taxon_names: Optional[List[str]] = None
        self.taxon_metadata: Optional[List[Dict[str, Any]]] = None  # Metadata per taxon
        self.linkage_matrix: Optional[np.ndarray] = None
        self.root_node: Optional[ClusterNode] = None

        print(f"TaxonClusterer initialized")
        print(f"  Redis: {redis_host}:{redis_port}/{redis_db}")
        print(f"  Neo4j: {neo4j_uri}")

    def load_embeddings(self, embedding_key: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load embeddings from Redis.

        Args:
            embedding_key: Redis key containing pickled embeddings

        Returns:
            Tuple of (embeddings array, taxon names list, taxon metadata list)

        Raises:
            ValueError: If key doesn't exist or data is invalid
        """
        print(f"Loading embeddings from Redis key: {embedding_key}")

        if not self.redis_client.exists(embedding_key):
            raise ValueError(f"Redis key '{embedding_key}' does not exist")

        # Load pickled data from Redis
        pickled_data = self.redis_client.get(embedding_key)
        data = pickle.loads(pickled_data)

        # Assume it's a pandas DataFrame from EmbeddingsComputer
        try:
            assert isinstance(data, pd.DataFrame)
            # Extract embedding columns (F0, F1, F2, ...)
            embedding_cols = [col for col in data.columns if col.startswith('F')]
            self.embeddings = data[embedding_cols].values

            # Extract taxon names from 'taxon' field (nomenclature)
            # If 'taxon' column doesn't exist, fall back to 'description'
            if 'taxon' in data.columns:
                self.taxon_names = data['taxon'].tolist()
            else:
                self.taxon_names = data['description'].tolist()

            # Extract metadata from other columns
            self.taxon_metadata = []
            for _, row in data.iterrows():
                metadata = {}

                # Flatten source/ingest dict for neo4j storage.
                # Use ingest (new format) if present, fall back to source (old format)
                row_dict = row.to_dict()
                if 'ingest' in data.columns and isinstance(row['ingest'], dict):
                    ingest = row['ingest']
                    for key in ingest.keys():
                        metadata[f'ingest_{key}'] = ingest[key]
                if 'source' in data.columns and isinstance(row['source'], dict):
                    source = row['source']
                    for key in source.keys():
                        metadata[f'source_{key}'] = source[key]

                # Add other metadata fields
                if 'filename' in data.columns:
                    metadata['filename'] = row.get('filename')
                if 'row' in data.columns:
                    metadata['row'] = row.get('row')
                if 'line_number' in data.columns:
                    metadata['line_number'] = row.get('line_number')
                if 'paragraph_number' in data.columns:
                    metadata['paragraph_number'] = row.get('paragraph_number')
                if 'page_number' in data.columns:
                    metadata['page_number'] = row.get('page_number')
                if 'empirical_page_number' in data.columns:
                    metadata['empirical_page_number'] = row.get('empirical_page_number')

                # Always include description
                metadata['description'] = row.get('description', '')

                self.taxon_metadata.append(metadata)
        except Exception as e:
            raise ValueError(f"Failed to parse data from Redis: {e}")

        print(f"✓ Loaded {len(self.taxon_names)} taxa with {self.embeddings.shape[1]}-dimensional embeddings")

        return self.embeddings, self.taxon_names, self.taxon_metadata

    def cluster(
        self,
        method: str = "average",
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Perform agglomerative hierarchical clustering.

        Args:
            method: Linkage method ('single', 'complete', 'average', 'ward')
            metric: Distance metric ('cosine', 'euclidean', 'cityblock')

        Returns:
            Linkage matrix from scipy

        Raises:
            ValueError: If embeddings haven't been loaded
        """
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call load_embeddings() first.")

        print(f"Performing agglomerative clustering...")
        print(f"  Method: {method}")
        print(f"  Metric: {metric}")

        # Perform hierarchical clustering
        self.linkage_matrix = linkage(
            self.embeddings,
            method=method,
            metric=metric
        )

        # Convert linkage matrix to tree structure
        self.root_node = self._build_tree_from_linkage(self.linkage_matrix)

        print(f"✓ Clustering complete")
        print(f"  Tree depth: {self._get_tree_depth(self.root_node)}")
        print(f"  Total nodes: {self._count_nodes(self.root_node)}")

        return self.linkage_matrix

    def _build_tree_from_linkage(self, Z: np.ndarray) -> ClusterNode:
        """
        Convert scipy linkage matrix to ClusterNode tree.

        Args:
            Z: Linkage matrix from scipy

        Returns:
            Root ClusterNode
        """
        n = len(Z) + 1  # Number of original observations
        scipy_tree = to_tree(Z)

        def convert_node(scipy_node, node_id: int) -> ClusterNode:
            """Recursively convert scipy tree to ClusterNode."""
            if scipy_node.is_leaf():
                # Leaf node - represents a taxon
                leaf_idx = scipy_node.id
                metadata = self.taxon_metadata[leaf_idx] if self.taxon_metadata else {}
                return ClusterNode(
                    node_id=node_id,
                    is_leaf=True,
                    taxon_name=self.taxon_names[leaf_idx],
                    distance=0.0,
                    count=1,
                    metadata=metadata
                )
            else:
                # Internal node - represents a pseudoclade
                left = convert_node(scipy_node.left, scipy_node.left.id)
                right = convert_node(scipy_node.right, scipy_node.right.id)

                return ClusterNode(
                    node_id=node_id,
                    is_leaf=False,
                    left_child=left,
                    right_child=right,
                    distance=scipy_node.dist,
                    count=scipy_node.count
                )

        return convert_node(scipy_tree, scipy_tree.id)

    def _get_tree_depth(self, node: Optional[ClusterNode]) -> int:
        """Calculate tree depth."""
        if node is None or node.is_leaf:
            return 1
        return 1 + max(
            self._get_tree_depth(node.left_child),
            self._get_tree_depth(node.right_child)
        )

    def _count_nodes(self, node: Optional[ClusterNode]) -> int:
        """Count total nodes in tree."""
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return 1 + self._count_nodes(node.left_child) + self._count_nodes(node.right_child)

    def store_in_neo4j(
        self,
        root_name: str = "Fungi",
        clear_existing: bool = True
    ):
        """
        Store the clustering tree in Neo4j.

        Creates:
        - Taxon nodes (leaf nodes) with properties: name, node_id
        - Pseudoclade nodes (internal nodes) with properties: name, node_id, count
        - PARENT_OF relationships with property: distance (cosine similarity)

        Args:
            root_name: Name for the root pseudoclade
            clear_existing: Whether to clear existing Taxon and Pseudoclade nodes
        """
        if self.root_node is None:
            raise ValueError("No clustering tree available. Call cluster() first.")

        print(f"Storing tree in Neo4j...")
        print(f"  Root name: {root_name}")

        with self.neo4j_driver.session() as session:
            # Optionally clear existing data
            if clear_existing:
                print("  Clearing existing Taxon and Pseudoclade nodes...")
                session.run("""
                    MATCH (n)
                    WHERE n:Taxon OR n:Pseudoclade
                    DETACH DELETE n
                """)

            # Create indexes for performance
            session.run("CREATE INDEX taxon_node_id IF NOT EXISTS FOR (t:Taxon) ON (t.node_id)")
            session.run("CREATE INDEX pseudoclade_node_id IF NOT EXISTS FOR (p:Pseudoclade) ON (p.node_id)")

            # Store tree recursively
            pseudoclade_counter = [0]  # Use list for mutability in nested function

            def store_node(node: ClusterNode, parent_id: Optional[int] = None, is_root: bool = False):
                """Recursively store nodes in Neo4j."""
                if node.is_leaf:
                    # Create Taxon node with metadata
                    taxon_props = {
                        'name': node.taxon_name,
                        'node_id': node.node_id
                    }

                    # Add metadata fields if available
                    if node.metadata:
                        for key, value in node.metadata.items():
                            # Convert values to Neo4j-compatible types
                            if value is not None and not isinstance(value, (bool, int, float, str)):
                                taxon_props[key] = str(value)
                            else:
                                taxon_props[key] = value

                    session.run("""
                        CREATE (t:Taxon $props)
                    """, props=taxon_props)

                    # Create relationship to parent if exists
                    if parent_id is not None:
                        session.run("""
                            MATCH (parent:Pseudoclade {node_id: $parent_id})
                            MATCH (child:Taxon {node_id: $child_id})
                            CREATE (parent)-[:PARENT_OF {distance: $distance}]->(child)
                        """, parent_id=parent_id, child_id=node.node_id, distance=node.distance)
                else:
                    # Create Pseudoclade node
                    if is_root:
                        pseudoclade_name = root_name
                    else:
                        pseudoclade_counter[0] += 1
                        pseudoclade_name = f"Pseudoclade_{pseudoclade_counter[0]}"

                    session.run("""
                        CREATE (p:Pseudoclade {
                            name: $name,
                            node_id: $node_id,
                            count: $count
                        })
                    """, name=pseudoclade_name, node_id=node.node_id, count=node.count)

                    # Create relationship to parent if exists
                    if parent_id is not None:
                        session.run("""
                            MATCH (parent:Pseudoclade {node_id: $parent_id})
                            MATCH (child:Pseudoclade {node_id: $child_id})
                            CREATE (parent)-[:PARENT_OF {distance: $distance}]->(child)
                        """, parent_id=parent_id, child_id=node.node_id, distance=node.distance)

                    # Recursively store children
                    if node.left_child:
                        store_node(node.left_child, node.node_id, False)
                    if node.right_child:
                        store_node(node.right_child, node.node_id, False)

            # Start from root
            store_node(self.root_node, None, True)

        print(f"✓ Tree stored in Neo4j")

        # Print summary statistics
        self._print_neo4j_stats()

    def _print_neo4j_stats(self):
        """Print statistics about stored data."""
        with self.neo4j_driver.session() as session:
            # Count taxa
            result = session.run("MATCH (t:Taxon) RETURN count(t) as count")
            taxon_count = result.single()['count']

            # Count pseudoclades
            result = session.run("MATCH (p:Pseudoclade) RETURN count(p) as count")
            pseudoclade_count = result.single()['count']

            # Count relationships
            result = session.run("MATCH ()-[r:PARENT_OF]->() RETURN count(r) as count")
            relationship_count = result.single()['count']

            print(f"  Taxon nodes: {taxon_count}")
            print(f"  Pseudoclade nodes: {pseudoclade_count}")
            print(f"  PARENT_OF relationships: {relationship_count}")

    def get_subtree(self, clade_name: str) -> List[str]:
        """
        Get all taxa descendant from a given clade.

        Args:
            clade_name: Name of the clade (Pseudoclade or Taxon)

        Returns:
            List of taxon names in the subtree
        """
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (root {name: $clade_name})-[:PARENT_OF*]->(t:Taxon)
                RETURN t.name as taxon_name
                ORDER BY taxon_name
            """, clade_name=clade_name)

            return [record['taxon_name'] for record in result]

    def get_tree_path(self, taxon_name: str) -> List[Tuple[str, float]]:
        """
        Get the path from root to a specific taxon.

        Args:
            taxon_name: Name of the taxon

        Returns:
            List of (node_name, distance) tuples from root to taxon
        """
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH path = (root:Pseudoclade)-[:PARENT_OF*]->(t:Taxon {name: $taxon_name})
                WHERE NOT (root)<-[:PARENT_OF]-()
                UNWIND nodes(path) as node
                RETURN node.name as name,
                       CASE WHEN node:Taxon THEN 0.0
                            ELSE relationships(path)[0].distance
                       END as distance
            """, taxon_name=taxon_name)

            return [(record['name'], record['distance']) for record in result]

    def close(self):
        """Close connections."""
        self.neo4j_driver.close()
        print("Connections closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def example_usage():
    """
    Example usage of TaxonClusterer.
    """
    # Initialize clusterer
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

        # Store in Neo4j
        clusterer.store_in_neo4j(root_name="Fungi", clear_existing=True)

        print("\n✓ Clustering and storage complete!")

        # Query examples
        print("\n--- Query Examples ---")

        # Get subtree
        fungi_taxa = clusterer.get_subtree("Fungi")
        print(f"\nAll taxa under Fungi: {len(fungi_taxa)} taxa")
        if fungi_taxa:
            print(f"  First 5: {fungi_taxa[:5]}")

        # Get path to specific taxon
        if clusterer.taxon_names:
            example_taxon = clusterer.taxon_names[0]
            path = clusterer.get_tree_path(example_taxon)
            print(f"\nPath to {example_taxon}:")
            for node_name, distance in path:
                print(f"  {node_name} (distance: {distance:.4f})")

    finally:
        clusterer.close()


if __name__ == "__main__":
    example_usage()
