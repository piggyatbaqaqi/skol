#!/usr/bin/env python3
"""
Pipeline to extract Taxon objects from CouchDB annotated files and save back to CouchDB.

This module provides a UDF-based PySpark pipeline that:
1. Reads annotated files from an ingest CouchDB database
2. Extracts Taxon objects using the SKOL pipeline
3. Saves Taxa as JSON documents to a taxon CouchDB database
4. Ensures idempotent operations using composite keys: (doc_id, url, line_number)
"""

import hashlib
from typing import Iterator, Optional, Dict, Any

import couchdb
from pyspark.sql import SparkSession, DataFrame, Row
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, MapType, IntegerType

from skol_classifier.couchdb_io import CouchDBConnection

from couchdb_file import read_couchdb_partition
from finder import parse_annotated, remove_interstitials
from taxon import group_paragraphs, Taxon

def generate_taxon_doc_id(doc_id: str, url: Optional[str], line_number: int) -> str:
    """
    Generate a unique, deterministic document ID for a taxon.

    This ensures idempotent writes - the same taxon from the same source
    will always have the same document ID.

    Args:
        doc_id: Source document ID from ingest database
        url: URL from the source line
        line_number: Line number from the source

    Returns:
        Unique document ID as a hash string
    """
    # Create composite key
    key_parts = [
        doc_id,
        url if url else "no_url",
        str(line_number)
    ]
    composite_key = ":".join(key_parts)

    # Generate deterministic hash
    hash_obj = hashlib.sha256(composite_key.encode('utf-8'))
    doc_hash = hash_obj.hexdigest()

    return f"taxon_{doc_hash}"


def extract_taxa_from_partition(
    partition: Iterator[Row],
    ingest_db_name: str
) -> Iterator[Taxon]:
    """
    Extract Taxa from a partition of CouchDB rows.

    This function processes annotated files from CouchDB and yields
    Taxon objects for further processing.

    Args:
        partition: Iterator of Rows with columns:
            - doc_id: CouchDB document ID
            - attachment_name: Attachment filename
            - value: Text content
            - human_url: Optional URL
        ingest_db_name: Database name for metadata tracking

    Yields:
        Taxon objects with nomenclature and description paragraphs
    """
    # Read lines from partition
    lines = read_couchdb_partition(partition, ingest_db_name)

    # Parse annotated content
    paragraphs = parse_annotated(lines)

    # Remove interstitial paragraphs
    filtered = remove_interstitials(paragraphs)

    # Convert to list to preserve paragraph objects for metadata extraction
    filtered_list = list(filtered)

    # Group into taxa (returns Taxon objects with references to paragraphs)
    taxa = group_paragraphs(iter(filtered_list))

    # Yield Taxon objects directly
    for taxon in taxa:
        # Only yield taxa that have nomenclature
        if taxon.has_nomenclature():
            yield taxon


def convert_taxa_to_rows(partition: Iterator[Taxon]) -> Iterator[Row]:
    """
    Convert Taxon objects to PySpark Rows suitable for DataFrame creation.

    Args:
        partition: Iterator of Taxon objects

    Yields:
        PySpark Row objects with fields:
            - taxon: String of concatenated nomenclature paragraphs
            - description: String of concatenated description paragraphs
            - source: Dict with keys doc_id, url, db_name
            - line_number: Line number of first nomenclature paragraph
            - paragraph_number: Paragraph number of first nomenclature paragraph
            - page_number: Page number of first nomenclature paragraph
            - empirical_page_number: Empirical page number of first nomenclature paragraph
    """
    for taxon in partition:
        taxon_dict = taxon.as_row()

        if '_id' not in taxon_dict:
            taxon_dict['_id'] = generate_taxon_doc_id(
                taxon_dict['source']['doc_id'],
                taxon_dict['source'].get('human_url'),
                taxon_dict['line_number'] or 0
            )
        if 'json_annotated' not in taxon_dict:
            taxon_dict['json_annotated'] = None
        # Convert dict to Row
        yield Row(**taxon_dict)


class TaxonExtractor:
    """
    Extract and save Taxa from CouchDB annotated files.

    This class encapsulates the complete pipeline for:
    1. Loading annotated documents from a CouchDB ingest database
    2. Extracting Taxon objects using the SKOL pipeline
    3. Saving Taxa to a CouchDB taxon database with idempotent keys

    Args:
        spark: SparkSession for distributed processing
        ingest_couchdb_url: URL of ingest CouchDB server
        ingest_db_name: Name of ingest database
        taxon_db_name: Name of taxon database
        taxon_couchdb_url: URL of taxon CouchDB server (defaults to ingest_couchdb_url)
        ingest_username: Optional username for ingest database
        ingest_password: Optional password for ingest database
        taxon_username: Optional username for taxon database (defaults to ingest_username)
        taxon_password: Optional password for taxon database (defaults to ingest_password)

    Example:
        >>> spark = SparkSession.builder.appName("TaxonExtractor").getOrCreate()
        >>>
        >>> extractor = TaxonExtractor(
        ...     spark=spark,
        ...     ingest_couchdb_url="http://localhost:5984",
        ...     ingest_db_name="mycobank_annotations",
        ...     taxon_db_name="mycobank_taxa",
        ...     ingest_username="admin",
        ...     ingest_password="secret"
        ... )
        >>>
        >>> # Step-by-step debugging
        >>> annotated_df = extractor.load_annotated_documents()
        >>> print(f"Loaded {annotated_df.count()} documents")
        >>>
        >>> taxa_df = extractor.extract_taxa(annotated_df)
        >>> print(f"Extracted {taxa_df.count()} taxa")
        >>> taxa_df.show(5)
        >>>
        >>> results = extractor.save_taxa(taxa_df)
        >>> print(f"Saved: {results.filter('success = true').count()}")
        >>>
        >>> # Or run the complete pipeline
        >>> results = extractor.run_pipeline()
        >>> results.filter("success = false").show()
    """

    def __init__(
        self,
        spark: SparkSession,
        ingest_couchdb_url: str,
        ingest_db_name: str,
        taxon_db_name: str,
        taxon_couchdb_url: Optional[str] = None,
        ingest_username: Optional[str] = None,
        ingest_password: Optional[str] = None,
        taxon_username: Optional[str] = None,
        taxon_password: Optional[str] = None
    ):
        self.spark = spark
        self.ingest_couchdb_url = ingest_couchdb_url
        self.ingest_db_name = ingest_db_name
        self.ingest_username = ingest_username
        self.ingest_password = ingest_password

        self.taxon_couchdb_url = taxon_couchdb_url or ingest_couchdb_url
        self.taxon_db_name = taxon_db_name
        self.taxon_username = taxon_username or ingest_username
        self.taxon_password = taxon_password or ingest_password

        # Schema for extracted taxa
        self._extract_schema = StructType([
            StructField("taxon", StringType(), False),
            StructField("description", StringType(), False),
            StructField("source", MapType(StringType(), StringType(), valueContainsNull=True), False),
            StructField("line_number", IntegerType(), True),
            StructField("paragraph_number", IntegerType(), True),
            StructField("page_number", IntegerType(), True),
            StructField("empirical_page_number", StringType(), True),
            StructField("_id", StringType(), True),
            StructField("json_annotated", StringType(), True)
        ])

        # Schema for save results
        self._save_schema = StructType([
            StructField("doc_id", StringType(), False),
            StructField("success", BooleanType(), False),
            StructField("error_message", StringType(), False),
        ])

    def load_annotated_documents(self, pattern: str = "*.txt.ann") -> DataFrame:
        """
        Load annotated documents from CouchDB ingest database.

        Args:
            pattern: Pattern for attachment names (default: "*.txt.ann")

        Returns:
            DataFrame with columns: doc_id, attachment_name, value
        """
        ingest_conn = CouchDBConnection(
            self.ingest_couchdb_url,
            self.ingest_db_name,
            self.ingest_username,
            self.ingest_password
        )
        return ingest_conn.load_distributed(self.spark, pattern)

    def extract_taxa(self, annotated_df: DataFrame) -> DataFrame:
        """
        Extract taxa from annotated documents DataFrame.

        Args:
            annotated_df: DataFrame with columns: doc_id, attachment_name, value
                         (or prediction DataFrame with doc_id, pos, value, prediction, ...)

        Returns:
            DataFrame with taxa information (taxon, description, source, line numbers, etc.)
        """
        # Select only the columns we need for extraction
        # This prevents column count mismatch errors when receiving prediction outputs
        required_cols = ["doc_id", "value"]

        # Add attachment_name if it exists (from CouchDB), otherwise we'll handle it
        if "attachment_name" in annotated_df.columns:
            required_cols.append("attachment_name")

        # Select only required columns to avoid schema mismatch
        annotated_df_filtered = annotated_df.select(*required_cols)

        # Debug: Print schema to verify
        print(f"[TaxonExtractor] Input DataFrame columns: {annotated_df.columns}")
        print(f"[TaxonExtractor] Filtered DataFrame columns: {annotated_df_filtered.columns}")
        print(f"[TaxonExtractor] Filtered DataFrame schema:")
        annotated_df_filtered.printSchema()

        # Extract to local variable to avoid serializing self
        db_name = self.ingest_db_name

        def extract_partition(partition):  # type: ignore[reportUnknownParameterType]
            # Extract Taxon objects
            taxa = extract_taxa_from_partition(iter(partition), db_name)  # type: ignore[reportUnknownArgumentType]
            # Convert to Rows for DataFrame
            return convert_taxa_to_rows(taxa)

        taxa_rdd = annotated_df_filtered.rdd.mapPartitions(extract_partition)  # type: ignore[reportUnknownArgumentType]
        taxa_df = self.spark.createDataFrame(taxa_rdd, self._extract_schema)

        return taxa_df

    def load_taxa(self, pattern: str = "taxon_*") -> DataFrame:
        """
        Load taxa from CouchDB taxon database.

        This method performs the inverse operation of save_taxa(), loading
        taxa documents from CouchDB and converting them back to a DataFrame.

        Args:
            pattern: Pattern for document IDs to load (default: "taxon_*")
                    Use "*" to load all documents
                    Use "taxon_abc*" to load specific subset

        Returns:
            DataFrame with taxa information matching the extract_taxa() schema:
            - taxon: String of concatenated nomenclature paragraphs
            - description: String of concatenated description paragraphs
            - source: Dict with keys doc_id, url, db_name
            - line_number: Line number of first nomenclature paragraph
            - paragraph_number: Paragraph number of first nomenclature paragraph
            - page_number: Page number of first nomenclature paragraph
            - empirical_page_number: Empirical page number of first nomenclature paragraph

        Example:
            >>> # Load all taxa
            >>> taxa_df = extractor.load_taxa()
            >>> print(f"Loaded {taxa_df.count()} taxa")
            >>>
            >>> # Load specific subset
            >>> subset_df = extractor.load_taxa(pattern="taxon_abc*")
        """
        # Extract to local variables to avoid serializing self
        couchdb_url = self.taxon_couchdb_url
        db_name = self.taxon_db_name
        username = self.taxon_username
        password = self.taxon_password
        extract_schema = self._extract_schema

        def load_partition(partition: Iterator[Row]) -> Iterator[Row]:
            """Load taxa from CouchDB for an entire partition."""
            # Connect to CouchDB once per partition
            try:
                server = couchdb.Server(couchdb_url)
                if username and password:
                    server.resource.credentials = (username, password)

                # Check if database exists
                if db_name not in server:
                    print(f"Database {db_name} does not exist")
                    return

                db = server[db_name]

                # Process each row (which contains doc_id)
                for row in partition:
                    try:
                        doc_id = row.doc_id if hasattr(row, 'doc_id') else str(row[0])

                        # Load document from CouchDB
                        if doc_id in db:
                            doc = db[doc_id]

                            # Convert CouchDB document to Row
                            taxon_data = {
                                'taxon': doc.get('taxon', ''),
                                'description': doc.get('description', ''),
                                'source': doc.get('source', {}),
                                'line_number': doc.get('line_number'),
                                'paragraph_number': doc.get('paragraph_number'),
                                'page_number': doc.get('page_number'),
                                'empirical_page_number': doc.get('empirical_page_number')
                            }

                            yield Row(**taxon_data)
                        else:
                            print(f"Document {doc_id} not found in database")

                    except Exception as e:  # pyright: ignore[reportUnknownExceptionType]
                        print(f"Error loading taxon {doc_id}: {e}")

            except Exception as e:  # pyright: ignore[reportUnknownExceptionType]
                print(f"Error connecting to CouchDB: {e}")

        # First, get list of document IDs matching pattern from CouchDB
        # We need to create a DataFrame with doc_ids to process
        doc_ids = self._get_matching_doc_ids(pattern)

        if not doc_ids:
            # Return empty DataFrame with correct schema
            return self.spark.createDataFrame([], extract_schema)

        # Create DataFrame with doc_ids for parallel processing
        doc_ids_rdd = self.spark.sparkContext.parallelize(doc_ids)
        doc_ids_df = doc_ids_rdd.map(lambda x: Row(doc_id=x)).toDF()

        # Load taxa using mapPartitions
        taxa_rdd = doc_ids_df.rdd.mapPartitions(load_partition)
        taxa_df = self.spark.createDataFrame(taxa_rdd, extract_schema)

        return taxa_df

    def _get_matching_doc_ids(self, pattern: str) -> list:
        """
        Get list of document IDs matching the pattern from CouchDB.

        Args:
            pattern: Pattern for document IDs (e.g., "taxon_*", "*")

        Returns:
            List of matching document IDs
        """
        try:
            server = couchdb.Server(self.taxon_couchdb_url)
            if self.taxon_username and self.taxon_password:
                server.resource.credentials = (self.taxon_username, self.taxon_password)

            # Check if database exists
            if self.taxon_db_name not in server:
                print(f"Database {self.taxon_db_name} does not exist")
                return []

            db = server[self.taxon_db_name]

            # Get all document IDs
            all_doc_ids = [doc_id for doc_id in db if not doc_id.startswith('_design/')]

            # Filter by pattern
            if pattern == "*":
                # Return all non-design documents
                return all_doc_ids
            else:
                # Simple pattern matching (prefix matching for now)
                # Convert glob pattern to prefix
                if pattern.endswith('*'):
                    prefix = pattern[:-1]
                    return [doc_id for doc_id in all_doc_ids if doc_id.startswith(prefix)]
                else:
                    # Exact match
                    return [doc_id for doc_id in all_doc_ids if doc_id == pattern]

        except Exception as e:  # pyright: ignore[reportUnknownExceptionType]
            print(f"Error getting document IDs from CouchDB: {e}")
            return []

    def save_taxa(self, taxa_df: DataFrame) -> DataFrame:
        """
        Save taxa DataFrame to CouchDB taxon database.

        Args:
            taxa_df: DataFrame with taxa information

        Returns:
            DataFrame with save results (doc_id, success, error_message)
        """
        # Extract to local variables to avoid serializing self
        couchdb_url = self.taxon_couchdb_url
        db_name = self.taxon_db_name
        username = self.taxon_username
        password = self.taxon_password

        def save_partition(partition: Iterator[Row]) -> Iterator[Row]:
            """Save taxa to CouchDB for an entire partition (idempotent)."""
            # Connect to CouchDB once per partition
            try:
                server = couchdb.Server(couchdb_url)
                if username and password:
                    server.resource.credentials = (username, password)

                # Get or create database
                if db_name in server:
                    db = server[db_name]
                else:
                    db = server.create(db_name)  # pyright: ignore[reportUnknownMemberType]

                # Process each taxon in the partition
                for row in partition:
                    success = False
                    error_msg = ""
                    doc_id = "unknown"

                    try:
                        # Extract source metadata from row
                        source_dict = row.source if hasattr(row, 'source') else {}  # type: ignore[reportUnknownMemberType]
                        source: Dict[str, Any] = dict(source_dict) if isinstance(source_dict, dict) else {}  # type: ignore[reportUnknownArgumentType]
                        source_doc_id: str = str(source.get('doc_id', 'unknown'))
                        source_url: Optional[str] = source.get('url')  # type: ignore[reportUnknownArgumentType]
                        line_number: Any = row.line_number if hasattr(row, 'line_number') else 0  # type: ignore[reportUnknownMemberType]

                        # Generate deterministic document ID
                        doc_id = generate_taxon_doc_id(
                            source_doc_id,
                            source_url if isinstance(source_url, str) else None,
                            int(line_number) if line_number else 0
                        )

                        # Convert row to dict for CouchDB storage
                        taxon_doc = row.asDict()

                        # Check if document already exists (idempotent)
                        if doc_id in db:
                            # Document exists - update it
                            existing_doc = db[doc_id]
                            taxon_doc['_id'] = doc_id
                            taxon_doc['_rev'] = existing_doc['_rev']
                        else:
                            # New document - create it
                            taxon_doc['_id'] = doc_id

                        db.save(taxon_doc)  # pyright: ignore[reportUnknownMemberType]
                        success = True

                    except Exception as e:  # pyright: ignore[reportUnknownExceptionType]
                        error_msg = str(e)
                        print(f"Error saving taxon {doc_id}: {e}")

                    yield Row(
                        doc_id=doc_id,
                        success=success,
                        error_message=error_msg
                    )

            except Exception as e:  # pyright: ignore[reportUnknownExceptionType]
                print(f"Error connecting to CouchDB: {e}")
                # Yield failures for all rows
                for row in partition:
                    yield Row(
                        doc_id="unknown_connection_error",
                        success=False,
                        error_message=str(e)
                    )

        results_df = taxa_df.rdd.mapPartitions(save_partition).toDF(self._save_schema)
        return results_df

    def run_pipeline(self, pattern: str = "*.txt.ann") -> DataFrame:
        """
        Run the complete pipeline: load, extract, and save taxa.

        This method:
        1. Loads annotated files from ingest CouchDB database
        2. Extracts Taxon objects in parallel using mapPartitions
        3. Saves Taxa to taxon CouchDB database with idempotent keys
        4. Returns a DataFrame with success/failure results

        Args:
            pattern: Pattern for attachment names (default: "*.txt.ann")

        Returns:
            DataFrame with columns: doc_id, success, error_message

        Example:
            >>> results = extractor.run_pipeline()
            >>> results.filter("success = true").count()
            >>> results.filter("success = false").show()
        """
        # Step 1: Load annotated documents from CouchDB
        annotated_df = self.load_annotated_documents(pattern)

        # Step 2: Extract taxa from annotated documents
        taxa_df = self.extract_taxa(annotated_df)

        # Step 3: Save taxa to CouchDB
        results_df = self.save_taxa(taxa_df)

        return results_df


# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract Taxa from CouchDB annotated files and save to CouchDB"
    )
    parser.add_argument(
        "--ingest-url",
        default="http://localhost:5984",
        help="CouchDB server URL for ingest database"
    )
    parser.add_argument(
        "--ingest-database",
        required=True,
        help="Name of ingest database (e.g., mycobank_annotations)"
    )
    parser.add_argument(
        "--ingest-username",
        help="Username for ingest database"
    )
    parser.add_argument(
        "--ingest-password",
        help="Password for ingest database"
    )
    parser.add_argument(
        "--taxon-url",
        help="CouchDB server URL for taxon database (defaults to ingest-url)"
    )
    parser.add_argument(
        "--taxon-database",
        required=True,
        help="Name of taxon database (e.g., mycobank_taxa)"
    )
    parser.add_argument(
        "--taxon-username",
        help="Username for taxon database (defaults to ingest-username)"
    )
    parser.add_argument(
        "--taxon-password",
        help="Password for taxon database (defaults to ingest-password)"
    )
    parser.add_argument(
        "--pattern",
        default="*.txt.ann",
        help="Pattern for attachment names (default: *.txt.ann)"
    )

    args = parser.parse_args()

    # Default taxon credentials to ingest credentials
    taxon_url = args.taxon_url or args.ingest_url
    taxon_username = args.taxon_username or args.ingest_username
    taxon_password = args.taxon_password or args.ingest_password

    # Create Spark session
    spark = SparkSession.builder \
        .appName("SKOL Taxon Extractor") \
        .getOrCreate()

    print(f"Extracting taxa from {args.ingest_database} to {args.taxon_database}...")

    # Create extractor instance
    extractor = TaxonExtractor(
        spark=spark,
        ingest_couchdb_url=args.ingest_url,
        ingest_db_name=args.ingest_database,
        taxon_db_name=args.taxon_database,
        taxon_couchdb_url=taxon_url,
        ingest_username=args.ingest_username,
        ingest_password=args.ingest_password,
        taxon_username=taxon_username,
        taxon_password=taxon_password
    )

    # Run pipeline
    results = extractor.run_pipeline(pattern=args.pattern)

    # Show results
    total = results.count()
    successes = results.filter("success = true").count()
    failures = results.filter("success = false").count()

    print(f"\nResults:")
    print(f"  Total taxa: {total}")
    print(f"  Successful saves: {successes}")
    print(f"  Failed saves: {failures}")

    if failures > 0:
        print("\nFailed documents:")
        results.filter("success = false").show(truncate=False)

    spark.stop()
