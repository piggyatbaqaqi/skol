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
from pyspark.sql.types import StructType, StructField, StringType, BooleanType

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
            - url: Optional URL
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
        # Convert dict to Row
        yield Row(**taxon_dict)


def save_taxa_to_couchdb_partition(
    partition: Iterator[Row],
    couchdb_url: str,
    taxon_db_name: str,
    username: Optional[str] = None,
    password: Optional[str] = None
) -> Iterator[Row]:
    """
    Save taxa to CouchDB for an entire partition (idempotent).

    This function creates deterministic document IDs based on
    (source.doc_id, source.url, line_number) to ensure
    idempotent writes.

    Args:
        partition: Iterator of Rows with columns from Taxon.as_row():
            - taxon: Nomenclature text
            - description: Description text
            - source: Dict with doc_id, url, db_name
            - line_number: Line number
            - paragraph_number, page_number, empirical_page_number
        couchdb_url: CouchDB server URL
        taxon_db_name: Target database name
        username: Optional username
        password: Optional password

    Yields:
        Rows with save results (doc_id, success, error_message)
    """
    # Connect to CouchDB once per partition
    try:
        server = couchdb.Server(couchdb_url)
        if username and password:
            server.resource.credentials = (username, password)

        # Get or create database
        if taxon_db_name in server:
            db = server[taxon_db_name]
        else:
            db = server.create(taxon_db_name)  # pyright: ignore[reportUnknownMemberType]

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


def load_annotated_documents(
    spark: SparkSession,
    ingest_couchdb_url: str,
    ingest_db_name: str,
    ingest_username: Optional[str] = None,
    ingest_password: Optional[str] = None,
    pattern: str = "*.txt.ann"
) -> DataFrame:
    """
    Load annotated documents from CouchDB.

    Returns:
        DataFrame with columns: doc_id, attachment_name, value
    """
    ingest_conn = CouchDBConnection(
        ingest_couchdb_url,
        ingest_db_name,
        ingest_username,
        ingest_password
    )
    return ingest_conn.load_distributed(spark, pattern)


def extract_taxa_dataframe(
    spark: SparkSession,
    annotated_df: DataFrame,
    ingest_db_name: str
) -> DataFrame:
    """
    Extract taxa from annotated documents DataFrame.

    Args:
        spark: SparkSession
        annotated_df: DataFrame with columns: doc_id, attachment_name, value
        ingest_db_name: Database name for metadata

    Returns:
        DataFrame with taxa information (taxon, description, source, line numbers, etc.)
    """
    from pyspark.sql.types import MapType, IntegerType

    extract_schema = StructType([
        StructField("taxon", StringType(), False),
        StructField("description", StringType(), False),
        StructField("source", MapType(StringType(), StringType(), valueContainsNull=True), False),
        StructField("line_number", IntegerType(), True),
        StructField("paragraph_number", IntegerType(), True),
        StructField("page_number", IntegerType(), True),
        StructField("empirical_page_number", StringType(), True),
    ])

    def extract_partition(partition):
        # Extract Taxon objects
        taxa = extract_taxa_from_partition(partition, ingest_db_name)
        # Convert to Rows for DataFrame
        return convert_taxa_to_rows(taxa)

    taxa_rdd = annotated_df.rdd.mapPartitions(extract_partition)
    taxa_df = spark.createDataFrame(taxa_rdd, extract_schema)

    return taxa_df


def save_taxa_dataframe(
    taxa_df: DataFrame,
    taxon_couchdb_url: str,
    taxon_db_name: str,
    taxon_username: Optional[str] = None,
    taxon_password: Optional[str] = None
) -> DataFrame:
    """
    Save taxa DataFrame to CouchDB.

    Args:
        taxa_df: DataFrame with taxa information
        taxon_couchdb_url: URL of taxon CouchDB server
        taxon_db_name: Name of taxon database
        taxon_username: Optional username
        taxon_password: Optional password

    Returns:
        DataFrame with save results (doc_id, success, error_message)
    """
    save_schema = StructType([
        StructField("doc_id", StringType(), False),
        StructField("success", BooleanType(), False),
        StructField("error_message", StringType(), False),
    ])

    def save_partition(partition):
        return save_taxa_to_couchdb_partition(
            partition,
            taxon_couchdb_url,
            taxon_db_name,
            taxon_username,
            taxon_password
        )

    results_df = taxa_df.rdd.mapPartitions(save_partition).toDF(save_schema)
    return results_df


def extract_and_save_taxa_pipeline(
    spark: SparkSession,
    ingest_couchdb_url: str,
    ingest_db_name: str,
    taxon_db_name: str,
    taxon_couchdb_url: Optional[str] = None,
    ingest_username: Optional[str] = None,
    ingest_password: Optional[str] = None,
    taxon_username: Optional[str] = None,
    taxon_password: Optional[str] = None,
    pattern: str = "*.txt.ann"
) -> DataFrame:
    """
    Complete pipeline to extract taxa from ingest database and save to taxon database.

    This pipeline:
    1. Loads annotated files from ingest CouchDB database
    2. Extracts Taxon objects in parallel using mapPartitions
    3. Saves Taxa to taxon CouchDB database with idempotent keys
    4. Returns a DataFrame with success/failure results

    Args:
        spark: SparkSession
        ingest_couchdb_url: URL of ingest CouchDB server
        ingest_db_name: Name of ingest database
        taxon_db_name: Name of taxon database
        taxon_couchdb_url: URL of taxon CouchDB server (can be same as ingest)
        ingest_username: Optional username for ingest database
        ingest_password: Optional password for ingest database
        taxon_username: Optional username for taxon database
        taxon_password: Optional password for taxon database
        pattern: Pattern for attachment names (default: "*.txt.ann")

    Returns:
        DataFrame with columns: doc_id, success, error_message

    Example:
        >>> spark = SparkSession.builder.appName("TaxonExtractor").getOrCreate()
        >>>
        >>> results = extract_and_save_taxa_pipeline(
        ...     spark=spark,
        ...     ingest_couchdb_url="http://localhost:5984",
        ...     ingest_db_name="mycobank_annotations",
        ...     taxon_couchdb_url="http://localhost:5984",
        ...     taxon_db_name="mycobank_taxa",
        ...     ingest_username="admin",
        ...     ingest_password="secret",
        ...     taxon_username="admin",
        ...     taxon_password="secret"
        ... )
        >>>
        >>> # Check results
        >>> results.filter("success = true").count()
        >>> results.filter("success = false").show()
    """
    taxon_couchdb_url = taxon_couchdb_url or ingest_couchdb_url
    taxon_username = taxon_username or ingest_username
    taxon_password = taxon_password or ingest_password

    # Step 1: Load annotated documents from CouchDB
    annotated_df = load_annotated_documents(
        spark,
        ingest_couchdb_url,
        ingest_db_name,
        ingest_username,
        ingest_password,
        pattern
    )

    # Step 2: Extract taxa from annotated documents
    taxa_df = extract_taxa_dataframe(spark, annotated_df, ingest_db_name)

    # Step 3: Save taxa to CouchDB
    results_df = save_taxa_dataframe(
        taxa_df,
        taxon_couchdb_url,
        taxon_db_name,
        taxon_username,
        taxon_password
    )

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

    # Run pipeline
    results = extract_and_save_taxa_pipeline(
        spark=spark,
        ingest_couchdb_url=args.ingest_url,
        ingest_db_name=args.ingest_database,
        taxon_couchdb_url=taxon_url,
        taxon_db_name=args.taxon_database,
        ingest_username=args.ingest_username,
        ingest_password=args.ingest_password,
        taxon_username=taxon_username,
        taxon_password=taxon_password,
        pattern=args.pattern
    )

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
