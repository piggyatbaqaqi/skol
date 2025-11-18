"""
Pipeline to extract Taxon objects from CouchDB annotated files and save back to CouchDB.

This module provides a UDF-based PySpark pipeline that:
1. Reads annotated files from an ingest CouchDB database
2. Extracts Taxon objects using the SKOL pipeline
3. Saves Taxa as JSON documents to a taxon CouchDB database
4. Ensures idempotent operations using composite keys: (doc_id, url, line_number)
"""

import json
import hashlib
from typing import Iterator, Optional, Dict, Any, List
from pyspark.sql import SparkSession, DataFrame, Row
from pyspark.sql.types import StructType, StructField, StringType, BooleanType
import couchdb

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


def taxon_to_json_doc(
    taxon: Taxon,
    first_nomenclature_para
) -> Optional[Dict[str, Any]]:
    """
    Convert a Taxon object to a JSON document for CouchDB storage.

    The document includes:
    - All paragraph dictionaries from the taxon
    - Source metadata extracted from first nomenclature paragraph
    - Composite key for idempotency (doc_id, url, line_number)

    Args:
        taxon: Taxon object to convert
        first_nomenclature_para: First nomenclature Paragraph object (for metadata)

    Returns:
        Dictionary ready for JSON serialization and CouchDB storage
    """
    # Collect all paragraph dictionaries
    paragraphs = list(taxon.dictionaries())

    if not paragraphs:
        return None

    # Extract metadata from first nomenclature paragraph's first line
    first_line = first_nomenclature_para.first_line
    if not first_line:
        return None

    source_doc_id = first_line.doc_id if first_line.doc_id else "unknown"
    source_url = first_line.url
    source_db_name = first_line.db_name if first_line.db_name else "unknown"
    line_number = first_line.line_number

    # Build the document
    doc = {
        "type": "taxon",
        "serial_number": paragraphs[0].get('serial_number', '0'),
        "source": {
            "doc_id": source_doc_id,
            "url": source_url,
            "db_name": source_db_name,
            "line_number": line_number
        },
        "paragraphs": paragraphs,
        # Denormalized fields for easy querying
        "nomenclature_count": sum(1 for p in paragraphs if p.get('label') == 'Nomenclature'),
        "description_count": sum(1 for p in paragraphs if p.get('label') == 'Description'),
    }

    return doc


def extract_taxa_from_partition(
    partition: Iterator[Row],
    ingest_db_name: str
) -> Iterator[Dict[str, Any]]:
    """
    Extract Taxa from a partition of CouchDB rows.

    This function processes annotated files from CouchDB and yields
    dictionaries containing taxon data ready for saving.

    Args:
        partition: Iterator of Rows with columns:
            - doc_id: CouchDB document ID
            - attachment_name: Attachment filename
            - value: Text content
            - url: Optional URL
        ingest_db_name: Database name for metadata tracking

    Yields:
        Dictionaries with:
            - source_doc_id: Original document ID
            - source_url: Original URL
            - source_db_name: Original database name
            - taxon_json: JSON string of taxon document
            - first_line_number: Line number for idempotency key
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

    # Convert each taxon to JSON
    for taxon in taxa:
        # Get first nomenclature paragraph for metadata
        if not taxon._nomenclatures:
            continue

        first_nomenclature = taxon._nomenclatures[0]

        taxon_doc = taxon_to_json_doc(taxon, first_nomenclature)

        if taxon_doc:
            # Extract keys from document
            source_doc_id = taxon_doc['source']['doc_id']
            source_url = taxon_doc['source']['url']
            source_db_name = taxon_doc['source']['db_name']
            first_line_num = taxon_doc['source']['line_number']

            yield {
                'source_doc_id': source_doc_id,
                'source_url': source_url if source_url else '',
                'source_db_name': source_db_name,
                'taxon_json': json.dumps(taxon_doc),
                'first_line_number': first_line_num
            }


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
    (source_doc_id, source_url, first_line_number) to ensure
    idempotent writes.

    Args:
        partition: Iterator of Rows with columns:
            - source_doc_id: Original document ID
            - source_url: Original URL
            - source_db_name: Original database name
            - taxon_json: JSON string of taxon document
            - first_line_number: Line number for key
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
            db = server.create(taxon_db_name)

        # Process each taxon in the partition
        for row in partition:
            success = False
            error_msg = ""

            try:
                # Generate deterministic document ID
                doc_id = generate_taxon_doc_id(
                    row.source_doc_id,
                    row.source_url if row.source_url else None,
                    row.first_line_number
                )

                # Parse JSON
                taxon_doc = json.loads(row.taxon_json)

                # Check if document already exists (idempotent)
                if doc_id in db:
                    # Document exists - update it
                    existing_doc = db[doc_id]
                    taxon_doc['_id'] = doc_id
                    taxon_doc['_rev'] = existing_doc['_rev']
                    db.save(taxon_doc)
                else:
                    # New document - create it
                    taxon_doc['_id'] = doc_id
                    db.save(taxon_doc)

                success = True

            except Exception as e:
                error_msg = str(e)
                print(f"Error saving taxon {doc_id}: {e}")

            yield Row(
                doc_id=generate_taxon_doc_id(
                    row.source_doc_id,
                    row.source_url if row.source_url else None,
                    row.first_line_number
                ),
                success=success,
                error_message=error_msg
            )

    except Exception as e:
        print(f"Error connecting to CouchDB: {e}")
        # Yield failures for all rows
        for row in partition:
            yield Row(
                doc_id=generate_taxon_doc_id(
                    row.source_doc_id,
                    row.source_url if row.source_url else None,
                    row.first_line_number
                ),
                success=False,
                error_message=str(e)
            )


def extract_and_save_taxa_pipeline(
    spark: SparkSession,
    ingest_couchdb_url: str,
    ingest_db_name: str,
    taxon_couchdb_url: str,
    taxon_db_name: str,
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
        taxon_couchdb_url: URL of taxon CouchDB server (can be same as ingest)
        taxon_db_name: Name of taxon database
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
    from skol_classifier.couchdb_io import CouchDBConnection

    # Load annotated files from ingest database
    ingest_conn = CouchDBConnection(
        ingest_couchdb_url,
        ingest_db_name,
        ingest_username,
        ingest_password
    )

    df = ingest_conn.load_distributed(spark, pattern)

    # Extract taxa from each partition
    extract_schema = StructType([
        StructField("source_doc_id", StringType(), False),
        StructField("source_url", StringType(), False),
        StructField("source_db_name", StringType(), False),
        StructField("taxon_json", StringType(), False),
        StructField("first_line_number", StringType(), False),
    ])

    def extract_partition(partition):
        return extract_taxa_from_partition(partition, ingest_db_name)

    taxa_df = df.rdd.mapPartitions(extract_partition).toDF(extract_schema)

    # Save taxa to taxon database
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
