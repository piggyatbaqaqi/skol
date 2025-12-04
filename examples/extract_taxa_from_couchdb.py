"""
Example: Extract Taxon objects from CouchDB attachments in PySpark.

This example demonstrates how to use the CouchDB-aware file reading functions
to extract taxa from annotated files stored in CouchDB.
"""

import sys
sys.path.insert(0, '..')

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from typing import Iterator, List, Dict, Any

from skol_classifier.couchdb_io import CouchDBConnection
from couchdb_file import read_couchdb_partition
from finder import parse_annotated, remove_interstitials
from taxon import group_paragraphs, Taxon
from paragraph import Paragraph


def process_partition_to_taxa(
    partition: Iterator,
    db_name: str
) -> Iterator[Dict[str, Any]]:
    """
    Process a partition of CouchDB rows and extract Taxon objects.

    This function is designed to be used with mapPartitions in PySpark.
    It converts CouchDB attachment content into structured Taxon data.

    Args:
        partition: Iterator of Rows with doc_id, attachment_name, value
        db_name: Database name for metadata tracking

    Yields:
        Dictionaries representing taxon paragraph data with CouchDB metadata

    Example:
        >>> df.rdd.mapPartitions(
        ...     lambda part: process_partition_to_taxa(part, "mycobank")
        ... )
    """
    # Step 1: Read CouchDB rows into Line objects with metadata
    lines = list(read_couchdb_partition(partition, db_name))

    print("DEBUG: line[0].human_url =", lines[0].human_url if lines else "No lines")

    # Step 2: Parse lines into Paragraph objects
    paragraphs = parse_annotated(lines)

    print(f"DEBUG: paragraphs[0].human_url =", paragraphs[0].human_url if paragraphs else "No paragraphs")

    # Step 3: Remove interstitial paragraphs (optional)
    filtered = remove_interstitials(paragraphs)

    print(f"DEBUG: filtered[0].human_url =", filtered[0].human_url if filtered else "No filtered paragraphs")

    # Step 4: Group paragraphs into Taxon objects
    taxa = group_paragraphs(filtered)

    # Step 5: Convert taxa to dictionaries and yield
    for taxon in taxa:
        for para_dict in taxon.dictionaries():
            # The para_dict contains:
            # - serial_number: Taxon ID
            # - filename: In format "db_name/doc_id/attachment_name"
            # - label: "Nomenclature" or "Description"
            # - paragraph_number: Sequential number
            # - page_number: Page in document
            # - empirical_page_number: Printed page number
            # - body: Full paragraph text

            # Parse the composite filename to extract CouchDB metadata
            parts = para_dict['filename'].split('/', 2)
            if len(parts) == 3:
                para_dict['db_name'] = parts[0]
                para_dict['doc_id'] = parts[1]
                para_dict['attachment_name'] = parts[2]

            yield para_dict


def extract_taxa_spark(
    spark: SparkSession,
    couchdb_url: str,
    database: str,
    db_name: str,
    username: str = None,
    password: str = None,
    pattern: str = "*.txt.ann"
):
    """
    Extract taxa from CouchDB using distributed Spark processing.

    Args:
        spark: SparkSession
        couchdb_url: CouchDB server URL (e.g., "http://localhost:5984")
        database: Database name
        db_name: Logical database name for metadata (ingest_db_name)
        username: Optional CouchDB username
        password: Optional CouchDB password
        pattern: Attachment filename pattern (default: "*.txt.ann")

    Returns:
        DataFrame with taxon data including CouchDB metadata

    Example:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("TaxonExtractor").getOrCreate()
        >>>
        >>> taxa_df = extract_taxa_spark(
        ...     spark=spark,
        ...     couchdb_url="http://localhost:5984",
        ...     database="mycobank_docs",
        ...     db_name="mycobank",
        ...     username="admin",
        ...     password="secret",
        ...     pattern="*.txt.ann"
        ... )
        >>>
        >>> # Show results
        >>> taxa_df.select("serial_number", "doc_id", "label", "body").show()
        >>>
        >>> # Save to CSV
        >>> taxa_df.write.csv("output/taxa", header=True)
    """
    # Create CouchDB connection
    conn = CouchDBConnection(couchdb_url, database, username, password)

    # Load attachments from CouchDB (distributed)
    print(f"Loading attachments matching '{pattern}' from {database}...")
    df = conn.load_distributed(spark, pattern)

    print(f"Found {df.count()} attachments")

    # Process partitions to extract taxa
    print("Processing partitions to extract taxa...")

    def process_partition(partition):
        return process_partition_to_taxa(partition, db_name)

    # Define schema for output
    schema = StructType([
        StructField("serial_number", StringType(), False),
        StructField("filename", StringType(), False),
        StructField("db_name", StringType(), True),
        StructField("doc_id", StringType(), True),
        StructField("attachment_name", StringType(), True),
        StructField("label", StringType(), False),
        StructField("paragraph_number", StringType(), False),
        StructField("page_number", StringType(), False),
        StructField("empirical_page_number", StringType(), True),
        StructField("body", StringType(), False)
    ])

    # Apply processing
    taxa_df = df.rdd.mapPartitions(process_partition).toDF(schema)

    return taxa_df


def extract_taxa_local(
    couchdb_url: str,
    database: str,
    db_name: str,
    username: str = None,
    password: str = None,
    pattern: str = "*.txt.ann"
) -> List[Taxon]:
    """
    Extract taxa from CouchDB using local (non-distributed) processing.

    This is useful for small datasets or testing. For large datasets,
    use extract_taxa_spark() instead.

    Args:
        couchdb_url: CouchDB server URL
        database: Database name
        db_name: Logical database name for metadata
        username: Optional username
        password: Optional password
        pattern: Attachment pattern

    Returns:
        List of Taxon objects

    Example:
        >>> taxa = extract_taxa_local(
        ...     couchdb_url="http://localhost:5984",
        ...     database="mycobank_docs",
        ...     db_name="mycobank",
        ...     pattern="*.txt.ann"
        ... )
        >>>
        >>> print(f"Found {len(taxa)} taxa")
        >>> for taxon in taxa:
        ...     print(taxon.as_row())
    """
    from couchdb_file import read_couchdb_files_from_connection
    from pyspark.sql import SparkSession

    # Create local Spark session
    spark = SparkSession.builder \
        .appName("TaxonExtractorLocal") \
        .master("local[*]") \
        .getOrCreate()

    # Create connection
    conn = CouchDBConnection(couchdb_url, database, username, password)

    # Load and read files
    print(f"Loading files from {database}...")
    lines = read_couchdb_files_from_connection(conn, spark, db_name, pattern)

    # Parse and extract
    print("Parsing paragraphs...")
    paragraphs = parse_annotated(lines)

    print("Filtering interstitials...")
    filtered = remove_interstitials(paragraphs)

    print("Grouping into taxa...")
    taxa = list(group_paragraphs(filtered))

    print(f"Found {len(taxa)} taxa")

    spark.stop()
    return taxa


# Example usage functions

def example_distributed():
    """Example: Extract taxa using distributed Spark processing."""
    from pyspark.sql import SparkSession

    # Create Spark session
    spark = SparkSession.builder \
        .appName("DistributedTaxonExtractor") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    # Extract taxa
    taxa_df = extract_taxa_spark(
        spark=spark,
        couchdb_url="http://localhost:5984",
        database="mycobank_annotations",
        db_name="mycobank",
        username="admin",
        password="password",
        pattern="*.txt.ann"
    )

    # Analyze results
    print(f"\nTotal paragraphs: {taxa_df.count()}")
    print("\nLabel distribution:")
    taxa_df.groupBy("label").count().show()

    print("\nTaxa per document:")
    taxa_df.groupBy("doc_id", "serial_number") \
        .count() \
        .groupBy("doc_id") \
        .count() \
        .withColumnRenamed("count", "taxa_count") \
        .show()

    # Save results
    print("\nSaving to parquet...")
    taxa_df.write.mode("overwrite").parquet("output/taxa.parquet")

    print("\nSaving to CSV...")
    taxa_df.write.mode("overwrite").csv("output/taxa.csv", header=True)

    spark.stop()


def example_local():
    """Example: Extract taxa using local processing."""
    import csv

    # Extract taxa
    taxa = extract_taxa_local(
        couchdb_url="http://localhost:5984",
        database="mycobank_annotations",
        db_name="mycobank",
        username="admin",
        password="password",
        pattern="*.txt.ann"
    )

    # Write to CSV
    with open("output/taxa_local.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=Taxon.FIELDNAMES)
        writer.writeheader()

        for taxon in taxa:
            for para_dict in taxon.dictionaries():
                writer.writerow(para_dict)

    print(f"\nWrote {len(taxa)} taxa to output/taxa_local.csv")


def example_with_filtering():
    """Example: Extract only nomenclature paragraphs from specific documents."""
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col

    spark = SparkSession.builder \
        .appName("FilteredTaxonExtractor") \
        .master("local[*]") \
        .getOrCreate()

    # Extract all taxa
    taxa_df = extract_taxa_spark(
        spark=spark,
        couchdb_url="http://localhost:5984",
        database="mycobank_annotations",
        db_name="mycobank",
        pattern="*.txt.ann"
    )

    # Filter for only nomenclature paragraphs
    nomenclature_df = taxa_df.filter(col("label") == "Nomenclature")

    # Filter for specific documents
    specific_docs = nomenclature_df.filter(
        col("doc_id").startswith("article_2023_")
    )

    print(f"\nFound {specific_docs.count()} nomenclature paragraphs "
          f"in articles from 2023")

    specific_docs.select("serial_number", "doc_id", "body").show(truncate=50)

    spark.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract Taxon objects from CouchDB annotations"
    )
    parser.add_argument(
        "--mode",
        choices=["distributed", "local", "filtering"],
        default="local",
        help="Execution mode"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:5984",
        help="CouchDB URL"
    )
    parser.add_argument(
        "--database",
        required=True,
        help="CouchDB database name"
    )
    parser.add_argument(
        "--db-name",
        required=True,
        help="Logical database name for metadata (ingest_db_name)"
    )
    parser.add_argument(
        "--username",
        help="CouchDB username"
    )
    parser.add_argument(
        "--password",
        help="CouchDB password"
    )
    parser.add_argument(
        "--pattern",
        default="*.txt.ann",
        help="Attachment filename pattern"
    )

    args = parser.parse_args()

    if args.mode == "distributed":
        example_distributed()
    elif args.mode == "local":
        example_local()
    else:
        example_with_filtering()
