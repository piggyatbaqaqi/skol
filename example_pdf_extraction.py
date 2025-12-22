"""
Example usage of PDFSectionExtractor class with DataFrame output.

This script demonstrates how to extract sections from PDF attachments
in CouchDB documents and work with the resulting PySpark DataFrames.
"""

from pyspark.sql import SparkSession
from pdf_section_extractor import PDFSectionExtractor

# Initialize Spark
spark = SparkSession.builder \
    .appName("PDFSectionExtractorExamples") \
    .getOrCreate()

# Initialize extractor (uses environment variables for credentials)
extractor = PDFSectionExtractor(verbosity=1, spark=spark)

# Example 1: Extract from a specific document
print("\n" + "="*70)
print("EXAMPLE 1: Extract from specific document")
print("="*70)

sections_df = extractor.extract_from_document(
    database='skol_dev',
    doc_id='00df9554e9834283b5e844c7a994ba5f',
    attachment_name='article.pdf'  # Optional: auto-detects if omitted
)

print(f"\nExtracted {sections_df.count()} sections")
print("\nDataFrame schema:")
sections_df.printSchema()

print("\nFirst 3 sections:")
sections_df.select("paragraph_number", "value", "page_number", "line_number") \
    .show(3, truncate=60, vertical=False)

# Example 2: Query DataFrame
print("\n" + "="*70)
print("EXAMPLE 2: Query sections by page number")
print("="*70)

# Get all sections from page 1
page1_sections = sections_df.filter(sections_df.page_number == 1)
print(f"\nSections on page 1: {page1_sections.count()}")
page1_sections.select("paragraph_number", "value").show(5, truncate=60)

# Example 3: Search for specific content using SQL
print("\n" + "="*70)
print("EXAMPLE 3: Search sections with SQL")
print("="*70)

# Register as temp view
sections_df.createOrReplaceTempView("sections")

# Find all sections mentioning "ascospores"
matching = spark.sql("""
    SELECT paragraph_number, value, page_number
    FROM sections
    WHERE LOWER(value) LIKE '%ascospores%'
""")

print(f"\nFound {matching.count()} sections mentioning 'ascospores'")
matching.show(3, truncate=80)

# Example 4: Auto-detect PDF attachment
print("\n" + "="*70)
print("EXAMPLE 4: Auto-detect PDF")
print("="*70)

# List attachments first
attachments = extractor.list_attachments(
    database='skol_dev',
    doc_id='00df9554e9834283b5e844c7a994ba5f'
)
print(f"\nAvailable attachments:")
for name, info in attachments.items():
    content_type = info.get('content_type', 'unknown')
    size = info.get('length', 0)
    print(f"  - {name}: {content_type} ({size:,} bytes)")

# Auto-detect and extract
sections2_df = extractor.extract_from_document(
    database='skol_dev',
    doc_id='00df9554e9834283b5e844c7a994ba5f'
    # attachment_name omitted - will auto-detect PDF
)
print(f"\nAuto-detected PDF and extracted {sections2_df.count()} sections")

# Example 5: Export to various formats
print("\n" + "="*70)
print("EXAMPLE 5: Export DataFrame")
print("="*70)

# Convert to Pandas (for smaller datasets)
print("\nConverting to Pandas DataFrame...")
sections_pandas = sections_df.limit(10).toPandas()
print(f"Pandas DataFrame shape: {sections_pandas.shape}")
print("\nFirst 3 rows:")
print(sections_pandas[['paragraph_number', 'page_number', 'value']].head(3))

# Example 6: Aggregate statistics
print("\n" + "="*70)
print("EXAMPLE 6: Statistics")
print("="*70)

# Count sections per page
print("\nSections per page:")
sections_df.groupBy("page_number") \
    .count() \
    .orderBy("page_number") \
    .show()

# Average section length per page
from pyspark.sql.functions import length, avg

print("\nAverage section length per page:")
sections_df.groupBy("page_number") \
    .agg(avg(length("value")).alias("avg_section_length")) \
    .orderBy("page_number") \
    .show()

print("\n" + "="*70)
print("EXAMPLES COMPLETE")
print("="*70)

# Clean up
spark.stop()
