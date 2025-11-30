#!/usr/bin/env python3
"""Example script showing how to use yedda_parser with PySpark."""

from yedda_parser import yedda_file_to_spark_df, get_label_statistics
from pyspark.sql import SparkSession


def main():
    """Parse article_reference.txt and create PySpark DataFrame."""

    # Create Spark session
    print("Creating Spark session...")
    spark = SparkSession.builder \
        .appName("YEDDA Article Parser") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    try:
        # Parse YEDDA file to DataFrame
        print("\nParsing test_data/article_reference.txt...")
        df = yedda_file_to_spark_df('test_data/article_reference.txt', spark)

        # Show schema
        print("\nDataFrame Schema:")
        df.printSchema()

        # Show basic statistics
        print(f"\nTotal rows: {df.count()}")

        # Show label statistics
        print("\nLabel Statistics:")
        stats = get_label_statistics(df)
        stats.show()

        # Show sample rows from each label
        print("\n=== Sample Nomenclature rows ===")
        df.filter(df.label == "Nomenclature").show(5, truncate=80)

        print("\n=== Sample Description rows ===")
        df.filter(df.label == "Description").show(5, truncate=80)

        print("\n=== Sample Misc-exposition rows ===")
        df.filter(df.label == "Misc-exposition").show(5, truncate=80)

        # Save to parquet for future use
        print("\nSaving DataFrame to parquet...")
        df.write.mode("overwrite").parquet("test_data/article_reference.parquet")
        print("Saved to: test_data/article_reference.parquet")

        # Example queries
        print("\n=== Example Query: Lines containing 'Glomus' ===")
        glomus_df = df.filter(df.line.contains("Glomus"))
        print(f"Found {glomus_df.count()} lines containing 'Glomus'")
        glomus_df.show(10, truncate=80)

        print("\n=== Example Query: Nomenclature lines with years ===")
        from pyspark.sql.functions import regexp_extract

        year_df = df.filter(df.label == "Nomenclature") \
            .withColumn("year", regexp_extract(df.line, r'\b(19|20)\d{2}\b', 0)) \
            .filter("year != ''")

        print(f"Found {year_df.count()} nomenclature lines with years")
        year_df.select("line", "year").show(10, truncate=70)

    finally:
        # Stop Spark session
        print("\nStopping Spark session...")
        spark.stop()
        print("Done!")


if __name__ == "__main__":
    main()
