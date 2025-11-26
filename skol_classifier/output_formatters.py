"""
Output formatting module for SKOL classifier.

This module provides classes for formatting predictions in various
formats (YEDA annotation format, labels only, probabilities, etc.).
"""

from typing import List
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, concat, lit, expr, udf, collect_list
from pyspark.sql.types import StringType, ArrayType
from pyspark.sql.window import Window


class YedaFormatter:
    """
    Formats predictions in YEDA annotation format.

    YEDA format: [@ text #Label*]
    """

    def __init__(self, coalesce_labels: bool = False, line_level: bool = False):
        """
        Initialize the formatter.

        Args:
            coalesce_labels: Whether to coalesce consecutive labels
            line_level: Whether data is line-level
        """
        self.coalesce_labels = coalesce_labels
        self.line_level = line_level

    def format(self, predictions: DataFrame) -> DataFrame:
        """
        Format predictions in YEDA annotation format.

        Args:
            predictions: DataFrame with predictions

        Returns:
            DataFrame with annotated_value column (and optionally coalesced)
        """
        # First add annotated_value column
        formatted = self.format_predictions(predictions)

        # Coalesce if requested
        if self.coalesce_labels and self.line_level:
            return self.coalesce_consecutive_labels(formatted, line_level=self.line_level)

        return formatted

    @staticmethod
    def format_predictions(predictions: DataFrame) -> DataFrame:
        """
        Format predictions in YEDA annotation format.

        Args:
            predictions: DataFrame with predictions

        Returns:
            DataFrame with annotated_value column
        """
        return predictions.withColumn(
            "annotated_value",
            concat(
                lit("[@ "),
                col("value"),
                lit(" #"),
                col("predicted_label"),
                lit("*]")
            )
        )

    @staticmethod
    def coalesce_consecutive_labels(
        predictions: DataFrame,
        line_level: bool = True
    ) -> DataFrame:
        """
        Coalesce consecutive predictions with the same label into blocks.

        This is useful for line-level predictions where you want to merge
        consecutive lines with the same label into a single annotation block.

        Args:
            predictions: DataFrame with predictions (must have predicted_label)
            line_level: Whether to treat as line-level data

        Returns:
            DataFrame with coalesced labels
        """
        if not line_level:
            # For paragraph-level, no coalescing needed
            return predictions

        # Define UDF to coalesce consecutive lines with same label
        def coalesce_lines(rows: List[tuple]) -> List[str]:
            """
            Coalesce consecutive lines with the same label.

            Args:
                rows: List of (line_number, value, predicted_label) tuples

            Returns:
                List of YEDA-formatted annotation blocks
            """
            if not rows:
                return []

            # Sort by row number
            sorted_rows = sorted(rows, key=lambda x: x[0])

            result = []
            current_label = sorted_rows[0][2]
            current_lines = [sorted_rows[0][1]]

            for i in range(1, len(sorted_rows)):
                row_num, value, label = sorted_rows[i]

                if label == current_label:
                    # Same label, accumulate
                    current_lines.append(value)
                else:
                    # Different label, output current block
                    block_text = '\n'.join(current_lines)
                    result.append(f"[@ {block_text} #{current_label}*]")

                    # Start new block
                    current_label = label
                    current_lines = [value]

            # Output final block
            block_text = '\n'.join(current_lines)
            result.append(f"[@ {block_text} #{current_label}*]")

            return result

        # UDF for coalescing
        coalesce_udf = udf(
            coalesce_lines,
            ArrayType(StringType())
        )

        # Check if DataFrame has filename or doc_id
        groupby_col = "filename" if "filename" in predictions.columns else "doc_id"

        # Group by document and coalesce
        return (
            predictions
            .groupBy(groupby_col)
            .agg(
                collect_list(
                    expr("struct(line_number, value, predicted_label)")
                ).alias("rows")
            )
            .withColumn("coalesced_annotations", coalesce_udf(col("rows")))
            .select(groupby_col, "coalesced_annotations")
        )


class FileOutputWriter:
    """
    Writes predictions to local files.
    """

    @staticmethod
    def save_annotated(
        predictions: DataFrame,
        output_path: str,
        coalesce_labels: bool = False,
        line_level: bool = False
    ) -> None:
        """
        Save annotated predictions to disk.

        Args:
            predictions: DataFrame with predictions
            output_path: Directory to save output files
            coalesce_labels: Whether to coalesce consecutive labels
            line_level: Whether data is line-level
        """
        # Format predictions
        if "annotated_value" not in predictions.columns:
            predictions = YedaFormatter.format_predictions(predictions)

        # Coalesce if requested
        if coalesce_labels and line_level:
            predictions = YedaFormatter.coalesce_consecutive_labels(
                predictions, line_level=True
            )
            # For coalesced output, we have a different structure
            predictions.write.mode("overwrite").json(output_path)
            return

        # Determine grouping column
        groupby_col = "filename" if "filename" in predictions.columns else "doc_id"

        # Check if we have line_number for ordering
        if "line_number" in predictions.columns:
            # Aggregate with ordering
            aggregated_df = (
                predictions.groupBy(groupby_col)
                .agg(
                    expr("sort_array(collect_list(struct(line_number, annotated_value))) AS sorted_list")
                )
                .withColumn("annotated_value_ordered", expr("transform(sorted_list, x -> x.annotated_value)"))
                .withColumn("final_aggregated", expr("array_join(annotated_value_ordered, '\n')"))
                .select(groupby_col, "final_aggregated")
            )
        else:
            # Aggregate without ordering
            aggregated_df = (
                predictions.groupBy(groupby_col)
                .agg(
                    collect_list("annotated_value").alias("annotations")
                )
                .withColumn("final_aggregated", expr("array_join(annotations, '\n')"))
                .select(groupby_col, "final_aggregated")
            )

        # Write to disk
        aggregated_df.write.partitionBy(groupby_col).mode("overwrite").text(output_path)

    @staticmethod
    def save_labels(predictions: DataFrame, output_path: str) -> None:
        """
        Save only the predicted labels.

        Args:
            predictions: DataFrame with predictions
            output_path: Path to save labels
        """
        predictions.select("predicted_label").write.mode("overwrite").text(output_path)

    @staticmethod
    def save_probabilities(predictions: DataFrame, output_path: str) -> None:
        """
        Save prediction probabilities.

        Args:
            predictions: DataFrame with predictions including probability column
            output_path: Path to save probabilities
        """
        if "probability" not in predictions.columns:
            raise ValueError("DataFrame does not have 'probability' column")

        predictions.select("value", "predicted_label", "probability").write.mode(
            "overwrite"
        ).json(output_path)


class CouchDBOutputWriter:
    """
    Writes predictions back to CouchDB as attachments.
    """

    def __init__(
        self,
        couchdb_url: str,
        database: str,
        username: str,
        password: str
    ):
        """
        Initialize the writer.

        Args:
            couchdb_url: CouchDB server URL
            database: Database name
            username: CouchDB username
            password: CouchDB password
        """
        from .couchdb_io import CouchDBConnection

        self.conn = CouchDBConnection(
            couchdb_url=couchdb_url,
            database=database,
            username=username,
            password=password
        )

    def save_annotated(
        self,
        predictions: DataFrame,
        suffix: str = ".ann",
        coalesce_labels: bool = False,
        line_level: bool = False
    ) -> None:
        """
        Save predictions to CouchDB as attachments.

        Args:
            predictions: DataFrame with predictions
            suffix: Suffix for attachment names
            coalesce_labels: Whether to coalesce consecutive labels
            line_level: Whether data is line-level
        """
        # Format predictions
        if "annotated_value" not in predictions.columns:
            predictions = YedaFormatter.format_predictions(predictions)

        # Coalesce if requested
        if coalesce_labels and line_level:
            predictions = YedaFormatter.coalesce_consecutive_labels(
                predictions, line_level=True
            )
            # For coalesced output, we have coalesced_annotations column
            # We need to join them into final_aggregated_pg
            from pyspark.sql.functions import expr
            predictions = predictions.withColumn(
                "final_aggregated_pg",
                expr("array_join(coalesced_annotations, '\n')")
            )
        else:
            # Aggregate annotated values by document
            groupby_col = "doc_id" if "doc_id" in predictions.columns else "filename"
            attachment_col = "attachment_name" if "attachment_name" in predictions.columns else "filename"

            # Check if we have line_number for ordering
            if "line_number" in predictions.columns:
                from pyspark.sql.functions import expr
                predictions = (
                    predictions.groupBy(groupby_col, attachment_col)
                    .agg(
                        expr("sort_array(collect_list(struct(line_number, annotated_value))) AS sorted_list")
                    )
                    .withColumn("annotated_value_ordered", expr("transform(sorted_list, x -> x.annotated_value)"))
                    .withColumn("final_aggregated_pg", expr("array_join(annotated_value_ordered, '\n')"))
                    .select(groupby_col, attachment_col, "final_aggregated_pg")
                )
            else:
                from pyspark.sql.functions import collect_list, expr
                predictions = (
                    predictions.groupBy(groupby_col, attachment_col)
                    .agg(
                        collect_list("annotated_value").alias("annotations")
                    )
                    .withColumn("final_aggregated_pg", expr("array_join(annotations, '\n')"))
                    .select(groupby_col, attachment_col, "final_aggregated_pg")
                )

            # Rename columns for CouchDB save
            if groupby_col != "doc_id":
                predictions = predictions.withColumnRenamed(groupby_col, "doc_id")
            if attachment_col != "attachment_name":
                predictions = predictions.withColumnRenamed(attachment_col, "attachment_name")

        # Use CouchDB connection to save
        self.conn.save_distributed(predictions, suffix=suffix)
