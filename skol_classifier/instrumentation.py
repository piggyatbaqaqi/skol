"""
Instrumentation utilities for diagnosing Spark task size and OOM issues.

This module provides utilities to:
- Monitor DataFrame sizes and partition counts
- Track closure sizes in UDFs and mapPartitions
- Log broadcast variable sizes
- Identify operations that create large task binaries
"""

import sys
import pickle
from typing import Any, Callable, Dict, Optional
from functools import wraps
import time

from pyspark.sql import DataFrame
from pyspark.broadcast import Broadcast


class SparkInstrumentation:
    """Utilities for instrumenting Spark operations."""

    def __init__(self, verbosity: int = 1):
        """
        Initialize instrumentation.

        Args:
            verbosity: Level of logging (0=none, 1=warnings, 2=info, 3=debug)
        """
        self.verbosity = verbosity
        self.metrics = {}

    def log(self, level: int, message: str) -> None:
        """
        Log message if verbosity is high enough.

        Args:
            level: Minimum verbosity level to log
            message: Message to log
        """
        if self.verbosity >= level:
            print(f"[Instrumentation] {message}")

    def measure_closure_size(self, func: Callable, name: str = "closure") -> int:
        """
        Measure the serialized size of a closure.

        This helps identify functions that capture large objects.

        Args:
            func: Function to measure
            name: Name for logging

        Returns:
            Size in bytes
        """
        try:
            serialized = pickle.dumps(func)
            size_bytes = len(serialized)
            size_kb = size_bytes / 1024

            if size_kb > 1000:
                self.log(1, f"⚠ WARNING: Large closure '{name}': {size_kb:.1f} KiB")
            elif size_kb > 100:
                self.log(2, f"  Closure '{name}': {size_kb:.1f} KiB")
            else:
                self.log(3, f"  Closure '{name}': {size_kb:.1f} KiB")

            self.metrics[f"closure_{name}_kb"] = size_kb
            return size_bytes

        except Exception as e:
            self.log(1, f"⚠ WARNING: Could not measure closure '{name}': {e}")
            return -1

    def measure_broadcast_size(self, broadcast_var: Broadcast, name: str = "broadcast") -> int:
        """
        Measure the size of a broadcast variable.

        Args:
            broadcast_var: Broadcast variable to measure
            name: Name for logging

        Returns:
            Size in bytes (estimate)
        """
        try:
            value = broadcast_var.value
            serialized = pickle.dumps(value)
            size_bytes = len(serialized)
            size_mb = size_bytes / (1024 * 1024)

            if size_mb > 4:
                self.log(1, f"⚠ WARNING: Large broadcast '{name}': {size_mb:.1f} MiB")
            elif size_mb > 1:
                self.log(2, f"  Broadcast '{name}': {size_mb:.1f} MiB")
            else:
                self.log(3, f"  Broadcast '{name}': {size_mb:.2f} MiB")

            self.metrics[f"broadcast_{name}_mb"] = size_mb
            return size_bytes

        except Exception as e:
            self.log(1, f"⚠ WARNING: Could not measure broadcast '{name}': {e}")
            return -1

    def analyze_dataframe(
        self,
        df: DataFrame,
        name: str = "dataframe",
        count: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze DataFrame characteristics.

        Args:
            df: DataFrame to analyze
            name: Name for logging
            count: Whether to count rows (expensive!)

        Returns:
            Dictionary with metrics
        """
        metrics = {
            "num_columns": len(df.columns),
            "num_partitions": df.rdd.getNumPartitions(),
        }

        self.log(2, f"  DataFrame '{name}':")
        self.log(2, f"    Columns: {metrics['num_columns']}")
        self.log(2, f"    Partitions: {metrics['num_partitions']}")

        if count:
            start = time.time()
            row_count = df.count()
            elapsed = time.time() - start
            metrics["num_rows"] = row_count
            self.log(2, f"    Rows: {row_count:,} (counted in {elapsed:.1f}s)")

        # Check lineage depth as proxy for execution plan size
        # Note: toDebugString() may not exist in Spark Connect mode
        try:
            lineage_str = df._jdf.toDebugString()
            lineage_depth = lineage_str.count('\n')
        except Exception:
            # Spark Connect or other mode without toDebugString
            lineage_depth = 0
            self.log(3, f"  (lineage depth unavailable - Spark Connect mode)")
        metrics["lineage_depth"] = lineage_depth

        if lineage_depth > 50:
            self.log(1, f"⚠ WARNING: Deep lineage in '{name}': {lineage_depth} stages")
            self.log(1, f"   Consider checkpointing to break lineage")
        elif lineage_depth > 20:
            self.log(2, f"  Lineage depth '{name}': {lineage_depth} stages")
        else:
            self.log(3, f"  Lineage depth '{name}': {lineage_depth} stages")

        self.metrics[f"df_{name}"] = metrics
        return metrics

    def checkpoint_if_needed(
        self,
        df: DataFrame,
        name: str = "dataframe",
        lineage_threshold: int = 30
    ) -> DataFrame:
        """
        Checkpoint DataFrame if lineage is too deep.

        Args:
            df: DataFrame to potentially checkpoint
            name: Name for logging
            lineage_threshold: Max lineage depth before checkpointing

        Returns:
            Original or checkpointed DataFrame
        """
        lineage_str = df._jdf.toDebugString()
        lineage_depth = lineage_str.count('\n')

        if lineage_depth > lineage_threshold:
            self.log(1, f"  Checkpointing '{name}' due to deep lineage ({lineage_depth} stages)")
            # Use cache instead of checkpoint for now (checkpoint requires checkpoint dir)
            df = df.cache()
            # Force materialization
            df.count()
            self.log(2, f"  ✓ Cached '{name}'")

        return df

    def instrument_mappartitions(
        self,
        df: DataFrame,
        func: Callable,
        operation_name: str = "mapPartitions"
    ) -> None:
        """
        Instrument a mapPartitions operation before execution.

        Args:
            df: DataFrame that will be mapped
            func: Function to apply to partitions
            operation_name: Name for logging
        """
        self.log(2, f"\n{'='*70}")
        self.log(2, f"Instrumenting: {operation_name}")
        self.log(2, f"{'='*70}")

        # Measure closure size
        self.measure_closure_size(func, operation_name)

        # Analyze input DataFrame
        self.analyze_dataframe(df, f"{operation_name}_input")

        self.log(2, f"{'='*70}\n")

    def get_metrics_summary(self) -> str:
        """
        Get summary of all collected metrics.

        Returns:
            Formatted string with metrics
        """
        if not self.metrics:
            return "[Instrumentation] No metrics collected"

        summary = ["\n" + "="*70]
        summary.append("Instrumentation Metrics Summary")
        summary.append("="*70)

        # Closure sizes
        closure_metrics = {k: v for k, v in self.metrics.items() if k.startswith("closure_")}
        if closure_metrics:
            summary.append("\nClosure Sizes:")
            for name, size_kb in sorted(closure_metrics.items(), key=lambda x: x[1], reverse=True):
                pretty_name = name.replace("closure_", "").replace("_kb", "")
                if size_kb > 1000:
                    summary.append(f"  ⚠ {pretty_name}: {size_kb:.1f} KiB")
                else:
                    summary.append(f"    {pretty_name}: {size_kb:.1f} KiB")

        # Broadcast sizes
        broadcast_metrics = {k: v for k, v in self.metrics.items() if k.startswith("broadcast_")}
        if broadcast_metrics:
            summary.append("\nBroadcast Variable Sizes:")
            for name, size_mb in sorted(broadcast_metrics.items(), key=lambda x: x[1], reverse=True):
                pretty_name = name.replace("broadcast_", "").replace("_mb", "")
                if size_mb > 4:
                    summary.append(f"  ⚠ {pretty_name}: {size_mb:.1f} MiB")
                else:
                    summary.append(f"    {pretty_name}: {size_mb:.2f} MiB")

        # DataFrame metrics
        df_metrics = {k: v for k, v in self.metrics.items() if k.startswith("df_")}
        if df_metrics:
            summary.append("\nDataFrame Lineage Depths:")
            for name, metrics in df_metrics.items():
                pretty_name = name.replace("df_", "")
                depth = metrics.get("lineage_depth", 0)
                partitions = metrics.get("num_partitions", 0)
                if depth > 50:
                    summary.append(f"  ⚠ {pretty_name}: {depth} stages, {partitions} partitions")
                else:
                    summary.append(f"    {pretty_name}: {depth} stages, {partitions} partitions")

        summary.append("="*70 + "\n")
        return "\n".join(summary)


def instrument_method(operation_name: str, verbosity: int = 1):
    """
    Decorator to instrument a method that performs Spark operations.

    Usage:
        @instrument_method("save_to_couchdb", verbosity=2)
        def save_to_couchdb(self, df):
            ...

    Args:
        operation_name: Name for logging
        verbosity: Logging level
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            instrumentation = SparkInstrumentation(verbosity=verbosity)

            instrumentation.log(2, f"\n{'='*70}")
            instrumentation.log(2, f"Starting: {operation_name}")
            instrumentation.log(2, f"{'='*70}")

            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time

                instrumentation.log(2, f"\n{'='*70}")
                instrumentation.log(2, f"Completed: {operation_name} in {elapsed:.1f}s")
                instrumentation.log(2, f"{'='*70}\n")

                return result

            except Exception as e:
                elapsed = time.time() - start_time
                instrumentation.log(1, f"\n✗ Failed: {operation_name} after {elapsed:.1f}s")
                instrumentation.log(1, f"  Error: {e}")
                raise

        return wrapper
    return decorator
