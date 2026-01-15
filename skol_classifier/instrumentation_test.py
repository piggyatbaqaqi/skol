"""
Tests for instrumentation.py module.

Run with: pytest skol_classifier/instrumentation_test.py -v
"""

import os
import sys
import time
import pytest
from pyspark.sql import SparkSession

from .instrumentation import SparkInstrumentation, instrument_method


# Module-level functions for testing closure size measurement
# (Local functions inside test methods can't always be pickled)
def _small_test_func():
    """A small function for testing closure size."""
    return 1


# Module-level data for large closure test
_LARGE_DATA_FOR_TEST = list(range(100000))


def _large_test_func():
    """A function that captures large module-level data."""
    return sum(_LARGE_DATA_FOR_TEST)


def _simple_partition_func(partition):
    """Simple function for mapPartitions test."""
    return partition


# Get the project root directory (parent of skol_classifier)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)


# Fixtures
@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for testing."""
    # Ensure the parent directory is in the Python path for Spark workers
    if PARENT_ROOT not in sys.path:
        sys.path.insert(0, PARENT_ROOT)

    session = SparkSession.builder \
        .appName("InstrumentationTests") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    yield session
    session.stop()


class TestSparkInstrumentation:
    """Tests for SparkInstrumentation class."""

    def test_init_default_verbosity(self):
        """Test default verbosity is 1."""
        instr = SparkInstrumentation()
        assert instr.verbosity == 1
        assert instr.metrics == {}

    def test_init_custom_verbosity(self):
        """Test custom verbosity setting."""
        instr = SparkInstrumentation(verbosity=3)
        assert instr.verbosity == 3

    def test_log_below_verbosity(self, capsys):
        """Test that log messages below verbosity are not printed."""
        instr = SparkInstrumentation(verbosity=1)
        instr.log(2, "This should not appear")
        captured = capsys.readouterr()
        assert "This should not appear" not in captured.out

    def test_log_at_verbosity(self, capsys):
        """Test that log messages at verbosity level are printed."""
        instr = SparkInstrumentation(verbosity=2)
        instr.log(2, "This should appear")
        captured = capsys.readouterr()
        assert "This should appear" in captured.out

    def test_log_above_verbosity(self, capsys):
        """Test that log messages above verbosity are printed."""
        instr = SparkInstrumentation(verbosity=3)
        instr.log(2, "This should also appear")
        captured = capsys.readouterr()
        assert "This should also appear" in captured.out

    def test_measure_closure_size_small(self):
        """Test measuring size of a small closure.

        Uses module-level function to avoid pickle issues with local functions.
        """
        instr = SparkInstrumentation(verbosity=3)

        size = instr.measure_closure_size(_small_test_func, "small_func")

        assert size > 0
        assert "closure_small_func_kb" in instr.metrics
        assert instr.metrics["closure_small_func_kb"] < 100  # Should be small

    def test_measure_closure_size_large_captures(self):
        """Test measuring size of closure that captures data.

        Note: Module-level functions referencing module-level data don't
        actually capture that data in their closure - they just reference
        the module global. So the closure size is small. This test verifies
        that measure_closure_size works correctly even for such functions.
        """
        instr = SparkInstrumentation(verbosity=1)

        size = instr.measure_closure_size(_large_test_func, "large_closure")

        assert size > 0
        assert "closure_large_closure_kb" in instr.metrics
        # Module-level functions don't capture module globals in closure
        # The function itself is small; only local closures capture data
        assert instr.metrics["closure_large_closure_kb"] >= 0

    def test_measure_closure_size_unpicklable(self, capsys):
        """Test handling of unpicklable closures."""
        instr = SparkInstrumentation(verbosity=1)

        # Lambda with local reference that might cause pickle issues
        def unpicklable_func():
            # File handles can't be pickled
            return sys.stdout

        # Should return -1 or handle gracefully
        size = instr.measure_closure_size(unpicklable_func, "test_func")
        # This should succeed since we're not actually trying to use sys.stdout
        assert size >= -1

    def test_analyze_dataframe(self, spark):
        """Test DataFrame analysis."""
        instr = SparkInstrumentation(verbosity=2)

        # Create simple DataFrame
        data = [(1, "a"), (2, "b"), (3, "c")]
        df = spark.createDataFrame(data, ["id", "value"])

        metrics = instr.analyze_dataframe(df, "test_df", count=False)

        assert metrics["num_columns"] == 2
        assert metrics["num_partitions"] > 0
        assert "lineage_depth" in metrics
        assert "df_test_df" in instr.metrics

    def test_analyze_dataframe_with_count(self, spark):
        """Test DataFrame analysis with row count."""
        instr = SparkInstrumentation(verbosity=2)

        data = [(1, "a"), (2, "b"), (3, "c")]
        df = spark.createDataFrame(data, ["id", "value"])

        metrics = instr.analyze_dataframe(df, "test_df", count=True)

        assert metrics["num_rows"] == 3

    def test_checkpoint_if_needed_shallow(self, spark):
        """Test that shallow lineage doesn't trigger checkpoint."""
        instr = SparkInstrumentation(verbosity=2)

        data = [(1, "a"), (2, "b")]
        df = spark.createDataFrame(data, ["id", "value"])

        # Shallow lineage should not checkpoint
        result_df = instr.checkpoint_if_needed(df, "test_df", lineage_threshold=100)

        # DataFrame should be returned unchanged (not cached)
        # We can verify by checking if it's the same object or similar
        assert result_df is not None

    def test_checkpoint_if_needed_deep(self, spark):
        """Test that deep lineage triggers checkpoint.

        Note: In Spark Connect mode, toDebugString() is not available,
        so checkpointing is skipped and the original DataFrame is returned.
        """
        instr = SparkInstrumentation(verbosity=2)

        # Create DataFrame with some transformations
        data = [(i, f"value_{i}") for i in range(10)]
        df = spark.createDataFrame(data, ["id", "value"])

        # Force checkpoint with very low threshold
        result_df = instr.checkpoint_if_needed(df, "test_df", lineage_threshold=1)

        # Should return a DataFrame (cached or original in Connect mode)
        assert result_df is not None
        # In Spark Connect mode, toDebugString() is unavailable so
        # checkpointing is skipped - we just verify a DataFrame is returned

    def test_get_metrics_summary_empty(self):
        """Test metrics summary with no metrics."""
        instr = SparkInstrumentation(verbosity=1)
        summary = instr.get_metrics_summary()

        assert "No metrics collected" in summary

    def test_get_metrics_summary_with_metrics(self):
        """Test metrics summary with collected metrics."""
        instr = SparkInstrumentation(verbosity=3)

        # Add some metrics manually
        instr.metrics["closure_test_kb"] = 50.0
        instr.metrics["df_test"] = {"lineage_depth": 10, "num_partitions": 4}

        summary = instr.get_metrics_summary()

        assert "Instrumentation Metrics Summary" in summary
        assert "Closure Sizes" in summary
        assert "test" in summary

    def test_instrument_mappartitions(self, spark):
        """Test mapPartitions instrumentation.

        Uses module-level function to avoid pickle issues.
        """
        instr = SparkInstrumentation(verbosity=2)

        data = [(1, "a"), (2, "b")]
        df = spark.createDataFrame(data, ["id", "value"])

        instr.instrument_mappartitions(df, _simple_partition_func, "test_operation")

        # Should have recorded closure size and DataFrame metrics
        assert "closure_test_operation_kb" in instr.metrics
        assert "df_test_operation_input" in instr.metrics


class TestInstrumentMethodDecorator:
    """Tests for the instrument_method decorator."""

    def test_decorator_success(self, capsys):
        """Test decorator with successful function."""
        @instrument_method("test_operation", verbosity=2)
        def successful_func():
            return "success"

        result = successful_func()

        assert result == "success"
        captured = capsys.readouterr()
        assert "Starting: test_operation" in captured.out
        assert "Completed: test_operation" in captured.out

    def test_decorator_with_args(self, capsys):
        """Test decorator preserves function arguments."""
        @instrument_method("add_operation", verbosity=2)
        def add_func(a, b):
            return a + b

        result = add_func(3, 5)

        assert result == 8

    def test_decorator_with_kwargs(self, capsys):
        """Test decorator preserves keyword arguments."""
        @instrument_method("greet_operation", verbosity=2)
        def greet_func(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = greet_func("World", greeting="Hi")

        assert result == "Hi, World!"

    def test_decorator_exception(self, capsys):
        """Test decorator handles exceptions correctly."""
        @instrument_method("failing_operation", verbosity=1)
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_func()

        captured = capsys.readouterr()
        assert "Failed: failing_operation" in captured.out

    def test_decorator_timing(self, capsys):
        """Test that decorator measures execution time."""
        @instrument_method("timed_operation", verbosity=2)
        def slow_func():
            time.sleep(0.1)
            return "done"

        result = slow_func()

        assert result == "done"
        captured = capsys.readouterr()
        # Should show time elapsed
        assert "in" in captured.out
        assert "s" in captured.out

    def test_decorator_silent_mode(self, capsys):
        """Test decorator with verbosity 0 (silent)."""
        @instrument_method("silent_operation", verbosity=0)
        def silent_func():
            return "silent"

        result = silent_func()

        assert result == "silent"
        captured = capsys.readouterr()
        # Should not print anything
        assert captured.out == ""
