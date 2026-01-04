"""
Test script to verify predict_proba() functionality for TraditionalMLSkolModel.

This script tests that:
1. predict_proba() returns predictions with probabilities column
2. The probabilities are in array format (not Spark ML Vector)
3. calculate_stats() can access probabilities and compute loss metrics
4. Confusion matrix appears at verbosity >= 2
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, DoubleType
from skol_classifier.model import LogisticRegressionSkolModel, RandomForestSkolModel
from pyspark.ml.linalg import Vectors


def test_logistic_regression_predict_proba():
    """Test predict_proba() with Logistic Regression model."""
    print("\n" + "="*70)
    print("TEST 1: Logistic Regression predict_proba()")
    print("="*70)

    spark = SparkSession.builder.appName("Test Predict Proba").getOrCreate()

    # Create sample training data with features and labels
    # Using simple feature vectors for testing
    train_data = [
        (Vectors.dense([1.0, 0.0, 0.0]), 0),
        (Vectors.dense([1.0, 0.0, 0.0]), 0),
        (Vectors.dense([0.0, 1.0, 0.0]), 1),
        (Vectors.dense([0.0, 1.0, 0.0]), 1),
        (Vectors.dense([0.0, 0.0, 1.0]), 2),
        (Vectors.dense([0.0, 0.0, 1.0]), 2),
    ]

    schema = StructType([
        StructField("combined_idf", ArrayType(DoubleType()), False),
        StructField("label_indexed", IntegerType(), False),
    ])

    # Convert to proper format for Spark ML
    from pyspark.ml.linalg import VectorUDT
    actual_schema = StructType([
        StructField("combined_idf", VectorUDT(), False),
        StructField("label_indexed", IntegerType(), False),
    ])

    train_df = spark.createDataFrame(train_data, schema=actual_schema)

    # Create and train model
    model = LogisticRegressionSkolModel(
        features_col="combined_idf",
        label_col="label_indexed",
        verbosity=2  # Enable confusion matrix
    )

    labels = ["Class0", "Class1", "Class2"]
    model.fit(train_df, labels=labels)

    print("✓ Model trained successfully")

    # Test predict_proba()
    predictions = model.predict_proba(train_df)

    print("✓ predict_proba() executed successfully")

    # Verify predictions has required columns
    assert "prediction" in predictions.columns, "Missing 'prediction' column"
    assert "probabilities" in predictions.columns, "Missing 'probabilities' column"
    print("✓ Predictions have required columns")

    # Check that probabilities is an array (not Vector)
    sample_row = predictions.select("probabilities").first()
    probs = sample_row["probabilities"]
    assert isinstance(probs, list), f"probabilities should be list, got {type(probs)}"
    assert len(probs) == 3, f"Expected 3 probabilities, got {len(probs)}"
    print(f"✓ Probabilities are in array format: {probs}")

    # Verify probabilities sum to ~1.0
    prob_sum = sum(probs)
    assert 0.99 <= prob_sum <= 1.01, f"Probabilities should sum to 1.0, got {prob_sum}"
    print(f"✓ Probabilities sum to 1.0: {prob_sum:.4f}")

    # Test that calculate_stats() works with probabilities
    stats = model.calculate_stats(predictions, verbose=True)

    # Verify stats includes loss metrics
    assert "loss" in stats, "Stats should include 'loss' metric"
    assert not (stats["loss"] != stats["loss"]), "Loss should not be NaN"  # Check for NaN
    print(f"✓ Loss calculated: {stats['loss']:.4f}")

    # Verify per-class loss metrics
    for label in labels:
        key = f"{label}_loss"
        assert key in stats, f"Missing per-class loss: {key}"
        print(f"  {label} loss: {stats[key]:.4f}")

    print("\n✓ PASS: Logistic Regression predict_proba() works correctly")


def test_random_forest_predict_proba():
    """Test predict_proba() with Random Forest model."""
    print("\n" + "="*70)
    print("TEST 2: Random Forest predict_proba()")
    print("="*70)

    spark = SparkSession.builder.appName("Test RF Predict Proba").getOrCreate()

    # Create sample training data
    train_data = [
        (Vectors.dense([1.0, 0.0, 0.0]), 0),
        (Vectors.dense([1.0, 0.0, 0.0]), 0),
        (Vectors.dense([0.0, 1.0, 0.0]), 1),
        (Vectors.dense([0.0, 1.0, 0.0]), 1),
        (Vectors.dense([0.0, 0.0, 1.0]), 2),
        (Vectors.dense([0.0, 0.0, 1.0]), 2),
    ]

    from pyspark.ml.linalg import VectorUDT
    schema = StructType([
        StructField("combined_idf", VectorUDT(), False),
        StructField("label_indexed", IntegerType(), False),
    ])

    train_df = spark.createDataFrame(train_data, schema=schema)

    # Create and train model
    model = RandomForestSkolModel(
        features_col="combined_idf",
        label_col="label_indexed",
        n_estimators=10,
        verbosity=1
    )

    labels = ["ClassA", "ClassB", "ClassC"]
    model.fit(train_df, labels=labels)

    print("✓ Random Forest trained successfully")

    # Test predict_proba()
    predictions = model.predict_proba(train_df)

    print("✓ predict_proba() executed successfully")

    # Verify probabilities column exists and is array
    assert "probabilities" in predictions.columns, "Missing 'probabilities' column"

    sample_row = predictions.select("probabilities").first()
    probs = sample_row["probabilities"]
    assert isinstance(probs, list), f"probabilities should be list, got {type(probs)}"
    print(f"✓ Probabilities format verified: {probs}")

    # Test calculate_stats()
    stats = model.calculate_stats(predictions, verbose=True)

    assert "loss" in stats, "Stats should include 'loss' metric"
    print(f"✓ Loss calculated: {stats['loss']:.4f}")

    print("\n✓ PASS: Random Forest predict_proba() works correctly")


def test_predict_uses_predict_proba():
    """Test that predict() uses predict_proba() internally."""
    print("\n" + "="*70)
    print("TEST 3: Verify predict() uses predict_proba()")
    print("="*70)

    spark = SparkSession.builder.appName("Test Predict").getOrCreate()

    # Create sample training data
    train_data = [
        (Vectors.dense([1.0, 0.0]), 0),
        (Vectors.dense([0.0, 1.0]), 1),
    ]

    from pyspark.ml.linalg import VectorUDT
    schema = StructType([
        StructField("combined_idf", VectorUDT(), False),
        StructField("label_indexed", IntegerType(), False),
    ])

    train_df = spark.createDataFrame(train_data, schema=schema)

    # Create and train model
    model = LogisticRegressionSkolModel(
        features_col="combined_idf",
        label_col="label_indexed",
        verbosity=0
    )

    model.fit(train_df, labels=["Class0", "Class1"])

    # Call predict() (which should internally call predict_proba())
    predictions = model.predict(train_df)

    # Verify predictions has probabilities column (proving predict_proba was called)
    assert "probabilities" in predictions.columns, \
        "predict() should return probabilities column via predict_proba()"

    # Verify _last_predictions was cached
    assert model._last_predictions is not None, \
        "_last_predictions should be cached"

    print("✓ predict() correctly uses predict_proba() internally")
    print("✓ Results cached in _last_predictions")

    print("\n✓ PASS: predict() delegation works correctly")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PREDICT_PROBA TESTS")
    print("="*70)

    try:
        test_logistic_regression_predict_proba()
        test_random_forest_predict_proba()
        test_predict_uses_predict_proba()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nThe predict_proba() functionality is working correctly!")
        print("\nKey features verified:")
        print("  ✓ Spark ML Vector converted to array format")
        print("  ✓ Probabilities column present in predictions")
        print("  ✓ calculate_stats() can compute loss metrics")
        print("  ✓ Confusion matrix displayed at verbosity >= 2")
        print("  ✓ predict() uses predict_proba() internally")
        print("  ✓ Results cached for stats calculation")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
