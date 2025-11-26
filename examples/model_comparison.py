"""
Example comparing different models and feature configurations using V2 API

This demonstrates how the unified SkolClassifierV2 API simplifies
model comparison by having all configuration in the constructor.
"""

from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2
from skol_classifier import get_file_list


def compare_models():
    """Compare different model configurations using V2 API."""

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("SKOL Model Comparison") \
        .master("local[*]") \
        .getOrCreate()

    # Load annotated data
    annotated_files = get_file_list(
        "data/annotated", pattern="**/*.txt.ann"
    )

    # Configurations to test
    configs = [
        {
            "name": "Logistic Regression (words only)",
            "model_type": "logistic",
            "use_suffixes": False,
            "maxIter": 10,
            "regParam": 0.01,
            "line_level": False
        },
        {
            "name": "Logistic Regression (words + suffixes)",
            "model_type": "logistic",
            "use_suffixes": True,
            "maxIter": 10,
            "regParam": 0.01,
            "line_level": False
        },
        {
            "name": "Random Forest (words only)",
            "model_type": "random_forest",
            "use_suffixes": False,
            "n_estimators": 100,
            "line_level": False
        },
        {
            "name": "Random Forest (words + suffixes)",
            "model_type": "random_forest",
            "use_suffixes": True,
            "n_estimators": 100,
            "line_level": False
        },
        {
            "name": "Logistic Regression (line-level, words only)",
            "model_type": "logistic",
            "use_suffixes": False,
            "maxIter": 10,
            "regParam": 0.01,
            "line_level": True
        },
        {
            "name": "Logistic Regression (line-level, words + suffixes)",
            "model_type": "logistic",
            "use_suffixes": True,
            "maxIter": 10,
            "regParam": 0.01,
            "line_level": True
        },
        {
            "name": "Random Forest (line-level, words only)",
            "model_type": "random_forest",
            "use_suffixes": False,
            "n_estimators": 100,
            "line_level": True
        },
        {
            "name": "Random Forest (line-level, words + suffixes)",
            "model_type": "random_forest",
            "use_suffixes": True,
            "n_estimators": 100,
            "line_level": True
        }
    ]

    # Test each configuration
    results = []
    for config in configs:
        print(f"\n{'=' * 60}")
        print(f"Testing: {config['name']}")
        print(f"{'=' * 60}")

        # Extract name from config
        name = config.pop("name")

        # Create classifier with V2 API - all config in constructor
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=annotated_files,
            **config  # Pass all model/feature configuration
        )

        # Train (no parameters needed - everything in constructor)
        stats = classifier.fit()

        # Store results
        results.append({
            "name": name,
            **stats
        })

        print(f"  Mode:      {'Line-level' if config.get('line_level', False) else 'Paragraph'}")
        print(f"  Train:     {stats.get('train_size', 'N/A')} samples")
        print(f"  Test:      {stats.get('test_size', 'N/A')} samples")
        print(f"  Accuracy:  {stats.get('accuracy', 0):.4f}")
        print(f"  Precision: {stats.get('precision', 0):.4f}")
        print(f"  Recall:    {stats.get('recall', 0):.4f}")
        print(f"  F1 Score:  {stats.get('f1_score', 0):.4f}")

    # Print comparison table
    print(f"\n{'=' * 80}")
    print("COMPARISON RESULTS")
    print(f"{'=' * 80}")
    print(f"{'Model':<45} {'Accuracy':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"{'-' * 80}")

    for result in results:
        print(
            f"{result['name']:<45} "
            f"{result.get('accuracy', 0):>8.4f} "
            f"{result.get('precision', 0):>10.4f} "
            f"{result.get('recall', 0):>8.4f} "
            f"{result.get('f1_score', 0):>8.4f}"
        )

    # Find best model
    best = max(results, key=lambda x: x.get('f1_score', 0))
    print(f"\n{'=' * 80}")
    print(f"Best model: {best['name']} (F1: {best.get('f1_score', 0):.4f})")
    print(f"{'=' * 80}")

    spark.stop()


if __name__ == "__main__":
    compare_models()
