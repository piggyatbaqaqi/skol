"""
Example comparing different models and feature configurations
"""

from skol_classifier import SkolClassifier, get_file_list


def compare_models():
    """Compare different model configurations."""

    # Initialize classifier
    classifier = SkolClassifier()

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
            "regParam": 0.01
        },
        {
            "name": "Logistic Regression (words + suffixes)",
            "model_type": "logistic",
            "use_suffixes": True,
            "maxIter": 10,
            "regParam": 0.01
        },
        {
            "name": "Random Forest (words only)",
            "model_type": "random_forest",
            "use_suffixes": False,
            "numTrees": 100
        },
        {
            "name": "Random Forest (words + suffixes)",
            "model_type": "random_forest",
            "use_suffixes": True,
            "numTrees": 100
        }
    ]

    # Test each configuration
    results = []
    for config in configs:
        print(f"\n{'=' * 60}")
        print(f"Testing: {config['name']}")
        print(f"{'=' * 60}")

        # Create fresh classifier for each test
        test_classifier = SkolClassifier()

        # Extract configuration parameters
        name = config.pop("name")

        # Train and evaluate
        stats = test_classifier.fit(
            annotated_file_paths=annotated_files,
            test_size=0.2,
            **config
        )

        # Store results
        results.append({
            "name": name,
            **stats
        })

        print(f"  Accuracy:  {stats['accuracy']:.4f}")
        print(f"  Precision: {stats['precision']:.4f}")
        print(f"  Recall:    {stats['recall']:.4f}")
        print(f"  F1 Score:  {stats['f1_score']:.4f}")

    # Print comparison table
    print(f"\n{'=' * 80}")
    print("COMPARISON RESULTS")
    print(f"{'=' * 80}")
    print(f"{'Model':<45} {'Accuracy':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"{'-' * 80}")

    for result in results:
        print(
            f"{result['name']:<45} "
            f"{result['accuracy']:>8.4f} "
            f"{result['precision']:>10.4f} "
            f"{result['recall']:>8.4f} "
            f"{result['f1_score']:>8.4f}"
        )

    # Find best model
    best = max(results, key=lambda x: x['f1_score'])
    print(f"\n{'=' * 80}")
    print(f"Best model: {best['name']} (F1: {best['f1_score']:.4f})")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    compare_models()
